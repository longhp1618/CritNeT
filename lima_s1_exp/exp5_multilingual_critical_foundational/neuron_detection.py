
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from critnet import CriticalNeuronConfig, NeuronDetector
from lima_s1_exp.dataloader import create_dataloader
from lima_s1_exp.utils import define_chat_template_base, prepare_tokenizer, process_data


def _parse_modules(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or []


def _merge_unique(base: Optional[List[str]], extra: List[str]) -> List[str]:
    merged = list(base or [])
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def _summary(indices: Dict[str, List[int]]) -> Dict[str, int]:
    by_module = {}
    total = 0
    for module_name, neuron_list in indices.items():
        count = len(neuron_list)
        by_module[module_name] = count
        total += count
    by_module["__total__"] = total
    return by_module


def _parse_top_k_percent(value: str) -> List[float]:
    """Parse one ratio or a comma-separated list (e.g. ``0.01,0.05,0.1``)."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--top_k_percent is empty.")
    ratios = [float(p) for p in parts]
    for r in ratios:
        if not (0.0 < r <= 1.0):
            raise ValueError(f"Each top_k_percent must be in (0, 1], got {r!r} in {ratios!r}.")
    return ratios


def main() -> None:
    # Example:
    # python -m lima_s1_exp.exp5_multilingual_critical_foundational.neuron_detection --base "Qwen/Qwen3-0.6B" --all_languages --top_k_percent 0.05 --stype lima_s1 --include_all_eligible_modules
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--stype", type=str, default="lima_s1", help="Dataset type used by process_data.")
    parser.add_argument("--output_dir", type=str, default="./neuron_train_data_detect")
    parser.add_argument("--language", type=str, help="Single language code (e.g., en, zh, sw)")
    parser.add_argument("--all_languages", action="store_true", help="Use en, zh, ar, sw in one run")
    parser.add_argument(
        "--top_k_percent",
        type=str,
        default="0.05",
        help="Global top-k ratio, or comma-separated ratios (e.g. 0.001,0.01,0.05) to select and save for each.",
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples per language")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--row_modules", type=str, default=None, help="Comma-separated row modules")
    parser.add_argument("--column_modules", type=str, default=None, help="Comma-separated column modules")
    parser.add_argument("--norm_modules", type=str, default=None, help="Comma-separated norm modules")
    parser.add_argument("--embedding_modules", type=str, default=None, help="Comma-separated embedding modules")

    parser.add_argument("--include_all_eligible_modules", action="store_true", help="Include attention norms (q_norm, k_norm)")
    parser.add_argument("--include_attention_norms", action="store_true", help="Append q_norm and k_norm")

    parser.add_argument(
        "--save_importance_cache_dir",
        type=str,
        default=None,
        help="Directory to write per-language *.pt importance caches (post gate-combine). "
        "Reuse with --importance_cache_dir to sweep --top_k_percent without recomputing gradients.",
    )
    parser.add_argument(
        "--importance_cache_dir",
        type=str,
        default=None,
        help="Load per-language *.pt caches from this directory and only run global top-k (no model load).",
    )
    args = parser.parse_args()

    args.language_template = False
    ratios = _parse_top_k_percent(args.top_k_percent)
    if args.importance_cache_dir and args.save_importance_cache_dir:
        raise ValueError("Use only one of --importance_cache_dir and --save_importance_cache_dir.")

    from_cache = args.importance_cache_dir is not None

    if not from_cache:
        print(f"Loading model: {args.base}")
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        if tokenizer.chat_template is None:
            args.language_template = True
        tokenizer, _ = prepare_tokenizer(args.base, chat_template=tokenizer.chat_template is None)
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
        model.train()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        tokenizer = None
        model = None

    row_modules = _parse_modules(args.row_modules)
    col_modules = _parse_modules(args.column_modules)
    norm_modules = _parse_modules(args.norm_modules)
    emb_modules = _parse_modules(args.embedding_modules)

    if args.include_all_eligible_modules:
        args.include_attention_norms = True

    if args.include_attention_norms:
        norm_modules = _merge_unique(norm_modules, ["q_norm", "k_norm"])

    config = CriticalNeuronConfig(
        base_model_name_or_path=args.base,
        row_modules=row_modules,
        column_modules=col_modules,
        norm_modules=norm_modules,
        embedding_modules=emb_modules,
        sparsity_ratio=ratios[0],
    )

    if args.all_languages:
        target_langs = ["en", "zh", "ar", "sw"]
    else:
        if not args.language:
            raise ValueError("Provide --language or set --all_languages.")
        target_langs = args.language.split(",")  # [args.language]

    text_dct: Optional[dict] = None
    if not from_cache:
        text_dct = process_data(args.stype)
        rng = random.Random(42)
        for lang in text_dct.keys():
            if args.max_samples != -1 and len(text_dct[lang]) > args.max_samples:
                text_dct[lang] = rng.sample(text_dct[lang], args.max_samples)

    importance_save_root: Optional[Path] = None
    if args.save_importance_cache_dir:
        importance_save_root = Path(args.save_importance_cache_dir)
        importance_save_root.mkdir(parents=True, exist_ok=True)

    cache_load_root: Optional[Path] = Path(args.importance_cache_dir) / args.base / args.stype if from_cache else None

    sweep_root = Path(args.output_dir) / args.base / args.stype
    multi_ratio = len(ratios) > 1
    if multi_ratio:
        print(f"Top-k sweep: {ratios}")

    print(f"Processing {target_langs}")
    for lang in target_langs:
        if not from_cache:
            assert text_dct is not None
            if lang not in text_dct:
                print(f"Skipping {lang}: not found in data. Available: {list(text_dct.keys())}")
                continue
            print(f"Processing {lang}")
        else:
            print(f"Processing {lang} (from importance cache)")

        shared_scores_cache: Optional[str] = None
        if multi_ratio and not from_cache:
            if importance_save_root is not None:
                shared_scores_cache = str(importance_save_root / args.base / args.stype / f"{lang}.pt")
            else:
                staging = sweep_root / "_multi_ratio_importance_cache"
                staging.mkdir(parents=True, exist_ok=True)
                shared_scores_cache = str(staging / f"{lang}.pt")

        for ri, ratio in enumerate(ratios):
            config.sparsity_ratio = ratio
            output_dir = sweep_root / f"ratio{ratio}"
            output_dir.mkdir(parents=True, exist_ok=True)

            if from_cache:
                assert cache_load_root is not None
                cache_pt = cache_load_root / f"{lang}.pt"
                if not cache_pt.is_file():
                    print(f"Skipping {lang}: missing cache file {cache_pt}")
                    break
                detector = NeuronDetector(model=None, config=config)
                print(f"\nSelecting neurons for {lang} from {cache_pt} (ratio={ratio})...")
                selected = detector.select_from_importance_cache(str(cache_pt))
            else:
                assert model is not None and tokenizer is not None
                if ri == 0:
                    if args.language_template:
                        tokenizer = define_chat_template_base(tokenizer, lang)

                    samples = text_dct[lang]
                    dataloader = create_dataloader(
                        raw=samples,
                        tokenizer=tokenizer,
                        bs=args.batch_size,
                        MAX_LEN=args.max_length,
                        shuffle=False,
                    )

                    print(f"\nDetecting neurons for {lang} (samples={len(samples)})...")
                    detector = NeuronDetector(model=model, config=config)
                    cache_path = None
                    if multi_ratio:
                        cache_path = shared_scores_cache
                    elif importance_save_root is not None:
                        cache_path = str(importance_save_root / args.base / args.stype / f"{lang}.pt")
                    selected = detector.detect(
                        dataloader=dataloader,
                        save_importance_cache_path=cache_path,
                    )
                else:
                    assert shared_scores_cache is not None
                    detector = NeuronDetector(model=None, config=config)
                    print(f"\nSelecting neurons for {lang} from {shared_scores_cache} (ratio={ratio})...")
                    selected = detector.select_from_importance_cache(shared_scores_cache)

            # Save both toolkit-native folder and legacy single-json file.
            lang_dir = output_dir / lang
            detector.save(str(lang_dir))

            legacy_json = output_dir / f"{lang}.json"
            with open(legacy_json, "w", encoding="utf-8") as f:
                json.dump(selected, f, indent=2)

            counts = _summary(selected)
            print(f"Saved: {lang_dir} and {legacy_json}")
            print(f"Total selected neurons: {counts['__total__']}")


if __name__ == "__main__":
    main()