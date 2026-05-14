"""Multilingual neuron-set statistics.

For a fixed sparsity ratio, computes set algebra (union / intersection /
per-task exclusive / non-shared) over per-language critical-neuron sets
and dumps the resulting JSON / CSV report.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM

from critnet import CriticalNeuronConfig, NeuronStatistician


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


def _load_indices(path: Path) -> Dict[str, List[int]]:
    if path.is_dir():
        indices_path = path / "neuron_indices.json"
        if not indices_path.is_file():
            raise FileNotFoundError(f"Missing neuron_indices.json in {path}")
        with open(indices_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_config(args: argparse.Namespace) -> CriticalNeuronConfig:
    row_modules = _parse_modules(args.row_modules)
    col_modules = _parse_modules(args.column_modules)
    norm_modules = _parse_modules(args.norm_modules)
    emb_modules = _parse_modules(args.embedding_modules)

    if args.include_all_eligible_modules:
        args.include_attention_norms = True
    if args.include_attention_norms:
        norm_modules = _merge_unique(norm_modules, ["q_norm", "k_norm"])

    return CriticalNeuronConfig(
        base_model_name_or_path=args.model_name,
        row_modules=row_modules,
        column_modules=col_modules,
        norm_modules=norm_modules,
        embedding_modules=emb_modules,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--neurons_path", type=str, default="./neuron_detect")
    parser.add_argument("--save_path", type=str, default="./neuron_statistics")
    parser.add_argument("--ratio", type=float, default=0.05,
                        help="Used only to locate the ratio<...> folder under --neurons_path.")
    parser.add_argument("--langs", type=str, default="en,zh,ar,sw",
                        help="Comma-separated languages")

    parser.add_argument("--row_modules", type=str, default=None)
    parser.add_argument("--column_modules", type=str, default=None)
    parser.add_argument("--norm_modules", type=str, default=None)
    parser.add_argument("--embedding_modules", type=str, default=None)
    parser.add_argument("--include_all_eligible_modules", action="store_true")
    parser.add_argument("--include_attention_norms", action="store_true")
    parser.add_argument("--stype", type=str, default="lima_s1")
    args = parser.parse_args()

    config = _build_config(args)

    base_dir = Path(args.neurons_path) / args.model_name / args.stype / f"ratio{args.ratio}"
    save_dir = Path(args.save_path) / args.model_name / args.stype / f"ratio{args.ratio}"
    save_dir.mkdir(parents=True, exist_ok=True)

    lang_list = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    task_indices: Dict[str, Dict[str, List[int]]] = {}
    for lang in lang_list:
        folder_path = base_dir / lang
        flat_path = base_dir / f"{lang}.json"
        if folder_path.is_dir():
            task_indices[lang] = _load_indices(folder_path)
        elif flat_path.is_file():
            task_indices[lang] = _load_indices(flat_path)
        else:
            print(f"Skipping {lang}: no neuron file found.")
    if not task_indices:
        raise ValueError(f"No neuron indices found under {base_dir}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    result = NeuronStatistician(model=model, config=config).analyze(task_indices)

    print(result.summary())
    result.save_report(str(save_dir))
    print(f"Saved statistics and neuron partitions to: {save_dir}")


if __name__ == "__main__":
    main()
