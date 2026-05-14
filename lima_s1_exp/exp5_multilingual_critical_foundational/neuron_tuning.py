import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.cache/triton")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from critnet import CriticalNeuronConfig, get_neuron_model
from lima_s1_exp.dataloader import build_lm_dataset, collate_fn
from lima_s1_exp.utils import prepare_tokenizer, process_data

logger = logging.getLogger(__name__)


def _apply_liger_kernels(model_name: str) -> bool:
    """Patch model classes with Liger fused kernels *before* from_pretrained.

    Must be called before AutoModelForCausalLM.from_pretrained so the
    patched class is used to construct the model.  Returns True if
    patching succeeded.
    """
    name_lower = model_name.lower()
    try:
        if "llama" in name_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama()
        elif "qwen" in name_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2()
        elif "mistral" in name_lower:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral
            apply_liger_kernel_to_mistral()
        else:
            logger.warning("No Liger kernel mapping for %s; skipping.", model_name)
            return False
        logger.info("Liger kernels applied for %s.", model_name)
        return True
    except ImportError:
        logger.warning("liger_kernel not installed; fused CE unavailable.")
        return False


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


def _load_or_build_config(args: argparse.Namespace) -> CriticalNeuronConfig:
    neuron_path = Path(args.neuron_path)
    if neuron_path.is_dir() and (neuron_path / "critical_neuron_config.json").is_file():
        cfg = CriticalNeuronConfig.from_pretrained(str(neuron_path))
        cfg.base_model_name_or_path = args.base
        return cfg

    with open(neuron_path, "r", encoding="utf-8") as f:
        neuron_indices: Dict[str, List[int]] = json.load(f)

    row_modules = _parse_modules(args.row_modules)
    col_modules = _parse_modules(args.column_modules)
    norm_modules = _parse_modules(args.norm_modules)
    emb_modules = _parse_modules(args.embedding_modules)

    if args.include_all_eligible_modules:
        args.include_attention_norms = True

    if args.include_attention_norms:
        norm_modules = _merge_unique(norm_modules, ["q_norm", "k_norm"])

    return CriticalNeuronConfig(
        base_model_name_or_path=args.base,
        row_modules=row_modules,
        column_modules=col_modules,
        norm_modules=norm_modules,
        embedding_modules=emb_modules,
        sparsity_ratio=args.sparsity_ratio,
        neuron_indices=neuron_indices,
    )


def _verify_freeze(model) -> None:
    """Assert that only dW adapter params are trainable."""
    leaks = [
        name for name, p in model.named_parameters()
        if p.requires_grad and ".dW" not in name
    ]
    if leaks:
        raise RuntimeError(
            f"Freeze check failed: {len(leaks)} non-adapter params are trainable. "
            f"First 5: {leaks[:5]}"
        )


def main() -> None:
    # Launch with: accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml neuron_tuning.py ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--stype", type=str, default="lima_s1", help="Dataset type for process_data")
    parser.add_argument("--neuron_path", type=str, required=True, help="Path to neuron JSON file or toolkit folder")
    parser.add_argument("--training_config", type=str, default="lima_s1_exp/configs/exp_config/sft_config.yaml")
    parser.add_argument("--output_adapter_dir", type=str, default="saved_adapters/neuron_tuning")
    parser.add_argument("--output_model_dir", type=str, default="saved_models/neuron_tuning")
    parser.add_argument("--save_merged_model", action="store_true", help="Merge sparse adapter and save full model")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--langs", type=str, default="en,zh,ar,sw", help="Comma-separated languages")
    parser.add_argument("--sparsity_ratio", type=float, default=0.05)

    parser.add_argument("--row_modules", type=str, default=None)
    parser.add_argument("--column_modules", type=str, default=None)
    parser.add_argument("--norm_modules", type=str, default=None)
    parser.add_argument("--embedding_modules", type=str, default=None)
    parser.add_argument("--include_all_eligible_modules", action="store_true")
    parser.add_argument("--include_attention_norms", action="store_true")
    args = parser.parse_args()

    _apply_liger_kernels(args.base)

    model = AutoModelForCausalLM.from_pretrained(args.base)
    tokenizer, _ = prepare_tokenizer(args.base)

    config = _load_or_build_config(args)
    neuron_model = get_neuron_model(model, config)
    neuron_model.print_trainable_parameters()
    _verify_freeze(neuron_model)

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    cfg = OmegaConf.load(args.training_config)
    trainer_args_dict = OmegaConf.to_container(cfg)
    trainer_args_dict["output_dir"] = str(
        Path("checkpoints/neuron_tuning") / args.base
    )
    # Liger kernels are already applied to the model class above;
    # disable the Trainer's own attempt (which fails on non-PreTrainedModel).
    trainer_args_dict["use_liger_kernel"] = False
    training_args = hf_parser.parse_dict(trainer_args_dict)[0]

    text_dct = process_data(args.stype)
    selected_langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    texts = []
    for lang in selected_langs:
        if lang in text_dct:
            texts.extend(text_dct[lang])

    if not texts:
        raise ValueError(f"No training samples found for langs={selected_langs}")

    train_dataset = build_lm_dataset(texts, tokenizer, args.max_length)

    trainer = Trainer(
        model=neuron_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate_fn(batch, args.max_length, tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = Path(args.output_adapter_dir) / args.stype / args.base
    adapter_dir.mkdir(parents=True, exist_ok=True)
    neuron_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.save_merged_model:
        merged_model = neuron_model.merge_and_unload()
        model_dir = Path(args.output_model_dir) / args.stype / args.base
        model_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))


if __name__ == "__main__":
    main()
