"""Full fine-tuning with specific neurons frozen (safety-preserving FT).

Trains the entire model while preventing selected neurons from being
updated.  Gradient hooks zero out frozen-neuron gradients during
backward (zero forward overhead), and a Trainer callback restores
frozen weights after each optimizer step to counteract weight-decay
drift.

Launch with:
    accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml \
        lima_s1_exp/exp4_freeze_ft_attack/freeze_ft.py \
        --base Qwen/Qwen3-4B-Instruct-2507 \
        --freeze_neurons_path neuron_pairwise_stats/Qwen/Qwen3-4B-Instruct-2507/lima_s10.13_safety0.06/exclusive_safety_neurons.json \
        --stype lima_s1 \
        --langs en,zh,ar,sw
"""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.cache/triton")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from critnet import CriticalNeuronConfig, freeze_neurons
from lima_s1_exp.dataloader import build_lm_dataset, collate_fn
from lima_s1_exp.utils import prepare_tokenizer, process_data

logger = logging.getLogger(__name__)


def _apply_liger_kernels(model_name: str) -> bool:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning with frozen safety neurons."
    )
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--stype", type=str, default="lima_s1")
    parser.add_argument(
        "--freeze_neurons_path",
        type=str,
        required=True,
        help="Path to JSON file mapping module names to neuron indices to freeze.",
    )
    parser.add_argument(
        "--training_config", type=str, default="lima_s1_exp/configs/exp_config/sft_config.yaml"
    )
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--langs", type=str, default="en,zh,ar,sw",
        help="Comma-separated languages",
    )
    args = parser.parse_args()

    _apply_liger_kernels(args.base)

    model = AutoModelForCausalLM.from_pretrained(args.base)
    tokenizer, _ = prepare_tokenizer(args.base)
    model.enable_input_require_grads()

    with open(args.freeze_neurons_path, "r", encoding="utf-8") as f:
        freeze_indices = json.load(f)

    config = CriticalNeuronConfig()
    frozen_handle = freeze_neurons(model, freeze_indices, config)
    frozen_handle.print_frozen_summary()

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    cfg = OmegaConf.load(args.training_config)
    trainer_args_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer_args_dict.pop("lora", None)
    trainer_args_dict["output_dir"] = str(
        Path("checkpoints/freeze_ft") / args.base
    )
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate_fn(batch, args.max_length, tokenizer),
        processing_class=tokenizer,
        callbacks=[frozen_handle.make_trainer_callback()],
    )

    trainer.train()

    out_root = args.output_model_dir or "saved_models/freeze_ft"
    model_dir = Path(out_root) / args.stype / args.base
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    logger.info("Model saved to %s", model_dir)


if __name__ == "__main__":
    main()
