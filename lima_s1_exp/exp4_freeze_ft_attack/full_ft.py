import argparse
import logging
import os
from pathlib import Path

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.cache/triton")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import transformers
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from lima_s1_exp.dataloader import build_lm_dataset, collate_fn
from lima_s1_exp.utils import prepare_tokenizer, process_data, resize_pad_embeddings

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


def main() -> None:
    # Launch with: accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml lima_s1_exp/exp4_freeze_ft_attack/full_ft.py ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        "--stype",
        type=str,
        default="lima_s1",
        help="Dataset type for process_data (lima_s1, wiki, fineweb, wiki_limas1, fineweb_limas1)",
    )
    parser.add_argument("--training_config", type=str, default="lima_s1_exp/configs/exp_config/sft_config.yaml")
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--langs", type=str, default="en,zh,ar,sw", help="Comma-separated languages")
    parser.add_argument(
        "--merge_before_save",
        action="store_true",
        help="If LoRA: merge adapters into base weights before save (larger disk; easier for engines that expect one folder).",
    )
    args = parser.parse_args()

    _apply_liger_kernels(args.base)

    if args.stype in ["wiki_limas1", "fineweb_limas1"]:
        pre = args.stype.split("_")[0]
        load_path = f"saved_models/full/{pre}/{args.base}"
        model = AutoModelForCausalLM.from_pretrained(load_path)
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base)
        tokenizer, _ = prepare_tokenizer(args.base)
        if "Mistral" in args.base:
            resize_pad_embeddings(model, tokenizer)

    model.enable_input_require_grads()

    hf_parser = transformers.HfArgumentParser(TrainingArguments)
    cfg = OmegaConf.load(args.training_config)
    trainer_args_dict = OmegaConf.to_container(cfg, resolve=True)
    lora_raw = trainer_args_dict.pop("lora", None)
    use_lora = isinstance(lora_raw, dict) and bool(lora_raw)

    if args.stype in ["wiki", "fineweb"]:
        trainer_args_dict["num_train_epochs"] = 1
    ckpt_subdir = "lora_ft" if use_lora else "full_ft"
    trainer_args_dict["output_dir"] = str(Path("checkpoints") / ckpt_subdir / args.base)
    trainer_args_dict["use_liger_kernel"] = False
    training_args = hf_parser.parse_dict(trainer_args_dict)[0]

    if use_lora:
        tm = lora_raw.get("target_modules")
        if not tm:
            tm = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=int(lora_raw.get("r", 16)),
            lora_alpha=int(lora_raw.get("lora_alpha", 32)),
            lora_dropout=float(lora_raw.get("lora_dropout", 0.05)),
            bias=str(lora_raw.get("bias", "none")),
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(tm),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
    )

    trainer.create_optimizer()
    num_opt = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
    num_req = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("optimizer params: %s  requires_grad params: %s", num_opt, num_req)

    trainer.train()

    if args.merge_before_save and use_lora and isinstance(model, PeftModel):
        model = model.merge_and_unload()

    out_root = args.output_model_dir
    if out_root is None:
        out_root = "saved_models/lora" if use_lora else "saved_models/full"
    model_dir = Path(out_root) / args.stype / args.base
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))


if __name__ == "__main__":
    main()
