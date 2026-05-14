import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from critnet import CriticalNeuronConfig, NeuronDeactivator


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
        idx_file = path / "neuron_indices.json"
        if not idx_file.is_file():
            raise FileNotFoundError(f"Missing neuron_indices.json in {path}")
        with open(idx_file, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_config(args: argparse.Namespace, neuron_indices: Dict[str, List[int]]) -> CriticalNeuronConfig:
    row_modules = _parse_modules(args.row_modules)
    col_modules = _parse_modules(args.column_modules)
    norm_modules = _parse_modules(args.norm_modules)
    emb_modules = _parse_modules(args.embedding_modules)

    if args.include_all_eligible_modules:
        args.include_attention_norms = True

    if args.include_attention_norms:
        norm_modules = _merge_unique(norm_modules, ["q_norm", "k_norm"])

    return CriticalNeuronConfig(
        base_model_name_or_path=args.model_path,
        row_modules=row_modules,
        column_modules=col_modules,
        norm_modules=norm_modules,
        embedding_modules=emb_modules,
        sparsity_ratio=args.ratio,
        neuron_indices=neuron_indices,
    )


def main() -> None:
    # Example:
    # python -m lima_s1_exp.exp3_safety_neuron_deactivation.deactivation --model_path "Qwen/Qwen3-0.6B" --neurons_file "./neuron_deactivation/<model>/ratio0.05/detect_union_neurons.json" --save_path "./deactivate_model_param/<model>/ratio0.05/union"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Base model path or HF id")
    parser.add_argument("--neurons_file", type=str, required=True, help="Neuron JSON file or toolkit folder")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory for deactivated model")
    parser.add_argument("--ratio", type=float, default=0.05)

    parser.add_argument("--row_modules", type=str, default=None)
    parser.add_argument("--column_modules", type=str, default=None)
    parser.add_argument("--norm_modules", type=str, default=None)
    parser.add_argument("--embedding_modules", type=str, default=None)
    parser.add_argument("--include_all_eligible_modules", action="store_true")
    parser.add_argument("--include_attention_norms", action="store_true")
    args = parser.parse_args()

    neuron_path = Path(args.neurons_file)
    neuron_indices = _load_indices(neuron_path)

    # If config exists in the folder, prefer loading it to guarantee module typing matches detection.
    if neuron_path.is_dir() and (neuron_path / "critical_neuron_config.json").is_file():
        config = CriticalNeuronConfig.from_pretrained(str(neuron_path))
        config.base_model_name_or_path = args.model_path
    else:
        config = _build_config(args, neuron_indices)

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    deactivator = NeuronDeactivator(model=model, config=config)
    result = deactivator.deactivate(neuron_indices=neuron_indices)
    print(result.summary())

    out_dir = Path(args.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    deactivator.save_pretrained(str(out_dir), tokenizer=tokenizer)
    print(f"Saved deactivated model to: {out_dir}")


if __name__ == "__main__":
    main()
