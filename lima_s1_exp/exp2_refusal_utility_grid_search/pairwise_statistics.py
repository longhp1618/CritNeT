#!/usr/bin/env python3
"""Cross-dataset neuron statistics for two detection runs (e.g. lima_s1 vs safety).

Loads neuron index dicts from ``{neurons_path}/{model}/{stype}/ratio{r}/{lang}/`` (or
legacy ``{lang}.json``), then runs :class:`NeuronStatistician` for each ratio pair.

With ``--num_workers 1`` (default), the HF model is loaded once on the main process.
With ``--num_workers`` > 1 or ``<= 0`` (auto), workers run CPU-only analysis and do not
load the model; they write per-pair JSON directly (large runs are much faster).

For each successful pair, writes under
``{save_path}/{model}/{stype_a}{r_a}_{stype_b}{r_b}/``:

* ``union_neurons.json``, ``shared_neurons.json``
* ``exclusive_{stype_a}_neurons.json``, ``exclusive_{stype_b}_neurons.json``

Default ``stype_a=lima_s1``, ``stype_b=safety`` yields folders like
``lima_s10.02_safety0.06``.

Example::

    python -m lima_s1_exp.exp2_refusal_utility_grid_search.pairwise_statistics \\
        --model_name meta-llama/Llama-3.1-8B-Instruct \\
        --neurons_path ./neuron_detect \\
        --stype_a lima_s1 --stype_b safety \\
        --lang en --pair_mode cartesian \\
        --save_path ./neuron_pairwise_stats
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

from critnet import CriticalNeuronConfig, NeuronStatistician, StatisticsResult


def _normalize_lang_label(lang: str) -> str:
    """Map accidental locale strings (e.g. ``en_US.UTF-8``) to dataset codes like ``en``."""
    lang = lang.strip()
    if "_" in lang and "." in lang:
        return lang.split("_", 1)[0].lower()
    return lang


def default_ratio_grid(start: float = 0.01, stop: float = 0.20, step: float = 0.01) -> List[float]:
    """Match detection sweep: 0.02, 0.04, …, 0.20 (same path labels as ``neuron_detection``)."""
    n = int(round((stop - start) / step)) + 1
    return [float(f"{(start + i * step):.2f}") for i in range(n)]


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


def _build_config(
    model_name: str,
    row_modules: Optional[str],
    column_modules: Optional[str],
    norm_modules: Optional[str],
    embedding_modules: Optional[str],
    include_all_eligible_modules: bool,
    include_attention_norms: bool,
) -> CriticalNeuronConfig:
    row_m = _parse_modules(row_modules)
    col_m = _parse_modules(column_modules)
    norm_m = _parse_modules(norm_modules)
    emb_m = _parse_modules(embedding_modules)
    if include_all_eligible_modules:
        include_attention_norms = True
    if include_attention_norms:
        norm_m = _merge_unique(norm_m, ["q_norm", "k_norm"])
    return CriticalNeuronConfig(
        base_model_name_or_path=model_name,
        row_modules=row_m,
        column_modules=col_m,
        norm_modules=norm_m,
        embedding_modules=emb_m,
    )


def _load_indices(path: Path) -> Dict[str, List[int]]:
    if path.is_dir():
        indices_path = path / "neuron_indices.json"
        if not indices_path.is_file():
            raise FileNotFoundError(f"Missing neuron_indices.json in {path}")
        with open(indices_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_task_indices(
    neurons_path: Path,
    model_name: str,
    stype: str,
    ratio: float,
    lang: str,
) -> Optional[Dict[str, List[int]]]:
    """Load one task's neuron dict from disk; return ``None`` if nothing is found."""
    ratio_dir = neurons_path / model_name / stype / f"ratio{ratio}"
    folder_path = ratio_dir / lang
    flat_path = ratio_dir / f"{lang}.json"
    try:
        if folder_path.is_dir():
            return _load_indices(folder_path)
        if flat_path.is_file():
            return _load_indices(flat_path)
    except FileNotFoundError:
        return None
    return None


def _neuron_count(indices: Dict[str, List[int]]) -> int:
    return sum(len(v) for v in indices.values())


def _ratio_tag(r: float) -> str:
    """Stable suffix for directory names (matches default ratio grid formatting)."""
    return f"{r:.2f}"


def pair_output_subdir(stype_a: str, stype_b: str, ratio_a: float, ratio_b: float) -> str:
    """e.g. ``lima_s10.02_safety0.06`` for lima_s1 @ 0.02 and safety @ 0.06."""
    return f"{stype_a}{_ratio_tag(ratio_a)}_{stype_b}{_ratio_tag(ratio_b)}"


def save_pair_neuron_jsons(
    pair_dir: Path,
    result: StatisticsResult,
    stype_a: str,
    stype_b: str,
) -> None:
    """Write union, shared, and per-stype exclusive neuron dicts (toolkit field names)."""
    save_pair_neuron_jsons_from_dicts(
        pair_dir,
        result.union,
        result.intersection,
        result.exclusive,
        stype_a,
        stype_b,
    )


def save_pair_neuron_jsons_from_dicts(
    pair_dir: Path,
    union: Dict[str, List[int]],
    intersection: Dict[str, List[int]],
    exclusive: Dict[str, Dict[str, List[int]]],
    stype_a: str,
    stype_b: str,
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    with open(pair_dir / "union_neurons.json", "w", encoding="utf-8") as f:
        json.dump(union, f, indent=2)
    with open(pair_dir / "shared_neurons.json", "w", encoding="utf-8") as f:
        json.dump(intersection, f, indent=2)
    with open(pair_dir / f"exclusive_{stype_a}_neurons.json", "w", encoding="utf-8") as f:
        json.dump(exclusive.get(stype_a, {}), f, indent=2)
    with open(pair_dir / f"exclusive_{stype_b}_neurons.json", "w", encoding="utf-8") as f:
        json.dump(exclusive.get(stype_b, {}), f, indent=2)


def _analyze_pair_worker(job: Tuple[str, str, str, str, str, float, float, str, bool]) -> Dict[str, Any]:
    """CPU-only pair analysis + optional JSON write. Picklable for ProcessPoolExecutor."""
    neurons_path, model_name, stype_a, stype_b, lang, ra, rb, out_root_str, write_json = job
    root = Path(neurons_path)
    idx_a = extract_task_indices(root, model_name, stype_a, ra, lang)
    idx_b = extract_task_indices(root, model_name, stype_b, rb, lang)
    base: Dict[str, Any] = {
        "stype_a": stype_a,
        "stype_b": stype_b,
        "ratio_a": ra,
        "ratio_b": rb,
        "lang": lang,
    }
    if idx_a is None or idx_b is None:
        return {
            **base,
            "status": "missing_file",
            "count_a": _neuron_count(idx_a) if idx_a else "",
            "count_b": _neuron_count(idx_b) if idx_b else "",
        }
    task_indices = {stype_a: idx_a, stype_b: idx_b}
    stat = NeuronStatistician(model=None, config=None)
    result = stat.analyze(task_indices)
    ca = _neuron_count(idx_a)
    cb = _neuron_count(idx_b)
    inter = result.intersection_count
    uni = result.union_count
    jacc = (inter / uni) if uni else 0.0
    dice = (2 * inter / (ca + cb)) if (ca + cb) else 0.0
    overlap_min = (inter / min(ca, cb)) if min(ca, cb) else 0.0
    if write_json:
        pair_dir = Path(out_root_str) / pair_output_subdir(stype_a, stype_b, ra, rb)
        save_pair_neuron_jsons_from_dicts(
            pair_dir,
            result.union,
            result.intersection,
            result.exclusive,
            stype_a,
            stype_b,
        )
    return {
        **base,
        "status": "ok",
        "count_a": ca,
        "count_b": cb,
        "intersection_count": inter,
        "union_count": uni,
        "jaccard": f"{jacc:.6f}",
        "dice": f"{dice:.6f}",
        "overlap_min": f"{overlap_min:.6f}",
    }


def iter_ratio_pairs(
    ratios: List[float],
    pair_mode: str,
) -> List[Tuple[float, float]]:
    if pair_mode == "cartesian":
        return list(product(ratios, ratios))
    if pair_mode == "diagonal":
        return [(r, r) for r in ratios]
    if pair_mode == "upper_triangle":
        out: List[Tuple[float, float]] = []
        for i, a in enumerate(ratios):
            for b in ratios[i:]:
                out.append((a, b))
        return out
    raise ValueError(f"Unknown pair_mode: {pair_mode}")


def resolve_worker_count(num_workers: int) -> int:
    """``<= 0`` means auto: cap at 16, at least 1."""
    if num_workers <= 0:
        return max(1, min(16, os.cpu_count() or 4))
    return num_workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise ratio statistics between two detection stypes.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--neurons_path", type=str, default="./neuron_detect")
    parser.add_argument("--save_path", type=str, default="./neuron_pairwise_stats")
    parser.add_argument(
        "--stype_a",
        type=str,
        default="lima_s1",
        help="First dataset key; first ratio in each pair applies here (default lima_s1).",
    )
    parser.add_argument(
        "--stype_b",
        type=str,
        default="safety",
        help="Second dataset key; second ratio applies here (default safety).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Dataset language subfolder / json stem (not system LC_MESSAGES; use 'en' not en_US.UTF-8).",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help="Comma-separated ratios, or omit to use 0.02–0.20 step 0.02.",
    )
    parser.add_argument(
        "--pair_mode",
        choices=("cartesian", "diagonal", "upper_triangle"),
        default="cartesian",
        help="cartesian: all (r_a,r_b) with r_a for stype_a, r_b for stype_b; diagonal; upper_triangle.",
    )
    parser.add_argument(
        "--no_pair_neuron_json",
        action="store_true",
        help="Do not write union/shared/exclusive JSON under each pair directory (summary CSV only).",
    )
    parser.add_argument(
        "--write_per_pair_reports",
        action="store_true",
        help="Also run statistician.save_report (statistics.csv, non_shared, etc.) in each pair directory.",
    )
    parser.add_argument("--skip_missing", action="store_true", default=True, help="Skip pairs with missing neuron files.")
    parser.add_argument("--no_skip_missing", action="store_false", dest="skip_missing")
    parser.add_argument("--device_map", type=str, default=None, help="Optional HF device_map (e.g. auto).")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")

    parser.add_argument("--row_modules", type=str, default=None)
    parser.add_argument("--column_modules", type=str, default=None)
    parser.add_argument("--norm_modules", type=str, default=None)
    parser.add_argument("--embedding_modules", type=str, default=None)
    parser.add_argument("--include_all_eligible_modules", action="store_true")
    parser.add_argument("--include_attention_norms", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel processes for pair jobs. 1 = sequential (loads model once). "
        "<=0 = auto (min(CPU, 16)). Workers are CPU-only and write JSON from disk.",
    )
    args = parser.parse_args()
    args.lang = _normalize_lang_label(args.lang)

    if args.ratios:
        ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    else:
        ratios = default_ratio_grid()

    pairs = iter_ratio_pairs(ratios, args.pair_mode)
    neurons_root = Path(args.neurons_path)
    out_root = Path(args.save_path) / args.model_name
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "pairwise_summary.csv"

    nw = resolve_worker_count(args.num_workers)
    if nw > 1 and args.write_per_pair_reports:
        print(
            "--write_per_pair_reports is ignored when num_workers > 1; use --num_workers 1 for full per-pair CSV.",
            file=sys.stderr,
        )

    fieldnames = [
        "stype_a",
        "stype_b",
        "ratio_a",
        "ratio_b",
        "lang",
        "count_a",
        "count_b",
        "intersection_count",
        "union_count",
        "jaccard",
        "dice",
        "overlap_min",
        "status",
    ]

    n_ok = 0
    n_skip = 0
    with open(summary_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        if nw == 1:
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
            load_kw: Dict[str, object] = {"torch_dtype": dtype}
            if args.device_map:
                load_kw["device_map"] = args.device_map

            print(f"Loading model once: {args.model_name}", file=sys.stderr)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kw)

            config = _build_config(
                args.model_name,
                args.row_modules,
                args.column_modules,
                args.norm_modules,
                args.embedding_modules,
                args.include_all_eligible_modules,
                args.include_attention_norms,
            )
            statistician = NeuronStatistician(model=model, config=config)

            for ra, rb in pairs:
                idx_a = extract_task_indices(neurons_root, args.model_name, args.stype_a, ra, args.lang)
                idx_b = extract_task_indices(neurons_root, args.model_name, args.stype_b, rb, args.lang)
                if idx_a is None or idx_b is None:
                    if not args.skip_missing:
                        raise FileNotFoundError(
                            f"Missing neurons for pair ({ra}, {rb}): "
                            f"a={idx_a is not None} b={idx_b is not None}"
                        )
                    writer.writerow(
                        {
                            "stype_a": args.stype_a,
                            "stype_b": args.stype_b,
                            "ratio_a": ra,
                            "ratio_b": rb,
                            "lang": args.lang,
                            "count_a": _neuron_count(idx_a) if idx_a else "",
                            "count_b": _neuron_count(idx_b) if idx_b else "",
                            "intersection_count": "",
                            "union_count": "",
                            "jaccard": "",
                            "dice": "",
                            "overlap_min": "",
                            "status": "missing_file",
                        }
                    )
                    n_skip += 1
                    continue

                task_indices = {args.stype_a: idx_a, args.stype_b: idx_b}
                result = statistician.analyze(task_indices)
                ca = _neuron_count(idx_a)
                cb = _neuron_count(idx_b)
                inter = result.intersection_count
                uni = result.union_count
                jacc = (inter / uni) if uni else 0.0
                dice = (2 * inter / (ca + cb)) if (ca + cb) else 0.0
                overlap_min = (inter / min(ca, cb)) if min(ca, cb) else 0.0

                writer.writerow(
                    {
                        "stype_a": args.stype_a,
                        "stype_b": args.stype_b,
                        "ratio_a": ra,
                        "ratio_b": rb,
                        "lang": args.lang,
                        "count_a": ca,
                        "count_b": cb,
                        "intersection_count": inter,
                        "union_count": uni,
                        "jaccard": f"{jacc:.6f}",
                        "dice": f"{dice:.6f}",
                        "overlap_min": f"{overlap_min:.6f}",
                        "status": "ok",
                    }
                )
                n_ok += 1

                pair_dir = out_root / pair_output_subdir(args.stype_a, args.stype_b, ra, rb)
                if not args.no_pair_neuron_json:
                    save_pair_neuron_jsons(pair_dir, result, args.stype_a, args.stype_b)
                if args.write_per_pair_reports:
                    result.save_report(str(pair_dir))
        else:
            print(f"Parallel mode: {nw} workers (no model load; mp_context=spawn).", file=sys.stderr)
            jobs = [
                (
                    str(neurons_root.resolve()),
                    args.model_name,
                    args.stype_a,
                    args.stype_b,
                    args.lang,
                    ra,
                    rb,
                    str(out_root.resolve()),
                    not args.no_pair_neuron_json,
                )
                for ra, rb in pairs
            ]
            results: List[Dict[str, Any]] = []
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as pool:
                futures = [pool.submit(_analyze_pair_worker, job) for job in jobs]
                for fut in as_completed(futures):
                    results.append(fut.result())

            results.sort(key=lambda r: (float(r["ratio_a"]), float(r["ratio_b"])))
            for row in results:
                if row["status"] == "missing_file":
                    if not args.skip_missing:
                        raise FileNotFoundError(
                            f"Missing neurons for pair ({row['ratio_a']}, {row['ratio_b']})"
                        )
                    writer.writerow(
                        {
                            "stype_a": row["stype_a"],
                            "stype_b": row["stype_b"],
                            "ratio_a": row["ratio_a"],
                            "ratio_b": row["ratio_b"],
                            "lang": row["lang"],
                            "count_a": row["count_a"],
                            "count_b": row["count_b"],
                            "intersection_count": "",
                            "union_count": "",
                            "jaccard": "",
                            "dice": "",
                            "overlap_min": "",
                            "status": "missing_file",
                        }
                    )
                    n_skip += 1
                else:
                    writer.writerow(
                        {
                            "stype_a": row["stype_a"],
                            "stype_b": row["stype_b"],
                            "ratio_a": row["ratio_a"],
                            "ratio_b": row["ratio_b"],
                            "lang": row["lang"],
                            "count_a": row["count_a"],
                            "count_b": row["count_b"],
                            "intersection_count": row["intersection_count"],
                            "union_count": row["union_count"],
                            "jaccard": row["jaccard"],
                            "dice": row["dice"],
                            "overlap_min": row["overlap_min"],
                            "status": "ok",
                        }
                    )
                    n_ok += 1

    print(f"Wrote {summary_csv} ({n_ok} ok, {n_skip} skipped).", file=sys.stderr)


if __name__ == "__main__":
    main()
