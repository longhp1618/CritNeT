#!/usr/bin/env python3
"""Deactivate union neurons for each pairwise folder, run safety eval (ASR), remove checkpoint.

Runs up to ``max_workers`` jobs in parallel. Each job pins one GPU via
``CUDA_VISIBLE_DEVICES`` (subprocess isolation). Best pair = **highest** ID ASR.
Pairs that already have a valid ``metrics.json`` under the safety predictions tree are
skipped (no deactivation / re-eval). Merges into ``lima_s1_exp/eval/predictions/safety/best.json``
after loading cached results and after **each** newly finished pair:

``{"<model_id>": {"best_ASR": float, "best_ratio": {"lima_s1": float, "safety": float}}}``
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_gpus(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_lima_safety_ratios(pair_subdir: str) -> Optional[Tuple[float, float]]:
    """Parse ``lima_s10.02_safety0.36`` → (0.02, 0.36) for lima_s1 and safety."""
    m = re.match(r"^lima_s1([\d.]+)_safety([\d.]+)$", pair_subdir)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def make_best_json_entry(result_row: Dict[str, Any]) -> Dict[str, Any]:
    """``best.json`` fragment for one model from a winning sweep row."""
    parsed = parse_lima_safety_ratios(result_row["pair_subdir"])
    if parsed:
        lima_r, safety_r = parsed
        ratio_obj: Dict[str, Any] = {"lima_s1": lima_r, "safety": safety_r}
    else:
        ratio_obj = {"lima_s1": None, "safety": None}
    return {"best_ASR": result_row["id_asr"], "best_ratio": ratio_obj}


def load_merged_best_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        return dict(loaded) if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def try_load_existing_eval_result(
    repo: Path,
    base_model: str,
    pair_dir: Path,
    neurons_json: str,
) -> Optional[Dict[str, Any]]:
    """If ``metrics.json`` exists and has ``id_asr``, return a result row (skip re-run)."""
    pair_name = pair_dir.name
    neurons_file = pair_dir / neurons_json
    pred_root = repo / "lima_s1_exp" / "eval" / "predictions" / "safety" / base_model / pair_name
    metrics_path = pred_root / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        id_asr = float(data["id_asr"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
        return None
    return {
        "ok": True,
        "pair_subdir": pair_name,
        "id_asr": id_asr,
        "union_neurons_file": str(neurons_file.resolve()),
        "eval_prediction_dir": str(pred_root.resolve()),
        "metrics_json": str(metrics_path.resolve()),
        "gpu": -1,
        "skipped_existing_eval": True,
    }


def discover_pair_dirs(
    root: Path,
    neurons_json: str,
    name_filter: str,
) -> List[Path]:
    """Default filter: dirname starts with ``lima_s1`` and contains ``_safety``."""
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if name_filter == "lima_safety":
            if not (p.name.startswith("lima_s1") and "_safety" in p.name):
                continue
        elif name_filter != "any":
            raise ValueError(f"Unknown --pair_dir_filter: {name_filter}")
        nf = p / neurons_json
        if nf.is_file():
            out.append(p)
    return out


def run_one_job(
    repo: Path,
    base_model: str,
    pair_dir: Path,
    gpu: int,
    neurons_json: str,
    ratio: float,
) -> Dict[str, Any]:
    pair_name = pair_dir.name
    neurons_file = pair_dir / neurons_json
    safe_tag = base_model.replace("/", "__")
    out_model = repo / "deactivate_tmp" / safe_tag / pair_name
    metrics_path = repo / "lima_s1_exp" / "eval" / "predictions" / "safety" / base_model / pair_name / "metrics.json"
    pred_root = repo / "lima_s1_exp" / "eval" / "predictions" / "safety" / base_model / pair_name

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        if out_model.exists():
            shutil.rmtree(out_model, ignore_errors=True)
        out_model.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                sys.executable,
                str(repo / "lima_s1_exp" / "exp3_safety_neuron_deactivation" / "deactivation.py"),
                "--model_path",
                base_model,
                "--neurons_file",
                str(neurons_file),
                "--save_path",
                str(out_model),
                "--include_all_eligible_modules",
            ],
            cwd=str(repo),
            env=env,
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(repo / "lima_s1_exp" / "eval" / "eval_safe.py"),
                "--model_name",
                base_model,
                "--local_model_path",
                str(out_model),
                "--template",
                "--sgl_tp",
                "1",
                "--sgl_dp",
                "1",
                "--result_subdir",
                pair_name,
                "--metrics_json",
                str(metrics_path),
            ],
            cwd=str(repo),
            env=env,
            check=True,
        )

        id_asr: Optional[float] = None
        if metrics_path.is_file():
            with open(metrics_path, encoding="utf-8") as f:
                id_asr = float(json.load(f)["id_asr"])

        return {
            "ok": True,
            "pair_subdir": pair_name,
            "id_asr": id_asr,
            "union_neurons_file": str(neurons_file.resolve()),
            "eval_prediction_dir": str(pred_root.resolve()),
            "metrics_json": str(metrics_path.resolve()),
            "gpu": gpu,
        }
    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "pair_subdir": pair_name,
            "error": f"subprocess exit {e.returncode}",
            "gpu": gpu,
        }
    except Exception as e:
        return {
            "ok": False,
            "pair_subdir": pair_name,
            "error": str(e),
            "gpu": gpu,
        }
    finally:
        if out_model.exists():
            shutil.rmtree(out_model, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".", help="Project root (contains deactivation.py, eval/).")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF id for base weights, tokenizer layout, and prediction paths.",
    )
    parser.add_argument(
        "--pair_stats_root",
        type=str,
        default=None,
        help="Directory containing lima_s1*safety* subfolders (default: neuron_pairwise_stats/<base_model>).",
    )
    parser.add_argument(
        "--neurons_json",
        type=str,
        default="exclusive_safety_neurons.json",
        help="Neuron JSON inside each pair folder (passed to deactivation.py).",
    )
    parser.add_argument("--deactivate_ratio", type=float, default=0.05)
    parser.add_argument(
        "--gpus",
        type=str,
        default="4,5,6,7",
        help="Comma-separated physical GPU ids; round-robin across jobs.",
    )
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--pair_dir_filter",
        choices=("lima_safety", "any"),
        default="lima_safety",
        help="lima_safety: only dirs like lima_s10.02_safety0.36; any: every subdir with neurons_json.",
    )
    parser.add_argument(
        "--best_json",
        type=str,
        default=None,
        help="Default: lima_s1_exp/eval/predictions/safety/best.json (merged across models).",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    pair_root = (
        Path(args.pair_stats_root)
        if args.pair_stats_root
        else repo / "neuron_pairwise_stats" / args.base_model
    ).resolve()

    best_json = (
        Path(args.best_json)
        if args.best_json
        else repo / "lima_s1_exp" / "eval" / "predictions" / "safety" / "best.json"
    ).resolve()

    if not pair_root.is_dir():
        raise FileNotFoundError(f"pair_stats_root is not a directory: {pair_root}")

    pairs = discover_pair_dirs(pair_root, args.neurons_json, args.pair_dir_filter)
    if not pairs:
        raise FileNotFoundError(
            f"No pair directories with {args.neurons_json} under {pair_root} (filter={args.pair_dir_filter})."
        )

    gpus = _parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one id.")
    # One thread per GPU (cap at max_workers) so two jobs never share a GPU concurrently.
    n_threads = max(1, min(args.max_workers, len(gpus)))

    print(f"Found {len(pairs)} pair dirs under {pair_root}", file=sys.stderr)
    print(f"Using {n_threads} worker threads on GPUs {gpus[:n_threads]}", file=sys.stderr)

    best_json.parent.mkdir(parents=True, exist_ok=True)
    merged: Dict[str, Any] = load_merged_best_json(best_json)

    results: List[Dict[str, Any]] = []
    pending_pairs: List[Path] = []
    for pdir in pairs:
        cached = try_load_existing_eval_result(repo, args.base_model, pdir, args.neurons_json)
        if cached is not None:
            results.append(cached)
        else:
            pending_pairs.append(pdir)

    n_cached = len(results)
    print(
        f"Eval cache: {n_cached} pairs skipped (existing metrics.json), {len(pending_pairs)} to run.",
        file=sys.stderr,
    )

    lock = threading.Lock()

    def recompute_and_save_best_json() -> None:
        """Recompute best for ``base_model`` (highest ASR) and write ``best.json``."""
        ok_asr = [r for r in results if r.get("ok") and r.get("id_asr") is not None]
        if ok_asr:
            b = max(ok_asr, key=lambda x: float(x["id_asr"]))
            entry = make_best_json_entry(b)
            if entry["best_ratio"]["lima_s1"] is None:
                print(
                    f"Warning: could not parse lima_s1/safety ratios from {b['pair_subdir']!r}; "
                    "best_ratio set to nulls.",
                    file=sys.stderr,
                )
            merged[args.base_model] = entry
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        if ok_asr:
            b = max(ok_asr, key=lambda x: float(x["id_asr"]))
            print(
                f"Updated {best_json}  {args.base_model!r}  best_ASR={b['id_asr']}  pair={b['pair_subdir']}",
                file=sys.stderr,
            )

    with lock:
        recompute_and_save_best_json()

    job_q: queue.Queue[Path] = queue.Queue()
    for pdir in pending_pairs:
        job_q.put(pdir)

    def worker_loop(gpu: int) -> None:
        while True:
            try:
                pdir = job_q.get_nowait()
            except queue.Empty:
                return
            r = run_one_job(repo, args.base_model, pdir, gpu, args.neurons_json, args.deactivate_ratio)
            with lock:
                results.append(r)
                recompute_and_save_best_json()

    threads = [
        threading.Thread(target=worker_loop, args=(gpus[i],))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    ok_asr = [r for r in ok if r.get("id_asr") is not None]
    if ok_asr:
        final = max(ok_asr, key=lambda x: float(x["id_asr"]))
        print(
            f"Finished sweep: {len(ok)} ok, {len(failed)} failed. "
            f"Final best (highest ASR): {final['pair_subdir']}  best_ASR={final['id_asr']}",
            file=sys.stderr,
        )
    else:
        print(
            f"Finished sweep: {len(ok)} ok, {len(failed)} failed. "
            f"No ASR in this run; {args.base_model!r} entry in best.json unchanged unless from a prior run.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
