#!/usr/bin/env python3
"""For each deactivation ratio: load English lima_s1 neurons, deactivate base model, run ``lima_s1_exp.eval.eval_utility_en``, delete checkpoint.

Mirrors the flow in ``pairwise_deactivate_safety.py`` / ``pairwise_deactivate_eval_safe.sh`` but for a single
``neuron_detect/<base>/lima_s1/ratio<r>/<lang>/`` tree. Evaluation is **MMLU English-only** (see ``eval_utility_en.py``).

For each ratio ``r``, neuron indices are read from ``<detect_root>/<base_model>/lima_s1/ratio<r>/<lang>/``
( toolkit folder with ``neuron_indices.json`` ), then ``deactivation.py`` is run with ``--ratio r``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_gpus(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def ratio_folder_name(r: float) -> str:
    """Stable decimal-style name (for tags only). Same float → same string (e.g. ``1e-5`` vs ``0.00001``)."""
    s = format(r, ".12f").rstrip("0").rstrip(".")
    if s == "" or s == "-":
        s = "0"
    return f"ratio{s}"


def _parse_ratio_suffix(dir_name: str) -> Optional[float]:
    """Parse ``ratio`` + suffix from neuron_detection dirs (``ratio0.0001``, ``ratio1e-07``, …)."""
    if not dir_name.startswith("ratio"):
        return None
    suffix = dir_name[len("ratio") :]
    if not suffix:
        return None
    try:
        return float(suffix)
    except ValueError:
        return None


def find_ratio_subdir(lima_stype_root: Path, r: float) -> Optional[Path]:
    """Resolve detection output folder: ``neuron_detection`` uses ``f'ratio{ratio}'`` (often scientific str)."""
    if not lima_stype_root.is_dir():
        return None
    # rel_tol handles float noise; abs_tol helps near-zero ratios
    for p in sorted(lima_stype_root.iterdir()):
        if not p.is_dir():
            continue
        v = _parse_ratio_suffix(p.name)
        if v is None:
            continue
        if math.isclose(v, r, rel_tol=1e-12, abs_tol=1e-18):
            return p
    return None


def list_ratio_dirs_on_disk(lima_stype_root: Path, limit: int = 40) -> List[str]:
    if not lima_stype_root.is_dir():
        return []
    names = sorted(
        p.name
        for p in lima_stype_root.iterdir()
        if p.is_dir() and p.name.startswith("ratio") and _parse_ratio_suffix(p.name) is not None
    )
    if len(names) > limit:
        return names[:limit] + [f"... and {len(names) - limit} more ratio* dirs"]
    return names


def neurons_en_dir(detect_root: Path, base_model: str, stype: str, r: float, lang: str) -> Tuple[Optional[Path], Path]:
    """Returns ``(ratio_dir, neuron_lang_path)``. ``ratio_dir`` is None if no matching ``ratio*`` folder."""
    lima = detect_root / base_model / stype
    ratio_dir = find_ratio_subdir(lima, r)
    if ratio_dir is None:
        return None, lima / ratio_folder_name(r) / lang
    return ratio_dir, ratio_dir / lang


def resolve_neurons_path(neuron_dir: Path) -> Path:
    """Directory with ``neuron_indices.json`` or sibling flat ``<lang>.json`` (``deactivation.py``)."""
    if neuron_dir.is_dir() and (neuron_dir / "neuron_indices.json").is_file():
        return neuron_dir
    flat = neuron_dir.parent / f"{neuron_dir.name}.json"
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"Expected {neuron_dir / 'neuron_indices.json'} or {flat} — missing English neuron file under {neuron_dir.parent}."
    )


def utility_prediction_dir(repo: Path, rel_model: str) -> Path:
    """Same layout rule as ``eval_utility_en.py`` for multi-slash ``model_name``."""
    sub = rel_model.replace("saved_models", "")
    return repo / "lima_s1_exp" / "eval" / "predictions" / sub


def try_skip_existing(repo: Path, rel_model: str) -> bool:
    summary_path = utility_prediction_dir(repo, rel_model) / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and len(data) > 0
    except (json.JSONDecodeError, OSError):
        return False


def run_one_ratio(
    repo: Path,
    base_model: str,
    gpu: int,
    r: float,
    *,
    detect_root: Path,
    stype: str,
    lang: str,
    dp_size: int,
    utility_extra: List[str],
    skip_existing: bool,
) -> Dict[str, Any]:
    safe_tag = base_model.replace("/", "__")
    # Output tag from float only so 1e-5 and 0.00001 share one prediction dir.
    r_tag = ratio_folder_name(r).replace("ratio", "r")
    # Subdir tag includes mmlu_en so outputs do not collide with full eval_utility runs.
    rel_model = f"deactivate_tmp/{safe_tag}/lima_s1_{lang}_mmlu_{r_tag}"
    out_model = repo / rel_model

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ratio_dir, neuron_dir = neurons_en_dir(detect_root, base_model, stype, r, lang)
    if ratio_dir is None:
        lima_root = detect_root / base_model / stype
        avail = list_ratio_dirs_on_disk(lima_root)
        return {
            "ok": False,
            "ratio": r,
            "error": (
                f"No neuron_detect folder matches ratio {r!r} under {lima_root} "
                f"(tried numeric match against ratio* dirs; e.g. 1e-5 vs 0.00001 is the same float). "
                f"Sample dirs: {avail}"
            ),
            "gpu": gpu,
        }
    try:
        neurons_file = resolve_neurons_path(neuron_dir)
    except FileNotFoundError as e:
        return {"ok": False, "ratio": r, "error": str(e), "gpu": gpu, "matched_ratio_dir": ratio_dir.name}

    if skip_existing and try_skip_existing(repo, rel_model):
        return {
            "ok": True,
            "ratio": r,
            "skipped_existing_eval": True,
            "neurons_file": str(neurons_file),
            "gpu": gpu,
        }

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
                "--ratio",
                str(r),
                "--include_all_eligible_modules",
            ],
            cwd=str(repo),
            env=env,
            check=True,
        )

        cmd = [
            sys.executable,
            "-m",
            "lima_s1_exp.eval.eval_utility_en",
            "--model_name",
            rel_model,
            "--dp_size",
            str(dp_size),
        ] + utility_extra
        subprocess.run(cmd, cwd=str(repo), env=env, check=True)

        pred = utility_prediction_dir(repo, rel_model)
        summary_path = pred / "summary.json"
        summary: Optional[Dict[str, Any]] = None
        if summary_path.is_file():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

        return {
            "ok": True,
            "ratio": r,
            "neurons_file": str(neurons_file),
            "eval_prediction_dir": str(pred.resolve()),
            "summary_json": str(summary_path.resolve()) if summary_path.is_file() else None,
            "summary": summary,
            "gpu": gpu,
        }
    except subprocess.CalledProcessError as e:
        return {"ok": False, "ratio": r, "error": f"subprocess exit {e.returncode}", "gpu": gpu}
    except Exception as e:
        return {"ok": False, "ratio": r, "error": str(e), "gpu": gpu}
    finally:
        if out_model.exists():
            shutil.rmtree(out_model, ignore_errors=True)


def main() -> None:
    default_ratios1 = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    default_ratios2 = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005]
    default_ratios = default_ratios1 + default_ratios2
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".", help="Project root.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF id; also relative path under detect_root.",
    )
    parser.add_argument(
        "--detect_root",
        type=str,
        default=None,
        help="Default: <repo>/neuron_detect",
    )
    parser.add_argument(
        "--stype",
        type=str,
        default="lima_s1",
        help="Subfolder under base_model (default lima_s1).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language subfolder / json for neuron indices.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        default=default_ratios,
        help=f"Deactivation ratios (default: {default_ratios}).",
    )
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids for workers.")
    parser.add_argument("--max_workers", type=int, default=1, help="Parallel jobs (one GPU each).")
    parser.add_argument("--dp_size", type=int, default=1, help="eval_utility_en --dp_size (default 1 per worker GPU).")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if summary.json already exists for that ratio's prediction dir.",
    )
    parser.add_argument("--debug", action="store_true", help="Forward --debug to eval_utility_en.")
    args, utility_extra = parser.parse_known_args()
    utility_extra = list(utility_extra)
    if args.debug and "--debug" not in utility_extra:
        utility_extra.insert(0, "--debug")

    repo = Path(args.repo_root).resolve()
    detect_root = Path(args.detect_root) if args.detect_root else repo / "neuron_detect"
    detect_root = detect_root.resolve()

    if not detect_root.is_dir():
        raise FileNotFoundError(f"detect_root is not a directory: {detect_root}")

    gpus = _parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one id.")
    n_threads = max(1, min(args.max_workers, len(gpus)))

    ratios = list(args.ratios)
    deduped: List[float] = []
    for x in ratios:
        if not any(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-18) for y in deduped):
            deduped.append(x)
    ratios = deduped
    print(
        f"Ratios: {ratios}\nDetect: {detect_root / args.base_model / args.stype}\n"
        f"Workers: {n_threads} on GPUs {gpus[:n_threads]} dp_size={args.dp_size}",
        file=sys.stderr,
    )

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    job_q: queue.Queue[float] = queue.Queue()
    for r in ratios:
        job_q.put(r)

    def worker_loop(gpu: int) -> None:
        while True:
            try:
                r = job_q.get_nowait()
            except queue.Empty:
                return
            out = run_one_ratio(
                repo,
                args.base_model,
                gpu,
                r,
                detect_root=detect_root,
                stype=args.stype,
                lang=args.lang,
                dp_size=args.dp_size,
                utility_extra=utility_extra,
                skip_existing=args.skip_existing,
            )
            with lock:
                results.append(out)
                tag = "SKIP" if out.get("skipped_existing_eval") else ("OK" if out.get("ok") else "FAIL")
                print(f"[{tag}] ratio={r} {out.get('error', '')}", file=sys.stderr)

    threads = [threading.Thread(target=worker_loop, args=(gpus[i],)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results.sort(key=lambda x: (x.get("ratio") is None, x.get("ratio") or 0.0))
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    report_path = (
        repo
        / "lima_s1_exp"
        / "eval"
        / "predictions"
        / "utility"
        / args.base_model.replace("/", "__")
        / f"lima_s1_{args.lang}_mmlu_deactivate_sweep.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"Wrote report: {report_path}", file=sys.stderr)
    print(f"Finished: {len(ok)} ok, {len(failed)} failed.", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
