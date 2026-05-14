# LIMA-S1 experiments (`lima_s1_exp`)

This package holds **paper-specific** pipelines: fixed dataset layouts, Accelerate configs, training YAMLs, SGLang-based eval, and bash wrappers. Install from the repo root (`pip install -e .`) and run shells from the **repository root** so paths like `lima_s1_exp/...` resolve.

## Scope, dependencies, and outputs

**Typical extra dependencies:** `accelerate`, `peft` (where scripts use it), and the stack expected by `eval/` (e.g. `sglang`, `vllm` imports in safety/utility runners). The core library only needs `torch` / `transformers` / `tqdm`.

**Outputs (gitignored by default):** model checkpoints (`saved_models/`, `saved_adapters/`), detection artifacts (`neuron_detect/`, `neuron_statistics/`, `neuron_pairwise_stats/`), deactivation dirs (`deactivate_model_param/`), and predictions under `lima_s1_exp/eval/predictions/`. Create `lima_s1_exp/results/` if you want a single place for tables and logs.

Python entrypoints use **`python -m lima_s1_exp.eval.<module>`** or paths like `lima_s1_exp/exp4_freeze_ft_attack/freeze_ft.py` as in the shell files.

## Shared layout (paths and roles)

| Path | Role |
|------|------|
| [`eval/`](eval/) | `eval_safe.py`, `eval_utility.py`, `eval_utility_en.py`, pairwise / sweep drivers, `grader.py`, `parser.py`. Predictions default under `lima_s1_exp/eval/predictions/`. |
| [`configs/acc_config/`](configs/acc_config/) | DeepSpeed / DDP YAMLs for `accelerate launch`. |
| [`configs/exp_config/`](configs/exp_config/) | SFT / LoRA YAMLs consumed by `freeze_ft.py`, `full_ft.py`, `neuron_tuning.py`. |
| [`ft_datasets/`](ft_datasets/) | CSVs for `lima`, `s1`, and `safety` splits used by `lima_s1_exp.utils.process_data`. |
| [`scripts/`](scripts/) | One subdirectory per experiment; each script `cd`s to repo root then calls `python` / `accelerate`. |

---

<details>
<summary><strong>Experiment 1 — Utility-neuron ablation (English)</strong></summary>

**Goal:** Deactivate a small fraction of **utility-related** critical neurons (from `neuron_detect/.../lima_s1/...`) and measure impact on **English** utility benchmarks (Global-MMLU Lite path via `eval_utility_en`).

**Code:**

- [`exp1_utility_neuron_ablation_en/run_ablation.py`](exp1_utility_neuron_ablation_en/run_ablation.py) — thin CLI to `eval.lima_s1_en_deactivate_utility`.
- [`eval/lima_s1_en_deactivate_utility.py`](eval/lima_s1_en_deactivate_utility.py) — parallel jobs: load per-ratio neuron JSON, deactivate, run `lima_s1_exp.eval.eval_utility_en`, optionally clean temp checkpoints.

**Shell:**

- [`scripts/exp1_utility_neuron_ablation_en/lima_s1_en_deactivate_eval_utility.sh`](scripts/exp1_utility_neuron_ablation_en/lima_s1_en_deactivate_eval_utility.sh)

**Environment (examples):** `BASE_MODEL`, `DETECT_ROOT` (default `neuron_detect`), `GPUS`, `MAX_WORKERS`, `DP_SIZE`, `SKIP_EXISTING`, `DEBUG`.

**Outputs:** Deactivated checkpoints under `lima_s1_exp/eval/predictions/deactivate_tmp/...`; sweep summaries under `lima_s1_exp/eval/predictions/utility/...`.

</details>

<details>
<summary><strong>Experiment 2 — Refusal vs. utility grid search</strong></summary>

**Goal:** Build **pairwise** statistics over folders like `lima_s10.xx_safety0.yy` under `neuron_pairwise_stats/<BASE_MODEL>/`, then optionally run **safety** eval (ASR) while sweeping deactivation ratios.

**Code:**

- [`exp2_refusal_utility_grid_search/pairwise_statistics.py`](exp2_refusal_utility_grid_search/pairwise_statistics.py) — Cartesian (or other) pairing of `lima_s1` vs `safety` detection trees; writes `exclusive_safety_neurons.json` etc. per pair.
- [`exp2_refusal_utility_grid_search/pairwise_deactivate.py`](exp2_refusal_utility_grid_search/pairwise_deactivate.py) — helper for pairwise deactivation flows (if used by your pipeline).

**Shell:**

- [`scripts/exp2_refusal_utility_grid_search/neuron_pairwise_statistics.sh`](scripts/exp2_refusal_utility_grid_search/neuron_pairwise_statistics.sh)
- [`scripts/exp2_refusal_utility_grid_search/pairwise_deactivate_eval_safe.sh`](scripts/exp2_refusal_utility_grid_search/pairwise_deactivate_eval_safe.sh) — calls `python -m lima_s1_exp.eval.pairwise_deactivate_safety` with `PAIR_ROOT`, `BEST_JSON` (default `lima_s1_exp/eval/predictions/safety/best.json`), GPUs, workers.

**Prerequisites:** Populated `neuron_detect/` (or equivalent) for both stypes; disk for `neuron_pairwise_stats/`.

</details>

<details>
<summary><strong>Experiment 3 — Safety-neuron deactivation</strong></summary>

**Goal:** Zero out **safety-critical** neuron sets and evaluate **both** utility and safety metrics (English-centric scripts in this repo).

**Code:**

- [`exp3_safety_neuron_deactivation/deactivation.py`](exp3_safety_neuron_deactivation/deactivation.py) — CLI around `critnet.NeuronDeactivator`.
- [`exp3_safety_neuron_deactivation/eval_safe.py`](exp3_safety_neuron_deactivation/eval_safe.py), [`eval_utility.py`](exp3_safety_neuron_deactivation/eval_utility.py) — re-exports or wrappers pointing at `lima_s1_exp.eval` (for convenience).

**Shell:**

- [`scripts/exp3_safety_neuron_deactivation/neuron_deactivate.sh`](scripts/exp3_safety_neuron_deactivation/neuron_deactivate.sh) — expects `NEURONS_FILE` under `./neuron_deactivation/...`, writes `./deactivate_model_param/...`.

</details>

<details>
<summary><strong>Experiment 4 — Fine-tuning with frozen neurons (“attack” / baseline)</strong></summary>

**Goal:** **Full** fine-tuning while **freezing** a chosen neuron subset (e.g. exclusive safety neurons from pairwise stats) using `critnet.freeze_neurons`, plus baselines without freezing.

**Code:**

- [`exp4_freeze_ft_attack/freeze_ft.py`](exp4_freeze_ft_attack/freeze_ft.py) — Accelerate + HF Trainer, `--freeze_neurons_path`, `--training_config` defaulting to `lima_s1_exp/configs/exp_config/sft_config.yaml`, `--stype`, `--langs`.
- [`exp4_freeze_ft_attack/full_ft.py`](exp4_freeze_ft_attack/full_ft.py) — same data path, no freeze (comparison).

**Shell (from repo root):**

- [`scripts/exp4_freeze_ft_attack/freeze_ft.sh`](scripts/exp4_freeze_ft_attack/freeze_ft.sh), [`freeze_ft2.sh`](scripts/exp4_freeze_ft_attack/freeze_ft2.sh) — Qwen3-4B vs Llama-3.1-8B style defaults.
- [`scripts/exp4_freeze_ft_attack/full_ft.sh`](scripts/exp4_freeze_ft_attack/full_ft.sh), [`full_ft2.sh`](scripts/exp4_freeze_ft_attack/full_ft2.sh) — unfrozen baselines.
- [`scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh`](scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh) — after training: `python -m lima_s1_exp.eval.eval_safe` then `python -m lima_s1_exp.eval.eval_utility_en` with `MODEL_DIR`, `DP_SIZE`, `TASKS`.

**Typical artifact layout:** checkpoints under `saved_models/freeze_ft/<stype>/<model_id>/` or `saved_models/full/...`; `freeze_neurons_path` often under `neuron_pairwise_stats/<model_id>/lima_s1*_safety*/exclusive_safety_neurons.json`.

</details>

<details>
<summary><strong>Experiment 5 — Multilingual critical / foundational neurons</strong></summary>

**Goal:** Language-aware **detection** on LIMA/S1 CSVs, **statistics** across languages, optional **critical-neuron tuning** with merged adapter eval on multilingual utility tasks.

**Code:**

- [`exp5_multilingual_critical_foundational/neuron_detection.py`](exp5_multilingual_critical_foundational/neuron_detection.py) — `critnet.NeuronDetector` + `lima_s1_exp` data helpers; writes under `NEURONS_ROOT` (default `./neuron_detect`).
- [`exp5_multilingual_critical_foundational/neuron_statistics.py`](exp5_multilingual_critical_foundational/neuron_statistics.py) — `NeuronStatistician` wrapper.
- [`exp5_multilingual_critical_foundational/neuron_tuning.py`](exp5_multilingual_critical_foundational/neuron_tuning.py) — `get_neuron_model` + Accelerate training.

**Shell:**

- [`neuron_detect.sh`](scripts/exp5_multilingual_critical_foundational/neuron_detect.sh), [`neuron_statistics.sh`](scripts/exp5_multilingual_critical_foundational/neuron_statistics.sh), [`neuron_pipeline.sh`](scripts/exp5_multilingual_critical_foundational/neuron_pipeline.sh), [`neuron_tune.sh`](scripts/exp5_multilingual_critical_foundational/neuron_tune.sh)

**Environment:** `BASE_MODEL`, `STYPE` (`lima_s1` vs `safety`), `NEURONS_ROOT`, `NEURON_PATH` / `NEURONS_FILE`, `LANGS`, `ACC_CONFIG`, `TRAINING_CONFIG`, optional importance-cache dirs for long runs.

</details>

<details>
<summary><strong>Quick reference: folder → paper role</strong></summary>

| Folder | Manuscript-facing role |
|--------|-------------------------|
| `exp1_utility_neuron_ablation_en` | English utility ablation (small fraction of utility neurons). |
| `exp2_refusal_utility_grid_search` | Refusal–utility neuron ratio grid + safety ASR sweep. |
| `exp3_safety_neuron_deactivation` | Deactivate safety neurons; joint utility/safety readouts. |
| `exp4_freeze_ft_attack` | FT with frozen safety-related neurons vs full FT. |
| `exp5_multilingual_critical_foundational` | Multilingual detection, aggregation, tuning, eval hooks. |

For **library-only** usage (no LIMA-S1 data), use **`critnet`** and the root [README.md](../README.md).

</details>
