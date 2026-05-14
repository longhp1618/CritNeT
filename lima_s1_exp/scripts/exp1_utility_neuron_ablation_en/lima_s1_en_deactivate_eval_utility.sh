#!/usr/bin/env bash
# Deactivate English lima_s1 neurons (per ratio folder under neuron_detect), run eval_utility_en (MMLU English-only), delete checkpoints.
# See lima_s1_exp/eval/lima_s1_en_deactivate_utility.py.
#
# Example:
#   bash lima_s1_exp/scripts/exp1_utility_neuron_ablation_en/lima_s1_en_deactivate_eval_utility.sh
#   GPUS=0 MAX_WORKERS=1 DP_SIZE=1 bash lima_s1_exp/scripts/exp1_utility_neuron_ablation_en/lima_s1_en_deactivate_eval_utility.sh

set -euo pipefail

cd "$(dirname "$0")/../../../" || exit 1

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DETECT_ROOT="${DETECT_ROOT:-neuron_detect}"
GPUS="${GPUS:-2,3}"
MAX_WORKERS="${MAX_WORKERS:-2}"
DP_SIZE="${DP_SIZE:-1}"
EXTRA=()
[[ "${SKIP_EXISTING:-1}" == "1" ]] && EXTRA+=(--skip_existing)
[[ "${DEBUG:-0}" == "1" ]] && EXTRA+=(--debug)

python -m lima_s1_exp.eval.lima_s1_en_deactivate_utility \
  --repo_root . \
  --base_model "${BASE_MODEL}" \
  --detect_root "${DETECT_ROOT}" \
  --stype lima_s1 \
  --lang en \
  --gpus "${GPUS}" \
  --max_workers "${MAX_WORKERS}" \
  --dp_size "${DP_SIZE}" \
  "${EXTRA[@]}" \
  "$@"

echo "Done. Predictions: lima_s1_exp/eval/predictions/deactivate_tmp/.../lima_s1_en_mmlu_r*/ ; sweep JSON: lima_s1_exp/eval/predictions/utility/*/lima_s1_en_mmlu_deactivate_sweep.json"
