#!/usr/bin/env bash
# Zero-out critical neurons and save full model (lima_s1_exp/exp3_safety_neuron_deactivation/deactivation.py).
# For many pairwise neuron sets + safety ASR sweep, see lima_s1_exp/scripts/exp2_refusal_utility_grid_search/pairwise_deactivate_eval_safe.sh.

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
RATIO="${RATIO:-0.05}"
STATS_SAVE="${STATS_SAVE:-./neuron_deactivation}"
NEURONS_FILE="${NEURONS_FILE:-${STATS_SAVE}/${BASE_MODEL}/ratio${RATIO}/detect_union_neurons.json}"
OUT_DIR="${OUT_DIR:-./deactivate_model_param/${BASE_MODEL}/ratio${RATIO}/union}"
INCLUDE_ALL="${INCLUDE_ALL:-1}"

cd "$(dirname "$0")/../../../" || exit 1

if [[ ! -e "${NEURONS_FILE}" ]]; then
  echo "ERROR: NEURONS_FILE not found: ${NEURONS_FILE}"
  exit 1
fi

extra=()
[[ "${INCLUDE_ALL}" == "1" ]] && extra+=(--include_all_eligible_modules)

python lima_s1_exp/exp3_safety_neuron_deactivation/deactivation.py \
  --model_path "${BASE_MODEL}" \
  --neurons_file "${NEURONS_FILE}" \
  --save_path "${OUT_DIR}" \
  "${extra[@]}"

echo "Deactivated model saved to: ${OUT_DIR}"
