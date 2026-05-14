#!/usr/bin/env bash
# Pairwise ratio statistics: two detection stypes (e.g. safety vs lima_s1) over ratio grids.
# Expects neuron_detection layout: NEURONS_PATH/MODEL/STYPE/ratio{R}/{NEURON_LANG}/neuron_indices.json
# Use NEURON_LANG (not LANG): LANG is the system locale (e.g. en_US.UTF-8), not the data subfolder "en".

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
NEURONS_PATH="${NEURONS_PATH:-./neuron_detect}"
SAVE_PATH="${SAVE_PATH:-./neuron_pairwise_stats}"
# ratio_a applies to STYPE_A, ratio_b to STYPE_B; pair dirs are STYPE_A{r_a}_STYPE_B{r_b}
STYPE_A="${STYPE_A:-lima_s1}"
STYPE_B="${STYPE_B:-safety}"
NEURON_LANG="${NEURON_LANG:-en}"
# cartesian | diagonal | upper_triangle
PAIR_MODE="${PAIR_MODE:-cartesian}"
# Optional: comma-separated ratios; empty = 0.02..0.98 step 0.02
RATIOS="${RATIOS:-}"
INCLUDE_ALL="${INCLUDE_ALL:-1}"
WRITE_REPORTS="${WRITE_REPORTS:-0}"
NO_PAIR_NEURON_JSON="${NO_PAIR_NEURON_JSON:-0}"
# 1 = sequential + load HF model; 0 or unset = auto worker count (CPU-only, no model)
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$(dirname "$0")/../../../" || exit 1

extra=()
[[ "${INCLUDE_ALL}" == "1" ]] && extra+=(--include_all_eligible_modules)
[[ "${WRITE_REPORTS}" == "1" ]] && extra+=(--write_per_pair_reports)
[[ "${NO_PAIR_NEURON_JSON}" == "1" ]] && extra+=(--no_pair_neuron_json)
extra+=(--num_workers "${NUM_WORKERS}")
[[ -n "${RATIOS}" ]] && extra+=(--ratios "${RATIOS}")

python lima_s1_exp/exp2_refusal_utility_grid_search/pairwise_statistics.py \
  --model_name "${BASE_MODEL}" \
  --neurons_path "${NEURONS_PATH}" \
  --save_path "${SAVE_PATH}" \
  --stype_a "${STYPE_A}" \
  --stype_b "${STYPE_B}" \
  --lang "${NEURON_LANG}" \
  --pair_mode "${PAIR_MODE}" \
  "${extra[@]}"

echo "Done. Summary: ${SAVE_PATH}/${BASE_MODEL}/pairwise_summary.csv"
