#!/usr/bin/env bash
# Cross-language neuron statistics (lima_s1_exp/exp5_multilingual_critical_foundational/neuron_statistics.py).

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
RATIO="${RATIO:-0.05}"
NEURONS_PATH="${NEURONS_PATH:-./neuron_detect}"
SAVE_PATH="${SAVE_PATH:-./neuron_statistics}"
LANGS="${LANGS:-en,zh,ar,sw}"
INCLUDE_ALL="${INCLUDE_ALL:-1}"

cd "$(dirname "$0")/../../../" || exit 1

extra=()
[[ "${INCLUDE_ALL}" == "1" ]] && extra+=(--include_all_eligible_modules)

python lima_s1_exp/exp5_multilingual_critical_foundational/neuron_statistics.py \
  --model_name "${BASE_MODEL}" \
  --neurons_path "${NEURONS_PATH}" \
  --save_path "${SAVE_PATH}" \
  --ratio "${RATIO}" \
  --langs "${LANGS}" \
  "${extra[@]}"

echo "Done. Reports under: ${SAVE_PATH}/${BASE_MODEL}/ratio${RATIO}/"
