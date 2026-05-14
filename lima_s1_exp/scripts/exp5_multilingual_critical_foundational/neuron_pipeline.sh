#!/usr/bin/env bash
# Run detection then statistics with shared settings.
# Tuning/deactivation: see printed hints or run neuron_tune.sh / neuron_deactivate.sh.

set -euo pipefail

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
export RATIO="${RATIO:-0.05}"
export STYPE="${STYPE:-lima_s1}"
export NEURONS_ROOT="${NEURONS_ROOT:-./neuron_train_data_detect}"
export NEURONS_PATH="${NEURONS_PATH:-${NEURONS_ROOT}}"
export SAVE_PATH="${SAVE_PATH:-./neuron_deactivation}"
export LANGS="${LANGS:-en,zh,ar,sw}"
export INCLUDE_ALL="${INCLUDE_ALL:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1/2: detection ==="
bash "${SCRIPT_DIR}/neuron_detect.sh"

echo ""
echo "=== Step 2/2: statistics ==="
bash "${SCRIPT_DIR}/neuron_statistics.sh"

echo ""
echo "=== Next steps (examples) ==="
UNION="${SAVE_PATH}/${BASE_MODEL}/ratio${RATIO}/detect_union_neurons.json"
echo "Tune:   NEURON_PATH=\"${UNION}\" bash ${SCRIPT_DIR}/neuron_tune.sh"
echo "Ablate: NEURONS_FILE=\"${UNION}\" OUT_DIR=\"./deactivate_model_param/${BASE_MODEL}/ratio${RATIO}/union\" bash ${SCRIPT_DIR}/neuron_deactivate.sh"
