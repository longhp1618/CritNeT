#!/usr/bin/env bash
# Critical-neuron fine-tuning with DeepSpeed ZeRO-2 via accelerate,
# followed by optional evaluation via lima_s1_exp/eval/eval_utility.py.
#
# Usage: bash lima_s1_exp/scripts/exp5_multilingual_critical_foundational/neuron_tune.sh
#   or:  EVAL=0 bash lima_s1_exp/scripts/exp5_multilingual_critical_foundational/neuron_tune.sh
#   or:  SAVE_MERGED=1 bash lima_s1_exp/scripts/exp5_multilingual_critical_foundational/neuron_tune.sh

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
STYPE="${STYPE:-lima_s1}"
RATIO="${RATIO:-0.05}"
STATS_SAVE="${STATS_SAVE:-./neuron_deactivation}"
NEURON_PATH="${NEURON_PATH:-${STATS_SAVE}/${BASE_MODEL}/${STYPE}/ratio${RATIO}/union_neurons.json}"
TRAINING_CONFIG="${TRAINING_CONFIG:-lima_s1_exp/configs/exp_config/sft_config.yaml}"
ACC_CONFIG="${ACC_CONFIG:-lima_s1_exp/configs/acc_config/ds_zero2_4.yaml}"
LANGS="${LANGS:-en,zh,ar,sw}"
INCLUDE_ALL="${INCLUDE_ALL:-1}"
SAVE_MERGED="${SAVE_MERGED:-0}"
EVAL="${EVAL:-1}"
DP_SIZE="${DP_SIZE:-4}"

cd "$(dirname "$0")/../../../" || exit 1

if [[ ! -e "${NEURON_PATH}" ]]; then
  echo "ERROR: NEURON_PATH not found: ${NEURON_PATH}"
  echo "Set NEURON_PATH to detect_union_neurons.json, a lang folder, or detection output dir."
  exit 1
fi

# Eval requires the full merged model on disk.
if [[ "${EVAL}" == "1" ]]; then
  SAVE_MERGED=1
fi

extra=()
[[ "${INCLUDE_ALL}" == "1" ]] && extra+=(--include_all_eligible_modules)
[[ "${SAVE_MERGED}" == "1" ]] && extra+=(--save_merged_model)

accelerate launch --config_file "${ACC_CONFIG}" \
  lima_s1_exp/exp5_multilingual_critical_foundational/neuron_tuning.py \
  --base "${BASE_MODEL}" \
  --stype "${STYPE}" \
  --neuron_path "${NEURON_PATH}" \
  --training_config "${TRAINING_CONFIG}" \
  --langs "${LANGS}" \
  --sparsity_ratio "${RATIO}" \
  "${extra[@]}"

ADAPTER_DIR="saved_adapters/neuron_tuning/${STYPE}/${BASE_MODEL}"
MERGED_DIR="saved_models/neuron_tuning/${STYPE}/${BASE_MODEL}"
echo "Adapter saved under: ./${ADAPTER_DIR}/"

# ---------- Evaluation ----------
if [[ "${EVAL}" == "1" ]]; then
  echo "=========================================="
  echo "Evaluating merged model: ${MERGED_DIR}"
  echo "=========================================="
  python -m lima_s1_exp.eval.eval_utility \
    --model_name "${MERGED_DIR}" \
    --dp_size "${DP_SIZE}"
fi
