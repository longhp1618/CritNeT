#!/usr/bin/env bash
# Safety + English utility evaluation for any saved HF checkpoint (freeze or full FT).
#
# Prereqs: same GPU / SGLang setup as lima_s1_exp/eval/eval_safe.py and lima_s1_exp/eval/eval_utility_en.py.
#
# Usage (from repo root):
#   bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh
#   MODEL_DIR=saved_models/freeze_ft/lima_s1/Qwen/Qwen3-4B-Instruct-2507 DP_SIZE=4 bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh

set -euo pipefail

BASE="${BASE:-Qwen/Qwen3-4B-Instruct-2507}"
STYPE="${STYPE:-lima_s1}"
MODEL_DIR="${MODEL_DIR:-saved_models/freeze_ft/${STYPE}/${BASE}}"
DP_SIZE="${DP_SIZE:-4}"
# English utility tasks (default in eval_utility_en.py matches this list)
TASKS="${TASKS:-math_easy,mmlu,belebele,arc_easy}"

cd "$(dirname "$0")/../../../" || exit 1

# if [[ ! -d "${MODEL_DIR}" ]]; then
#   echo "ERROR: MODEL_DIR not found: ${MODEL_DIR}"
#   echo "Train first, or set MODEL_DIR to your saved HuggingFace checkpoint directory."
#   exit 1
# fi

echo "=========================================="
echo "Safety eval (MultiJail, English)"
echo "Model: ${MODEL_DIR}"
echo "=========================================="
python -m lima_s1_exp.eval.eval_safe \
  --model_name "${MODEL_DIR}" \
  --template \
  --local_model_path "${MODEL_DIR}" \
  --sgl_dp "${DP_SIZE}" \
  --metrics_json "lima_s1_exp/eval/predictions/safety/${MODEL_DIR}/metrics.json"

echo ""
echo "=========================================="
echo "Utility eval (English): tasks=${TASKS}"
echo "Model: ${MODEL_DIR}"
echo "=========================================="
python -m lima_s1_exp.eval.eval_utility_en \
  --model_name "${MODEL_DIR}" \
  --dp_size "${DP_SIZE}" \
  --tasks "${TASKS}"

echo ""
echo "Safety outputs: lima_s1_exp/eval/predictions/safety/${MODEL_DIR}/"
echo "Utility outputs: lima_s1_exp/eval/predictions/ ; summary at .../summary.json"
