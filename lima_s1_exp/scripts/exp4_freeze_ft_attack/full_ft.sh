#!/usr/bin/env bash
# Full fine-tuning (no neuron freeze), same data/settings as freeze_ft.sh baseline.
# After training: English safety (MultiJail) + utility (math_easy, mmlu, belebele, arc_easy).
#
# Usage: bash lima_s1_exp/scripts/exp4_freeze_ft_attack/full_ft.sh
# Override eval GPUs: DP_SIZE=8 MODEL_DIR=... bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh

set -euo pipefail
cd "$(dirname "$0")/../../../" || exit 1

accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml \
  lima_s1_exp/exp4_freeze_ft_attack/full_ft.py \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --training_config lima_s1_exp/configs/exp_config/sft_config.yaml \
  --stype lima_s1 \
  --langs en

MODEL_DIR=saved_models/full/lima_s1/Qwen/Qwen3-4B-Instruct-2507 DP_SIZE=4 bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh
