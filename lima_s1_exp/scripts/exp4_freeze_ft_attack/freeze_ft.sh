#!/usr/bin/env bash
# Freeze safety neurons, full fine-tune the rest. After training, English safety + utility eval:
#   bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh

set -euo pipefail
cd "$(dirname "$0")/../../../" || exit 1

accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml \
  lima_s1_exp/exp4_freeze_ft_attack/freeze_ft.py \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --freeze_neurons_path neuron_pairwise_stats/Qwen/Qwen3-4B-Instruct-2507/lima_s10.13_safety0.06/exclusive_safety_neurons.json \
  --stype lima_s1 \
  --langs en

MODEL_DIR=saved_models/freeze_ft/lima_s1/Qwen/Qwen3-4B-Instruct-2507 DP_SIZE=4 bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh