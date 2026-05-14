#!/usr/bin/env bash
# Full fine-tuning (no neuron freeze), same data/settings as freeze_ft2.sh baseline.
# After training: English safety + utility eval.
#
# Usage: bash lima_s1_exp/scripts/exp4_freeze_ft_attack/full_ft2.sh

set -euo pipefail
cd "$(dirname "$0")/../../../" || exit 1

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml --main_process_port 29501 \
  lima_s1_exp/exp4_freeze_ft_attack/full_ft.py \
  --base meta-llama/Llama-3.1-8B-Instruct \
  --training_config lima_s1_exp/configs/exp_config/sft_config.yaml \
  --stype lima_s1 \
  --langs en

CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_DIR=saved_models/full/lima_s1/meta-llama/Llama-3.1-8B-Instruct DP_SIZE=4 bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh
