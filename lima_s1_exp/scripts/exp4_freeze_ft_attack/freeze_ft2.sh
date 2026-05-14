#!/usr/bin/env bash
# Freeze safety neurons, full fine-tune the rest. After training, English safety + utility eval:
#   bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh

set -euo pipefail
cd "$(dirname "$0")/../../../" || exit 1

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file lima_s1_exp/configs/acc_config/ds_zero2_4.yaml --main_process_port 29501 \
  lima_s1_exp/exp4_freeze_ft_attack/freeze_ft.py \
  --base meta-llama/Llama-3.1-8B-Instruct \
  --freeze_neurons_path neuron_pairwise_stats/meta-llama/Llama-3.1-8B-Instruct/lima_s10.14_safety0.04/exclusive_safety_neurons.json \
  --stype lima_s1 \
  --langs en

CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_DIR=saved_models/freeze_ft/lima_s1/meta-llama/Llama-3.1-8B-Instruct DP_SIZE=4 bash lima_s1_exp/scripts/exp4_freeze_ft_attack/eval_freeze_ft_en.sh