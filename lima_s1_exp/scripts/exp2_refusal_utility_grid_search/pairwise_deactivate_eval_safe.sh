#!/usr/bin/env bash
# For each lima_s1*safety* pairwise folder: deactivate union neurons → eval_safe (ASR) → delete checkpoint.
# Uses one dedicated GPU per worker (default 4 threads on GPUs 4–7).
#
# Example:
#   bash lima_s1_exp/scripts/exp2_refusal_utility_grid_search/pairwise_deactivate_eval_safe.sh
#   BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct GPUS=4,5,6,7 MAX_WORKERS=4 bash lima_s1_exp/scripts/exp2_refusal_utility_grid_search/pairwise_deactivate_eval_safe.sh

set -euo pipefail

cd "$(dirname "$0")/../../../" || exit 1

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PAIR_ROOT="${PAIR_ROOT:-neuron_pairwise_stats/${BASE_MODEL}}"
NEURONS_JSON="${NEURONS_JSON:-exclusive_safety_neurons.json}"
GPUS="${GPUS:-0,1,2,3}"
MAX_WORKERS="${MAX_WORKERS:-4}"
DEACTIVATE_RATIO="${DEACTIVATE_RATIO:-0.05}"
BEST_JSON="${BEST_JSON:-lima_s1_exp/eval/predictions/safety/best.json}"

python -m lima_s1_exp.eval.pairwise_deactivate_safety \
  --repo_root . \
  --base_model "${BASE_MODEL}" \
  --pair_stats_root "${PAIR_ROOT}" \
  --neurons_json "${NEURONS_JSON}" \
  --gpus "${GPUS}" \
  --max_workers "${MAX_WORKERS}" \
  --deactivate_ratio "${DEACTIVATE_RATIO}" \
  --best_json "${BEST_JSON}"

echo "Done. See: ${BEST_JSON}"
