#!/usr/bin/env bash
# Run critical-neuron detection (lima_s1_exp/exp5_multilingual_critical_foundational/neuron_detection.py).
# Usage: STYPE=safety ALL_LANGS=0 NEURON_LANG=en BASE_MODEL=... SAVE_IMPORTANCE_CACHE_DIR=./importance_cache bash lima_s1_exp/scripts/exp5_multilingual_critical_foundational/neuron_detect.sh
# (Use NEURON_LANG for the dataset code; LANG is often en_US.UTF-8 and is the wrong path segment.)

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"

INCLUDE_ALL="${INCLUDE_ALL:-1}"      # 1 = --include_all_eligible_modules
STYPE="${STYPE:-lima_s1}"
RATIO="${RATIO:-0.05}"
NEURONS_ROOT="${NEURONS_ROOT:-./neuron_detect}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
ALL_LANGS="${ALL_LANGS:-1}"          # 1 = --all_languages
# Optional: directory for per-lang *.pt caches (|W*grad| after gate merge). Mutually exclusive with IMPORTANCE_CACHE_DIR.
SAVE_IMPORTANCE_CACHE_DIR="${SAVE_IMPORTANCE_CACHE_DIR:-}"
# Optional: load caches from here and only re-run global top-k (no GPU model). Mutually exclusive with SAVE_IMPORTANCE_CACHE_DIR.
IMPORTANCE_CACHE_DIR="${IMPORTANCE_CACHE_DIR:-}"
NEURON_LANG="${NEURON_LANG:-en}"

cd "$(dirname "$0")/../../../" || exit 1

extra=()
[[ "${ALL_LANGS}" == "1" ]] && extra+=(--all_languages) || extra+=(--language "${NEURON_LANG}")
[[ "${INCLUDE_ALL}" == "1" ]] && extra+=(--include_all_eligible_modules)
[[ -n "${SAVE_IMPORTANCE_CACHE_DIR}" ]] && extra+=(--save_importance_cache_dir "${SAVE_IMPORTANCE_CACHE_DIR}")
[[ -n "${IMPORTANCE_CACHE_DIR}" ]] && extra+=(--importance_cache_dir "${IMPORTANCE_CACHE_DIR}")

python lima_s1_exp/exp5_multilingual_critical_foundational/neuron_detection.py \
  --base "${BASE_MODEL}" \
  --stype "${STYPE}" \
  --output_dir "${NEURONS_ROOT}" \
  --top_k_percent "${RATIO}" \
  --max_samples "${MAX_SAMPLES}" \
  "${extra[@]}"

echo "Done. Outputs under: ${NEURONS_ROOT}/${BASE_MODEL}/ratio${RATIO}/"
