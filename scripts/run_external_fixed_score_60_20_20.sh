#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SEEDS=(2806 2807 2808 2809 2810 2811 2812 2813 2814 2815)

run_dataset() {
  local dataset_name="$1"
  local text_features="$2"
  local combined_features="$3"
  local qwen_scores="$4"
  local output_dir="$5"

  python -m crop.experiments.exp15_prefix_aware \
    --dataset_name "$dataset_name" \
    --step_text_features "$text_features" \
    --step_combined_features "$combined_features" \
    --qwen_scores_csv "$qwen_scores" \
    --output_dir "$output_dir" \
    --seeds "${SEEDS[@]}" \
    --alphas 0.05 \
    --lambda_grid_size 101 \
    --table_fixed_only
}

run_dataset \
  ProcessBench \
  outputs/strengthened/final/external_process/processbench/processbench_text_steps.npz \
  outputs/strengthened/final/external_process/processbench/processbench_combined_steps.npz \
  outputs/strengthened/final/external_process/processbench_qwen_prm/qwen_prm_scores.csv \
  outputs/fixed_score_60_20_20/processbench &

run_dataset \
  Math-Shepherd \
  outputs/strengthened/final/external_process/math_shepherd/math_shepherd_text_steps.npz \
  outputs/strengthened/final/external_process/math_shepherd/math_shepherd_combined_steps.npz \
  outputs/strengthened/final/external_process/math_shepherd_qwen_prm/qwen_prm_scores.csv \
  outputs/fixed_score_60_20_20/math_shepherd &

run_dataset \
  PRMBench \
  outputs/strengthened/final/external_process/prmbench/prmbench_text_steps.npz \
  outputs/strengthened/final/external_process/prmbench/prmbench_combined_steps.npz \
  outputs/strengthened/final/external_process/prmbench_full_qwen_prm/qwen_prm_scores.csv \
  outputs/fixed_score_60_20_20/prmbench &

run_dataset \
  PRM800K \
  outputs/strengthened/final/external_process/prm800k/prm800k_text_steps.npz \
  outputs/strengthened/final/external_process/prm800k/prm800k_combined_steps.npz \
  outputs/strengthened/final/external_process/prm800k_qwen_prm/qwen_prm_scores.csv \
  outputs/fixed_score_60_20_20/prm800k &

wait
python scripts/build_fixed_score_table.py
