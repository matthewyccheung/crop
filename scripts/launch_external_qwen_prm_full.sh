#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCORER="crop/scripts/score_qwen_prm.py"
BASE="outputs/strengthened/final/external_process"
RUN_ROOT="${BASE}/qwen_full_runs"
LOG_DIR="${RUN_ROOT}/logs"
PID_FILE="${RUN_ROOT}/pids.tsv"
MANIFEST="${RUN_ROOT}/manifest.tsv"

mkdir -p "${LOG_DIR}"
: > "${PID_FILE}"
: > "${MANIFEST}"
printf "dataset\tpart\tgpu\tstart\tmax_traces\toutput_csv\tlog\n" >> "${MANIFEST}"

launch_shard() {
  local dataset="$1"
  local part="$2"
  local gpu="$3"
  local input_jsonl="$4"
  local output_dir="$5"
  local start="$6"
  local max_traces="$7"
  local output_csv="${output_dir}/shards/part_${part}.csv"
  local log="${LOG_DIR}/${dataset}_part_${part}.log"

  mkdir -p "${output_dir}/shards"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${dataset}" "${part}" "${gpu}" "${start}" "${max_traces}" "${output_csv}" "${log}" >> "${MANIFEST}"

  setsid env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    /usr/bin/time -f "elapsed_sec %e" \
    "${PYTHON_BIN}" "${SCORER}" \
      --input_jsonl "${input_jsonl}" \
      --output_csv "${output_csv}" \
      --start "${start}" \
      --max_traces "${max_traces}" \
      --flush_every 25 \
      --device_map auto \
    > "${log}" 2>&1 < /dev/null &

  printf "%s\t%s\t%s\t%s\n" "${dataset}" "${part}" "${gpu}" "$!" >> "${PID_FILE}"
}

launch_shard "math_shepherd" 0 0 "${BASE}/math_shepherd/math_shepherd_normalized.jsonl" "${BASE}/math_shepherd_qwen_prm" 0 5000
launch_shard "math_shepherd" 1 1 "${BASE}/math_shepherd/math_shepherd_normalized.jsonl" "${BASE}/math_shepherd_qwen_prm" 5000 5000

launch_shard "prmbench" 0 2 "${BASE}/prmbench/prmbench_normalized.jsonl" "${BASE}/prmbench_full_qwen_prm" 0 1667
launch_shard "prmbench" 1 3 "${BASE}/prmbench/prmbench_normalized.jsonl" "${BASE}/prmbench_full_qwen_prm" 1667 1667
launch_shard "prmbench" 2 4 "${BASE}/prmbench/prmbench_normalized.jsonl" "${BASE}/prmbench_full_qwen_prm" 3334 1666

launch_shard "prm800k" 0 5 "${BASE}/prm800k/prm800k_normalized.jsonl" "${BASE}/prm800k_qwen_prm" 0 2667
launch_shard "prm800k" 1 6 "${BASE}/prm800k/prm800k_normalized.jsonl" "${BASE}/prm800k_qwen_prm" 2667 2667
launch_shard "prm800k" 2 7 "${BASE}/prm800k/prm800k_normalized.jsonl" "${BASE}/prm800k_qwen_prm" 5334 2666

echo "Launched Qwen PRM shards. PIDs: ${PID_FILE}"
echo "Manifest: ${MANIFEST}"
