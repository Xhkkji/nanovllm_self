#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
DATASET="${DATASET:-data/serving_benchmarks/synthetic_serving_qwen3_tokenized.jsonl}"
PROFILE="${PROFILE:-short_in_short_out}"
RUNS="${RUNS:-3}"
LIMIT="${LIMIT:-2}"
WARMUP="${WARMUP:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-8}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
NCCL_PORT_BASE="${NCCL_PORT_BASE:-29710}"
LOAD_MODE="${LOAD_MODE:-closed_loop}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-4}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-4}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-180}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.05}"
RESULT_TAG="${RESULT_TAG:-sync_gpu_single_pair_stability}"

cd "${ROOT_DIR}"

for run_idx in $(seq 1 "${RUNS}"); do
  tag="${RESULT_TAG}_r${run_idx}"
  port="$((NCCL_PORT_BASE + run_idx))"
  echo "[sync-gpu-stability] run=${run_idx}/${RUNS} tag=${tag} port=${port}"
  PREFILL_GPU="${PREFILL_GPU}" \
  DECODE_GPU="${DECODE_GPU}" \
  KV_TRANSFER_BACKENDS="sync_gpu" \
  KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE}" \
  NCCL_PORT="${port}" \
  DATASET="${DATASET}" \
  PROFILES="${PROFILE}" \
  LIMIT="${LIMIT}" \
  WARMUP="${WARMUP}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
  MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP}" \
  DECODE_MODE="continuous" \
  LOAD_MODE="${LOAD_MODE}" \
  CONCURRENCY="${CONCURRENCY}" \
  MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS}" \
  MAX_PENDING_SENDS="${MAX_PENDING_SENDS}" \
  MAX_PENDING_RECVS="${MAX_PENDING_RECVS}" \
  REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S}" \
  STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S}" \
  POLL_INTERVAL_S="${POLL_INTERVAL_S}" \
  RESULT_TAG="${tag}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash pd_self/multiprocess/scripts/run_pipeline_profile_matrix.sh

  mode="pipeline_pd_sync_gpu"
  if [[ "${LOAD_MODE}" != "batch" ]]; then
    mode="${mode}_${LOAD_MODE}_c${CONCURRENCY}"
  fi
  work_dir="pd_self/multiprocess/result/${mode}/${tag}/${PROFILE}/work"
  "${PYTHON_BIN}" pd_self/multiprocess/evaluation/check_pd_resource_cleanup.py \
    --work-dir "${work_dir}" \
    --output "pd_self/multiprocess/result/stability/${RESULT_TAG}/${PROFILE}/run_${run_idx}_resource_cleanup.json"
done

echo "[sync-gpu-stability] done"
