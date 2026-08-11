#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
DATASET="${DATASET:-data/serving_benchmarks/synthetic_serving_qwen3_tokenized.jsonl}"
PROFILES="${PROFILES:-short_in_short_out mixed_chat}"
LIMIT="${LIMIT:-3}"
WARMUP="${WARMUP:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-8}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
NCCL_PORT="${NCCL_PORT:-29690}"
DECODE_MODE="${DECODE_MODE:-continuous}"
LOAD_MODE="${LOAD_MODE:-closed_loop}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-4}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-4}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-180}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.05}"
RESULT_TAG="${RESULT_TAG:-pd_correctness_matrix}"
RUN_BENCHMARKS="${RUN_BENCHMARKS:-true}"
FAIL_ON_SHM_CANDIDATES="${FAIL_ON_SHM_CANDIDATES:-false}"

cd "${ROOT_DIR}"

mode_for_backend() {
  local backend="$1"
  local mode="pipeline_pd"
  if [[ "${backend}" != "shared_memory" ]]; then
    mode="pipeline_pd_${backend}"
  fi
  if [[ "${LOAD_MODE}" != "batch" ]]; then
    mode="${mode}_${LOAD_MODE}_c${CONCURRENCY}"
  fi
  echo "${mode}"
}

if [[ "${RUN_BENCHMARKS}" == "true" || "${RUN_BENCHMARKS}" == "1" ]]; then
  for backend in shared_memory sync_gpu; do
    echo "[pd-correctness-matrix] running backend=${backend}"
    PREFILL_GPU="${PREFILL_GPU}" \
    DECODE_GPU="${DECODE_GPU}" \
    KV_TRANSFER_BACKENDS="${backend}" \
    KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE}" \
    NCCL_PORT="${NCCL_PORT}" \
    DATASET="${DATASET}" \
    PROFILES="${PROFILES}" \
    LIMIT="${LIMIT}" \
    WARMUP="${WARMUP}" \
    MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
    MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP}" \
    DECODE_MODE="${DECODE_MODE}" \
    LOAD_MODE="${LOAD_MODE}" \
    CONCURRENCY="${CONCURRENCY}" \
    MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS}" \
    MAX_PENDING_SENDS="${MAX_PENDING_SENDS}" \
    MAX_PENDING_RECVS="${MAX_PENDING_RECVS}" \
    REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S}" \
    STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S}" \
    POLL_INTERVAL_S="${POLL_INTERVAL_S}" \
    RESULT_TAG="${RESULT_TAG}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash pd_self/multiprocess/scripts/run_pipeline_profile_matrix.sh
  done
fi

baseline_mode="$(mode_for_backend shared_memory)"
candidate_mode="$(mode_for_backend sync_gpu)"
correctness_dir="pd_self/multiprocess/result/correctness/${RESULT_TAG}"
mkdir -p "${correctness_dir}"

for profile in ${PROFILES}; do
  left="pd_self/multiprocess/result/${baseline_mode}/${RESULT_TAG}/${profile}/synthetic_metrics.jsonl"
  right="pd_self/multiprocess/result/${candidate_mode}/${RESULT_TAG}/${profile}/synthetic_metrics.jsonl"
  output="${correctness_dir}/${profile}.json"

  echo "[pd-correctness-matrix] checking profile=${profile}"
  "${PYTHON_BIN}" pd_self/multiprocess/evaluation/check_pipeline_correctness.py \
    --left "${left}" \
    --right "${right}" \
    --output "${output}"

  work_dir="pd_self/multiprocess/result/${candidate_mode}/${RESULT_TAG}/${profile}/work"
  resource_output="${correctness_dir}/${profile}_resource_cleanup.json"
  resource_args=(
    pd_self/multiprocess/evaluation/check_pd_resource_cleanup.py
    --work-dir "${work_dir}"
    --output "${resource_output}"
  )
  if [[ "${FAIL_ON_SHM_CANDIDATES}" == "true" || "${FAIL_ON_SHM_CANDIDATES}" == "1" ]]; then
    resource_args+=(--fail-on-shm-candidates)
  fi
  "${PYTHON_BIN}" "${resource_args[@]}"
done

echo "[pd-correctness-matrix] done output_dir=${correctness_dir}"
