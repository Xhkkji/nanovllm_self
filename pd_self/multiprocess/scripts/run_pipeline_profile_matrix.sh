#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
KV_TRANSFER_BACKENDS="${KV_TRANSFER_BACKENDS:-shared_memory}"
NCCL_PORT="${NCCL_PORT:-29577}"
DATASET="${DATASET:-data/serving_benchmarks/synthetic_serving_qwen3_tokenized.jsonl}"
PROFILES="${PROFILES:-short_in_short_out short_in_long_out long_in_short_out long_in_long_out mixed_chat}"
LIMIT="${LIMIT:-10}"
WARMUP="${WARMUP:-2}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-16}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
DECODE_MODE="${DECODE_MODE:-continuous}"
LOAD_MODE="${LOAD_MODE:-batch}"
CONCURRENCY="${CONCURRENCY:-4}"
SCHEDULER="${SCHEDULER:-none}"
NUM_WORKER_SLOTS="${NUM_WORKER_SLOTS:-1}"
INITIAL_BACKLOG_S="${INITIAL_BACKLOG_S:-}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-4}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-4}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-120}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.05}"
RESULT_TAG="${RESULT_TAG:-}"

cd "${ROOT_DIR}"

echo "[pipeline-matrix] profiles=${PROFILES}"
echo "[pipeline-matrix] kv_transfer_backends=${KV_TRANSFER_BACKENDS}"
echo "[pipeline-matrix] dataset=${DATASET}"
echo "[pipeline-matrix] result_tag=${RESULT_TAG:-none}"
echo "[pipeline-matrix] limit=${LIMIT} warmup=${WARMUP} max_output_tokens_cap=${MAX_OUTPUT_TOKENS_CAP} nccl_port=${NCCL_PORT} load_mode=${LOAD_MODE} concurrency=${CONCURRENCY} scheduler=${SCHEDULER} num_worker_slots=${NUM_WORKER_SLOTS} initial_backlog_s=${INITIAL_BACKLOG_S:-none}"
echo "[pipeline-matrix] max_active_decode_requests=${MAX_ACTIVE_DECODE_REQUESTS} max_pending_sends=${MAX_PENDING_SENDS} max_pending_recvs=${MAX_PENDING_RECVS}"

for backend in ${KV_TRANSFER_BACKENDS}; do
  for profile in ${PROFILES}; do
    mode_dir="pipeline_pd"
    if [[ "${backend}" != "shared_memory" ]]; then
      mode_dir="pipeline_pd_${backend}"
    fi
    if [[ "${LOAD_MODE}" != "batch" ]]; then
      mode_dir="${mode_dir}_${LOAD_MODE}_c${CONCURRENCY}"
    fi

    if [[ -n "${RESULT_TAG}" ]]; then
      result_dir="pd_self/multiprocess/result/${mode_dir}/${RESULT_TAG}/${profile}"
    else
      result_dir="pd_self/multiprocess/result/${mode_dir}/${profile}"
    fi
    work_dir="${result_dir}/work"
    metrics_path="${result_dir}/synthetic_metrics.jsonl"
    summary_path="${result_dir}/synthetic_summary.json"

    echo "[pipeline-matrix] backend=${backend} profile=${profile} result_dir=${result_dir}"

    "${PYTHON_BIN}" pd_self/multiprocess/evaluation/benchmark_synthetic_pd_pipeline.py \
      --dataset "${DATASET}" \
      --profile "${profile}" \
      --limit "${LIMIT}" \
      --warmup "${WARMUP}" \
      --max-total-tokens "${MAX_TOTAL_TOKENS}" \
      --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
      --prefill-gpu "${PREFILL_GPU}" \
      --decode-gpu "${DECODE_GPU}" \
      --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
      --kv-transfer-backend "${backend}" \
      --nccl-port "${NCCL_PORT}" \
      --decode-mode "${DECODE_MODE}" \
      --load-mode "${LOAD_MODE}" \
      --concurrency "${CONCURRENCY}" \
      --scheduler "${SCHEDULER}" \
      --num-worker-slots "${NUM_WORKER_SLOTS}" \
      --initial-backlog-s "${INITIAL_BACKLOG_S}" \
      --max-active-decode-requests "${MAX_ACTIVE_DECODE_REQUESTS}" \
      --max-pending-sends "${MAX_PENDING_SENDS}" \
      --max-pending-recvs "${MAX_PENDING_RECVS}" \
      --request-timeout-s "${REQUEST_TIMEOUT_S}" \
      --startup-timeout-s "${STARTUP_TIMEOUT_S}" \
      --poll-interval-s "${POLL_INTERVAL_S}" \
      --python-bin "${PYTHON_BIN}" \
      --work-dir "${work_dir}" \
      --output "${metrics_path}" \
      --summary-output "${summary_path}"
  done
done

echo "[pipeline-matrix] done"
