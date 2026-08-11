#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-shared_memory}"
NCCL_PORT="${NCCL_PORT:-29577}"
DATASET="${DATASET:-data/serving_benchmarks/synthetic_serving_qwen3_tokenized.jsonl}"
LIMIT="${LIMIT:-5}"
WARMUP="${WARMUP:-0}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-16}"
DECODE_MODE="${DECODE_MODE:-continuous}"
LOAD_MODE="${LOAD_MODE:-batch}"
CONCURRENCY="${CONCURRENCY:-4}"
SCHEDULER="${SCHEDULER:-none}"
NUM_WORKER_SLOTS="${NUM_WORKER_SLOTS:-1}"
INITIAL_BACKLOG_S="${INITIAL_BACKLOG_S:-}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-4}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-4}"
PROFILE_ARG=()

if [[ -n "${PROFILE:-}" ]]; then
  PROFILE_ARG=(--profile "${PROFILE}")
fi

cd "${ROOT_DIR}"

echo "[pipeline-pd] dataset=${DATASET} profile=${PROFILE:-all} limit=${LIMIT} warmup=${WARMUP} max_output_tokens_cap=${MAX_OUTPUT_TOKENS_CAP} decode_mode=${DECODE_MODE} load_mode=${LOAD_MODE} concurrency=${CONCURRENCY} scheduler=${SCHEDULER} num_worker_slots=${NUM_WORKER_SLOTS} initial_backlog_s=${INITIAL_BACKLOG_S:-none} max_active_decode_requests=${MAX_ACTIVE_DECODE_REQUESTS} max_pending_sends=${MAX_PENDING_SENDS} max_pending_recvs=${MAX_PENDING_RECVS} kv_transfer_backend=${KV_TRANSFER_BACKEND} nccl_port=${NCCL_PORT}"

"${PYTHON_BIN}" pd_self/multiprocess/evaluation/benchmark_synthetic_pd_pipeline.py \
  --dataset "${DATASET}" \
  --limit "${LIMIT}" \
  --warmup "${WARMUP}" \
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
  --prefill-gpu "${PREFILL_GPU}" \
  --decode-gpu "${DECODE_GPU}" \
  --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
  --kv-transfer-backend "${KV_TRANSFER_BACKEND}" \
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
  --python-bin "${PYTHON_BIN}" \
  "${PROFILE_ARG[@]}"
