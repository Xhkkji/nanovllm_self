#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

DATASET="${DATASET:-data/serving_benchmarks/sharegpt_qwen3_tokenized_5k.jsonl}"
PROFILE="${PROFILE:-}"
LIMIT="${LIMIT:-10}"
WARMUP="${WARMUP:-1}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-16}"
PREFILL_GPUS="${PREFILL_GPUS:-0,2}"
DECODE_GPUS="${DECODE_GPUS:-1,3}"
SCHEDULER="${SCHEDULER:-load_aware}"
INITIAL_BACKLOG_S="${INITIAL_BACKLOG_S:-}"
LOAD_MODE="${LOAD_MODE:-closed_loop}"
CONCURRENCY="${CONCURRENCY:-4}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-shared_memory}"
NCCL_PORT_BASE="${NCCL_PORT_BASE:-29670}"
RESULT_TAG="${RESULT_TAG:-default}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-120}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.05}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-4}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-4}"
WORKER_FEEDBACK="${WORKER_FEEDBACK:-true}"
WORKER_FEEDBACK_SCALE_S="${WORKER_FEEDBACK_SCALE_S:-1.0}"

PROFILE_ARG=()
if [[ -n "${PROFILE}" ]]; then
  PROFILE_ARG=(--profile "${PROFILE}")
fi

WORKER_FEEDBACK_ARG=(--worker-feedback)
if [[ "${WORKER_FEEDBACK}" == "false" || "${WORKER_FEEDBACK}" == "0" ]]; then
  WORKER_FEEDBACK_ARG=(--no-worker-feedback)
fi

mode="agent_pd_multi_pair_${KV_TRANSFER_BACKEND}_${SCHEDULER}"
result_dir="pd_self/multiprocess/result/${mode}/${RESULT_TAG}/${PROFILE:-all}"

cd "${ROOT_DIR}"

echo "[agent-pd-multi-pair] dataset=${DATASET}"
echo "[agent-pd-multi-pair] prefill_gpus=${PREFILL_GPUS} decode_gpus=${DECODE_GPUS} backend=${KV_TRANSFER_BACKEND} scheduler=${SCHEDULER}"
echo "[agent-pd-multi-pair] profile=${PROFILE:-all} limit=${LIMIT} warmup=${WARMUP} cap=${MAX_OUTPUT_TOKENS_CAP} load_mode=${LOAD_MODE} concurrency=${CONCURRENCY}"
echo "[agent-pd-multi-pair] result_dir=${result_dir}"

"${PYTHON_BIN}" pd_self/multiprocess/evaluation/benchmark_agent_pd_multi_pair.py \
  --dataset "${DATASET}" \
  --limit "${LIMIT}" \
  --warmup "${WARMUP}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
  --prefill-gpus "${PREFILL_GPUS}" \
  --decode-gpus "${DECODE_GPUS}" \
  --scheduler "${SCHEDULER}" \
  --initial-backlog-s "${INITIAL_BACKLOG_S}" \
  --load-mode "${LOAD_MODE}" \
  --concurrency "${CONCURRENCY}" \
  --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
  --kv-transfer-backend "${KV_TRANSFER_BACKEND}" \
  --nccl-port-base "${NCCL_PORT_BASE}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  --startup-timeout-s "${STARTUP_TIMEOUT_S}" \
  --poll-interval-s "${POLL_INTERVAL_S}" \
  --max-active-decode-requests "${MAX_ACTIVE_DECODE_REQUESTS}" \
  --max-pending-sends "${MAX_PENDING_SENDS}" \
  --max-pending-recvs "${MAX_PENDING_RECVS}" \
  "${WORKER_FEEDBACK_ARG[@]}" \
  --worker-feedback-scale-s "${WORKER_FEEDBACK_SCALE_S}" \
  --python-bin "${PYTHON_BIN}" \
  --work-dir "${result_dir}/work" \
  --output "${result_dir}/synthetic_metrics.jsonl" \
  --summary-output "${result_dir}/synthetic_summary.json" \
  "${PROFILE_ARG[@]}"
