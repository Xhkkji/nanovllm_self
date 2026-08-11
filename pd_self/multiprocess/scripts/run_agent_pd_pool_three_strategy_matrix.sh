#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

DATASET="${DATASET:-data/serving_benchmarks/agent_trace_qwen3_tokenized.jsonl}"
PROFILE="${PROFILE:-agent_multi_step}"
LIMIT="${LIMIT:-16}"
WARMUP="${WARMUP:-0}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-16}"
PREFILL_GPUS="${PREFILL_GPUS:-0,2}"
DECODE_GPUS="${DECODE_GPUS:-1,3}"
LOAD_MODE="${LOAD_MODE:-closed_loop}"
CONCURRENCY="${CONCURRENCY:-2}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
NCCL_PORT_BASE="${NCCL_PORT_BASE:-29910}"
RESULT_TAG="${RESULT_TAG:-agent_pd_pool_4gpu_three_strategy}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-240}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.05}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-1}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-1}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-1}"
WORKER_FEEDBACK="${WORKER_FEEDBACK:-true}"
WORKER_FEEDBACK_SCALE_S="${WORKER_FEEDBACK_SCALE_S:-1.0}"

cd "${ROOT_DIR}"

RESULT_ROOT="pd_self/multiprocess/result/agent_pd_pool_matrix/${RESULT_TAG}/${PROFILE}"
mkdir -p "${RESULT_ROOT}"

strategies=(round_robin load_aware affinity_load_aware)
index=0
for strategy in "${strategies[@]}"; do
  strategy_dir="${RESULT_ROOT}/${strategy}"
  echo "[agent-pd-pool-matrix] running strategy=${strategy}"
  echo "[agent-pd-pool-matrix] result_dir=${strategy_dir}"

  worker_feedback_flag="--worker-feedback"
  if [[ "${WORKER_FEEDBACK}" == "false" || "${WORKER_FEEDBACK}" == "0" ]]; then
    worker_feedback_flag="--no-worker-feedback"
  fi

  "${PYTHON_BIN}" pd_self/multiprocess/evaluation/benchmark_agent_pd_pool.py \
    --dataset "${DATASET}" \
    --profile "${PROFILE}" \
    --limit "${LIMIT}" \
    --warmup "${WARMUP}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
    --prefill-gpus "${PREFILL_GPUS}" \
    --decode-gpus "${DECODE_GPUS}" \
    --scheduler "${strategy}" \
    --load-mode "${LOAD_MODE}" \
    --concurrency "${CONCURRENCY}" \
    --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
    --nccl-port "$((NCCL_PORT_BASE + index))" \
    --request-timeout-s "${REQUEST_TIMEOUT_S}" \
    --startup-timeout-s "${STARTUP_TIMEOUT_S}" \
    --poll-interval-s "${POLL_INTERVAL_S}" \
    --max-active-decode-requests "${MAX_ACTIVE_DECODE_REQUESTS}" \
    --max-pending-sends "${MAX_PENDING_SENDS}" \
    --max-pending-recvs "${MAX_PENDING_RECVS}" \
    --worker-feedback-scale-s "${WORKER_FEEDBACK_SCALE_S}" \
    "${worker_feedback_flag}" \
    --work-dir "${strategy_dir}/work" \
    --output "${strategy_dir}/synthetic_metrics.jsonl" \
    --summary-output "${strategy_dir}/synthetic_summary.json"

  "${PYTHON_BIN}" pd_self/multiprocess/evaluation/check_pd_resource_cleanup.py \
    --work-dir "${strategy_dir}/work" \
    --output "${strategy_dir}/resource_cleanup.json"

  index=$((index + 1))
done

"${PYTHON_BIN}" pd_self/multiprocess/evaluation/compare_agent_pd_pool_strategies.py \
  --round-robin "${RESULT_ROOT}/round_robin/synthetic_summary.json" \
  --load-aware "${RESULT_ROOT}/load_aware/synthetic_summary.json" \
  --affinity-load-aware "${RESULT_ROOT}/affinity_load_aware/synthetic_summary.json" \
  --output "${RESULT_ROOT}/strategy_compare_summary.json" \
  --markdown-output "${RESULT_ROOT}/strategy_compare_summary.md"

echo "[agent-pd-pool-matrix] comparison=${RESULT_ROOT}/strategy_compare_summary.json"
