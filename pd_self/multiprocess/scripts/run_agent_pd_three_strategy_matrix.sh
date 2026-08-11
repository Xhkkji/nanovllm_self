#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

DATASET="${DATASET:-data/serving_benchmarks/agent_trace_qwen3_tokenized.jsonl}"
PROFILE="${PROFILE:-agent_multi_step}"
LIMIT="${LIMIT:-40}"
WARMUP="${WARMUP:-4}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-64}"
PREFILL_GPUS="${PREFILL_GPUS:-0,2}"
DECODE_GPUS="${DECODE_GPUS:-1,3}"
INITIAL_BACKLOG_S="${INITIAL_BACKLOG_S:-0,0}"
LOAD_MODE="${LOAD_MODE:-closed_loop}"
CONCURRENCY="${CONCURRENCY:-4}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-shared_memory}"
NCCL_PORT_BASE="${NCCL_PORT_BASE:-29670}"
RESULT_TAG="${RESULT_TAG:-agent_trace_three_strategy}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-300}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-180}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-0.05}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-4}"
MAX_PENDING_RECVS="${MAX_PENDING_RECVS:-4}"
WORKER_FEEDBACK_SCALE_S="${WORKER_FEEDBACK_SCALE_S:-1.0}"

cd "${ROOT_DIR}"

for strategy in round_robin load_aware affinity_load_aware; do
  echo "[agent-pd-matrix] running strategy=${strategy}"
  DATASET="${DATASET}" \
  PROFILE="${PROFILE}" \
  LIMIT="${LIMIT}" \
  WARMUP="${WARMUP}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS}" \
  MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP}" \
  PREFILL_GPUS="${PREFILL_GPUS}" \
  DECODE_GPUS="${DECODE_GPUS}" \
  SCHEDULER="${strategy}" \
  INITIAL_BACKLOG_S="${INITIAL_BACKLOG_S}" \
  LOAD_MODE="${LOAD_MODE}" \
  CONCURRENCY="${CONCURRENCY}" \
  KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE}" \
  KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND}" \
  NCCL_PORT_BASE="$((NCCL_PORT_BASE + 10))" \
  RESULT_TAG="${RESULT_TAG}" \
  REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S}" \
  STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S}" \
  POLL_INTERVAL_S="${POLL_INTERVAL_S}" \
  MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS}" \
  MAX_PENDING_SENDS="${MAX_PENDING_SENDS}" \
  MAX_PENDING_RECVS="${MAX_PENDING_RECVS}" \
  WORKER_FEEDBACK_SCALE_S="${WORKER_FEEDBACK_SCALE_S}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash pd_self/multiprocess/scripts/run_agent_pd_multi_pair_benchmark.sh
done

COMPARE_DIR="pd_self/multiprocess/result/compare/agent_pd_three_strategy/${RESULT_TAG}/${PROFILE}"
python_args=(
  pd_self/multiprocess/evaluation/compare_agent_pd_strategies.py
  --round-robin
  "pd_self/multiprocess/result/agent_pd_multi_pair_${KV_TRANSFER_BACKEND}_round_robin/${RESULT_TAG}/${PROFILE}/synthetic_summary.json"
  --load-aware
  "pd_self/multiprocess/result/agent_pd_multi_pair_${KV_TRANSFER_BACKEND}_load_aware/${RESULT_TAG}/${PROFILE}/synthetic_summary.json"
  --affinity-load-aware
  "pd_self/multiprocess/result/agent_pd_multi_pair_${KV_TRANSFER_BACKEND}_affinity_load_aware/${RESULT_TAG}/${PROFILE}/synthetic_summary.json"
  --output
  "${COMPARE_DIR}/strategy_compare_summary.json"
)
"${PYTHON_BIN}" "${python_args[@]}"
