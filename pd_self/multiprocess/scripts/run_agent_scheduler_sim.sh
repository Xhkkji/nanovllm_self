#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

DATASET="${DATASET:-data/serving_benchmarks/sharegpt_qwen3_tokenized_5k.jsonl}"
LIMIT="${LIMIT:-200}"
PROFILE="${PROFILE:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
INITIAL_BACKLOG_S="${INITIAL_BACKLOG_S:-}"
ARRIVAL_MODE="${ARRIVAL_MODE:-burst}"
REQUEST_RATE="${REQUEST_RATE:-4.0}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-256}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
RESULT_TAG="${RESULT_TAG:-default}"

PROFILE_ARG=()
if [[ -n "${PROFILE}" ]]; then
  PROFILE_ARG=(--profile "${PROFILE}")
fi

cd "${ROOT_DIR}"

OUTPUT_DIR="pd_self/multiprocess/result/agent_scheduler/${RESULT_TAG}/${ARRIVAL_MODE}_w${NUM_WORKERS}_cap${MAX_OUTPUT_TOKENS_CAP}/${PROFILE:-all}"

echo "[agent-scheduler] dataset=${DATASET}"
echo "[agent-scheduler] profile=${PROFILE:-all} limit=${LIMIT} num_workers=${NUM_WORKERS} initial_backlog_s=${INITIAL_BACKLOG_S:-none} arrival_mode=${ARRIVAL_MODE} request_rate=${REQUEST_RATE} cap=${MAX_OUTPUT_TOKENS_CAP}"
echo "[agent-scheduler] output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" pd_self/multiprocess/evaluation/benchmark_agent_load_scheduler.py \
  --dataset "${DATASET}" \
  --limit "${LIMIT}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
  --num-workers "${NUM_WORKERS}" \
  --initial-backlog-s "${INITIAL_BACKLOG_S}" \
  --arrival-mode "${ARRIVAL_MODE}" \
  --request-rate "${REQUEST_RATE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${PROFILE_ARG[@]}"
