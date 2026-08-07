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
LIMIT="${LIMIT:-5}"
WARMUP="${WARMUP:-0}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-16}"
DECODE_MODE="${DECODE_MODE:-continuous}"
MAX_ACTIVE_DECODE_REQUESTS="${MAX_ACTIVE_DECODE_REQUESTS:-4}"
PROFILE_ARG=()

if [[ -n "${PROFILE:-}" ]]; then
  PROFILE_ARG=(--profile "${PROFILE}")
fi

cd "${ROOT_DIR}"

echo "[pipeline-pd] profile=${PROFILE:-all} limit=${LIMIT} warmup=${WARMUP} max_output_tokens_cap=${MAX_OUTPUT_TOKENS_CAP} decode_mode=${DECODE_MODE} max_active_decode_requests=${MAX_ACTIVE_DECODE_REQUESTS} kv_transfer_backend=${KV_TRANSFER_BACKEND} nccl_port=${NCCL_PORT}"

"${PYTHON_BIN}" pd_self/multiprocess/evaluation/benchmark_synthetic_pd_pipeline.py \
  --limit "${LIMIT}" \
  --warmup "${WARMUP}" \
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
  --prefill-gpu "${PREFILL_GPU}" \
  --decode-gpu "${DECODE_GPU}" \
  --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
  --kv-transfer-backend "${KV_TRANSFER_BACKEND}" \
  --nccl-port "${NCCL_PORT}" \
  --decode-mode "${DECODE_MODE}" \
  --max-active-decode-requests "${MAX_ACTIVE_DECODE_REQUESTS}" \
  --python-bin "${PYTHON_BIN}" \
  "${PROFILE_ARG[@]}"
