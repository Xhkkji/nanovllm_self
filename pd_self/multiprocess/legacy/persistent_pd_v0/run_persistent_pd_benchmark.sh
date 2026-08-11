#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

# Legacy persistent PD benchmark。
# 当前主线使用 run_pipeline_pd_benchmark.sh / run_pipeline_profile_matrix.sh。
#
# 这个脚本只是薄封装：
# - prefill/decode worker 的启动和生命周期由 legacy benchmark_synthetic_pd_persistent.py 管。
# - 这里主要把常用参数转成环境变量，方便命令行快速跑。
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
LIMIT="${LIMIT:-5}"
WARMUP="${WARMUP:-0}"
MAX_OUTPUT_TOKENS_CAP="${MAX_OUTPUT_TOKENS_CAP:-16}"
PROFILE_ARG=()

if [[ -n "${PROFILE:-}" ]]; then
  PROFILE_ARG=(--profile "${PROFILE}")
fi

cd "${ROOT_DIR}"

echo "[persistent-pd] profile=${PROFILE:-all} limit=${LIMIT} warmup=${WARMUP} max_output_tokens_cap=${MAX_OUTPUT_TOKENS_CAP}"

"${PYTHON_BIN}" pd_self/multiprocess/legacy/persistent_pd_v0/benchmark_synthetic_pd_persistent.py \
  --limit "${LIMIT}" \
  --warmup "${WARMUP}" \
  --max-output-tokens-cap "${MAX_OUTPUT_TOKENS_CAP}" \
  --prefill-gpu "${PREFILL_GPU}" \
  --decode-gpu "${DECODE_GPU}" \
  --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
  --python-bin "${PYTHON_BIN}" \
  "${PROFILE_ARG[@]}"
