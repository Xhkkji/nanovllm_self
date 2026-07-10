#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"
OUT_FILE="${1:-${ROOT_DIR}/results/runtime_bench.jsonl}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUT_FILE}")"
rm -f "${OUT_FILE}"

run_case() {
  local prompt="$1"
  local batch_size="$2"
  local gen_len="$3"
  local prefill_backend="$4"
  local decode_backend="$5"

  echo "==> prompt=${prompt} bs=${batch_size} gen=${gen_len} prefill=${prefill_backend} decode=${decode_backend}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/benchmark_runtime.py" \
    --prompt "${prompt}" \
    --batch-size "${batch_size}" \
    --gen-len "${gen_len}" \
    --prefill-backend "${prefill_backend}" \
    --decode-backend "${decode_backend}" \
    --output "${OUT_FILE}"
}

# 第一版只扫最有价值的几组
run_case short 1 64 torch torch
run_case short 1 64 torch flashattn
run_case medium 8 64 torch torch
run_case medium 8 64 torch flashattn

echo
echo "Results saved to: ${OUT_FILE}"

