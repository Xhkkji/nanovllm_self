#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/benchmark_decode.py" "$@"
