#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"
PROFILES="${PROFILES:-short_in_short_out short_in_long_out long_in_short_out long_in_long_out mixed_chat}"
BASELINE_MODE="${BASELINE_MODE:-pipeline_pd}"
CANDIDATE_MODE="${CANDIDATE_MODE:-pipeline_pd_sync_gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-pd_self/multiprocess/result/correctness/pipeline_pd_vs_sync_gpu}"

cd "${ROOT_DIR}"

for profile in ${PROFILES}; do
  left="pd_self/multiprocess/result/${BASELINE_MODE}/${profile}/synthetic_metrics.jsonl"
  right="pd_self/multiprocess/result/${CANDIDATE_MODE}/${profile}/synthetic_metrics.jsonl"
  output="${OUTPUT_DIR}/${profile}.json"

  echo "[correctness] profile=${profile}"
  "${PYTHON_BIN}" pd_self/multiprocess/evaluation/check_pipeline_correctness.py \
    --left "${left}" \
    --right "${right}" \
    --output "${output}"
done

echo "[correctness] done"
