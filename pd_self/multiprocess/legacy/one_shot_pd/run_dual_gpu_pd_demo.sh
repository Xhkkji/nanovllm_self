#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

# Legacy one-shot PD demo。
# 当前主线使用 run_pipeline_pd_benchmark.sh / run_pipeline_profile_matrix.sh。

PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
KV_CACHE_QUANT_MODE="${KV_CACHE_QUANT_MODE:-int8_mock}"
PROMPT="${PROMPT:-What is KV cache?}"

PAYLOAD_FILE="${PAYLOAD_FILE:-/tmp/nanovllm_pd_payload.pkl}"
DONE_FILE="${DONE_FILE:-/tmp/nanovllm_pd_decode.done}"

RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/pd_self/multiprocess/result}"
LOG_DIR="${LOG_DIR:-${RESULT_DIR}}"
METRICS_DIR="${METRICS_DIR:-${RESULT_DIR}}"
PREFILL_LOG="${LOG_DIR}/dual_gpu_pd_prefill.log"
DECODE_LOG="${LOG_DIR}/dual_gpu_pd_decode.log"
PREFILL_METRICS="${METRICS_DIR}/dual_gpu_pd_prefill_metrics.json"
DECODE_METRICS="${METRICS_DIR}/dual_gpu_pd_decode_metrics.json"

mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
rm -f "${PAYLOAD_FILE}" "${DONE_FILE}" "${PREFILL_LOG}" "${DECODE_LOG}" \
      "${PREFILL_METRICS}" "${DECODE_METRICS}"

cd "${ROOT_DIR}"

cleanup() {
  if [[ -n "${PREFILL_PID:-}" ]] && kill -0 "${PREFILL_PID}" 2>/dev/null; then
    kill "${PREFILL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[dual-gpu-pd] starting prefill worker on GPU ${PREFILL_GPU}"
CUDA_VISIBLE_DEVICES="${PREFILL_GPU}" "${PYTHON_BIN}" \
  pd_self/multiprocess/legacy/one_shot_pd/prefill_worker.py \
  --prompt "${PROMPT}" \
  --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
  --out "${PAYLOAD_FILE}" \
  --done-file "${DONE_FILE}" \
  --metrics-out "${PREFILL_METRICS}" \
  >"${PREFILL_LOG}" 2>&1 &
PREFILL_PID=$!

echo "[dual-gpu-pd] waiting for payload ${PAYLOAD_FILE}"
for _ in $(seq 1 600); do
  if [[ -f "${PAYLOAD_FILE}" ]]; then
    break
  fi
  if ! kill -0 "${PREFILL_PID}" 2>/dev/null; then
    echo "[dual-gpu-pd] prefill worker exited before payload was written"
    cat "${PREFILL_LOG}" || true
    exit 1
  fi
  sleep 0.2
done

if [[ ! -f "${PAYLOAD_FILE}" ]]; then
  echo "[dual-gpu-pd] timed out waiting for payload"
  cat "${PREFILL_LOG}" || true
  exit 1
fi

echo "[dual-gpu-pd] starting decode worker on GPU ${DECODE_GPU}"
CUDA_VISIBLE_DEVICES="${DECODE_GPU}" "${PYTHON_BIN}" \
  pd_self/multiprocess/legacy/one_shot_pd/decode_worker.py \
  --kv-cache-quant-mode "${KV_CACHE_QUANT_MODE}" \
  --infile "${PAYLOAD_FILE}" \
  --done-file "${DONE_FILE}" \
  --metrics-out "${DECODE_METRICS}" \
  >"${DECODE_LOG}" 2>&1

wait "${PREFILL_PID}"
trap - EXIT

echo "[dual-gpu-pd] done"
echo "[dual-gpu-pd] prefill log: ${PREFILL_LOG}"
echo "[dual-gpu-pd] decode log : ${DECODE_LOG}"
echo "[dual-gpu-pd] prefill metrics: ${PREFILL_METRICS}"
echo "[dual-gpu-pd] decode metrics : ${DECODE_METRICS}"
