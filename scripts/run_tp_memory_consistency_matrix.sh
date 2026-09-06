#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/torchrun}"

RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/results/tp_smoke}"
PROMPT="${TP_PROMPT:-What is a large language model?}"
MAX_TOKENS="${TP_MAX_TOKENS:-16}"

mkdir -p "${RESULT_DIR}"
cd "${ROOT_DIR}"

run_case() {
  local tp_size="$1"
  local devices="$2"
  local output="${RESULT_DIR}/tp${tp_size}.json"

  echo "[tp-matrix] running tp=${tp_size} devices=${devices}"

  if [[ "${tp_size}" == "1" ]]; then
    TP_SIZE=1 \
    TP_PROMPT="${PROMPT}" \
    TP_MAX_TOKENS="${MAX_TOKENS}" \
    TP_RESULT_PATH="${output}" \
    CUDA_VISIBLE_DEVICES="${devices}" \
      "${PYTHON_BIN}" tests/test_tp_generate.py
  else
    TP_SIZE="${tp_size}" \
    TP_PROMPT="${PROMPT}" \
    TP_MAX_TOKENS="${MAX_TOKENS}" \
    TP_RESULT_PATH="${output}" \
    CUDA_VISIBLE_DEVICES="${devices}" \
      "${TORCHRUN_BIN}" \
        --nproc_per_node="${tp_size}" \
        tests/test_tp_generate.py
  fi
}

run_case 1 "0"
run_case 2 "0,1"
run_case 4 "0,1,2,3"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

result_dir = Path("${RESULT_DIR}")
rows = {}
for tp in (1, 2, 4):
    path = result_dir / f"tp{tp}.json"
    rows[tp] = json.loads(path.read_text(encoding="utf-8"))

base_tokens = rows[1]["generated_token_ids"]
summary = {
    "result_dir": str(result_dir),
    "baseline_tp": 1,
    "cases": {},
}

for tp, row in rows.items():
    tokens = row["generated_token_ids"]
    summary["cases"][str(tp)] = {
        "generated_token_ids": tokens,
        "matches_tp1": tokens == base_tokens,
        "rank_memory": row["rank_memory"],
        "output_text": row["output_text"],
    }

summary_path = result_dir / "summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("===== TP MEMORY / CONSISTENCY SUMMARY =====")
for tp in (1, 2, 4):
    case = summary["cases"][str(tp)]
    peak = [round(item["max_memory_allocated_mb"], 2) for item in case["rank_memory"]]
    print(f"tp={tp} matches_tp1={case['matches_tp1']} peak_allocated_mb={peak}")
print(f"summary_written {summary_path}")
PY
