import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]


def read_json(path: Path):
    """读取 JSON 文件，主要用于加载每条请求对应的 decode_metrics。"""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    """读取 metrics.jsonl，并返回所有逐请求记录。"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path: str) -> Path:
    """把相对路径解析到仓库根目录，方便脚本从任意工作目录执行。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def load_rows_by_request(metrics_path: Path, include_warmup: bool):
    """按 request_id 建索引，便于左右两份 benchmark 输出逐请求对拍。"""
    rows = {}
    for row in read_jsonl(metrics_path):
        if not include_warmup and row.get("phase") != "measure":
            continue
        rows[row["request_id"]] = row
    return rows


def comparable_decode_fields(row):
    """
    ########################### 异步 PD correctness 对拍 ###########################

    benchmark 的 metrics.jsonl 只保存 decode_metrics 文件路径。
    这里读取每条请求的 decode_metrics，并抽取可稳定比较的输出侧字段。

    当前最小 correctness 目标：
    - shared_memory PD 和 sync_gpu async PD 对同一 request 生成相同 token 序列。
    - generated_tokens / final_token_lens 也一致。

    注意：
    - 不比较耗时字段。
    - 不比较 seq_idx / decode_finished_ids，因为不同运行中的内部 seq 编号不一定
      是 correctness 语义的一部分。
    """
    metrics = read_json(resolve_path(row["decode_metrics"]))
    return {
        "generated_tokens": metrics.get("generated_tokens"),
        "decode_step_tokens": metrics.get("decode_step_tokens", []),
        "final_token_lens": metrics.get("final_token_lens", []),
    }


def main():
    """对比两份 pipeline PD 输出，检查生成 token 序列和长度是否一致。"""
    parser = argparse.ArgumentParser(
        description="Compare pipeline PD correctness between two benchmark outputs."
    )
    parser.add_argument("--left", required=True, help="Baseline synthetic_metrics.jsonl")
    parser.add_argument("--right", required=True, help="Candidate synthetic_metrics.jsonl")
    parser.add_argument("--output", required=True, help="Output correctness summary JSON")
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Also compare warmup rows. Default compares measured rows only.",
    )
    args = parser.parse_args()

    left_path = resolve_path(args.left)
    right_path = resolve_path(args.right)
    left_rows = load_rows_by_request(left_path, args.include_warmup)
    right_rows = load_rows_by_request(right_path, args.include_warmup)

    common_ids = sorted(set(left_rows) & set(right_rows))
    missing_left = sorted(set(right_rows) - set(left_rows))
    missing_right = sorted(set(left_rows) - set(right_rows))

    mismatches = []
    for request_id in common_ids:
        left_fields = comparable_decode_fields(left_rows[request_id])
        right_fields = comparable_decode_fields(right_rows[request_id])
        if left_fields == right_fields:
            continue
        mismatches.append(
            {
                "request_id": request_id,
                "left": left_fields,
                "right": right_fields,
            }
        )

    summary = {
        "left": str(left_path),
        "right": str(right_path),
        "include_warmup": args.include_warmup,
        "num_left": len(left_rows),
        "num_right": len(right_rows),
        "num_common": len(common_ids),
        "num_missing_left": len(missing_left),
        "num_missing_right": len(missing_right),
        "num_mismatches": len(mismatches),
        "passed": not missing_left and not missing_right and not mismatches,
        "missing_left": missing_left[:20],
        "missing_right": missing_right[:20],
        "mismatches": mismatches[:20],
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("correctness_written", output_path)

    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
