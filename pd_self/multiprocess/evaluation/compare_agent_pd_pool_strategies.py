import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    """读取单个策略的 synthetic_summary.json。"""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    """读取单个策略的逐请求 synthetic_metrics.jsonl。"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def avg(values: list[float]) -> float:
    """计算平均值；空列表返回 0，便于 JSON 汇总。"""
    return sum(values) / len(values) if values else 0.0


def ratio(candidate, baseline):
    """计算候选策略相对 baseline 的比值，分母为空时返回 None。"""
    if candidate is None or baseline in (None, 0):
        return None
    return candidate / baseline


def strategy_metrics(summary_path: Path) -> dict:
    """
    ########################### Agent-aware PD Pool 策略对比 ###########################
    从单个 pool 策略结果目录里提取关键指标。

    这里不关心 nano-vLLM 内部 scheduler，只观察外层 Agent-aware coordinator
    把请求路由到哪个 prefill worker / decode worker，以及最终吞吐和延迟。
    """
    summary = read_json(summary_path)
    metrics_path = summary_path.with_name("synthetic_metrics.jsonl")
    if not metrics_path.exists():
        metrics_path = summary_path.with_name(summary_path.stem.replace("summary", "metrics") + ".jsonl")
    rows = read_jsonl(metrics_path)
    measured = [row for row in rows if row.get("phase", "measure") == "measure"]

    prefill_counts = Counter(str(row.get("prefill_worker_id")) for row in measured)
    decode_counts = Counter(str(row.get("decode_worker_id")) for row in measured)
    route_counts = Counter(
        f"P{row.get('prefill_worker_id')}->D{row.get('decode_worker_id')}"
        for row in measured
    )
    rank_route_counts = Counter(
        f"{row.get('src_rank')}->{row.get('dst_rank')}"
        for row in measured
    )
    affinity_hits = sum(1 for row in measured if row.get("affinity_hit"))
    cross_routes = sum(
        1
        for row in measured
        if row.get("prefill_worker_id") != row.get("decode_worker_id")
    )

    time_summary = summary.get("time", {})
    transfer_times = [
        float(row.get("transfer_time_s", 0.0))
        for row in measured
        if row.get("transfer_time_s") is not None
    ]

    return {
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "scheduler": summary.get("scheduler"),
        "measured_requests": summary.get("measured_requests", len(measured)),
        "generated_tokens": summary.get("generated_tokens", 0),
        "pipeline_measure_wall_time_s": summary.get("pipeline_measure_wall_time_s"),
        "pipeline_throughput_generated_tok_s": summary.get(
            "pipeline_throughput_generated_tok_s"
        ),
        "wall_e2e_avg_s": time_summary.get("wall_e2e_time_s", {}).get("avg"),
        "wall_e2e_p90_s": time_summary.get("wall_e2e_time_s", {}).get("p90"),
        "core_e2e_avg_s": time_summary.get("core_e2e_time_s", {}).get("avg"),
        "core_e2e_p90_s": time_summary.get("core_e2e_time_s", {}).get("p90"),
        "transfer_avg_s": avg(transfer_times),
        "affinity_hits": affinity_hits,
        "affinity_hit_rate": affinity_hits / len(measured) if measured else 0.0,
        "cross_route_count": cross_routes,
        "cross_route_rate": cross_routes / len(measured) if measured else 0.0,
        "prefill_worker_counts": dict(prefill_counts),
        "decode_worker_counts": dict(decode_counts),
        "route_counts": dict(route_counts),
        "rank_route_counts": dict(rank_route_counts),
        "prefill_pool_summary": summary.get("prefill_pool_summary", {}),
        "decode_pool_summary": summary.get("decode_pool_summary", {}),
    }


def markdown_table(results: dict) -> str:
    """生成一个简短 Markdown 表，方便直接贴到文档或简历材料里。"""
    lines = [
        "| strategy | reqs | tok/s | wall avg(s) | wall p90(s) | affinity | cross-route |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            "| {name} | {reqs} | {tok_s:.4f} | {avg_s:.4f} | {p90_s:.4f} | {aff:.2%} | {cross:.2%} |".format(
                name=name,
                reqs=result.get("measured_requests", 0),
                tok_s=float(result.get("pipeline_throughput_generated_tok_s") or 0.0),
                avg_s=float(result.get("wall_e2e_avg_s") or 0.0),
                p90_s=float(result.get("wall_e2e_p90_s") or 0.0),
                aff=float(result.get("affinity_hit_rate") or 0.0),
                cross=float(result.get("cross_route_rate") or 0.0),
            )
        )
    return "\n".join(lines)


def parse_args():
    """解析三策略 pool summary 路径和输出路径。"""
    parser = argparse.ArgumentParser(
        description="Compare round_robin/load_aware/affinity_load_aware Agent PD pool results."
    )
    parser.add_argument("--round-robin", required=True)
    parser.add_argument("--load-aware", required=True)
    parser.add_argument("--affinity-load-aware", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def main():
    """读取三份 Agent PD pool 结果，生成统一策略对比 JSON 和可选 Markdown 表。"""
    args = parse_args()
    paths = {
        "round_robin": Path(args.round_robin),
        "load_aware": Path(args.load_aware),
        "affinity_load_aware": Path(args.affinity_load_aware),
    }
    results = {name: strategy_metrics(path) for name, path in paths.items()}
    baseline = results["round_robin"]

    comparison = {}
    for name, result in results.items():
        comparison[name] = {
            **result,
            "throughput_over_round_robin": ratio(
                result["pipeline_throughput_generated_tok_s"],
                baseline["pipeline_throughput_generated_tok_s"],
            ),
            "wall_e2e_avg_over_round_robin": ratio(
                result["wall_e2e_avg_s"],
                baseline["wall_e2e_avg_s"],
            ),
            "wall_e2e_p90_over_round_robin": ratio(
                result["wall_e2e_p90_s"],
                baseline["wall_e2e_p90_s"],
            ),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": "round_robin",
                "strategies": comparison,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_table(comparison) + "\n", encoding="utf-8")

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("comparison_written", output_path)
    if args.markdown_output:
        print("comparison_markdown_written", args.markdown_output)


if __name__ == "__main__":
    main()
