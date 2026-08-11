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


def strategy_metrics(summary_path: Path) -> dict:
    """从一个策略目录提取吞吐、延迟、affinity 和 pair 分布指标。"""
    summary = read_json(summary_path)
    metrics_path = summary_path.with_name("synthetic_metrics.jsonl")
    rows = read_jsonl(metrics_path)
    measured = [row for row in rows if row.get("phase", "measure") == "measure"]
    affinity_hits = sum(1 for row in measured if row.get("affinity_hit"))
    pair_counts = Counter(row.get("worker_pair_id") for row in measured)
    feedback_loads = [
        float(row.get("worker_feedback_load_s", 0.0))
        for row in measured
        if row.get("worker_feedback_load_s") is not None
    ]
    time_summary = summary.get("time", {})

    return {
        "summary_path": str(summary_path),
        "scheduler": summary.get("scheduler"),
        "measured_requests": summary.get("measured_requests", len(measured)),
        "generated_tokens": summary.get("generated_tokens", 0),
        "pipeline_measure_wall_time_s": summary.get("pipeline_measure_wall_time_s"),
        "pipeline_throughput_generated_tok_s": summary.get(
            "pipeline_throughput_generated_tok_s"
        ),
        "core_e2e_avg_s": time_summary.get("core_e2e_time_s", {}).get("avg"),
        "core_e2e_p90_s": time_summary.get("core_e2e_time_s", {}).get("p90"),
        "wall_e2e_avg_s": time_summary.get("wall_e2e_time_s", {}).get("avg"),
        "wall_e2e_p90_s": time_summary.get("wall_e2e_time_s", {}).get("p90"),
        "affinity_hits": affinity_hits,
        "affinity_hit_rate": (
            affinity_hits / len(measured) if measured else 0.0
        ),
        "worker_pair_counts": dict(pair_counts),
        "worker_load_imbalance": summary.get("agent_scheduler_workers", {}).get(
            "worker_load_imbalance"
        ),
        "avg_worker_feedback_load_s": (
            sum(feedback_loads) / len(feedback_loads) if feedback_loads else 0.0
        ),
    }


def ratio(candidate, baseline):
    """计算候选策略相对 baseline 的比值，分母为空时返回 None。"""
    if candidate is None or baseline in (None, 0):
        return None
    return candidate / baseline


def parse_args():
    """解析三策略 summary 路径和对比结果输出路径。"""
    parser = argparse.ArgumentParser(
        description="Compare round_robin/load_aware/affinity_load_aware Agent PD results."
    )
    parser.add_argument("--round-robin", required=True)
    parser.add_argument("--load-aware", required=True)
    parser.add_argument("--affinity-load-aware", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    """读取三份真实 multi-PD 结果，生成统一的策略对比 summary。"""
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
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("comparison_written", output_path)


if __name__ == "__main__":
    main()
