import argparse
import json
import math
import os
import sys
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT_DIR)

from benchmark_synthetic_common import (
    DEFAULT_DATASET,
    DEFAULT_RESULT_DIR,
    cap_max_tokens,
    load_jsonl,
    write_json,
    write_jsonl,
)
from pd_self.multiprocess.agent_scheduler import (
    AgentSchedulerConfig,
    build_agent_request_estimate,
    init_worker_states,
    parse_initial_backlogs,
    schedule_request,
    worker_summary,
)


def percentile(values, pct):
    """计算百分位数，用于模拟调度结果的延迟/排队时间统计。"""
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[lower]
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def arrival_time_s(idx, args):
    """根据到达模式生成第 idx 个请求的模拟到达时间。"""
    if args.arrival_mode == "burst":
        return 0.0
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be positive when arrival_mode=constant_rate")
    return idx / args.request_rate


def config_from_args(args):
    """把命令行参数转换为 AgentSchedulerConfig，方便调节复杂度和服务时间模型。"""
    return AgentSchedulerConfig(
        short_input_threshold=args.short_input_threshold,
        long_input_threshold=args.long_input_threshold,
        short_output_threshold=args.short_output_threshold,
        long_output_threshold=args.long_output_threshold,
        prefill_complexity_weight=args.prefill_complexity_weight,
        decode_complexity_weight=args.decode_complexity_weight,
        tool_complexity_weight=args.tool_complexity_weight,
        step_complexity_weight=args.step_complexity_weight,
        prefill_tokens_per_s=args.prefill_tokens_per_s,
        decode_tokens_per_s=args.decode_tokens_per_s,
        base_request_overhead_s=args.base_request_overhead_s,
        tool_call_time_s=args.tool_call_time_s,
        step_overhead_s=args.step_overhead_s,
        queue_weight=args.queue_weight,
        finish_time_weight=args.finish_time_weight,
        complexity_capacity=args.complexity_capacity,
        affinity_max_extra_wait_s=args.affinity_max_extra_wait_s,
    )


def estimate_to_row(req):
    """把 AgentRequestEstimate 转成可写入 metrics.jsonl 的基础字段。"""
    return {
        "request_index": req.request_index,
        "request_id": req.request_id,
        "session_id": req.session_id,
        "profile": req.profile,
        "task_type": req.task_type,
        "input_tokens": req.input_tokens,
        "output_tokens": req.output_tokens,
        "estimated_tool_calls": req.estimated_tool_calls,
        "estimated_steps": req.estimated_steps,
        "complexity_score": req.complexity_score,
    }


def simulate(requests, scheduler, args):
    """纯模拟调度：不启动 GPU，只用请求长度和虚拟 worker 状态比较策略效果。"""
    config = config_from_args(args)
    initial_backlogs = parse_initial_backlogs(args.initial_backlog_s, args.num_workers)
    workers = init_worker_states(args.num_workers, initial_backlogs)
    session_to_worker = {}
    rows = []

    for idx, req in enumerate(requests):
        arrival_t = arrival_time_s(idx, args)
        schedule_meta = schedule_request(
            workers,
            scheduler,
            idx,
            req,
            arrival_t,
            config,
            session_to_worker=session_to_worker,
        )
        rows.append(
            {
                **estimate_to_row(req),
                "scheduler": scheduler,
                **schedule_meta,
            }
        )

    return rows, workers


def summarize(rows, workers, args):
    """汇总模拟结果，包括吞吐、e2e、排队时间和 worker 负载不均衡度。"""
    if not rows:
        return {}
    e2e = [row["estimated_e2e_time_s"] for row in rows]
    queue_wait = [row["queue_wait_time_s"] for row in rows]
    service = [row["estimated_service_time_s"] for row in rows]
    first_arrival = min(row["arrival_time_s"] for row in rows)
    last_finish = max(row["finish_time_s"] for row in rows)
    makespan = last_finish - first_arrival
    total_output_tokens = sum(row["output_tokens"] for row in rows)
    workers_summary = worker_summary(workers)

    return {
        "num_requests": len(rows),
        "num_workers": args.num_workers,
        "initial_backlog_s": parse_initial_backlogs(args.initial_backlog_s, args.num_workers),
        "arrival_mode": args.arrival_mode,
        "request_rate": args.request_rate,
        "max_output_tokens_cap": args.max_output_tokens_cap,
        "generated_tokens": total_output_tokens,
        "makespan_s": makespan,
        "throughput_generated_tok_s": (
            total_output_tokens / makespan if makespan > 0 else 0.0
        ),
        "estimated_e2e_time_s": {
            "avg": sum(e2e) / len(e2e),
            "p50": percentile(e2e, 0.50),
            "p90": percentile(e2e, 0.90),
            "p99": percentile(e2e, 0.99),
            "max": max(e2e),
        },
        "queue_wait_time_s": {
            "avg": sum(queue_wait) / len(queue_wait),
            "p50": percentile(queue_wait, 0.50),
            "p90": percentile(queue_wait, 0.90),
            "p99": percentile(queue_wait, 0.99),
            "max": max(queue_wait),
        },
        "estimated_service_time_s": {
            "avg": sum(service) / len(service),
            "p50": percentile(service, 0.50),
            "p90": percentile(service, 0.90),
            "p99": percentile(service, 0.99),
            "max": max(service),
        },
        **workers_summary,
    }


def compare_summaries(round_robin, load_aware):
    """把候选策略和 round_robin baseline 做比值对比。"""
    def ratio(a, b):
        """内部安全除法，避免分母为 0 时抛异常。"""
        return a / b if b else None

    rr_e2e = round_robin["estimated_e2e_time_s"]
    la_e2e = load_aware["estimated_e2e_time_s"]
    rr_q = round_robin["queue_wait_time_s"]
    la_q = load_aware["queue_wait_time_s"]
    return {
        "candidate_over_round_robin_throughput": ratio(
            load_aware["throughput_generated_tok_s"],
            round_robin["throughput_generated_tok_s"],
        ),
        "candidate_avg_e2e_over_round_robin": ratio(la_e2e["avg"], rr_e2e["avg"]),
        "candidate_p90_e2e_over_round_robin": ratio(la_e2e["p90"], rr_e2e["p90"]),
        "candidate_p99_e2e_over_round_robin": ratio(la_e2e["p99"], rr_e2e["p99"]),
        "candidate_avg_queue_over_round_robin": ratio(la_q["avg"], rr_q["avg"]),
        "round_robin_worker_load_imbalance": round_robin["worker_load_imbalance"],
        "candidate_worker_load_imbalance": load_aware["worker_load_imbalance"],
    }


def select_requests(args):
    """从数据集中筛选参与模拟的请求，支持 profile 和 total_tokens 过滤。"""
    rows = []
    for row in load_jsonl(args.dataset):
        if args.profile and row.get("profile") != args.profile:
            continue
        total_tokens = int(row.get("total_tokens", 0))
        if args.max_total_tokens and total_tokens > args.max_total_tokens:
            continue
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break
    return rows


def parse_args():
    """解析 Agent 调度 simulation 的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Simulate Agent-aware load scheduling on serving requests."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--initial-backlog-s",
        default="",
        help="Comma-separated existing backlog seconds for each worker, e.g. 0,20,0,40.",
    )
    parser.add_argument("--arrival-mode", choices=["burst", "constant_rate"], default="burst")
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--prefill-tokens-per-s", type=float, default=4000.0)
    parser.add_argument("--decode-tokens-per-s", type=float, default=14.0)
    parser.add_argument("--base-request-overhead-s", type=float, default=0.02)
    parser.add_argument("--tool-call-time-s", type=float, default=0.30)
    parser.add_argument("--step-overhead-s", type=float, default=0.05)
    parser.add_argument("--short-input-threshold", type=int, default=256)
    parser.add_argument("--long-input-threshold", type=int, default=1024)
    parser.add_argument("--short-output-threshold", type=int, default=128)
    parser.add_argument("--long-output-threshold", type=int, default=128)
    parser.add_argument("--prefill-complexity-weight", type=float, default=0.3)
    parser.add_argument("--decode-complexity-weight", type=float, default=1.0)
    parser.add_argument("--tool-complexity-weight", type=float, default=64.0)
    parser.add_argument("--step-complexity-weight", type=float, default=32.0)
    parser.add_argument("--queue-weight", type=float, default=1.0)
    parser.add_argument("--finish-time-weight", type=float, default=1.0)
    parser.add_argument("--complexity-capacity", type=float, default=10000.0)
    parser.add_argument(
        "--candidate-scheduler",
        default="load_aware",
        choices=["load_aware", "affinity_load_aware"],
        help="Scheduler compared against round_robin in this simulation.",
    )
    parser.add_argument(
        "--affinity-max-extra-wait-s",
        type=float,
        default=2.0,
        help="Max extra queue wait allowed before affinity_load_aware migrates a session.",
    )
    return parser.parse_args()

def main():
    """simulation 主入口：读取请求、分别跑 baseline/候选策略，并写出对比结果。"""
    args = parse_args()
    output_dir = Path(
        args.output_dir
        or Path(DEFAULT_RESULT_DIR)
        / "agent_scheduler"
        / f"{args.arrival_mode}_w{args.num_workers}_cap{args.max_output_tokens_cap}"
        / (args.profile or "all")
    )
    raw_rows = select_requests(args)
    if not raw_rows:
        raise RuntimeError("no requests selected")

    config = config_from_args(args)
    requests = [
        build_agent_request_estimate(
            row,
            idx,
            cap_max_tokens(row, args.max_output_tokens_cap),
            config,
        )
        for idx, row in enumerate(raw_rows)
    ]
    rr_rows, rr_workers = simulate(requests, "round_robin", args)
    la_rows, la_workers = simulate(requests, args.candidate_scheduler, args)
    rr_summary = summarize(rr_rows, rr_workers, args)
    la_summary = summarize(la_rows, la_workers, args)
    summary = {
        "dataset": args.dataset,
        "profile": args.profile,
        "limit": args.limit,
        "selected_requests": len(requests),
        "round_robin": rr_summary,
        args.candidate_scheduler: la_summary,
        "comparison": compare_summaries(rr_summary, la_summary),
    }

    write_jsonl(output_dir / "round_robin_metrics.jsonl", rr_rows)
    write_jsonl(output_dir / f"{args.candidate_scheduler}_metrics.jsonl", la_rows)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["comparison"], ensure_ascii=False, indent=2))
    print("summary_written", output_dir / "summary.json")


if __name__ == "__main__":
    main()
