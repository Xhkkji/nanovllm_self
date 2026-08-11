import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT_DIR)

from benchmark_synthetic_common import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_RESULT_DIR,
    cap_max_tokens,
    select_requests,
    summarize,
    write_json,
    write_jsonl,
)
from benchmark_synthetic_pd_pipeline import (  # noqa: E402
    atomic_write_json,
    build_metrics_rows,
    build_paths,
    clean_work_dir,
    read_json,
    request_base,
    start_worker,
    wait_for_file,
)
from pd_self.multiprocess.agent_scheduler import (  # noqa: E402
    AgentSchedulerConfig,
    build_agent_request_estimate,
    init_worker_states,
    parse_initial_backlogs,
    schedule_request,
    worker_summary,
)


def parse_csv(value):
    """解析逗号分隔的 GPU 列表，例如 '0,2' -> ['0', '2']。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def make_pair_args(args, pair):
    """为单个 PD pair 拼出启动 persistent worker 需要的最小参数对象。"""
    # 多 PD pair：每个 pair 都复用已有 persistent worker，只是 GPU、work_dir 和 NCCL 端口不同。
    # 这里用 SimpleNamespace 拼出 start_worker 需要的最小参数集合，避免改动原 worker 启动逻辑。
    return SimpleNamespace(
        prefill_gpu=pair["prefill_gpu"],
        decode_gpu=pair["decode_gpu"],
        kv_cache_quant_mode=args.kv_cache_quant_mode,
        kv_transfer_backend=args.kv_transfer_backend,
        nccl_port=pair["nccl_port"],
        poll_interval_s=args.poll_interval_s,
        decode_mode=args.decode_mode,
        max_active_decode_requests=args.max_active_decode_requests,
        max_pending_sends=args.max_pending_sends,
        max_pending_recvs=args.max_pending_recvs,
        python_bin=args.python_bin,
    )


def build_worker_pairs(args, root_work_dir: Path):
    """根据 prefill/decode GPU 列表构造多组 PD pair 的描述信息和工作目录。"""
    prefill_gpus = parse_csv(args.prefill_gpus)
    decode_gpus = parse_csv(args.decode_gpus)
    if len(prefill_gpus) != len(decode_gpus):
        raise ValueError("--prefill-gpus and --decode-gpus must have same length")
    if args.num_worker_pairs and args.num_worker_pairs != len(prefill_gpus):
        raise ValueError("--num-worker-pairs must match the GPU pair list length")

    pairs = []
    for pair_id, (prefill_gpu, decode_gpu) in enumerate(zip(prefill_gpus, decode_gpus)):
        pair_work_dir = root_work_dir / f"pair_{pair_id}"
        pairs.append(
            {
                "pair_id": pair_id,
                "prefill_gpu": prefill_gpu,
                "decode_gpu": decode_gpu,
                "nccl_port": args.nccl_port_base + pair_id,
                "work_dir": pair_work_dir,
                "prefill_log": pair_work_dir / "pipeline_prefill.log",
                "decode_log": pair_work_dir / "pipeline_decode.log",
            }
        )
    return pairs


def start_worker_pairs(args, pairs):
    """启动所有 PD pair 的 prefill/decode worker，并等待每个 worker 写 ready 文件。"""
    procs = []
    ready = []
    for pair in pairs:
        clean_work_dir(pair["work_dir"])
        pair_args = make_pair_args(args, pair)
        prefill_proc = start_worker(pair_args, "prefill", pair["work_dir"], pair["prefill_log"])
        decode_proc = start_worker(pair_args, "decode", pair["work_dir"], pair["decode_log"])
        procs.extend([prefill_proc, decode_proc])

    for pair in pairs:
        wait_for_file(pair["work_dir"] / "prefill_worker.ready.json", args.startup_timeout_s)
        wait_for_file(pair["work_dir"] / "decode_worker.ready.json", args.startup_timeout_s)
        ready.append(
            {
                "pair_id": pair["pair_id"],
                "prefill_gpu": pair["prefill_gpu"],
                "decode_gpu": pair["decode_gpu"],
                "nccl_port": pair["nccl_port"],
                "prefill_ready": read_json(pair["work_dir"] / "prefill_worker.ready.json"),
                "decode_ready": read_json(pair["work_dir"] / "decode_worker.ready.json"),
            }
        )
    return procs, ready


def shutdown_worker_pairs(pairs, procs):
    """向所有 worker 写 shutdown 文件并等待退出，超时后兜底 kill 子进程。"""
    for pair in pairs:
        try:
            (pair["work_dir"] / "shutdown").write_text("shutdown\n", encoding="utf-8")
        except FileNotFoundError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)


def raise_if_worker_error(paths_by_base):
    """检查所有已提交请求是否产生 worker error 文件，有则立即抛出详细错误。"""
    for paths in paths_by_base.values():
        for key in ("prefill_error", "decode_error"):
            error_path = paths[key]
            if error_path.exists():
                raise RuntimeError(
                    f"worker error: {error_path}\n{error_path.read_text(encoding='utf-8')}"
                )


def wait_for_decode_done(paths_by_base, timeout_s, poll_interval_s):
    """batch 模式下等待一批请求全部写出 decode_done，并记录完成时间。"""
    deadline = time.time() + timeout_s
    completed_at = {}
    while time.time() < deadline:
        raise_if_worker_error(paths_by_base)
        for base, paths in paths_by_base.items():
            if base in completed_at:
                continue
            if paths["decode_done"].exists():
                completed_at[base] = perf_counter()
        if len(completed_at) == len(paths_by_base):
            return completed_at
        time.sleep(poll_interval_s)
    missing = sorted(set(paths_by_base) - set(completed_at))
    raise TimeoutError(f"timed out waiting decode_done for {missing[:5]}")


def build_scheduler(args, num_pairs):
    """创建 Agent-aware 调度器运行态，包括 worker 状态和 session->worker 亲和表。"""
    initial_backlogs = parse_initial_backlogs(args.initial_backlog_s, num_pairs)
    return {
        "config": AgentSchedulerConfig(),
        "workers": init_worker_states(num_pairs, initial_backlogs),
        "initial_backlog_s": initial_backlogs,
        "session_to_worker": {},
    }


def read_worker_state(path: Path) -> dict:
    """读取 worker 写出的真实状态文件；文件暂未生成时返回空状态。"""
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}


def feedback_load_s(args, prefill_state: dict, decode_state: dict) -> float:
    """把 prefill/decode worker state 转成调度器可使用的轻量负载分数。"""
    # Runtime feedback 最小模型：
    # - active decode 是最直接的 decode 侧拥塞信号；
    # - pending recv/send 表示传输层还有未完成 handoff；
    # - request_queue_depth 表示 prefill 侧还有未消费的文件请求；
    # - busy 标志作为轻量补偿，避免刚进入执行但计数尚未增加时完全看不见负载。
    load_units = (
        float(decode_state.get("active_decode_requests", 0))
        + float(decode_state.get("pending_recvs", 0))
        + float(prefill_state.get("pending_sends", 0))
        + float(prefill_state.get("request_queue_depth", 0))
    )
    if prefill_state.get("busy"):
        load_units += 0.5
    if decode_state.get("busy"):
        load_units += 0.5
    return load_units * args.worker_feedback_scale_s


def refresh_worker_feedback(args, pairs, scheduler_runtime):
    """调度前读取各 PD pair 的真实 worker state，并更新 WorkerState.feedback_load_s。"""
    feedback_rows = []
    for pair, worker in zip(pairs, scheduler_runtime["workers"]):
        prefill_state = read_worker_state(pair["work_dir"] / "prefill_worker_state.json")
        decode_state = read_worker_state(pair["work_dir"] / "decode_worker_state.json")
        load_s = feedback_load_s(args, prefill_state, decode_state) if args.worker_feedback else 0.0
        worker.feedback_load_s = load_s
        worker.feedback_state = {
            "pair_id": pair["pair_id"],
            "prefill": prefill_state,
            "decode": decode_state,
            "feedback_load_s": load_s,
        }
        feedback_rows.append(worker.feedback_state)
    return feedback_rows


def submit_one(args, pairs, scheduler_runtime, entry):
    """提交单条请求：估计复杂度、选择 PD pair、写 request.json 并返回追踪元数据。"""
    idx, row, phase, measure_index = entry
    request_id = row.get("id", f"agent-pd-{idx:06d}")
    max_tokens = max(2, cap_max_tokens(row, args.max_output_tokens_cap))
    estimate = build_agent_request_estimate(
        row,
        idx,
        max_tokens,
        scheduler_runtime["config"],
    )
    worker_feedback_states = refresh_worker_feedback(args, pairs, scheduler_runtime)

    # Agent-aware 多 PD pair 调度：
    # scheduler 先基于请求复杂度和每个 pair 的虚拟 backlog 选择 pair_id；
    # 随后 request.json 会真实写入对应 pair 的 work_dir，由该 pair 的 P/D worker 执行。
    schedule_meta = schedule_request(
        scheduler_runtime["workers"],
        args.scheduler,
        idx,
        estimate,
        arrival_t_s=0.0,
        config=scheduler_runtime["config"],
        session_to_worker=scheduler_runtime["session_to_worker"],
    )
    pair = pairs[schedule_meta["worker_id"]]
    base = request_base(idx, request_id)
    paths = build_paths(pair["work_dir"], base)
    request = {
        **row,
        "id": request_id,
        "max_tokens": max_tokens,
    }

    now = perf_counter()
    metadata = {
        "phase": phase,
        "request_index": idx,
        "measure_index": measure_index,
        "request_id": request_id,
        "profile": row.get("profile"),
        "input_tokens_dataset": row.get("input_tokens"),
        "max_tokens": max_tokens,
        "target_output_tokens": row.get("max_tokens", row.get("output_len")),
        "total_tokens_dataset": row.get("total_tokens"),
        "scheduler": args.scheduler,
        # Agent trace 字段原样保留到真实 PD metrics，方便按 program/session/step
        # 还原一条 Agent 任务的完整执行轨迹，而不是只看到单条 request。
        "program_id": row.get("program_id", estimate.session_id),
        "session_id": estimate.session_id,
        "step_id": row.get("step_id"),
        "num_steps": row.get("num_steps"),
        "task_kind": row.get("task_kind"),
        "affinity_hit": schedule_meta["affinity_hit"],
        "preferred_worker_id": schedule_meta["preferred_worker_id"],
        "worker_pair_id": pair["pair_id"],
        "worker_slot_id": pair["pair_id"],
        "prefill_gpu": pair["prefill_gpu"],
        "decode_gpu": pair["decode_gpu"],
        "task_type": estimate.task_type,
        "estimated_tool_calls": estimate.estimated_tool_calls,
        "estimated_steps": estimate.estimated_steps,
        "complexity_score": estimate.complexity_score,
        "slot_queue_wait_time_s": schedule_meta["queue_wait_time_s"],
        "worker_feedback_load_s": schedule_meta["worker_feedback_load_s"],
        "worker_feedback_state": schedule_meta["worker_feedback_state"],
        "all_worker_feedback_states": worker_feedback_states,
        "estimated_service_time_s": schedule_meta["estimated_service_time_s"],
        "estimated_slot_e2e_time_s": schedule_meta["estimated_e2e_time_s"],
    }
    atomic_write_json(paths["request"], request)
    return base, paths, metadata, now


def submit_entries(args, pairs, scheduler_runtime, entries):
    """batch 提交一组请求，不等待单条完成，用于 warmup 或 batch load 模式。"""
    paths_by_base = {}
    metadata_by_base = {}
    submit_times = {}
    first_submit_t = None
    for entry in entries:
        base, paths, metadata, now = submit_one(args, pairs, scheduler_runtime, entry)
        paths_by_base[base] = paths
        metadata_by_base[base] = metadata
        submit_times[base] = now
        first_submit_t = now if first_submit_t is None else first_submit_t
    return paths_by_base, metadata_by_base, submit_times, first_submit_t


def run_closed_loop(args, pairs, scheduler_runtime, entries):
    """闭环压测：保持固定并发，某条请求完成后立刻补交下一条请求。"""
    paths_by_base = {}
    metadata_by_base = {}
    submit_times = {}
    completed_at = {}
    first_submit_t = None
    last_done_t = None
    next_idx = 0
    last_activity_t = time.time()

    def submit_until_full():
        """内部 helper：在 inflight 未达到 concurrency 前持续提交新请求。"""
        nonlocal first_submit_t, next_idx, last_activity_t
        while next_idx < len(entries):
            inflight = len(paths_by_base) - len(completed_at)
            if inflight >= args.concurrency:
                break
            base, paths, metadata, now = submit_one(
                args,
                pairs,
                scheduler_runtime,
                entries[next_idx],
            )
            paths_by_base[base] = paths
            metadata_by_base[base] = metadata
            submit_times[base] = now
            first_submit_t = now if first_submit_t is None else first_submit_t
            next_idx += 1
            last_activity_t = time.time()

    submit_until_full()
    while len(completed_at) < len(entries):
        raise_if_worker_error(paths_by_base)
        made_progress = False
        for base, paths in list(paths_by_base.items()):
            if base in completed_at:
                continue
            if paths["decode_done"].exists():
                completed_at[base] = perf_counter()
                last_done_t = completed_at[base]
                made_progress = True
        if made_progress:
            last_activity_t = time.time()
            submit_until_full()
        elif time.time() - last_activity_t > args.request_timeout_s:
            missing = sorted(set(paths_by_base) - set(completed_at))
            raise TimeoutError(f"timed out waiting decode_done for {missing[:5]}")
        else:
            time.sleep(args.poll_interval_s)

    return paths_by_base, metadata_by_base, submit_times, completed_at, first_submit_t, last_done_t


def parse_args():
    """解析真实多 PD pair Agent benchmark 的命令行参数。"""
    parser = argparse.ArgumentParser(description="Agent-aware multi PD-pair benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=16)
    parser.add_argument("--prefill-gpus", default="0,2")
    parser.add_argument("--decode-gpus", default="1,3")
    parser.add_argument("--num-worker-pairs", type=int, default=0)
    parser.add_argument(
        "--scheduler",
        default="load_aware",
        choices=["round_robin", "load_aware", "affinity_load_aware"],
    )
    parser.add_argument("--initial-backlog-s", default="")
    parser.add_argument("--load-mode", default="closed_loop", choices=["batch", "closed_loop"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument("--kv-transfer-backend", default="shared_memory", choices=["shared_memory", "sync_gpu"])
    parser.add_argument("--python-bin", default="/home/xhk/miniconda3/envs/pytorch/bin/python")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--nccl-port-base", type=int, default=29670)
    parser.add_argument("--decode-mode", default="continuous", choices=["continuous"])
    parser.add_argument("--max-active-decode-requests", type=int, default=4)
    parser.add_argument("--max-pending-sends", type=int, default=4)
    parser.add_argument("--max-pending-recvs", type=int, default=4)
    parser.add_argument(
        "--worker-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read prefill/decode worker_state files before routing each request.",
    )
    parser.add_argument(
        "--worker-feedback-scale-s",
        type=float,
        default=1.0,
        help="Scale applied when converting worker_state counters into scheduler load seconds.",
    )
    return parser.parse_args()


def main():
    """真实多 PD pair benchmark 主入口：启动 worker、提交请求、汇总 metrics、关闭进程。"""
    args = parse_args()
    mode = f"agent_pd_multi_pair_{args.kv_transfer_backend}_{args.scheduler}"
    profile_dir = args.profile or "all"
    result_dir = Path(DEFAULT_RESULT_DIR) / mode / profile_dir
    args.work_dir = args.work_dir or str(result_dir / "work")
    args.output = args.output or str(result_dir / "synthetic_metrics.jsonl")
    args.summary_output = args.summary_output or str(result_dir / "synthetic_summary.json")

    requests = select_requests(
        args.dataset,
        limit=args.limit + args.warmup,
        profile=args.profile,
        max_total_tokens=args.max_total_tokens,
    )
    if not requests:
        raise RuntimeError("no benchmark requests selected")

    root_work_dir = Path(args.work_dir)
    root_work_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_worker_pairs(args, root_work_dir)
    scheduler_runtime = build_scheduler(args, len(pairs))
    procs = []
    ready = []
    try:
        procs, ready = start_worker_pairs(args, pairs)

        warmup_entries = [
            (idx, row, "warmup", None)
            for idx, row in enumerate(requests[: args.warmup])
        ]
        measure_entries = [
            (args.warmup + idx, row, "measure", idx)
            for idx, row in enumerate(requests[args.warmup :])
        ]

        rows = []
        warmup_first_submit_t = None
        warmup_last_done_t = None
        if warmup_entries:
            warmup_paths, warmup_meta, warmup_submit, warmup_first_submit_t = submit_entries(
                args,
                pairs,
                scheduler_runtime,
                warmup_entries,
            )
            warmup_completed = wait_for_decode_done(
                warmup_paths,
                args.request_timeout_s,
                args.poll_interval_s,
            )
            warmup_last_done_t = max(warmup_completed.values())
            rows.extend(
                build_metrics_rows(
                    args,
                    warmup_paths,
                    warmup_meta,
                    warmup_submit,
                    warmup_completed,
                )
            )

        if args.load_mode == "closed_loop":
            (
                measure_paths,
                measure_meta,
                measure_submit,
                measure_completed,
                measure_first_submit_t,
                measure_last_done_t,
            ) = run_closed_loop(args, pairs, scheduler_runtime, measure_entries)
        else:
            measure_paths, measure_meta, measure_submit, measure_first_submit_t = submit_entries(
                args,
                pairs,
                scheduler_runtime,
                measure_entries,
            )
            measure_completed = wait_for_decode_done(
                measure_paths,
                args.request_timeout_s,
                args.poll_interval_s,
            )
            measure_last_done_t = max(measure_completed.values())

        rows.extend(
            build_metrics_rows(
                args,
                measure_paths,
                measure_meta,
                measure_submit,
                measure_completed,
            )
        )
        rows = sorted(rows, key=lambda row: row["request_index"])
        measured_rows = [row for row in rows if row.get("phase") == "measure"]
        summary = summarize(
            measured_rows,
            [
                "prefill_time_s",
                "payload_write_time_s",
                "transfer_time_s",
                "restore_time_s",
                "decode_time_s",
                "decode_compute_time_s",
                "core_e2e_time_s",
                "wall_e2e_time_s",
            ],
        )
        total_generated = sum(row.get("generated_tokens", 0) for row in measured_rows)
        first_submit_candidates = [
            t for t in (warmup_first_submit_t, measure_first_submit_t) if t is not None
        ]
        first_submit_t = min(first_submit_candidates) if first_submit_candidates else None
        last_done_candidates = [
            t for t in (warmup_last_done_t, measure_last_done_t) if t is not None
        ]
        last_done_t = max(last_done_candidates) if last_done_candidates else None
        total_wall_s = (
            last_done_t - first_submit_t
            if first_submit_t is not None and last_done_t is not None
            else 0.0
        )
        measure_wall_s = (
            measure_last_done_t - measure_first_submit_t
            if measure_first_submit_t is not None
            else 0.0
        )
        summary.update(
            {
                "mode": mode,
                "dataset": args.dataset,
                "profile": args.profile,
                "limit": args.limit,
                "warmup": args.warmup,
                "measured_requests": len(measured_rows),
                "kv_cache_quant_mode": args.kv_cache_quant_mode,
                "kv_transfer_backend": args.kv_transfer_backend,
                "scheduler": args.scheduler,
                "load_mode": args.load_mode,
                "concurrency": args.concurrency,
                "prefill_gpus": parse_csv(args.prefill_gpus),
                "decode_gpus": parse_csv(args.decode_gpus),
                "worker_pairs": ready,
                "worker_feedback": args.worker_feedback,
                "worker_feedback_scale_s": args.worker_feedback_scale_s,
                "agent_scheduler_workers": worker_summary(scheduler_runtime["workers"]),
                "pipeline_total_wall_time_s": total_wall_s,
                "pipeline_measure_wall_time_s": measure_wall_s,
                "pipeline_throughput_generated_tok_s": (
                    total_generated / measure_wall_s if measure_wall_s > 0 else 0.0
                ),
                "result_dir": str(result_dir),
            }
        )
        write_jsonl(args.output, rows)
        write_json(args.summary_output, summary)
        print("metrics_written", args.output)
        print("summary_written", args.summary_output)
    finally:
        shutdown_worker_pairs(pairs, procs)


if __name__ == "__main__":
    main()
