import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
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
    wait_for_file,
)
from pd_self.multiprocess.agent_scheduler import (  # noqa: E402
    AgentSchedulerConfig,
    build_agent_request_estimate,
    resolve_session_id,
)


def parse_csv(value: str) -> list[str]:
    """解析逗号分隔的 GPU 列表，例如 '0,2' -> ['0', '2']。"""
    return [item.strip() for item in value.split(",") if item.strip()]


def pool_visible_devices(args) -> str:
    """
    ########################### NCCL 池化 PD GPU 可见性 ###########################
    pool 模式下所有 worker 需要看到同一份 GPU 列表。

    原先每个 worker 只暴露自己的单卡，例如 rank3 进程只看到 cuda:0。
    这会导致 PyTorch/NCCL 收尾时把 global rank=3 误当成本地 cuda:3，
    从而触发 invalid device ordinal。

    这里统一设置：
      CUDA_VISIBLE_DEVICES = PREFILL_GPUS + DECODE_GPUS
    再通过 --local-cuda-device 告诉每个 worker 实际使用哪个本地 index。
    """
    return ",".join([*parse_csv(args.prefill_gpus), *parse_csv(args.decode_gpus)])


def worker_state_path(worker: dict) -> Path:
    """返回 worker 写出的 runtime feedback 状态文件路径。"""
    if worker["role"] == "prefill":
        return worker["work_dir"] / "prefill_worker_state.json"
    return worker["work_dir"] / "decode_worker_state.json"


def read_worker_state(worker: dict) -> dict:
    """读取某个 pool worker 的真实状态；文件还没生成时返回空字典。"""
    try:
        return read_json(worker_state_path(worker))
    except FileNotFoundError:
        return {}


def feedback_load_s(args, worker: dict, state: dict) -> float:
    """
    ########################### NCCL 池化 PD 调度反馈 ###########################
    把 worker_state 中的轻量计数转换成调度器可用的负载秒数。

    第一版只做足够简单的启发式：
    - prefill 看 request_queue_depth / pending_sends / busy；
    - decode 看 active_decode_requests / pending_recvs / busy。
    这个分数只影响 coordinator 选 worker，不改变 nano-vLLM 内部 scheduler。
    """
    if not args.worker_feedback:
        return 0.0

    if worker["role"] == "prefill":
        units = (
            float(state.get("request_queue_depth", 0))
            + float(state.get("pending_sends", 0))
        )
    else:
        units = (
            float(state.get("active_decode_requests", 0))
            + float(state.get("pending_recvs", 0))
        )

    if state.get("busy"):
        units += 0.5
    return units * args.worker_feedback_scale_s


def estimate_prefill_time_s(row: dict, config: AgentSchedulerConfig) -> float:
    """根据输入 token 数估算 prefill 侧占用时间，用于 pool 调度。"""
    input_tokens = int(row.get("input_tokens", 0))
    return config.base_request_overhead_s + input_tokens / config.prefill_tokens_per_s


def estimate_decode_time_s(max_tokens: int, config: AgentSchedulerConfig) -> float:
    """根据输出 token 上限估算 decode 侧占用时间，用于 pool 调度。"""
    return config.base_request_overhead_s + max_tokens / config.decode_tokens_per_s


def worker_queue_wait_s(worker: dict) -> float:
    """读取 driver 侧维护的虚拟排队时间。"""
    return max(0.0, worker.get("available_at_s", 0.0))


def select_least_loaded_worker(
    workers: list[dict],
    service_time_s: float,
    request_index: int,
    scheduler: str,
) -> dict:
    """
    从同类 pool 中选择 worker。

    round_robin：
      完全按请求序号轮询，作为 baseline。

    load_aware / affinity_load_aware：
      选择“虚拟排队 + 真实 feedback + 本次服务时间”最低的 worker。
    """
    if scheduler == "round_robin":
        return workers[request_index % len(workers)]

    best_worker = None
    best_score = None
    for worker in workers:
        score = (
            worker_queue_wait_s(worker)
            + worker.get("feedback_load_s", 0.0)
            + service_time_s
        )
        if best_score is None or score < best_score:
            best_worker = worker
            best_score = score
    return best_worker


def assign_virtual_work(worker: dict, service_time_s: float, max_tokens: int) -> dict:
    """
    更新 driver 侧虚拟 worker 状态。

    这不是 nano-vLLM 内部真实调度，只是 coordinator 做路由时的轻量 backlog 模型。
    真实负载仍然通过 worker_state.json 反馈回来。
    """
    start_t_s = worker_queue_wait_s(worker)
    finish_t_s = start_t_s + service_time_s
    worker["available_at_s"] = finish_t_s
    worker["busy_time_s"] = worker.get("busy_time_s", 0.0) + service_time_s
    worker["num_requests"] = worker.get("num_requests", 0) + 1
    worker["generated_tokens"] = worker.get("generated_tokens", 0) + max_tokens
    return {
        "slot_queue_wait_time_s": start_t_s,
        "slot_finish_time_s": finish_t_s,
    }


def build_pool_workers(args, root_work_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    ########################### NCCL 池化 PD worker 拓扑 ###########################
    构造独立的 prefill pool 和 decode pool。

    rank 规则：
      prefill rank: 0 .. num_prefill-1
      decode rank : num_prefill .. num_prefill+num_decode-1

    例子：
      PREFILL_GPUS=0,2 DECODE_GPUS=1,3
      rank 0: P0 on GPU0
      rank 1: P1 on GPU2
      rank 2: D0 on GPU1
      rank 3: D1 on GPU3
    """
    prefill_gpus = parse_csv(args.prefill_gpus)
    decode_gpus = parse_csv(args.decode_gpus)
    if not prefill_gpus or not decode_gpus:
        raise ValueError("prefill_gpus and decode_gpus must be non-empty")

    world_size = len(prefill_gpus) + len(decode_gpus)

    prefill_workers = []
    for idx, gpu in enumerate(prefill_gpus):
        work_dir = root_work_dir / f"prefill_{idx}"
        prefill_workers.append(
            {
                "role": "prefill",
                "prefill_worker_id": idx,
                "worker_id": idx,
                "gpu": gpu,
                "global_rank": idx,
                "local_cuda_device": idx,
                "world_size": world_size,
                "work_dir": work_dir,
                "log": work_dir / "persistent_prefill.log",
                "available_at_s": 0.0,
                "busy_time_s": 0.0,
                "num_requests": 0,
                "generated_tokens": 0,
                "feedback_load_s": 0.0,
                "feedback_state": {},
            }
        )

    decode_workers = []
    for idx, gpu in enumerate(decode_gpus):
        work_dir = root_work_dir / f"decode_{idx}"
        decode_workers.append(
            {
                "role": "decode",
                "decode_worker_id": idx,
                "worker_id": idx,
                "gpu": gpu,
                "global_rank": len(prefill_gpus) + idx,
                "local_cuda_device": len(prefill_gpus) + idx,
                "world_size": world_size,
                "work_dir": work_dir,
                "log": work_dir / "persistent_decode.log",
                "available_at_s": 0.0,
                "busy_time_s": 0.0,
                "num_requests": 0,
                "generated_tokens": 0,
                "feedback_load_s": 0.0,
                "feedback_state": {},
            }
        )

    return prefill_workers, decode_workers


def refresh_worker_feedback(args, workers: list[dict]) -> list[dict]:
    """调度前刷新 pool worker 的真实状态，用于 load-aware 路由。"""
    rows = []
    for worker in workers:
        state = read_worker_state(worker)
        worker["feedback_load_s"] = feedback_load_s(args, worker, state)
        worker["feedback_state"] = {
            "role": worker["role"],
            "worker_id": worker["worker_id"],
            "global_rank": worker["global_rank"],
            "gpu": worker["gpu"],
            "state": state,
            "feedback_load_s": worker["feedback_load_s"],
        }
        rows.append(worker["feedback_state"])
    return rows


def start_prefill_pool_worker(args, worker: dict) -> subprocess.Popen:
    """
    ########################### NCCL 池化 PD prefill worker 启动 ###########################
    每个 worker 都加入同一个 NCCL world，并且看到同一份 pool GPU 列表。
    local_cuda_device 决定当前 worker 实际使用哪张本地可见卡。
    """
    worker["work_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = pool_visible_devices(args)

    cmd = [
        args.python_bin,
        "pd_self/multiprocess/persistent_prefill_worker.py",
        "--pool-mode",
        "--work-dir",
        str(worker["work_dir"]),
        "--kv-cache-quant-mode",
        args.kv_cache_quant_mode,
        "--kv-transfer-backend",
        "sync_gpu",
        "--nccl-port",
        str(args.nccl_port),
        "--global-rank",
        str(worker["global_rank"]),
        "--world-size",
        str(worker["world_size"]),
        "--prefill-worker-id",
        str(worker["prefill_worker_id"]),
        "--local-cuda-device",
        str(worker["local_cuda_device"]),
        "--poll-interval-s",
        str(args.poll_interval_s),
        "--max-pending-sends",
        str(args.max_pending_sends),
        "--no-wait-decode-done",
    ]

    with worker["log"].open("w", encoding="utf-8") as f:
        return subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def start_decode_pool_worker(args, worker: dict) -> subprocess.Popen:
    """
    ########################### NCCL 池化 PD decode worker 启动 ###########################
    decode worker 只扫描自己的 work_dir；任意 prefill 选中它时，会把 payload_ready 写到这里。
    """
    worker["work_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = pool_visible_devices(args)

    cmd = [
        args.python_bin,
        "pd_self/multiprocess/persistent_decode_worker.py",
        "--pool-mode",
        "--work-dir",
        str(worker["work_dir"]),
        "--kv-cache-quant-mode",
        args.kv_cache_quant_mode,
        "--kv-transfer-backend",
        "sync_gpu",
        "--decode-mode",
        "continuous",
        "--nccl-port",
        str(args.nccl_port),
        "--global-rank",
        str(worker["global_rank"]),
        "--world-size",
        str(worker["world_size"]),
        "--decode-worker-id",
        str(worker["decode_worker_id"]),
        "--local-cuda-device",
        str(worker["local_cuda_device"]),
        "--poll-interval-s",
        str(args.poll_interval_s),
        "--max-active-decode-requests",
        str(args.max_active_decode_requests),
        "--max-pending-recvs",
        str(args.max_pending_recvs),
    ]

    with worker["log"].open("w", encoding="utf-8") as f:
        return subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def start_pool_workers(
    args,
    prefill_workers: list[dict],
    decode_workers: list[dict],
) -> tuple[list[subprocess.Popen], list[dict]]:
    """启动所有 P/D pool worker，并等待 ready 文件出现。"""
    procs = []
    ready = []

    for worker in [*prefill_workers, *decode_workers]:
        clean_work_dir(worker["work_dir"])

    # NCCL init_process_group 会等全体 rank 加入。
    # 所以这里先把所有进程都 Popen 出去，再等待 ready 文件。
    for worker in prefill_workers:
        procs.append(start_prefill_pool_worker(args, worker))
    for worker in decode_workers:
        procs.append(start_decode_pool_worker(args, worker))

    for worker in prefill_workers:
        ready_file = worker["work_dir"] / "prefill_worker.ready.json"
        wait_for_file(ready_file, args.startup_timeout_s)
        ready.append(
            {
                "role": "prefill",
                "prefill_worker_id": worker["prefill_worker_id"],
                "global_rank": worker["global_rank"],
                "local_cuda_device": worker["local_cuda_device"],
                "gpu": worker["gpu"],
                "ready": read_json(ready_file),
            }
        )

    for worker in decode_workers:
        ready_file = worker["work_dir"] / "decode_worker.ready.json"
        wait_for_file(ready_file, args.startup_timeout_s)
        ready.append(
            {
                "role": "decode",
                "decode_worker_id": worker["decode_worker_id"],
                "global_rank": worker["global_rank"],
                "local_cuda_device": worker["local_cuda_device"],
                "gpu": worker["gpu"],
                "ready": read_json(ready_file),
            }
        )

    return procs, ready


def shutdown_pool_workers(workers: list[dict], procs: list[subprocess.Popen]) -> None:
    """向所有 worker 写 shutdown 文件并等待退出，超时后兜底 kill 子进程。"""
    for worker in workers:
        try:
            (worker["work_dir"] / "shutdown").write_text("shutdown\n", encoding="utf-8")
        except FileNotFoundError:
            pass

    for proc in procs:
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
            proc.wait(timeout=10)


def raise_if_worker_error(paths_by_base: dict[str, dict]) -> None:
    """检查已提交请求是否产生 prefill/decode error 文件，有则立即抛出。"""
    for paths in paths_by_base.values():
        for key in ("prefill_error", "decode_error"):
            error_path = paths[key]
            if error_path.exists():
                raise RuntimeError(
                    f"worker error: {error_path}\n{error_path.read_text(encoding='utf-8')}"
                )


def wait_for_decode_done(
    paths_by_base: dict[str, dict],
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, float]:
    """batch 模式等待所有请求完成 decode，并记录每条请求完成时间。"""
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


def schedule_pool_request(
    args,
    config: AgentSchedulerConfig,
    prefill_workers: list[dict],
    decode_workers: list[dict],
    session_to_decode_worker: dict[str, int],
    row: dict,
    idx: int,
    request_id: str,
    max_tokens: int,
) -> tuple[dict, dict, dict]:
    """
    ########################### NCCL 池化 PD 两阶段调度 ###########################
    池化版把 P 和 D 分开选：
      1. prefill 侧主要看 input_tokens 和 P worker feedback；
      2. decode 侧主要看 output_tokens、D worker feedback 和 session affinity。

    这一步是 coordinator 逻辑，不修改 nano-vLLM 内部 scheduler。
    """
    estimate = build_agent_request_estimate(row, idx, max_tokens, config)
    prefill_feedback = refresh_worker_feedback(args, prefill_workers)
    decode_feedback = refresh_worker_feedback(args, decode_workers)

    prefill_service_s = estimate_prefill_time_s(row, config)
    decode_service_s = estimate_decode_time_s(max_tokens, config)

    prefill = select_least_loaded_worker(
        prefill_workers,
        prefill_service_s,
        idx,
        args.scheduler,
    )

    affinity_hit = False
    preferred_decode_worker_id = session_to_decode_worker.get(estimate.session_id)
    if args.scheduler == "affinity_load_aware" and preferred_decode_worker_id is not None:
        preferred = decode_workers[preferred_decode_worker_id]
        best = select_least_loaded_worker(
            decode_workers,
            decode_service_s,
            idx,
            "load_aware",
        )
        extra_wait_s = (
            worker_queue_wait_s(preferred)
            + preferred.get("feedback_load_s", 0.0)
            - worker_queue_wait_s(best)
            - best.get("feedback_load_s", 0.0)
        )
        if extra_wait_s <= config.affinity_max_extra_wait_s:
            decode = preferred
            affinity_hit = True
        else:
            decode = best
            session_to_decode_worker[estimate.session_id] = decode["decode_worker_id"]
    else:
        decode = select_least_loaded_worker(
            decode_workers,
            decode_service_s,
            idx,
            args.scheduler,
        )
        if args.scheduler == "affinity_load_aware":
            session_to_decode_worker[estimate.session_id] = decode["decode_worker_id"]

    prefill_slot_meta = assign_virtual_work(prefill, prefill_service_s, max_tokens)
    decode_slot_meta = assign_virtual_work(decode, decode_service_s, max_tokens)

    route_meta = {
        "scheduler": args.scheduler,
        "program_id": row.get("program_id", resolve_session_id(row, request_id)),
        "session_id": estimate.session_id,
        "step_id": row.get("step_id"),
        "num_steps": row.get("num_steps"),
        "task_kind": row.get("task_kind"),
        "task_type": estimate.task_type,
        "estimated_tool_calls": estimate.estimated_tool_calls,
        "estimated_steps": estimate.estimated_steps,
        "complexity_score": estimate.complexity_score,
        "affinity_hit": affinity_hit,
        "preferred_decode_worker_id": preferred_decode_worker_id,
        "prefill_worker_id": prefill["prefill_worker_id"],
        "decode_worker_id": decode["decode_worker_id"],
        "src_rank": prefill["global_rank"],
        "dst_rank": decode["global_rank"],
        "prefill_gpu": prefill["gpu"],
        "decode_gpu": decode["gpu"],
        "prefill_feedback_load_s": prefill["feedback_load_s"],
        "decode_feedback_load_s": decode["feedback_load_s"],
        "prefill_feedback_state": prefill["feedback_state"],
        "decode_feedback_state": decode["feedback_state"],
        "all_prefill_feedback_states": prefill_feedback,
        "all_decode_feedback_states": decode_feedback,
        "estimated_prefill_service_time_s": prefill_service_s,
        "estimated_decode_service_time_s": decode_service_s,
        "estimated_service_time_s": prefill_service_s + decode_service_s,
        "prefill_slot_queue_wait_time_s": prefill_slot_meta["slot_queue_wait_time_s"],
        "decode_slot_queue_wait_time_s": decode_slot_meta["slot_queue_wait_time_s"],
        "estimated_slot_e2e_time_s": max(
            prefill_slot_meta["slot_finish_time_s"],
            decode_slot_meta["slot_finish_time_s"],
        ),
    }
    return prefill, decode, route_meta


def submit_one_pool(
    args,
    config: AgentSchedulerConfig,
    prefill_workers: list[dict],
    decode_workers: list[dict],
    session_to_decode_worker: dict[str, int],
    entry,
) -> tuple[str, dict, dict, float]:
    """提交单条请求：先选 P/D，再把 request.json 写到选中的 prefill work_dir。"""
    idx, row, phase, measure_index = entry
    request_id = row.get("id", f"agent-pool-{idx:06d}")
    max_tokens = max(2, cap_max_tokens(row, args.max_output_tokens_cap))
    prefill, decode, route_meta = schedule_pool_request(
        args,
        config,
        prefill_workers,
        decode_workers,
        session_to_decode_worker,
        row,
        idx,
        request_id,
        max_tokens,
    )

    base = request_base(idx, request_id)
    prefill_paths = build_paths(prefill["work_dir"], base)
    decode_paths = build_paths(decode["work_dir"], base)

    request = {
        **row,
        "id": request_id,
        "max_tokens": max_tokens,
        "pd_pool": {
            # ########################### NCCL 池化 PD 请求路由元数据 ###########################
            # prefill worker 读取这些字段后，会把 KV 发给指定 decode rank，
            # 并把 payload_ready / recv_ready / done 等文件写到 decode_work_dir。
            "prefill_worker_id": prefill["prefill_worker_id"],
            "decode_worker_id": decode["decode_worker_id"],
            "src_rank": prefill["global_rank"],
            "dst_rank": decode["global_rank"],
            "decode_work_dir": str(decode["work_dir"]),
        },
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
        "request_path": str(prefill_paths["request"]),
        "decode_done_path": str(decode_paths["decode_done"]),
        **route_meta,
    }

    atomic_write_json(prefill_paths["request"], request)
    return base, decode_paths, metadata, now


def submit_entries(
    args,
    config: AgentSchedulerConfig,
    prefill_workers: list[dict],
    decode_workers: list[dict],
    session_to_decode_worker: dict[str, int],
    entries,
) -> tuple[dict, dict, dict, float | None]:
    """batch 提交一批请求，不等待单条完成。"""
    paths_by_base = {}
    metadata_by_base = {}
    submit_times = {}
    first_submit_t = None

    for entry in entries:
        base, paths, metadata, now = submit_one_pool(
            args,
            config,
            prefill_workers,
            decode_workers,
            session_to_decode_worker,
            entry,
        )
        paths_by_base[base] = paths
        metadata_by_base[base] = metadata
        submit_times[base] = now
        first_submit_t = now if first_submit_t is None else first_submit_t

    return paths_by_base, metadata_by_base, submit_times, first_submit_t


def run_closed_loop(
    args,
    config: AgentSchedulerConfig,
    prefill_workers: list[dict],
    decode_workers: list[dict],
    session_to_decode_worker: dict[str, int],
    entries,
) -> tuple[dict, dict, dict, dict, float | None, float | None]:
    """闭环压测：保持固定并发，某条请求完成后立刻补交下一条。"""
    paths_by_base = {}
    metadata_by_base = {}
    submit_times = {}
    completed_at = {}
    first_submit_t = None
    last_done_t = None
    next_idx = 0
    last_activity_t = time.time()
    max_inflight = max(1, args.concurrency)

    def submit_until_full():
        """内部 helper：把 inflight 请求数补到 concurrency。"""
        nonlocal first_submit_t, next_idx, last_activity_t
        while next_idx < len(entries):
            inflight = len(paths_by_base) - len(completed_at)
            if inflight >= max_inflight:
                break
            base, paths, metadata, now = submit_one_pool(
                args,
                config,
                prefill_workers,
                decode_workers,
                session_to_decode_worker,
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


def pool_worker_summary(workers: list[dict]) -> dict:
    """汇总 pool worker 的虚拟负载，写入 summary 方便看是否负载均衡。"""
    busy_times = [worker.get("busy_time_s", 0.0) for worker in workers]
    avg_busy = sum(busy_times) / len(busy_times) if busy_times else 0.0
    return {
        "worker_ids": [worker["worker_id"] for worker in workers],
        "global_ranks": [worker["global_rank"] for worker in workers],
        "gpus": [worker["gpu"] for worker in workers],
        "worker_busy_time_s": busy_times,
        "worker_num_requests": [worker.get("num_requests", 0) for worker in workers],
        "worker_feedback_load_s": [worker.get("feedback_load_s", 0.0) for worker in workers],
        "worker_load_imbalance": max(busy_times) / avg_busy if avg_busy > 0 else 0.0,
    }


def parse_args():
    """解析 NCCL 池化 PD benchmark 参数。"""
    parser = argparse.ArgumentParser(description="Agent-aware pooled PD NCCL benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=16)
    parser.add_argument("--prefill-gpus", default="0,2")
    parser.add_argument("--decode-gpus", default="1,3")
    parser.add_argument(
        "--scheduler",
        default="load_aware",
        choices=["round_robin", "load_aware", "affinity_load_aware"],
    )
    parser.add_argument("--load-mode", default="closed_loop", choices=["batch", "closed_loop"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument("--python-bin", default="/home/xhk/miniconda3/envs/pytorch/bin/python")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--nccl-port", type=int, default=29810)
    parser.add_argument("--max-active-decode-requests", type=int, default=4)
    parser.add_argument("--max-pending-sends", type=int, default=1)
    parser.add_argument("--max-pending-recvs", type=int, default=1)
    parser.add_argument(
        "--worker-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read worker_state files before routing each request.",
    )
    parser.add_argument(
        "--worker-feedback-scale-s",
        type=float,
        default=1.0,
        help="Scale applied when converting worker_state counters into scheduler load seconds.",
    )
    return parser.parse_args()


def main():
    """NCCL 池化 PD benchmark 主入口：启动 pool、提交请求、汇总 metrics、关闭进程。"""
    args = parse_args()
    # ########################### NCCL 池化 PD benchmark 兼容字段 ###########################
    # pool driver 固定使用 sync_gpu 后端，但后续复用 benchmark_synthetic_pd_pipeline.py
    # 里的 build_metrics_rows()。旧汇总函数会读取 args.kv_transfer_backend，
    # 所以这里显式补上，避免每个 metrics 汇总点都分支判断。
    args.kv_transfer_backend = "sync_gpu"
    mode = f"agent_pd_pool_sync_gpu_{args.scheduler}"
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
    prefill_workers, decode_workers = build_pool_workers(args, root_work_dir)
    config = AgentSchedulerConfig()
    session_to_decode_worker = {}
    procs = []
    ready = []

    try:
        procs, ready = start_pool_workers(args, prefill_workers, decode_workers)

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
                config,
                prefill_workers,
                decode_workers,
                session_to_decode_worker,
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
            ) = run_closed_loop(
                args,
                config,
                prefill_workers,
                decode_workers,
                session_to_decode_worker,
                measure_entries,
            )
        else:
            measure_paths, measure_meta, measure_submit, measure_first_submit_t = submit_entries(
                args,
                config,
                prefill_workers,
                decode_workers,
                session_to_decode_worker,
                measure_entries,
            )
            measure_completed = wait_for_decode_done(
                measure_paths,
                args.request_timeout_s,
                args.poll_interval_s,
            )
            measure_last_done_t = max(measure_completed.values()) if measure_completed else None

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
            if measure_first_submit_t is not None and measure_last_done_t is not None
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
                "kv_transfer_backend": "sync_gpu",
                "scheduler": args.scheduler,
                "load_mode": args.load_mode,
                "concurrency": args.concurrency,
                "prefill_gpus": parse_csv(args.prefill_gpus),
                "decode_gpus": parse_csv(args.decode_gpus),
                "nccl_port": args.nccl_port,
                "world_size": len(prefill_workers) + len(decode_workers),
                "pool_workers": ready,
                "worker_feedback": args.worker_feedback,
                "worker_feedback_scale_s": args.worker_feedback_scale_s,
                "prefill_pool_summary": pool_worker_summary(prefill_workers),
                "decode_pool_summary": pool_worker_summary(decode_workers),
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
        shutdown_pool_workers([*prefill_workers, *decode_workers], procs)


if __name__ == "__main__":
    main()
