from dataclasses import dataclass


@dataclass
class AgentSchedulerConfig:
    # Agent-aware 调度：这些参数只用于估计任务复杂度和服务时间。
    # 第一版保持轻量，后续可以替换为真实 runtime 统计出来的吞吐/排队指标。
    short_input_threshold: int = 256
    long_input_threshold: int = 1024
    short_output_threshold: int = 128
    long_output_threshold: int = 128
    prefill_complexity_weight: float = 0.3
    decode_complexity_weight: float = 1.0
    tool_complexity_weight: float = 64.0
    step_complexity_weight: float = 32.0
    prefill_tokens_per_s: float = 4000.0
    decode_tokens_per_s: float = 14.0
    base_request_overhead_s: float = 0.02
    tool_call_time_s: float = 0.30
    step_overhead_s: float = 0.05
    queue_weight: float = 1.0
    finish_time_weight: float = 1.0
    complexity_capacity: float = 10000.0
    affinity_max_extra_wait_s: float = 2.0


@dataclass
class AgentRequestEstimate:
    request_index: int
    request_id: str
    session_id: str
    profile: str | None
    task_type: str
    input_tokens: int
    output_tokens: int
    estimated_tool_calls: int
    estimated_steps: int
    complexity_score: float


@dataclass
class WorkerState:
    worker_id: int
    available_at_s: float = 0.0
    busy_time_s: float = 0.0
    num_requests: int = 0
    generated_tokens: int = 0
    feedback_load_s: float = 0.0
    feedback_state: dict | None = None


def worker_queue_wait_s(worker: WorkerState, arrival_t_s: float) -> float:
    """合并虚拟排队时间和真实 worker feedback，得到调度时使用的当前等待代价。"""
    return max(0.0, worker.available_at_s - arrival_t_s) + worker.feedback_load_s


def resolve_session_id(row, request_id: str) -> str:
    """从输入样本里解析 Agent 会话标识，用于后续 session affinity 路由。"""
    # Agent-aware PD 分离：session_id 用来表达“这几条请求属于同一个 Agent 会话/程序”。
    # 后续做 Prefix Cache 或 KV locality 时，同一 session 尽量回到同一个 PD pair，
    # 这样更容易复用前缀/KV 命中；普通 ShareGPT 数据没有这些字段时退化为 request_id。
    return str(
        row.get("program_id")
        or row.get("session_id")
        or row.get("conversation_id")
        or row.get("source_id")
        or request_id
    )


def parse_initial_backlogs(initial_backlog_s: str, num_workers: int) -> list[float]:
    """解析命令行传入的初始虚拟 backlog，给每个 worker/pair 一个起始排队时间。"""
    if not initial_backlog_s:
        return [0.0] * num_workers
    values = [
        float(item.strip())
        for item in initial_backlog_s.split(",")
        if item.strip()
    ]
    if len(values) != num_workers:
        raise ValueError(
            f"--initial-backlog-s expects {num_workers} values, got {len(values)}"
        )
    return values


def init_worker_states(num_workers: int, initial_backlogs: list[float]) -> list[WorkerState]:
    """创建调度器内部的 worker 状态数组，表示每个虚拟/真实 PD pair 的负载。"""
    return [
        WorkerState(
            worker_id=i,
            available_at_s=initial_backlogs[i],
            busy_time_s=initial_backlogs[i],
        )
        for i in range(num_workers)
    ]


def infer_task_type(prompt: str, input_tokens: int, output_tokens: int, config: AgentSchedulerConfig) -> str:
    """根据 prompt 关键词和输入/输出长度，粗略推断 Agent step 的任务类型。"""
    prompt_lower = str(prompt).lower()
    tool_keywords = (
        "search",
        "calculate",
        "weather",
        "python",
        "code",
        "api",
        "database",
        "file",
    )
    if any(keyword in prompt_lower for keyword in tool_keywords):
        return "tool_like"
    if input_tokens >= config.long_input_threshold:
        return "long_context"
    if output_tokens >= config.long_output_threshold:
        return "decode_heavy"
    if (
        input_tokens <= config.short_input_threshold
        and output_tokens <= config.short_output_threshold
    ):
        return "simple_qa"
    return "mixed_agent"


def build_agent_request_estimate(
    row,
    request_index: int,
    output_tokens: int,
    config: AgentSchedulerConfig,
) -> AgentRequestEstimate:
    """把一条 benchmark/Agent step 输入转换成调度器使用的复杂度估计对象。"""
    input_tokens = int(row.get("input_tokens", 0))
    request_id = row.get("id", f"agent-{request_index:06d}")
    task_type = row.get("task_type") or infer_task_type(
        row.get("prompt", ""),
        input_tokens,
        output_tokens,
        config,
    )
    estimated_tool_calls = 1 if task_type == "tool_like" else 0
    estimated_steps = 1
    if input_tokens >= config.long_input_threshold:
        estimated_steps += 1
    if output_tokens >= config.long_output_threshold:
        estimated_steps += 1
    estimated_steps += estimated_tool_calls

    # Agent-aware 调度：复杂度不是单纯 token 数，而是把长输入、长输出、
    # 工具调用和多步骤任务都折算到同一个 score，方便和 worker 负载一起比较。
    complexity_score = (
        input_tokens * config.prefill_complexity_weight
        + output_tokens * config.decode_complexity_weight
        + estimated_tool_calls * config.tool_complexity_weight
        + max(0, estimated_steps - 1) * config.step_complexity_weight
    )

    return AgentRequestEstimate(
        request_index=request_index,
        request_id=request_id,
        session_id=resolve_session_id(row, request_id),
        profile=row.get("profile"),
        task_type=task_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_tool_calls=estimated_tool_calls,
        estimated_steps=estimated_steps,
        complexity_score=complexity_score,
    )


def load_aware_score(
    worker: WorkerState,
    req: AgentRequestEstimate,
    arrival_t_s: float,
    service_t_s: float,
    config: AgentSchedulerConfig,
) -> float:
    """计算一个请求放到某个 worker 上的负载分数，分数越低越适合路由过去。"""
    queue_wait = worker_queue_wait_s(worker, arrival_t_s)
    projected_finish = max(worker.available_at_s, arrival_t_s) + service_t_s

    # Agent-aware 调度：选择“预计完成最早且当前积压最低”的 worker。
    # 后续接真实多实例时，这里的 available_at_s 可以替换为真实 queue/backlog 指标。
    return (
        queue_wait * config.queue_weight
        + projected_finish * config.finish_time_weight
        + req.complexity_score / config.complexity_capacity
    )


def select_worker_round_robin(
    workers: list[WorkerState],
    request_index: int,
) -> WorkerState:
    """轮询策略：完全按请求序号取模选择 worker，不感知负载和任务复杂度。"""
    return workers[request_index % len(workers)]


def select_worker_load_aware(
    workers: list[WorkerState],
    req: AgentRequestEstimate,
    arrival_t_s: float,
    service_t_s: float,
    config: AgentSchedulerConfig,
) -> WorkerState:
    """负载感知策略：遍历所有 worker，选择预计排队/完成代价最低的一个。"""
    best_worker = None
    best_score = None
    for worker in workers:
        score = load_aware_score(worker, req, arrival_t_s, service_t_s, config)
        if best_score is None or score < best_score:
            best_score = score
            best_worker = worker
    return best_worker


def select_worker_affinity_load_aware(
    workers: list[WorkerState],
    req: AgentRequestEstimate,
    arrival_t_s: float,
    service_t_s: float,
    config: AgentSchedulerConfig,
    session_to_worker: dict[str, int],
) -> tuple[WorkerState, bool, int | None]:
    """会话亲和负载感知策略：同 session 优先回原 worker，过载时允许迁移。"""
    # Agent-aware 异步/多 PD 分离：会话亲和策略。
    # 目标不是盲目固定 worker，而是在“保留 KV/Prefix locality”和“避免排队过长”之间取舍：
    # 1. 如果这个 session 之前已经路由到某个 PD pair，就把它作为 preferred worker；
    # 2. 同时计算全局 load-aware 的最佳 worker；
    # 3. preferred worker 额外等待时间不超过阈值时，继续留在原 pair；
    # 4. 否则迁移到更空闲的 pair，保证复杂 Agent 任务不会拖垮单个 decode 队列。
    preferred_worker_id = session_to_worker.get(req.session_id)
    best_worker = select_worker_load_aware(
        workers,
        req,
        arrival_t_s,
        service_t_s,
        config,
    )
    if preferred_worker_id is None:
        session_to_worker[req.session_id] = best_worker.worker_id
        return best_worker, False, None

    preferred_worker = workers[preferred_worker_id]
    preferred_wait = worker_queue_wait_s(preferred_worker, arrival_t_s)
    best_wait = worker_queue_wait_s(best_worker, arrival_t_s)
    extra_wait = preferred_wait - best_wait
    if extra_wait <= config.affinity_max_extra_wait_s:
        return preferred_worker, True, preferred_worker_id

    session_to_worker[req.session_id] = best_worker.worker_id
    return best_worker, False, preferred_worker_id


def estimate_service_time_s(req: AgentRequestEstimate, config: AgentSchedulerConfig) -> float:
    """用轻量模型估计请求服务时间，作为 driver 侧路由的虚拟排队依据。"""
    # Agent-aware 调度：这里使用轻量服务时间模型做路由估计。
    # 真实 PD 执行仍由 benchmark driver 完成；这个估计值只影响“发给哪个 worker slot”。
    prefill_time_s = req.input_tokens / config.prefill_tokens_per_s
    decode_time_s = req.output_tokens / config.decode_tokens_per_s
    tool_time_s = req.estimated_tool_calls * config.tool_call_time_s
    step_overhead_s = max(0, req.estimated_steps - 1) * config.step_overhead_s
    return (
        config.base_request_overhead_s
        + prefill_time_s
        + decode_time_s
        + tool_time_s
        + step_overhead_s
    )


def select_worker(
    workers: list[WorkerState],
    scheduler: str,
    request_index: int,
    req: AgentRequestEstimate,
    arrival_t_s: float,
    service_t_s: float,
    config: AgentSchedulerConfig,
    session_to_worker: dict[str, int] | None = None,
) -> WorkerState:
    """统一的 worker 选择入口，根据 scheduler 名称分发到具体调度策略。"""
    if scheduler == "round_robin":
        return select_worker_round_robin(workers, request_index)
    if scheduler == "load_aware":
        return select_worker_load_aware(
            workers,
            req,
            arrival_t_s,
            service_t_s,
            config,
        )
    if scheduler == "affinity_load_aware":
        worker, _, _ = select_worker_affinity_load_aware(
            workers,
            req,
            arrival_t_s,
            service_t_s,
            config,
            session_to_worker or {},
        )
        return worker
    raise ValueError(f"unknown scheduler: {scheduler}")


def assign_request_to_worker(
    worker: WorkerState,
    arrival_t_s: float,
    service_t_s: float,
    output_tokens: int,
) -> dict:
    """把请求记账到选中的 worker 上，并返回这次调度产生的时间线元数据。"""
    start_t_s = max(arrival_t_s, worker.available_at_s)
    finish_t_s = start_t_s + service_t_s
    queue_wait_t_s = start_t_s - arrival_t_s

    worker.available_at_s = finish_t_s
    worker.busy_time_s += service_t_s
    worker.num_requests += 1
    worker.generated_tokens += output_tokens

    return {
        "worker_id": worker.worker_id,
        "worker_feedback_load_s": worker.feedback_load_s,
        "worker_feedback_state": worker.feedback_state,
        "arrival_time_s": arrival_t_s,
        "start_time_s": start_t_s,
        "finish_time_s": finish_t_s,
        "queue_wait_time_s": queue_wait_t_s,
        "estimated_service_time_s": service_t_s,
        "estimated_e2e_time_s": finish_t_s - arrival_t_s,
    }


def schedule_request(
    workers: list[WorkerState],
    scheduler: str,
    request_index: int,
    req: AgentRequestEstimate,
    arrival_t_s: float,
    config: AgentSchedulerConfig,
    session_to_worker: dict[str, int] | None = None,
) -> dict:
    """调度器主入口：估计服务时间、选择 worker、更新 worker 状态并返回调度结果。"""
    service_t_s = estimate_service_time_s(req, config)
    affinity_hit = False
    preferred_worker_id = None
    if scheduler == "affinity_load_aware":
        if session_to_worker is None:
            session_to_worker = {}
        worker, affinity_hit, preferred_worker_id = select_worker_affinity_load_aware(
            workers,
            req,
            arrival_t_s,
            service_t_s,
            config,
            session_to_worker,
        )
    else:
        worker = select_worker(
            workers,
            scheduler,
            request_index,
            req,
            arrival_t_s,
            service_t_s,
            config,
        )

    schedule_meta = assign_request_to_worker(
        worker,
        arrival_t_s,
        service_t_s,
        req.output_tokens,
    )
    schedule_meta.update(
        {
            "session_id": req.session_id,
            "affinity_hit": affinity_hit,
            "preferred_worker_id": preferred_worker_id,
        }
    )
    return schedule_meta


def worker_summary(workers: list[WorkerState]) -> dict:
    """汇总所有 worker 的虚拟负载，用于 benchmark summary 和调度效果对比。"""
    busy_times = [worker.busy_time_s for worker in workers]
    avg_busy = sum(busy_times) / len(busy_times) if busy_times else 0.0
    return {
        "worker_busy_time_s": busy_times,
        "worker_num_requests": [worker.num_requests for worker in workers],
        "worker_generated_tokens": [worker.generated_tokens for worker in workers],
        "worker_feedback_load_s": [worker.feedback_load_s for worker in workers],
        "worker_load_imbalance": max(busy_times) / avg_busy if avg_busy > 0 else 0.0,
    }
