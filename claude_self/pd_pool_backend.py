import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pd_self.multiprocess.agent_scheduler import (
    AgentSchedulerConfig,
    build_agent_request_estimate,
    init_worker_states,
    parse_initial_backlogs,
    schedule_request,
)
from pd_self.multiprocess.evaluation.benchmark_synthetic_pd_pipeline import (
    atomic_write_json,
    build_paths,
    read_json,
    request_base,
)


@dataclass
class PDWorkerSlot:
    """
    Agent-aware PD 分离：一个可路由的 worker 槽位。

    prefill worker 和 decode worker 都可以用这个结构描述。
    注意这里不启动 worker，只描述已经启动好的 worker 在哪里。
    """

    worker_id: int
    global_rank: int
    work_dir: Path


class PDPoolBackend:
    """
    Agent 接入真实 PD Pool 的最小后端。

    它负责四件事：
    1. 接收 Agent 传来的 prompt / max_tokens / metadata；
    2. 根据 Agent metadata 选择 prefill worker 和 decode worker；
    3. 写 request.json 到 prefill worker 的 work_dir；
    4. 等 decode worker 写出 decode_done，然后读取 decode_metrics 返回文本。

    重要设计：
    - Agent 不知道 PD 分离细节；
    - PD worker 不知道 LangGraph；
    - 两边只通过 request.json / decode_metrics.json 连接。
    """

    def __init__(
        self,
        prefill_work_dirs: list[str],
        decode_work_dirs: list[str],
        prefill_global_ranks: list[int],
        decode_global_ranks: list[int],
        model_path: str = "/home/xhk/model/Qwen3-0.6B/",
        scheduler: str = "affinity_load_aware",
        initial_backlog_s: str = "",
        request_timeout_s: float = 300.0,
        poll_interval_s: float = 0.05,
    ):
        if len(prefill_work_dirs) != len(prefill_global_ranks):
            raise ValueError("prefill_work_dirs and prefill_global_ranks size mismatch")
        if len(decode_work_dirs) != len(decode_global_ranks):
            raise ValueError("decode_work_dirs and decode_global_ranks size mismatch")

        self.prefill_workers = [
            PDWorkerSlot(
                worker_id=i,
                global_rank=prefill_global_ranks[i],
                work_dir=Path(prefill_work_dirs[i]),
            )
            for i in range(len(prefill_work_dirs))
        ]

        self.decode_workers = [
            PDWorkerSlot(
                worker_id=i,
                global_rank=decode_global_ranks[i],
                work_dir=Path(decode_work_dirs[i]),
            )
            for i in range(len(decode_work_dirs))
        ]

        self.scheduler = scheduler
        self.config = AgentSchedulerConfig()
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s

        # Agent-aware 调度：decode 侧保留 session affinity。
        # 同一个 Agent session 的多步推理尽量回到同一个 decode worker。
        self.session_to_decode_worker = {}

        # driver 侧虚拟负载。
        # 这里先复用 agent_scheduler 的 WorkerState，不侵入 nano-vLLM 内部 scheduler。
        prefill_backlogs = parse_initial_backlogs(
            initial_backlog_s,
            len(self.prefill_workers),
        ) if initial_backlog_s else [0.0] * len(self.prefill_workers)

        self.prefill_states = init_worker_states(
            len(self.prefill_workers),
            prefill_backlogs,
        )
        self.decode_states = init_worker_states(
            len(self.decode_workers),
            [0.0] * len(self.decode_workers),
        )

        # 只加载 tokenizer，用于：
        # 1. 估计 input_tokens；
        # 2. 把 decode_metrics 里的 decode_step_tokens 解码成文本。
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.request_index = 0
        self.route_history = []

    def _estimate_input_tokens(self, prompt: str) -> int:
        """
        Agent-aware 调度：估算 prefill 复杂度。

        第一版直接用 tokenizer 得到真实 prompt token 数。
        后续如果要更轻，可以改成 len(prompt) // 4。
        """
        return len(self.tokenizer.encode(prompt))

    def _select_prefill_worker(self, request_index: int, row: dict, max_tokens: int):
        """
        选择 prefill worker。

        第一版保持简单：
        - 用同一套 schedule_request；
        - 不做 session affinity；
        - 主要根据 input_tokens 估计 prefill 侧压力。
        """
        estimate = build_agent_request_estimate(
            row,
            request_index=request_index,
            output_tokens=max_tokens,
            config=self.config,
        )

        meta = schedule_request(
            workers=self.prefill_states,
            scheduler="load_aware" if self.scheduler != "round_robin" else "round_robin",
            request_index=request_index,
            req=estimate,
            arrival_t_s=0.0,
            config=self.config,
            session_to_worker=None,
        )

        worker_id = meta["worker_id"]
        return self.prefill_workers[worker_id], meta

    def _select_decode_worker(self, request_index: int, row: dict, max_tokens: int):
        """
        选择 decode worker。

        Agent-aware 的重点主要在 decode 侧：
        - 多步 Agent session 尽量保持 affinity；
        - 如果原 decode worker 太忙，则允许迁移；
        - 避免复杂任务一直堵住同一个 D worker。
        """
        estimate = build_agent_request_estimate(
            row,
            request_index=request_index,
            output_tokens=max_tokens,
            config=self.config,
        )

        meta = schedule_request(
            workers=self.decode_states,
            scheduler=self.scheduler,
            request_index=request_index,
            req=estimate,
            arrival_t_s=0.0,
            config=self.config,
            session_to_worker=self.session_to_decode_worker,
        )

        worker_id = meta["worker_id"]
        return self.decode_workers[worker_id], meta

    def _wait_decode_done(self, paths: dict):
        """
        等待 decode worker 完成。

        persistent_decode_worker 的协议是：
        - 写 decode_metrics.json；
        - 再写 decode_done。

        所以看到 decode_done 后，可以安全读取 decode_metrics。
        """
        deadline = time.perf_counter() + self.request_timeout_s

        while time.perf_counter() < deadline:
            if paths["decode_done"].exists():
                return

            if paths["decode_error"].exists():
                raise RuntimeError(
                    paths["decode_error"].read_text(encoding="utf-8")
                )

            time.sleep(self.poll_interval_s)

        raise TimeoutError(f"timed out waiting decode_done: {paths['decode_done']}")

    def _decode_output_text(self, decode_metrics: dict) -> str:
        """
        从 decode_metrics 里恢复输出文本。

        当前 persistent_decode_worker 已经写了 decode_step_tokens。
        这些就是 decode 阶段生成的新 token，可以直接 tokenizer.decode。
        """
        token_ids = decode_metrics.get("decode_step_tokens", [])
        if not token_ids:
            return ""

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip()

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        metadata: dict | None = None,
    ) -> str:
        """
        Agent 调用的唯一入口。

        输入来自 Agent._llm_node：
        - prompt：完整 Agent prompt；
        - max_tokens：本轮最大输出长度；
        - metadata：session_id / step_id / task_type 等调度语义。
        """
        metadata = metadata or {}
        request_index = self.request_index
        self.request_index += 1

        request_id = metadata.get("id", f"agent-pd-{request_index:06d}")

        row = {
            **metadata,
            "id": request_id,
            "prompt": prompt,
            "input_tokens": self._estimate_input_tokens(prompt),
            "max_tokens": max_tokens,
        }

        prefill, prefill_meta = self._select_prefill_worker(
            request_index,
            row,
            max_tokens,
        )
        decode, decode_meta = self._select_decode_worker(
            request_index,
            row,
            max_tokens,
        )

        base = request_base(request_index, request_id)
        prefill_paths = build_paths(prefill.work_dir, base)
        decode_paths = build_paths(decode.work_dir, base)

        request = {
            **row,
            "prompt": prompt,
            "max_tokens": max(2, max_tokens),

            # Agent-aware PD Pool 路由元数据。
            # prefill worker 读取这个字段后，会把 KV 交给指定 decode worker。
            "pd_pool": {
                "prefill_worker_id": prefill.worker_id,
                "decode_worker_id": decode.worker_id,
                "src_rank": prefill.global_rank,
                "dst_rank": decode.global_rank,
                "decode_work_dir": str(decode.work_dir),
            },
        }

        self.route_history.append(
            {
                "request_id": request_id,
                "session_id": row.get("session_id"),
                "step_id": row.get("step_id"),
                "task_type": row.get("task_type"),
                "prefill_worker_id": prefill.worker_id,
                "decode_worker_id": decode.worker_id,
                "src_rank": prefill.global_rank,
                "dst_rank": decode.global_rank,
                "prefill_meta": prefill_meta,
                "decode_meta": decode_meta,
            }
        )

        # 写 request.json 后，persistent_prefill_worker 会自动扫描并处理。
        atomic_write_json(prefill_paths["request"], request)

        # decode_done 写在目标 decode worker 的 work_dir。
        self._wait_decode_done(decode_paths)

        decode_metrics = read_json(decode_paths["decode_metrics"])
        return self._decode_output_text(decode_metrics)
