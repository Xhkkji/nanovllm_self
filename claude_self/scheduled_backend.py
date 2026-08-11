import os
import sys

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

class ScheduledBackend:
    """
    Agent-aware 调度包装器。

    作用：
    1. 对 Agent 每次 LLM 调用执行一次 schedule_request。
    2. 记录这次请求会被路由到哪个 worker/pair。
    3. 第一版仍然调用 inner_backend 本地生成，避免一开始就把 Agent 和 PD 多进程耦合死。

    后续迁移：
    - 当前：schedule -> inner_backend.generate_text()
    - 后面：schedule -> 写 request.json 到 pair_i/work_dir -> 等 decode_done
    """
    
    def __init__(
      self,
      inner_backend,
      scheduler: str = "affinity_load_aware",
      num_workers: int = 2,
      initial_backlog_s: str = "",
    ):
      self.inner_backend = inner_backend
      self.scheduler = scheduler
      self.config = AgentSchedulerConfig()
      initial_backlogs = parse_initial_backlogs(initial_backlog_s, num_workers)
      self.workers = init_worker_states(num_workers, initial_backlogs)

      # Agent-aware 调度：记录 session_id -> worker_id。
      # affinity_load_aware 用它把同一个 Agent session 尽量路由回同一个 worker。
      self.session_to_worker = {}
      # 保存每次模型调用的调度结果，方便 demo/benchmark 打印。
      self.route_history = []
    
    def _estimate_input_tokens(self, prompt: str) -> int:
      """
      第一版简化估计输入长度。

      如果 inner_backend 是 NanovllmBackend，可以直接用 tokenizer/encoder。
      如果后面换成 LangGraph/OpenAI wrapper，也可以替换这里。
      """
      if hasattr(self.inner_backend, "llm"):
        encoded = self.inner_backend.llm.encoder(prompt)
        return len(encoded["input_ids"])
      
      # 兜底：粗略估计，避免没有 tokenizer 时直接挂掉。
      return max(1, len(prompt) // 4)
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        metadata: dict | None = None,
    ) -> str:
      if metadata is None:
        metadata = {}
      
      row = {
        **metadata,
        "prompt": prompt,
        "input_tokens": self._estimate_input_tokens(prompt),
        "max_tokens": max_tokens,
      }
      
      request_index = len(self.route_history)
      estimate = build_agent_request_estimate(
        row,
        request_index=request_index,
        output_tokens=max_tokens,
        config=self.config,
      )
      
      # Agent-aware 调度：这里是真正把 Agent step 喂给调度器的地方。
      schedule_meta = schedule_request(
          self.workers,
          self.scheduler,
          request_index,
          estimate,
          arrival_t_s=0.0,
          config=self.config,
          session_to_worker=self.session_to_worker,
      )

      route_record = {
          "request_id": estimate.request_id,
          "session_id": estimate.session_id,
          "step_id": metadata.get("step_id"),
          "task_type": estimate.task_type,
          "worker_id": schedule_meta["worker_id"],
          "affinity_hit": schedule_meta["affinity_hit"],
          "preferred_worker_id": schedule_meta["preferred_worker_id"],
          "queue_wait_time_s": schedule_meta["queue_wait_time_s"],
          "estimated_service_time_s": schedule_meta["estimated_service_time_s"],
          "estimated_e2e_time_s": schedule_meta["estimated_e2e_time_s"],
      }
      
      self.route_history.append(route_record)
      # 第一版仍然走本地 backend，先验证 Agent + Scheduler 语义链路。
      return self.inner_backend.generate_text(
          prompt,
          max_tokens=max_tokens,
          metadata=metadata,
      )
