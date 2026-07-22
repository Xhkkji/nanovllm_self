from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue
from time import perf_counter
from typing import Optional

from .payload import HandoffPayload

# 关键点：
# RequestState 是服务对象。
# Sequence 是模型执行对象。
# 不要把 request 生命周期都塞进 Sequence。
# request_id 应该进入 HandoffPayload，不能再临时拼 req-{seq_idx}。

class RequestStatus(Enum):
    WAITING_PREFILL = auto()
    PREFILLING = auto()
    WAITING_DECODE = auto()
    DECODING = auto()
    FINISHED = auto()
    ABORTED = auto()
    ERROR = auto()

@dataclass
class TokenEvent:
    request_id: str
    token_id: int | None
    text: str = ""
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None

@dataclass
class RequestState:
    request_id: str
    seq_idx: int
    prompt: str
    input_ids: list[int]
    output_ids: list[int] = field(default_factory=list)

    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = True

    status: RequestStatus = RequestStatus.WAITING_PREFILL
    handoff_payload: Optional[HandoffPayload] = None
    stream_queue: Queue = field(default_factory=Queue)  # 每次实例化都新建
 
    created_time: float = field(default_factory=perf_counter)
    finished_time: float | None = None
    error: str | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.input_ids + self.output_ids
    
    def mark_finished(self, finish_reason: str = "stop"):
        """
        正常结束
        """
        self.status = RequestStatus.FINISHED          # ① 状态切换
        self.finished_time = perf_counter()          # ② 记录结束时间
        self.stream_queue.put(TokenEvent(            # ③ 发送结束信号
            request_id=self.request_id,
            token_id=None,                           # 没有新的 token
            finished=True,                           # 标记结束
            finish_reason=finish_reason,             # 结束原因
        ))

    def mark_error(self, error: Exception | str):
        """
        异常结束
        """
        self.status = RequestStatus.ERROR           # ① 错误状态
        self.error = str(error)                     # ② 记录错误信息
        self.finished_time = perf_counter()         # ③ 记录失败时间
        self.stream_queue.put(TokenEvent(           # ④ 发送错误信号
            request_id=self.request_id,
            token_id=None,
            finished=True,
            finish_reason="error",                  # 明确标记为错误
            error=str(error),                       # 携带错误详情
        ))




