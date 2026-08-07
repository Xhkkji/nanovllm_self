from dataclasses import dataclass

@dataclass
class DecodeStepOutput:
    """
    用于在线decode的流式输出
    """
    scheduled: list
    token_ids: list[int]
    seq_need_compute_logits: list[int]
    finished_seq_ids: list[int]


@dataclass
class OnlineSchedulerStepMetrics:
    """
    【第四章收口改动】
    OnlinePDCoordinator 每推进一轮 step 后记录的调度观测信息。
    命名里显式带 Online/Scheduler/Step，避免和模型性能 profile、decode 输出混淆。
    """
    step_id: int
    emitted_events: int

    prefill_scheduled_decode_tokens: int
    prefill_scheduled_prefill_tokens: int
    prefill_schedule_reason: str

    decode_scheduled_decode_tokens: int
    decode_scheduled_prefill_tokens: int
    decode_schedule_reason: str

    waiting_prefill_requests: int
    prefill_running_sequences: int
    decode_running_sequences: int

    finished_requests: int
    aborted_requests: int
    error_requests: int

    prefill_used_blocks: int
    prefill_free_blocks: int
    decode_used_blocks: int
    decode_free_blocks: int
