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


