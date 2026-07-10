# 定义 prefill 和 decode 之间的交接数据
from dataclasses import dataclass
from typing import List

@dataclass
class HandoffPayload:
    """
    decode 继续生成时，需要知道：prompt 边界
    当前历史长度
    KV 在哪里
    采样参数是什么
    """
    # 请求唯一标识
    seq_idx: int
    # 当前完整token序列(包含prompt)
    token_ids: List[int]
    # 参与prefill的token数
    num_prompt_tokens: int
    # 已经写入 KV cache 的 token 数
    num_cached_tokens: int
    # 当前序列对应的物理 block 映射
    block_table: List[int]
    # 采样相关参数
    temperature: float
    max_tokens: int
    ignore_eos: bool
    finished: bool