# 定义 prefill 和 decode 之间的交接数据
from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class SharedMemoryKVRef:
    """
    跨进程定位一份 KV 数据所需的最小信息。
    decode 进程可以只靠这些字段 attach 到 shared memory。
    """
    shm_name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

@dataclass
class KVTransferMeta:
    """
    只描述“这次 handoff 的 KV 在哪里、大小是多少”。
    不直接携带大 tensor。
    """
    handoff_id: str
    producer_id: str

    # 已经真正写入kvcache的token数
    num_cached_tokens: int
    num_kv_blocks: int
    last_block_num_tokens: int

    # field(default_factory=list) 指定每次创建实例时，该字段默认值通过调用 list() 生成一个新的空列表，而非所有实例共享同一个列表对象
    # 仅用于 trace/debug，不参与 decode 侧真实 block 恢复
    src_block_table: List[int] = field(default_factory=list)

    # shared memory / external backend 的跨进程引用
    storage_ref: Optional["SharedMemoryKVRef"] = None

@dataclass
class HandoffPayload:
    """
    prefill -> decode 的轻量交接对象。
    
    decode 继续生成时，需要知道：prompt 边界
    当前历史长度
    KV 在哪里
    采样参数是什么

    20260713 向vllm和mooncake的方向推进
    只放轻量请求信息。
    真正的大块 KV 数据由 connector 自己管理。
    """
    # 请求唯一标识
    seq_idx: int
    request_id: str
    # 当前完整token序列(包含prompt)
    token_ids: List[int]
    # 参与prefill的token数
    num_prompt_tokens: int
    # 已经写入 KV cache 的 token 数
    num_cached_tokens: int
    # 采样相关参数
    temperature: float
    max_tokens: int
    ignore_eos: bool
    finished: bool

    # 此处不保存kvblock
    # 带transfer metadata
    # Optional[KVTransferMeta] 是 Python 类型注解中的一种写法，表示该字段可以接受两种类型的值：
    transfer_meta: Optional[KVTransferMeta] = None
