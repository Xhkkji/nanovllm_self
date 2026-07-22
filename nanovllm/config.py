from dataclasses import dataclass
import os
from transformers import AutoConfig
import torch

@dataclass
class Config:
    model_path: str
    device: str = "cuda:0"

    dtype: torch.dtype = torch.float16
    # 新增：KV cache 单独的存储 dtype
    kv_cache_dtype: torch.dtype = torch.bfloat16
    # 新增：attention 计算时 q/k/v 使用的 dtype
    attention_compute_dtype: torch.dtype = torch.bfloat16
    # 新增：decode flash 的实现方式
    # 可选: "paged_flash" / "gathered_flash" / "torch"
    # decode_attention_backend: str = "paged_flash"

    max_model_len: int = 2048
    max_num_seqs: int = 256  # 一轮调度里，最多同时处理多少条序列，“并发序列数上限”
    num_blocks: int = 256
    block_size: int = 16  # 每个block存储的token数
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 16384  # 一轮调度里，最多处理多少个 token
    kvcache_block_size: int = 16
    use_cache: bool = True
    eos: int = -1
    # chunk prefill
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 256