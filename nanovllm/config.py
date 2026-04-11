from dataclasses import dataclass
import os
from transformers import AutoConfig

@dataclass
class Config:
    model_path: str
    device: str = "cuda:0"
    dtype: str = "float16"
    max_model_len: int = 2048
    max_num_seqs: int = 256
    block_size: int = 16  # 每个block存储的token数
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 256
    kvcache_block_size: int = 256
    use_cache: bool = False