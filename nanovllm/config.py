from dataclasses import dataclass
import os
from transformers import AutoConfig
import torch

def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype

    if dtype in (None, "auto"):
        return None

    name_to_dtype = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }

    if dtype not in name_to_dtype:
        raise ValueError(f"unsupported dtype: {dtype}")

    return name_to_dtype[dtype]

@dataclass
class Config:
    model_path: str
    device: str = "cuda:0"

    dtype: torch.dtype | str = torch.float16
    # 新增：KV cache 单独的存储 dtype
    # KV cache 存储 dtype；auto 表示跟随模型 dtype.原先值：torch.bfloat16
    kv_cache_dtype: torch.dtype | str = torch.bfloat16
    # 新增：attention 计算时 q/k/v 使用的 dtype
    # KV cache 存储 dtype；auto 表示跟随模型 dtype.原先值：torch.bfloat16
    attention_compute_dtype: torch.dtype | str = torch.bfloat16
    # 新增：decode flash 的实现方式
    # 可选: "paged_flash" / "gathered_flash" / "torch"
    # decode_attention_backend: str = "paged_flash"
    
    # 新增：KV cache 量化模式
    kv_cache_quant_mode: str = "none"  # "none" / "int8_mock"
    kv_cache_scale_dtype: torch.dtype | str = "fp32"

    max_model_len: int = 2048
    max_num_seqs: int = 256  # 一轮调度里，最多同时处理多少条序列，“并发序列数上限”
    num_blocks: int = 256
    block_size: int = 256  # 每个block存储的token数；flash-attn paged KV cache 要求能被 256 整除
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 16384  # 一轮调度里，最多处理多少个 token
    kvcache_block_size: int = 256
    use_cache: bool = True
    eos: int = -1
    # chunk prefill
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 256

    def __post_init__(self):
        self.dtype = resolve_torch_dtype(self.dtype) or torch.float16

        resolved_kv_dtype = resolve_torch_dtype(self.kv_cache_dtype) or self.dtype
        self.kv_cache_dtype = resolved_kv_dtype or self.dtype
        self.attention_compute_dtype = (
            resolve_torch_dtype(self.attention_compute_dtype) or self.dtype
            )
        
        self.kv_cache_scale_dtype = (
            resolve_torch_dtype(self.kv_cache_scale_dtype) or torch.float32
        )
        
        if self.kv_cache_quant_mode not in ("none", "int8_mock"):
            raise ValueError(
                f"unsupported kv_cache_quant_mode: {self.kv_cache_quant_mode}"
            )

        if self.kv_cache_quant_mode == "int8_mock":
            # int8_mock 下 KV cache 本体存 int8
            self.kv_cache_dtype = torch.int8
