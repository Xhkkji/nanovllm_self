import torch
from dataclasses import dataclass

@dataclass
class Context:
    """
    一个context对应一个batch，可能对应多个seq
    """
    # is_prefill: bool = False
    # Batch 中有 3 个序列，Query 长度分别为 [3, 5, 2]
    # cu_seqlens_q = [0, 3, 8, 10]
    #                ↑  ↑  ↑  ↑
    #          seq0边界 seq1边界 seq2边界 结束
    # 描述 Query 序列的累积长
    cu_seqlens_q: torch.Tensor | None = None  # 前缀和索引
    # 同样是这 3 个序列，但历史长度不同
    # 序列0: 已有 10 个历史 token
    # 序列1: 已有 8 个历史 token
    # 序列2: 已有 12 个历史 token
    # cu_seqlens_k = [0, 10, 18, 30]
    # 描述 Key/Value 序列的累积长度
    cu_seqlens_k: torch.Tensor | None = None
    # Query 长度分别为 [3, 5, 2]，取5
    max_seqlen_q: int = 0
    # KV 历史长度分别为 [10, 8, 12]，取12
    max_seqlen_k: int = 0
    # slot_mapping：将当前批次中每个 token 的逻辑位置，映射到它在物理 KV 缓存池中的具体存储位置
    slot_mapping: torch.Tensor | None = None  # 逻辑 token 位置 → 物理 KV Cache 位置的映射
    # context_lens：每条 seq 当前完整可见的 KV 长度。
    # 在统一调度下：
    # - chunked prefill 时，等于 prefix + 本轮新增
    # - decode 时，等于当前完整历史长度
    context_lens: torch.Tensor | None = None  # 每个序列已有的上下文长度（历史 token 数）
    # 针对当前这批seqs所有seq的blocktable，补齐长度的二维blocktable矩阵，用于查找
    block_tables: torch.Tensor | None = None
    seq_need_compute_logits: torch.Tensor | None = None


def get_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, seq_need_compute_logits=None):
    return Context(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        seq_need_compute_logits=seq_need_compute_logits,
    )
