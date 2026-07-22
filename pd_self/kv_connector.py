from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import torch

from .payload import KVTransferMeta
from .kv_store import KVStoreBackend, KVStoreItem

class KVConnector:
    """
    控制面 / 编排层。

    它做的事情是：
    1. producer 侧：从 runner.kv_cache 抽取一段 KV
    2. 把这段 KV 交给 backend 存起来
    3. 生成一个轻量 transfer_meta 返回给 payload
    4. consumer 侧：根据 transfer_meta 去 backend 取 KV
    5. 写回 decode 侧本地 kv_cache

    这层不直接决定 KV 存在哪。
    """
    def __init__(self, config, role: str, engine_id: str, kv_store_backend: KVStoreBackend):
        assert role in ("producer", "consumer")
        self.config = config
        self.role = role  # "producer" or "consumer"
        self.engine_id = engine_id
        self.kv_store_backend = kv_store_backend
        self.model_runner = None
    
    def register_model_runner(self, model_runner):
        """
        绑定model_runner
        """
        self.model_runner = model_runner
    
    def _build_handoff_id(self, seq) -> str:
        """
        handoff_id 建议与一次具体 handoff 唯一绑定。
        当前先用：
        engine_id + seq_idx + cached_tokens + uuid
        """
        return (
            f"{self.engine_id}:"
            f"{seq.seq_idx}:"
            f"{seq.num_cached_tokens}:"
            f"{uuid.uuid4().hex[:8]}"
        )
    
    def _compute_last_block_num_tokens(self, num_cached: int) -> int:
        """
        注意这里必须基于 num_cached_tokens，而不是 len(seq.token_ids)。
        因为 handoff 边界时，通常第一枚生成 token 已经 append，
        但还没写入 KV。
        """
        if num_cached == 0:
            return 0
        
        block_size = self.config.block_size
        rem = num_cached % block_size
        return block_size if rem == 0 else rem
    
    def _export_kv_snapshot(self, seq) -> tuple[torch.Tensor, int ,int, list[int]]:
        """
        从 producer runner 的 paged KV cache 中，
        抽出当前 seq 已缓存部分对应的 block 快照。
        """
        assert self.model_runner is not None
        num_cached = seq.num_cached_tokens
        if num_cached == 0:
            return None, 0, 0, []
        
        block_size = self.config.block_size
        num_kv_blocks = (num_cached + block_size - 1) // block_size
        last_block_num_tokens = self._compute_last_block_num_tokens(num_cached)

        # 只导出真正已缓存部分对应的 block
        src_block_ids = list(seq.block_table[:num_kv_blocks])
        
        # 当前 kv_cache 布局：
        # kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # cpu快照：把 GPU 上某一时刻的 KV 数据，完整复制一份到 CPU 内存里，变成一份独立、可传递、后续不会被原地改写的数据副本
        # 并且把独立的block处理成连续的
        # 利用src_block_ids从model_runner的kvcache中取出对应的块
        kv_blocks = (
            self.model_runner.kv_cache[:, :, src_block_ids].detach().cpu().clone()
        )
        return kv_blocks, num_kv_blocks, last_block_num_tokens, src_block_ids


    def save_kv(self, seq) -> Optional[KVTransferMeta]:
        """
        从 prefill runner 导出 block 级 KV。

        注意：
        num_cached_tokens 才表示真正已入 KV 的 token 数。
        刚完成prefill的seq，token_ids 里可能已经包含了第一枚生成 token，但那一枚通常还没写入 KV。

        20260713 
        producer 侧：
        从本地 runner 的 kv_cache 抽取当前已缓存部分，
        存入 connector 自己管理的 store，
        然后只返回 metadata。

        producer 侧入口：
        从本地 kv_cache 抽 KV -> 交给 backend -> 返回轻量 metadata
        """
        assert self.role == "producer"
        assert self.model_runner is not None

        num_cached = seq.num_cached_tokens
        if num_cached == 0:
            return None
        
        kv_blocks, num_kv_blocks, last_block_num_tokens, src_block_ids = \
            self._export_kv_snapshot(seq)
        
        handoff_id = self._build_handoff_id(seq)

        # 储存kvcache数据
        stored_item = self.kv_store_backend.put(
            handoff_id, 
            KVStoreItem(
                kv_blocks=kv_blocks,
                num_kv_blocks=num_kv_blocks,
                last_block_num_tokens=last_block_num_tokens,
            )
        )
        # 返回对应的元数据
        return KVTransferMeta(
            handoff_id=handoff_id,
            producer_id=self.engine_id,
            num_cached_tokens=num_cached,
            num_kv_blocks=num_kv_blocks,
            last_block_num_tokens=last_block_num_tokens,
            src_block_table=src_block_ids,  # 仅用于 debug
            storage_ref=stored_item.storage_ref,
        ) 

    def load_kv(self, transfer_meta: Optional[KVTransferMeta], dst_block_ids: list[int], consume: bool = True) -> None:
        """
        consumer 侧入口：
        根据 transfer_meta 从 backend 拉 KV，
        再写入 decode 侧本地 block。

        consume=True:
        - 更像“一次性交付”
        - decode 成功恢复后就把 store 中的数据删掉

        consume=False:
        - 适合将来调试 / 重放 / 多副本消费者
        """
        assert self.role == "consumer"
        assert self.model_runner is not None

        if transfer_meta is None or not dst_block_ids:
            return

        if transfer_meta.storage_ref is not None:
            # shared memory / remote backend 路径
            item = self.kv_store_backend.pop_by_ref(
                ref=transfer_meta.storage_ref,
                num_kv_blocks=transfer_meta.num_kv_blocks,
                last_block_num_tokens=transfer_meta.last_block_num_tokens,
                unlink=consume,
            )
        else:
            # dict backend 路径
            # 利用元数据取出kv数据
            if consume:
                item = self.kv_store_backend.pop(transfer_meta.handoff_id)
            else:
                item = self.kv_store_backend.get(transfer_meta.handoff_id)
        
        # 把kv数据获取到GPU上
        device_kv_blocks = item.kv_blocks.to(
            self.model_runner.kv_cache.device,
            non_blocking=True,
            )

        # 把获取的数据放到consumer的kvcache里面
        # kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        for i, block_id in enumerate(dst_block_ids):
            # non_blokcing=True：CPU 发出搬运指令后，不等待搬运完成，立刻执行下一行代码。这叫异步传输。
            # 技术原理：它利用 CUDA 的流（Stream）机制，把数据拷贝和 GPU 计算重叠进行。搬运归搬运，计算归计算，两者互不阻塞。
            self.model_runner.kv_cache[:, :, block_id].copy_(device_kv_blocks[:, :, i])


    def discard(self, transfer_meta: Optional[KVTransferMeta]) -> None:
        """
        如果某次 handoff 在 decode 前就被取消 / 失败，
        可以显式清理 backend 中残留的数据。
        """
        if transfer_meta is None:
            return

        if transfer_meta.storage_ref is not None:
            self.kv_store_backend.delete_by_ref(transfer_meta.storage_ref)
            return
            
        self.kv_store_backend.delete(transfer_meta.handoff_id)
