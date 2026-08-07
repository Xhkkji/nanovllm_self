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
    
    def _export_kv_snapshot(self, seq):
        """
        从 producer runner 的 paged KV cache 中，
        抽出当前 seq 已缓存部分对应的 block 快照。
        
        20260726
        导出当前 seq 已缓存部分对应的 KV block。
        int8_mock 时，同时导出 scale block。
        
        普通 bf16/fp16:
        scale_blocks = None

        int8_mock:
        scale_blocks = prefill_runner.kv_scale_cache 对应 block 的快照
        """
        assert self.model_runner is not None
        num_cached = seq.num_cached_tokens
        if num_cached == 0:
            return None, None, 0, 0, []
        
        block_size = self.config.block_size
        num_kv_blocks = (num_cached + block_size - 1) // block_size
        last_block_num_tokens = self._compute_last_block_num_tokens(num_cached)

        # 只导出真正已缓存部分对应的 block
        src_block_ids = list(seq.block_table[:num_kv_blocks])
        
        device_direct = getattr(self.kv_store_backend, "device_direct", False)
        if device_direct:
            # GPU P2P / NCCL 路径：保留在 producer GPU 上。
            kv_blocks = (
                self.model_runner.kv_cache[: ,:, src_block_ids]
                .detach()
                .contiguous()
            )
            
            scale_blocks = None
            if getattr(self.model_runner, "kv_scale_cache", None) is not None:
                scale_blocks = (
                    self.model_runner.kv_scale_cache[:, :, src_block_ids]
                    .detach()
                    .contiguous()
                )
        
        else:
            # 当前 kv_cache 布局：
            # kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            # cpu快照：把 GPU 上某一时刻的 KV 数据，完整复制一份到 CPU 内存里，变成一份独立、可传递、后续不会被原地改写的数据副本
            # 并且把独立的block处理成连续的
            # 利用src_block_ids从model_runner的kvcache中取出对应的块
            kv_blocks = (
                self.model_runner.kv_cache[:, :, src_block_ids].detach().cpu().clone()
            )
            
            # 取出量化scale_block
            scale_blocks = None
            if getattr(self.model_runner, "kv_scale_cache", None) is not None:
                scale_blocks = (
                    self.model_runner.kv_scale_cache[:, :, src_block_ids]
                    .detach()
                    .cpu()
                    .clone()
                )
            
        return (
            kv_blocks,
            scale_blocks,
            num_kv_blocks,
            last_block_num_tokens,
            src_block_ids,
        )
        
    def flush_kv_transfer(self, transfer_meta: KVTransferMeta) -> None:
        """
        producer 侧触发真实 KV 发送。

        shared_memory:
        backend 没有 send_pending，这里直接 no-op。
        因为 put() 已经完成写 shared memory。

        sync_gpu:
        backend 有 send_pending。
        put() 只是暂存 GPU KV；
        send_pending() 才会 blocking dist.send。
        """
        assert self.role == "producer"
        
        if transfer_meta is None:
            return

        send_pending = getattr(self.kv_store_backend, "send_pending", None)
        if send_pending is None:
            return
        
        send_pending(transfer_meta.handoff_id)


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
        
        (
            kv_blocks,
            scale_blocks,
            num_kv_blocks,
            last_block_num_tokens,
            src_block_ids,
        ) = self._export_kv_snapshot(seq)
        
        handoff_id = self._build_handoff_id(seq)

        # KVStoreItem 里真正带大 tensor(stored_item)
        # KVTransferMeta 里只带 metadata/ref(storage_ref)
        
        # 储存kvcache数据
        stored_item = self.kv_store_backend.put(
            handoff_id, 
            KVStoreItem(
                kv_blocks=kv_blocks,
                scale_blocks=scale_blocks,
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
            scale_storage_ref=stored_item.scale_storage_ref,
            kv_cache_quant_mode=self.config.kv_cache_quant_mode,
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
                kv_ref=transfer_meta.storage_ref,
                scale_ref=transfer_meta.scale_storage_ref,
                num_kv_blocks=transfer_meta.num_kv_blocks,
                last_block_num_tokens=transfer_meta.last_block_num_tokens,
                unlink=consume,
            )
            if consume:
                # pop_by_ref 不依赖 backend._records；同进程调试时额外清掉残留 metadata。
                self.kv_store_backend.delete(transfer_meta.handoff_id)
        else:
            # dict backend 路径
            # 利用元数据取出kv数据
            if consume:
                item = self.kv_store_backend.pop(transfer_meta.handoff_id)
            else:
                item = self.kv_store_backend.get(transfer_meta.handoff_id)
        
        self.load_kv_item(item, dst_block_ids)
    
    # ########################### 异步 PD restore 复用入口 ###########################
    def load_kv_item(self, item, dst_block_ids: list[int]) -> None:
        """
        已经拿到 KVStoreItem 后，把 KV 写入 decode 侧本地 KV cache。

        同步路径：
        load_kv() -> backend.pop_by_ref() -> load_kv_item()

        异步路径：
        backend.submit_recv_by_ref()
        -> backend.finish_recv()
        -> load_kv_item()
        """
        assert self.role == "consumer"
        assert self.model_runner is not None

        if item is None or not dst_block_ids:
            return

        device_kv_blocks = item.kv_blocks.to(
            self.model_runner.kv_cache.device,
            non_blocking=True,
        )

        device_scale_blocks = None
        if item.scale_blocks is not None:
            if self.model_runner.kv_scale_cache is None:
                raise RuntimeError(
                    "received scale_blocks but decode runner has no kv_scale_cache"
                )

            device_scale_blocks = item.scale_blocks.to(
                self.model_runner.kv_scale_cache.device,
                non_blocking=True,
            )

        for i, block_id in enumerate(dst_block_ids):
            # kv_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            self.model_runner.kv_cache[:, :, block_id].copy_(
                device_kv_blocks[:, :, i]
            )

            if device_scale_blocks is not None:
                # kv_scale_cache: [2, num_layers, num_blocks, block_size, num_kv_heads, 1]
                self.model_runner.kv_scale_cache[:, :, block_id].copy_(
                    device_scale_blocks[:, :, i]
                )
    # ########################### 异步 PD restore 复用入口 ###########################


    def discard(self, transfer_meta: Optional[KVTransferMeta]) -> None:
        """
        如果某次 handoff 在 decode 前就被取消 / 失败，
        可以显式清理 backend 中残留的数据。
        """
        if transfer_meta is None:
            return

        if transfer_meta.storage_ref is not None:
            self.kv_store_backend.delete_by_ref(transfer_meta.storage_ref)
            self.kv_store_backend.delete_by_ref(transfer_meta.scale_storage_ref)
            return
            
        self.kv_store_backend.delete(transfer_meta.handoff_id)
