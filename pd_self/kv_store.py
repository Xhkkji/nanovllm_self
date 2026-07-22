from __future__ import annotations

# ABC（Abstract Base Class）：
# 让这个类变成“抽象类”。这意味着这个类不能直接实例化（不能 x = MyClass()），它只能被别的类继承。

# abstractmethod（抽象方法装饰器）：
# 被这个装饰器标记的方法，子类必须重写（Override），否则子类也无法实例化。
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
from multiprocessing import shared_memory

import numpy as np
import torch

from .payload import SharedMemoryKVRef

@dataclass
class KVStoreItem:
    """
    backend 的数据对象。

    Dict backend:
    - kv_blocks 直接存在 Python 内存

    SharedMemory backend:
    - put() 时把 kv_blocks 写进 shared memory
    - 返回 storage_ref
    - pop_by_ref() 时根据 storage_ref 读出 KV
    """
    kv_blocks: torch.Tensor
    num_kv_blocks: int
    last_block_num_tokens: int
    storage_ref: SharedMemoryKVRef | None = None

class KVStoreBackend(ABC):
    """
    抽象类
    数据面抽象层。

    这层不关心 Sequence / block_manager / payload，
    只负责：
    1. 存一份 handoff 对应的 KV
    2. 取一份 handoff 对应的 KV
    3. 清理一份 handoff 对应的 KV
    """
    @abstractmethod
    def put(self, handoff_id: str, item: KVStoreItem) -> KVStoreItem:
        raise NotImplementedError
    
    @abstractmethod
    def get(self, handoff_id: str) -> KVStoreItem:
        raise NotImplementedError

    @abstractmethod
    def pop(self, handoff_id: str) -> KVStoreItem:
        raise NotImplementedError

    @abstractmethod
    def delete(self, handoff_id: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def delete_by_ref(self, ref) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, handoff_id: str) -> bool:
        raise NotImplementedError

class DictKVStoreBackend(KVStoreBackend):
    """
    单进程 / 单机最小实现。

    这版非常适合你当前阶段：
    - 逻辑简单
    - 方便调试
    - 后面替换 backend 时，connector / engine 基本不动
    """
    def __init__(self):
        self._store: Dict[str, KVStoreItem] = {}
    
    def put(self, handoff_id: str, item: KVStoreItem) -> None:
        self._store[handoff_id] = item
        return item
    
    def get(self, handoff_id: str) -> KVStoreItem:
        if handoff_id not in self._store:
            raise KeyError(f"KV handoff_id not found: {handoff_id}")
        return self._store[handoff_id]
    
    def pop(self, handoff_id: str) -> KVStoreItem:
        if handoff_id not in self._store:
            raise KeyError(f"KV handoff_id not found: {handoff_id}")
        return self._store.pop(handoff_id)
    
    def delete(self, handoff_id: str) -> None:
        self._store.pop(handoff_id, None)
    
    def delete_by_ref(self, ref) -> None:
        # dict backend 没有跨进程 ref，保持 no-op
        return

    def exists(self, handoff_id: str) -> bool:
        return handoff_id in self._store

class SharedMemoryKVStoreBackend(KVStoreBackend):
    """
    单机多进程 shared memory backend。

    注意：
    - backend 内部 _refs 只适合同进程调试
    - 真正跨进程时，decode 应该通过 transfer_meta.storage_ref attach
    """
    def __init__(self):
        self._records: Dict[str, tuple[SharedMemoryKVRef, int, int]] = {}

    def put(self, handoff_id: str, item: KVStoreItem) -> KVStoreItem:
        if item.kv_blocks is None:
            raise ValueError("SharedMemoryKVStoreBackend.put requires kv_blocks")

        # 第一版先转成 float32 存进 shared memory，避免 bf16 无法直接转 numpy 的问题。
        kv_blocks = item.kv_blocks.detach().contiguous().cpu().to(torch.float32)
        np_array = kv_blocks.numpy()

        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        shm_array = np.ndarray(
            np_array.shape,
            dtype=np_array.dtype,
            buffer=shm.buf,  # ← 关键！直接使用共享内存作为底层存储
        )
        # 使用 [...] 确保是元素级赋值，而不是引用赋值
        shm_array[...] = np_array

        storage_ref = SharedMemoryKVRef(
            shm_name=shm.name,
            shape=tuple(np_array.shape),
            dtype=str(np_array.dtype),
            nbytes=np_array.nbytes,
        )

        shm.close()

        stored_item = KVStoreItem(
            kv_blocks=None,
            num_kv_blocks=item.num_kv_blocks,
            last_block_num_tokens=item.last_block_num_tokens,
            storage_ref=storage_ref,
        )
        self._records[handoff_id] = (
            storage_ref,
            item.num_kv_blocks,
            item.last_block_num_tokens,
        )
        return stored_item

    def get(self, handoff_id: str) -> KVStoreItem:
        if handoff_id not in self._records:
            raise KeyError(f"KV handoff_id not found: {handoff_id}")
        ref, num_kv_blocks, last_block_num_tokens = self._records[handoff_id]
        return self.load_by_ref(ref, num_kv_blocks, last_block_num_tokens)

    def pop(self, handoff_id: str) -> KVStoreItem:
        if handoff_id not in self._records:
            raise KeyError(f"KV handoff_id not found: {handoff_id}")
        ref, num_kv_blocks, last_block_num_tokens = self._records.pop(handoff_id)
        return self.load_by_ref(ref, num_kv_blocks, last_block_num_tokens, unlink=True)

    def load_by_ref(
        self,
        ref: SharedMemoryKVRef,
        num_kv_blocks: int,
        last_block_num_tokens: int,
        unlink: bool = False,
    ) -> KVStoreItem:
        shm = shared_memory.SharedMemory(name=ref.shm_name)
        np_array = np.ndarray(
            ref.shape,
            dtype=np.dtype(ref.dtype),
            buffer=shm.buf,
        )

        # 第一版用 clone，保证关闭 shm 后 tensor 仍然有效。
        kv_blocks = torch.from_numpy(np_array).clone()

        shm.close()
        if unlink:
            shm.unlink()

        return KVStoreItem(
            kv_blocks=kv_blocks,
            num_kv_blocks=num_kv_blocks,
            last_block_num_tokens=last_block_num_tokens,
            storage_ref=ref,
        )

    def pop_by_ref(
        self,
        ref: SharedMemoryKVRef,
        num_kv_blocks: int,
        last_block_num_tokens: int,
        unlink: bool = True,
    ) -> KVStoreItem:
        shm = shared_memory.SharedMemory(name=ref.shm_name)
        np_array = np.ndarray(
            ref.shape,
            dtype=np.dtype(ref.dtype),
            buffer=shm.buf,
        )

        kv_blocks = torch.from_numpy(np_array).clone()

        shm.close()
        if unlink:
            shm.unlink()

        return KVStoreItem(
            kv_blocks=kv_blocks,
            num_kv_blocks=num_kv_blocks,
            last_block_num_tokens=last_block_num_tokens,
            storage_ref=ref,
        )

    def delete(self, handoff_id: str) -> None:
        record = self._records.pop(handoff_id, None)
        if record is None:
            return
        ref, _, _ = record
        try:
            shm = shared_memory.SharedMemory(name=ref.shm_name)
        except FileNotFoundError:
            return
        shm.close()

        try:
            shm.unlink()
        except FileNotFoundError:
            pass
    
    def delete_by_ref(self, ref) -> None:
        if ref is None:
            return
        try:
            shm = shared_memory.SharedMemory(name=ref.shm_name)
        except FileNotFoundError:
            return
        shm.close()

        try:
            shm.unlink()
        except FileNotFoundError:
            pass

    def exists(self, handoff_id: str) -> bool:
        return handoff_id in self._records
