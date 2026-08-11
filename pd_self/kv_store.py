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

from .payload import SharedMemoryKVRef, SyncGpuKVRef


@dataclass
class AsyncSendState:
    """
    异步 PD producer 侧状态。

    submit_send() 返回后，NCCL send 可能还没完成。
    kv_blocks / scale_blocks 必须保存在这里，直到所有 work.wait() 完成，
    否则底层发送 buffer 可能提前释放。
    """
    handoff_id: str
    works: list
    kv_blocks: torch.Tensor
    scale_blocks: torch.Tensor | None

@dataclass
class AsyncRecvState:
    """
    异步 PD consumer 侧状态。

    submit_recv_by_ref() 只提交 irecv 并保存接收 buffer；
    finish_recv() 在传输完成后把这些 buffer 包成 KVStoreItem。
    """
    kv_ref: object
    scale_ref: object | None
    works: list
    kv_blocks: torch.Tensor
    scale_blocks: torch.Tensor | None
    num_kv_blocks: int
    last_block_num_tokens: int

# pd同步传输逻辑
@dataclass
class PendingGpuKVItem:
    kv_blocks: torch.Tensor
    scale_blocks: torch.Tensor | None
    num_kv_blocks: int
    last_block_num_tokens: int
    storage_ref: object
    scale_storage_ref: object | None = None

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
    20260726
    - kv_blocks / scale_blocks 写入 shared memory
    - item 里只返回 storage_ref / scale_storage_ref
    """
    # shared memory backend 里真正返回的是 ref
    # 不一定保留 tensor,dict模式才保留
    kv_blocks: torch.Tensor | None
    num_kv_blocks: int
    last_block_num_tokens: int
    
    # int8_mock 时携带 scale blocks
    # kv_blocks:
    # [2, num_layers, num_kv_blocks, block_size, num_kv_heads, head_dim]

    # scale_blocks:
    # [2, num_layers, num_kv_blocks, block_size, num_kv_heads, 1]
    scale_blocks: torch.Tensor | None = None
    
    storage_ref: SharedMemoryKVRef | None = None
    scale_storage_ref: SharedMemoryKVRef | None = None
    

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
    
    20260727
    这里需要同时支持：
    - KV blocks
    - int8_mock scale blocks

    KV 和 scale 是两份 tensor，因此对应两块 shared memory。
    """
    def __init__(self):
        self._records: Dict[str, tuple[SharedMemoryKVRef, SharedMemoryKVRef | None, int, int]] = {}

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        torch.Tensor -> numpy.ndarray。

        注意：
        - numpy 不支持 torch.bfloat16，所以 bf16 先转 fp32
        - int8 / fp16 / fp32 可以直接转 numpy
        """
        tensor = tensor.detach().contiguous().cpu()
        
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()
    
    def _write_tensor_to_shm(self, tensor: torch.Tensor) -> SharedMemoryKVRef:
        """
        把一整个 tensor 写到一块 shared memory，并返回定位信息。
        """
        np_array = self._tensor_to_numpy(tensor)
        # 分配一个大小为nparray的共享空间
        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        shm_array = np.ndarray(
            np_array.shape,
            dtype=np_array.dtype,
            buffer=shm.buf,
        )
        # 把np_array写入共享空间
        shm_array[...] = np_array
        
        ref = SharedMemoryKVRef(
            shm_name=shm.name,
            shape=tuple(np_array.shape),
            dtype=str(np_array.dtype),
            nbytes=np_array.nbytes,
        )
        
        shm.close()
        return ref

    def _read_tensor_from_shm(
        self,
        ref: SharedMemoryKVRef,
        unlink: bool,
    ) -> torch.Tensor:
        """
        根据 ref attach 到 shared memory，并读出一份独立 tensor。

        clone 的原因：
        - torch.from_numpy(np_array) 共享 shm 底层内存
        - clone 后 tensor 独立
        - 后面 shm.close()/unlink() 不会影响返回值
        """
        shm = shared_memory.SharedMemory(name=ref.shm_name)
        np_array = np.ndarray(
            ref.shape,
            dtype=np.dtype(ref.dtype),
            buffer=shm.buf,
        )

        tensor = torch.from_numpy(np_array).clone()

        shm.close()
        if unlink:
            shm.unlink()

        return tensor
    
    def _load_from_record(
        self,
        record: tuple[SharedMemoryKVRef, SharedMemoryKVRef | None, int, int],
        unlink: bool,
    ) -> KVStoreItem:
        """
        从一条 record 恢复 KVStoreItem。

        record:
        - kv_ref
        - scale_ref
        - num_kv_blocks
        - last_block_num_tokens
        """
        kv_ref, scale_ref, num_kv_blocks, last_block_num_tokens = record

        kv_blocks = self._read_tensor_from_shm(kv_ref, unlink=unlink)

        scale_blocks = None
        if scale_ref is not None:
            scale_blocks = self._read_tensor_from_shm(scale_ref, unlink=unlink)
        
        return KVStoreItem(
            kv_blocks=kv_blocks,
            scale_blocks=scale_blocks,
            num_kv_blocks=num_kv_blocks,
            last_block_num_tokens=last_block_num_tokens,
            storage_ref=kv_ref,
            scale_storage_ref=scale_ref,
        )
        
    def put(self, handoff_id: str, item: KVStoreItem) -> KVStoreItem:
        if item.kv_blocks is None:
            raise ValueError("SharedMemoryKVStoreBackend.put requires kv_blocks")

        # 第一版先转成 float32 存进 shared memory，避免 bf16 无法直接转 numpy 的问题。
        kv_ref = self._write_tensor_to_shm(item.kv_blocks)
        scale_ref = None
        if item.scale_blocks is not None:
            scale_ref = self._write_tensor_to_shm(item.scale_blocks)

        self._records[handoff_id] = (
            kv_ref,
            scale_ref,
            item.num_kv_blocks,
            item.last_block_num_tokens,
        )
        return KVStoreItem(
            kv_blocks=None,
            scale_blocks=None,
            num_kv_blocks=item.num_kv_blocks,
            last_block_num_tokens=item.last_block_num_tokens,
            storage_ref=kv_ref,
            scale_storage_ref=scale_ref,
        )

    def get(self, handoff_id: str) -> KVStoreItem:
        """
        只读加载：保留 _records，也不 unlink shared memory。
        """
        if handoff_id not in self._records:
            raise KeyError(f"KV handoff_id not found: {handoff_id}")
        
        return self._load_from_record(
            self._records[handoff_id],
            unlink=False
        )
        
    def pop(self, handoff_id: str) -> KVStoreItem:
        """
        按 handoff_id 消费：删除 _records，并 unlink 对应 shared memory。
        """
        if handoff_id not in self._records:
            raise KeyError(f"KV handoff_id not found: {handoff_id}")
        
        return self._load_from_record(
            self._records.pop(handoff_id),
            unlink=True
        )

    def pop_by_ref(
        self,
        kv_ref: SharedMemoryKVRef,
        num_kv_blocks: int,
        last_block_num_tokens: int,
        scale_ref: SharedMemoryKVRef | None = None,
        unlink: bool = True,
    ) -> KVStoreItem:
        """
        按 ref 加载，不依赖 _records。

        unlink=True 时消费 shared memory；unlink=False 适合调试、多消费者或重放。
        """
        return self._load_from_record(
            (kv_ref,
            scale_ref,
            num_kv_blocks,
            last_block_num_tokens,),
            unlink=unlink
        )
    
    def _delete_ref(self, ref: SharedMemoryKVRef | None) -> None:
        """
        删除一块 shared memory。
        """
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

    def delete(self, handoff_id: str) -> None:
        """
        按 handoff_id 清理 _records 和对应 shared memory。
        """
        record = self._records.pop(handoff_id, None)
        if record is None:
            return
        kv_ref, scale_ref, _, _ = record
        self._delete_ref(kv_ref)
        self._delete_ref(scale_ref)
    
    def delete_by_ref(self, ref) -> None:
        """
        只清理一块 shared memory，不访问 _records。
        """
        self._delete_ref(ref)

    def exists(self, handoff_id: str) -> bool:
        return handoff_id in self._records

class SyncGpuKVStoreBackend(KVStoreBackend):
    """
    单机双卡同步 GPU KV 传输 backend。

    第一版设计：
    - producer put() 只暂存 GPU tensor，不立刻 send。
    - producer send_pending() 阻塞发送 KV。
    - consumer pop_by_ref() 阻塞接收 KV。
    - 传输完成前 producer 不能释放 pending tensor。
    """
    device_direct = True

    def __init__(self, rank: int, peer_rank: int | None = None):
        self.rank = rank
        self.peer_rank = peer_rank
        self._pending: dict[str, PendingGpuKVItem] = {}

    def set_peer_rank(self, peer_rank: int) -> None:
        """
        ########################### NCCL 池化 PD ###########################
        设置“本条请求”的目标 decode rank。

        单 pair 时代 peer_rank 固定；池化时代同一个 prefill worker
        可能连续把不同请求发给不同 decode worker，所以 prefill 在 run_prefill()
        前根据 request["pd_pool"]["dst_rank"] 动态设置。
        """
        self.peer_rank = peer_rank

    def _dtype_to_torch(self, dtype: str):
        if dtype == "torch.bfloat16":
            return torch.bfloat16
        if dtype == "torch.float16":
            return torch.float16
        if dtype == "torch.float32":
            return torch.float32
        if dtype == "torch.int8":
            return torch.int8
        raise ValueError(f"unsupported dtype: {dtype}")

    def _tensor_nbytes(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()
    # ########################### 异步 PD 数据面 ###########################
    def submit_send(self, handoff_id: str) -> AsyncSendState:
        """
        producer 侧异步提交 GPU KV send。

        注意：
        - 这里 pop _pending，但不能释放 tensor。
        - tensor 必须挂在 AsyncSendState 里，直到所有 work 完成。
        """
        item = self._pending.pop(handoff_id)
        dst_rank = item.storage_ref.dst_rank

        works = [
            torch.distributed.isend(item.kv_blocks, dst=dst_rank)
        ]
        if item.scale_blocks is not None:
            works.append(torch.distributed.isend(item.scale_blocks, dst=dst_rank))

        return AsyncSendState(
            handoff_id=handoff_id,
            works=works,
            kv_blocks=item.kv_blocks,
            scale_blocks=item.scale_blocks,
        )
    
    def submit_recv_by_ref(
        self,
        kv_ref,
        scale_ref=None,
        num_kv_blocks: int = 0,
        last_block_num_tokens: int = 0,
    ) -> AsyncRecvState:
        """
        consumer 侧异步提交 GPU KV recv。

        这里只分配接收 buffer 并提交 irecv，不返回 KVStoreItem。
        等所有 work 完成后，再 finish_recv()。
        """
        device = torch.device("cuda")
        dtype = self._dtype_to_torch(kv_ref.dtype)

        kv_blocks = torch.empty(
            kv_ref.shape,
            device=device,
            dtype=dtype,
        )

        works = [
            torch.distributed.irecv(kv_blocks, src=kv_ref.src_rank)
        ]

        scale_blocks = None
        if kv_ref.scale_shape is not None:
            scale_dtype = self._dtype_to_torch(kv_ref.scale_dtype)
            scale_blocks = torch.empty(
                kv_ref.scale_shape,
                device=device,
                dtype=scale_dtype,
            )
            works.append(torch.distributed.irecv(scale_blocks, src=kv_ref.src_rank))

        return AsyncRecvState(
            kv_ref=kv_ref,
            scale_ref=scale_ref,
            works=works,
            kv_blocks=kv_blocks,
            scale_blocks=scale_blocks,
            num_kv_blocks=num_kv_blocks,
            last_block_num_tokens=last_block_num_tokens,
        )
        
    def transfer_done(self, state) -> bool:
        return all(work.is_completed() for work in state.works)


    def wait_transfer(self, state) -> None:
        for work in state.works:
            work.wait()


    def finish_recv(self, state: AsyncRecvState) -> KVStoreItem:
        """
        consumer 侧在 irecv 全部完成后调用。
        返回形式和 pop_by_ref() 一样，方便上层复用旧 restore 逻辑。
        """
        self.wait_transfer(state)

        return KVStoreItem(
            kv_blocks=state.kv_blocks,
            scale_blocks=state.scale_blocks,
            num_kv_blocks=state.num_kv_blocks,
            last_block_num_tokens=state.last_block_num_tokens,
            storage_ref=state.kv_ref,
            scale_storage_ref=state.scale_ref,
        )
    # ########################### 异步 PD 数据面 ###########################

    def put(self, handoff_id: str, item: KVStoreItem) -> KVStoreItem:
        """
        producer 侧调用。

        注意：
        - item.kv_blocks 必须已经在 producer GPU 上。
        - 这里不能 CPU 化。
        - 这里只保存 pending，不做 blocking send。
        """
        if item.kv_blocks is None:
            raise ValueError("SyncGpuKVStoreBackend.put requires kv_blocks")
        if self.peer_rank is None:
            raise RuntimeError("sync_gpu requires peer_rank before put()")

        kv_blocks = item.kv_blocks.detach().contiguous()
        scale_blocks = (
            item.scale_blocks.detach().contiguous()
            if item.scale_blocks is not None
            else None
        )

        ref = SyncGpuKVRef(
            handoff_id=handoff_id,
            shape=tuple(kv_blocks.shape),
            dtype=str(kv_blocks.dtype),
            nbytes=self._tensor_nbytes(kv_blocks),
            src_rank=self.rank,
            dst_rank=self.peer_rank,
            scale_shape=tuple(scale_blocks.shape) if scale_blocks is not None else None,
            scale_dtype=str(scale_blocks.dtype) if scale_blocks is not None else None,
            scale_nbytes=self._tensor_nbytes(scale_blocks) if scale_blocks is not None else 0,
        )        

        self._pending[handoff_id] = PendingGpuKVItem(
            kv_blocks=kv_blocks,
            scale_blocks=scale_blocks,
            num_kv_blocks=item.num_kv_blocks,
            last_block_num_tokens=item.last_block_num_tokens,
            storage_ref=ref,
        )
        
        return KVStoreItem(
            kv_blocks=None,
            scale_blocks=None,
            num_kv_blocks=item.num_kv_blocks,
            last_block_num_tokens=item.last_block_num_tokens,
            storage_ref=ref,
            scale_storage_ref=None,
        )
    
    def send_pending(self, handoff_id: str) -> None:
        """
        producer 侧在 decode 已经 recv_ready 后调用。

        这是同步传输点：
        - send 不完成，prefill worker 不继续释放这份 pending KV。
        """
        item = self._pending.pop(handoff_id)
        dst_rank = item.storage_ref.dst_rank

        torch.distributed.send(item.kv_blocks, dst=dst_rank)
        if item.scale_blocks is not None:
            torch.distributed.send(item.scale_blocks, dst=dst_rank)

    def pop_by_ref(
        self,
        kv_ref,
        scale_ref=None,
        num_kv_blocks: int=0,
        last_block_num_tokens: int = 0,
        unlink: bool = True,
    ):
        """
        consumer 侧调用。

        这里会阻塞等待 producer 侧 dist.send。
        返回的 kv_blocks 已经在 decode GPU 上。
        """
        device = torch.device("cuda")
        dtype = self._dtype_to_torch(kv_ref.dtype)
        
        kv_blocks = torch.empty(
            kv_ref.shape,
            device=device,
            dtype=dtype,
        )
        
        # 同步接收的时候，kv和scale都存在同一个SyncGpuKVRef中
        torch.distributed.recv(kv_blocks, src=kv_ref.src_rank)
        
        scale_blocks = None
        if kv_ref.scale_shape is not None:
            scale_dtype = self._dtype_to_torch(kv_ref.scale_dtype)
            scale_blocks = torch.empty(
                kv_ref.scale_shape,
                device=device,
                dtype=scale_dtype,
            )
            torch.distributed.recv(scale_blocks, src=kv_ref.src_rank)
        
        return KVStoreItem(
            kv_blocks=kv_blocks,
            scale_blocks=scale_blocks,
            num_kv_blocks=num_kv_blocks,
            last_block_num_tokens=last_block_num_tokens,
            storage_ref=kv_ref,
            scale_storage_ref=None,
        )
    
    def get(self, handoff_id: str) -> KVStoreItem:
        raise NotImplementedError("sync gpu backend only supports put + send_pending + pop_by_ref")

    def pop(self, handoff_id: str) -> KVStoreItem:
        raise NotImplementedError("sync gpu backend only supports pop_by_ref")

    def delete(self, handoff_id: str) -> None:
        self._pending.pop(handoff_id, None)

    def delete_by_ref(self, ref) -> None:
        if ref is not None:
            self.delete(ref.handoff_id)

    def exists(self, handoff_id: str) -> bool:
        return handoff_id in self._pending
