import os
import torch
import torch.distributed as dist


# _TP_RANK：当前进程是第几张卡（从 0 开始）。
# _TP_SIZE：总共用了多少张卡做张量并行。
_TP_RANK = 0
_TP_SIZE = 1


def init_tensor_parallel(tp_size: int):
    """
    Tensor Parallel 初始化。

    第一版只支持：
    - 单机多卡
    - torchrun 启动
    - 每个进程绑定一张 GPU

    torchrun 会自动注入：
    - RANK
    - WORLD_SIZE
    - LOCAL_RANK
    """
    global _TP_RANK, _TP_SIZE

    _TP_SIZE = tp_size

    if tp_size == 1:
        torch.cuda.set_device(0)
        _TP_RANK = 0
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != tp_size:
        raise ValueError(f"world_size={world_size}, tp_size={tp_size}")

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

    _TP_RANK = rank


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_size() -> int:
    return _TP_SIZE


def tp_all_reduce(x):
    """
    Row Parallel Linear 后需要 all_reduce。

    每个 rank 算出一部分结果，最后 sum 成完整 hidden。
    """
    if _TP_SIZE == 1:
        return x

    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x