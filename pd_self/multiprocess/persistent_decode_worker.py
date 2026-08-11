import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from time import perf_counter

# 常驻 decode worker。
#
# 这个文件是 persistent PD benchmark 的 consumer 侧：
# 1. 进程启动后只加载一次 tokenizer / model / DecodeEngine。
# 2. 一直轮询 work_dir 里的 *.prefill_done。
# 3. 对每个 prefill_done，读取同 base 的 *.payload.pkl。
# 4. 根据 payload.transfer_meta 里的 SharedMemoryKVRef attach shared memory，
#    把 prefill 侧导出的 KV restore 到 decode 侧本地 KV cache。
# 5. 调用 DecodeEngine 持续 decode 到请求完成。
# 6. 写出 *.decode_metrics.json 和 *.decode_done，通知 prefill/driver 这条请求完成。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from pd_self.decode_engine import DecodeEngine
from pd_self.kv_connector import KVConnector
from pd_self.kv_store import SharedMemoryKVStoreBackend, SyncGpuKVStoreBackend
from pd_self.multiprocess.control_plane import control_socket_path, make_control_listener


def sync_cuda():
    """同步当前 CUDA 设备，保证 restore/decode 相关计时覆盖真实 GPU 执行时间。"""
    # CUDA kernel 是异步提交的；benchmark 计时必须同步后再读 perf_counter。
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def destroy_process_group(args) -> None:
    """销毁 NCCL process group。"""
    if not torch.distributed.is_initialized():
        return
    torch.distributed.destroy_process_group()


def build_config(args):
    """构造常驻 decode worker 的 nano-vLLM 配置，并与 prefill worker 的 KV 配置对齐。"""
    # 单 pair 模式下 local_cuda_device 默认是 0；
    # pool 模式下 driver 会让所有 worker 看见同一份 pool GPU 列表，
    # 再用 --local-cuda-device 指定当前 worker 真正使用哪张本地可见卡。
    device = f"cuda:{args.local_cuda_device}"
    return Config(
        model_path=args.model_path,
        device=device,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=512,
        block_size=256,
        num_blocks=64,
        kv_cache_quant_mode=args.kv_cache_quant_mode,
        kv_cache_scale_dtype="fp32",
        attention_compute_dtype="bf16",
    )


def atomic_write_json(path: Path, obj) -> None:
    """原子写 metrics/error JSON，避免 driver 在文件半写入时读取。"""
    # 防止 driver 读到半写入的 metrics/error JSON。
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def write_worker_state(
    args,
    work_dir: Path,
    active_requests: dict | None = None,
    pending_recvs: dict | None = None,
    processed_bases: set | None = None,
    busy: bool = False,
) -> None:
    """写出 decode worker 当前状态，供多 PD pair driver 做真实 worker feedback 调度。"""
    # Agent-aware / Runtime feedback：
    # 外层调度器不直接读取 DecodeEngine 内部对象，而是通过这个轻量 JSON
    # 获得 active decode 数、pending recv 数等信号。这样保持调度层和引擎内部解耦。
    active_requests = active_requests or {}
    pending_recvs = pending_recvs or {}
    processed_bases = processed_bases or set()
    atomic_write_json(
        work_dir / args.state_file,
        {
            "role": "persistent_decode",
            "updated_time_s": time.time(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "decode_mode": args.decode_mode,
            "busy": busy,
            "active_decode_requests": len(active_requests),
            "pending_recvs": len(pending_recvs),
            "processed_bases": len(processed_bases),
            "max_active_decode_requests": args.max_active_decode_requests,
            "max_pending_recvs": args.max_pending_recvs,
        },
    )


def request_base(done_path: Path) -> str:
    """从 xxx.prefill_done 文件名提取请求 base，用于找到同请求的 payload/metrics。"""
    # decode worker 以 prefill_done 作为“可以开始 restore”的信号。
    # base 用来找到同一条请求的 payload / metrics / done 文件。
    suffix = ".prefill_done"
    name = done_path.name
    if not name.endswith(suffix):
        raise ValueError(f"unexpected done filename: {done_path}")
    return name[: -len(suffix)]

def payload_ready_base(ready_path: Path) -> str:
    """
    从 xxx.payload_ready 得到 xxx。

    sync_gpu 模式下，decode 不能等 prefill_done，
    必须先从 payload_ready 开始进入 recv 握手。
    """
    suffix = ".payload_ready"
    name = ready_path.name
    if not name.endswith(suffix):
        raise ValueError(f"unexpected payload_ready filename: {ready_path}")
    return name[: -len(suffix)]

def is_sync_gpu_backend(args) -> bool:
    """判断当前 decode worker 是否使用 sync_gpu/NCCL 传输后端。"""
    return getattr(args, "kv_transfer_backend", "shared_memory") == "sync_gpu"


def build_paths(work_dir: Path, base: str):
    """
    decode 侧文件协议。

    shared_memory 模式：
      decode 扫 *.prefill_done。

    sync_gpu 模式：
      decode 扫 *.payload_ready。
      因为如果 decode 等 prefill_done，会和 prefill 互相等待导致死锁：
        prefill 等 recv_ready
        decode 等 prefill_done
    """
    return {
        "payload": work_dir / f"{base}.payload.pkl",
        "payload_ready": work_dir / f"{base}.payload_ready",
        "recv_ready": work_dir / f"{base}.recv_ready",
        "prefill_done": work_dir / f"{base}.prefill_done",
        "decode_metrics": work_dir / f"{base}.decode_metrics.json",
        "decode_done": work_dir / f"{base}.decode_done",
        "decode_error": work_dir / f"{base}.decode_error.json",
    }


def run_decode_to_finish(engine):
    """旧 run_to_finish 模式：单条 payload restore 后持续 step，直到 scheduler 全部完成。"""
    # DecodeEngine.step() 每次推进一轮调度。
    # 对单请求来说，通常一轮生成一个 token；多请求时可能调度多个 seq。
    # 这里循环到 scheduler 为空，表示这条 payload 恢复出的 seq 已全部完成。
    decode_steps = 0
    decode_step_tokens = []
    decode_finished_ids = []
    empty_steps = 0
    max_empty_steps = 10000

    while not engine.scheduler.is_finished():
        out = engine.step()
        sync_cuda()
        if not out.scheduled:
            # 正常情况下不应该长期空转；这个保护用于发现 scheduler/block 状态异常。
            empty_steps += 1
            if empty_steps > max_empty_steps:
                raise RuntimeError("persistent decode worker made no progress")
            continue
        empty_steps = 0
        decode_steps += 1
        decode_step_tokens.extend(out.token_ids)
        decode_finished_ids.extend(out.finished_seq_ids)

    return decode_steps, decode_step_tokens, decode_finished_ids


def process_payload(args, engine, done_path: Path):
    """旧 shared_memory 串行路径：读取 payload、restore KV、完整 decode 并写完成文件。"""
    # 处理单条已经 prefill_done 的请求。
    #
    # 输入：
    #   *.payload.pkl，里面包含 token_ids、采样参数、transfer_meta。
    # 核心动作：
    #   engine.restore_payloads([payload]) 会分配 decode 本地 blocks，
    #   调用 KVConnector.load_kv() 从 shared memory 拉取 KV，
    #   写入 decode worker 自己的 runner.kv_cache。
    # 输出：
    #   *.decode_metrics.json + *.decode_done。
    work_dir = Path(args.work_dir)
    base = request_base(done_path)
    paths = build_paths(work_dir, base)

    total_t0 = perf_counter()
    payload_read_t0 = perf_counter()
    # 读取 payload 只涉及轻量 metadata；大块 KV 不在这个 pickle 文件里。
    with paths["payload"].open("rb") as f:
        payload = pickle.load(f)
    payload_read_time_s = perf_counter() - payload_read_t0

    with torch.inference_mode():
        restore_t0 = perf_counter()
        # restore 是 PD 分离的关键成本：
        # shared memory -> CPU tensor clone -> decode GPU KV cache。
        # 当前 shared memory backend 仍是 CPU 共享内存，不是 GPU direct transfer。
        results, restored = engine.restore_payloads([payload])
        sync_cuda()
        restore_time_s = perf_counter() - restore_t0

        decode_t0 = perf_counter()
        # restore 完成后，decode 侧已经拥有完整上下文 KV，
        # 后续就和普通 decode 一样逐 token 生成到 max_tokens/eos。
        decode_steps, decode_step_tokens, decode_finished_ids = run_decode_to_finish(engine)
        decode_time_s = perf_counter() - decode_t0

    final_token_lens = [len(seq.token_ids) for seq in restored]
    generated_tokens = sum(
        max(0, len(seq.token_ids) - seq.num_prompt_tokens) for seq in restored
    )

    meta = payload.transfer_meta
    metrics = {
        # decode metrics 记录 restore/decode 两段核心耗时。
        # generated_tokens 是从恢复出的 seq 最终长度减 prompt 长度得到的。
        "role": "persistent_decode",
        "decode_mode": args.decode_mode,
        "request_id": payload.request_id,
        "payload_path": str(paths["payload"]),
        "kv_cache_quant_mode": args.kv_cache_quant_mode,
        "num_kv_blocks": meta.num_kv_blocks if meta is not None else 0,
        "kv_nbytes": meta.storage_ref.nbytes if meta is not None and meta.storage_ref is not None else 0,
        "scale_nbytes": meta.scale_storage_ref.nbytes if meta is not None and meta.scale_storage_ref is not None else 0,
        "payload_read_time_s": payload_read_time_s,
        "restore_time_s": restore_time_s,
        "decode_time_s": decode_time_s,
        "total_time_s": perf_counter() - total_t0,
        "restored_count": len(restored),
        "finished_in_prefill_results": len(results),
        "decode_steps": decode_steps,
        "decode_step_tokens": decode_step_tokens,
        "decode_finished_ids": decode_finished_ids,
        "generated_tokens": generated_tokens,
        "final_token_lens": final_token_lens,
    }
    atomic_write_json(paths["decode_metrics"], metrics)
    # decode_done 是本条请求生命周期结束信号：
    # driver 看到它后汇总结果，prefill 看到它后可清理 producer 侧 metadata。
    paths["decode_done"].write_text("done\n", encoding="utf-8")


def restore_payload_for_continuous(args, engine, signal_path: Path | None = None, base: str | None = None, control=None):
    """
    continuous decode 的文件协议入口。

    shared_memory:
      signal_path = xxx.prefill_done
      decode 此时直接读 payload.pkl，然后从 shared memory restore。

    sync_gpu:
      signal_path = xxx.payload_ready
      decode 此时先读 payload.pkl，拿到 transfer_meta 里的 shape/dtype/rank。
      然后写 recv_ready，通知 prefill 可以 blocking send。
      接着调用 restore_active_payload()，内部会进入 dist.recv。
    """
    work_dir = Path(args.work_dir)
    if base is None:
        if is_sync_gpu_backend(args):
            base = payload_ready_base(signal_path)
        else:
            base = request_base(signal_path)
    paths = build_paths(work_dir, base)

    payload_read_t0 = perf_counter()
    with paths["payload"].open("rb") as f:
        payload = pickle.load(f)
    payload_read_time_s = perf_counter() - payload_read_t0

    
    if is_sync_gpu_backend(args):
        # 关键点：
        # 这行必须在 restore_active_payload() 之前。
        #
        # restore_active_payload()
        # -> restore_sequence()
        # -> kv_connector.load_kv()
        # -> backend.pop_by_ref()
        # -> dist.recv()
        #
        # 如果 decode 不先写 recv_ready，prefill 不会进入 dist.send，
        # decode 进入 dist.recv 后就会一直等。
        if control is None:
            raise RuntimeError("sync_gpu requires control connection")
        control.send(
            {
                "type": "recv_ready",
                "base": base,
            }
        )

    finished_state, active_state = engine.restore_active_payload(
        payload,
        payload_path=str(paths["payload"]),
    )
    state = finished_state or active_state
    state.payload_read_time_s = payload_read_time_s

    if finished_state is not None:
        # prefill 阶段已经生成结束的请求不会进入 active decode 集合。
        # worker 直接写完成文件，driver 就能按普通请求汇总。
        finish_continuous_request(args, paths, finished_state)
        return []

    return [(active_state.seq_idx, base, paths)]

def submit_async_recv_for_continuous(args, backend, work_dir: Path, base: str, control):
    """
    异步 PD decode 侧提交入口。

    sync_gpu 异步路径：
    1. 读 payload metadata
    2. 根据 SyncGpuKVRef 在 decode GPU 上分配 recv buffer
    3. 提交 irecv
    4. 发 recv_ready
    5. 返回 pending recv 状态
    """
    paths = build_paths(work_dir, base)

    payload_read_t0 = perf_counter()
    with paths["payload"].open("rb") as f:
        payload = pickle.load(f)
    payload_read_time_s = perf_counter() - payload_read_t0

    meta = payload.transfer_meta
    if meta is None:
        raise RuntimeError(f"sync_gpu payload {base} has no transfer_meta")

    # 异步 PD 注意时序：
    # torch.distributed.irecv 在 NCCL 后端下可能也会等待对端 isend 进入。
    # 所以这里先通知 prefill 可以提交 isend，再提交本端 irecv，
    # 避免 “decode 卡在 irecv / prefill 卡在等 recv_ready” 的死锁。
    if args.pool_mode:
        # ########################### NCCL 池化 PD 文件握手 ###########################
        # 池化后，一个 decode worker 可能接收多个 prefill worker 的 payload。
        # 为了避免维护多条 socket 连接，第一版直接用文件作为 recv_ready。
        #
        # 时序：
        #   1. decode 看到 payload_ready
        #   2. decode 读 payload，知道 src_rank / shape / dtype
        #   3. decode 写 recv_ready
        #   4. prefill 看到 recv_ready 后提交 isend
        #   5. decode 提交 irecv 并等待完成
        paths["recv_ready"].write_text("ready\n", encoding="utf-8")
    else:
        control.send(
            {
                "type": "recv_ready",
                "base": base,
            }
        )

    recv_state = backend.submit_recv_by_ref(
        kv_ref=meta.storage_ref,
        scale_ref=meta.scale_storage_ref,
        num_kv_blocks=meta.num_kv_blocks,
        last_block_num_tokens=meta.last_block_num_tokens,
    )

    return {
        "payload": payload,
        "paths": paths,
        "recv_state": recv_state,
        "payload_read_time_s": payload_read_time_s,
        "transfer_t0": perf_counter(),
    }


def active_state_to_metrics(args, state):
    """把 DecodeEngine 的 active/finished 状态转换成 benchmark 可以落盘的 metrics 字典。"""
    # 把 DecodeEngine 内部状态转换成 benchmark JSON。
    # 这里不反向访问 Sequence，因为 finished 后 Sequence 已经从 scheduler.running 释放。
    return {
        "role": "persistent_decode",
        "decode_mode": "continuous",
        "request_id": state.request_id,
        "payload_path": state.payload_path,
        "kv_cache_quant_mode": args.kv_cache_quant_mode,
        "num_kv_blocks": state.num_kv_blocks,
        "kv_nbytes": state.kv_nbytes,
        "scale_nbytes": state.scale_nbytes,
        "payload_read_time_s": state.payload_read_time_s,
        "transfer_time_s": state.transfer_time_s,
        "restore_time_s": state.restore_time_s,
        "decode_time_s": state.decode_time_s,
        "decode_compute_time_s": state.decode_compute_time_s,
        "total_time_s": perf_counter() - state.total_t0,
        "restored_count": state.restored_count,
        "finished_in_prefill_results": state.finished_in_prefill_results,
        "decode_steps": state.decode_steps,
        "decode_step_tokens": state.decode_step_tokens,
        "decode_finished_ids": state.decode_finished_ids,
        "generated_tokens": state.generated_tokens,
        "final_token_lens": state.final_token_lens,
    }


def finish_continuous_request(args, paths, state):
    """continuous 模式下完成单条请求：先写 decode metrics，再写 decode_done 信号。"""
    # 一条 continuous active request 完成时，统一落盘 metrics 和 done。
    # decode_done 必须在 metrics 原子写完之后再写，避免 driver 先看到 done 后读不到 metrics。
    metrics = active_state_to_metrics(args, state)
    atomic_write_json(paths["decode_metrics"], metrics)
    paths["decode_done"].write_text("done\n", encoding="utf-8")


def recv_control_message(control, timeout_s: float):
    """
    从 prefill->decode 控制连接读取一条消息。

    返回值：
    - dict：收到一条正常控制消息，比如 payload_ready。
    - None：当前没有消息。
    - EOFError：对端已经关闭 socket。

    multiprocessing.Connection.poll() 在对端关闭时也可能返回 True，
    随后的 recv() 会抛 EOFError。把这段集中起来，主循环就不用在
    多个 recv 点重复处理同一种收尾报错。
    """
    if control is None or not control.poll(timeout_s):
        return None
    return control.recv()


def run_continuous_decode_loop(args, engine, work_dir: Path, control=None):
    """
    Continuous decode 主循环。

    shared_memory 模式：
      扫 *.prefill_done。
      因为 prefill_done 表示 shared memory 已经写好。

    sync_gpu 模式：
      扫 *.payload_ready。
      因为 decode 必须提前读取 payload metadata，并写 recv_ready，
      prefill 才会进入 blocking dist.send。
    """
    processed_bases = set()
    active_requests = {}
    # 异步 PD pending_recvs：
    # sync_gpu 收到 payload_ready 后，不再 blocking recv。
    # decode 先提交 irecv 并保存到 pending_recvs；
    # 后续主循环一边 continuous_step，一边轮询这些 recv 是否完成。
    pending_recvs = {}
    seq_idx_to_base = {}
    empty_steps = 0
    max_empty_steps = 10000
    shutdown_path = work_dir / args.shutdown_file
    write_worker_state(args, work_dir, active_requests, pending_recvs, processed_bases)

    while not shutdown_path.exists() or active_requests or pending_recvs:
        made_progress = False
        write_worker_state(
            args,
            work_dir,
            active_requests,
            pending_recvs,
            processed_bases,
            busy=bool(active_requests or pending_recvs),
        )
        
        if is_sync_gpu_backend(args):
            if args.pool_mode:
                # ########################### NCCL 池化 PD decode 入口 ###########################
                # decode worker 只扫描自己的 work_dir。
                # 任意 prefill worker 如果选择了这个 decode，都会把 payload_ready 写到这里。
                ready_paths = sorted(work_dir.glob("*.payload_ready"))
                signal_bases = [payload_ready_base(path) for path in ready_paths]
        
            else:
                signal_bases = []
                if len(pending_recvs) < args.max_pending_recvs:
                    # 没有 active request 时，可以稍微等一下 socket；
                    # 有 active request 时，不阻塞 decode step。
                    #
                    # 异步 PD 这里看 pending_recvs 容量，而不是 active_requests。
                    # 收到 payload_ready 后会先进入 pending_recvs，不会立刻占用 active slot。
                    poll_timeout = args.poll_interval_s if not active_requests else 0.0
                    try:
                        msg = recv_control_message(control, poll_timeout)
                    except EOFError:
                        if not active_requests and not pending_recvs:
                            break
                        control = None
                        msg = None

                    if msg is not None:
                        if msg.get("type") != "payload_ready":
                            raise RuntimeError(f"unexpected control message: {msg}")
                        signal_bases.append(msg["base"])
                
                # 尽量把 socket 里已经到达的 payload_ready 一次性捞出来，
                # 但不要超过 max_active_decode_requests。
                while(
                    control is not None
                    and len(pending_recvs) + len(signal_bases) < args.max_pending_recvs
                ):
                    try:
                        msg = recv_control_message(control, 0.0)
                    except EOFError:
                        control = None
                        break
                    if msg is None:
                        break
                    if msg.get("type") != "payload_ready":
                        raise RuntimeError(f"unexpected control message: {msg}")
                    signal_bases.append(msg["base"])
                signal_paths = []

        else:
            signal_bases = []
            signal_paths = sorted(work_dir.glob("*.prefill_done"))

        signals = signal_bases if is_sync_gpu_backend(args) else signal_paths
        for signal in signals:
            if is_sync_gpu_backend(args):
                # sync_gpu 异步版里，收到 signal 后只是提交 irecv，
                # 不会立刻加入 active_requests。
                # 所以这里限制的是 pending_recvs，而不是 active_requests。
                if len(pending_recvs) >= args.max_pending_recvs:
                    break

                base = signal
                signal_path = None
            else:
                # shared_memory 还是旧逻辑：
                # 读取 payload 后马上 restore，并加入 active_requests。
                if len(active_requests) >= args.max_active_decode_requests:
                    break

                signal_path = signal
                base = request_base(signal_path)

            paths = build_paths(work_dir, base)

            if base in processed_bases:
                continue
            if paths["decode_done"].exists() or paths["decode_error"].exists():
                processed_bases.add(base)
                continue
            if not paths["payload"].exists():
                continue

            try:
                if is_sync_gpu_backend(args):
                    # 异步 GPU P2P 路径：
                    # 这里只提交 irecv，并给 prefill 回 recv_ready。
                    # 注意：这里不调用 restore_payload_for_continuous()，
                    # 因为那个函数内部会走 blocking recv。
                    pending_recvs[base] = submit_async_recv_for_continuous(
                        args=args,
                        backend=engine.kv_connector.kv_store_backend,
                        work_dir=work_dir,
                        base=base,
                        control=control,
                    )
                    processed_bases.add(base)
                    made_progress = True
                    continue

                # shared_memory 继续走原来的同步 restore 路径。
                restored_entries = restore_payload_for_continuous(
                    args,
                    engine,
                    signal_path=signal_path,
                    base=base,
                    control=control,
                )
                processed_bases.add(base)
                made_progress = True

                for seq_idx, seq_base, seq_paths in restored_entries:
                    active_requests[seq_base] = {
                        "paths": seq_paths,
                    }
                    seq_idx_to_base[seq_idx] = seq_base

            except Exception as exc:
                processed_bases.add(base)
                atomic_write_json(
                    paths["decode_error"],
                    {
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"decode_error {signal_path}: {exc}", flush=True)
        
        # ########################### 异步 PD recv 完成检查 ###########################
        # sync_gpu 的 irecv 提交后会停在 pending_recvs。
        # 每轮主循环先检查哪些 recv 已经完成；完成后再把 KV 注入 decode KV cache，
        # 并把请求加入 active_requests，让后面的 continuous_step 批量 decode。
        if is_sync_gpu_backend(args) and pending_recvs:
            backend = engine.kv_connector.kv_store_backend
            for base, item in list(pending_recvs.items()):
                if len(active_requests) >= args.max_active_decode_requests:
                    break

                if active_requests and not backend.transfer_done(item["recv_state"]):
                    continue
                if not active_requests:
                    # 异步 PD 的第一条请求或 active 全部完成后的补位请求，
                    # 当前没有 decode 计算可以重叠，直接 wait 比空转轮询更稳。
                    backend.wait_transfer(item["recv_state"])

                kv_item = backend.finish_recv(item["recv_state"])
                transfer_time_s = perf_counter() - item["transfer_t0"]

                finished_state, active_state = engine.restore_active_payload_from_kv_item(
                    payload=item["payload"],
                    kv_item=kv_item,
                    payload_path=str(item["paths"]["payload"]),
                    payload_read_time_s=item["payload_read_time_s"],
                    transfer_time_s=transfer_time_s,
                )

                pending_recvs.pop(base, None)
                made_progress = True
                write_worker_state(
                    args,
                    work_dir,
                    active_requests,
                    pending_recvs,
                    processed_bases,
                    busy=True,
                )

                if finished_state is not None:
                    finish_continuous_request(args, item["paths"], finished_state)
                    continue

                active_requests[base] = {
                    "paths": item["paths"],
                }
                seq_idx_to_base[active_state.seq_idx] = base
        # ########################### 异步 PD recv 完成检查 ###########################
        
        if active_requests:
            with torch.inference_mode():
                finished_states, out = engine.continuous_step()

            if not out.scheduled:
                empty_steps += 1
                if empty_steps > max_empty_steps:
                    raise RuntimeError("persistent continuous decode worker made no progress")
            else:
                empty_steps = 0
                made_progress = True

            for state in finished_states:
                base = seq_idx_to_base.pop(state.seq_idx, None)
                if base is None:
                    continue

                entry = active_requests.pop(base, None)
                if entry is None:
                    continue

                finish_continuous_request(args, entry["paths"], state)
                write_worker_state(
                    args,
                    work_dir,
                    active_requests,
                    pending_recvs,
                    processed_bases,
                    busy=bool(active_requests or pending_recvs),
                )

        if not made_progress:
            time.sleep(args.poll_interval_s)


def parse_args():
    """解析常驻 decode worker 参数，包括 decode 模式、传输后端和异步 recv 上限。"""
    parser = argparse.ArgumentParser(description="Persistent decode worker for local PD benchmark.")
    parser.add_argument("--model-path", default="/home/xhk/model/Qwen3-0.6B/")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--ready-file", default="decode_worker.ready.json")
    parser.add_argument("--shutdown-file", default="shutdown")
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument(
        "--decode-mode",
        default="run_to_finish",
        choices=["run_to_finish", "continuous"],
        help="run_to_finish keeps old per-request behavior; continuous batches active decode requests.",
    )
    parser.add_argument(
        "--max-active-decode-requests",
        type=int,
        default=4,
        help="Max restored requests kept active in continuous decode mode.",
    )
    parser.add_argument(
        "--max-pending-recvs",
        type=int,
        default=4,
        help="Max in-flight async PD consumer recvs before applying backpressure.",
    )
    parser.add_argument(
        "--kv-transfer-backend",
        default="shared_memory",
        choices=["shared_memory", "sync_gpu"],
        help="KV transfer backend. sync_gpu scans payload_ready instead of prefill_done.",
    )
    parser.add_argument("--nccl-port", default="29577")
    parser.add_argument("--state-file", default="decode_worker_state.json")
    
    # 池化
    parser.add_argument("--global-rank", type=int, default=1)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--pool-mode", action="store_true")
    parser.add_argument("--decode-worker-id", type=int, default=0)
    parser.add_argument("--local-cuda-device", type=int, default=0)
    
    return parser.parse_args()


def main():
    """常驻 decode worker 主入口：初始化模型/控制面，持续接收 payload 并执行 decode。"""
    args = parse_args()
    
    if args.kv_transfer_backend == "sync_gpu":
        if args.decode_mode != "continuous":
            raise ValueError(
                "sync_gpu only supports continuous decode mode; "
                "run_to_finish may deadlock because it waits for prefill_done."
            )
            
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    control_listener = None
    control = None
    if args.kv_transfer_backend == "sync_gpu" and not args.pool_mode:
        control_listener = make_control_listener(
            control_socket_path(work_dir)
        )
    
    if args.kv_transfer_backend == "sync_gpu":
        torch.cuda.set_device(args.local_cuda_device)
        # ########################### NCCL 池化 PD 初始化 ###########################
        # decode worker 使用 global rank 加入同一个 NCCL world。
        # 它后续根据 payload.transfer_meta.storage_ref.src_rank 接收来自任意 prefill 的 KV。
        
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{args.nccl_port}",
            rank=args.global_rank,
            world_size=args.world_size,
            # ########################### NCCL 池化 PD 本地设备绑定 ###########################
            # ########################### NCCL 池化 PD 本地设备绑定 ###########################
            # pool 模式下 global_rank 表示通信身份，local_cuda_device 表示本进程用的本地 GPU。
            # 这两个概念必须拆开；否则 rank 2/3 很容易被误当成本地 cuda:2/cuda:3。
            device_id=torch.device(f"cuda:{args.local_cuda_device}"),
        )
        backend = SyncGpuKVStoreBackend(rank=args.global_rank, peer_rank=None)
    else:
        backend = SharedMemoryKVStoreBackend()

    print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("torch_current_device", torch.cuda.current_device(), flush=True)
    print("torch_device_name", torch.cuda.get_device_name(args.local_cuda_device), flush=True)

    
    init_t0 = perf_counter()
    # 常驻 decode worker 只初始化一次模型。
    # 这使得 persistent benchmark 的 wall time 接近每条请求真实执行时间，
    # 不再包含每条请求重复加载模型的成本。
    config = build_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    runner = ModelRunner(config)
    
    connector = KVConnector(
        config=config,
        role="consumer",
        engine_id="persistent-decode-worker-0",
        kv_store_backend=backend,
    )
    connector.register_model_runner(runner)
    engine = DecodeEngine(
        config=config,
        tokenizer=tokenizer,
        model_runner=runner,
        kv_connector=connector,
    )
    sync_cuda()
    init_time_s = perf_counter() - init_t0
    
    if args.kv_transfer_backend == "sync_gpu" and not args.pool_mode:
        control = control_listener.accept()

    atomic_write_json(
        work_dir / args.ready_file,
        {
            "role": "persistent_decode",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_current_device": int(torch.cuda.current_device()),
            "torch_device_name": torch.cuda.get_device_name(args.local_cuda_device),
            "kv_cache_quant_mode": args.kv_cache_quant_mode,
            "kv_transfer_backend": args.kv_transfer_backend,
            "control_connected": control is not None,
            "decode_mode": args.decode_mode,
            "max_active_decode_requests": args.max_active_decode_requests,
            "max_pending_recvs": args.max_pending_recvs,
            "model_init_time_s": init_time_s,
        },
    )
    
    # ready 文件通知 driver：decode 侧模型和 KV cache 已准备好。
    print("persistent_decode_ready", work_dir / args.ready_file, flush=True)
    write_worker_state(args, work_dir)

    try:
        if args.decode_mode == "continuous":
            run_continuous_decode_loop(args, engine, work_dir, control=control)
            return

        # 主循环：扫描 prefill_done，而不是 request.json。
        # 这保证 decode 只处理已经完成 KV handoff 的请求。
        while not (work_dir / args.shutdown_file).exists():
            done_paths = sorted(work_dir.glob("*.prefill_done"))
            made_progress = False
            write_worker_state(
                args,
                work_dir,
                busy=False,
            )
            for done_path in done_paths:
                base = request_base(done_path)
                paths = build_paths(work_dir, base)
                if paths["decode_done"].exists() or paths["decode_error"].exists():
                    continue
                if not paths["payload"].exists():
                    # 理论上 prefill_done 写出前 payload 已原子写入；
                    # 这个判断是为了容错，不让 decode 读不存在的文件。
                    continue
                made_progress = True
                try:
                    write_worker_state(
                        args,
                        work_dir,
                        busy=True,
                    )
                    process_payload(args, engine, done_path)
                    write_worker_state(
                        args,
                        work_dir,
                        busy=False,
                    )
                except Exception as exc:
                    # 写 error 文件而不是让 worker 直接退出，方便 driver 报出具体请求的错误。
                    atomic_write_json(
                        paths["decode_error"],
                        {
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"decode_error {done_path}: {exc}", flush=True)
            if not made_progress:
                time.sleep(args.poll_interval_s)
    finally:
        if control is not None:
            control.close()
        if control_listener is not None:
            control_listener.close()
        if args.kv_transfer_backend == "sync_gpu":
            destroy_process_group(args)
        print("persistent_decode_shutdown", flush=True)


if __name__ == "__main__":
    main()
