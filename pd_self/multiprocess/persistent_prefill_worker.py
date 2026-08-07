import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from time import perf_counter

# 常驻 prefill worker。
#
# 这个文件是 persistent PD benchmark 的 producer 侧：
# 1. 进程启动后只加载一次 tokenizer / model / PrefillEngine。
# 2. 一直轮询 work_dir 里的 *.request.json。
# 3. 每发现一个新请求，就执行 prefill，生成 HandoffPayload。
# 4. HandoffPayload 里只保存轻量 metadata；真正的 KV blocks 已经通过
#    KVConnector -> SharedMemoryKVStoreBackend 写入 shared memory。
# 5. prefill 侧写出 *.payload.pkl 和 *.prefill_done，decode 侧看到
#    *.prefill_done 后再去读取 payload 并 restore KV。
# 6. 串行模式下，prefill 侧等待对应 *.decode_done 后再清理本进程残留 metadata。
# 7. Pipeline 模式下，prefill 侧不等待 decode_done，而是继续处理下一条请求；
#    主循环后台扫描已完成请求并清理 metadata。
#
# 第一版是串行请求：driver 会等一条请求 decode 完成后再发下一条。
# 后续要做 pipeline 时，可以让 driver 连续投递多个 request，worker 循环本身
# 已经具备持续处理多个文件的基本结构。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from pd_self.kv_connector import KVConnector
from pd_self.kv_store import SharedMemoryKVStoreBackend, SyncGpuKVStoreBackend
from pd_self.multiprocess.control_plane import connect_control, control_socket_path
from pd_self.prefill_engine import PrefillEngine


def sync_cuda():
    # 所有计时点前后都显式 synchronize，避免 CUDA 异步执行导致时间偏小。
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_config(args):
    # benchmark 先固定一个较小但足够覆盖 synthetic smoke 的配置。
    # 注意这里 device 永远写 cuda:0，因为每个 worker 进程通过
    # CUDA_VISIBLE_DEVICES 只暴露自己负责的那张物理卡。
    return Config(
        model_path=args.model_path,
        device="cuda:0",
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
    # 先写 .tmp 再 replace，避免 decode/driver 读到半写入的 JSON 文件。
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def atomic_write_pickle(path: Path, obj) -> None:
    # payload.pkl 同样需要原子写。否则 decode 侧可能在 pickle 尚未写完时读取。
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(obj, f)
    tmp_path.replace(path)


def request_base(request_path: Path) -> str:
    # 文件协议约定：
    #   0000_synth-0000.request.json
    # 的 base 是：
    #   0000_synth-0000
    # 后续 payload / metrics / done 都复用这个 base，保证一条请求的文件能对齐。
    suffix = ".request.json"
    name = request_path.name
    if not name.endswith(suffix):
        raise ValueError(f"unexpected request filename: {request_path}")
    return name[: -len(suffix)]


def build_paths(work_dir: Path, base: str):
    """
    prefill 侧文件协议。

    shared_memory 模式：
    - payload.pkl 写好后，prefill_done 就可以写。
    - decode 看到 prefill_done 后读取 payload 并 restore。

    sync_gpu 模式：
    - payload.pkl 写好后，先写 payload_ready。
    - decode 看到 payload_ready 后读取 payload，写 recv_ready，并进入 dist.recv。
    - prefill 看到 recv_ready 后调用 send_pending/flush_kv_transfer。
    - send 完成后再写 prefill_done。

    因此：
    - payload_ready = payload metadata 可读
    - recv_ready = decode 已准备好接收 GPU KV
    - prefill_done = KV handoff 已完成
    """
    return {
        "payload": work_dir / f"{base}.payload.pkl",

        # 新增：payload 元数据已写好，decode 可以读取 payload.pkl。
        "payload_ready": work_dir / f"{base}.payload_ready",

        # 新增：decode 已经读到 payload，并准备调用 dist.recv。
        "recv_ready": work_dir / f"{base}.recv_ready",

        # 注意：同步 P2P 模式下，prefill_done 表示 KV send 已完成。
        "prefill_done": work_dir / f"{base}.prefill_done",

        "prefill_metrics": work_dir / f"{base}.prefill_metrics.json",
        "prefill_error": work_dir / f"{base}.prefill_error.json",
        "decode_done": work_dir / f"{base}.decode_done",
    }
    
def is_sync_gpu_backend(args) -> bool:
    """
    第一版用参数区分文件协议。

    shared_memory:
    payload -> prefill_done

    sync_gpu:
    payload -> payload_ready -> recv_ready -> send -> prefill_done
    """
    return getattr(args, "kv_transfer_backend", "shared_memory") == "sync_gpu"


def finalize_pending_sends(args, backend, pending_sends: dict, block: bool = False) -> bool:
    """
    ########################### 异步 PD producer 侧收尾 ###########################

    sync_gpu 异步 PD 中，process_request() 只负责提交 isend，不在原地等待。
    真正的 send 完成检查、metrics 写入、prefill_done 写入，都放到这里统一处理。

    为什么 prefill_done 必须在这里写：
    - prefill_done 的语义是 “KV handoff 已完成”。
    - submit_send() 只是提交异步 NCCL send，不代表数据已经传完。
    - 因此必须等 send_state 完成后，才能写 prefill_done。

    block=False:
    - 主循环中使用，只轮询已完成的 send，不阻塞 prefill 新请求。

    block=True:
    - worker 退出时使用，尽量把已经提交的 send 收尾完成，避免半路销毁进程组。
    """
    made_progress = False
    for base, item in list(pending_sends.items()):
        send_state = item["send_state"]
        if not block and not backend.transfer_done(send_state):
            continue

        wait_t0 = perf_counter()
        backend.wait_transfer(send_state)
        sync_cuda()
        wait_time_s = perf_counter() - wait_t0

        metrics = item["metrics"]
        send_complete_latency_s = perf_counter() - item["send_complete_t0"]
        metrics["send_submit_time_s"] = item["send_submit_time_s"]
        metrics["send_complete_latency_s"] = send_complete_latency_s
        metrics["send_finalize_wait_time_s"] = wait_time_s
        # 兼容旧 benchmark 字段：
        # transfer_time_s 在异步 PD producer 路径下表示 send 从提交后到确认完成的延迟。
        metrics["transfer_time_s"] = send_complete_latency_s
        metrics["total_time_s"] = perf_counter() - item["total_t0"]

        paths = item["paths"]
        atomic_write_json(paths["prefill_metrics"], metrics)
        paths["prefill_done"].write_text("done\n", encoding="utf-8")

        pending_sends.pop(base, None)
        made_progress = True

    return made_progress


def process_request(args, engine, backend, request_path: Path, seq_idx: int, control=None):
    """
    处理单条请求的完整 prefill 生命周期。

    shared_memory 模式：
      run_prefill()
      -> save_kv() 内部把 KV 写入 shared memory
      -> 写 payload.pkl
      -> 写 prefill_done
      -> decode 之后读取 shared memory

    sync_gpu 模式：
      run_prefill()
      -> save_kv() 内部只把 GPU KV 暂存在 producer backend._pending
      -> 写 payload.pkl
      -> 写 payload_ready
      -> 等 decode 写 recv_ready
      -> submit_send() 提交异步 isend
      -> 返回 pending_send，由主循环后台检查完成
      -> send 完成后写 prefill_done

    注意：
    - sync_gpu 模式下，不要在 payload_ready 之前 send。
    - sync_gpu 模式下，prefill_done 不能作为 decode 开始读取 payload 的信号；
      它表示 KV send/recv 已完成。
    """
    work_dir = Path(args.work_dir)
    base = request_base(request_path)
    paths = build_paths(work_dir, base)

    with request_path.open(encoding="utf-8") as f:
        request = json.load(f)

    prompt = request["prompt"]
    request_id = request.get("id", base)
    max_tokens = int(request.get("max_tokens", request.get("output_len", 2)))
    # max_tokens=1 会在 prefill 侧直接完成，没有 decode 阶段。
    # 常驻 PD benchmark 需要测 restore/decode，所以强制至少 2。
    max_tokens = max(2, max_tokens)

    total_t0 = perf_counter()
    # 这里是真正的 prefill 执行点。
    # KVConnector.save_kv 会在 PrefillEngine._make_payload 内部被调用，
    # 把当前 seq 对应的 KV blocks 导出到 shared memory，并在 payload 中保存 ref。
    payload = None
    meta = None
    handoff_id = None

    # sync_gpu 专用状态。
    # handoff_sent=False 表示 GPU KV 还只停留在 producer backend._pending 中；
    # 如果后面失败，需要 backend.delete(handoff_id) 清理这份 pending tensor。
    handoff_sent = False
    try:
        prefill_t0 = perf_counter()
        # 真正执行 prefill。
        # 这里内部会走：
        # PrefillEngine.run_prefill()
        # -> PrefillEngine._make_payload()
        # -> KVConnector.save_kv()
        #
        # shared_memory:
        #   save_kv() -> backend.put() 直接把 KV 写入 shared memory
        #
        # sync_gpu:
        #   save_kv() -> backend.put() 只把 GPU KV 暂存在 _pending
        payload = engine.run_prefill(
            texts=[prompt],
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
            start_seq_id=seq_idx,
        )[0]
        # 用 benchmark request_id 覆盖默认 req-{seq_idx}，方便结果文件和 dataset 对齐。
        payload.request_id = request_id
        
        sync_cuda()
        
        prefill_time_s = perf_counter() - prefill_t0
        payload_write_t0 = perf_counter()

        # payload.pkl 只保存轻量 metadata。
        # 真正大块 KV 不在 pkl 里：
        # - shared_memory: pkl 里是 SharedMemoryKVRef
        # - sync_gpu: pkl 里是 SyncGpuKVRef，真实 tensor 在 producer backend._pending
        atomic_write_pickle(paths["payload"], payload)
        payload_write_time_s = perf_counter() - payload_write_t0

        meta = payload.transfer_meta
        handoff_id = meta.handoff_id if meta is not None else None

        transfer_wait_time_s = 0.0
        transfer_time_s = 0.0
        send_submit_time_s = 0.0
        send_complete_latency_s = 0.0
        send_finalize_wait_time_s = 0.0
        pending_send = None
        
        if is_sync_gpu_backend(args):
            if meta is None:
                raise RuntimeError(
                    f"sync_gpu request {request_id} has no transfer_meta"
                )
            
            if control is None:
                raise RuntimeError("sync_gpu requires control connection")

            # 通知 decode：payload.pkl 已经写好，可以读取 metadata。
            control.send(
                {
                    "type": "payload_ready",
                    "base": base,
                    "request_id": request_id,
                    "handoff_id": handoff_id,
                }
            )
            
            # 写 payload.pkl
            # -> control.send(payload_ready)
            # -> control.recv(recv_ready)
            # -> flush_kv_transfer()
            # -> dist.send()
            # -> 写 prefill_done
            wait_t0 = perf_counter()
            while True:
                if (work_dir / args.shutdown_file).exists():
                    raise RuntimeError(
                        f"shutdown before recv_ready for {request_id}"
                    )

                if perf_counter() - wait_t0 > args.decode_wait_timeout_s:
                    raise TimeoutError(f"timed out waiting recv_ready for {request_id}")

                # 轮询非阻塞地检查 control 连接上是否有新消息
                # 有新消息是指decode端已经准备好传输
                if not control.poll(args.poll_interval_s):
                    continue

                try:
                    msg = control.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        f"control connection closed before recv_ready for {request_id}"
                    ) from exc
                if msg.get("type") == "recv_ready" and msg.get("base") == base:
                    break

                raise RuntimeError(f"unexpected control message for {request_id}: {msg}")

            transfer_wait_time_s = perf_counter() - wait_t0
            
            
            # ########################### 异步 PD producer 侧发送提交 ###########################
            # 这里是 producer pending_sends 的核心变化：
            # - 旧逻辑：submit_send() 后立刻 wait_transfer()，prefill 被卡住。
            # - 新逻辑：这里只提交 isend，把 send_state 返回给主循环。
            # - 主循环通过 finalize_pending_sends() 后台轮询完成，再写 prefill_done。
            #
            # 注意：send_state 里持有 kv_blocks / scale_blocks 引用。
            # 在 send 完成前不能释放，否则 NCCL 的发送 buffer 生命周期不安全。
            send_submit_t0 = perf_counter()
            send_state = backend.submit_send(handoff_id)
            send_submit_time_s = perf_counter() - send_submit_t0
            handoff_sent = True
            pending_send = {
                "send_state": send_state,
                "send_submit_time_s": send_submit_time_s,
                "send_complete_t0": perf_counter(),
                "total_t0": total_t0,
                "paths": paths,
            }
            # ########################### 异步 PD producer 侧发送提交 ###########################
        
        metrics = {
            # metrics 只记录 prefill 侧能准确观测到的内容：
            # prompt token 数、KV block 数、shared memory 大小、prefill 时间等。
            # decode restore/decode 时间由 persistent_decode_worker 单独记录。
            "role": "persistent_prefill",
            "request_id": request_id,
            "profile": request.get("profile"),
            "input_tokens_dataset": request.get("input_tokens"),
            "max_tokens": max_tokens,
            "total_tokens_dataset": request.get("total_tokens"),
            "payload_path": str(paths["payload"]),
            "payload_finished": payload.finished,
            "num_prompt_tokens": payload.num_prompt_tokens,
            "num_cached_tokens": payload.num_cached_tokens,
            "token_len": len(payload.token_ids),
            "num_kv_blocks": meta.num_kv_blocks if meta is not None else 0,
            "kv_nbytes": meta.storage_ref.nbytes if meta is not None and meta.storage_ref is not None else 0,
            "scale_nbytes": (
                meta.scale_storage_ref.nbytes
                if meta is not None and meta.scale_storage_ref is not None
                else getattr(meta.storage_ref, "scale_nbytes", 0)
                if meta is not None and meta.storage_ref is not None
                else 0
            ),
            "prefill_time_s": prefill_time_s,
            "payload_write_time_s": payload_write_time_s,
            "transfer_wait_time_s": transfer_wait_time_s,
            "send_submit_time_s": send_submit_time_s,
            "send_complete_latency_s": send_complete_latency_s,
            "send_finalize_wait_time_s": send_finalize_wait_time_s,
            "transfer_time_s": transfer_time_s,
            "total_time_s": perf_counter() - total_t0,
        }

        if pending_send is not None:
            # ########################### 异步 PD producer 侧延迟完成 ###########################
            # sync_gpu 异步路径不能在这里写 metrics / prefill_done。
            # 因为 isend 只是提交，真实传输完成要等主循环后台 finalize_pending_sends()。
            pending_send["metrics"] = metrics
            return base, handoff_id, pending_send
            # ########################### 异步 PD producer 侧延迟完成 ###########################
    
        atomic_write_json(paths["prefill_metrics"], metrics)
        # shared_memory:
        #   写到这里表示 payload + shared memory ref 都可用了。
        #
        # sync_gpu:
        #   写到这里表示 payload_ready/recv_ready/send 都完成了。
        paths["prefill_done"].write_text("done\n", encoding="utf-8")

    except Exception:
        # 这里主要保护 sync_gpu。
        #
        # 如果异常发生在 flush_kv_transfer() 之前：
        #   backend._pending 里还持有 GPU KV tensor。
        #   必须 delete，否则这块显存会残留到 worker 退出。
        #
        # 如果异常发生在 flush_kv_transfer() 之后：
        #   send_pending() 内部已经 pop 了 _pending。
        #   这里不再 delete，避免误删或混淆状态。
        if (
            is_sync_gpu_backend(args)
            and handoff_id is not None
            and not handoff_sent
        ):
            backend.delete(handoff_id)
        
        # 清理未完成的握手标记，避免 decode 看到旧文件后误处理。
        #
        # 注意：
        # - payload.pkl 可以不删，因为 error 文件会让 driver 报错退出。
        # - prefill_error 由 main() 外层 except 写，这里不要写，避免重复职责。
        for key in ("payload_ready", "recv_ready", "prefill_done"):
            paths[key].unlink(missing_ok=True)
        
        raise
    
    if args.wait_decode_done:
        wait_t0 = perf_counter()
        # 串行模式：prefill 进程等 decode_done。
        # 原因：shared memory 由 prefill 创建，如果 producer 进程提前退出或过早清理，
        # decode 侧可能 attach 失败。这里也保证一条请求的资源生命周期闭环。
        while not paths["decode_done"].exists():
            if (work_dir / args.shutdown_file).exists():
                break
            if perf_counter() - wait_t0 > args.decode_wait_timeout_s:
                raise TimeoutError(f"timed out waiting decode_done for {request_id}")
            time.sleep(args.poll_interval_s)

        if handoff_id is not None:
            # shared_memory:
            #   decode restore 时通常已经 pop_by_ref(unlink=True) 消费并 unlink。
            #   这里 delete 主要清理 producer backend._records metadata。
            #
            # sync_gpu:
            #   send_pending() 已经 pop _pending。
            #   这里 delete 是 no-op 级别的兜底清理。
            backend.delete(handoff_id)

    return base, handoff_id, None


def cleanup_completed_handoffs(work_dir: Path, backend, pending_handoffs: dict[str, str]) -> None:
    # Pipeline 模式使用：
    # prefill 不阻塞等待 decode_done，因此 producer backend._records 会暂时保存
    # handoff_id -> SharedMemoryKVRef metadata。
    # 一旦发现某个 base 的 decode_done，说明 decode 已经 restore 并 unlink 了 shared memory，
    # 这里就可以清理 producer 侧 metadata。这个操作不影响 decode 已经拿到的 KV。
    for base, handoff_id in list(pending_handoffs.items()):
        paths = build_paths(work_dir, base)
        if not paths["decode_done"].exists():
            continue
        backend.delete(handoff_id)
        pending_handoffs.pop(base, None)


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent prefill worker for local PD benchmark.")
    parser.add_argument("--model-path", default="/home/xhk/model/Qwen3-0.6B/")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--ready-file", default="prefill_worker.ready.json")
    parser.add_argument("--shutdown-file", default="shutdown")
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--decode-wait-timeout-s", type=float, default=300.0)
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument(
        "--wait-decode-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait each request's decode_done before processing the next request.",
    )
    parser.add_argument(
        "--kv-transfer-backend",
        default="shared_memory",
        choices=["shared_memory", "sync_gpu"],
        help="KV transfer backend. sync_gpu uses payload_ready/recv_ready handshake.",
    )
    parser.add_argument(
        "--max-pending-sends",
        type=int,
        default=4,
        help="Max in-flight async PD producer sends before applying backpressure.",
    )
    parser.add_argument("--nccl-port", default="29577")
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    if args.kv_transfer_backend == "sync_gpu":
        torch.cuda.set_device(0)
        # 这里初始化 distributed
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{args.nccl_port}",
            rank=0,
            world_size=2,
        )
        backend = SyncGpuKVStoreBackend(rank=0, peer_rank=1)  # prefill
    else:
        backend = SharedMemoryKVStoreBackend()

    print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("torch_current_device", torch.cuda.current_device(), flush=True)
    print("torch_device_name", torch.cuda.get_device_name(0), flush=True)


    init_t0 = perf_counter()
    # 常驻 worker 的核心收益在这里：
    # tokenizer/model/runner/engine 只初始化一次，后续所有请求复用同一套对象。
    config = build_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    runner = ModelRunner(config)
    
    connector = KVConnector(
        config=config,
        role="producer",
        engine_id="persistent-prefill-worker-0",
        kv_store_backend=backend,
    )
    connector.register_model_runner(runner)
    engine = PrefillEngine(
        config=config,
        tokenizer=tokenizer,
        model_runner=runner,
        kv_connector=connector,
    )
    sync_cuda()
    init_time_s = perf_counter() - init_t0
    
    control = None
    if args.kv_transfer_backend == "sync_gpu":
        control = connect_control(
        control_socket_path(work_dir),
        timeout_s=args.decode_wait_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    

    atomic_write_json(
        work_dir / args.ready_file,
        {
            "role": "persistent_prefill",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_current_device": int(torch.cuda.current_device()),
            "torch_device_name": torch.cuda.get_device_name(0),
            "kv_cache_quant_mode": args.kv_cache_quant_mode,
            "kv_transfer_backend": args.kv_transfer_backend,
            "control_connected": control is not None,
            "max_pending_sends": args.max_pending_sends,
            "model_init_time_s": init_time_s,
        },
    )
    # ready 文件告诉 benchmark driver：prefill worker 已完成模型加载，可以开始投递请求。
    print("persistent_prefill_ready", work_dir / args.ready_file, flush=True)

    seq_idx = 0
    pending_handoffs = {}
    # ########################### 异步 PD producer pending_sends ###########################
    # sync_gpu 异步 PD 中，prefill 提交 isend 后不会阻塞等待传输完成。
    # 每条尚未完成的 send 都保存在 pending_sends 中：
    # - key: request base
    # - value: send_state / metrics / paths / timing
    #
    # 主循环每轮调用 finalize_pending_sends() 轮询完成状态。
    # send 完成后才写 prefill_metrics 和 prefill_done。
    pending_sends = {}
    # ########################### 异步 PD producer pending_sends ###########################
    try:
        # 主循环：只要没有 shutdown 文件，就持续扫描新请求。
        # 串行 benchmark 下，每条请求会等 decode_done。
        # pipeline benchmark 下，--no-wait-decode-done 会让 prefill 连续处理 request，
        # decode worker 同时消费 prefill_done，从而形成 P/D overlap。
        while not (work_dir / args.shutdown_file).exists():
            made_progress = finalize_pending_sends(
                args,
                backend,
                pending_sends,
                block=False,
            )
            cleanup_completed_handoffs(work_dir, backend, pending_handoffs)
            request_paths = sorted(work_dir.glob("*.request.json"))
            for request_path in request_paths:
                # ########################### 异步 PD producer pending_sends ###########################
                # request_paths 可能一次性包含很多请求。
                # 如果只在 while 顶部 finalize，已完成的 send 可能要等这一整批 prefill
                # 都处理完后才写 prefill_done，导致 transfer_time_s 被人为拉长。
                # 因此每处理新请求前先轻量轮询一次 pending_sends。
                made_progress = finalize_pending_sends(
                    args,
                    backend,
                    pending_sends,
                    block=False,
                ) or made_progress
                # ########################### 异步 PD producer pending_sends ###########################

                if (
                    is_sync_gpu_backend(args)
                    and len(pending_sends) >= args.max_pending_sends
                ):
                    # ########################### 异步 PD producer backpressure ###########################
                    # producer 侧已经有太多未完成 isend。
                    # 这里停止继续处理新的 request，回到 while 顶部优先 finalize pending_sends。
                    # 这样可以避免 prefill 过快生产，导致 send buffer / NCCL work 无限制堆积。
                    break
                    # ########################### 异步 PD producer backpressure ###########################

                base = request_base(request_path)
                paths = build_paths(work_dir, base)
                # 已完成或已失败的请求不重复处理。
                if (
                    base in pending_sends
                    or paths["prefill_done"].exists()
                    or paths["prefill_error"].exists()
                ):
                    continue
                made_progress = True
                try:
                    base, handoff_id, pending_send = process_request(
                        args,
                        engine,
                        backend,
                        request_path,
                        seq_idx,
                        control=control,
                    )
                    if pending_send is not None:
                        # ########################### 异步 PD producer pending_sends ###########################
                        # sync_gpu 异步路径：send 已提交但未完成。
                        # 这里先放入 pending_sends；后续 finalize_pending_sends()
                        # 检测完成后再写 prefill_done。
                        pending_sends[base] = pending_send
                        # ########################### 异步 PD producer pending_sends ###########################
                    if not args.wait_decode_done and handoff_id is not None:
                        pending_handoffs[base] = handoff_id
                    seq_idx += 1

                    # ########################### 异步 PD producer pending_sends ###########################
                    # 当前请求可能刚提交了 isend；立刻轮询一次。
                    # 如果传输已经完成，可以马上写 prefill_done，减少控制面可见延迟。
                    made_progress = finalize_pending_sends(
                        args,
                        backend,
                        pending_sends,
                        block=False,
                    ) or made_progress
                    # ########################### 异步 PD producer pending_sends ###########################
                except Exception as exc:
                    # 异常不让 worker 直接崩掉，而是写 error 文件给 driver。
                    # driver 等待 decode_done 时会同时检查 error 文件并抛出可读错误。
                    atomic_write_json(
                        paths["prefill_error"],
                        {
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"prefill_error {request_path}: {exc}", flush=True)
            if not made_progress:
                time.sleep(args.poll_interval_s)
    finally:
        if pending_sends:
            finalize_pending_sends(
                args,
                backend,
                pending_sends,
                block=True,
            )
        cleanup_completed_handoffs(work_dir, backend, pending_handoffs)
        if control is not None:
            control.close()
        if args.kv_transfer_backend == "sync_gpu" and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        print("persistent_prefill_shutdown", flush=True)


if __name__ == "__main__":
    main()
