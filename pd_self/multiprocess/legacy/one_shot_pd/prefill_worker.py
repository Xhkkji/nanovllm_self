import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from time import perf_counter

# Legacy one-shot prefill worker。
# 当前主线使用 persistent_prefill_worker.py；该文件只作为早期 demo 归档保留。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer
import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from pd_self.kv_connector import KVConnector
from pd_self.kv_store import SharedMemoryKVStoreBackend
from pd_self.prefill_engine import PrefillEngine


def sync_cuda():
    """在有 CUDA 时同步当前设备，用于让 prefill/写 payload 的计时更接近真实 GPU 时间。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_config(args):
    """构造 prefill worker 使用的最小 nano-vLLM Config，并保证与 decode 侧 KV 配置一致。"""
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
    
    
def main():
    """短生命周期 prefill demo 入口：加载请求，执行 prefill，把 KV payload 写给 decode 侧。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/xhk/model/Qwen3-0.6B/")
    parser.add_argument("--prompt", default="What is KV cache?")
    parser.add_argument("--request-json", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--out", default="/tmp/nanovllm_pd_payload.pkl")
    parser.add_argument("--done-file", default="/tmp/nanovllm_pd_decode.done")
    parser.add_argument("--metrics-out", default="pd_self/multiprocess/result/dual_gpu_pd_prefill_metrics.json")
    parser.add_argument("--kv-cache-quant-mode", default="none", choices=["none", "int8_mock"])
    args = parser.parse_args()

    request_meta = {}
    if args.request_json:
        with open(args.request_json, encoding="utf-8") as f:
            request_meta = json.load(f)
        args.prompt = request_meta["prompt"]
        args.request_id = request_meta.get("id", args.request_id)
        args.max_tokens = int(
            request_meta.get("max_tokens", request_meta.get("output_len", args.max_tokens))
        )

    total_t0 = perf_counter()
    if os.path.exists(args.done_file):
        os.remove(args.done_file)

    config = build_config(args)
    print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch_current_device", torch.cuda.current_device())
    print("torch_device_name", torch.cuda.get_device_name(0))

    init_t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    runner = ModelRunner(config)
    backend = SharedMemoryKVStoreBackend()

    connector = KVConnector(
        config=config,
        role="producer",
        engine_id="prefill-worker-0",
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

    prefill_t0 = perf_counter()
    payload = engine.run_prefill(
        texts=[args.prompt],
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        start_seq_id=0,
    )[0]
    if args.request_id is not None:
        payload.request_id = args.request_id
    sync_cuda()
    prefill_time_s = perf_counter() - prefill_t0

    payload_write_t0 = perf_counter()
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    payload_write_time_s = perf_counter() - payload_write_t0

    meta = payload.transfer_meta
    print("payload_written", args.out)
    print("storage_ref", type(meta.storage_ref).__name__ if meta and meta.storage_ref else None)
    print("scale_storage_ref", type(meta.scale_storage_ref).__name__ if meta and meta.scale_storage_ref else None)
    if meta is not None and meta.storage_ref is not None:
        print("kv_nbytes", meta.storage_ref.nbytes)
    if meta is not None and meta.scale_storage_ref is not None:
        print("scale_nbytes", meta.scale_storage_ref.nbytes)

    # 保持 producer 存活，避免 Python resource_tracker 在 decode attach 前清理 shm。
    print("waiting_decode_done", args.done_file)
    wait_t0 = perf_counter()
    while not os.path.exists(args.done_file):
        time.sleep(0.2)
    wait_decode_time_s = perf_counter() - wait_t0

    metrics = {
        "role": "prefill",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_current_device": int(torch.cuda.current_device()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "kv_cache_quant_mode": args.kv_cache_quant_mode,
        "payload_path": args.out,
        "done_file": args.done_file,
        "request_id": args.request_id,
        "profile": request_meta.get("profile"),
        "input_tokens_dataset": request_meta.get("input_tokens"),
        "max_tokens": args.max_tokens,
        "total_tokens_dataset": request_meta.get("total_tokens"),
        "payload_finished": payload.finished,
        "num_prompt_tokens": payload.num_prompt_tokens,
        "num_cached_tokens": payload.num_cached_tokens,
        "token_len": len(payload.token_ids),
        "num_kv_blocks": meta.num_kv_blocks if meta is not None else 0,
        "kv_nbytes": meta.storage_ref.nbytes if meta is not None and meta.storage_ref is not None else 0,
        "scale_nbytes": meta.scale_storage_ref.nbytes if meta is not None and meta.scale_storage_ref is not None else 0,
        "model_init_time_s": init_time_s,
        "prefill_time_s": prefill_time_s,
        "payload_write_time_s": payload_write_time_s,
        "wait_decode_time_s": wait_decode_time_s,
        "total_time_s": perf_counter() - total_t0,
    }
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("metrics_written", args.metrics_out)


if __name__ == "__main__":
    main()    
