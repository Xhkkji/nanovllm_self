import argparse
import json
import os
import pickle
import sys
from time import perf_counter

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
from pd_self.kv_store import SharedMemoryKVStoreBackend


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_config(args):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/xhk/model/Qwen3-0.6B/")
    parser.add_argument("--infile", default="/tmp/nanovllm_pd_payload.pkl")
    parser.add_argument("--done-file", default="/tmp/nanovllm_pd_decode.done")
    parser.add_argument("--metrics-out", default="pd_self/multiprocess/result/dual_gpu_pd_decode_metrics.json")
    parser.add_argument("--kv-cache-quant-mode", default="none", choices=["none", "int8_mock"])
    parser.add_argument("--run-to-finish", action="store_true")
    args = parser.parse_args()

    total_t0 = perf_counter()
    config = build_config(args)
    print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch_current_device", torch.cuda.current_device())
    print("torch_device_name", torch.cuda.get_device_name(0))

    init_t0 = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    runner = ModelRunner(config)

    # 注意：这里新建 backend 是故意的。
    # decode 侧通过 payload 里的 SharedMemoryKVRef attach，不依赖 producer 进程的 _records。
    backend = SharedMemoryKVStoreBackend()

    connector = KVConnector(
        config=config,
        role="consumer",
        engine_id="decode-worker-0",
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

    payload_read_t0 = perf_counter()
    with open(args.infile, "rb") as f:
        payload = pickle.load(f)
    payload_read_time_s = perf_counter() - payload_read_t0

    with torch.inference_mode():
        restore_t0 = perf_counter()
        results, restored = engine.restore_payloads([payload])
        sync_cuda()
        restore_time_s = perf_counter() - restore_t0
        print("restore_results", results)
        print("restored_count", len(restored))

        decode_t0 = perf_counter()
        decode_steps = 0
        decode_step_tokens = []
        decode_finished_ids = []

        if args.run_to_finish:
            empty_steps = 0
            max_empty_steps = 10000
            while not engine.scheduler.is_finished():
                out = engine.step()
                sync_cuda()
                if not out.scheduled:
                    empty_steps += 1
                    if empty_steps > max_empty_steps:
                        raise RuntimeError("decode worker made no progress")
                    continue
                empty_steps = 0
                decode_steps += 1
                decode_step_tokens.extend(out.token_ids)
                decode_finished_ids.extend(out.finished_seq_ids)
        else:
            out = engine.step()
            sync_cuda()
            decode_steps = 1 if out.scheduled else 0
            decode_step_tokens = list(out.token_ids)
            decode_finished_ids = list(out.finished_seq_ids)

        decode_step_time_s = perf_counter() - decode_t0
        final_token_lens = [len(seq.token_ids) for seq in restored]
        generated_tokens = sum(
            max(0, len(seq.token_ids) - seq.num_prompt_tokens) for seq in restored
        )
        print("decode_steps", decode_steps)
        print("decode_step_tokens", decode_step_tokens)
        print("decode_finished_ids", decode_finished_ids)

    done_write_t0 = perf_counter()
    with open(args.done_file, "w") as f:
        f.write("done\n")
    done_write_time_s = perf_counter() - done_write_t0

    meta = payload.transfer_meta
    metrics = {
        "role": "decode",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_current_device": int(torch.cuda.current_device()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "kv_cache_quant_mode": args.kv_cache_quant_mode,
        "payload_path": args.infile,
        "done_file": args.done_file,
        "request_id": payload.request_id,
        "run_to_finish": args.run_to_finish,
        "num_kv_blocks": meta.num_kv_blocks if meta is not None else 0,
        "kv_nbytes": meta.storage_ref.nbytes if meta is not None and meta.storage_ref is not None else 0,
        "scale_nbytes": meta.scale_storage_ref.nbytes if meta is not None and meta.scale_storage_ref is not None else 0,
        "model_init_time_s": init_time_s,
        "payload_read_time_s": payload_read_time_s,
        "restore_time_s": restore_time_s,
        "decode_step_time_s": decode_step_time_s,
        "done_write_time_s": done_write_time_s,
        "total_time_s": perf_counter() - total_t0,
        "restored_count": len(restored),
        "decode_steps": decode_steps,
        "decode_step_tokens": decode_step_tokens,
        "decode_finished_ids": decode_finished_ids,
        "generated_tokens": generated_tokens,
        "final_token_lens": final_token_lens,
    }
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("metrics_written", args.metrics_out)


if __name__ == "__main__":
    main()
