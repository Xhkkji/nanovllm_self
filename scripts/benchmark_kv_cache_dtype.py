import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from nanovllm.config import Config
from pd_self.online_coordinator import OnlinePDCoordinator


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"

PROMPTS = {
    "short": "What is a large language model?",
    "medium": (
        "Explain how a transformer model works, including embeddings, "
        "self-attention, feed-forward networks, and autoregressive decoding."
    ),
}

DTYPE_CASES = {
    "bf16": ("bf16", "bf16"),
    "fp16": ("fp16", "fp16"),
}


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def kv_cache_memory_mb(model_runner) -> float:
    kv_cache = model_runner.kv_cache
    return kv_cache.numel() * kv_cache.element_size() / (1024 ** 2)


def make_config(kv_cache_dtype: str, attention_compute_dtype: str, batch_size: int):
    return Config(
        model_path=MODEL_PATH,
        device="cuda:0",
        max_num_seqs=max(16, batch_size),
        max_num_batched_tokens=4096,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        block_size=256,
        num_blocks=256,
        kv_cache_dtype=kv_cache_dtype,
        attention_compute_dtype=attention_compute_dtype,
    )


def submit_batch(engine, prompt_name: str, batch_size: int, gen_len: int):
    prompt = PROMPTS[prompt_name]
    return [
        engine.submit(
            prompt,
            max_tokens=gen_len,
            temperature=0.0,
            ignore_eos=True,
        )
        for _ in range(batch_size)
    ]


def drain_until_finished(engine, request_ids, max_steps=10000):
    for _ in range(max_steps):
        if all(engine.is_finished(request_id) for request_id in request_ids):
            return
        engine.step()
    raise RuntimeError("online engine did not finish within max_steps")


def run_warmup(engine, prompt_name: str, batch_size: int):
    request_ids = submit_batch(engine, prompt_name, batch_size, gen_len=2)
    with torch.inference_mode():
        drain_until_finished(engine, request_ids)
    cuda_sync()


def run_timed_case(prompt_name: str, batch_size: int, gen_len: int, dtype_name: str):
    kv_cache_dtype, attention_compute_dtype = DTYPE_CASES[dtype_name]
    config = make_config(kv_cache_dtype, attention_compute_dtype, batch_size)
    engine = OnlinePDCoordinator(config, kv_backend="dict")

    try:
        run_warmup(engine, prompt_name, batch_size)

        request_ids = submit_batch(engine, prompt_name, batch_size, gen_len)
        prompt_tokens = sum(len(engine.requests[request_id].input_ids) for request_id in request_ids)
        first_token_sec = {}

        cuda_sync()
        t0 = perf_counter()
        with torch.inference_mode():
            for _ in range(10000):
                if all(engine.is_finished(request_id) for request_id in request_ids):
                    break

                events = engine.step()
                cuda_sync()
                now = perf_counter()

                for event in events:
                    if event.token_id is None:
                        continue
                    first_token_sec.setdefault(event.request_id, now - t0)
            else:
                raise RuntimeError("online engine did not finish within max_steps")

        cuda_sync()
        total_time_s = perf_counter() - t0

        generated_tokens = sum(len(engine.requests[request_id].output_ids) for request_id in request_ids)
        ttft_values = [first_token_sec[request_id] for request_id in request_ids]
        avg_ttft_s = sum(ttft_values) / len(ttft_values)
        max_ttft_s = max(ttft_values)
        decode_time_s = max(total_time_s - avg_ttft_s, 1e-9)
        inter_token_count = max(generated_tokens - batch_size, 0)
        itl_ms = (decode_time_s / inter_token_count) * 1000.0 if inter_token_count > 0 else 0.0

        prefill_kv_mb = kv_cache_memory_mb(engine.prefill_engine.model_runner)
        decode_kv_mb = kv_cache_memory_mb(engine.decode_engine.model_runner)
        outputs = [engine.requests[request_id].token_ids for request_id in request_ids]

        return {
            "dtype_name": dtype_name,
            "kv_cache_dtype": str(config.kv_cache_dtype),
            "attention_compute_dtype": str(config.attention_compute_dtype),
            "dtype_nbytes": dtype_nbytes(config.kv_cache_dtype),
            "prompt_name": prompt_name,
            "batch_size": batch_size,
            "gen_len": gen_len,
            "block_size": config.block_size,
            "num_blocks": config.num_blocks,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "avg_ttft_ms": avg_ttft_s * 1000.0,
            "max_ttft_ms": max_ttft_s * 1000.0,
            "itl_ms": itl_ms,
            "throughput_tok_s": generated_tokens / total_time_s if total_time_s > 0 else 0.0,
            "decode_tok_s": inter_token_count / decode_time_s if decode_time_s > 0 else 0.0,
            "total_time_s": total_time_s,
            "prefill_kv_cache_mb": prefill_kv_mb,
            "decode_kv_cache_mb": decode_kv_mb,
            "pd_total_kv_cache_mb": prefill_kv_mb + decode_kv_mb,
            "outputs": outputs,
        }
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_result(row):
    print(
        f"[{row['dtype_name']}] prompt={row['prompt_name']} "
        f"bs={row['batch_size']} gen={row['gen_len']} "
        f"block={row['block_size']}"
    )
    print(
        f"  avg_TTFT(ms)={row['avg_ttft_ms']:.2f} "
        f"max_TTFT(ms)={row['max_ttft_ms']:.2f} "
        f"ITL(ms)={row['itl_ms']:.2f} "
        f"decode_tok/s={row['decode_tok_s']:.2f} "
        f"throughput_tok/s={row['throughput_tok_s']:.2f}"
    )
    print(
        f"  prompt_tokens={row['prompt_tokens']} "
        f"generated_tokens={row['generated_tokens']} "
        f"total_time_s={row['total_time_s']:.4f} "
        f"pd_kv_cache_mb={row['pd_total_kv_cache_mb']:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="KV cache dtype benchmark matrix for online PD.")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=["short", "medium"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--dtypes", nargs="+", choices=list(DTYPE_CASES), default=["bf16", "fp16"])
    parser.add_argument("--output", default="logs/kv_cache_dtype_benchmark_20260725.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with output_path.open("w", encoding="utf-8") as f:
        for prompt_name in args.prompts:
            for batch_size in args.batch_sizes:
                for dtype_name in args.dtypes:
                    row = run_timed_case(prompt_name, batch_size, args.gen_len, dtype_name)
                    rows.append(row)
                    print_result(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()

    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
