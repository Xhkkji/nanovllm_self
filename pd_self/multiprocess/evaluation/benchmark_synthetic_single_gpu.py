import argparse
import os
import sys
from time import perf_counter

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT_DIR)

from benchmark_synthetic_common import (
    DEFAULT_DATASET,
    cap_max_tokens,
    default_metrics_path,
    default_summary_path,
    select_requests,
    summarize,
    write_json,
    write_jsonl,
)
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def ensure_attention_backend_labels(llm):
    for layer in llm.model_runner.model.layers:
        attn = layer.p_attn
        if not hasattr(attn, "prefill_backend"):
            attn.prefill_backend = "default"
        if not hasattr(attn, "decode_backend"):
            attn.decode_backend = "default"


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU synthetic serving benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output = args.output or default_metrics_path("single_gpu", args.profile)
    args.summary_output = args.summary_output or default_summary_path("single_gpu", args.profile)
    requests = select_requests(
        args.dataset,
        limit=args.limit + args.warmup,
        profile=args.profile,
        max_total_tokens=args.max_total_tokens,
    )
    if not requests:
        raise RuntimeError("no benchmark requests selected")

    init_t0 = perf_counter()
    llm = LLM_self(enable_profile=False)
    ensure_attention_backend_labels(llm)
    sync_cuda()
    model_init_time_s = perf_counter() - init_t0

    metrics_rows = []
    for idx, row in enumerate(requests):
        phase = "warmup" if idx < args.warmup else "measure"
        measure_index = idx - args.warmup if phase == "measure" else None
        max_tokens = cap_max_tokens(row, args.max_output_tokens_cap)
        input_ids = llm.tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"]
        sampling_params = [
            SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        ]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sync_cuda()
        t0 = perf_counter()
        result = llm.generate(input_ids, sampling_params=sampling_params, return_metrics=True)
        sync_cuda()
        e2e_time_s = perf_counter() - t0

        engine_metrics = result["metrics"]
        generated_tokens = int(engine_metrics.get("generated_tokens", 0))
        core_e2e_time_s = float(engine_metrics.get("total_time_s", e2e_time_s))
        metrics_rows.append(
            {
                "mode": "single_gpu",
                "phase": phase,
                "request_index": idx,
                "measure_index": measure_index,
                "request_id": row.get("id"),
                "profile": row.get("profile"),
                "input_tokens_dataset": row.get("input_tokens"),
                "input_tokens_actual": int(input_ids.numel()),
                "max_tokens": max_tokens,
                "target_output_tokens": row.get("max_tokens", row.get("output_len")),
                "total_tokens_dataset": row.get("total_tokens"),
                "model_init_time_s": model_init_time_s if idx == 0 else 0.0,
                "core_e2e_time_s": core_e2e_time_s,
                "wall_e2e_time_s": e2e_time_s,
                "generated_tokens": generated_tokens,
                "throughput_generated_tok_s": generated_tokens / core_e2e_time_s
                if core_e2e_time_s > 0
                else 0.0,
                "peak_memory_mb": peak_memory_mb(),
                "engine_metrics": engine_metrics,
            }
        )
        print(
            f"[single][{phase}] {idx + 1}/{len(requests)} id={row.get('id')} "
            f"profile={row.get('profile')} generated={generated_tokens} "
            f"core={core_e2e_time_s:.4f}s wall={e2e_time_s:.4f}s"
        )

    measured_rows = [row for row in metrics_rows if row.get("phase") == "measure"]
    summary = summarize(measured_rows, ["core_e2e_time_s", "wall_e2e_time_s"])
    summary.update(
        {
            "mode": "single_gpu",
            "dataset": args.dataset,
            "limit": args.limit,
            "warmup": args.warmup,
            "total_selected_requests": len(requests),
            "measured_requests": len(measured_rows),
            "profile": args.profile,
            "max_total_tokens": args.max_total_tokens,
            "max_output_tokens_cap": args.max_output_tokens_cap,
            "model_init_time_s": model_init_time_s,
            "result_dir": os.path.dirname(args.summary_output),
        }
    )
    write_jsonl(args.output, metrics_rows)
    write_json(args.summary_output, summary)
    print("metrics_written", args.output)
    print("summary_written", args.summary_output)


if __name__ == "__main__":
    main()
