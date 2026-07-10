import argparse
import json
from pathlib import Path

from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams


PROMPTS = {
    "short": "What is a large language model?",
    "medium": (
        "Explain how a transformer model works, including embeddings, "
        "self-attention, feed-forward networks, and autoregressive decoding."
    ),
    "long": (
        "Write a clear technical overview of large language model inference systems. "
        "Cover tokenization, embeddings, rotary position encoding, attention, KV cache, "
        "prefill, decode, batching, prefix caching, scheduler design, and GPU kernels. "
        "Then explain the tradeoff between latency and throughput in serving systems."
    ),
}


def set_backends(llm, prefill_backend, decode_backend):
    for layer in llm.model_runner.model.layers:
        layer.p_attn.prefill_backend = prefill_backend
        layer.p_attn.decode_backend = decode_backend


def set_cuda_graph(llm, enabled):
    llm.model_runner.enable_cuda_graph = enabled


def make_inputs(tokenizer, prompt_name, batch_size):
    prompt = PROMPTS[prompt_name]
    texts = [prompt] * batch_size
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    return encoded["input_ids"]


def run_case(prompt_name, batch_size, gen_len, prefill_backend, decode_backend, cuda_graph, warmup_runs=0):
    llm = LLM_self(enable_profile=False)
    set_backends(llm, prefill_backend, decode_backend)
    set_cuda_graph(llm, cuda_graph)
    inputs = make_inputs(llm.tokenizer, prompt_name, batch_size)
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=gen_len, ignore_eos=True)
        for _ in range(batch_size)
    ]

    # 预热运行不计入最终结果。
    # 对 CUDA Graph 路径来说，这一步会把 graph warmup / capture 成本提前吃掉，
    # 正式返回的指标更接近 steady-state。
    for _ in range(warmup_runs):
        llm.generate(inputs, sampling_params=sampling_params, return_metrics=False)

    result = llm.generate(inputs, sampling_params=sampling_params, return_metrics=True)
    metrics = result["metrics"]
    metrics["prompt_name"] = prompt_name
    metrics["batch_size"] = batch_size
    metrics["gen_len"] = gen_len
    metrics["cuda_graph"] = cuda_graph
    metrics["warmup_runs"] = warmup_runs
    return metrics


def print_result(metrics):
    print(
        f"[{metrics['prompt_name']}] "
        f"bs={metrics['batch_size']} "
        f"gen={metrics['gen_len']} "
        f"prefill={metrics['prefill_backend']} "
        f"decode={metrics['decode_backend']} "
        f"warmup_runs={metrics['warmup_runs']} "
        f"cuda_graph={'on' if metrics['cuda_graph'] else 'off'}"
    )
    print(
        f"  TTFT(ms)={metrics['ttft_ms']:.2f} "
        f"ITL(ms)={metrics['itl_ms']:.2f} "
        f"prefill_tok/s={metrics['prefill_tok_s']:.2f} "
        f"decode_tok/s={metrics['decode_tok_s']:.2f} "
        f"throughput_tok/s={metrics['throughput_tok_s']:.2f}"
    )
    print(
        f"  prompt_tokens={metrics['prompt_tokens']} "
        f"generated_tokens={metrics['generated_tokens']} "
        f"total_time_s={metrics['total_time_s']:.4f}"
    )
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal structured benchmark for nanovllm_self.")
    parser.add_argument("--prompt", choices=list(PROMPTS.keys()), default="medium")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--prefill-backend", choices=["torch", "flashattn"], default="torch")
    parser.add_argument("--decode-backend", choices=["torch", "flashattn"], default="flashattn")
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="off")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional jsonl output path")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_case(
        prompt_name=args.prompt,
        batch_size=args.batch_size,
        gen_len=args.gen_len,
        prefill_backend=args.prefill_backend,
        decode_backend=args.decode_backend,
        cuda_graph=(args.cuda_graph == "on"),
        warmup_runs=args.warmup_runs,
    )
    print_result(metrics)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
