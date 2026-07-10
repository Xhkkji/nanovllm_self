import argparse
import json
from pathlib import Path

from benchmark_runtime import run_case


EXACT_BUCKET_CASES = [
    ("short", 1, 64),
    ("short", 2, 64),
    ("medium", 4, 64),
    ("medium", 8, 64),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exact-bucket CUDA Graph on/off comparisons for bs in [1,2,4,8]."
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--output",
        default="/home/xhk/nanovllm_self/results/cuda_graph_exact_buckets.jsonl",
        help="Optional jsonl output path.",
    )
    return parser.parse_args()


def print_case_header(prompt_name, batch_size, gen_len):
    print(f"=== Case: prompt={prompt_name} bs={batch_size} gen={gen_len} ===")


def print_metrics(label, metrics):
    print(
        f"{label}: "
        f"TTFT(ms)={metrics['ttft_ms']:.2f} "
        f"ITL(ms)={metrics['itl_ms']:.2f} "
        f"decode_tok/s={metrics['decode_tok_s']:.2f} "
        f"throughput_tok/s={metrics['throughput_tok_s']:.2f}"
    )


def compute_delta(base, graph):
    def pct(old, new, larger_is_better):
        if old == 0:
            return 0.0
        if larger_is_better:
            return (new - old) / old * 100.0
        return (old - new) / old * 100.0

    return {
        "itl_gain_pct": pct(base["itl_ms"], graph["itl_ms"], larger_is_better=False),
        "decode_gain_pct": pct(base["decode_tok_s"], graph["decode_tok_s"], larger_is_better=True),
        "throughput_gain_pct": pct(base["throughput_tok_s"], graph["throughput_tok_s"], larger_is_better=True),
    }


def append_jsonl(path, row):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for prompt_name, batch_size, gen_len in EXACT_BUCKET_CASES:
        print_case_header(prompt_name, batch_size, gen_len)

        base = run_case(
            prompt_name=prompt_name,
            batch_size=batch_size,
            gen_len=gen_len,
            prefill_backend="torch",
            decode_backend="flashattn",
            cuda_graph=False,
            warmup_runs=args.warmup_runs,
        )
        graph = run_case(
            prompt_name=prompt_name,
            batch_size=batch_size,
            gen_len=gen_len,
            prefill_backend="torch",
            decode_backend="flashattn",
            cuda_graph=True,
            warmup_runs=args.warmup_runs,
        )
        delta = compute_delta(base, graph)

        print_metrics("baseline", base)
        print_metrics("graph   ", graph)
        print(
            "delta   : "
            f"ITL {delta['itl_gain_pct']:+.1f}% | "
            f"decode {delta['decode_gain_pct']:+.1f}% | "
            f"throughput {delta['throughput_gain_pct']:+.1f}%"
        )
        print()

        append_jsonl(
            output_path,
            {
                "prompt_name": prompt_name,
                "batch_size": batch_size,
                "gen_len": gen_len,
                "warmup_runs": args.warmup_runs,
                "baseline": base,
                "graph": graph,
                "delta": delta,
            },
        )

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
