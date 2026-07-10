import argparse

from benchmark_runtime import print_result, run_case


def parse_args():
    parser = argparse.ArgumentParser(description="Compare flash decode with and without CUDA Graph.")
    parser.add_argument("--prompt", choices=["short", "medium", "long"], default="short")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    base = run_case(
        prompt_name=args.prompt,
        batch_size=args.batch_size,
        gen_len=args.gen_len,
        prefill_backend="torch",
        decode_backend="flashattn",
        cuda_graph=False,
        warmup_runs=args.warmup_runs,
    )
    graph = run_case(
        prompt_name=args.prompt,
        batch_size=args.batch_size,
        gen_len=args.gen_len,
        prefill_backend="torch",
        decode_backend="flashattn",
        cuda_graph=True,
        warmup_runs=args.warmup_runs,
    )

    print("=== Baseline (steady-state) ===")
    print_result(base)
    print("=== CUDA Graph (steady-state) ===")
    print_result(graph)

    if base["itl_ms"] > 0:
        itl_gain = (base["itl_ms"] - graph["itl_ms"]) / base["itl_ms"] * 100.0
    else:
        itl_gain = 0.0

    if base["decode_tok_s"] > 0:
        decode_gain = (graph["decode_tok_s"] - base["decode_tok_s"]) / base["decode_tok_s"] * 100.0
    else:
        decode_gain = 0.0

    if base["throughput_tok_s"] > 0:
        throughput_gain = (graph["throughput_tok_s"] - base["throughput_tok_s"]) / base["throughput_tok_s"] * 100.0
    else:
        throughput_gain = 0.0

    print("=== Delta ===")
    print(
        f"ITL: {base['itl_ms']:.2f} ms -> {graph['itl_ms']:.2f} ms "
        f"({itl_gain:+.1f}%)"
    )
    print(
        f"decode_tok/s: {base['decode_tok_s']:.2f} -> {graph['decode_tok_s']:.2f} "
        f"({decode_gain:+.1f}%)"
    )
    print(
        f"throughput_tok/s: {base['throughput_tok_s']:.2f} -> {graph['throughput_tok_s']:.2f} "
        f"({throughput_gain:+.1f}%)"
    )
    print(f"warmup_runs discarded before timing: {args.warmup_runs}")


if __name__ == "__main__":
    main()
