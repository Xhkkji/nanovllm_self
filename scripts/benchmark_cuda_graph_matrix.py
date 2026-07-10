import argparse
import json
from pathlib import Path

from benchmark_runtime import run_case


MATRIX_CASES = [
    ("short", 1, 64),
    ("short", 2, 64),
    ("short", 3, 64),
    ("medium", 4, 64),
    ("medium", 5, 64),
    ("medium", 8, 64),
    ("medium", 9, 64),
    ("short", 17, 32),
]


def select_bucket(batch_size: int, graph_buckets):
    for bs in graph_buckets:
        if batch_size <= bs:
            return bs
    return None


def is_exact_bucket(batch_size: int, graph_buckets):
    return batch_size in graph_buckets


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


def print_case_result(row):
    print(
        f"[{row['prompt_name']}] "
        f"bs={row['batch_size']} "
        f"gen={row['gen_len']} "
        f"type={row['bucket_type']} "
        f"bucket={row['selected_bucket']}"
    )
    print(
        f"  baseline: "
        f"ITL={row['baseline']['itl_ms']:.2f} ms "
        f"decode_tok/s={row['baseline']['decode_tok_s']:.2f} "
        f"throughput={row['baseline']['throughput_tok_s']:.2f}"
    )
    print(
        f"  graph   : "
        f"ITL={row['graph']['itl_ms']:.2f} ms "
        f"decode_tok/s={row['graph']['decode_tok_s']:.2f} "
        f"throughput={row['graph']['throughput_tok_s']:.2f}"
    )
    print(
        f"  delta   : "
        f"ITL {row['delta']['itl_gain_pct']:+.1f}% | "
        f"decode {row['delta']['decode_gain_pct']:+.1f}% | "
        f"throughput {row['delta']['throughput_gain_pct']:+.1f}%"
    )
    print()


def write_markdown_summary(rows, output_md_path):
    output_path = Path(output_md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# CUDA Graph Benchmark Matrix Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- torch prefill + flash decode")
    lines.append("- compare `cuda_graph=off` vs `cuda_graph=on`")
    lines.append("- cover both exact-bucket and up-round bucket cases")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| prompt | bs | gen | type | selected_bucket | baseline_itl_ms | graph_itl_ms | itl_gain | baseline_decode_tok/s | graph_decode_tok/s | decode_gain | baseline_throughput_tok/s | graph_throughput_tok/s | throughput_gain |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['prompt_name']} | {row['batch_size']} | {row['gen_len']} | "
            f"{row['bucket_type']} | {row['selected_bucket']} | "
            f"{row['baseline']['itl_ms']:.2f} | {row['graph']['itl_ms']:.2f} | "
            f"{row['delta']['itl_gain_pct']:+.1f}% | "
            f"{row['baseline']['decode_tok_s']:.2f} | {row['graph']['decode_tok_s']:.2f} | "
            f"{row['delta']['decode_gain_pct']:+.1f}% | "
            f"{row['baseline']['throughput_tok_s']:.2f} | {row['graph']['throughput_tok_s']:.2f} | "
            f"{row['delta']['throughput_gain_pct']:+.1f}% |"
        )

    exact_rows = [row for row in rows if row["bucket_type"] == "exact"]
    upround_rows = [row for row in rows if row["bucket_type"] == "up_round"]

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if exact_rows:
        avg_exact_itl = sum(row["delta"]["itl_gain_pct"] for row in exact_rows) / len(exact_rows)
        avg_exact_decode = sum(row["delta"]["decode_gain_pct"] for row in exact_rows) / len(exact_rows)
        lines.append(
            f"- exact bucket average gain: ITL {avg_exact_itl:+.1f}%, decode {avg_exact_decode:+.1f}%"
        )
    if upround_rows:
        avg_up_itl = sum(row["delta"]["itl_gain_pct"] for row in upround_rows) / len(upround_rows)
        avg_up_decode = sum(row["delta"]["decode_gain_pct"] for row in upround_rows) / len(upround_rows)
        lines.append(
            f"- up-round bucket average gain: ITL {avg_up_itl:+.1f}%, decode {avg_up_decode:+.1f}%"
        )
    lines.append("- warmup runs are discarded before timing, so this is a steady-state benchmark.")
    lines.append("- TTFT here should not be interpreted as strict cold-start TTFT.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a more formal CUDA Graph benchmark matrix for exact and up-round bucket cases."
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--jsonl-output",
        default="/home/xhk/nanovllm_self/results/cuda_graph_benchmark_matrix.jsonl",
    )
    parser.add_argument(
        "--md-output",
        default="/home/xhk/nanovllm_self/results/cuda_graph_benchmark_matrix_summary.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    graph_buckets = [1, 2, 4, 8] + list(range(16, 256 + 1, 16))

    jsonl_path = Path(args.jsonl_output)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()

    rows = []
    for prompt_name, batch_size, gen_len in MATRIX_CASES:
        selected_bucket = select_bucket(batch_size, graph_buckets)
        bucket_type = "exact" if is_exact_bucket(batch_size, graph_buckets) else "up_round"

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

        row = {
            "prompt_name": prompt_name,
            "batch_size": batch_size,
            "gen_len": gen_len,
            "warmup_runs": args.warmup_runs,
            "bucket_type": bucket_type,
            "selected_bucket": selected_bucket,
            "baseline": base,
            "graph": graph,
            "delta": delta,
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)
        print_case_result(row)

    write_markdown_summary(rows, args.md_output)
    print(f"JSONL saved to: {jsonl_path}")
    print(f"Markdown summary saved to: {args.md_output}")


if __name__ == "__main__":
    main()
