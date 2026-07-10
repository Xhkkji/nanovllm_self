import argparse
import json
from pathlib import Path

from benchmark_runtime import run_case


CASES = [
    ("short", 1, 64),
    ("short", 8, 64),
    ("medium", 1, 64),
    ("medium", 8, 64),
    ("long", 1, 64),
    ("long", 8, 64),
]


def compute_delta(torch_metrics, flash_metrics):
    def pct(old, new, larger_is_better):
        if old == 0:
            return 0.0
        if larger_is_better:
            return (new - old) / old * 100.0
        return (old - new) / old * 100.0

    return {
        "ttft_gain_pct": pct(torch_metrics["ttft_ms"], flash_metrics["ttft_ms"], larger_is_better=False),
        "prefill_gain_pct": pct(torch_metrics["prefill_tok_s"], flash_metrics["prefill_tok_s"], larger_is_better=True),
        "throughput_gain_pct": pct(torch_metrics["throughput_tok_s"], flash_metrics["throughput_tok_s"], larger_is_better=True),
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
        f"decode={row['decode_backend']} "
        f"cuda_graph={'on' if row['cuda_graph'] else 'off'}"
    )
    print(
        f"  torch prefill: "
        f"TTFT={row['torch_prefill']['ttft_ms']:.2f} ms "
        f"prefill_tok/s={row['torch_prefill']['prefill_tok_s']:.2f} "
        f"throughput={row['torch_prefill']['throughput_tok_s']:.2f}"
    )
    print(
        f"  flash prefill: "
        f"TTFT={row['flash_prefill']['ttft_ms']:.2f} ms "
        f"prefill_tok/s={row['flash_prefill']['prefill_tok_s']:.2f} "
        f"throughput={row['flash_prefill']['throughput_tok_s']:.2f}"
    )
    print(
        f"  delta        : "
        f"TTFT {row['delta']['ttft_gain_pct']:+.1f}% | "
        f"prefill {row['delta']['prefill_gain_pct']:+.1f}% | "
        f"throughput {row['delta']['throughput_gain_pct']:+.1f}%"
    )
    print()


def write_markdown_summary(rows, output_md_path):
    output_path = Path(output_md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Prefill Backend Benchmark Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- compare `torch prefill` vs `flash prefill`")
    lines.append("- decode backend fixed to `flashattn`")
    lines.append("- focus on TTFT / prefill throughput / end-to-end throughput")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| prompt | bs | gen | torch_ttft_ms | flash_ttft_ms | ttft_gain | torch_prefill_tok/s | flash_prefill_tok/s | prefill_gain | torch_throughput_tok/s | flash_throughput_tok/s | throughput_gain |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['prompt_name']} | {row['batch_size']} | {row['gen_len']} | "
            f"{row['torch_prefill']['ttft_ms']:.2f} | {row['flash_prefill']['ttft_ms']:.2f} | "
            f"{row['delta']['ttft_gain_pct']:+.1f}% | "
            f"{row['torch_prefill']['prefill_tok_s']:.2f} | {row['flash_prefill']['prefill_tok_s']:.2f} | "
            f"{row['delta']['prefill_gain_pct']:+.1f}% | "
            f"{row['torch_prefill']['throughput_tok_s']:.2f} | {row['flash_prefill']['throughput_tok_s']:.2f} | "
            f"{row['delta']['throughput_gain_pct']:+.1f}% |"
        )

    avg_ttft = sum(row["delta"]["ttft_gain_pct"] for row in rows) / len(rows)
    avg_prefill = sum(row["delta"]["prefill_gain_pct"] for row in rows) / len(rows)
    avg_throughput = sum(row["delta"]["throughput_gain_pct"] for row in rows) / len(rows)

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- average TTFT gain: {avg_ttft:+.1f}%")
    lines.append(f"- average prefill throughput gain: {avg_prefill:+.1f}%")
    lines.append(f"- average end-to-end throughput gain: {avg_throughput:+.1f}%")
    lines.append("- current benchmark uses warmup before timing, so it reflects steady-state behavior.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark torch prefill vs flash prefill under flash decode."
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="off")
    parser.add_argument(
        "--jsonl-output",
        default="/home/xhk/nanovllm_self/results/prefill_backend_benchmark.jsonl",
    )
    parser.add_argument(
        "--md-output",
        default="/home/xhk/nanovllm_self/results/prefill_backend_benchmark_summary.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cuda_graph = args.cuda_graph == "on"

    jsonl_path = Path(args.jsonl_output)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()

    rows = []
    for prompt_name, batch_size, gen_len in CASES:
        torch_prefill = run_case(
            prompt_name=prompt_name,
            batch_size=batch_size,
            gen_len=gen_len,
            prefill_backend="torch",
            decode_backend="flashattn",
            cuda_graph=cuda_graph,
            warmup_runs=args.warmup_runs,
        )
        flash_prefill = run_case(
            prompt_name=prompt_name,
            batch_size=batch_size,
            gen_len=gen_len,
            prefill_backend="flashattn",
            decode_backend="flashattn",
            cuda_graph=cuda_graph,
            warmup_runs=args.warmup_runs,
        )
        delta = compute_delta(torch_prefill, flash_prefill)

        row = {
            "prompt_name": prompt_name,
            "batch_size": batch_size,
            "gen_len": gen_len,
            "decode_backend": "flashattn",
            "cuda_graph": cuda_graph,
            "warmup_runs": args.warmup_runs,
            "torch_prefill": torch_prefill,
            "flash_prefill": flash_prefill,
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
