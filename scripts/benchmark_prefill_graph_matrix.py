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


COMBOS = [
    ("torch", False),
    ("flashattn", False),
    ("torch", True),
    ("flashattn", True),
]


def combo_key(prefill_backend, cuda_graph):
    graph_label = "graph_on" if cuda_graph else "graph_off"
    return f"{prefill_backend}_{graph_label}"


def append_jsonl(path, row):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(old, new, larger_is_better):
    if old == 0:
        return 0.0
    if larger_is_better:
        return (new - old) / old * 100.0
    return (old - new) / old * 100.0


def compute_summary(metrics_by_combo):
    base = metrics_by_combo["torch_graph_off"]
    flash = metrics_by_combo["flashattn_graph_off"]
    graph = metrics_by_combo["torch_graph_on"]
    full = metrics_by_combo["flashattn_graph_on"]

    return {
        "flash_prefill_vs_base": {
            "ttft_gain_pct": pct(base["ttft_ms"], flash["ttft_ms"], larger_is_better=False),
            "prefill_gain_pct": pct(base["prefill_tok_s"], flash["prefill_tok_s"], larger_is_better=True),
            "throughput_gain_pct": pct(base["throughput_tok_s"], flash["throughput_tok_s"], larger_is_better=True),
        },
        "graph_vs_base": {
            "ttft_gain_pct": pct(base["ttft_ms"], graph["ttft_ms"], larger_is_better=False),
            "itl_gain_pct": pct(base["itl_ms"], graph["itl_ms"], larger_is_better=False),
            "throughput_gain_pct": pct(base["throughput_tok_s"], graph["throughput_tok_s"], larger_is_better=True),
        },
        "full_stack_vs_base": {
            "ttft_gain_pct": pct(base["ttft_ms"], full["ttft_ms"], larger_is_better=False),
            "itl_gain_pct": pct(base["itl_ms"], full["itl_ms"], larger_is_better=False),
            "prefill_gain_pct": pct(base["prefill_tok_s"], full["prefill_tok_s"], larger_is_better=True),
            "throughput_gain_pct": pct(base["throughput_tok_s"], full["throughput_tok_s"], larger_is_better=True),
        },
    }


def print_case_result(prompt_name, batch_size, gen_len, metrics_by_combo, summary):
    print(f"[{prompt_name}] bs={batch_size} gen={gen_len}")
    for key in ["torch_graph_off", "flashattn_graph_off", "torch_graph_on", "flashattn_graph_on"]:
        m = metrics_by_combo[key]
        print(
            f"  {key}: "
            f"TTFT={m['ttft_ms']:.2f} ms "
            f"ITL={m['itl_ms']:.2f} ms "
            f"prefill_tok/s={m['prefill_tok_s']:.2f} "
            f"decode_tok/s={m['decode_tok_s']:.2f} "
            f"throughput={m['throughput_tok_s']:.2f}"
        )
    print(
        "  flash_prefill_vs_base: "
        f"TTFT {summary['flash_prefill_vs_base']['ttft_gain_pct']:+.1f}% | "
        f"prefill {summary['flash_prefill_vs_base']['prefill_gain_pct']:+.1f}% | "
        f"throughput {summary['flash_prefill_vs_base']['throughput_gain_pct']:+.1f}%"
    )
    print(
        "  graph_vs_base       : "
        f"TTFT {summary['graph_vs_base']['ttft_gain_pct']:+.1f}% | "
        f"ITL {summary['graph_vs_base']['itl_gain_pct']:+.1f}% | "
        f"throughput {summary['graph_vs_base']['throughput_gain_pct']:+.1f}%"
    )
    print(
        "  full_stack_vs_base  : "
        f"TTFT {summary['full_stack_vs_base']['ttft_gain_pct']:+.1f}% | "
        f"ITL {summary['full_stack_vs_base']['itl_gain_pct']:+.1f}% | "
        f"prefill {summary['full_stack_vs_base']['prefill_gain_pct']:+.1f}% | "
        f"throughput {summary['full_stack_vs_base']['throughput_gain_pct']:+.1f}%"
    )
    print()


def write_markdown_summary(rows, output_md_path):
    output_path = Path(output_md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Prefill + Graph Matrix Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- decode backend fixed to `flashattn`")
    lines.append("- compare four combinations:")
    lines.append("  - `torch prefill + graph off`")
    lines.append("  - `flash prefill + graph off`")
    lines.append("  - `torch prefill + graph on`")
    lines.append("  - `flash prefill + graph on`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| prompt | bs | gen | base_ttft | flash_ttft | graph_ttft | full_ttft | base_itl | graph_itl | full_itl | base_prefill_tok/s | flash_prefill_tok/s | full_prefill_tok/s | base_throughput | flash_throughput | graph_throughput | full_throughput |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        m = row["metrics_by_combo"]
        lines.append(
            f"| {row['prompt_name']} | {row['batch_size']} | {row['gen_len']} | "
            f"{m['torch_graph_off']['ttft_ms']:.2f} | "
            f"{m['flashattn_graph_off']['ttft_ms']:.2f} | "
            f"{m['torch_graph_on']['ttft_ms']:.2f} | "
            f"{m['flashattn_graph_on']['ttft_ms']:.2f} | "
            f"{m['torch_graph_off']['itl_ms']:.2f} | "
            f"{m['torch_graph_on']['itl_ms']:.2f} | "
            f"{m['flashattn_graph_on']['itl_ms']:.2f} | "
            f"{m['torch_graph_off']['prefill_tok_s']:.2f} | "
            f"{m['flashattn_graph_off']['prefill_tok_s']:.2f} | "
            f"{m['flashattn_graph_on']['prefill_tok_s']:.2f} | "
            f"{m['torch_graph_off']['throughput_tok_s']:.2f} | "
            f"{m['flashattn_graph_off']['throughput_tok_s']:.2f} | "
            f"{m['torch_graph_on']['throughput_tok_s']:.2f} | "
            f"{m['flashattn_graph_on']['throughput_tok_s']:.2f} |"
        )

    avg_full_ttft = sum(row["summary"]["full_stack_vs_base"]["ttft_gain_pct"] for row in rows) / len(rows)
    avg_full_itl = sum(row["summary"]["full_stack_vs_base"]["itl_gain_pct"] for row in rows) / len(rows)
    avg_full_prefill = sum(row["summary"]["full_stack_vs_base"]["prefill_gain_pct"] for row in rows) / len(rows)
    avg_full_tp = sum(row["summary"]["full_stack_vs_base"]["throughput_gain_pct"] for row in rows) / len(rows)

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- average full-stack TTFT gain vs base: {avg_full_ttft:+.1f}%")
    lines.append(f"- average full-stack ITL gain vs base: {avg_full_itl:+.1f}%")
    lines.append(f"- average full-stack prefill throughput gain vs base: {avg_full_prefill:+.1f}%")
    lines.append(f"- average full-stack end-to-end throughput gain vs base: {avg_full_tp:+.1f}%")
    lines.append("- current benchmark uses warmup before timing, so it reflects steady-state behavior.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark combined prefill backend and CUDA Graph configurations."
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--jsonl-output",
        default="/home/xhk/nanovllm_self/results/prefill_graph_matrix.jsonl",
    )
    parser.add_argument(
        "--md-output",
        default="/home/xhk/nanovllm_self/results/prefill_graph_matrix_summary.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    jsonl_path = Path(args.jsonl_output)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()

    rows = []
    for prompt_name, batch_size, gen_len in CASES:
        metrics_by_combo = {}
        for prefill_backend, cuda_graph in COMBOS:
            metrics = run_case(
                prompt_name=prompt_name,
                batch_size=batch_size,
                gen_len=gen_len,
                prefill_backend=prefill_backend,
                decode_backend="flashattn",
                cuda_graph=cuda_graph,
                warmup_runs=args.warmup_runs,
            )
            metrics_by_combo[combo_key(prefill_backend, cuda_graph)] = metrics

        summary = compute_summary(metrics_by_combo)
        row = {
            "prompt_name": prompt_name,
            "batch_size": batch_size,
            "gen_len": gen_len,
            "warmup_runs": args.warmup_runs,
            "metrics_by_combo": metrics_by_combo,
            "summary": summary,
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)
        print_case_result(prompt_name, batch_size, gen_len, metrics_by_combo, summary)

    write_markdown_summary(rows, args.md_output)
    print(f"JSONL saved to: {jsonl_path}")
    print(f"Markdown summary saved to: {args.md_output}")


if __name__ == "__main__":
    main()
