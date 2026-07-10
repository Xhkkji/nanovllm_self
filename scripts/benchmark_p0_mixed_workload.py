import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.scheduler import Scheduler

MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEVICE = "cuda:0"

PROMPT_SHORT = "What is a large language model?"
PROMPT_MEDIUM = "How does a transformer model work?"
PROMPT_MIXED = [
    "What is machine learning?",
    "Explain how the Transformer neural network architecture works in detail.",
]

CASES = [
    {
        "case_name": "single_short",
        "prompts": [PROMPT_SHORT],
        "gen_len": 32,
    },
    {
        "case_name": "mixed_pair",
        "prompts": PROMPT_MIXED,
        "gen_len": 32,
    },
    {
        "case_name": "batch8_short",
        "prompts": [PROMPT_SHORT] * 8,
        "gen_len": 32,
    },
]

COMBOS = [
    ("torch", "flashattn", False),
    ("flashattn", "flashattn", False),
    ("torch", "flashattn", True),
    ("flashattn", "flashattn", True),
]


def set_backends(llm, prefill_backend, decode_backend):
    for layer in llm.model_runner.model.layers:
        layer.p_attn.prefill_backend = prefill_backend
        layer.p_attn.decode_backend = decode_backend


def set_cuda_graph(llm, enabled):
    llm.model_runner.enable_cuda_graph = enabled
    if enabled:
        llm.model_runner.init_graph_states()


def reset_engine(llm):
    llm.scheduler = Scheduler(llm.config, llm.model_runner.block_manager)
    llm.model_runner.kv_cache.zero_()
    bm = llm.model_runner.block_manager
    bm.hash_to_block_id.clear()
    bm.free_blocks_idx.clear()
    bm.free_blocks_idx.extend(range(bm.num_blocks))
    bm.used_blocks_idx.clear()
    for block in bm.blocks:
        block.reset()


def make_inputs(tokenizer, prompts):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    return encoded["input_ids"]


def make_sampling_params(batch_size, gen_len):
    return [
        SamplingParams(temperature=0.0, max_tokens=gen_len, ignore_eos=True)
        for _ in range(batch_size)
    ]


def run_case(llm, prompts, gen_len, prefill_backend, decode_backend, cuda_graph, warmup_runs):
    reset_engine(llm)
    set_backends(llm, prefill_backend, decode_backend)
    set_cuda_graph(llm, cuda_graph)

    inputs = make_inputs(llm.tokenizer, prompts)
    sampling_params = make_sampling_params(len(prompts), gen_len)

    for _ in range(warmup_runs):
        llm.generate(inputs, sampling_params=sampling_params, return_metrics=False)

    result = llm.generate(inputs, sampling_params=sampling_params, return_metrics=True)
    metrics = result["metrics"]
    metrics["outputs"] = result["outputs"]
    return metrics


def run_hf(tokenizer, hf, prompts, gen_len):
    outputs = []
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.inference_mode():
            output = hf.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_len,
                do_sample=False,
            )
        outputs.append(output[0].tolist())
    return outputs


def pct(old, new, larger_is_better):
    if old == 0:
        return 0.0
    if larger_is_better:
        return (new - old) / old * 100.0
    return (old - new) / old * 100.0


def compare_to_hf(case_name, prompts, gen_len, combo_name, metrics, hf_outputs):
    for idx, (self_out, hf_out) in enumerate(zip(metrics["outputs"], hf_outputs)):
        if self_out != hf_out:
            raise AssertionError(
                f"{case_name}/{combo_name} seq{idx} mismatch\nself={self_out}\nhf={hf_out}"
            )


def print_case(case_name, gen_len, metrics_by_combo, summary):
    print(f"[{case_name}] gen={gen_len}")
    for key in ["torch_graph_off", "flashattn_graph_off", "torch_graph_on", "flashattn_graph_on"]:
        m = metrics_by_combo[key]
        print(
            f"  {key}: TTFT={m['ttft_ms']:.2f} ms ITL={m['itl_ms']:.2f} ms "
            f"prefill_tok/s={m['prefill_tok_s']:.2f} decode_tok/s={m['decode_tok_s']:.2f} "
            f"throughput={m['throughput_tok_s']:.2f}"
        )
    print(
        f"  base->full: TTFT {summary['full_stack_vs_base']['ttft_gain_pct']:+.1f}% | "
        f"ITL {summary['full_stack_vs_base']['itl_gain_pct']:+.1f}% | "
        f"prefill {summary['full_stack_vs_base']['prefill_gain_pct']:+.1f}% | "
        f"throughput {summary['full_stack_vs_base']['throughput_gain_pct']:+.1f}%"
    )
    print()


def compute_summary(metrics_by_combo):
    base = metrics_by_combo["torch_graph_off"]
    flash = metrics_by_combo["flashattn_graph_off"]
    graph = metrics_by_combo["torch_graph_on"]
    full = metrics_by_combo["flashattn_graph_on"]
    return {
        "flash_prefill_vs_base": {
            "ttft_gain_pct": pct(base["ttft_ms"], flash["ttft_ms"], False),
            "prefill_gain_pct": pct(base["prefill_tok_s"], flash["prefill_tok_s"], True),
            "throughput_gain_pct": pct(base["throughput_tok_s"], flash["throughput_tok_s"], True),
        },
        "graph_vs_base": {
            "ttft_gain_pct": pct(base["ttft_ms"], graph["ttft_ms"], False),
            "itl_gain_pct": pct(base["itl_ms"], graph["itl_ms"], False),
            "throughput_gain_pct": pct(base["throughput_tok_s"], graph["throughput_tok_s"], True),
        },
        "full_stack_vs_base": {
            "ttft_gain_pct": pct(base["ttft_ms"], full["ttft_ms"], False),
            "itl_gain_pct": pct(base["itl_ms"], full["itl_ms"], False),
            "prefill_gain_pct": pct(base["prefill_tok_s"], full["prefill_tok_s"], True),
            "throughput_gain_pct": pct(base["throughput_tok_s"], full["throughput_tok_s"], True),
        },
    }


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown_summary(rows, output_md_path):
    output_path = Path(output_md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# P0 Mixed Workload Benchmark Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- mixed workload / mixed prompt length benchmark")
    lines.append("- decode backend fixed to `flashattn`")
    lines.append("- compare four combinations:")
    lines.append("  - `torch prefill + graph off`")
    lines.append("  - `flash prefill + graph off`")
    lines.append("  - `torch prefill + graph on`")
    lines.append("  - `flash prefill + graph on`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| case | bs | gen | base_ttft | full_ttft | base_itl | full_itl | base_prefill_tok/s | full_prefill_tok/s | base_throughput | full_throughput |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        base = row["metrics_by_combo"]["torch_graph_off"]
        full = row["metrics_by_combo"]["flashattn_graph_on"]
        lines.append(
            f"| {row['case_name']} | {row['batch_size']} | {row['gen_len']} | "
            f"{base['ttft_ms']:.2f} | {full['ttft_ms']:.2f} | "
            f"{base['itl_ms']:.2f} | {full['itl_ms']:.2f} | "
            f"{base['prefill_tok_s']:.2f} | {full['prefill_tok_s']:.2f} | "
            f"{base['throughput_tok_s']:.2f} | {full['throughput_tok_s']:.2f} |"
        )

    avg_full_ttft = sum(row["summary"]["full_stack_vs_base"]["ttft_gain_pct"] for row in rows) / len(rows)
    avg_full_itl = sum(row["summary"]["full_stack_vs_base"]["itl_gain_pct"] for row in rows) / len(rows)
    avg_full_prefill = sum(row["summary"]["full_stack_vs_base"]["prefill_gain_pct"] for row in rows) / len(rows)
    avg_full_throughput = sum(row["summary"]["full_stack_vs_base"]["throughput_gain_pct"] for row in rows) / len(rows)

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- average full-stack TTFT gain vs base: {avg_full_ttft:+.1f}%")
    lines.append(f"- average full-stack ITL gain vs base: {avg_full_itl:+.1f}%")
    lines.append(f"- average full-stack prefill throughput gain vs base: {avg_full_prefill:+.1f}%")
    lines.append(f"- average full-stack end-to-end throughput gain vs base: {avg_full_throughput:+.1f}%")
    lines.append("- warmup runs are discarded before timing, so this reflects steady-state behavior.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark P0 mixed workload matrix.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--jsonl-output", default="/home/xhk/nanovllm_self/results/p0_mixed_workload_matrix.jsonl")
    parser.add_argument("--md-output", default="/home/xhk/nanovllm_self/results/p0_mixed_workload_summary_20260626.md")
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLM_self(enable_profile=False)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    jsonl_path = Path(args.jsonl_output)
    if jsonl_path.exists():
        jsonl_path.unlink()

    rows = []
    for case in CASES:
        prompts = case["prompts"]
        gen_len = case["gen_len"]
        batch_size = len(prompts)
        hf_outputs = run_hf(tokenizer, hf, prompts, gen_len)

        metrics_by_combo = {}
        for prefill_backend, decode_backend, cuda_graph in COMBOS:
            combo_name = f"{prefill_backend}_{'graph_on' if cuda_graph else 'graph_off'}"
            metrics = run_case(
                llm,
                prompts,
                gen_len,
                prefill_backend,
                decode_backend,
                cuda_graph,
                args.warmup_runs,
            )
            compare_to_hf(case["case_name"], prompts, gen_len, combo_name, metrics, hf_outputs)
            metrics_by_combo[combo_name] = metrics

        summary = compute_summary(metrics_by_combo)
        row = {
            "case_name": case["case_name"],
            "batch_size": batch_size,
            "gen_len": gen_len,
            "warmup_runs": args.warmup_runs,
            "metrics_by_combo": metrics_by_combo,
            "summary": summary,
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)
        print_case(case["case_name"], gen_len, metrics_by_combo, summary)

    write_markdown_summary(rows, args.md_output)
    print(f"JSONL saved to: {jsonl_path}")
    print(f"Markdown summary saved to: {args.md_output}")


if __name__ == "__main__":
    main()
