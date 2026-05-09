import argparse
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.block_manager import block_manager as BlockManager
from nanovllm.models.qwen3 import Qwen3Model


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEVICE = "cuda:0"
DEFAULT_PROMPT = "Introduce the acg in China where nearby Japan."


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_timings(name, prompt_len, steps, total_sec, step_times):
    ms = [x * 1000 for x in step_times]
    print(f"[{name}]")
    print(f"  prompt_tokens   : {prompt_len}")
    print(f"  generated_steps : {steps}")
    print(f"  total_time_sec  : {total_sec:.4f}")
    print(f"  throughput_tok_s: {steps / total_sec:.2f}" if total_sec > 0 else "  throughput_tok_s: inf")
    print(f"  step_avg_ms     : {statistics.mean(ms):.3f}")
    print(f"  step_p50_ms     : {statistics.median(ms):.3f}")
    print(f"  step_max_ms     : {max(ms):.3f}")
    if len(ms) >= 10:
        head = ms[:10]
        tail = ms[-10:]
        print(f"  first10_ms      : {[round(x, 3) for x in head]}")
        print(f"  last10_ms       : {[round(x, 3) for x in tail]}")
    print()


def run_hf(prompt, steps):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    all_tokens = input_ids.clone()
    past = None
    step_times = []

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            step_start = time.perf_counter()
            current = all_tokens if past is None else all_tokens[:, -1:]
            out = model(current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
            cuda_sync()
            step_times.append(time.perf_counter() - step_start)
    total_sec = time.perf_counter() - t0
    return input_ids.shape[1], total_sec, step_times


def run_self(prompt, steps, block_size):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = Qwen3Model(config).to(DEVICE).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    block_manager = BlockManager(
        num_blocks=100,
        block_size=block_size,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
    )
    seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
    seq.block_size = block_size
    seq.block_table = block_manager.allocate_with_prefill(seq)

    all_tokens = input_ids.clone()
    is_prefill = True
    step_times = []

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            step_start = time.perf_counter()
            if is_prefill:
                current = all_tokens[0]
                positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)
                out = model(current, positions=positions, block_manager=block_manager, seq=seq, is_prefill=True)
                logits = out[-1, :].unsqueeze(0)
                is_prefill = False
            else:
                current = all_tokens[0, -1:]
                positions = torch.tensor([[len(seq.token_ids) - 1]], device=DEVICE)
                out = model(current, positions=positions, block_manager=block_manager, seq=seq, is_prefill=False)
                logits = out.unsqueeze(0)

            next_token = logits.argmax(dim=-1, keepdim=True)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)

            token_id = int(next_token[0, 0].item())
            seq.append_token(token_id)
            if len(seq.token_ids) > len(seq.block_table) * block_manager.block_size:
                seq.block_table.append(block_manager.allocate_block(1)[0])

            cuda_sync()
            step_times.append(time.perf_counter() - step_start)
    total_sec = time.perf_counter() - t0
    return input_ids.shape[1], total_sec, step_times


def main():
    parser = argparse.ArgumentParser(description="Benchmark greedy decode for HF and nanovllm_self.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--mode", choices=["hf", "self", "both"], default="both")
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    if args.mode in ("hf", "both"):
        prompt_len, total_sec, step_times = run_hf(args.prompt, args.steps)
        summarize_timings("HF", prompt_len, args.steps, total_sec, step_times)

    if args.mode in ("self", "both"):
        prompt_len, total_sec, step_times = run_self(args.prompt, args.steps, args.block_size)
        summarize_timings("SELF", prompt_len, args.steps, total_sec, step_times)


if __name__ == "__main__":
    main()
