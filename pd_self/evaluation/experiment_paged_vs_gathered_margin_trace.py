import copy
import gc
import math
from types import MethodType

import torch
from flash_attn import flash_attn_varlen_func
from transformers import AutoTokenizer

from nanovllm.engine.model_runner import ModelRunner
from pd_self.decode_engine import DecodeEngine
from pd_self.prefill_engine import PrefillEngine
from pd_self.evaluation.experiment_decode_fp32_qkv import PROMPTS, build_config


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
LONG_SEQ_ID = 1
MAX_TOKENS = 32


def clone_payload(payload):
    return copy.deepcopy(payload)


def find_first_diff(a, b):
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return limit
    return None


def top2_summary(logits_row):
    values, indices = torch.topk(logits_row.float(), k=2)
    return {
        "token0": int(indices[0].item()),
        "logit0": float(values[0].item()),
        "token1": int(indices[1].item()),
        "logit1": float(values[1].item()),
        "margin": float((values[0] - values[1]).item()),
    }


def patch_decode_gathered_flash_fp16(model_runner):
    patched = []

    for layer in model_runner.model.layers:
        p_attn = layer.p_attn
        old_forward_unified = p_attn.forward_unified

        def new_forward_unified(self, q, k, v, context):
            is_decode = (
                context is not None
                and context.block_tables is not None
                and context.context_lens is not None
                and context.max_seqlen_q == 1
            )
            if not is_decode:
                return old_forward_unified(q, k, v, context)

            self.write_kv_cache(k, v, context.slot_mapping)

            k_batch, v_batch, _ = self.get_kv_cache(context)
            seq_lens = context.context_lens.tolist()
            flat_k = []
            flat_v = []
            for i, seq_len in enumerate(seq_lens):
                flat_k.append(k_batch[i, :seq_len])
                flat_v.append(v_batch[i, :seq_len])

            flat_k = torch.cat(flat_k, dim=0).to(torch.float16)
            flat_v = torch.cat(flat_v, dim=0).to(torch.float16)
            q_cast = q.to(torch.float16)

            cu_q = torch.arange(0, q.size(0) + 1, dtype=torch.int32, device=q.device)
            cu_k = torch.zeros(q.size(0) + 1, dtype=torch.int32, device=q.device)
            cu_k[1:] = torch.cumsum(context.context_lens, dim=0)

            out = flash_attn_varlen_func(
                q=q_cast,
                k=flat_k,
                v=flat_v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=1,
                max_seqlen_k=int(context.context_lens.max().item()),
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )
            return out.to(self.dtype)

        p_attn.forward_unified = MethodType(new_forward_unified, p_attn)
        patched.append((p_attn, old_forward_unified))

    return patched


def restore_patch(patched):
    for p_attn, old_forward_unified in patched:
        p_attn.forward_unified = old_forward_unified


def run_trace(decode_seq_ids, use_gathered_flash=False):
    config = build_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_runner = ModelRunner(config)
    prefill_engine = PrefillEngine(config, tokenizer, model_runner)
    decode_engine = DecodeEngine(config, tokenizer, model_runner)
    patched = []

    try:
        if use_gathered_flash:
            patched = patch_decode_gathered_flash_fp16(model_runner)

        with torch.no_grad():
            payloads = prefill_engine.run_prefill(
                texts=PROMPTS,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                start_seq_id=0,
            )

            results = {}
            trace = []
            for payload in payloads:
                if payload.seq_idx not in decode_seq_ids:
                    continue
                if payload.finished:
                    results[payload.seq_idx] = list(payload.token_ids)
                    continue
                seq = decode_engine.restore_sequence(clone_payload(payload))
                decode_engine.scheduler.running.append(seq)
                results[seq.seq_idx] = list(seq.token_ids)

            while not decode_engine.scheduler.is_finished():
                scheduled = decode_engine.scheduler.schedule()
                input_ids, positions, context = model_runner.prepare_model_input(scheduled)
                logits = model_runner.model(input_ids, positions, context)

                target_row = None
                for i, seq in enumerate(scheduled):
                    if seq.seq_idx == LONG_SEQ_ID:
                        target_row = i
                        break

                step_entry = None
                if target_row is not None:
                    need_rows = context.seq_need_compute_logits.tolist()
                    if target_row in need_rows:
                        logits_row = logits[need_rows.index(target_row)]
                        step_entry = {
                            "position": int(positions[target_row].item()),
                            **top2_summary(logits_row),
                        }

                if context.seq_need_compute_logits.numel() > 0:
                    temperatures = model_runner.prepare_sampler(scheduled, context)
                    token_ids = model_runner.sampler(logits, temperatures)
                    if isinstance(token_ids, torch.Tensor):
                        token_ids = token_ids.reshape(-1).tolist()
                    token_ids = [int(x) for x in token_ids]
                else:
                    token_ids = []

                decode_engine.scheduler.postprocess(scheduled, token_ids, context.seq_need_compute_logits.tolist())
                for seq in scheduled:
                    results[seq.seq_idx] = list(seq.token_ids)

                if step_entry is not None:
                    trace.append(step_entry)

        return [results[seq_id] for seq_id in sorted(results.keys())], trace
    finally:
        if patched:
            restore_patch(patched)
        del decode_engine, prefill_engine, model_runner, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_len = len(tokenizer(PROMPTS[1], add_special_tokens=False)["input_ids"])

    paged_batch, paged_batch_trace = run_trace({0, 1}, use_gathered_flash=False)
    paged_alone, paged_alone_trace = run_trace({1}, use_gathered_flash=False)
    gathered_batch, gathered_batch_trace = run_trace({0, 1}, use_gathered_flash=True)
    gathered_alone, gathered_alone_trace = run_trace({1}, use_gathered_flash=True)

    paged_diff = find_first_diff(paged_batch[-1], paged_alone[-1])
    gathered_diff = find_first_diff(gathered_batch[-1], gathered_alone[-1])

    print(
        f"paged first_diff_abs={paged_diff} "
        f"first_diff_gen_step={None if paged_diff is None else paged_diff - prompt_len}"
    )
    print(
        f"gathered first_diff_abs={gathered_diff} "
        f"first_diff_gen_step={None if gathered_diff is None else gathered_diff - prompt_len}"
    )
    print()
    print("position | paged_batch_top2 | paged_alone_top2 | gathered_batch_top2 | gathered_alone_top2")

    for pb, pa, gb, ga in zip(
        paged_batch_trace,
        paged_alone_trace,
        gathered_batch_trace,
        gathered_alone_trace,
    ):
        print(
            f"{pb['position']:>8} | "
            f"({pb['token0']},{pb['margin']:.3f}) ({pb['token1']}) | "
            f"({pa['token0']},{pa['margin']:.3f}) ({pa['token1']}) | "
            f"({gb['token0']},{gb['margin']:.3f}) ({gb['token1']}) | "
            f"({ga['token0']},{ga['margin']:.3f}) ({ga['token1']})"
        )


if __name__ == "__main__":
    main()
