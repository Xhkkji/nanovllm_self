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


def patch_decode_gathered_bf16_flash(model_runner):
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

            flat_k = torch.cat(flat_k, dim=0).to(torch.bfloat16)
            flat_v = torch.cat(flat_v, dim=0).to(torch.bfloat16)
            q_bf16 = q.to(torch.bfloat16)

            cu_q = torch.arange(
                0,
                q.size(0) + 1,
                dtype=torch.int32,
                device=q.device,
            )
            cu_k = torch.zeros(
                q.size(0) + 1,
                dtype=torch.int32,
                device=q.device,
            )
            cu_k[1:] = torch.cumsum(context.context_lens, dim=0)

            out = flash_attn_varlen_func(
                q=q_bf16,
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


def run_variant(decode_seq_ids, patch_gathered_bf16=False):
    config = build_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_runner = ModelRunner(config)
    prefill_engine = PrefillEngine(config, tokenizer, model_runner)
    decode_engine = DecodeEngine(config, tokenizer, model_runner)
    patched = []

    try:
        if patch_gathered_bf16:
            patched = patch_decode_gathered_bf16_flash(model_runner)

        with torch.no_grad():
            payloads = prefill_engine.run_prefill(
                texts=PROMPTS,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                start_seq_id=0,
            )

            results = {}
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
                token_ids, seq_need_compute_logits = model_runner.run(scheduled)
                decode_engine.scheduler.postprocess(scheduled, token_ids, seq_need_compute_logits)
                for seq in scheduled:
                    results[seq.seq_idx] = list(seq.token_ids)

        return [results[seq_id] for seq_id in sorted(results.keys())]
    finally:
        if patched:
            restore_patch(patched)
        del decode_engine, prefill_engine, model_runner, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_len = len(tokenizer(PROMPTS[1], add_special_tokens=False)["input_ids"])

    baseline_batch = run_variant({0, 1}, patch_gathered_bf16=False)
    baseline_alone = run_variant({1}, patch_gathered_bf16=False)
    patched_batch = run_variant({0, 1}, patch_gathered_bf16=True)
    patched_alone = run_variant({1}, patch_gathered_bf16=True)

    baseline_diff = find_first_diff(baseline_batch[-1], baseline_alone[-1])
    patched_diff = find_first_diff(patched_batch[-1], patched_alone[-1])

    print(
        f"baseline_flash_fp16_storage: first_diff_abs={baseline_diff} "
        f"first_diff_gen_step={None if baseline_diff is None else baseline_diff - prompt_len}"
    )
    print(
        f"gathered_bf16_flash_from_fp16_storage: first_diff_abs={patched_diff} "
        f"first_diff_gen_step={None if patched_diff is None else patched_diff - prompt_len}"
    )


if __name__ == "__main__":
    main()
