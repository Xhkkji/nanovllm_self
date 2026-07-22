import copy
import gc

import torch
from transformers import AutoTokenizer

from nanovllm.engine.model_runner import ModelRunner
from pd_self.decode_engine import DecodeEngine
from pd_self.prefill_engine import PrefillEngine
from pd_self.evaluation.experiment_decode_fp32_qkv import (
    DisableTF32,
    PROMPTS,
    build_config,
    patch_decode_qkv_fp32,
    restore_patch,
)


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
LONG_SEQ_ID = 1
TARGET_POSITION = 42
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


def patch_kv_cache_bf16(model_runner):
    old_cache = model_runner.kv_cache
    model_runner.kv_cache = torch.zeros_like(old_cache, dtype=torch.bfloat16)
    model_runner.bind_kvcache_to_attention()
    return old_cache


def restore_kv_cache(model_runner, old_cache):
    model_runner.kv_cache = old_cache
    model_runner.bind_kvcache_to_attention()


def top2_summary(logits_row):
    values, indices = torch.topk(logits_row.float(), k=2)
    return {
        "token0": int(indices[0].item()),
        "logit0": float(values[0].item()),
        "token1": int(indices[1].item()),
        "logit1": float(values[1].item()),
        "margin": float((values[0] - values[1]).item()),
    }


def run_pair_prefill_and_decode_with_results(
    decode_seq_ids, use_fp32_qkv=False, use_bf16_kv=False
):
    config = build_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_runner = ModelRunner(config)
    prefill_engine = PrefillEngine(config, tokenizer, model_runner)
    decode_engine = DecodeEngine(config, tokenizer, model_runner)

    patched = []
    old_kv_cache = None
    active_results = {}

    try:
        if use_bf16_kv:
            old_kv_cache = patch_kv_cache_bf16(model_runner)
        if use_fp32_qkv:
            patched = patch_decode_qkv_fp32(model_runner)

        with torch.no_grad():
            payloads = prefill_engine.run_prefill(
                texts=PROMPTS,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                ignore_eos=True,
                start_seq_id=0,
            )

            for payload in payloads:
                if payload.seq_idx not in decode_seq_ids:
                    continue
                if payload.finished:
                    active_results[payload.seq_idx] = list(payload.token_ids)
                    continue
                seq = decode_engine.restore_sequence(clone_payload(payload))
                decode_engine.scheduler.running.append(seq)
                active_results[seq.seq_idx] = list(seq.token_ids)

            target_capture = None

            while not decode_engine.scheduler.is_finished():
                scheduled = decode_engine.scheduler.schedule()
                input_ids, positions, context = model_runner.prepare_model_input(scheduled)
                logits = model_runner.model(input_ids, positions, context)

                target_row = None
                for i, seq in enumerate(scheduled):
                    if seq.seq_idx == LONG_SEQ_ID:
                        target_row = i
                        break

                if target_row is not None and int(positions[target_row].item()) == TARGET_POSITION:
                    need_rows = context.seq_need_compute_logits.tolist()
                    if target_row in need_rows:
                        logits_row = logits[need_rows.index(target_row)]
                        target_capture = top2_summary(logits_row)

                if context.seq_need_compute_logits.numel() > 0:
                    temperatures = model_runner.prepare_sampler(scheduled, context)
                    token_ids = model_runner.sampler(logits, temperatures)
                    if isinstance(token_ids, torch.Tensor):
                        token_ids = token_ids.reshape(-1).tolist()
                    token_ids = [int(x) for x in token_ids]
                else:
                    token_ids = []

                decode_engine.scheduler.postprocess(
                    scheduled,
                    token_ids,
                    context.seq_need_compute_logits.tolist(),
                )

                for seq in scheduled:
                    active_results[seq.seq_idx] = list(seq.token_ids)

        ordered = [active_results[seq_id] for seq_id in sorted(active_results.keys())]
        return ordered, target_capture
    finally:
        if patched:
            restore_patch(patched)
        if old_kv_cache is not None:
            restore_kv_cache(model_runner, old_kv_cache)
        del decode_engine, prefill_engine, model_runner, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def run_variant(name, use_fp32_qkv=False, use_bf16_kv=False, disable_tf32=False):
    if disable_tf32:
        with DisableTF32():
            batch_outputs, batch_capture = run_pair_prefill_and_decode_with_results(
                decode_seq_ids={0, 1},
                use_fp32_qkv=use_fp32_qkv,
                use_bf16_kv=use_bf16_kv,
            )
            alone_outputs, alone_capture = run_pair_prefill_and_decode_with_results(
                decode_seq_ids={1},
                use_fp32_qkv=use_fp32_qkv,
                use_bf16_kv=use_bf16_kv,
            )
    else:
        batch_outputs, batch_capture = run_pair_prefill_and_decode_with_results(
            decode_seq_ids={0, 1},
            use_fp32_qkv=use_fp32_qkv,
            use_bf16_kv=use_bf16_kv,
        )
        alone_outputs, alone_capture = run_pair_prefill_and_decode_with_results(
            decode_seq_ids={1},
            use_fp32_qkv=use_fp32_qkv,
            use_bf16_kv=use_bf16_kv,
        )

    long_batch = batch_outputs[-1]
    long_alone = alone_outputs[-1]
    prompt_len = len(
        AutoTokenizer.from_pretrained(MODEL_PATH)(
            PROMPTS[1], add_special_tokens=False
        )["input_ids"]
    )
    first_diff_abs = find_first_diff(long_batch, long_alone)
    first_diff_gen_step = None if first_diff_abs is None else first_diff_abs - prompt_len

    print(f"=== {name} ===")
    print(f"first_diff_abs={first_diff_abs} first_diff_gen_step={first_diff_gen_step}")
    print(f"batch_capture={batch_capture}")
    print(f"alone_capture={alone_capture}")
    print()


def main():
    run_variant("baseline")
    run_variant("fp32_qkv_no_tf32", use_fp32_qkv=True, disable_tf32=True)
    run_variant("bf16_kv_cache", use_bf16_kv=True)
    run_variant(
        "bf16_kv_cache_plus_fp32_qkv_no_tf32",
        use_bf16_kv=True,
        use_fp32_qkv=True,
        disable_tf32=True,
    )


if __name__ == "__main__":
    main()
