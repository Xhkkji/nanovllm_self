import copy
import gc

import torch
from transformers import AutoTokenizer

from nanovllm.engine.model_runner import ModelRunner
from pd_self.decode_engine import DecodeEngine
from pd_self.prefill_engine import PrefillEngine
from pd_self.evaluation.experiment_decode_fp32_qkv import PROMPTS, build_config
from pd_self.evaluation.experiment_mixed_len_logit_margin import patch_kv_cache_bf16


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


def set_backend(model_runner, backend: str):
    for layer in model_runner.model.layers:
        layer.p_attn.set_attention_backend(backend)


def run_variant(decode_seq_ids, backend="flashattn", use_bf16_kv=False):
    config = build_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_runner = ModelRunner(config)
    prefill_engine = PrefillEngine(config, tokenizer, model_runner)
    decode_engine = DecodeEngine(config, tokenizer, model_runner)
    old_kv_cache = None

    try:
        set_backend(model_runner, backend)
        if use_bf16_kv:
            old_kv_cache = patch_kv_cache_bf16(model_runner)

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
        if old_kv_cache is not None:
            model_runner.kv_cache = old_kv_cache
            model_runner.bind_kvcache_to_attention()
        del decode_engine, prefill_engine, model_runner, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def summarize_case(name, batch_outputs, alone_outputs, prompt_len):
    long_batch = batch_outputs[-1]
    long_alone = alone_outputs[-1]
    diff = find_first_diff(long_batch, long_alone)
    gen_step = None if diff is None else diff - prompt_len
    print(f"{name}: first_diff_abs={diff} first_diff_gen_step={gen_step}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_len = len(tokenizer(PROMPTS[1], add_special_tokens=False)["input_ids"])

    cases = [
        ("flash_fp16kv", "flashattn", False),
        ("torch_fp16kv", "torch", False),
        ("flash_bf16kv", "flashattn", True),
        ("torch_bf16kv", "torch", True),
    ]

    for name, backend, use_bf16_kv in cases:
        batch_outputs = run_variant({0, 1}, backend=backend, use_bf16_kv=use_bf16_kv)
        alone_outputs = run_variant({1}, backend=backend, use_bf16_kv=use_bf16_kv)
        summarize_case(name, batch_outputs, alone_outputs, prompt_len)


if __name__ == "__main__":
    main()
