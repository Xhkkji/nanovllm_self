import copy
import gc
from types import MethodType

import torch
from transformers import AutoTokenizer

from nanovllm.engine.model_runner import ModelRunner
from pd_self.decode_engine import DecodeEngine
from pd_self.prefill_engine import PrefillEngine
from pd_self.evaluation.experiment_decode_fp32_qkv import PROMPTS, build_config
from pd_self.evaluation.experiment_mixed_len_logit_margin import patch_kv_cache_bf16


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
LONG_SEQ_ID = 1
TARGET_POSITION = 35
MAX_TOKENS = 32


def clone_payload(payload):
    return copy.deepcopy(payload)


class KVStoreTracer:
    def __init__(self, model_runner, target_layer=0):
        self.model_runner = model_runner
        self.target_layer = target_layer
        self.current_meta = None
        self.capture = None
        self._patch()

    def set_meta(self, meta):
        self.current_meta = meta

    def _patch(self):
        layer = self.model_runner.model.layers[self.target_layer]
        p_attn = layer.p_attn
        tracer = self

        def wrapped_write(this, k, v, slot_mapping):
            meta = tracer.current_meta
            if meta and meta.get("capture_now") and meta.get("row") is not None:
                row = meta["row"]
                slot = int(slot_mapping[row].item())
                block_id = slot // this.block_size
                offset = slot % this.block_size

                raw_k = k[row].detach().float().cpu()
                raw_v = v[row].detach().float().cpu()

                old_method(k, v, slot_mapping)

                stored_k = this.k_cache[block_id, offset].detach().float().cpu()
                stored_v = this.v_cache[block_id, offset].detach().float().cpu()
                tracer.capture = {
                    "seq_ids": list(meta["seq_ids"]),
                    "row": row,
                    "slot": slot,
                    "block_id": int(block_id),
                    "offset": int(offset),
                    "raw_k": raw_k,
                    "raw_v": raw_v,
                    "stored_k": stored_k,
                    "stored_v": stored_v,
                    "cache_dtype": str(this.k_cache.dtype),
                }
                return None

            return old_method(k, v, slot_mapping)

        old_method = p_attn.write_kv_cache
        p_attn.write_kv_cache = MethodType(wrapped_write, p_attn)
        self.old_method = old_method
        self.p_attn = p_attn

    def close(self):
        self.p_attn.write_kv_cache = self.old_method


def run_scenario(decode_seq_ids, use_bf16_kv=False):
    config = build_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_runner = ModelRunner(config)
    prefill_engine = PrefillEngine(config, tokenizer, model_runner)
    decode_engine = DecodeEngine(config, tokenizer, model_runner)
    tracer = KVStoreTracer(model_runner, target_layer=0)
    old_kv_cache = None

    try:
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

            for payload in payloads:
                if payload.seq_idx not in decode_seq_ids or payload.finished:
                    continue
                decode_engine.scheduler.running.append(
                    decode_engine.restore_sequence(clone_payload(payload))
                )

            while not decode_engine.scheduler.is_finished():
                scheduled = decode_engine.scheduler.schedule()
                input_ids, positions, context = model_runner.prepare_model_input(scheduled)

                target_row = None
                for i, seq in enumerate(scheduled):
                    if seq.seq_idx == LONG_SEQ_ID:
                        target_row = i
                        break

                capture_now = bool(
                    target_row is not None and int(positions[target_row].item()) == TARGET_POSITION
                )
                tracer.set_meta(
                    {
                        "capture_now": capture_now,
                        "row": target_row,
                        "seq_ids": [seq.seq_idx for seq in scheduled],
                    }
                )

                logits = model_runner.model(input_ids, positions, context)

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

                if tracer.capture is not None:
                    break

        return tracer.capture
    finally:
        tracer.close()
        if old_kv_cache is not None:
            model_runner.kv_cache = old_kv_cache
            model_runner.bind_kvcache_to_attention()
        del decode_engine, prefill_engine, model_runner, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def summarize(name, batch_capture, alone_capture):
    print(f"=== {name} ===")
    print("batch meta:", {k: batch_capture[k] for k in ["cache_dtype", "seq_ids", "slot", "block_id", "offset"]})
    print("alone meta:", {k: alone_capture[k] for k in ["cache_dtype", "seq_ids", "slot", "block_id", "offset"]})

    raw_k_diff = (batch_capture["raw_k"] - alone_capture["raw_k"]).abs().max().item()
    raw_v_diff = (batch_capture["raw_v"] - alone_capture["raw_v"]).abs().max().item()
    stored_k_diff = (batch_capture["stored_k"] - alone_capture["stored_k"]).abs().max().item()
    stored_v_diff = (batch_capture["stored_v"] - alone_capture["stored_v"]).abs().max().item()

    store_err_batch_k = (batch_capture["stored_k"] - batch_capture["raw_k"]).abs().max().item()
    store_err_alone_k = (alone_capture["stored_k"] - alone_capture["raw_k"]).abs().max().item()
    store_err_batch_v = (batch_capture["stored_v"] - batch_capture["raw_v"]).abs().max().item()
    store_err_alone_v = (alone_capture["stored_v"] - alone_capture["raw_v"]).abs().max().item()

    print(f"raw_k_diff={raw_k_diff:.6f} raw_v_diff={raw_v_diff:.6f}")
    print(f"stored_k_diff={stored_k_diff:.6f} stored_v_diff={stored_v_diff:.6f}")
    print(
        f"batch_store_err_k={store_err_batch_k:.6f} alone_store_err_k={store_err_alone_k:.6f} "
        f"batch_store_err_v={store_err_batch_v:.6f} alone_store_err_v={store_err_alone_v:.6f}"
    )
    print()


def main():
    fp16_batch = run_scenario({0, 1}, use_bf16_kv=False)
    fp16_alone = run_scenario({1}, use_bf16_kv=False)
    summarize("fp16_kv_cache", fp16_batch, fp16_alone)

    bf16_batch = run_scenario({0, 1}, use_bf16_kv=True)
    bf16_alone = run_scenario({1}, use_bf16_kv=True)
    summarize("bf16_kv_cache", bf16_batch, bf16_alone)


if __name__ == "__main__":
    main()
