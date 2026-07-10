import argparse
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEVICE = "cuda:0"

SHARED_PREFIX = (
    "Large language model inference systems usually separate prefill and decode. "
    "During prefill, the model processes the full prompt and writes key value tensors into a paged KV cache. "
    "During decode, the system only computes the newest query token and attends to the full cached history. "
    "A production runtime must balance correctness, memory efficiency, batch scheduling, and latency stability. "
    "Prefix caching reuses identical full blocks from previous requests so repeated prompt prefixes do not need to be recomputed. "
    "This optimization is especially useful for agent systems, tool prompts, long system prompts, and multi-turn workflows. "
)

LONG_SHARED_PREFIX = " ".join([SHARED_PREFIX] * 10)
PREFIX_PROMPT = LONG_SHARED_PREFIX
TARGET_PROMPT_A = (
    LONG_SHARED_PREFIX
    + " Now explain why prefix caching helps reduce TTFT in an inference server."
)
TARGET_PROMPT_B = (
    LONG_SHARED_PREFIX
    + " Now explain why decode latency matters for interactive chat applications."
)


class LongPrefixPrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(
            model_path=MODEL_PATH,
            device=DEVICE,
            max_num_seqs=256,
            max_num_batched_tokens=16384,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            block_size=256,
            num_blocks=256,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        cls.hf = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        ).eval()
        cls.model_runner = ModelRunner(cls.config)

        for layer in cls.model_runner.model.layers:
            layer.p_attn.prefill_backend = "flashattn"
            layer.p_attn.decode_backend = "flashattn"
            layer.p_attn.forward_backend = "flashattn"

    def _reset_runner_state(self):
        self.model_runner.kv_cache.zero_()
        bm = self.model_runner.block_manager
        bm.hash_to_block_id.clear()
        bm.free_blocks_idx.clear()
        bm.free_blocks_idx.extend(range(bm.num_blocks))
        bm.used_blocks_idx.clear()
        for block in bm.blocks:
            block.reset()

    def _make_seq(self, text, seq_idx):
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        seq = Sequence(seq_idx=seq_idx, token_ids=token_ids, block_size=self.config.block_size)
        seq.temperature = 0.0
        seq.max_tokens = 1
        seq.ignore_eos = True
        return seq, token_ids

    def _hf_next_token(self, prompt):
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.inference_mode():
            logits = self.hf(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            ).logits[:, -1, :]
        return int(logits.argmax(dim=-1).item())

    def _self_nonprefix_next_token(self, prompt):
        self._reset_runner_state()
        seq, _ = self._make_seq(prompt, 0)
        self.model_runner.block_manager.allocate(seq)
        seq.num_new_tokens = len(seq) - seq.num_cached_tokens
        input_ids, positions, context = self.model_runner.prepare_model_input([seq])
        with torch.inference_mode():
            logits = self.model_runner.model(input_ids, positions, context)
        return int(logits.argmax(dim=-1).item())

    def _self_prefix_prefill_result(self, target_prompt):
        self._reset_runner_state()

        prefix_seq, prefix_ids = self._make_seq(PREFIX_PROMPT, 0)
        self.model_runner.block_manager.allocate(prefix_seq)
        prefix_seq.num_new_tokens = len(prefix_seq) - prefix_seq.num_cached_tokens
        prefix_input_ids, prefix_positions, prefix_context = self.model_runner.prepare_model_input([prefix_seq])
        with torch.inference_mode():
            _ = self.model_runner.model(prefix_input_ids, prefix_positions, prefix_context)
        prefix_seq.num_cached_tokens += prefix_seq.num_new_tokens
        prefix_seq.num_new_tokens = 0

        target_seq, target_ids = self._make_seq(target_prompt, 1)
        self.model_runner.block_manager.allocate(target_seq)
        target_seq.num_new_tokens = len(target_seq) - target_seq.num_cached_tokens
        input_ids, positions, context = self.model_runner.prepare_model_input([target_seq])
        with torch.inference_mode():
            logits = self.model_runner.model(input_ids, positions, context)

        expected_cached = (len(prefix_ids) // self.config.block_size) * self.config.block_size
        return {
            "next_token": int(logits.argmax(dim=-1).item()),
            "num_cached_tokens": int(target_seq.num_cached_tokens),
            "input_ids_len": int(input_ids.numel()),
            "full_len": int(len(target_seq)),
            "has_block_tables": context.block_tables is not None,
            "q_total": int(context.cu_seqlens_q[-1].item()),
            "k_total": int(context.cu_seqlens_k[-1].item()),
            "prefix_len": len(prefix_ids),
            "target_len": len(target_ids),
            "expected_cached": expected_cached,
        }

    def _check_target(self, target_prompt):
        result = self._self_prefix_prefill_result(target_prompt)
        hf_next = self._hf_next_token(target_prompt)
        nonprefix_next = self._self_nonprefix_next_token(target_prompt)

        self.assertGreater(result["num_cached_tokens"], 0)
        self.assertEqual(result["num_cached_tokens"], result["expected_cached"])
        self.assertEqual(result["num_cached_tokens"] % self.config.block_size, 0)
        self.assertTrue(result["has_block_tables"])
        self.assertGreater(result["k_total"], result["q_total"])
        self.assertEqual(
            result["input_ids_len"],
            result["full_len"] - result["num_cached_tokens"],
        )
        self.assertEqual(result["next_token"], hf_next)
        self.assertEqual(result["next_token"], nonprefix_next)

    def test_long_prefix_prefill_target_a(self):
        self._check_target(TARGET_PROMPT_A)

    def test_long_prefix_prefill_target_b(self):
        self._check_target(TARGET_PROMPT_B)


def main():
    parser = argparse.ArgumentParser(description="Long-prefix prefill checks for nanovllm_self.")
    parser.add_argument("-k", "--keyword", default=None)
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(LongPrefixPrefillTest)

    if args.keyword:
        filtered = unittest.TestSuite()
        for test in suite:
            if args.keyword in test.id():
                filtered.addTest(test)
        suite = filtered

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
