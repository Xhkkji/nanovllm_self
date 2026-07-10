import argparse
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEVICE = "cuda:0"

PREFIX_PROMPT = "Explain what a large language model is and how it learns patterns from data."
TARGET_PROMPT = "Explain what a large language model is and how it learns patterns from data. Give two concrete examples."


class PrefixPrefillTest(unittest.TestCase):
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
        return seq

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

    def _self_prefill_next_token(self, prompt):
        self._reset_runner_state()
        seq = self._make_seq(prompt, 0)
        self.model_runner.block_manager.allocate(seq)
        input_ids, positions, context = self.model_runner.prepare_prefill([seq])
        with torch.inference_mode():
            logits = self.model_runner.model(input_ids, positions, context)
        return int(logits.argmax(dim=-1).item())

    def _self_prefix_prefill_result(self):
        self._reset_runner_state()

        prefix_seq = self._make_seq(PREFIX_PROMPT, 0)
        self.model_runner.block_manager.allocate(prefix_seq)

        # 先把 prefix 对应的 KV 真正写进全局 cache。
        prefix_input_ids, prefix_positions, prefix_context = self.model_runner.prepare_prefill([prefix_seq])
        with torch.inference_mode():
            _ = self.model_runner.model(prefix_input_ids, prefix_positions, prefix_context)

        target_seq = self._make_seq(TARGET_PROMPT, 1)
        self.model_runner.block_manager.allocate(target_seq)
        input_ids, positions, context = self.model_runner.prepare_prefill([target_seq])

        with torch.inference_mode():
            logits = self.model_runner.model(input_ids, positions, context)

        return {
            "next_token": int(logits.argmax(dim=-1).item()),
            "num_cached_tokens": int(target_seq.num_cached_tokens),
            "input_ids_len": int(input_ids.numel()),
            "full_len": int(len(target_seq)),
            "has_block_tables": context.block_tables is not None,
            "q_total": int(context.cu_seqlens_q[-1].item()),
            "k_total": int(context.cu_seqlens_k[-1].item()),
        }

    def test_prefix_prefill_hits_cache(self):
        result = self._self_prefix_prefill_result()
        self.assertGreater(
            result["num_cached_tokens"],
            0,
            msg=f"expected prefix cache hit, got num_cached_tokens={result['num_cached_tokens']}",
        )
        self.assertTrue(result["has_block_tables"], msg="prefix prefill should build block_tables")
        self.assertGreater(result["k_total"], result["q_total"], msg="prefix prefill should have k_total > q_total")
        self.assertEqual(
            result["input_ids_len"],
            result["full_len"] - result["num_cached_tokens"],
            msg="prefill should only compute uncached suffix tokens",
        )

    def test_prefix_prefill_next_token_matches_hf_and_nonprefix(self):
        hf_next = self._hf_next_token(TARGET_PROMPT)
        nonprefix_next = self._self_prefill_next_token(TARGET_PROMPT)
        prefix_result = self._self_prefix_prefill_result()

        self.assertEqual(prefix_result["next_token"], hf_next)
        self.assertEqual(prefix_result["next_token"], nonprefix_next)


def main():
    parser = argparse.ArgumentParser(description="Prefix-prefill checks for nanovllm_self.")
    parser.add_argument(
        "-k",
        "--keyword",
        default=None,
        help="Only run tests whose name contains this keyword.",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(PrefixPrefillTest)

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
