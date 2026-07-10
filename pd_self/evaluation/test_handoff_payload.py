import unittest
import gc
import torch

from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from pd_self.prefill_engine import PrefillEngine


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"


class HandoffPayloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(
            model_path=MODEL_PATH,
            device="cuda:0",
            max_num_seqs=256,
            max_num_batched_tokens=16384,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            block_size=256,
            num_blocks=256,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def _build_prefill_engine(self):
        model_runner = ModelRunner(self.config)
        return PrefillEngine(self.config, self.tokenizer, model_runner)

    def _payload_id(self, payload):
        if hasattr(payload, "seq_id"):
            return payload.seq_id
        return payload.seq_idx

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_handoff_payload_contains_prefill_state(self):
        prompt = "What is a large language model?"
        prefill_engine = self._build_prefill_engine()

        payload = prefill_engine.run_prefill(
            texts=[prompt],
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
            start_seq_id=7,
        )[0]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        self.assertEqual(self._payload_id(payload), 7)
        self.assertEqual(payload.num_prompt_tokens, len(prompt_ids))
        self.assertEqual(payload.num_cached_tokens, len(prompt_ids))
        self.assertEqual(payload.token_ids[: len(prompt_ids)], prompt_ids)
        self.assertGreater(len(payload.token_ids), len(prompt_ids))
        self.assertGreater(len(payload.block_table), 0)
        self.assertEqual(payload.temperature, 0.0)
        self.assertEqual(payload.max_tokens, 32)
        self.assertTrue(payload.ignore_eos)
        self.assertFalse(payload.finished)

    def test_handoff_payload_finished_when_max_tokens_is_one(self):
        prompt = "What is a large language model?"
        prefill_engine = self._build_prefill_engine()

        payload = prefill_engine.run_prefill(
            texts=[prompt],
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            start_seq_id=9,
        )[0]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        self.assertEqual(self._payload_id(payload), 9)
        self.assertEqual(payload.num_prompt_tokens, len(prompt_ids))
        self.assertEqual(payload.token_ids[: len(prompt_ids)], prompt_ids)
        self.assertEqual(len(payload.token_ids), len(prompt_ids) + 1)
        self.assertTrue(payload.finished)
        # finished 请求当前实现下已经释放 KV 相关状态
        self.assertEqual(payload.num_cached_tokens, 0)
        self.assertEqual(payload.block_table, [])

    def test_multi_payloads_preserve_order_and_ids(self):
        prompts = [
            "What is a large language model?",
            "How does a transformer model work?",
        ]
        prefill_engine = self._build_prefill_engine()

        payloads = prefill_engine.run_prefill(
            texts=prompts,
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
            start_seq_id=20,
        )

        self.assertEqual(len(payloads), 2)
        self.assertEqual([self._payload_id(payload) for payload in payloads], [20, 21])

        for prompt, payload in zip(prompts, payloads):
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            self.assertEqual(payload.token_ids[: len(prompt_ids)], prompt_ids)
            self.assertEqual(payload.num_prompt_tokens, len(prompt_ids))
            self.assertGreaterEqual(len(payload.token_ids), len(prompt_ids) + 1)
            self.assertEqual(payload.num_cached_tokens, len(prompt_ids))
            self.assertGreater(len(payload.block_table), 0)
            self.assertFalse(payload.finished)


if __name__ == "__main__":
    unittest.main(verbosity=2)
