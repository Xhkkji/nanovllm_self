import argparse
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEVICE = "cuda:0"

PROMPTS_EQUAL_LEN = [
    "Explain what a large language model is.",
    "Explain how the Transformer neural network architecture works.",
]

PROMPTS_MIXED_LEN = [
    "What is machine learning?",
    "Explain how the Transformer neural network architecture works in detail.",
]

class CompareWithHFTest(unittest.TestCase):
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

    def _make_seq(self, text, seq_idx, max_new_tokens):
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        seq = Sequence(seq_idx=seq_idx, token_ids=token_ids, block_size=self.config.block_size)
        seq.temperature = 0.0
        seq.max_tokens = max_new_tokens
        return seq

    def _run_self(self, prompts, max_new_tokens):
        self._reset_runner_state()
        scheduler = Scheduler(self.config, self.model_runner.block_manager)
        seqs = [self._make_seq(text, i, max_new_tokens) for i, text in enumerate(prompts)]

        for seq in seqs:
            scheduler.add(seq)

        with torch.inference_mode():
            while not scheduler.is_finished():
                seq_list, is_prefill = scheduler.schedule()
                token_ids = self.model_runner.run(seq_list, is_prefill)
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                token_ids = [int(x) for x in token_ids]
                scheduler.postprocess(seq_list, token_ids)

        return [seq.token_ids for seq in seqs]

    def _run_hf_single(self, prompt, max_new_tokens):
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.inference_mode():
            output = self.hf.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return output[0].tolist()

    def _assert_all_equal(self, prompts, self_outputs, hf_outputs):
        for i, prompt in enumerate(prompts):
            self.assertEqual(
                self_outputs[i],
                hf_outputs[i],
                msg=(
                    f"seq{i} mismatch\n"
                    f"prompt: {prompt}\n"
                    f"self: {self_outputs[i]}\n"
                    f"hf:   {hf_outputs[i]}"
                ),
            )

    def test_single_seq_greedy_matches_hf(self):
        max_new_tokens = 16
        prompts = [PROMPTS_EQUAL_LEN[0]]
        self_outputs = self._run_self(prompts, max_new_tokens)
        hf_outputs = [self._run_hf_single(prompt, max_new_tokens) for prompt in prompts]
        self._assert_all_equal(prompts, self_outputs, hf_outputs)

    def test_multi_seq_greedy_matches_hf_equal_len(self):
        max_new_tokens = 16
        prompts = PROMPTS_EQUAL_LEN
        self_outputs = self._run_self(prompts, max_new_tokens)
        hf_outputs = [self._run_hf_single(prompt, max_new_tokens) for prompt in prompts]
        self._assert_all_equal(prompts, self_outputs, hf_outputs)

    def test_multi_seq_greedy_matches_hf_mixed_len(self):
        max_new_tokens = 16
        prompts = PROMPTS_MIXED_LEN
        self_outputs = self._run_self(prompts, max_new_tokens)
        hf_outputs = [self._run_hf_single(prompt, max_new_tokens) for prompt in prompts]
        self._assert_all_equal(prompts, self_outputs, hf_outputs)


def main():
    parser = argparse.ArgumentParser(description="Compare nanovllm_self greedy outputs with HF.")
    parser.add_argument(
        "-k",
        "--keyword",
        default=None,
        help="Only run tests whose name contains this keyword.",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(CompareWithHFTest)

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
