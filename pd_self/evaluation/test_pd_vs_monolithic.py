import unittest
import gc

import torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams
from pd_self.coordinator import PDCoordinator


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"


class PDVsMonolithicTest(unittest.TestCase):
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

        # 单体主链
        cls.llm = LLM_self(enable_profile=False)

        # 逻辑 PD
        cls.pd = PDCoordinator(cls.config)

    @classmethod
    def tearDownClass(cls):
        for attr in ("llm", "pd", "tokenizer", "config"):
            if hasattr(cls, attr):
                delattr(cls, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def _run_monolithic(self, prompts: list[str], max_tokens: int):
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

        sampling_params = []
        for _ in prompts:
            sampling_params.append(SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            ))

        with torch.inference_mode():
            outputs = self.llm.generate(input_ids, sampling_params=sampling_params)
        return outputs

    def _run_pd(self, prompts: list[str], max_tokens: int):
        return self.pd.generate(
            texts=prompts,
            max_tokens=max_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

    def test_short_prompt_greedy_matches_monolithic(self):
        prompt = "What is a large language model?"
        max_tokens = 32

        mono_token_ids = self._run_monolithic([prompt], max_tokens)[0]
        pd_token_ids = self._run_pd([prompt], max_tokens)[0]

        self.assertEqual(mono_token_ids, pd_token_ids)

    def test_medium_prompt_greedy_matches_monolithic(self):
        prompt = "How does a transformer model work?"
        max_tokens = 32

        mono_token_ids = self._run_monolithic([prompt], max_tokens)[0]
        pd_token_ids = self._run_pd([prompt], max_tokens)[0]

        self.assertEqual(mono_token_ids, pd_token_ids)

    def test_multi_prompt_greedy_matches_monolithic(self):
        prompts = [
            "What is a large language model?",
            "How does a transformer model work?",
        ]
        max_tokens = 32

        mono_outputs = self._run_monolithic(prompts, max_tokens)
        pd_outputs = self._run_pd(prompts, max_tokens)

        self.assertEqual(len(mono_outputs), len(pd_outputs))
        self.assertEqual(mono_outputs, pd_outputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
