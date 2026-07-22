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

PROMPTS = [
    "What is a large language model?",
    "How does a transformer model work?",
]


class FlashDecodeCompareWithHFTest(unittest.TestCase):
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

        # 当前统一主链由 forward_backend 控制。
        # 这里显式固定为 flashattn，验证当前 paged-flash 主链。
        for layer in cls.model_runner.model.layers:
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

    def _make_seq(self, text, seq_idx, max_new_tokens):
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        seq = Sequence(seq_idx=seq_idx, token_ids=token_ids, block_size=self.config.block_size)
        seq.temperature = 0.0
        seq.max_tokens = max_new_tokens
        seq.ignore_eos = True
        return seq

    def _run_self(self, prompts, max_new_tokens):
        self._reset_runner_state()
        scheduler = Scheduler(self.config, self.model_runner.block_manager)
        seqs = [self._make_seq(text, i, max_new_tokens) for i, text in enumerate(prompts)]

        for seq in seqs:
            scheduler.add(seq)

        with torch.inference_mode():
            while not scheduler.is_finished():
                seq_list = scheduler.schedule()
                token_ids, seq_need_compute_logits = self.model_runner.run(seq_list)
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                token_ids = [int(x) for x in token_ids]
                scheduler.postprocess(seq_list, token_ids, seq_need_compute_logits)

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

    def test_single_seq_greedy_matches_hf(self):
        max_new_tokens = 8
        prompts = [PROMPTS[0]]
        self_outputs = self._run_self(prompts, max_new_tokens)
        hf_outputs = [self._run_hf_single(prompt, max_new_tokens) for prompt in prompts]
        self.assertEqual(self_outputs[0], hf_outputs[0])

    def test_multi_seq_greedy_matches_hf(self):
        max_new_tokens = 4
        prompts = PROMPTS
        self_outputs = self._run_self(prompts, max_new_tokens)
        hf_outputs = [self._run_hf_single(prompt, max_new_tokens) for prompt in prompts]
        self.assertEqual(self_outputs, hf_outputs)


def main():
    parser = argparse.ArgumentParser(description="Compare torch-prefill + flash-decode outputs with HF.")
    parser.add_argument("-k", "--keyword", default=None)
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(FlashDecodeCompareWithHFTest)

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
