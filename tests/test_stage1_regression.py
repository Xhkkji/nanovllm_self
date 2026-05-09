import argparse
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nanovllm.engine.Sequence import Sequence
from nanovllm.engine.block_manager import block_manager as BlockManager
from nanovllm.models.qwen3 import Qwen3Model


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEVICE = "cuda:0"
PROMPT = "Introduce the acg in China where nearby Japan."
CHINESE_PROMPT = "请用自然、生动、带有画面感的语言，介绍中国的ACG文化，并简要对比日本ACG文化。"


class Stage1RegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        cls.config = AutoConfig.from_pretrained(MODEL_PATH)
        cls.hf = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE
        ).eval()
        cls.self_model = Qwen3Model(cls.config).to(DEVICE).eval()
        cls.input_ids = cls.tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)
        cls.chinese_input_ids = cls.tokenizer(
            CHINESE_PROMPT, return_tensors="pt"
        )["input_ids"].to(DEVICE)

    def _hf_prefill_topk(self, input_ids, k=10):
        with torch.no_grad():
            out = self.hf(input_ids, use_cache=True)
            logits = out.logits[:, -1, :]
            topk_ids = torch.topk(logits, k=k, dim=-1).indices[0].tolist()
        return topk_ids

    def _self_prefill_topk(self, input_ids, block_size=16, k=10):
        bm = BlockManager(
            num_blocks=100,
            block_size=block_size,
            num_layers=self.self_model.num_layers,
            num_kv_heads=self.self_model.num_kv_heads,
            head_dim=self.self_model.head_dim,
        )
        seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
        seq.block_size = block_size
        seq.block_table = bm.allocate_with_prefill(seq)
        positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            out = self.self_model(
                input_ids[0],
                positions=positions,
                block_manager=bm,
                seq=seq,
                is_prefill=True,
            )
            logits = out[-1, :].unsqueeze(0)
            topk_ids = torch.topk(logits, k=k, dim=-1).indices[0].tolist()
        return topk_ids

    def _hf_greedy_ids(self, input_ids, steps):
        tokens = input_ids.clone()
        past = None
        generated = []
        with torch.no_grad():
            for _ in range(steps):
                current = tokens if past is None else tokens[:, -1:]
                out = self.hf(current, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                generated.append(int(next_token.item()))
        return generated

    def _self_greedy_ids(self, input_ids, steps, block_size=16):
        bm = BlockManager(
            num_blocks=100,
            block_size=block_size,
            num_layers=self.self_model.num_layers,
            num_kv_heads=self.self_model.num_kv_heads,
            head_dim=self.self_model.head_dim,
        )
        seq = Sequence(seq_idx=0, token_ids=input_ids[0].tolist())
        seq.block_size = block_size
        seq.block_table = bm.allocate_with_prefill(seq)

        tokens = input_ids.clone()
        generated = []
        is_prefill = True
        with torch.no_grad():
            for _ in range(steps):
                if is_prefill:
                    current = tokens[0]
                    positions = torch.arange(0, len(seq.token_ids), device=DEVICE).unsqueeze(0)
                    out = self.self_model(
                        current,
                        positions=positions,
                        block_manager=bm,
                        seq=seq,
                        is_prefill=True,
                    )
                    logits = out[-1, :].unsqueeze(0)
                    is_prefill = False
                else:
                    current = tokens[0, -1:]
                    positions = torch.tensor([[len(seq.token_ids) - 1]], device=DEVICE)
                    out = self.self_model(
                        current,
                        positions=positions,
                        block_manager=bm,
                        seq=seq,
                        is_prefill=False,
                    )
                    logits = out.unsqueeze(0)

                next_token = logits.argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)

                token_id = int(next_token[0, 0].item())
                generated.append(token_id)
                seq.append_token(token_id)
                if len(seq.token_ids) > len(seq.block_table) * bm.block_size:
                    seq.block_table.append(bm.allocate_block(1)[0])

        return generated

    def _assert_token_lists_equal(self, expected, actual, *, label):
        if expected != actual:
            mismatch_idx = next(
                idx for idx, (lhs, rhs) in enumerate(zip(expected, actual)) if lhs != rhs
            )
            self.fail(
                f"{label} diverged at generated token {mismatch_idx}: "
                f"hf={expected[mismatch_idx]} self={actual[mismatch_idx]}"
            )

    def test_prefill_topk_matches_hf(self):
        self_topk = self._self_prefill_topk(self.input_ids, k=10)
        hf_topk = self._hf_prefill_topk(self.input_ids, k=10)
        self.assertEqual(self_topk[0], hf_topk[0])
        self.assertEqual(set(self_topk), set(hf_topk))

    def test_greedy_first_64_tokens_match_hf(self):
        steps = 64
        self._assert_token_lists_equal(
            self._hf_greedy_ids(self.input_ids, steps=steps),
            self._self_greedy_ids(self.input_ids, steps=steps),
            label="english_greedy_64",
        )

    def test_block_growth_does_not_change_greedy_result(self):
        steps = 64
        small_block = self._self_greedy_ids(self.input_ids, steps=steps, block_size=4)
        default_block = self._self_greedy_ids(self.input_ids, steps=steps, block_size=16)
        self._assert_token_lists_equal(
            default_block,
            small_block,
            label="english_block_growth",
        )

    def test_chinese_prefill_topk_matches_hf(self):
        self_topk = self._self_prefill_topk(self.chinese_input_ids, k=10)
        hf_topk = self._hf_prefill_topk(self.chinese_input_ids, k=10)
        self.assertEqual(self_topk[0], hf_topk[0])
        self.assertEqual(set(self_topk), set(hf_topk))

    def test_chinese_greedy_first_64_tokens_match_hf(self):
        steps = 64
        self._assert_token_lists_equal(
            self._hf_greedy_ids(self.chinese_input_ids, steps=steps),
            self._self_greedy_ids(self.chinese_input_ids, steps=steps),
            label="chinese_greedy_64",
        )

    def test_block_boundaries_match_hf(self):
        steps = 16
        for prompt_len in (15, 16, 17, 31, 32, 33):
            sliced = self.chinese_input_ids[:, :prompt_len]
            self._assert_token_lists_equal(
                self._hf_greedy_ids(sliced, steps=steps),
                self._self_greedy_ids(sliced, steps=steps),
                label=f"boundary_len_{prompt_len}",
            )


def main():
    parser = argparse.ArgumentParser(description="Run stage-1 regression checks for nanovllm_self.")
    parser.add_argument(
        "-k",
        "--keyword",
        default=None,
        help="Only run tests whose name contains this keyword.",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Stage1RegressionTest)

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
