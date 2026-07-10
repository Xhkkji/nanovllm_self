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

SHORT_PROMPT = "What is a large language model?"
MEDIUM_PROMPT = "How does a transformer model work?"

PREFIX_PROMPT = (
    "Explain what a large language model is and how it learns patterns from data."
)
TARGET_PROMPT = (
    "Explain what a large language model is and how it learns patterns from data. "
    "Give two concrete examples."
)

SHARED_PREFIX = (
    "Large language model inference systems usually separate prefill and decode. "
    "During prefill, the model processes the full prompt and writes key value tensors into a paged KV cache. "
    "During decode, the system only computes the newest query token and attends to the full cached history. "
    "A production runtime must balance correctness, memory efficiency, batch scheduling, and latency stability. "
    "Prefix caching reuses identical full blocks from previous requests so repeated prompt prefixes do not need to be recomputed. "
    "This optimization is especially useful for agent systems, tool prompts, long system prompts, and multi-turn workflows. "
)

LONG_SHARED_PREFIX = " ".join([SHARED_PREFIX] * 10)
LONG_TARGET_A = LONG_SHARED_PREFIX + " Now explain why prefix caching helps reduce TTFT in an inference server."
LONG_TARGET_B = LONG_SHARED_PREFIX + " Now explain why decode latency matters for interactive chat applications."


class P0MixedWorkloadRegressionTest(unittest.TestCase):
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

    def _reset_engine_state(self):
        self.model_runner.kv_cache.zero_()
        bm = self.model_runner.block_manager
        bm.hash_to_block_id.clear()
        bm.free_blocks_idx.clear()
        bm.free_blocks_idx.extend(range(bm.num_blocks))
        bm.used_blocks_idx.clear()
        for block in bm.blocks:
            block.reset()

    def _set_combo(self, prefill_backend, decode_backend, cuda_graph=False):
        for layer in self.model_runner.model.layers:
            layer.p_attn.prefill_backend = prefill_backend
            layer.p_attn.decode_backend = decode_backend
            layer.p_attn.forward_backend = (
                "flashattn"
                if prefill_backend == "flashattn" and decode_backend == "flashattn"
                else "torch"
            )
        self.model_runner.enable_cuda_graph = cuda_graph
        if cuda_graph:
            if not self.model_runner.graph_states:
                self.model_runner.init_graph_states()

    def _make_seq(self, text, seq_idx, max_new_tokens):
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        seq = Sequence(seq_idx=seq_idx, token_ids=token_ids, block_size=self.config.block_size)
        seq.temperature = 0.0
        seq.max_tokens = max_new_tokens
        seq.ignore_eos = True
        return seq

    def _run_self(self, prompts, max_new_tokens, prefill_backend, decode_backend, cuda_graph=False):
        self._reset_engine_state()
        self._set_combo(prefill_backend, decode_backend, cuda_graph=cuda_graph)
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

    def _run_hf(self, prompts, max_new_tokens):
        outputs = []
        for prompt in prompts:
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
            outputs.append(output[0].tolist())
        return outputs

    def _assert_outputs_match_hf(self, prompts, self_outputs, hf_outputs):
        for idx, prompt in enumerate(prompts):
            self.assertEqual(
                self_outputs[idx],
                hf_outputs[idx],
                msg=(
                    f"seq{idx} mismatch\n"
                    f"prompt: {prompt}\n"
                    f"self: {self_outputs[idx]}\n"
                    f"hf:   {hf_outputs[idx]}"
                ),
            )

    def _run_prefix_long_result(self, target_prompt):
        self._reset_engine_state()
        self._set_combo("flashattn", "flashattn", cuda_graph=False)

        prefix_seq = Sequence(
            seq_idx=0,
            token_ids=self.tokenizer(LONG_SHARED_PREFIX, add_special_tokens=False)["input_ids"],
            block_size=self.config.block_size,
        )
        prefix_seq.temperature = 0.0
        prefix_seq.max_tokens = 1
        prefix_seq.ignore_eos = True
        self.model_runner.block_manager.allocate(prefix_seq)
        prefix_seq.num_new_tokens = len(prefix_seq) - prefix_seq.num_cached_tokens
        prefix_input_ids, prefix_positions, prefix_context = self.model_runner.prepare_model_input([prefix_seq])
        with torch.inference_mode():
            _ = self.model_runner.model(prefix_input_ids, prefix_positions, prefix_context)
        prefix_seq.num_cached_tokens += prefix_seq.num_new_tokens
        prefix_seq.num_new_tokens = 0

        target_ids = self.tokenizer(target_prompt, add_special_tokens=False)["input_ids"]
        target_seq = Sequence(seq_idx=1, token_ids=target_ids, block_size=self.config.block_size)
        target_seq.temperature = 0.0
        target_seq.max_tokens = 1
        target_seq.ignore_eos = True
        self.model_runner.block_manager.allocate(target_seq)
        target_seq.num_new_tokens = len(target_seq) - target_seq.num_cached_tokens
        input_ids, positions, context = self.model_runner.prepare_model_input([target_seq])
        with torch.inference_mode():
            logits = self.model_runner.model(input_ids, positions, context)

        expected_cached = (len(prefix_seq.token_ids) // self.config.block_size) * self.config.block_size
        return {
            "next_token": int(logits.argmax(dim=-1).item()),
            "num_cached_tokens": int(target_seq.num_cached_tokens),
            "input_ids_len": int(input_ids.numel()),
            "full_len": int(len(target_seq)),
            "has_block_tables": context.block_tables is not None,
            "q_total": int(context.cu_seqlens_q[-1].item()),
            "k_total": int(context.cu_seqlens_k[-1].item()),
            "expected_cached": expected_cached,
        }

    def _hf_next_token(self, prompt):
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.inference_mode():
            logits = self.hf(input_ids=input_ids, attention_mask=attention_mask, use_cache=True).logits[:, -1, :]
        return int(logits.argmax(dim=-1).item())

    @unittest.skip("unified torch fallback path is still being refactored; flash path is the current validated mainline")
    def test_torch_prefill_flash_decode_equal_len(self):
        self_outputs = self._run_self(PROMPTS_EQUAL_LEN, 16, "torch", "flashattn", cuda_graph=False)
        hf_outputs = self._run_hf(PROMPTS_EQUAL_LEN, 16)
        self._assert_outputs_match_hf(PROMPTS_EQUAL_LEN, self_outputs, hf_outputs)

    @unittest.skip("unified torch fallback path is still being refactored; flash path is the current validated mainline")
    def test_torch_prefill_flash_decode_mixed_len(self):
        self_outputs = self._run_self(PROMPTS_MIXED_LEN, 16, "torch", "flashattn", cuda_graph=False)
        hf_outputs = self._run_hf(PROMPTS_MIXED_LEN, 16)
        self._assert_outputs_match_hf(PROMPTS_MIXED_LEN, self_outputs, hf_outputs)

    def test_flash_prefill_flash_decode_equal_len(self):
        self_outputs = self._run_self(PROMPTS_EQUAL_LEN, 16, "flashattn", "flashattn", cuda_graph=False)
        hf_outputs = self._run_hf(PROMPTS_EQUAL_LEN, 16)
        self._assert_outputs_match_hf(PROMPTS_EQUAL_LEN, self_outputs, hf_outputs)

    def test_flash_prefill_flash_decode_mixed_len(self):
        self_outputs = self._run_self(PROMPTS_MIXED_LEN, 16, "flashattn", "flashattn", cuda_graph=False)
        hf_outputs = self._run_hf(PROMPTS_MIXED_LEN, 16)
        self._assert_outputs_match_hf(PROMPTS_MIXED_LEN, self_outputs, hf_outputs)

    def test_long_prefix_prefill_matches_hf_and_hits_cache(self):
        result_a = self._run_prefix_long_result(LONG_TARGET_A)
        result_b = self._run_prefix_long_result(LONG_TARGET_B)
        hf_next_a = self._hf_next_token(LONG_TARGET_A)
        hf_next_b = self._hf_next_token(LONG_TARGET_B)

        for result, hf_next in ((result_a, hf_next_a), (result_b, hf_next_b)):
            self.assertGreater(result["num_cached_tokens"], 0)
            self.assertEqual(result["num_cached_tokens"], result["expected_cached"])
            self.assertEqual(result["num_cached_tokens"] % self.config.block_size, 0)
            self.assertTrue(result["has_block_tables"])
            self.assertGreater(result["k_total"], result["q_total"])
            self.assertEqual(result["input_ids_len"], result["full_len"] - result["num_cached_tokens"])
            self.assertEqual(result["next_token"], hf_next)


def main():
    parser = argparse.ArgumentParser(description="P0 mixed workload regression checks for nanovllm_self.")
    parser.add_argument("-k", "--keyword", default=None, help="Only run tests whose name contains this keyword.")
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(P0MixedWorkloadRegressionTest)

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
