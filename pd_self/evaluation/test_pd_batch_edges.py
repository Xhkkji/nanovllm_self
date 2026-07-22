import unittest
import gc

import torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.Sequence import SequenceStatus
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams
from pd_self.coordinator import PDCoordinator
from pd_self.kv_connector import KVConnector
from pd_self.kv_store import DictKVStoreBackend
from pd_self.prefill_engine import PrefillEngine


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"


class PDBatchEdgeCasesTest(unittest.TestCase):
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

        cls.llm = LLM_self(enable_profile=False)
        cls.pd = PDCoordinator(cls.config)

    @classmethod
    def tearDownClass(cls):
        for attr in ("llm", "pd", "tokenizer", "config"):
            if hasattr(cls, attr):
                delattr(cls, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def _payload_id(self, payload):
        if hasattr(payload, "seq_id"):
            return payload.seq_id
        return payload.seq_idx

    def _run_monolithic(self, prompts: list[str], max_tokens: int):
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]

        sampling_params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
            for _ in prompts
        ]

        with torch.inference_mode():
            outputs = self.llm.generate(input_ids, sampling_params=sampling_params)
        return outputs

    def _run_monolithic_serial(self, prompts: list[str], max_tokens):
        """
        当前 LLM_self.generate 的 batch 接口只接收 padding 后的 input_ids，
        没有额外的 attention_mask / prompt_lens 来告诉引擎每条样本的真实长度。
        因此混合长短 prompt 时，直接 batch 化会把 pad token 当成真实 prompt token。

        对 mixed-length case，测试时改用逐条单独跑 monolithic，作为更可靠的参考输出。
        """
        if isinstance(max_tokens, int):
            max_tokens = [max_tokens] * len(prompts)

        outputs = []
        for prompt, cur_max_tokens in zip(prompts, max_tokens):
            encoded = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            sampling_params = [
                SamplingParams(
                    temperature=0.0,
                    max_tokens=cur_max_tokens,
                    ignore_eos=True,
                )
            ]
            with torch.inference_mode():
                output = self.llm.generate(input_ids, sampling_params=sampling_params)[0]
            outputs.append(output)
        return outputs

    def _run_pd(self, prompts: list[str], max_tokens: int):
        return self.pd.generate(
            texts=prompts,
            max_tokens=max_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

    def _build_prefill_engine(self):
        model_runner = ModelRunner(self.config)
        kv_store_backend = DictKVStoreBackend()
        kv_connector = KVConnector(
            config=self.config,
            role="producer",
            engine_id="eval-prefill",
            kv_store_backend=kv_store_backend,
        )
        kv_connector.register_model_runner(model_runner)
        return PrefillEngine(self.config, self.tokenizer, model_runner, kv_connector)

    @unittest.expectedFailure
    def test_mixed_prompt_lengths_match_monolithic(self):
        """
        已知问题回归测试：
        mixed-length batch 在当前 PD 路径下会在较长序列的 decode 阶段出现偏差。

        这不是 evaluation 参考构造的问题：
        - 短序列可以与 serial monolithic 完全对齐
        - 长序列在 handoff 后若干步 batched decode 之后开始偏离
        """
        prompts = [
            "What is a large language model?",
            (
                "Explain how a transformer works in detail, including token embeddings, "
                "self-attention, multi-head attention, feed-forward layers, residual "
                "connections, and why positional information is needed."
            ),
        ]
        max_tokens = 32

        mono_outputs = self._run_monolithic_serial(prompts, max_tokens)
        pd_outputs = self._run_pd(prompts, max_tokens)

        self.assertEqual(len(mono_outputs), len(pd_outputs))
        self.assertEqual(mono_outputs, pd_outputs)

    def test_batch_max_tokens_one_finishes_in_prefill(self):
        prompts = [
            "What is a large language model?",
            "How does a transformer model work?",
        ]
        prefill_engine = self._build_prefill_engine()

        payloads = prefill_engine.run_prefill(
            texts=prompts,
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            start_seq_id=100,
        )

        self.assertEqual(len(payloads), len(prompts))

        for i, (prompt, payload) in enumerate(zip(prompts, payloads)):
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            self.assertEqual(self._payload_id(payload), 100 + i)
            self.assertEqual(payload.num_prompt_tokens, len(prompt_ids))
            self.assertEqual(payload.token_ids[: len(prompt_ids)], prompt_ids)
            self.assertEqual(len(payload.token_ids), len(prompt_ids) + 1)
            self.assertTrue(payload.finished)
            self.assertEqual(payload.num_cached_tokens, 0)
            self.assertIsNone(payload.transfer_meta)

    def test_short_prompt_handoffs_no_later_than_long_prompt(self):
        prompts = [
            "What is a large language model?",
            (
                "Explain how a transformer works in detail, including token embeddings, "
                "self-attention, multi-head attention, feed-forward layers, residual "
                "connections, and why positional information is needed."
            ),
        ]
        prefill_engine = self._build_prefill_engine()
        seqs = prefill_engine.build_sequences(
            texts=prompts,
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
            start_seq_id=200,
        )

        for seq in seqs:
            prefill_engine.scheduler.add(seq)

        handoff_step = {}
        step = 0

        with torch.inference_mode():
            while len(handoff_step) < len(seqs):
                scheduled = prefill_engine.scheduler.schedule()
                token_ids, seq_need_compute_logits = prefill_engine.model_runner.run(scheduled)
                prefill_engine.scheduler.postprocess(scheduled, token_ids, seq_need_compute_logits)

                for seq in scheduled:
                    if seq.seq_idx in handoff_step:
                        continue

                    if seq.status == SequenceStatus.FINISHED:
                        handoff_step[seq.seq_idx] = step
                        continue

                    if prefill_engine._is_handoff_ready(seq):
                        handoff_step[seq.seq_idx] = step
                        if seq in prefill_engine.scheduler.running:
                            prefill_engine.scheduler.running.remove(seq)

                step += 1

        short_seq_id = seqs[0].seq_idx
        long_seq_id = seqs[1].seq_idx

        self.assertIn(short_seq_id, handoff_step)
        self.assertIn(long_seq_id, handoff_step)
        self.assertLessEqual(handoff_step[short_seq_id], handoff_step[long_seq_id])


if __name__ == "__main__":
    unittest.main(verbosity=2)
