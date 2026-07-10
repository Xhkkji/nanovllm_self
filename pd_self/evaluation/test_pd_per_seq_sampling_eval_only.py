import unittest
import gc

import torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.Sequence import SequenceStatus
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams
from pd_self.decode_engine import DecodeEngine
from pd_self.prefill_engine import PrefillEngine


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"


class PDPerSeqSamplingEvalOnlyTest(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        for attr in ("llm", "tokenizer", "config"):
            if hasattr(cls, attr):
                delattr(cls, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def _run_monolithic_serial(self, prompts, max_tokens_list, temperatures, ignore_eos_list):
        outputs = []
        for prompt, max_tokens, temperature, ignore_eos in zip(
            prompts, max_tokens_list, temperatures, ignore_eos_list
        ):
            input_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            sampling_params = [
                SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                )
            ]
            with torch.inference_mode():
                output = self.llm.generate(input_ids, sampling_params=sampling_params)[0]
            outputs.append(output)
        return outputs

    def _run_pd_eval_only(self, prompts, max_tokens_list, temperatures, ignore_eos_list):
        """
        这是 evaluation-only workaround。

        当前公开 PD 接口：
        - PDCoordinator.generate(texts, max_tokens, temperature, ignore_eos)
        - PrefillEngine.run_prefill(texts, max_tokens, temperature, ignore_eos)

        都要求整批请求共享同一套采样参数。

        因此测试里如果想覆盖“同一 batch 内每条请求不同 max_tokens / temperature”，
        就只能绕过公开入口，直接在 evaluation 中手工构造和推进 Sequence。
        """
        model_runner = ModelRunner(self.config)
        prefill_engine = PrefillEngine(self.config, self.tokenizer, model_runner)
        decode_engine = DecodeEngine(self.config, self.tokenizer, model_runner)

        seqs = prefill_engine.build_sequences(
            texts=prompts,
            temperature=0.0,
            max_tokens=max(max_tokens_list),
            ignore_eos=True,
            start_seq_id=0,
        )

        for seq, max_tokens, temperature, ignore_eos in zip(
            seqs, max_tokens_list, temperatures, ignore_eos_list
        ):
            seq.max_tokens = max_tokens
            seq.temperature = temperature
            seq.ignore_eos = ignore_eos
            prefill_engine.scheduler.add(seq)

        payloads = {}
        handed_off = set()

        with torch.inference_mode():
            while len(handed_off) < len(seqs):
                scheduled = prefill_engine.scheduler.schedule()
                token_ids, seq_need_compute_logits = prefill_engine.model_runner.run(scheduled)
                prefill_engine.scheduler.postprocess(scheduled, token_ids, seq_need_compute_logits)

                for seq in scheduled:
                    if seq.seq_idx in handed_off:
                        continue

                    if seq.status == SequenceStatus.FINISHED:
                        payloads[seq.seq_idx] = prefill_engine._make_payload(seq, finished=True)
                        handed_off.add(seq.seq_idx)
                        continue

                    if prefill_engine._is_handoff_ready(seq):
                        if seq in prefill_engine.scheduler.running:
                            prefill_engine.scheduler.running.remove(seq)
                        payloads[seq.seq_idx] = prefill_engine._make_payload(seq, finished=False)
                        handed_off.add(seq.seq_idx)

        ordered_payloads = [payloads[seq.seq_idx] for seq in seqs]
        outputs = decode_engine.run_decode(ordered_payloads)

        del decode_engine
        del prefill_engine
        del model_runner
        gc.collect()
        torch.cuda.empty_cache()
        return outputs

    def test_eval_only_workaround_supports_per_seq_max_tokens(self):
        prompts = [
            "What is a large language model?",
            "How does a transformer model work?",
        ]
        max_tokens_list = [1, 8]
        temperatures = [0.0, 0.0]
        ignore_eos_list = [True, True]

        mono_outputs = self._run_monolithic_serial(
            prompts,
            max_tokens_list=max_tokens_list,
            temperatures=temperatures,
            ignore_eos_list=ignore_eos_list,
        )
        pd_outputs = self._run_pd_eval_only(
            prompts,
            max_tokens_list=max_tokens_list,
            temperatures=temperatures,
            ignore_eos_list=ignore_eos_list,
        )

        self.assertEqual(mono_outputs, pd_outputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
