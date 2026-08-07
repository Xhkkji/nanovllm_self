import gc
import unittest

import torch

from nanovllm.config import Config
from pd_self.coordinator import PDCoordinator
from pd_self.online_coordinator import OnlinePDCoordinator
from pd_self.request_state import RequestStatus


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"


class OnlinePDCorrectnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(
            model_path=MODEL_PATH,
            device="cuda:0",
            max_num_seqs=64,
            max_num_batched_tokens=4096,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            block_size=256,
            num_blocks=256,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    def _run_offline_pd(self, prompts: list[str], max_tokens: int):
        pd = PDCoordinator(self.config, kv_backend="dict")
        try:
            return pd.generate(
                texts=prompts,
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
            )
        finally:
            del pd
            gc.collect()
            torch.cuda.empty_cache()

    def _run_online_pd(self, prompts: list[str], max_tokens: int):
        online = OnlinePDCoordinator(self.config, kv_backend="dict")
        try:
            request_ids = [
                online.submit(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    ignore_eos=True,
                )
                for prompt in prompts
            ]

            for _ in range(1000):
                online.step()
                if all(online.is_finished(request_id) for request_id in request_ids):
                    break

            for request_id in request_ids:
                self.assertEqual(
                    online.requests[request_id].status,
                    RequestStatus.FINISHED,
                )

            return [
                online.requests[request_id].token_ids
                for request_id in request_ids
            ]
        finally:
            del online
            gc.collect()
            torch.cuda.empty_cache()

    def _run_online_pd_staggered(self, prompts: list[str], max_tokens: int):
        online = OnlinePDCoordinator(self.config, kv_backend="dict")
        try:
            request_id_a = online.submit(
                prompts[0],
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
            )

            for _ in range(2):
                online.step()

            request_id_b = online.submit(
                prompts[1],
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
            )

            request_ids = [request_id_a, request_id_b]
            for _ in range(1000):
                online.step()
                if all(online.is_finished(request_id) for request_id in request_ids):
                    break

            for request_id in request_ids:
                self.assertEqual(
                    online.requests[request_id].status,
                    RequestStatus.FINISHED,
                )

            return [
                online.requests[request_id].token_ids
                for request_id in request_ids
            ]
        finally:
            del online
            gc.collect()
            torch.cuda.empty_cache()

    def test_single_short_prompt_matches_offline_pd(self):
        prompts = ["What is a large language model?"]
        max_tokens = 4

        expected = self._run_offline_pd(prompts, max_tokens)
        actual = self._run_online_pd(prompts, max_tokens)

        self.assertEqual(actual, expected)

    def test_multi_prompt_matches_offline_pd(self):
        prompts = [
            "What is a large language model?",
            "How does a transformer model work?",
        ]
        max_tokens = 4

        expected = self._run_offline_pd(prompts, max_tokens)
        actual = self._run_online_pd(prompts, max_tokens)

        self.assertEqual(actual, expected)

    def test_staggered_arrival_matches_offline_pd(self):
        prompts = [
            "What is a large language model?",
            "How does a transformer model work?",
        ]
        max_tokens = 4

        expected = self._run_offline_pd(prompts, max_tokens)
        actual = self._run_online_pd_staggered(prompts, max_tokens)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
