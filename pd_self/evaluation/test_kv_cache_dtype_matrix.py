import gc
import unittest

import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.attention import PagedAttention
from pd_self.online_coordinator import OnlinePDCoordinator
from pd_self.request_state import RequestStatus


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"


class KVCacheDTypeMatrixTest(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _base_config(self, **kwargs):
        params = dict(
            model_path=MODEL_PATH,
            device="cuda:0",
            max_num_seqs=4,
            max_num_batched_tokens=512,
            max_model_len=512,
            block_size=256,
            num_blocks=64,
        )
        params.update(kwargs)
        return Config(**params)

    def test_config_resolves_kv_cache_dtype_aliases(self):
        cases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }

        for alias, expected_dtype in cases.items():
            with self.subTest(alias=alias):
                config = self._base_config(
                    kv_cache_dtype=alias,
                    attention_compute_dtype=alias,
                )
                self.assertEqual(config.kv_cache_dtype, expected_dtype)
                self.assertEqual(config.attention_compute_dtype, expected_dtype)

    def test_default_kv_cache_dtype_keeps_bf16_mainline(self):
        config = self._base_config()

        self.assertEqual(config.block_size, 256)
        self.assertEqual(config.kv_cache_dtype, torch.bfloat16)
        self.assertEqual(config.attention_compute_dtype, torch.bfloat16)

    def test_store_kv_cache_casts_to_cache_dtype(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        attention = PagedAttention(num_heads=8, num_kv_heads=8, head_dim=16)
        attention.block_size = 256
        attention.k_cache = torch.zeros(
            1, 256, 8, 16,
            dtype=torch.float16,
            device=device,
        )
        attention.v_cache = torch.zeros_like(attention.k_cache)

        k = torch.ones(1, 8, 16, dtype=torch.bfloat16, device=device)
        v = torch.ones(1, 8, 16, dtype=torch.bfloat16, device=device) * 2
        slot_mapping = torch.tensor([0], dtype=torch.int32, device=device)

        attention.store_kv_cache_torch(k, v, slot_mapping)

        self.assertEqual(attention.k_cache.dtype, torch.float16)
        self.assertEqual(attention.v_cache.dtype, torch.float16)
        self.assertEqual(attention.k_cache[0, 0].dtype, torch.float16)
        self.assertTrue(torch.allclose(attention.k_cache[0, 0].float(), k[0].float()))
        self.assertTrue(torch.allclose(attention.v_cache[0, 0].float(), v[0].float()))

    def test_model_runner_binds_bf16_kv_cache_dtype_to_attention_layers(self):
        config = self._base_config(
            kv_cache_dtype="bf16",
            attention_compute_dtype="bf16",
        )
        runner = ModelRunner(config)
        try:
            self.assertEqual(runner.kv_cache.dtype, torch.bfloat16)
            for layer in runner.model.layers:
                self.assertEqual(layer.p_attn.k_cache.dtype, torch.bfloat16)
                self.assertEqual(layer.p_attn.v_cache.dtype, torch.bfloat16)
                self.assertEqual(layer.p_attn.kv_cache_dtype, torch.bfloat16)
                self.assertEqual(layer.p_attn.attention_compute_dtype, torch.bfloat16)
        finally:
            del runner

    def test_online_pd_runs_with_fp16_kv_cache_and_fp16_compute(self):
        config = self._base_config(
            kv_cache_dtype="fp16",
            attention_compute_dtype="fp16",
        )
        engine = OnlinePDCoordinator(config, kv_backend="dict")
        try:
            self.assertEqual(engine.prefill_engine.model_runner.kv_cache.dtype, torch.float16)
            self.assertEqual(engine.decode_engine.model_runner.kv_cache.dtype, torch.float16)

            request_id = engine.submit(
                "What is KV cache?",
                max_tokens=2,
                temperature=0.0,
                ignore_eos=True,
            )
            events = list(engine.stream(request_id))
            token_events = [event for event in events if event.token_id is not None]

            self.assertEqual(engine.requests[request_id].status, RequestStatus.FINISHED)
            self.assertEqual(len(token_events), 2)
        finally:
            del engine

    def test_flashattn_guard_rejects_fp32_kv_cache(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        attention = PagedAttention(num_heads=8, num_kv_heads=8, head_dim=16)
        attention.k_cache = torch.zeros(
            1, 256, 8, 16,
            dtype=torch.float32,
            device=device,
        )
        attention.v_cache = torch.zeros_like(attention.k_cache)
        attention.attention_compute_dtype = torch.float32

        q = torch.zeros(1, 8, 16, dtype=torch.float32, device=device)
        k = torch.zeros(1, 8, 16, dtype=torch.float32, device=device)
        v = torch.zeros(1, 8, 16, dtype=torch.float32, device=device)

        with self.assertRaisesRegex(RuntimeError, "fp16/bf16 kv cache"):
            attention.prefill_flashattn(q, k, v, context=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
