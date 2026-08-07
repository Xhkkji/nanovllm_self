import unittest

from nanovllm.config import Config
from pd_self.online_coordinator import OnlinePDCoordinator
from pd_self.request_state import RequestStatus


class OnlinePDCoordinatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(
            model_path="/home/xhk/model/Qwen3-0.6B/",
            device="cuda:0",
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            block_size=256,
            num_blocks=256,
        )
        # 多个测试复用同一个 coordinator，避免重复加载两份模型。
        cls.engine = OnlinePDCoordinator(cls.config, kv_backend="dict")

    def test_single_request_stream_to_finish(self):
        request_id = self.engine.submit(
            "What is a large language model?",
            max_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )

        events = list(self.engine.stream(request_id))
        state = self.engine.requests[request_id]

        self.assertEqual(state.status, RequestStatus.FINISHED)
        self.assertGreater(
            len([event for event in events if event.token_id is not None]),
            0,
        )
        self.assertTrue(events[-1].finished)

    def test_abort_request(self):
        request_id = self.engine.submit(
            "Explain prefix caching in LLM serving.",
            max_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        )

        self.engine.step()
        self.assertTrue(self.engine.abort_request(request_id))

        state = self.engine.requests[request_id]
        self.assertEqual(state.status, RequestStatus.ABORTED)
        events = self.engine.poll_events(request_id)
        self.assertTrue(events)
        self.assertTrue(events[-1].finished)

    def test_staggered_arrival(self):
        request_id_a = self.engine.submit(
            "What is a large language model?",
            max_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )

        for _ in range(2):
            self.engine.step()

        request_id_b = self.engine.submit(
            "How does a transformer model work?",
            max_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )

        for _ in range(1000):
            self.engine.step()
            if (
                self.engine.is_finished(request_id_a)
                and self.engine.is_finished(request_id_b)
            ):
                break

        self.assertTrue(self.engine.is_finished(request_id_a))
        self.assertTrue(self.engine.is_finished(request_id_b))
        self.assertEqual(
            self.engine.requests[request_id_a].status,
            RequestStatus.FINISHED,
        )
        self.assertEqual(
            self.engine.requests[request_id_b].status,
            RequestStatus.FINISHED,
        )

    def test_online_scheduler_metrics_are_recorded(self):
        request_id = self.engine.submit(
            "What is KV cache in transformer inference?",
            max_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )

        self.engine.step()
        # 【第四章收口改动】验证在线调度观测接口可用；这里只测字段存在和基本范围，不绑定具体调度策略。
        metrics = self.engine.get_last_online_scheduler_metrics()

        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.step_id, 0)
        self.assertGreaterEqual(metrics.emitted_events, 0)
        self.assertGreaterEqual(metrics.prefill_used_blocks, 0)
        self.assertGreaterEqual(metrics.prefill_free_blocks, 0)
        self.assertGreaterEqual(metrics.decode_used_blocks, 0)
        self.assertGreaterEqual(metrics.decode_free_blocks, 0)
        self.assertIsInstance(metrics.prefill_schedule_reason, str)
        self.assertIsInstance(metrics.decode_schedule_reason, str)

        self.engine.abort_request(request_id)


if __name__ == "__main__":
    unittest.main()
