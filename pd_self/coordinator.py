# 将pd串联
from typing import List

from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner

from .prefill_engine import PrefillEngine
from .decode_engine import DecodeEngine
from .kv_connector import KVConnector
from .kv_store import DictKVStoreBackend, SharedMemoryKVStoreBackend

class PDCoordinator:
    """
    先调 prefill
    再把 payload 交给 decode
    第一版先让两边共用同一个 model_runner
    这是“逻辑 PD”
    后面再拆成不同 worker / 不同 runner

    20260711 关键变化：pd不再共用同一个 runner
    把 KV handoff 从 payload 内嵌数据，改成 connector 管理。
    """
    def __init__(self, config: Config, kv_backend: str = "dict"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        self.prefill_runner = ModelRunner(config)
        self.decode_runner = ModelRunner(config)
        
        if kv_backend == "shared_memory":
            self.kv_store_backend = SharedMemoryKVStoreBackend()
        else:
            # 单进程版共享 store，模拟外部 KV transfer backend
            self.kv_store_backend = DictKVStoreBackend()

        self.prefill_connector = KVConnector(
            config=config,
            role="producer",
            engine_id="prefill-worker-0",
            kv_store_backend=self.kv_store_backend,
        )
        self.decode_connector = KVConnector(
            config=config,
            role="consumer",
            engine_id="decode-worker-0",
            kv_store_backend=self.kv_store_backend,
        )

        self.prefill_connector.register_model_runner(self.prefill_runner)
        self.decode_connector.register_model_runner(self.decode_runner)

        self.prefill_engine = PrefillEngine(
            config=config,
            tokenizer=self.tokenizer,
            model_runner=self.prefill_runner,
            kv_connector=self.prefill_connector,
        )
        self.decode_engine = DecodeEngine(
            config=config,
            tokenizer=self.tokenizer,
            model_runner=self.decode_runner,
            kv_connector=self.decode_connector,
        )

    def generate(
        self, 
        texts: List[str], 
        max_tokens: int = 64, 
        temperature: float = 0.0, 
        ignore_eos: bool = True):
        payloads = self.prefill_engine.run_prefill(
            texts=texts,
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        return self.decode_engine.run_decode(payloads)
