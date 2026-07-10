# 将pd串联
from typing import List

from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from .prefill_engine import PrefillEngine
from .decode_engine import DecodeEngine

class PDCoordinator:
    """
    先调 prefill
    再把 payload 交给 decode
    第一版先让两边共用同一个 model_runner
    这是“逻辑 PD”
    后面再拆成不同 worker / 不同 runner
    """
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        # 第一版：共用同一个 runner / 同一份 kv cache / 同一个 block manager
        self.model_runner = ModelRunner(config)

        self.prefill_engine = PrefillEngine(config, self.tokenizer, self.model_runner)
        self.decode_engine = DecodeEngine(config, self.tokenizer, self.model_runner)

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
