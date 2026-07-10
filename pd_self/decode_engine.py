# 只负责在已有上下文基础上继续生成

from nanovllm.engine.Sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
import torch

class DecodeEngine:
    """
    从 handoff payload 恢复出 Sequence
    挂到自己的 scheduler / running 队列
    继续 decode 到结束
    """
    def __init__(self, config, tokenizer, model_runner):
        self.config = config
        self.tokenizer = tokenizer
        self.model_runner = model_runner
        self.scheduler = Scheduler(config, model_runner.block_manager)

    def restore_sequence(self, payload) -> Sequence:
        # 把从prefill传过来的payload参数取到seq里面
        seq = Sequence(
            seq_idx=payload.seq_idx,
            token_ids=list(payload.token_ids),
            block_size=self.config.block_size,
        )

        seq.num_prompt_tokens = payload.num_prompt_tokens
        seq.num_cached_tokens = payload.num_cached_tokens
        seq.block_table = list(payload.block_table)
        seq.temperature = payload.temperature
        seq.max_tokens = payload.max_tokens
        seq.ignore_eos = payload.ignore_eos
        seq.status = SequenceStatus.RUNNING
        return seq
    
    
    def run_decode(self, payloads):
        results = {}
        restored = []

        for payload in payloads:
            if payload.finished:
                results[payload.seq_idx] = list(payload.token_ids)
                continue
        
            seq = self.restore_sequence(payload)
            self.scheduler.running.append(seq)
            restored.append(seq)

        with torch.inference_mode():
            while not self.scheduler.is_finished():
                scheduled = self.scheduler.schedule()
                token_ids, seq_need_compute_logits = self.model_runner.run(scheduled)
                self.scheduler.postprocess(scheduled, token_ids, seq_need_compute_logits)

        for seq in restored:
            results[seq.seq_idx] = list(seq.token_ids)

        # 按照payloads顺序排序
        ordered_ids = [payload.seq_idx for payload in payloads]
        return [results[seq_idx] for seq_idx in ordered_ids]
    
        
