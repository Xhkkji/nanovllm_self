# 只负责把请求推进到 prefill_done
from typing import List

from nanovllm.engine.Sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from .payload import HandoffPayload
import torch

class PrefillEngine:
    """
    职责非常单一：
    接收文本或 token
    构造 Sequence
    分配 block
    一直跑到 seq.is_prefill_done
    返回 HandoffPayload

    注意：
    第一版先只做单请求
    这样最稳
    """
    def __init__(self, config, tokenizer, model_runner):
        self.config = config
        self.tokenizer = tokenizer
        self.model_runner = model_runner
        self.scheduler = Scheduler(config, model_runner.block_manager)

    def _build_sequence(self, text: str, seq_idx: int, temperature: float, max_tokens: int, ignore_eos: bool) -> Sequence:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        seq = Sequence(seq_idx=seq_idx, token_ids=token_ids, block_size=self.config.block_size)
        seq.temperature = temperature
        seq.max_tokens = max_tokens
        seq.ignore_eos = ignore_eos
        return seq
    
    def build_sequences(self,
        texts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 64,
        ignore_eos: bool = True,
        start_seq_id: int = 0,
    ) -> List[Sequence]:
        seqs = []
        for i, text in enumerate(texts):
            seq = self._build_sequence(
                text=text,
                seq_idx=start_seq_id + i,
                temperature=temperature,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )
            seqs.append(seq)
        return seqs
    
    def _make_payload(self, seq, finished):
        return HandoffPayload(
            seq_idx=seq.seq_idx,
            token_ids=list(seq.token_ids),
            num_prompt_tokens=seq.num_prompt_tokens,
            num_cached_tokens=seq.num_cached_tokens,
            block_table=list(seq.block_table),
            temperature=seq.temperature,
            max_tokens=seq.max_tokens,
            ignore_eos=seq.ignore_eos,
            finished=finished,
        )
    
    def _is_handoff_ready(self, seq: Sequence) -> bool:
        # prompt 已经全部写入 KV，且第一枚生成 token 已经产出
        return seq.is_prefill_done and seq.num_completion_tokens >= 1
    
    def run_prefill(self, texts: List[str], temperature: float = 0.0, max_tokens: int = 64, ignore_eos: bool = True, start_seq_id: int=0) -> List[HandoffPayload]:
        # 目前是单seq单请求模式
        # 推进到多seq
        # handoff 条件：
        # 1. prompt 已经全部 cached
        # 2. 第一枚生成 token 已经 append 到 token_ids
                
        seqs = self.build_sequences(texts, temperature, max_tokens, ignore_eos, start_seq_id)
        for seq in seqs:
            self.scheduler.add(seq)
        total = len(seqs)
        payloads = {}
        handed_off = set()

        with torch.inference_mode():
            while len(handed_off) < total:
                scheduled = self.scheduler.schedule()
                token_ids, seq_need_compute_logits = self.model_runner.run(scheduled)
                
                self.scheduler.postprocess(scheduled, token_ids, seq_need_compute_logits)

                # 扫描本轮参与过的 seq，哪些已经可以从 prefill 侧摘走
                for seq in scheduled:
                    if seq.seq_idx in handed_off:
                        continue
                    
                    if seq.status == SequenceStatus.FINISHED:
                        payloads[seq.seq_idx] = self._make_payload(seq, finished=True)
                        handed_off.add(seq.seq_idx)
                        continue
                    
                    if self._is_handoff_ready(seq):
                        # 逻辑 PD：不释放 block，不反分配，只是从 prefill scheduler 的 running 中摘掉
                        if seq in self.scheduler.running:
                            self.scheduler.running.remove(seq)
                        
                        payloads[seq.seq_idx] = self._make_payload(seq, finished=False)
                        handed_off.add(seq.seq_idx)
        # 按原顺序返回
        return [payloads[seq.seq_idx] for seq in seqs]
                    
