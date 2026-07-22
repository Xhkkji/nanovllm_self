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
    def __init__(self, config, tokenizer, model_runner, kv_connector):
        self.config = config
        self.tokenizer = tokenizer
        self.model_runner = model_runner
        self.kv_connector = kv_connector
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
    
    def _make_payload(self, seq: Sequence, finished: bool) -> HandoffPayload:
        transfer_meta = None
        if not finished:
            transfer_meta = self.kv_connector.save_kv(seq)

        return HandoffPayload(
            seq_idx=seq.seq_idx,
            request_id=f"req-{seq.seq_idx}",
            token_ids=list(seq.token_ids),
            num_prompt_tokens=seq.num_prompt_tokens,
            num_cached_tokens=seq.num_cached_tokens,
            temperature=seq.temperature,
            max_tokens=seq.max_tokens,
            ignore_eos=seq.ignore_eos,
            finished=finished,
            transfer_meta=transfer_meta,    
        )
    
    def _is_handoff_ready(self, seq: Sequence) -> bool:
        # prompt 已经全部写入 KV，且第一枚生成 token 已经产出
        return seq.is_prefill_done and seq.num_completion_tokens >= 1
    
    def add_request(
        self,
        text: str,
        seq_idx: int,
        temperature: float = 0.0,
        max_tokens: int = 64,
        ignore_eos: bool = True,
    ) -> Sequence:
        """
        创建seq并直接把seq加入到调度器里面
        """
        seq = self._build_sequence(
            text=text,
            seq_idx=seq_idx,
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[HandoffPayload]:
        """
        针对self.scheduler
        推进 prefill worker 一轮。

        返回：
        - 本轮已经到达 handoff 边界的 payloads
        - 可能为空
        """
        if self.scheduler.is_finished():
            return []
        
        scheduled = self.scheduler.schedule()
        if not scheduled:
            return []
        
        token_ids, seq_need_compute_logits = self.model_runner.run(scheduled)
        self.scheduler.postprocess(scheduled, token_ids, seq_need_compute_logits)

        payloads = []

        # 扫描本轮参与过的 seq，哪些已经可以从 prefill 侧摘走
        for seq in scheduled:
            if seq.status == SequenceStatus.FINISHED:
                payloads.append(
                    self._make_payload(seq, finished=True)
                )
                continue
            
            if self._is_handoff_ready(seq):
                # 所有prompt已全部写入KV，第一个token已生成
                payload = self._make_payload(seq, finished=False)
                
                if seq in self.scheduler.running:
                    self.scheduler.running.remove(seq)

                # prefill侧kv已导出，释放本地block
                self.model_runner.block_manager.deallocate(seq)
                payloads.append(payload)
        
        return payloads

    def run_prefill(self, texts: List[str], temperature: float = 0.0, max_tokens: int = 64, ignore_eos: bool = True, start_seq_id: int=0):
        """
        离线批处理，直接处理一批text
        在线预处理使用step逻辑，不执行此函数
        推进到多seq
        handoff 条件：
        1. prompt 已经全部 cached
        2. 第一枚生成 token 已经 append 到 token_ids
        """
        
        seqs = []

        for i, text in enumerate(texts):
            seq = self.add_request(
                text=text,
                seq_idx=start_seq_id + i,
                temperature=temperature,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )
            seqs.append(seq)

        total = len(seqs)
        payloads: dict[int, HandoffPayload] = {}
        handed_off: set[int] = set()

        # 离线超时保护
        empty_steps = 0
        max_empty_steps = 10000

        with torch.inference_mode():
            while len(handed_off) < total:
                new_payloads = self.step()
                
                if not new_payloads:
                    empty_steps += 1
                    if empty_steps > max_empty_steps:
                        raise RuntimeError("PrefillEngine.run_prefill made no progress")
                    continue

                empty_steps = 0
                for payload in new_payloads:
                    if payload.seq_idx in handed_off:
                        continue

                    payloads[payload.seq_idx] = payload
                    handed_off.add(payload.seq_idx)
                        
        # 按原顺序返回
        return [payloads[seq.seq_idx] for seq in seqs]
                    
    def abort_by_request_id(self, request_id: str) -> bool:
        """
        prefill 失败，要释放 prefill blocks。
        KV save 后但 decode 没 restore，要 discard transfer_meta。
        decode restore 失败，要释放 decode local blocks，也要 discard transfer_meta。
        decode 完成后，scheduler.postprocess 已经 deallocate，但 request state 也要 finish。
        abort 要同时查 prefill scheduler 和 decode scheduler。
        """
        for seq in list(self.scheduler.waiting):
            if getattr(seq, "request_id", None) == request_id:
                self.scheduler.abort(seq)
                return True

        for seq in list(self.scheduler.running):
            if getattr(seq, "request_id", None) == request_id:
                self.scheduler.abort(seq)
                return True

        return False