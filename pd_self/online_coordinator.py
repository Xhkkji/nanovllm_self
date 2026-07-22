import uuid
from time import perf_counter
from typing import Iterable

import torch

from nanovllm.engine.Sequence import SequenceStatus
from .coordinator import PDCoordinator
from .request_state import RequestState, RequestStatus, TokenEvent

class OnlinePDCoordinator(PDCoordinator):
    def __init__(self, config, kv_backend: str = "dict"):
        super().__init__(config, kv_backend=kv_backend)
        self.requests: dict[str, RequestState] = {}
        self.seq_to_request: dict[int, str] = {}
        self.next_seq_idx = 0

    def submit(
        self,
        text: str,                    # 用户输入的提示词
        max_tokens: int = 64,         # 最大生成token数
        temperature: float = 0.0,     # 采样温度（0=贪婪）
        ignore_eos: bool = True,      # 是否忽略结束符
        request_id: str | None = None,# 可选的请求ID
    ) -> str:    
        request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        seq_idx = self.next_seq_idx
        self.next_seq_idx += 1

        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        state = RequestState(
            request_id=request_id,
            seq_idx=seq_idx,
            prompt=text,
            input_ids=input_ids,
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        self.requests[request_id] = state
        self.seq_to_request[seq_idx] = request_id

        # 将文本处理成seq，并加入prefill的scheduler中
        seq = self.prefill_engine.add_request(
            text=text,
            seq_idx=seq_idx,
            temperature=temperature,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        seq.request_id = request_id
        state.status = RequestStatus.WAITING_PREFILL
        return request_id
    
    def step(self) -> list[TokenEvent]:
        events: list[TokenEvent] = []

        with torch.inference_mode():
            # 执行一次prefill,获取所有已经完成prefill并已经生成第一个token的seq信息
            payloads = self.prefill_engine.step()

            for payload in payloads:
                request_id = payload.request_id
                state = self.requests[request_id]
                state.handoff_payload = payload
                # 注意：prefill handoff 时，payload.token_ids 里通常已经包含第一枚生成 token
                # payload.token_ids 已经包含:prompt tokens + first generated token
                state.output_ids = payload.token_ids[len(state.input_ids):]

                if state.output_ids:
                    # 所以这里要把first generated token写进state.output_ids
                    first_token_id = state.output_ids[-1]
                    first_text = self.tokenizer.decode([first_token_id], skip_special_tokens=True)
                    # 新生成token的事件信息
                    event = TokenEvent(
                        request_id=request_id,
                        token_id=first_token_id,
                        text=first_text,
                        finished=False,
                    )
                    state.stream_queue.put(event)
                    events.append(event)

                if payload.finished:
                    state.mark_finished(finish_reason="length_or_eos")
                    events.append(state.stream_queue.get())
                    continue

                state.status = RequestStatus.WAITING_DECODE

            if payloads:
                _, restored = self.decode_engine.restore_payloads(payloads)
                for seq in restored:
                    request_id = getattr(seq, "request_id", None)
                    if request_id is None:
                        request_id = self.seq_to_request[seq.seq_idx]
                        seq.request_id = request_id
                    self.requests[request_id].status = RequestStatus.DECODING

            decode_out = self.decode_engine.step()

            if decode_out.scheduled:
                events.extend(self._collect_decode_events(decode_out))

        return events
    
    def _collect_decode_events(self, decode_out):
        """
        decode结束时，decode会生成这批seq的最后一个token(投机解码机制)并退出decode
        此时最后一个token还没有处理，需要在这一步处理
        """
        scheduled, token_ids, seq_need_compute_logits, finished_seq_ids = decode_out
        events = []
        token_ids = [int(x) for x in token_ids]

        for local_idx, token_id in zip(seq_need_compute_logits, token_ids):
            seq = scheduled[local_idx]
            request_id = getattr(seq, "request_id", None) or self.seq_to_request[seq.seq_idx]
            state = self.requests[request_id]

            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            # 把最后一个token加入到对应的seq中
            state.output_ids.append(token_id)

            event = TokenEvent(
                request_id=request_id,
                token_id=token_id,
                text=text,
                finished=False,
            )
            state.stream_queue.put(event)
            events.append(event)

            if seq.seq_idx in finished_seq_ids:
                state.mark_finished(finish_reason="length_or_eos")
                events.append(state.stream_queue.get())

        return events
    
    def is_finished(self, request_id: str) -> bool:
        return self.requests[request_id].status in (
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
            RequestStatus.ERROR,
        )

    def generate(self, texts, max_tokens=64, temperature=0.0, ignore_eos=True):
        """
        同步批处理
        用于离线批处理和单元测试，在线模式中用不上
        """
        
        request_ids = [
            self.submit(
                text=text,
                max_tokens=max_tokens,
                temperature=temperature,
                ignore_eos=ignore_eos,
            )
            for text in texts
        ]

        while not all(self.is_finished(rid) for rid in request_ids):
            self.step()

        outputs = []
        for rid in request_ids:
            state = self.requests[rid]
            outputs.append(state.input_ids + state.output_ids)
        return outputs
    
    def abort_request(self, request_id: str, reason: str = "abort") -> bool:
        state = self.requests.get(request_id)
        if state is None:
            return False

        if state.status in (
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
            RequestStatus.ERROR,
        ):
            return True

        self.prefill_engine.scheduler.abort_by_seq_idx(state.seq_idx)
        self.decode_engine.scheduler.abort_by_seq_idx(state.seq_idx)

        if state.handoff_payload is not None:
            self._cleanup_payload(state.handoff_payload)

        state.status = RequestStatus.ABORTED
        state.finished_time = perf_counter()
        state.stream_queue.put(TokenEvent(
            request_id=request_id,
            token_id=None,
            finished=True,
            finish_reason=reason,
        ))
        return True


    def _cleanup_payload(self, payload):
        if payload is None:
            return
        if payload.transfer_meta is not None:
            self.prefill_connector.discard(payload.transfer_meta)
            self.decode_connector.discard(payload.transfer_meta)