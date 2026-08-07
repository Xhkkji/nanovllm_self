import uuid
from time import perf_counter
from typing import Iterable

import torch

from nanovllm.engine.Sequence import SequenceStatus
from .coordinator import PDCoordinator
from .request_state import RequestState, RequestStatus, TokenEvent
from .runtime_types import OnlineSchedulerStepMetrics

class OnlinePDCoordinator(PDCoordinator):
    def __init__(self, config, kv_backend: str = "dict"):
        super().__init__(config, kv_backend=kv_backend)
        self.requests: dict[str, RequestState] = {}
        self.seq_to_request: dict[int, str] = {}
        self.next_seq_idx = 0
        # 【第四章收口改动】在线调度观测状态：只记录最近一轮，避免第一版引入复杂 metrics backend。
        self.online_scheduler_step_id = 0
        self.last_online_scheduler_metrics: OnlineSchedulerStepMetrics | None = None

    def submit(
        self,
        text: str,                    # 用户输入的提示词
        max_tokens: int = 64,         # 最大生成token数
        temperature: float = 0.0,     # 采样温度（0=贪婪）
        ignore_eos: bool = True,      # 是否忽略结束符
        request_id: str | None = None,# 可选的请求ID
    ) -> str:    
        request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        # 【第四章收口改动】在线请求 ID 必须唯一，否则 request table 会被覆盖。
        if request_id in self.requests:
            raise ValueError(f"duplicate request_id: {request_id}")

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
        # 【第四章收口改动】先注册 request state，但 admission 失败时必须回滚。
        self.requests[request_id] = state
        self.seq_to_request[seq_idx] = request_id

        try:
            # 【本次在线化改动】submit 时先构造 seq，再做 admission check，再加入 prefill scheduler
            self.prefill_engine.add_request(
                text=text,
                seq_idx=seq_idx,
                temperature=temperature,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
                request_id=request_id,
                check_admission=True,
            )
        except Exception:
            # 【第四章收口改动】admission / add_request 失败时清理服务层残留元数据。
            self.requests.pop(request_id, None)
            self.seq_to_request.pop(seq_idx, None)
            raise

        state.status = RequestStatus.WAITING_PREFILL
        return request_id
    
    def step(self) -> list[TokenEvent]:
        events: list[TokenEvent] = []
        try:
            with torch.inference_mode():
                # 【第四章收口改动】进入 prefill step 前同步 waiting/running 状态，便于外部观测。
                self._sync_prefill_states()
                # 执行一次prefill,获取所有已经完成prefill并已经生成第一个token的seq信息
                payloads = self.prefill_engine.step()
                # 【第四章收口改动】prefill step 后再同步一次，覆盖 chunked prefill 仍未 handoff 的请求。
                self._sync_prefill_states()

                for payload in payloads:
                    request_id = payload.request_id
                    state = self.requests[request_id]
                    state.handoff_payload = payload
                    # 【本次在线化改动】prefill handoff 时，payload.token_ids 里通常已经包含第一枚生成 token
                    # payload.token_ids 已经包含:prompt tokens + first generated token
                    state.output_ids = payload.token_ids[len(state.input_ids):]

                    if state.output_ids:
                        # 【本次在线化改动】这里取 first generated token。
                        # handoff 边界通常只会带 1 个 completion token，但这里显式取第 0 个，
                        # 语义更稳定，也方便以后扩展成多 token handoff。
                        first_token_id = state.output_ids[0]
                        first_text = self.tokenizer.decode([first_token_id], skip_special_tokens=True)
                        # 【本次在线化改动】把 prefill handoff 边界的首 token 变成可 stream 的事件
                        event = TokenEvent(
                            request_id=request_id,
                            token_id=first_token_id,
                            text=first_text,
                            finished=False,
                        )
                        state.stream_queue.put(event)
                        events.append(event)

                    if payload.finished:
                        # 【第四章收口改动】正常完成统一走 finish_request，资源清理和状态收束集中处理。
                        self.finish_request(request_id, finish_reason="length_or_eos")
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
        except Exception as e:
            self._mark_active_requests_error(e)
            raise

        # 【第四章收口改动】每轮 step 结束后记录调度和资源快照，供在线化调试/压测读取。
        self.record_online_scheduler_metrics(emitted_events=len(events))
        return events
    
    def _collect_decode_events(self, decode_out):
        """
        decode结束时，decode会生成这批seq的最后一个token(投机解码机制)并退出decode
        此时最后一个token还没有处理，需要在这一步处理
        """
        events = []
        # 【本次在线化改动】DecodeStepOutput 是 decode 侧的结构化返回，online coordinator 负责转成 request 事件
        scheduled = decode_out.scheduled
        token_ids = [int(x) for x in decode_out.token_ids]
        seq_need_compute_logits = decode_out.seq_need_compute_logits
        finished_seq_ids = set(decode_out.finished_seq_ids)

        for local_idx, token_id in zip(seq_need_compute_logits, token_ids):
            seq = scheduled[local_idx]
            request_id = getattr(seq, "request_id", None) or self.seq_to_request[seq.seq_idx]
            state = self.requests[request_id]

            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            # 【本次在线化改动】把 decode 侧产出的 token 追加到 request 状态里
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
                # 【第四章收口改动】decode 侧完成后也统一走 finish_request。
                self.finish_request(request_id, finish_reason="length_or_eos")

        return events
    
    def is_finished(self, request_id: str) -> bool:
        return self.requests[request_id].status in (
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
            RequestStatus.ERROR,
        )

    def stream(self, request_id: str):
        """
        同步 generator 版本,把 TokenEvent 以 generator 形式暴露给外部
        第一版不引入 async / FastAPI，只提供在线服务的最小消费接口。
        """
        while not self.is_finished(request_id):
            # 推进所有 active requests。
            self.step()
            for event in self.poll_events(request_id):
                yield event

        for event in self.poll_events(request_id):
            yield event
    
    def poll_events(self, request_id: str) -> list[TokenEvent]:
        state = self.requests[request_id]
        events = []

        while not state.stream_queue.empty():
            # 阻塞流式获取TokenEvent
            events.append(state.stream_queue.get())
        return events

    def count_requests_by_status(self) -> dict[RequestStatus, int]:
        """
        【第四章收口改动】
        明确命名为 request status 统计，只统计服务层 RequestState，
        不和 scheduler.waiting/running 的 Sequence 队列混用。
        """
        counts = {status: 0 for status in RequestStatus}
        for state in self.requests.values():
            counts[state.status] += 1
        return counts

    def record_online_scheduler_metrics(self, emitted_events: int) -> None:
        """
        【第四章收口改动】
        记录最近一轮 OnlinePDCoordinator.step() 的调度快照。
        这里全部读取现有状态，不参与调度决策，避免观测逻辑影响执行路径。
        """
        prefill_schedule = self.prefill_engine.scheduler.last_schedule_info
        decode_schedule = self.decode_engine.scheduler.last_schedule_info
        status_counts = self.count_requests_by_status()
        prefill_blocks = self.prefill_engine.model_runner.block_manager.get_block_usage_snapshot()
        decode_blocks = self.decode_engine.model_runner.block_manager.get_block_usage_snapshot()

        self.last_online_scheduler_metrics = OnlineSchedulerStepMetrics(
            step_id=self.online_scheduler_step_id,
            emitted_events=emitted_events,
            prefill_scheduled_decode_tokens=prefill_schedule.num_decode_tokens,
            prefill_scheduled_prefill_tokens=prefill_schedule.num_prefill_tokens,
            prefill_schedule_reason=prefill_schedule.reason,
            decode_scheduled_decode_tokens=decode_schedule.num_decode_tokens,
            decode_scheduled_prefill_tokens=decode_schedule.num_prefill_tokens,
            decode_schedule_reason=decode_schedule.reason,
            waiting_prefill_requests=len(self.prefill_engine.scheduler.waiting),
            prefill_running_sequences=len(self.prefill_engine.scheduler.running),
            decode_running_sequences=len(self.decode_engine.scheduler.running),
            finished_requests=status_counts[RequestStatus.FINISHED],
            aborted_requests=status_counts[RequestStatus.ABORTED],
            error_requests=status_counts[RequestStatus.ERROR],
            prefill_used_blocks=prefill_blocks["used_blocks"],
            prefill_free_blocks=prefill_blocks["free_blocks"],
            decode_used_blocks=decode_blocks["used_blocks"],
            decode_free_blocks=decode_blocks["free_blocks"],
        )
        self.online_scheduler_step_id += 1

    def get_last_online_scheduler_metrics(self) -> OnlineSchedulerStepMetrics | None:
        """
        【第四章收口改动】读取最近一轮在线调度指标；尚未 step 过时返回 None。
        """
        return self.last_online_scheduler_metrics

    def _sync_prefill_states(self) -> None:
        """
        【第四章收口改动】
        最小状态机同步：
        - waiting 队列里的请求标记为 WAITING_PREFILL
        - running 且尚未 handoff 的 prefill 请求标记为 PREFILLING
        Decode 状态由 restore_payloads 后统一设置为 DECODING。
        """
        terminal = {
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
            RequestStatus.ERROR,
        }

        for seq in list(self.prefill_engine.scheduler.waiting):
            request_id = getattr(seq, "request_id", None)
            if request_id in self.requests and self.requests[request_id].status not in terminal:
                self.requests[request_id].status = RequestStatus.WAITING_PREFILL

        for seq in list(self.prefill_engine.scheduler.running):
            request_id = getattr(seq, "request_id", None)
            if request_id in self.requests and self.requests[request_id].status not in terminal:
                self.requests[request_id].status = RequestStatus.PREFILLING

    def _cleanup_request_resources(self, state: RequestState) -> None:
        """
        【第四章收口改动】
        request 级统一资源清理入口。
        第一版只清三类资源：
        - prefill scheduler 中可能残留的 seq
        - decode scheduler 中可能残留的 seq
        - 尚未消费或异常残留的 handoff payload / transfer_meta
        """
        self.prefill_engine.abort_by_request_id(state.request_id)
        self.decode_engine.abort_by_request_id(state.request_id)

        if state.handoff_payload is not None:
            self._cleanup_payload(state.handoff_payload)

    def _cleanup_payload(self, payload):
        if payload is None:
            return
        if payload.transfer_meta is not None:
            self.prefill_connector.discard(payload.transfer_meta)
            self.decode_connector.discard(payload.transfer_meta)

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

        # 【第四章收口改动】abort 统一走 request 级资源清理。
        self._cleanup_request_resources(state)

        state.status = RequestStatus.ABORTED
        state.finished_time = perf_counter()
        state.stream_queue.put(TokenEvent(
            request_id=request_id,
            token_id=None,
            finished=True,
            finish_reason=reason,
        ))
        return True

    def finish_request(self, request_id: str, finish_reason: str = "stop") -> None:
        state = self.requests.get(request_id)
        if state is None:
            return

        if state.status in (
            RequestStatus.FINISHED,
            RequestStatus.ABORTED,
            RequestStatus.ERROR,
        ):
            return

        state.mark_finished(finish_reason=finish_reason)

        # 【第四章收口改动】正常完成也走统一清理，确保 scheduler / payload 没有残留。
        self._cleanup_request_resources(state)

    def _mark_active_requests_error(self, error: Exception | str) -> None:
        for state in self.requests.values():
            if state.status in (
                RequestStatus.FINISHED,
                RequestStatus.ABORTED,
                RequestStatus.ERROR,
            ):
                continue

            # 【第四章收口改动】异常路径只发 error event，不再先发 abort event，避免重复终止事件。
            self._cleanup_request_resources(state)
            state.mark_error(error)

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
