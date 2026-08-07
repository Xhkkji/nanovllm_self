from dataclasses import dataclass, field
from time import perf_counter

import torch
from nanovllm.engine.Sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from .runtime_types import DecodeStepOutput


@dataclass
class ActiveDecodeRequest:
    """
    continuous decode 中一条“已恢复但未完成”的请求状态。

    注意这里不保存 Sequence 本体：
    - 真正参与推理的 Sequence 放在 scheduler.running 里。
    - 这里保存的是服务层状态和 benchmark 指标。
    - 两边通过 seq_idx 建立一一对应关系。
    """
    request_id: str
    seq_idx: int
    payload_path: str | None = None
    num_prompt_tokens: int = 0
    num_kv_blocks: int = 0
    kv_nbytes: int = 0
    scale_nbytes: int = 0

    # payload_read_time_s 由 worker 填，因为读取 pickle 属于文件协议，不属于 engine。
    payload_read_time_s: float = 0.0
    # 异步 PD 中，transfer_time_s 表示从提交 irecv 到 recv 完成的时间。
    # 同步 / shared_memory 路径保持 0，方便同一套 metrics 对比。
    transfer_time_s: float = 0.0
    restore_time_s: float = 0.0
    # decode_time_s 是请求侧观察到的 decode wall time：
    # 如果一轮 batch step 同时调度多条 seq，每条 seq 都累计同一段 step_time。
    decode_time_s: float = 0.0
    # decode_compute_time_s 是按 batch size 分摊后的计算时间：
    # 用来观察 continuous batching 是否摊薄了单请求计算成本。
    decode_compute_time_s: float = 0.0

    decode_steps: int = 0
    decode_step_tokens: list[int] = field(default_factory=list)
    decode_finished_ids: list[int] = field(default_factory=list)
    final_token_lens: list[int] = field(default_factory=list)

    restored_count: int = 0
    finished_in_prefill_results: int = 0
    generated_tokens: int = 0

    total_t0: float = field(default_factory=perf_counter)


class DecodeEngine:
    """
    从 handoff payload 恢复出 Sequence
    挂到自己的 scheduler / running 队列
    继续 decode 到结束
    """
    def __init__(self, config, tokenizer, model_runner, kv_connector):
        self.config = config
        self.tokenizer = tokenizer
        self.model_runner = model_runner
        self.kv_connector = kv_connector
        self.scheduler = Scheduler(config, model_runner.block_manager)

        # continuous decode 用的轻量状态表。
        #
        # scheduler.running 里保存真正要推理的 Sequence；
        # active_decode_requests 里保存同一条 Sequence 对应的请求级状态和指标。
        # 当某个 seq finished 后，scheduler 会移除 Sequence，
        # continuous_step() 同步 pop 这里的 ActiveDecodeRequest。
        self.active_decode_requests: dict[int, ActiveDecodeRequest] = {}
    

    # ########################### continuous decode ###########################
    def restore_active_payload(self, payload, payload_path: str | None = None):
        """
        continuous decode 的单请求 restore 入口。

        这里一次只接收一个 payload，原因是当前 multiprocess 文件协议就是：
            一个 *.payload.pkl 对应一条请求。

        这个函数只做三件事：
        1. 调用 restore_payloads([payload])，把 prefill 侧 handoff 过来的 KV
           恢复到 decode 侧本地 KV cache。
        2. 如果请求还没完成，把恢复出的 Sequence 加入 scheduler.running。
        3. 创建 ActiveDecodeRequest，记录这条请求的状态和耗时。

        真正的“批量 decode”不在这里发生，而是在 continuous_step() 中发生。
        多次调用 restore_active_payload() 会让多条 seq 同时留在 scheduler.running；
        continuous_step() 每轮从 scheduler.running 里批量选 seq 做一次 batched forward。

        返回：
        - finished_state: prefill 阶段已完成时返回状态，否则 None
        - active_state: restore 进 scheduler 的状态，否则 None
        """
        total_t0 = perf_counter()
        meta = payload.transfer_meta
        num_kv_blocks = meta.num_kv_blocks if meta is not None else 0
        kv_nbytes = (
            meta.storage_ref.nbytes
            if meta is not None and meta.storage_ref is not None
            else 0
        )
        scale_nbytes = (
            meta.scale_storage_ref.nbytes
            if meta is not None and meta.scale_storage_ref is not None
            else getattr(meta.storage_ref, "scale_nbytes", 0)
            if meta is not None and meta.storage_ref is not None
            else 0
        )

        restore_t0 = perf_counter()
        results, restored = self.restore_payloads([payload])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        restore_time_s = perf_counter() - restore_t0

        if not restored:
            # max_tokens 很小时可能 prefill 阶段已经完成。
            # 这种请求没有进入 scheduler.running，也不加入 active_decode_requests。
            final_token_lens = [len(tokens) for tokens in results.values()]
            generated_tokens = sum(
                max(0, token_len - payload.num_prompt_tokens)
                for token_len in final_token_lens
            )

            state = ActiveDecodeRequest(
                request_id=payload.request_id,
                seq_idx=payload.seq_idx,
                payload_path=payload_path,
                num_prompt_tokens=payload.num_prompt_tokens,
                num_kv_blocks=num_kv_blocks,
                kv_nbytes=kv_nbytes,
                scale_nbytes=scale_nbytes,
                restore_time_s=restore_time_s,
                restored_count=0,
                finished_in_prefill_results=len(results),
                final_token_lens=final_token_lens,
                generated_tokens=generated_tokens,
                total_t0=total_t0,
            )
            return state, None

        # 当前协议下一次只 restore 一个 payload，所以 restored 里最多一条 seq。
        # 这条 seq 已经由 restore_payloads() 放入 scheduler.running；
        # 后续 continuous_step() 会和其他 active seq 一起批量推进。
        seq = restored[0]
        state = ActiveDecodeRequest(
            request_id=payload.request_id,
            seq_idx=seq.seq_idx,
            payload_path=payload_path,
            num_prompt_tokens=seq.num_prompt_tokens,
            num_kv_blocks=num_kv_blocks,
            kv_nbytes=kv_nbytes,
            scale_nbytes=scale_nbytes,
            restore_time_s=restore_time_s,
            restored_count=1,
            finished_in_prefill_results=len(results),
            final_token_lens=[len(seq.token_ids)],
            total_t0=total_t0,
        )
        self.active_decode_requests[seq.seq_idx] = state
        return None, state
    
    def continuous_step(self):
        """
        推进 continuous decode 一轮。

        链路：
        1. scheduler.schedule() 从 scheduler.running 里选出本轮可运行 seq。
        2. model_runner.run(scheduled) 对这些 seq 做一次 batched forward。
        3. scheduler.postprocess() 给需要 logits 的 seq 追加新 token，并释放 finished seq。
        4. 这里根据 DecodeStepOutput 更新 active_decode_requests 中的请求级指标。

        所谓批量 decode 指的是：
            一轮 step 同时推进多条请求，每条请求通常生成 1 个 token。
        不是一条请求一次生成多个 token。

        返回：
        - finished_states: 本轮完成的请求状态列表
        - step_out: 原始 DecodeStepOutput，方便调试
        """
        if not self.active_decode_requests:
            return [], DecodeStepOutput([], [], [], [])

        step_t0 = perf_counter()
        out = self.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time_s = perf_counter() - step_t0

        if not out.scheduled:
            return [], out

        scheduled_count = len(out.scheduled)
        per_seq_compute_time_s = step_time_s / scheduled_count

        # 所有 scheduled seq 都参与了本轮 forward，因此请求侧 latency 都累计 step_time。
        # 同时记录一个按 batch size 分摊的 compute time，便于评估 batching 收益。
        for seq in out.scheduled:
            state = self.active_decode_requests.get(seq.seq_idx)
            if state is None:
                continue
            state.decode_steps += 1
            state.decode_time_s += step_time_s
            state.decode_compute_time_s += per_seq_compute_time_s
            state.final_token_lens.append(len(seq.token_ids))

        # token_ids 只对应需要采样 logits 的 seq。
        # seq_need_compute_logits 是 scheduled 内部的局部下标，不是全局 seq_idx。
        for local_idx, token_id in zip(out.seq_need_compute_logits, out.token_ids):
            seq = out.scheduled[local_idx]
            state = self.active_decode_requests.get(seq.seq_idx)
            if state is not None:
                state.decode_step_tokens.append(int(token_id))

        finished_states = []
        for seq_idx in out.finished_seq_ids:
            state = self.active_decode_requests.pop(seq_idx, None)
            if state is None:
                continue
            state.decode_finished_ids.append(seq_idx)
            final_token_len = state.final_token_lens[-1] if state.final_token_lens else 0
            state.generated_tokens = max(0, final_token_len - state.num_prompt_tokens)
            finished_states.append(state)

        return finished_states, out
    # ########################### continuous decode ###########################


    def _allocate_local_blocks(self, payload):
        """
        decode 侧用自己的 block manager 分配 block。
        注意：不能复用 payload.block_table。
        初始化分配，使用bm.allocate_block()
        """
        transfer_meta = payload.transfer_meta
        if transfer_meta is None or transfer_meta.num_kv_blocks == 0:
            return []
        bm = self.model_runner.block_manager
        local_block_ids = bm.allocate_block(transfer_meta.num_kv_blocks)
        return list(local_block_ids)
    
    def _restore_block_manager_meta(self, seq):
        """
        恢复 decode 侧 block_manager 的元信息。
        这样后续 append/hash 流程才不会乱。
        """
        bm = self.model_runner.block_manager
        prev_hash = -1
        num_cached = seq.num_cached_tokens

        for block_idx, block_id in enumerate(seq.block_table):
            start = block_idx * seq.block_size
            # （满块， 非满块）
            end = min(start + seq.block_size, num_cached)

            if start >= num_cached:
                break

            block_tokens = seq.token_ids[start:end]
            # block_id是逻辑->物理映射的逻辑所以，直接seq里面的索引
            bm.blocks[block_id].token_ids = list(block_tokens)
            bm.blocks[block_id].prev_hash = prev_hash
            bm.blocks[block_id].ref_count = 1

            # 满块才注册hash
            if len(block_tokens) == seq.block_size:
                h = bm.compute_hash(block_tokens, prev_hash)
                bm.blocks[block_id].update(h, block_tokens, prev_hash)
                bm.hash_to_block_id[h] = block_id
                prev_hash = h
            else:
                bm.blocks[block_id].hash = -1


    # ########################### 异步 PD restore 入口 ###########################
    def restore_sequence_from_kv_item(self, payload, kv_item) -> Sequence:
        """
        异步 recv 完成后使用。

        和 restore_sequence() 的区别：
        - restore_sequence() 内部会调用 kv_connector.load_kv()
        也就是会进入 backend.pop_by_ref() / dist.recv。
        - restore_sequence_from_kv_item() 已经拿到了 kv_item，
        所以只负责分配本地 block，并把 kv_item 写进 decode KV cache。
        """
        seq = Sequence(
            seq_idx=payload.seq_idx,
            token_ids=list(payload.token_ids),
            block_size=self.config.block_size,
        )
        seq.request_id = payload.request_id
        seq.num_prompt_tokens = payload.num_prompt_tokens
        seq.num_cached_tokens = payload.num_cached_tokens
        seq.temperature = payload.temperature
        seq.max_tokens = payload.max_tokens
        seq.ignore_eos = payload.ignore_eos
        seq.status = SequenceStatus.RUNNING

        local_block_ids = []
        try:
            local_block_ids = self._allocate_local_blocks(payload)
            seq.block_table = local_block_ids

            self.kv_connector.load_kv_item(
                item=kv_item,
                dst_block_ids=local_block_ids,
            )

            self._restore_block_manager_meta(seq)
            return seq

        except Exception:
            if local_block_ids:
                self.model_runner.block_manager.deallocate(seq)
            if payload.transfer_meta is not None:
                self.kv_connector.discard(payload.transfer_meta)
            raise
    
    def restore_active_payload_from_kv_item(
        self,
        payload,
        kv_item,
        payload_path: str | None = None,
        payload_read_time_s: float = 0.0,
        transfer_time_s: float = 0.0,
    ):
        """
        异步 recv 完成后，把 payload + kv_item 恢复成 active decode request。
        """
        total_t0 = perf_counter()
        meta = payload.transfer_meta

        num_kv_blocks = meta.num_kv_blocks if meta is not None else 0
        kv_nbytes = (
            meta.storage_ref.nbytes
            if meta is not None and meta.storage_ref is not None
            else 0
        )
        scale_nbytes = (
            meta.scale_storage_ref.nbytes
            if meta is not None and meta.scale_storage_ref is not None
            else getattr(meta.storage_ref, "scale_nbytes", 0)
            if meta is not None and meta.storage_ref is not None
            else 0
        )

        restore_t0 = perf_counter()

        if payload.finished:
            results = {payload.seq_idx: list(payload.token_ids)}
            restored = []
        else:
            seq = self.restore_sequence_from_kv_item(payload, kv_item)
            self.scheduler.running.append(seq)
            results = {}
            restored = [seq]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        restore_time_s = perf_counter() - restore_t0

        if not restored:
            final_token_lens = [len(tokens) for tokens in results.values()]
            generated_tokens = sum(
                max(0, token_len - payload.num_prompt_tokens)
                for token_len in final_token_lens
            )

            state = ActiveDecodeRequest(
                request_id=payload.request_id,
                seq_idx=payload.seq_idx,
                payload_path=payload_path,
                num_prompt_tokens=payload.num_prompt_tokens,
                num_kv_blocks=num_kv_blocks,
                kv_nbytes=kv_nbytes,
                scale_nbytes=scale_nbytes,
                payload_read_time_s=payload_read_time_s,
                transfer_time_s=transfer_time_s,
                restore_time_s=restore_time_s,
                restored_count=0,
                finished_in_prefill_results=len(results),
                final_token_lens=final_token_lens,
                generated_tokens=generated_tokens,
                total_t0=total_t0,
            )
            return state, None

        seq = restored[0]
        state = ActiveDecodeRequest(
            request_id=payload.request_id,
            seq_idx=seq.seq_idx,
            payload_path=payload_path,
            num_prompt_tokens=seq.num_prompt_tokens,
            num_kv_blocks=num_kv_blocks,
            kv_nbytes=kv_nbytes,
            scale_nbytes=scale_nbytes,
            payload_read_time_s=payload_read_time_s,
            transfer_time_s=transfer_time_s,
            restore_time_s=restore_time_s,
            restored_count=1,
            finished_in_prefill_results=len(results),
            final_token_lens=[len(seq.token_ids)],
            total_t0=total_t0,
        )

        self.active_decode_requests[seq.seq_idx] = state
        return None, state
    # ########################### 异步 PD restore 入口 ###########################
    
    
    def restore_sequence(self, payload) -> Sequence:
        # 把从prefill传过来的payload参数取到seq里面
        seq = Sequence(
            seq_idx=payload.seq_idx,
            token_ids=list(payload.token_ids),
            block_size=self.config.block_size,
        )
        seq.request_id = payload.request_id
        seq.num_prompt_tokens = payload.num_prompt_tokens
        seq.num_cached_tokens = payload.num_cached_tokens
        seq.temperature = payload.temperature
        seq.max_tokens = payload.max_tokens
        seq.ignore_eos = payload.ignore_eos
        seq.status = SequenceStatus.RUNNING

        local_block_ids = []
        try:
            # decode侧重新分配自己的block
            local_block_ids = self._allocate_local_blocks(payload)
            seq.block_table = local_block_ids
            # 把 prefill 侧导出的 KV block 注入 decode 侧 KV cache
            self.kv_connector.load_kv(
                transfer_meta=payload.transfer_meta,
                dst_block_ids=local_block_ids,
            )
            # 恢复bm元信息,即将获取到的seq的block信息存入decode端的bm中
            self._restore_block_manager_meta(seq)
            return seq
        except Exception:
            if local_block_ids:
                self.model_runner.block_manager.deallocate(seq)
            if payload.transfer_meta is not None:
                self.kv_connector.discard(payload.transfer_meta)
            raise
            

    def restore_payloads(self, payloads):
        """
        20260721 更新
        把 handoff payload 恢复成 decode 侧本地 Sequence。

        返回：
        - results: 已经 finished 的请求结果
        - restored: 本次恢复进入 decode scheduler 的 seq
        """
        results = {}
        restored = []

        for payload in payloads:
            if payload.finished:
                # 储存已经完成的
                results[payload.seq_idx] = list(payload.token_ids)
                continue
        
            seq = self.restore_sequence(payload)
            self.scheduler.running.append(seq)
            restored.append(seq)
        
        return results, restored


    def step(self):
        """
        推进 decode worker 一轮。

        返回：
        - scheduled: 本轮实际跑的 seq
        - token_ids: 本轮采样出的 token
        - seq_need_compute_logits: token_ids 对应 scheduled 里的局部下标
        """
        if self.scheduler.is_finished():
            return DecodeStepOutput([], [], [], [])

        scheduled = self.scheduler.schedule()
        if not scheduled:
            return DecodeStepOutput([], [], [], [])
        
        token_ids, seq_need_compute_logits = self.model_runner.run(scheduled)

        finished = self.scheduler.postprocess(
            scheduled,
            token_ids,
            seq_need_compute_logits,
        )

        # 【本次在线化改动】返回结构化结果，便于 online coordinator 组装 token event
        return DecodeStepOutput(
            scheduled=scheduled, 
            token_ids=[int(x) for x in token_ids],
            seq_need_compute_logits=seq_need_compute_logits,
            finished_seq_ids=[seq.seq_idx for seq in finished],
        )

    def run_decode(self, payloads):
        results, restored = self.restore_payloads(payloads)

        empty_steps = 0
        max_empty_steps = 10000

        with torch.inference_mode():
            while not self.scheduler.is_finished():
                # 【本次在线化改动】DecodeStepOutput 替代旧 tuple，run_decode 这里只关心是否有调度结果
                decode_out = self.step()
                if not decode_out.scheduled:
                    empty_steps += 1
                    if empty_steps > max_empty_steps:
                        raise RuntimeError("DecodeEngine.run_decode made no progress")
                    continue
                empty_steps = 0

        for seq in restored:
            results[seq.seq_idx] = list(seq.token_ids)

        # 按照payloads顺序排序
        ordered_ids = [payload.seq_idx for payload in payloads]
        return [results[seq_idx] for seq_idx in ordered_ids]
    
    def cleanup_sequence(self, seq):
        self.scheduler.abort(seq)

    def abort_by_request_id(self, request_id: str) -> bool:
        for seq in list(self.scheduler.waiting):
            if getattr(seq, "request_id", None) == request_id:
                self.scheduler.abort(seq)
                return True

        for seq in list(self.scheduler.running):
            if getattr(seq, "request_id", None) == request_id:
                self.scheduler.abort(seq)
                return True

        return False
