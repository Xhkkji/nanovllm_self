# 只负责在已有上下文基础上继续生成

from nanovllm.engine.Sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
import torch
from .runtime_types import DecodeStepOutput

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

        # 多返回一个值：已经完成decode的seq_idx
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
                scheduled, token_ids, seq_need_compute_logits, finished_seq_ids = self.step()
                if not scheduled:
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
