from transformers import AutoConfig
from nanovllm.engine.block_manager import block_manager as bm
from nanovllm.engine.Sequence import Sequence, SequenceStatus

from collections import deque

@dataclass
class ScheduleOutput:
    seqs: list[Sequence]
    decode_seqs: list[Sequence]
    prefill_seqs: list[Sequence]
    new_prefill_seqs: list[Sequence]
    num_decode_tokens: int
    num_prefill_tokens: int
    token_budget_used: int
    token_budget_remaining: int
    blocked_decode: int = 0
    blocked_prefill: int = 0
    reason: str = "ok"

class Scheduler:
    def __init__(self, config, block_manager):
        self.config = config
        self.block_manager = block_manager
        self.max_num_seqs = config.max_num_seqs  # 一轮调度里，最多同时处理多少条序列，“并发序列数上限”
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 一轮调度里，最多处理多少个 token
        self.eos = config.eos

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def add(self, seq):
        self.waiting.append(seq)  # 全是需要prefill的seq
    
    def is_finished(self):
        return (not self.waiting) and (not self.running)

    def schedule(self) -> (list[Sequence]):
        """
        未完成 prefill 的请求会一直留在 running
        waiting 只放还没接纳的请求
        postprocess() 不再区分 prefill/decode 两套函数
        所有活跃 seq 都统一推进 num_cached_tokens += num_new_tokens
        当waiting队列不为空，优先处理waiting队列（需要prefill的seq） 
        
        20260721,schedule 支持空返回 + decode-first
        在线推理前的优化
        在线化前推荐调度顺序：
        1. decode first
        2. running prefill chunk
        3. waiting new prefill

        允许返回空列表，方便在线 engine loop 常驻运行。
        """
        # seq_list = deque()
        scheduled_decode = []
        scheduled_prefill = []
        # scheduled_running = []
        scheduled_new = []
        # preempted = []  # 被抢占的seq

        num_seqs = 0
        token_budget = self.max_num_batched_tokens

        # 1. 优先 decode，保护 ITL / TPOT
        for seq in list(self.running):
            if token_budget <= 0 or num_seqs >= self.max_num_seqs:
                break
            
            if not seq.is_prefill_done:
                # 若还处于prefill
                continue

            if not self.block_manager.can_append(seq):
                # 查看block容量是否足够
                continue
            
            self.block_manager.may_append(seq)

            seq.num_new_tokens = 1
            seq.status = SequenceStatus.RUNNING
            scheduled_decode.append(seq)

            token_budget -= 1
            num_seqs += 1
        
        # 2. 推进已经进入 running 的 prefill chunk
        for seq in list(self.running):
            if token_budget <= 0 or num_seqs >= self.max_num_seqs:
                break
            
            if seq.is_prefill_done:
                continue

            remaining = len(seq) - seq.num_cached_tokens
            num_new_tokens = remaining

            if self.config.enable_chunked_prefill:
                num_new_tokens = min(num_new_tokens, self.config.prefill_chunk_size)

            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens <= 0:
                continue

            seq.num_new_tokens = num_new_tokens
            seq.status = SequenceStatus.RUNNING
            scheduled_prefill.append(seq)

            token_budget -= num_new_tokens
            num_seqs += 1

        # 3. 接纳 waiting 中的新 prefill 请求
        while self.waiting and token_budget > 0 and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    break
                self.block_manager.allocate(seq)

            remaining = len(seq) - seq.num_cached_tokens
            num_new_tokens = remaining

            if self.config.enable_chunked_prefill:
                num_new_tokens = min(num_new_tokens, self.config.prefill_chunk_size)

            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens <= 0:
                break

            seq.num_new_tokens = num_new_tokens
            seq.status = SequenceStatus.RUNNING

            self.waiting.popleft()
            self.running.append(seq)
            scheduled_new.append(seq)

            token_budget -= num_new_tokens
            num_seqs += 1

        return scheduled_decode + scheduled_prefill + scheduled_new
        
    def preempty(self, seq):
        """
        调度失败，重新放回wait队列
        """
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def postprocess(self, seqs, token_ids, seq_need_compute_logits):
        """
        token_ids为decode新生成的token，现在处理新token，判断是否已经生成结束
        标准版需要实现seq:List[seq], token_ids:List[int],即批量seq
        改进后，只给真正需要采样的 seq 追加 token
        seq_need_logits:[seqidx0, seqidx2, seqidx3]
        token_ids:[seq0_vocabId, seq2_vocabId, seq3_vocabId]
        """
        assert len(token_ids) == len(seq_need_compute_logits)
        finished_list = []
        for seq_idx, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_idx]
            # 把新生成的token加入到对应seq中
            seq.append_token(token_id)
            # 若生成结束
            if (not seq.ignore_eos and token_id == self.eos) or \
                (seq.num_completion_tokens == seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                finished_list.append(seq)
            
        # 所有本轮参与的活跃 seq，都推进 cache 进度
        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens += seq.num_new_tokens
                seq.num_new_tokens = 0

        return finished_list

    def abort(self, seq: Sequence) -> None:
        """
        在线请求取消 / 超时 / 异常时统一清理。
        """
        if seq in self.waiting:
            self.waiting.remove(seq)

        if seq in self.running:
            self.running.remove(seq)
        
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED
        seq.num_new_tokens = 0
    
    def abort_by_seq_idx(self, seq_idx: int) -> bool:
        for seq in list(self.waiting):
            if seq.seq_idx == seq_idx:
                self.abort(seq)
                return True

        for seq in list(self.running):
            if seq.seq_idx == seq_idx:
                self.abort(seq)
                return True

        return False







