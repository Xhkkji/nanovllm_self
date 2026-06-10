from transformers import AutoConfig
from nanovllm.engine.block_manager import block_manager as bm
from nanovllm.engine.Sequence import Sequence, SequenceStatus

from collections import deque

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
        self.waiting.appendleft(seq)  # 全是需要prefill的seq
    
    def is_finished(self):
        if not self.waiting and not self.running:
            return True
        else:
            return False

    def schedule(self) -> (list[Sequence], bool):
        # 当waiting队列不为空，优先处理waiting队列（需要prefill的seq）
        seq_list = deque()
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            seq_list.append(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.appendleft(seq)
            self.block_manager.allocate(seq)  # 分配块并进行前缀共享, 将分配的块表关联到序列
            num_seqs += 1
            num_batched_tokens += len(seq)
        if seq_list:
            return seq_list, True
        
        # 没有需要prefill的seq了才处理decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 资源不足时进行抢占机制，抢占最后加入的，队列头的等待最久，优先保障队列头
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempty(self.running.pop())
                else:
                    self.preempty(seq)
            else:
                # 执行while正常结束的逻辑
                num_seqs += 1
                seq_list.append(seq)
                self.block_manager.may_append(seq)
        assert seq_list
        # deque extendleft的时候会把队列倒过来接在右边，因此需要手动把seq_list反过来
        self.running.extendleft(reversed(seq_list))
        return seq_list, False
    
    def preempty(self, seq):
        """
        调度失败，重新放回wait队列
        """
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def postprocess(self, seq_list, token_ids):
        """
        token_ids为decode新生成的token，现在处理新token，判断是否已经生成结束
        标准版需要实现seq:List[seq], token_ids:List[int],即批量seq
        """
        finished_list = []
        for i, seq in enumerate(seq_list):
            seq.append_token(token_ids[i])
            # 如果调度需要结束
            if token_ids[i] == self.eos or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished_list.append(seq)
        return finished_list








