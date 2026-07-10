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
        """
        # seq_list = deque()
        scheduled_running = []
        scheduled_new = []
        preempted = []  # 被抢占的seq

        num_seqs = 0
        token_budget = self.max_num_batched_tokens

        # 先推进running队列，此时prefill和decode在schedule中不分
        # req_index 是一个游标（Cursor）或索引，用于在 while 循环中记录当前正在处理 self.running 列表中的哪一条序列
        req_index = 0
        while req_index < len(self.running) and token_budget > 0 and num_seqs < self.max_num_seqs:
            seq = self.running[req_index]
            # 当前这条 seq 还需要推进多少 token
            remaining = len(seq) - seq.num_cached_tokens
            
            # 查看当前seq的prefill是否进行完
            if not seq.is_prefill_done:
                num_new_tokens = remaining
                # 若开启chunked_prefill功能
                if self.config.enable_chunked_prefill:
                    num_new_tokens = min(num_new_tokens,self.config.prefill_chunk_size)
                num_new_tokens = min(num_new_tokens, token_budget)
            else:
                # 如果该seq已经prefill完成，走decode，一次只推进 1 个 token
                num_new_tokens = min(1, token_budget)
            
            if num_new_tokens <= 0:
                # 处理完成，游标后移，继续处理下一个seq
                req_index += 1
                continue
            
            # 只有 decode 可能需要 append 新 block
            if seq.is_prefill_done:
                while not self.block_manager.can_append(seq):
                    # 从队列尾部开始抢占
                    # 抢占会一直持续，直到 can_append或者列表长度缩减到等于当前的 req_index
                    if self.running:
                        preempted_seq = self.running.pop()
                        self.preempty(preempted_seq)  # 放回wait队列
                        preempted.append(preempted_seq)  # 调度中断的seq都在waiting队列，此处与还没有开始服务的seq作区分
                        if len(self.running) == req_index:
                            # 若进入这个分支，则seq[req_index]也已经被驱逐
                            break
                    else:
                        break

                if len(self.running) == req_index:
                    # 一路退出大循环
                    # 当前的 schedule() 调用会立即终止调度过程
                    break
                
                self.block_manager.may_append(seq)
            
            seq.num_new_tokens = num_new_tokens
            seq.status = SequenceStatus.RUNNING
            scheduled_running.append(seq)

            token_budget -= num_new_tokens
            num_seqs += 1
            req_index += 1

        # 若能进入下面逻辑，则没有抢占发生，正常执行prefill
        if not preempted:
            # 如果本轮调度中已经发生过抢占（即 preempted 列表不为空），则暂停接纳 waiting 队列中的新请求
            # 为了在系统资源紧张时优先保障 已运行序列 的稳定性和公平性
            while self.waiting and token_budget > 0 and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                remaining = len(seq) - seq.num_cached_tokens
                num_new_tokens = remaining
                if self.config.enable_chunked_prefill:
                    num_new_tokens = min(num_new_tokens, self.config.prefill_chunk_size)
                num_new_tokens = min(num_new_tokens, token_budget)

                if num_new_tokens <= 0:
                    break
                
                if not seq.block_table:
                    if not self.block_manager.can_allocate(seq):
                        break
                    self.block_manager.allocate(seq)
                
                seq.num_new_tokens = num_new_tokens
                seq.status = SequenceStatus.RUNNING

                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new.append(seq)

                token_budget -= num_new_tokens
                num_seqs += 1
        
        # scheduled_running 是“续跑”的，scheduled_new 是“新跑”的。两者合并后的列表 
        # scheduled就是本轮模型推理需要处理的所有序列
        scheduled = scheduled_running + scheduled_new
        assert scheduled
        return scheduled
        
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
                self.running.remove(seq)
                finished_list.append(seq)
            
        # 所有本轮参与的活跃 seq，都推进 cache 进度
        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens += seq.num_new_tokens
                seq.num_new_tokens = 0

        return finished_list








