import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PagedAttention(nn.Module):
    """
    这个类是个工具类，应该在Transformer block中调用，非主逻辑函数
    """
    def __init__(self, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads if num_kv_heads > 0 else 1
        self.dtype = dtype
        # 以下在modelrunner中逐层绑定
        self.k_cache = None
        self.v_cache = None
        self.block_size = None
        self.layer_id = None

    def store_kv_cache(self, k, v, slot_mapping):
        """
        k: [num_tokens, num_kv_heads, head_dim]
        v: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens]
        """
        num_tokens = k.size(0)
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            # 把线性槽位映射到kv的block维度
            block_id = slot // self.block_size
            offset = slot % self.block_size
            self.k_cache[block_id, offset] = k[i]
            self.v_cache[block_id, offset] = v[i]

    def get_kv_cache(self, context):
        """
        根据 block_tables + context_lens，从全局 cache 中取出当前 batch 真正可见的历史 KV。

        这里只给一个最简单的“单 batch 拼接思路”版本。
        如果后面你要高性能 batch 化，可以再改造成更矢量化的实现。

        prefill过程，context.context_len字段没有赋值，需要单独处理
        """
        # block_tables: [batch, max_num_blocks]
        # context_lens: [batch]
        block_table = context.block_tables
        # context_lens = context.context_lens  # 若直接赋值，prefill路径是None
        all_k = []
        all_v = []

        # block_table是context里的，size(0)表示有多少个seq
        for i in range(block_table.size(0)):
            # 先确定当前这条 seq 的完整 KV 长度
            # decode: 用 context_lens
            # prefix prefill: 用 cu_seqlens_k 的差值
            if context.is_prefill:
                seq_len = (context.cu_seqlens_k[i+1] - context.cu_seqlens_k[i]).item()
            else:
                seq_len = context.context_lens[i].item()

            # 取出第 i 条 seq 的物理 block 列表
            block_table_ids = block_table[i].tolist()

            # 去掉padding的-1
            block_table_ids = [bid for bid in block_table_ids if bid != -1]
            if not block_table_ids:
                continue

            # 取出该 seq 用到的所有 block
            # shape: [num_blocks_for_seq, block_size, num_kv_heads, head_dim]
            seq_k_blocks = self.k_cache[block_table_ids]
            seq_v_blocks = self.v_cache[block_table_ids]

            # 把block展平到一维
            # 展平成连续 token 视角
            # [num_blocks * block_size, num_kv_heads, head_dim]
            seq_k = seq_k_blocks.reshape(-1, self.num_kv_heads, self.head_dim)
            seq_v = seq_v_blocks.reshape(-1, self.num_kv_heads, self.head_dim)
            seq_k = seq_k[:seq_len]
            seq_v = seq_v[:seq_len]
            all_k.append(seq_k)
            all_v.append(seq_v)
        return all_k, all_v

    def prefill(self, q, k, v, context):
        """
        当前这个版本给出的是“简单正确优先”的思路：
        1. 先写 cache
        2. 没 prefix 命中时，直接用当前局部 k/v 算 attention
        3. 有 prefix 命中时，再从 cache 里拼完整历史

        要求：
        q: [total_q_tokens, num_heads, head_dim]
        k: [total_q_tokens, num_kv_heads, head_dim]
        v: [total_q_tokens, num_kv_heads, head_dim]
        返回:
        attn_output: [num_heads, total_q_tokens, head_dim]
        """
        self.store_kv_cache(k, v, context.slot_mapping)

        # 没有prefix的时候
        if context.block_tables is None:
            cu_q = context.cu_seqlens_q  # 例如 [0, 7, 14]
            outputs = []

            # 关键点：
            # 虽然 q/k/v 是展平后的 total_tokens，
            # 但每条 seq 必须单独做 attention，不能跨 seq 互相看到
            # 遍历所有seq的起点
            for i in range(len(cu_q) - 1):
                q_start = cu_q[i].item()
                q_end = cu_q[i+1].item()

                seq_q = q[q_start:q_end]
                seq_k = k[q_start:q_end]
                seq_v = v[q_start:q_end]

                if self.num_heads != self.num_kv_heads:
                    # GQA 复制 KV 头以匹配 Q 头数
                    seq_k = seq_k.repeat_interleave(self.groups, dim=1)  # [seq_len, num_heads, head_dim], 扩充num_heads
                    seq_v = seq_v.repeat_interleave(self.groups, dim=1)  # [seq_len, num_heads, head_dim]
                
                seq_q = seq_q.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
                seq_k = seq_k.permute(1, 0, 2)
                seq_v = seq_v.permute(1, 0, 2) 

                # 计算attention
                seq_len = seq_q.size(1)
                # 当前 seq 内部做 causal attention
                scores = torch.matmul(seq_q, seq_k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [num_heads, seq_len, seq_len]
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))
                # print(f"[DEBUG] prefill: scores={scores}")
                attn_weights = F.softmax(scores, dim=-1)  # [num_heads, seq_len, seq_len]
                seq_output = torch.matmul(attn_weights, seq_v)  # [num_heads, seq_len, head_dim]
                outputs.append(seq_output)    

            # 按 token 维拼回去，保持和输入展平顺序一致
            attn_output = torch.cat(outputs, dim=1)  # [num_heads, total_tokens, head_dim]
            return attn_output
        
        # 有prefix的prefill
        
        all_k, all_v = self.get_kv_cache(context)
        outputs = []
        cu_q = context.cu_seqlens_q  # [0, q0, q1+q0...]

        # all_k表示可用的历史 KV 有多少，这里一定以all_k作为循环条件
        for i in range(len(all_k)):
            q_start = cu_q[i].item()
            q_end = cu_q[i+1].item()

            # 本轮需要处理的这个seq的query
            seq_q = q[q_start:q_end]

            # 当前 seq 可见的完整历史 KV
            # [seq_k_len, num_kv_heads, head_dim]
            full_k = all_k[i]
            full_v = all_v[i]

            if self.num_heads != self.num_kv_heads:
                # GQA 复制 KV 头以匹配 Q 头数
                full_k = full_k.repeat_interleave(self.groups, dim=1)  # [seq_len, num_heads, head_dim], 扩充num_heads
                full_v = full_v.repeat_interleave(self.groups, dim=1)  # [seq_len, num_heads, head_dim]
            
            # 转成 attention 计算形状
            # seq_q: [seq_q_len, num_heads, head_dim] -> [num_heads, seq_q_len, head_dim]
            # full_k: [seq_k_len, num_heads, head_dim] -> [num_heads, seq_k_len, head_dim]
            seq_q = seq_q.permute(1, 0, 2).float()
            full_k = full_k.permute(1, 0, 2).float()
            full_v = full_v.permute(1, 0, 2).float()

            scores = torch.matmul(seq_q, full_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            q_len = seq_q.size(1)
            k_len = full_k.size(1)

            # prefix prefill 的关键 mask
            # query 只对应“新增 token”
            # key 对应“完整历史（prefix + 新 token）”
            # 所以 mask 要右移 (k_len - q_len)
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, device=scores.device),
                diagonal=1 + (k_len - q_len)
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            seq_out = torch.matmul(attn_weights, full_v)  # [num_heads, q_len, head_dim]
            outputs.append(seq_out)
        
        # 4. 把每条 seq 的输出按原顺序拼回去
        # outputs 里每项形状: [num_heads, seq_q_len, head_dim]
        attn_output = torch.cat(outputs, dim=1)  # [num_heads, total_q_tokens, head_dim]
        return attn_output.to(self.dtype)


    def forward(self, q, k, v, context):
        """
        decode 路径：
        q, k, v:[num_kv_heads, head_dim]
        由于每个seq的长度不同，取出的kv是变长的，需要padding，然后mask
        """
        local_k = k
        local_v = v
        if local_k.dim() == 2:
            # 若[num_kv_heads, head_dim], qk需先处理成q,k:[1, num_kv_heads, head_dim]再存入kvcache
            local_k = local_k.unsqueeze(0)
            local_v = local_v.unsqueeze(0)
        
        self.store_kv_cache(local_k, local_v, context.slot_mapping)

        # [seq_len, num_kv_heads, head_dim],每个seq的seq_len不等
        all_k, all_v = self.get_kv_cache(context)

        # 当前先按 batch 中逐条 seq 处理，逻辑最清楚
        outputs = []
        for i in range(len(all_k)):
            seq_k = all_k[i]
            seq_v = all_v[i]

            # GQA: 复制 KV 头以匹配 Q 头数，待优化，可利用广播机制
            if self.num_heads != self.num_kv_heads:
                # [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
                seq_k = seq_k.repeat_interleave(self.groups, dim=1)
                seq_v = seq_v.repeat_interleave(self.groups, dim=1)

            seq_k = seq_k.permute(1, 2, 0).contiguous() # contiguous()确保存储连续。含义: [heads, head_dim, seq_len]
            seq_v = seq_v.permute(1, 0, 2).contiguous()  # [num_heads, seq_len, head_dim]

            # attention计算
            # 把q[i] unsqueeze成all_k的维度,[num_heads, head_dim] ->[1, num_heads, head_dim]
            seq_q = q[i].float()
            seq_q  = seq_q.unsqueeze(0)
            seq_k = seq_k.float()  # [num_heads, head_dim, seq_len]
            seq_v = seq_v.float()  # [num_heads, seq_len, head_dim]

            # scores: [1, num_heads, seq_len]
            # 矩阵乘法，k取转置，除以缩放因子
            # all_k: [num_heads, head_dim, seq_len]
            # scores: [1, num_heads, seq_len]
            # q的最后一维和k的最后两维做点积运算， 得到[1, num_heads, seq_len(score)]
            # scores:所有头对历史序列的打分，因此最后一个维度式seq_len，每个值都是该头的一个自注意力打分
            # scores = torch.matmul(q, all_k / self.head_dim ** 0.5)
            scores = torch.einsum('bhd,hds->bhs', seq_q, seq_k) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)  # 对最后一个维度的打分归一化
            # [1, num_heads, head_dim], attn_weight的最后一维和v的最后两维做点积运算,运用打分对v进行加权求和
            # output = torch.matmul(attn_weights, seq_v)
            # out:[1, num_heads, head_dim] -> [num_heads, head_dim]
            out = torch.einsum('bhs,hsd->bhd', attn_weights, seq_v).squeeze(0)
            outputs.append(out)
        # outputs:[batch_size, num_heads, head_dim]
        return torch.stack(outputs, dim=0).to(self.dtype)


