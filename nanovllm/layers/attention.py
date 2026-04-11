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

    def forward(self, q, block_manager, seq, layer):
        """
        q: [num_heads, head_dim] 当前 token 的 Query
        block_manager: BlockManager 实例
        seq: Sequence 实例（包含 block_table）
        layer: 当前层索引
        当前默认kv已经计算存储完毕，从block_table中取出所有的kv堆在一起进行注意力计算
        """


        block_size = block_manager.block_size
        block_table = seq.block_table
        num_tokens = len(seq.token_ids)  # 该seq的token的总数
        # 收集所有的kv
        all_k = []
        all_v = []

        for block_idx, block_id in enumerate(block_table):
            start = block_idx * block_size
            end = min(start+block_size, num_tokens)  # 避免最后不满的块越界
            for offset in range(start, end):
                k, v = block_manager.get_kv(block_id, offset, layer)  # layer不可省略
                all_k.append(k)  # [num_heads, head_dim]
                all_v.append(v)  # [num_heads, head_dim]


        # 把列表在第一个维度堆叠起来
        # 堆叠成连续序列
        # all_k: [seq_len, num_kv_heads, head_dim]
        all_k = torch.stack(all_k, dim=0)
        # all_v: [seq_len, num_kv_heads, head_dim]
        all_v = torch.stack(all_v, dim=0)

        # GQA: 复制 KV 头以匹配 Q 头数
        if self.num_heads != self.num_kv_heads:
            # [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
            all_k = all_k.repeat_interleave(self.groups, dim=1)
            all_v = all_v.repeat_interleave(self.groups, dim=1)

        all_k = all_k.permute(1, 2, 0)  # 含义: [heads, head_dim, seq_len]
        all_v = all_v.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]

        # attention计算
        # 把q unsqueeze成all_k的维度,[num_heads, head_dim] ->[1, num_heads, head_dim]
        q = torch.unsqueeze(q, dim=0)
        # scores: [1, num_heads, seq_len]
        # 矩阵乘法，k取转置，除以缩放因子
        # all_k: [num_heads, head_dim, seq_len]
        # scores: [1, num_heads, seq_len]
        # q的最后一维和k的最后两维做点积运算， 得到[1, num_heads, seq_len(score)]
        # scores:所有头对历史序列的打分，因此最后一个维度式seq_len，每个值都是该头的一个自注意力打分
        # scores = torch.matmul(q, all_k / self.head_dim ** 0.5)
        scores = torch.einsum('bhd,hds->bhs', q, all_k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # 对最后一个维度的打分归一化
        # [1, num_heads, head_dim], attn_weight的最后一维和v的最后两维做点积运算,运用打分对v进行加权求和
        # output = torch.matmul(attn_weights, all_v)
        output = torch.einsum('bhs,hsd->bhd', attn_weights, all_v)
        return output


