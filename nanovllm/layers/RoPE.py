from functools import lru_cache
import torch
from torch import nn

def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    RoPE 的数学定义就是基于“成对旋转”的
    RoPE 的核心操作是在二维平面上旋转一个点
    RoPE 把高维向量拆成多个二维平面，在每个平面上独立旋转
    假设 x 的最后一维是 head_size = 128：
    x1 形状：[..., 64] → 取第 0、2、4... 126 维
    x2 形状：[..., 64] → 取第 1、3、5... 127 维
    配对逻辑：x[0] 与 x[1] 组成第一对，x[2] 与 x[3] 组成第二对，依此类推。
    每对 (x1[i], x2[i]) 就是一个二维平面上的坐标，用同一个旋转角度 θ_i 旋转。
    """
    # 将x沿着最后一个维度head_dim分割成两块
    # 提升精度到float是避免运算时精度损失
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    """
    预计算所有位置（0 到 max_position_embeddings-1）的 cos 和 sin 值，避免重复计算
    """
    def __init__(self, 
        head_dim: int, 
        rotary_dim: int,  # 旋转维度，通常等于head_dim
        max_position_embeddings: int,  # 最长序列长度
        base: float  # 频率基数，默认10000
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        assert head_dim == rotary_dim  # Q 和 K 的全部维度都应用旋转
        #  倒数频率,shape = [rotary_dim//2],例如 [f0, f1, f2] （每个维度的固有频率）
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        Absolute_PE = torch.arange(0, max_position_embeddings)
        # 两个一维向量[i]、[j]做内积，得到[i, j]内积矩阵
        # Absolute_PE 的形状：[max_position]，表示绝对位置 m。
        # 外积，得到形状 [max_position, rotary_dim/2] 的矩阵。其中元素 [m, i] = m * inv_freq[i]，即每个位置对每个维度的旋转角度。
        freqs = torch.einsum('i,j->ij', Absolute_PE, inv_freq)
        # 对每个角度计算 cos 和 sin，形状均为 [max_position, rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # concat为[max_position, rotary_dim]
        # 然后扩展为[max_position, 1, rotary_dim],便于后续广播为了方便广播到 [batch, seq_len, head_num, rotary_dim] 的 Q/K 张量(即head_num维度)
        cache = torch.cat((cos, sin), dim=1).unsqueeze(1)
        # 注册到网络缓存区，persistent=False表示不加入到state_dict
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,   # 形状 [batch, seq_len] 或 [seq_len]，表示每个 token 的绝对位置
        query: torch.Tensor,       # 形状 [seq_len, head_num, head_size]
        key: torch.Tensor,         # 同上
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据 positions 中的每个索引值，从 cos_sin_cache 的第 0 维（位置维）取出对应行的数据。
        # 索引操作会保留 positions 的前面所有维度，并附加 cos_sin_cache 的剩余维度

        cos_sin = self.cos_sin_cache[positions]  # [seq_len, 1, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)  # 各为 [seq_len, 1, rotary_dim/2]

        # query\key 形状 [seq_len, head_num, head_size]，head_size = rotary_dim。
        # cos, sin 形状 [seq_len, 1, rotary_dim/2]，广播到所有头。
        query = apply_rotary_embedding(query, cos, sin)
        key = apply_rotary_embedding(key, cos, sin)
        return query, key



