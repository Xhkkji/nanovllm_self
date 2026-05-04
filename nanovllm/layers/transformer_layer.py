import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import PagedAttention
from transformers import AutoModelForCausalLM


class TransformerLayer(nn.Module):
    def __init__(self, layer_id, hidden_size, intermediate_size, num_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        """
        layer_id: 推理时要经过很多层transformer block，每一层的kv都不一样，所以每一层的kv都需要存储
        """
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_head = num_heads
        self.head_dim = head_dim

        # 权重在总模型中赋值
        # kqv线性投影矩阵
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

        # FFN
        # self.w1 = nn.Linear(hidden_size, 4 * hidden_size, bias=False, dtype=dtype)
        # self.w2 = nn.Linear(4 * hidden_size, hidden_size, bias=False, dtype=dtype)
        # qwen模型使用SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

        # layernorm
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        # PagedAttention
        self.p_attn = PagedAttention(num_heads, head_dim)



    def forward(self, x, block_manager, seq, is_prefill=False):
        residual = x
        x = self.ln1(x)

        q = self.q_proj(x).view(self.num_head, self.head_dim)
        k = self.k_proj(x).view(self.num_head, self.head_dim)
        v = self.v_proj(x).view(self.num_head, self.head_dim)

        token_pos = len(seq.token_ids) - 1  # 传入的seq是动态增长的
        block_idx = token_pos // block_manager.block_size  # 算出最后一个token属于哪个块
        offset = token_pos % block_manager.block_size
        block_id = seq.block_table[block_idx]
        block_manager.set_kv(block_id, offset, self.layer_id, k, v)  # 一个层传入一次，按顺序与层数对应

        attn_out = self.p_attn(q, block_manager, seq, self.layer_id)
        attn_out = self.o_proj(attn_out.view(-1))  # 自动展成1维
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        # SwiGLU
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        x = residual + x
        return x



