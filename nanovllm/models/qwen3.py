import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
# from ..layers.transformer_layer import TransformerLayer
from tqdm import tqdm
from transformers import AutoConfig
from ..layers.attention import PagedAttention
from ..layers.RMSNorm import RMSNorm

class QwenDecoderLayer(nn.Module):
    def __init__(self, layer_id, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        """
        layer_id: 推理时要经过很多层transformer block，每一层的kv都不一样，所以每一层的kv都需要存储
        """
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

        # 权重在总模型中赋值
        # kqv线性投影矩阵
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype)  # [1024, 2048]
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype) #  [1024, 1024]
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)  # [1024, 1024]
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False, dtype=dtype)  # [2048, 1024]

        # FFN
        # self.w1 = nn.Linear(hidden_size, 4 * hidden_size, bias=False, dtype=dtype)
        # self.w2 = nn.Linear(4 * hidden_size, hidden_size, bias=False, dtype=dtype)
        # qwen模型使用SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

        # layernorm
        # self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        # self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln1 = RMSNorm(hidden_size, dtype=dtype)
        self.ln2 = RMSNorm(hidden_size, dtype=dtype)

        # PagedAttention
        self.p_attn = PagedAttention(num_heads, self.num_kv_heads, self.head_dim)

    def prefill(self, x, block_manager, seq):
        """
        prefill阶段一次性处理整个输入序列，计算并存储所有的kv
        x与seq对应，seq会记录哪些token的kv已经存储过了（通过num_cached_tokens），避免重复计算
        但是seq在prefill阶段不做修改
        """
        x = x.unsqueeze(0)  # [1, token_len, hidden_size]，添加batch维度，方便统一处理
        residual = x
        x = self.ln1(x)
        # x.shape: [1, token_len, hidden_size]
        batch, seq_len, encode_dim = x.shape
        # x = x.view(batch * seq_len, encode_dim)  # 展平为[batch*seq_len, hidden_size]，一次性投影所有token
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)  # [batch, seq_len, num_heads, head_dim]
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)  # [batch, seq_len, num_kv_heads, head_dim]
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)  # [batch, seq_len, num_kv_heads, head_dim]

        block_manager.set_kv_prefill(k, v, seq, self.layer_id)  # prefill阶段一次性存储所有token的kv

        if self.num_heads != self.num_kv_heads:
            # GQA 复制 KV 头以匹配 Q 头数
            groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(groups, dim=2)  # [batch, seq_len, num_heads, head_dim]
            v = v.repeat_interleave(groups, dim=2)  # [batch, seq_len, num_heads, head_dim]
        q = q.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, num_heads, seq_len, seq_len]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        # print(f"[DEBUG] prefill: scores={scores}")
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)  # [batch, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)  # [batch, seq_len
        x = attn_output + residual  # 残差连接

        residual = x
        x = self.ln2(x)
        # SwiGLU
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        x = residual + x
        x = x.squeeze(0)
        return x


    def forward(self, x, block_manager, seq, is_prefill=False):
        residual = x
        x = self.ln1(x)

        q = self.q_proj(x).view(self.num_heads, self.head_dim)  # [2048]
        k = self.k_proj(x).view(self.num_kv_heads, self.head_dim)  # [1024]
        v = self.v_proj(x).view(self.num_kv_heads, self.head_dim)  # [1024]

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

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config = AutoConfig.from_pretrained("/home/xhk/model/Qwen3-0.6B/")
        self.dtype = torch.bfloat16
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        # self.hidden_size = hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.intermediate_size = config.intermediate_size
        self.num_kv_heads = config.num_key_value_heads

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # Transformer
        self.layers = nn.ModuleList([
            QwenDecoderLayer(i, self.hidden_size, self.intermediate_size, self.num_heads, self.num_kv_heads, self.head_dim, self.dtype)
            for i in range(self.num_layers)
        ])

        # 最后一次的层归一化
        self.norm = nn.LayerNorm(self.hidden_size, dtype=self.dtype)
        # 最后的线性层，映射到词汇表每个单词的概率
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False, dtype=self.dtype)

        self.load_pretrain_weights()

    def load_pretrain_weights(self, model="/home/xhk/model/Qwen3-0.6B/"):
        print(f'开始加载权重:{model}')
        pretrained = AutoModelForCausalLM.from_pretrained(
            model,
            # torch_dtype=torch.float16,
            device_map="cuda:0")
        pretrained_layers = pretrained.model.layers

        for i, layer in tqdm(enumerate(self.layers)):
            # 打印形状对比
            # print(f"\nLayer {i}:")
            # print(f"  q_proj: src={pretrained_layers[i].self_attn.q_proj.weight.shape}, "
            #       f"dst={layer.q_proj.weight.shape}")
            # print(f"  k_proj: src={pretrained_layers[i].self_attn.k_proj.weight.shape}, "
            #       f"dst={layer.k_proj.weight.shape}")
            # print(f"  v_proj: src={pretrained_layers[i].self_attn.v_proj.weight.shape}, "
            #       f"dst={layer.v_proj.weight.shape}")
            # print(f"  o_proj: src={pretrained_layers[i].self_attn.o_proj.weight.shape}, "
            #       f"dst={layer.o_proj.weight.shape}")
            # q_proj: src = torch.Size([2048, 1024]), dst = torch.Size([1024, 1024])
            # k_proj: src = torch.Size([1024, 1024]), dst = torch.Size([1024, 1024])
            # v_proj: src = torch.Size([1024, 1024]), dst = torch.Size([1024, 1024])
            # o_proj: src = torch.Size([1024, 2048]), dst = torch.Size([1024, 1024])
            # pytorch的weight是反的，即[output, input]
            # QKV 投影
            layer.q_proj.weight.data = pretrained_layers[i].self_attn.q_proj.weight.data
            layer.k_proj.weight.data = pretrained_layers[i].self_attn.k_proj.weight.data
            layer.v_proj.weight.data = pretrained_layers[i].self_attn.v_proj.weight.data
            layer.o_proj.weight.data = pretrained_layers[i].self_attn.o_proj.weight.data

            # FFN
            layer.gate_proj.weight.data = pretrained_layers[i].mlp.gate_proj.weight.data.clone()
            layer.up_proj.weight.data = pretrained_layers[i].mlp.up_proj.weight.data.clone()
            layer.down_proj.weight.data = pretrained_layers[i].mlp.down_proj.weight.data.clone()

            # LayerNorm（如果有 bias）
            if hasattr(layer.ln1, 'weight'):
                layer.ln1.weight.data = pretrained_layers[i].input_layernorm.weight.data
                layer.ln2.weight.data = pretrained_layers[i].post_attention_layernorm.weight.data

        # 复制 embedding 和 lm_head
        self.embed_tokens.weight.data = pretrained.model.embed_tokens.weight.data
        self.lm_head.weight.data = pretrained.lm_head.weight.data

        print("✅ Weight loading completed!")


    def forward(self, token_ids, positions, block_manager, seq, is_prefill=False):
        """
            token_ids: [num_tokens] 当前要处理的 token
            positions: [num_tokens] 位置编码（可选）
        """

        x = self.embed_tokens(token_ids)  # [num_tokens, hidden_size]
        # print(f"[DEBUG] forward: self.hidden_size={self.hidden_size}")
        # print(f"[DEBUG] forward: x.shape={x.shape}")

        # if token_ids.dim() == 1 and token_ids.size(0) == 1:  # 一维张量且只有一个元素
        if not is_prefill:
            if x.dim() == 2:
                x = x.squeeze(0)  # [1, hidden_size] -> [hidden_size], 去掉token_len维度
            for layer in self.layers:
                x = layer.forward(x, block_manager, seq)  # Transformer前向过程中保存了kv
            # x.shape=torch.Size([1024])
            # print(f"[DEBUG] decode forward: after layers, x.shape={x.shape}")
        
        else:
            # prefill
            # x.shape:torch.Size([1, token_len, 1024])
            print(f"prefill..")
            for layer in self.layers:
                x = layer.prefill(x, block_manager, seq)
            # x.shape=torch.Size([11, 1024])
            # print(f"[DEBUG] prefill forward: after layers, x.shape={x.shape}")
        
        x = self.norm(x)
        logits = self.lm_head(x)
        # 此处没有batch维度
        return logits

