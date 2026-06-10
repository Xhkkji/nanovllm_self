import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
# from ..layers.transformer_layer import TransformerLayer
from tqdm import tqdm
from transformers import AutoConfig
from ..layers.attention import PagedAttention
from ..layers.RMSNorm import RMSNorm
from ..layers.RoPE import RotaryEmbedding

class QwenDecoderLayer(nn.Module):
    def __init__(self, layer_id, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, config, dtype=torch.bfloat16):
        super().__init__()
        """
        layer_id: 推理时要经过很多层transformer block，每一层的kv都不一样，所以每一层的kv都需要存储
        """
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.config = config

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
        self.ln1 = RMSNorm(hidden_size, dtype=dtype, eps=self.config.rms_norm_eps)
        self.ln2 = RMSNorm(hidden_size, dtype=dtype, eps=self.config.rms_norm_eps)
        # 控制注意力分数的方差，防止梯度消失或爆炸，在计算出kqv之后、RoPE操作之前使用
        self.q_norm = RMSNorm(self.head_dim, dtype=dtype, eps=self.config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, dtype=dtype, eps=self.config.rms_norm_eps)


        # PagedAttention
        self.p_attn = PagedAttention(num_heads, self.num_kv_heads, self.head_dim)


    def prefill(self, x, positions, rotary_embedding, context):
        """
        prefill阶段一次性处理整个输入序列，计算并存储所有的kv
        
        """
        residual = x
        x = self.ln1(x)
        # x.shape: [token_len, hidden_size]
        seq_len, encode_dim = x.shape
        # x = x.view(batch * seq_len, encode_dim)  # 展平为[batch*seq_len, hidden_size]，一次性投影所有token
        q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim)  # [seq_len, num_heads, head_dim]
        k = self.k_proj(x).view(seq_len, self.num_kv_heads, self.head_dim)  # [seq_len, num_kv_heads, head_dim]
        v = self.v_proj(x).view(seq_len, self.num_kv_heads, self.head_dim)  # [seq_len, num_kv_heads, head_dim]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # 旋转位置编码
        q, k = rotary_embedding(positions, q, k)

        # if self.num_heads != self.num_kv_heads:
        #     # GQA 复制 KV 头以匹配 Q 头数
        #     groups = self.num_heads // self.num_kv_heads
        #     k = k.repeat_interleave(groups, dim=1)  # [seq_len, num_heads, head_dim], 扩充num_heads
        #     v = v.repeat_interleave(groups, dim=1)  # [seq_len, num_heads, head_dim]
        # q = q.permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
        # k = k.permute(1, 0, 2)
        # v = v.permute(1, 0, 2)  

        attn_output = self.p_attn.prefill(q, k, v, context)     
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(seq_len, self.num_heads * self.head_dim)  # [seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)  # [seq_len]
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


    def forward(self, x, positions, rotary_embedding, context):
        """
        需要处理batch维度
        """
        residual = x
        x = self.ln1(x)

        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)  # [b, 2048]
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)  # [b, 1024]
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)  # [b, 1024]
        # [Debug] decode: q.shape:torch.Size([16, 128]), k.shapetorch.Size([8, 128])
        # print(f"[Debug] decode: q.shape:{q.shape}, k.shape{k.shape}")
        
        q = self.q_norm(q)
        k = self.k_norm(k)


        # q = q.unsqueeze(0)
        # k = k.unsqueeze(0)
        q, k = rotary_embedding(positions, q, k)
        # q = q.squeeze(0)
        # k = k.squeeze(0)

        # token_pos = len(seq.token_ids) - 1  # 传入的seq是动态增长的
        # block_idx = token_pos // block_manager.block_size  # 算出最后一个token属于哪个块
        # offset = token_pos % block_manager.block_size
        # block_id = seq.block_table[block_idx]
        # # block_manager.set_kv(block_id, offset, self.layer_id, k, v)  # 一个层传入一次，按顺序与层数对应
        # attn_out = self.p_attn(q, block_manager, seq, self.layer_id)  # 内部已经利用精度提升计算了，返回的是原dtype
        
        # attn_out = [batch, num_heads, head_dim]
        attn_out = self.p_attn(q, k, v, context)  # 内部已经利用精度提升计算了，返回的是原dtype
        attn_out = self.o_proj(attn_out.view(residual.size(0), -1))  # 自动展成1维
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

        # RoPE
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_dim = self.head_dim
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.rotary_embedding = RotaryEmbedding(self.head_dim, self.rotary_dim, self.max_position_embeddings, rope_theta)

        # Transformer
        self.layers = nn.ModuleList([
            QwenDecoderLayer(i, self.hidden_size, self.intermediate_size, self.num_heads, self.num_kv_heads, self.head_dim, config, self.dtype)
            for i in range(self.num_layers)
        ])

        # 最后一次的层归一化
        # self.norm = nn.LayerNorm(self.hidden_size, dtype=self.dtype)
        self.norm = RMSNorm(self.hidden_size, dtype=self.dtype)
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
            
            layer.q_norm.weight.data = pretrained_layers[i].self_attn.q_norm.weight.data
            layer.k_norm.weight.data = pretrained_layers[i].self_attn.k_norm.weight.data


        # 复制 embedding 和 lm_head
        self.embed_tokens.weight.data = pretrained.model.embed_tokens.weight.data
        self.lm_head.weight.data = pretrained.lm_head.weight.data
        # RMSNorm赋值
        self.norm.weight.data = pretrained.model.norm.weight.data

        print("✅ Weight loading completed!")


    def forward(self, token_ids, positions, context):
        """
            token_ids: [num_tokens] 当前要处理的 token
            positions: [num_tokens] 位置编码（可选）
            prefill和decode传入的都是一维向量，可以等同处理
        """

        x = self.embed_tokens(token_ids)  # [num_tokens, hidden_size]
        # print(f"[DEBUG] forward: self.hidden_size={self.hidden_size}")
        # print(f"[DEBUG] forward: x.shape={x.shape}")

        if context.is_prefill:
            # prefill
            # x.shape:torch.Size([token_len, embeding_size])
            print(f"prefill..")
            for layer in self.layers:
                x = layer.prefill(x, positions=positions, rotary_embedding=self.rotary_embedding, context=context)
            # x : [total_new_tokens, hidden_size]
            # print(f"[DEBUG] prefill forward: after layers, x.shape={x.shape}")
        
            
        else:
            # decode阶段，token_ids: [batch]，即一批seq的所有最后元素
            for layer in self.layers:
                x = layer.forward(x, positions=positions, rotary_embedding=self.rotary_embedding, context=context)  # Transformer前向过程中保存了kv
            # x :[num_seqs, vocab_size]
            # print(f"[DEBUG] decode forward: after layers, x.shape={x.shape}")
            
        # x : [total_new_tokens, hidden_size]
        x = self.norm(x)
        if context.is_prefill:
            # 取出每条seq的最后一个token
            last_indice = context.cu_seqlens_q[1:] - 1
            # x :[num_seqs, vocab_size]
            x = x[last_indice].contiguous()

        logits = self.lm_head(x)
        # 此处没有batch维度
        return logits

