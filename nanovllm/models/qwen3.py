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

from nanovllm.distributed.parallel_state import (
    get_tp_rank,
    get_tp_size,
    tp_all_reduce,
)

class QwenDecoderLayer(nn.Module):
    def __init__(
        self, 
        layer_id, 
        hidden_size, 
        intermediate_size, 
        num_heads, 
        num_kv_heads, 
        head_dim, 
        config, 
        dtype=torch.bfloat16
    ):
        super().__init__()
        """
        layer_id: 推理时要经过很多层transformer block，每一层的kv都不一样，所以每一层的kv都需要存储
        """
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.config = config
        
        # TP 基本信息。
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        
        assert num_heads % self.tp_size == 0
        assert num_kv_heads % self.tp_size == 0
        assert intermediate_size % self.tp_size == 0

        # 每个 rank 只负责一部分 attention heads。
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        self.intermediate_size = intermediate_size // self.tp_size

        # 权重在总模型中赋值
        # kqv线性投影矩阵
        # Column Parallel:
        # 每个 rank 只保存 q/k/v/gate/up 的一部分输出通道。
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype)  # [1024, 2048]
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype) #  [1024, 1024]
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)  # [1024, 1024]
        
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False, dtype=dtype)
       
        # qwen模型使用SwiGLU
        # Row Parallel:
        # 每个 rank 输入一部分通道，输出 hidden_size，之后 all_reduce。
        # 所以实际上O部分是先以行做点积(线性变换中的乘法)，然后竖着加起来，对应的是线性变换中的加操作
        # num_heads和num_kv_heads已经按照tp切分过
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False, dtype=dtype)  # [2048, 1024]
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False, dtype=dtype)

        # layernorm
        # self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        # self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln1 = RMSNorm(hidden_size, dtype=dtype, eps=self.config.rms_norm_eps)
        self.ln2 = RMSNorm(hidden_size, dtype=dtype, eps=self.config.rms_norm_eps)
        # 控制注意力分数的方差，防止梯度消失或爆炸，在计算出kqv之后、RoPE操作之前使用
        self.q_norm = RMSNorm(self.head_dim, dtype=dtype, eps=self.config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, dtype=dtype, eps=self.config.rms_norm_eps)

        # PagedAttention
        # 注意这里传入的是 local heads。
        self.p_attn = PagedAttention(self.num_heads, self.num_kv_heads, self.head_dim)

    def forward(self, x, positions, rotary_embedding, context):
        """
        通用入口，不再区分prefill和decode
        统一前向：
        x: [total_new_tokens, hidden_size]
        """
        residual = x
        x = self.ln1(x)
        # x.shape: [token_len, hidden_size]
        total_new_tokens = x.size(0)
        seq_len, encode_dim = x.shape
        # x = x.view(batch * seq_len(total_new_tokens), encode_dim)  # 展平为[batch*seq_len, hidden_size]，一次性投影所有token
        q = self.q_proj(x).view(total_new_tokens, self.num_heads, self.head_dim)  # [seq_len, num_heads, head_dim]
        k = self.k_proj(x).view(total_new_tokens, self.num_kv_heads, self.head_dim)  # [seq_len, num_kv_heads, head_dim]
        v = self.v_proj(x).view(total_new_tokens, self.num_kv_heads, self.head_dim)  # [seq_len, num_kv_heads, head_dim]

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

        # 直接调用forward通用接口，不分prefill和decode
        # 在p_attn中保存kvcache
        attn_output = self.p_attn(q, k, v, context)     
        # 统一：attn_out:[total_new_tokens, num_heads, head_dims]
        attn_output = attn_output.contiguous().view(total_new_tokens, self.num_heads * self.head_dim)  # [total_new_tokens, hidden_size]
        attn_output = self.o_proj(attn_output)  # [total_new_tokens]
        
        # TP: o_proj 是 Row Parallel，每个 rank 只算了一部分 head 的贡献。
        # 这里 all_reduce 后，所有 rank 得到一致的完整 hidden。 
        attn_output = tp_all_reduce(attn_output)
        
        x = attn_output + residual  # 残差连接

        residual = x
        x = self.ln2(x)
        # SwiGLU
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        
        # TP: down_proj 是 Row Parallel，每个 rank 只算了一部分 FFN 中间维度。
        # all_reduce 后得到完整 MLP 输出。
        x = tp_all_reduce(x)
        
        x = residual + x
        # 统一主链下始终保持 [total_new_tokens, hidden_size] 形状；
        # 不能在单 token 场景把 batch/token 维挤掉，否则下一层会收到 1D 张量。
        return x


    def prefill(self, x, positions, rotary_embedding, context):
        """
        prefill阶段一次性处理整个输入序列，计算并存储所有的kv
        暂且废弃不用
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
        # 统一：attn_out:[seq_len, num_heads, head_dims]
        # attn_output = attn_output.permute(1, 0, 2).contiguous().view(seq_len, self.num_heads * self.head_dim)  # [seq_len, hidden_size]
        attn_output = attn_output.contiguous().view(seq_len, self.num_heads * self.head_dim)  # [seq_len, hidden_size]
        
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


    def decode(self, x, positions, rotary_embedding, context):
        """
        需要处理batch维度
        前forward函数。暂废弃不用
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
    def __init__(self, config, model_path="/home/xhk/model/Qwen3-0.6B/"):
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

        # TP 最小主线：embedding 每个 rank 暂时完整复制一份。
        # dtype 必须和后续 Linear 保持一致，否则第一层 q_proj 会出现
        # float / bfloat16 mismatch。
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, dtype=self.dtype)

        # RoPE
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_dim = self.head_dim
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.rotary_embedding = RotaryEmbedding(self.head_dim, self.rotary_dim, self.max_position_embeddings, rope_theta)

        # tp
        self.tp_size = get_tp_size()
        self.tp_rank = get_tp_rank()
        assert self.num_heads % self.tp_size == 0
        assert self.num_kv_heads % self.tp_size == 0
        
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_num_kv_heads = self.num_kv_heads // self.tp_size
        
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

        # TP/大模型测试主线：模型结构由外部传入的 config 决定，权重也必须从同一个
        # model_path 加载。否则会出现“14B 结构加载 0.6B 权重”的 shape mismatch。
        self.load_pretrain_weights(model_path)

    def load_pretrain_weights(self, model="/home/xhk/model/Qwen3-0.6B/"):
        print(f'开始加载权重:{model}')
        # 第一版为了最小实现，可以先 CPU 加载完整权重。
        # 后续做大模型时，再换成 safetensors 分片加载，避免每个 rank 都先吃完整模型内存。
        pretrained = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            )
        
        tp_rank = get_tp_rank()
        tp_size = get_tp_size()
        
        # embedding / norm / lm_head 第一版先复制到每个 rank。
        # 对于 Embedding 层、最终的 LayerNorm 层和输出层（lm_head），张量并行（TP）不进行切分。
        # 因此，每一张 GPU 卡都需要一份完整的权重副本。
        # TP 最小主线：copy 权重时显式转成目标参数 dtype/device。
        # 否则源 HF 权重 dtype 会污染本地模块，导致 forward 里出现 dtype mismatch。
        self.embed_tokens.weight.data.copy_(
            pretrained.model.embed_tokens.weight.data.to(
                dtype=self.embed_tokens.weight.dtype,
                device=self.embed_tokens.weight.device,
            )
        )
        self.norm.weight.data.copy_(
            pretrained.model.norm.weight.data.to(
                dtype=self.norm.weight.dtype,
                device=self.norm.weight.device,
            )
        )
        self.lm_head.weight.data.copy_(
            pretrained.lm_head.weight.data.to(
                dtype=self.lm_head.weight.dtype,
                device=self.lm_head.weight.device,
            )
        )
            
        pretrained_layers = pretrained.model.layers

        for i, layer in tqdm(enumerate(self.layers)):
            src = pretrained_layers[i].self_attn
            
            local_q_heads = layer.num_heads
            local_kv_heads = layer.num_kv_heads
            hd = layer.head_dim
            
            q_start = tp_rank * local_q_heads * hd
            q_end = (tp_rank + 1) * local_q_heads * hd

            kv_start = tp_rank * local_kv_heads * hd
            kv_end = (tp_rank + 1) * local_kv_heads * hd
            
            # q/k/v: 切 output rows, 按行切分。
            layer.q_proj.weight.data.copy_(
                src.q_proj.weight.data[q_start:q_end, :].to(
                    dtype=layer.q_proj.weight.dtype,
                    device=layer.q_proj.weight.device,
                )
            )
            layer.k_proj.weight.data.copy_(
                src.k_proj.weight.data[kv_start:kv_end, :].to(
                    dtype=layer.k_proj.weight.dtype,
                    device=layer.k_proj.weight.device,
                )
            )
            layer.v_proj.weight.data.copy_(
                src.v_proj.weight.data[kv_start:kv_end, :].to(
                    dtype=layer.v_proj.weight.dtype,
                    device=layer.v_proj.weight.device,
                )
            )
            
            # o_proj: 切 input columns，按列切。
            layer.o_proj.weight.data.copy_(
                src.o_proj.weight.data[:, q_start:q_end].to(
                    dtype=layer.o_proj.weight.dtype,
                    device=layer.o_proj.weight.device,
                )
            )
            
            mlp = pretrained_layers[i].mlp
            local_intermediated = layer.intermediate_size
            
            ffn_start = tp_rank * local_intermediated
            ffn_end = (tp_rank + 1) * local_intermediated
            
            # gate/up: 切 output rows。
            layer.gate_proj.weight.data.copy_(
                mlp.gate_proj.weight.data[ffn_start:ffn_end, :].to(
                    dtype=layer.gate_proj.weight.dtype,
                    device=layer.gate_proj.weight.device,
                )
            )
            layer.up_proj.weight.data.copy_(
                mlp.up_proj.weight.data[ffn_start:ffn_end, :].to(
                    dtype=layer.up_proj.weight.dtype,
                    device=layer.up_proj.weight.device,
                )
            )

            # down: 切 input columns。
            layer.down_proj.weight.data.copy_(
                mlp.down_proj.weight.data[:, ffn_start:ffn_end].to(
                    dtype=layer.down_proj.weight.dtype,
                    device=layer.down_proj.weight.device,
                )
            )
            
            # norm 复制。
            layer.ln1.weight.data.copy_(
                pretrained_layers[i].input_layernorm.weight.data.to(
                    dtype=layer.ln1.weight.dtype,
                    device=layer.ln1.weight.device,
                )
            )
            layer.ln2.weight.data.copy_(
                pretrained_layers[i].post_attention_layernorm.weight.data.to(
                    dtype=layer.ln2.weight.dtype,
                    device=layer.ln2.weight.device,
                )
            )

            # q_norm/k_norm 每个 head_dim 一份，直接复制。
            layer.q_norm.weight.data.copy_(
                src.q_norm.weight.data.to(
                    dtype=layer.q_norm.weight.dtype,
                    device=layer.q_norm.weight.device,
                )
            )
            layer.k_norm.weight.data.copy_(
                src.k_norm.weight.data.to(
                    dtype=layer.k_norm.weight.dtype,
                    device=layer.k_norm.weight.device,
                )
            )

        del pretrained
            
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
            
            
        #     # QKV 投影
        #     layer.q_proj.weight.data = pretrained_layers[i].self_attn.q_proj.weight.data
        #     layer.k_proj.weight.data = pretrained_layers[i].self_attn.k_proj.weight.data
        #     layer.v_proj.weight.data = pretrained_layers[i].self_attn.v_proj.weight.data
        #     layer.o_proj.weight.data = pretrained_layers[i].self_attn.o_proj.weight.data

        #     # FFN
        #     layer.gate_proj.weight.data = pretrained_layers[i].mlp.gate_proj.weight.data.clone()
        #     layer.up_proj.weight.data = pretrained_layers[i].mlp.up_proj.weight.data.clone()
        #     layer.down_proj.weight.data = pretrained_layers[i].mlp.down_proj.weight.data.clone()

        #     # LayerNorm（如果有 bias）
        #     if hasattr(layer.ln1, 'weight'):
        #         layer.ln1.weight.data = pretrained_layers[i].input_layernorm.weight.data
        #         layer.ln2.weight.data = pretrained_layers[i].post_attention_layernorm.weight.data
            
        #     layer.q_norm.weight.data = pretrained_layers[i].self_attn.q_norm.weight.data
        #     layer.k_norm.weight.data = pretrained_layers[i].self_attn.k_norm.weight.data


        # # 复制 embedding 和 lm_head
        # self.embed_tokens.weight.data = pretrained.model.embed_tokens.weight.data
        # self.lm_head.weight.data = pretrained.lm_head.weight.data
        # # RMSNorm赋值
        # self.norm.weight.data = pretrained.model.norm.weight.data

        # print("✅ Weight loading completed!")


    def forward(self, token_ids, positions, context):
        """
            token_ids: [num_tokens] 当前要处理的 token
            positions: [num_tokens] 位置编码（可选）
            prefill和decode传入的都是一维向量，可以等同处理
        """

        x = self.embed_tokens(token_ids)  # [num_tokens, hidden_size]
        # print(f"[DEBUG] forward: self.hidden_size={self.hidden_size}")
        # print(f"[DEBUG] forward: x.shape={x.shape}")


        # x.shape:torch.Size([token_len, embeding_size])
        for layer in self.layers:
            x = layer.forward(x, positions=positions, rotary_embedding=self.rotary_embedding, context=context)
        # print(f"[DEBUG] prefill forward: after layers, x.shape={x.shape}")

        # x : [total_new_tokens, hidden_size]
        x = self.norm(x)

        # 只保留需要logit运算的seq
        if context.seq_need_compute_logits.numel() > 0:
            # 取出每条seq的最后一个token
            # context.seq_need_compute_logits + 1:cu_sl_q存的是每个seq的起始位置
            # 取第seqi时，cu[i+1]是seqi+1的起始位置，再-1就是seqi的最后一个token
            last_indice = context.cu_seqlens_q[context.seq_need_compute_logits + 1] - 1
            # x :[num_seqs, vocab_size]
            x = x[last_indice].contiguous()
        else:
            # 没有任何序列需要计算 logits
            x = x[:0]  # 返回空张量

        logits = self.lm_head(x)
        # 此处没有batch维度
        return logits

