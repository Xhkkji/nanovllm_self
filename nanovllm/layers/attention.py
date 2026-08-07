import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from time import perf_counter
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


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

        # 新增：存储 / 计算 / 路径 三个维度显式拆开
        self.kv_cache_dtype = torch.float16
        self.attention_compute_dtype = torch.float16
        self.decode_backend = "paged_flash"
        
        # kvcache量化
        # kv_cache_quant_mode 控制当前 attention 如何写/读 KV cache
        # scale_cache 只在 int8_mock 下使用
        self.kv_cache_quant_mode = "none"
        self.k_scale_cache = None
        self.v_scale_cache = None

        # 以下在modelrunner中逐层绑定
        self.k_cache = None
        self.v_cache = None
        self.block_size = None
        self.layer_id = None

        self.enable_profile = False
        self.profile_decode = dict(
            store=0.0,
            load=0.0,
            attn=0.0,
            gqa_expand=0.0,
            permute=0.0,
            qk=0.0,
            softmax=0.0,
            av=0.0,
            calls=0,
        )
        # CUDA Graph capture 期间不能做 synchronize / perf_counter 这类 profiling 操作
        self.in_cuda_graph_capture = False

        # 当前实现后端标记
        # 以后可以扩展为：
        # "torch" / "flashattn" / "triton"
        self.kv_writer_backend = "torch"
        self.forward_backend = "flashattn"

    def _should_profile(self):
        return self.enable_profile and (not self.in_cuda_graph_capture)

    def quantize_int8_per_token(self, x):
        """
        这两个函数实现了一个非常经典的 Per-Token 对称量化（Symmetric Quantization） 方法。
        它是在大模型推理中，为了压缩 KV Cache 而经常采用的基础技术。

        简单来说，它把一个高精度的浮点数张量，映射到 int8 范围（-127 到 127）
        并且为每一个 Token 的每一个注意力头单独计算一个缩放因子，从而在压缩内存的同时，尽可能保留精度。
        
        
        这个函数的核心任务，是把输入的浮点数张量 x 压缩成 int8 整数。

        输入 x：形状是 [num_tokens, num_kv_heads, head_dim]，这对应着在 Transformer 模型中，一组 Token 的 Key 或 Value 值，按注意力头分组后的表示。

        计算绝对最大值 (Absmax)：x_fp32.abs().amax(dim=-1, keepdim=True)
        这一步在 head_dim（即最后一维）上，找出每一个 Token 的每一个注意力头里，所有元素中绝对值最大的那个。keepdim=True 确保输出的形状是 [num_tokens, num_kv_heads, 1]。

        计算缩放因子 (Scale)：(absmax / 127.0).clamp(min=1e-6)
        这是量化的核心。我们想把 [-absmax, absmax] 这个范围映射到 [-127, 127]，所以缩放因子 scale = absmax / 127。加一个很小的 1e-6 是为了防止 absmax 为 0 时导致除零错误。

        执行量化 (Quantization)：torch.round(x_fp32 / scale).clamp(-127, 127).to(torch.int8)

        首先将每个元素除以它对应的 scale，让数值范围落到 [-127, 127] 附近。
        round 四舍五入取整。
        clamp 把数值截断在 -127 到 127 之间，防止极少数异常值溢出。
        最后将数据类型转为 int8，完成内存压缩。
        返回：返回量化后的 int8 张量 q 和计算出的 scale 张量。scale 在后续反量化时必须用到。
        x: [num_tokens, num_kv_heads, head_dim]
        return:
        q:     int8  [num_tokens, num_kv_heads, head_dim]
        scale: float [num_tokens, num_kv_heads, 1]
        """
        x_fp32 = x.float()
        absmax = x_fp32.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / 127.0).clamp(min=1e-6)
        q = torch.round(x_fp32 / scale).clamp(-127, 127).to(torch.int8)
        return q, scale
    
    def dequantize_int8_per_token(self, q, scale):
        """
        这个函数是量化的逆过程，用于把 int8 数据还原回浮点数，以便参与如 Flash Attention 等需要浮点数运算的模块。

        输入 q 和 scale：q 是量化后的 int8 张量，scale 是在量化时计算并保存的缩放因子。

        执行反量化 (Dequantization)：(q.float() * scale.float()).to(self.attention_compute_dtype)
        逻辑非常简单直接：

        将 int8 的 q 转回 float 类型。
        将 q 乘回它对应的 scale，就恢复到了原始的数值范围。
        最后将数据类型转换为模型计算时使用的浮点类型（如 fp16 或 bf16），以便进行后续的注意力计算。
        
        
        q:     int8  [*, num_kv_heads, head_dim]
        scale: float [*, num_kv_heads, 1]
        """
        return (q.float() * scale.float()).to(self.attention_compute_dtype)
    
    
    def write_kv_cache(self, k, v, slot_mapping):
        """
        统一的 KV 写入入口。
        当前内部调用 PyTorch 版实现。
        以后如果接 Triton，只替换这里的分发逻辑即可。

        输入：
        k: [num_tokens, num_kv_heads, head_dim]
        v: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens]
        """
        if self.kv_writer_backend == "torch":
            return self.store_kv_cache_torch(k, v, slot_mapping)
        elif self.kv_writer_backend == "triton":
            return self.store_kv_cache_triton(k, v, slot_mapping)
        else:
            raise NotImplementedError(f"unknown kv writer backend: {self.kv_writer_backend}")


    def store_kv_cache_torch(self, k, v, slot_mapping):
        """
        torch版存储kvcache
        k: [num_tokens, num_kv_heads, head_dim]
        v: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens]

        20260724 第一版低精度 KV cache 只做 dtype cast，不做真正 int8/fp8 量化。
        
        20260725 int8量化
        普通模式：
        K/V cast 到 bf16/fp16 后直接写入 cache

        int8_mock：
        K/V 先量化成 int8
        int8 K/V 写入 kv_cache
        scale 写入 scale_cache
        """
        # in8量化
        block_id = slot_mapping // self.block_size
        offset = slot_mapping % self.block_size
        
        if self.kv_cache_quant_mode == "int8_mock":
            qk, k_scale = self.quantize_int8_per_token(k)
            qv, v_scale = self.quantize_int8_per_token(v)
        
            self.k_cache[block_id, offset] = qk
            self.v_cache[block_id, offset] = qv
            
            self.k_scale_cache[block_id, offset] = k_scale.to(self.k_scale_cache.dtype)
            self.v_scale_cache[block_id, offset] = v_scale.to(self.v_scale_cache.dtype)
            return
        
        
        # 逐元素运算
        cache_dtype = self.k_cache.dtype
        if k.dtype != cache_dtype:
            k = k.to(cache_dtype)
        if v.dtype != cache_dtype:
            v = v.to(cache_dtype)

        block_id = slot_mapping // self.block_size
        offset = slot_mapping % self.block_size

        self.k_cache[block_id, offset] = k
        self.v_cache[block_id, offset] = v
        # 等价于（但效率更高）：
        # self.k_cache[1, 1] = k[0]  # token0 → 块1，偏移1
        # self.k_cache[0, 2] = k[1]  # token1 → 块0，偏移2
        # self.k_cache[2, 1] = k[2]  # token2 → 块2，偏移1
        # self.k_cache[1, 3] = k[3]  # token3 → 块1，偏移3


        # num_tokens = k.size(0)
        # for i in range(num_tokens):
        #     slot = slot_mapping[i].item()
        #     # 把线性槽位映射到kv的block维度
        #     block_id = slot // self.block_size
        #     offset = slot % self.block_size
        #     self.k_cache[block_id, offset] = k[i]
        #     self.v_cache[block_id, offset] = v[i]

    def get_kv_cache(self, context):
        """
        根据 block_tables + context_lens，从全局 cache 中取出当前 batch 真正可见的历史 KV。

        返回：
        k_batch: [batch, max_seq_len, num_kv_heads, head_dim]
        v_batch: [batch, max_seq_len, num_kv_heads, head_dim]
        kv_mask: [batch, max_seq_len]，True 表示有效位置

        prefill过程，context.context_len字段没有赋值，需要单独处理
        """
        # block_tables: [batch, max_num_blocks]
        # context_lens: [batch]
        block_table = context.block_tables
        seq_lens = context.context_lens

        batch_size = block_table.size(0)
        max_seqs_len = int(seq_lens.max().item())

        # 若量化模式，k_cache.dtype = int8_mock
        # k\v_batch作为最终结果需要bf16/fp16格式
        cache_out_dtype = (
            self.attention_compute_dtype
            if self.kv_cache_quant_mode == "int8_mock"
            else self.k_cache.dtype
        )
        k_batch = torch.zeros(
            batch_size,
            max_seqs_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=cache_out_dtype,
            device=self.k_cache.device,
        )

        v_batch = torch.zeros(
            batch_size,
            max_seqs_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=cache_out_dtype,
            device=self.v_cache.device,
        )

        kv_mask = torch.zeros(
        batch_size,
        max_seqs_len,
        dtype=torch.bool,
        device=self.k_cache.device,
        )

        # 简化（非矢量化）
        for i in range(batch_size):
            row = block_table[i]
            valid_block_ids = row[row != -1]

            if valid_block_ids.numel() == 0:
                continue

            # 取出当前seq的长度
            seq_len = int(seq_lens[i].item())

            # 拉平为 token 维
            # [num_blocks * block_size, num_kv_heads, head_dim]
            seq_k = self.k_cache[valid_block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
            seq_v = self.v_cache[valid_block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]

            if self.kv_cache_quant_mode == "int8_mock":
                # 取出scale值，float类型
                seq_k_scale = self.k_scale_cache[valid_block_ids].reshape(-1, self.num_kv_heads, 1)[:seq_len]
                seq_v_scale = self.v_scale_cache[valid_block_ids].reshape(-1, self.num_kv_heads, 1)[:seq_len]
                seq_k = self.dequantize_int8_per_token(seq_k, seq_k_scale)
                seq_v = self.dequantize_int8_per_token(seq_v, seq_v_scale)

            k_batch[i, :seq_len] = seq_k
            v_batch[i, :seq_len] = seq_v
            kv_mask[i, :seq_len] = True

        if self.kv_cache_quant_mode != "int8_mock":
            compute_dtype = self.attention_compute_dtype
            if compute_dtype is not None:
                k_batch = k_batch.to(compute_dtype)
                v_batch = v_batch.to(compute_dtype)

        return k_batch, v_batch, kv_mask


    def prefill(self, q, k, v, context):
        """
        prefill 路径统一入口。
        职责：
        1. 先写 KV
        2. 再根据 backend 选择 prefill attention 实现
        """
        self.write_kv_cache(k, v, context.slot_mapping)
        if self.forward_backend == "torch":
            return self.prefill_torch(q, k, v, context)
        elif self.forward_backend == "flashattn":
            return self.prefill_flashattn(q, k, v, context)
        else:
            raise NotImplementedError(f"unknown prefill backend: {self.forward_backend}")

    def prefill_flashattn(self, q, k, v, context):
        """
         统一 varlen flash attention 路线。

        当前虽然函数名仍叫 prefill_flashattn，
        但它实际同时支持：
        - 普通 prefill
        - prefix / chunked prefill
        - decode(q_len=1) 这个特例

        q: [total_q_tokens, num_heads, head_dim]
        k: [total_q_tokens, num_kv_heads, head_dim]
        v: [total_q_tokens, num_kv_heads, head_dim]

        out: [total_q_tokens, num_kv_heads, head_dim]
        """
        # 量化保护，flash-attn 当前不能直接吃 int8 KV cache
        if self.kv_cache_quant_mode != "none":
            raise RuntimeError(
                "int8_mock KV cache does not support flash-attn; "
                "use torch attention backend for quantization correctness tests"
            )
        
        # dtype保护
        if self.k_cache.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"flashattn only supports fp16/bf16 kv cache, got {self.k_cache.dtype}"
            )

        # dtype保护
        if self.k_cache.dtype != self.attention_compute_dtype:
            raise RuntimeError(
                f"flashattn only supports the same dtype, got {self.k_cache.dtype} and {self.attention_compute_dtype}"
            )

        # 没 prefix 的情况：直接用当前局部 q/k/v
        if context.block_tables is None:
            q = q.to(self.attention_compute_dtype)
            k = k.to(self.attention_compute_dtype)
            v = v.to(self.attention_compute_dtype)
            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )
            return out.to(self.dtype)

        # prefix prefill：注意这里不再使用局部 k/v，
        # 而是直接用全局 cache + block_table
        # 有 prefix cache / chunked prefill / decode 历史时：
        # 直接让 flash 内核从 paged cache 中读取完整 KV

        # 改进偏移问题
        q = q.to(self.attention_compute_dtype)
        out = flash_attn_varlen_func(
            q=q,
            k=self.k_cache,
            v=self.v_cache,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            causal=True,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            block_table=context.block_tables,
        )
        return out.to(self.dtype)


    def prefill_torch(self, q, k, v, context):
        """
        当前这个版本给出的是“简单正确优先”的思路：
        1. 先写 cache
        2. 没 prefix 命中时，直接用当前局部 k/v 算 attention
        3. 有 prefix 命中时，再从 cache 里拼完整历史

        要求：
        q: [total_q_tokens, num_heads, head_dim]
        k: [total_q_tokens, num_kv_heads, head_dim]
        v: [total_q_tokens, num_kv_heads, head_dim]

        k_batch: [batch, max_seq_len, num_kv_heads, head_dim]
        v_batch: [batch, max_seq_len, num_kv_heads, head_dim]
        kv_mask: [batch, max_seq_len]
        返回:
        # attn_output: [num_heads, total_q_tokens, head_dim]
        attn_output: [total_q_tokens, num_heads, head_dim]

        20260703：已经更新为统一语义”的 torch attention
        """

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

                seq_q = seq_q.permute(1, 0, 2).float()  # [num_heads, seq_len, head_dim]
                seq_k = seq_k.permute(1, 0, 2).float()
                seq_v = seq_v.permute(1, 0, 2) .float()

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
            return attn_output.permute(1, 0, 2).to(self.dtype)

        # 有prefix的prefill
        # 2. 有 prefix / chunked prefill / decode 历史时：
        # 从全局 cache 取完整可见 KV，再按 q_len/k_len 做 mask
        k_batch, v_batch, kv_mask = self.get_kv_cache(context)
        outputs = []
        cu_q = context.cu_seqlens_q  # [0, q0, q1+q0...]
        cu_k = context.cu_seqlens_k
        batch_size = k_batch.size(0)
        # all_k表示可用的历史 KV 有多少，这里一定以all_k作为循环条件
        for i in range(batch_size):
            q_start = cu_q[i].item()
            q_end = cu_q[i+1].item()

            # 本轮需要处理的这个seq的query
            seq_q = q[q_start:q_end]

            # 当前 seq 完整可见历史长度
            k_len = int(cu_k[i + 1].item() - cu_k[i].item())
            # 当前 seq 可见的完整历史 KV
            # [seq_k_len, num_kv_heads, head_dim]
            full_k = k_batch[i, :k_len]
            full_v = v_batch[i, :k_len]

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
        return attn_output.permute(1, 0, 2).to(self.dtype)

    def forward_unified(self, q, k, v, context):
        """
        统一 attention 主链：
        - q: [total_new_tokens, num_heads, head_dim]
        - k: [total_new_tokens, num_kv_heads, head_dim]
        - v: [total_new_tokens, num_kv_heads, head_dim]

        不再区分整批是 prefill 还是 decode。
        每条 seq 的 q_len 由 context.cu_seqlens_q 决定，
        每条 seq 的完整可见历史由 context.cu_seqlens_k 决定。
        """
        should_profile = self._should_profile()
        if should_profile:
            torch.cuda.synchronize()
            t0 = perf_counter()

        self.write_kv_cache(k, v, context.slot_mapping)

        if should_profile:
            torch.cuda.synchronize()
            self.profile_decode["store"] += perf_counter() - t0

        if self.forward_backend == "torch":
            if should_profile:
                torch.cuda.synchronize()
                t0 = perf_counter()
            out = self.prefill_torch(q, k, v, context)
            if should_profile:
                torch.cuda.synchronize()
                self.profile_decode["attn"] += perf_counter() - t0
                self.profile_decode["calls"] += 1
            return out
        elif self.forward_backend == "flashattn":
            # 不需要getkvcache，直接吃 k_cache/v_cache
            # 再结合 context_lens + block_table
            # 直接在 paged cache 上做 attention
            if should_profile:
                torch.cuda.synchronize()
                t0 = perf_counter()
            out = self.prefill_flashattn(q, k, v, context)
            if should_profile:
                torch.cuda.synchronize()
                self.profile_decode["attn"] += perf_counter() - t0
                self.profile_decode["calls"] += 1
            return out
        else:
            raise NotImplementedError(f"unknown decode backend: {self.decode_backend}")

    def forward(self, q, k, v, context):
        # 通用的接口
        return self.forward_unified(q, k, v, context)

    def decode_flashattn(self, q, context):
        """
        FlashAttention decode 路径

        q: [batch, num_heads, head_dim]
        context.context_lens: [batch]
        context.block_tables: [batch, max_num_blocks]
        self.k_cache/self.v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        """
        # flash_attn_with_kvcache 期望 q 带一个 query_len 维
        # [batch, num_heads, head_dim] -> [batch, 1, num_heads, head_dim]
        # flash-attn 只支持 fp16 / bf16
        q = q.to(self.k_cache.dtype)
        q = q.unsqueeze(1).contiguous()

        # q：当前步的 Query 张量。形状通常是 [batch_size, num_heads, 1, head_dim]（序列长度为1，因为是逐词生成）。
        # k_cache / v_cache：这是预先分配好的缓存空间（显存），用于存储之前所有 token 的 Key 和 Value。
        # cache_seqlens=context.context_lens：一个整数数组，表示当前 batch 中每个序列已经生成了多少 token（即缓存中已有的有效长度）。
        # block_table=context.block_tables：这是一个映射表，用于 PagedAttention（分页注意力）。因为 KVCache 在显存中不是连续存储的，而是分块（page）存放的，block_table 告诉函数去哪里找每个序列的缓存块。
        # causal=True：启用因果掩码（Causal Mask）。确保当前 token 只能看到它自己以及它之前的 token，不能看到未来的 token（这在自回归生成中是必须的）。
        # softmax_scale=...：注意力分数缩放系数。标准 Transformer 里通常是 1 / sqrt(head_dim)，这里直接用数学公式计算出来了。
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            causal=True,
            softmax_scale=1.0 / math.sqrt(self.head_dim)
        )
        # 返回通常是 [batch, 1, num_heads, head_dim]
        out = out.squeeze(1)
        return out.to(self.dtype)

    def decode_attention_torch(self, q, k_batch, v_batch, kv_mask):
        """
        pytorch的低性能decode 路径：
        q:       [batch, num_heads, head_dim]
        k_batch: [batch, max_seq_len, num_kv_heads, head_dim]
        v_batch: [batch, max_seq_len, num_kv_heads, head_dim]
        kv_mask: [batch, max_seq_len]
        """
        should_profile = not self.in_cuda_graph_capture
        if should_profile:
            torch.cuda.synchronize()
            t0 = perf_counter()
        outputs = []


        # GQA: 复制 KV 头以匹配 Q 头数，待优化，可利用广播机制
        if self.num_heads != self.num_kv_heads:
            # [batch, seq_len, num_kv_heads, head_dim] -> [batch, seq_len, num_heads, head_dim]
            if should_profile:
                torch.cuda.synchronize()
                t_sub = perf_counter()
            k_batch = k_batch.repeat_interleave(self.groups, dim=2)
            v_batch = v_batch.repeat_interleave(self.groups, dim=2)
            if should_profile:
                torch.cuda.synchronize()
                self.profile_decode["gqa_expand"] += perf_counter() - t_sub

        if should_profile:
            torch.cuda.synchronize()
            t_sub = perf_counter()
        # k: [batch, num_heads, head_dim, max_seq_len]
        # v: [batch, num_heads, max_seq_len, head_dim]
        q = q.float()
        k_batch = k_batch.permute(0, 2, 3, 1).contiguous().float() # contiguous()确保存储连续。含义: [batchs, heads, head_dim, seq_len]
        v_batch = v_batch.permute(0, 2, 1, 3).contiguous().float()  # [batchs, num_heads, seq_len, head_dim]

        # attention计算
        if should_profile:
            torch.cuda.synchronize()
            self.profile_decode["permute"] += perf_counter() - t_sub

        # scores: [1, num_heads, seq_len]
        # 矩阵乘法，k取转置，除以缩放因子
        # qk: [batch, num_heads, max_seq_len]
        # scores: [batch, num_heads, seq_len]
        # q的最后一维和k的最后两维做点积运算， 得到[1, num_heads, seq_len(score)]
        # scores:所有头对历史序列的打分，因此最后一个维度式seq_len，每个值都是该头的一个自注意力打分
        # scores = torch.matmul(q, all_k / self.head_dim ** 0.5)
        if should_profile:
            torch.cuda.synchronize()
            t_sub = perf_counter()
        scores = torch.einsum('bhd,bhds->bhs', q, k_batch) / math.sqrt(self.head_dim)
        if should_profile:
            torch.cuda.synchronize()
            self.profile_decode["qk"] += perf_counter() - t_sub

        if should_profile:
            torch.cuda.synchronize()
            t_sub = perf_counter()
        # shape: [batch_size, 1, seq_len]
        # False表示要保留，True表示要屏蔽
        # masked_fill 需要的是：
        # True = 屏蔽
        # False = 保留
        # [[[False, False, False, True, True]],   # batch 0
        # [[False, False, True, True, True]]]    # batch 1
        scores = scores.masked_fill(~kv_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  # 对最后一个维度的打分归一化
        if should_profile:
            torch.cuda.synchronize()
            self.profile_decode["softmax"] += perf_counter() - t_sub

        # [batch, num_heads, head_dim], attn_weight的最后一维和v的最后两维做点积运算,运用打分对v进行加权求和
        # output = torch.matmul(attn_weights, seq_v)
        # out:[batch, num_heads, head_dim]
        if should_profile:
            torch.cuda.synchronize()
            t_sub = perf_counter()
        outputs = torch.einsum('bhs,bhsd->bhd', attn_weights, v_batch)
        if should_profile:
            torch.cuda.synchronize()
            self.profile_decode["av"] += perf_counter() - t_sub


        # outputs:[batch_size, num_heads, head_dim]
        if should_profile:
            torch.cuda.synchronize()
            self.profile_decode["attn"] += perf_counter() - t0
            self.profile_decode["calls"] += 1
        return outputs.to(self.dtype)


