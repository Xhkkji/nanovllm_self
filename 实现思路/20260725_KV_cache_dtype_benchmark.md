# 20260725 KV Cache DType Benchmark

## 1. 测试目标

本轮测试用于回答第一阶段 `KV cache dtype` 可配置后的四个问题：

1. `bf16 KV cache` 和 `fp16 KV cache` 输出是否一致？
2. `TTFT / ITL / throughput` 差多少？
3. KV cache 显存理论节省是多少？
4. 哪个 dtype 更适合作为当前默认值？

本轮只比较：

```text
bf16 KV cache + bf16 attention compute
fp16 KV cache + fp16 attention compute
```

不比较 `fp32`，因为当前 flash-attn paged KV cache 路径不支持 `fp32 KV cache`，放进同一性能矩阵不公平。

## 2. 测试环境和命令

代码路径：

```text
/home/xhk/nanovllm_self
```

benchmark 脚本：

```text
scripts/benchmark_kv_cache_dtype.py
```

原始日志：

```text
logs/kv_cache_dtype_benchmark_20260725.jsonl
```

执行命令：

```bash
PYTHONPATH=/home/xhk/nanovllm_self \
/home/xhk/miniconda3/envs/pytorch/bin/python \
scripts/benchmark_kv_cache_dtype.py \
  --prompts short medium \
  --batch-sizes 1 4 \
  --gen-len 32 \
  --dtypes bf16 fp16 \
  --output logs/kv_cache_dtype_benchmark_20260725.jsonl
```

测试矩阵：

```text
prompt: short / medium
batch_size: 1 / 4
gen_len: 32
dtype: bf16 / fp16
block_size: 256
num_blocks: 256
```

计时口径：

- 不把模型加载计入正式计时。
- 每个 case 先跑一个短 warmup request。
- `avg_TTFT` 是 batch 内 request 首 token 时间的平均值。
- `ITL` 是首 token 之后的平均 token 间隔近似值。
- `pd_kv_cache_mb` 是 prefill engine + decode engine 两份 KV cache 的理论总容量。

## 3. 测试结果

| dtype | prompt | bs | gen | avg TTFT ms | ITL ms | decode tok/s | throughput tok/s | total s | PD KV cache MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | short | 1 | 32 | 82.41 | 19.98 | 50.06 | 45.61 | 0.7017 | 14336.00 |
| fp16 | short | 1 | 32 | 74.43 | 20.65 | 48.43 | 44.79 | 0.7145 | 14336.00 |
| bf16 | short | 4 | 32 | 144.44 | 5.42 | 184.50 | 156.76 | 0.8165 | 14336.00 |
| fp16 | short | 4 | 32 | 182.56 | 7.30 | 136.96 | 117.65 | 1.0879 | 14336.00 |
| bf16 | medium | 1 | 32 | 72.31 | 20.13 | 49.69 | 45.96 | 0.6962 | 14336.00 |
| fp16 | medium | 1 | 32 | 92.03 | 26.62 | 37.57 | 34.89 | 0.9172 | 14336.00 |
| bf16 | medium | 4 | 32 | 145.96 | 5.37 | 186.20 | 157.65 | 0.8119 | 14336.00 |
| fp16 | medium | 4 | 32 | 151.98 | 5.56 | 179.74 | 152.04 | 0.8419 | 14336.00 |

## 4. bf16 vs fp16 对比

| prompt | bs | 输出是否一致 | fp16 decode tok/s 相对 bf16 | fp16 throughput 相对 bf16 |
|---|---:|---|---:|---:|
| short | 1 | 否 | -3.25% | -1.80% |
| short | 4 | 否 | -25.77% | -24.95% |
| medium | 1 | 是 | -24.39% | -24.10% |
| medium | 4 | 是 | -3.47% | -3.56% |

输出分叉位置：

| prompt | bs | 分叉情况 |
|---|---:|---|
| short | 1 | 第 0 条 seq 在绝对 token index 31 开始分叉 |
| short | 4 | 4 条 seq 都在绝对 token index 31 开始分叉 |
| medium | 1 | 未分叉 |
| medium | 4 | 未分叉 |

这里的绝对 token index 包含 prompt token。`short` prompt 长度是 7，因此 index 31 大约对应第 25 个生成 token 附近。

## 5. 问题回答

### 5.1 bf16 KV cache 和 fp16 KV cache 输出是否一致？

不完全一致。

本轮结果：

```text
medium prompt: bf16/fp16 输出一致
short prompt : bf16/fp16 在 token index 31 附近发生分叉
```

结论：

```text
fp16 KV cache 可以跑通，但不能假设和 bf16 完全 token-level 一致。
```

这属于低精度 KV cache 常见现象：即使每步 logits 只有很小差异，greedy decode 在 top-1/token 边界接近时也可能分叉，后续 token 会沿不同上下文继续生成。

### 5.2 TTFT / ITL / throughput 差多少？

本轮单次 benchmark 中，`bf16` 整体更稳。

观察：

- `short bs=1`：fp16 TTFT 更低，但 decode/throughput 略低。
- `short bs=4`：fp16 明显慢于 bf16。
- `medium bs=1`：fp16 明显慢于 bf16。
- `medium bs=4`：fp16 略慢于 bf16。

结论：

```text
当前实现里 fp16 KV cache 没有带来稳定性能收益。
bf16 仍然是更合理的默认主链路。
```

注意：本轮是单次小矩阵，不是严格统计学 benchmark。后续如果要做正式性能结论，需要每个 case 多跑几次取均值和 p50/p90。

### 5.3 KV cache 显存理论节省是多少？

本轮 `bf16` 和 `fp16` 没有显存节省差异。

原因：

```text
bf16 = 2 bytes / element
fp16 = 2 bytes / element
```

因此：

```text
bf16 KV cache 和 fp16 KV cache 理论 KV 显存相同。
```

本轮配置：

```text
num_layers = 28
num_blocks = 256
block_size = 256
num_kv_heads = 8
head_dim = 128
dtype_bytes = 2
PD 两端各有一份 KV cache
```

单 engine KV cache 理论容量：

```text
2 * 28 * 256 * 256 * 8 * 128 * 2 bytes = 7168 MB
```

PD prefill + decode 双 engine 总容量：

```text
7168 MB * 2 = 14336 MB
```

真正能降低 KV cache 显存的是后续：

```text
int8 / fp8 KV cache + scale metadata
```

### 5.4 哪个 dtype 更适合作为默认？

当前建议继续使用：

```text
kv_cache_dtype = bf16
attention_compute_dtype = bf16
```

原因：

1. 默认在线主链路测试已经通过。
2. online PD vs offline PD correctness 对拍通过。
3. handoff payload 测试通过。
4. fp16/fp16 虽然可跑，但没有稳定性能优势。
5. fp16 在 short prompt case 出现 token-level 分叉。
6. bf16 动态范围更大，作为默认更稳。

## 6. 当前发现的问题

### 6.1 fp16 并不等于更快

本轮 fp16 在 3/4 个 case 中 throughput 低于 bf16。

可能原因：

- 当前模型权重/attention 主链路更贴近 bf16。
- GPU 对 bf16/fp16 的具体性能差异受硬件和 kernel 路径影响。
- 当前在线 PD benchmark 是小 batch、小 gen_len，kernel launch 和调度开销占比高。
- 单次测量存在波动，需要多轮均值确认。

### 6.2 bf16/fp16 可能产生 token drift

`short` prompt 在 token index 31 附近分叉。

这说明 dtype 切换不只是 storage 层变化，也会影响 decode 结果。后续做 int8/fp8 时必须保留 drift 对拍。

### 6.3 当前 KV cache 容量配置偏大

`num_blocks=256`、`block_size=256` 在 PD 双 engine 下理论 KV cache 约 14GB。

这个配置适合跑通 flash-attn paged KV cache 和中小 benchmark，但后续做更大 batch 或更多 worker 时，需要考虑：

- 按显存自动估算 `num_blocks`
- prefill/decode worker 分别配置 KV cache 容量
- online admission control 结合真实 free blocks

## 7. 下一步建议

推荐下一步进入真正 KV cache 量化 mock：

```text
int8 KV cache + scale metadata
```

最小路线：

1. 保留当前 `bf16` 作为 baseline。
2. 增加 `kv_cache_quant_mode = none / int8_mock`。
3. `write_kv_cache()` 中对 K/V 做 per-token 或 per-block int8 quantize。
4. `get_kv_cache()` / flash 前做 dequantize。
5. `KVConnector` / `HandoffPayload` 增加 scale metadata 传递。
6. 沿用本轮 benchmark 脚本，加 `int8_mock` case。
7. 对比 memory saving、decode speed、token drift。

第一版不追求 kernel-level 性能，只验证接口、显存估算和误差传播。

## 8. 面向第一段实习的量化实现边界

如果这个项目用于找第一段 AI Infra / 推理优化实习，量化部分不需要做到工业级 kernel。

不建议现在做：

```text
真实 FP8 CUDA kernel
FlashAttention 内部直接消费 int8/fp8 KV
TensorRT-LLM 级别量化推理
完整 weight / activation quantization
Triton int8 attention kernel
```

更合适的目标是把 KV cache 量化做成一个完整工程闭环：

```text
KV cache 写入量化
scale metadata 管理
KV cache 读取反量化
正确性 / drift 对拍
memory benchmark
PD handoff metadata 设计
```

推荐完成度：

```text
P0: int8_mock 单机 torch attention 路径跑通
P1: int8_mock PD handoff 支持 scale metadata
P2: benchmark + md，总结 memory saving / drift / throughput
P3: 写清楚和 vLLM / Mooncake 的边界
```

可以在面试中这样表述：

```text
我没有实现工业级 FP8 kernel，但我完整实现了 KV cache 从写入量化、scale metadata 管理、读取反量化、PD handoff 传输、正确性漂移和显存收益 benchmark 的闭环。
```

## 9. 最简化 int8_mock 修改样例

下面是基于当前 `nanovllm_self` 代码的最小修改样例。它不是最终高性能方案，目标是先把量化 KV cache 的压缩/解压语义跑通。

第一版建议只支持：

```text
kv_cache_quant_mode = none / int8_mock
int8_mock 只走 torch attention path
flash-attn 路径继续用于 bf16/fp16 baseline
```

### 9.1 Config 增加量化模式

位置：`nanovllm/config.py`

```python
@dataclass
class Config:
    model_path: str
    device: str = "cuda:0"

    dtype: torch.dtype | str = torch.float16
    kv_cache_dtype: torch.dtype | str = torch.bfloat16
    attention_compute_dtype: torch.dtype | str = torch.bfloat16

    # int8_mock 是教学/验证路径，不是高性能 kernel 路径。
    kv_cache_quant_mode: str = "none"  # "none" / "int8_mock"
    kv_cache_scale_dtype: torch.dtype | str = "fp32"

    block_size: int = 256
    num_blocks: int = 256

    def __post_init__(self):
        self.dtype = resolve_torch_dtype(self.dtype) or torch.float16
        self.kv_cache_dtype = resolve_torch_dtype(self.kv_cache_dtype) or self.dtype
        self.attention_compute_dtype = (
            resolve_torch_dtype(self.attention_compute_dtype) or self.dtype
        )
        self.kv_cache_scale_dtype = (
            resolve_torch_dtype(self.kv_cache_scale_dtype) or torch.float32
        )

        if self.kv_cache_quant_mode not in ("none", "int8_mock"):
            raise ValueError(f"unsupported kv_cache_quant_mode: {self.kv_cache_quant_mode}")

        if self.kv_cache_quant_mode == "int8_mock":
            # flash-attn 不能直接消费 int8 KV，第一版强制走 torch attention。
            self.kv_cache_dtype = torch.int8
```

### 9.2 ModelRunner 分配 scale cache

位置：`nanovllm/engine/model_runner.py`

```python
class ModelRunner(nn.Module):
    def __init__(self, config: Config):
        ...
        self.kv_cache = torch.zeros(
            2,
            self.model.num_layers,
            self.config.num_blocks,
            self.config.block_size,
            self.model.num_kv_heads,
            self.model.head_dim,
            dtype=self.config.kv_cache_dtype,
            device=self.config.device,
        )

        self.kv_scale_cache = None
        if self.config.kv_cache_quant_mode == "int8_mock":
            # per-token-per-kv-head scale: [2, layers, blocks, block_size, kv_heads, 1]
            self.kv_scale_cache = torch.ones(
                2,
                self.model.num_layers,
                self.config.num_blocks,
                self.config.block_size,
                self.model.num_kv_heads,
                1,
                dtype=self.config.kv_cache_scale_dtype,
                device=self.config.device,
            )

        self.bind_kvcache_to_attention()
```

### 9.3 bind_kvcache_to_attention 绑定 scale cache

位置：`nanovllm/engine/model_runner.py`

```python
def bind_kvcache_to_attention(self):
    for layer_id, layer in enumerate(self.model.layers):
        p_attn = layer.p_attn

        p_attn.k_cache = self.kv_cache[0, layer_id]
        p_attn.v_cache = self.kv_cache[1, layer_id]
        p_attn.block_size = self.config.block_size
        p_attn.layer_id = layer_id

        p_attn.kv_cache_dtype = self.kv_cache.dtype
        p_attn.attention_compute_dtype = self.config.attention_compute_dtype
        p_attn.kv_cache_quant_mode = self.config.kv_cache_quant_mode

        if self.kv_scale_cache is not None:
            p_attn.k_scale_cache = self.kv_scale_cache[0, layer_id]
            p_attn.v_scale_cache = self.kv_scale_cache[1, layer_id]
        else:
            p_attn.k_scale_cache = None
            p_attn.v_scale_cache = None
```

### 9.4 Attention 初始化新增字段

位置：`nanovllm/layers/attention.py`

```python
class PagedAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, dtype=torch.bfloat16):
        ...
        self.kv_cache_quant_mode = "none"
        self.k_scale_cache = None
        self.v_scale_cache = None
```

### 9.5 int8 量化 / 反量化工具函数

位置：`nanovllm/layers/attention.py`

```python
def quantize_int8_per_token(self, x):
    # x: [num_tokens, num_kv_heads, head_dim]
    x_fp32 = x.float()
    absmax = x_fp32.abs().amax(dim=-1, keepdim=True)
    scale = (absmax / 127.0).clamp(min=1e-6)
    q = torch.round(x_fp32 / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8_per_token(self, q, scale):
    return (q.float() * scale.float()).to(self.attention_compute_dtype)
```

### 9.6 write_kv_cache 支持 int8_mock

位置：`nanovllm/layers/attention.py`

```python
def store_kv_cache_torch(self, k, v, slot_mapping):
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

    cache_dtype = self.k_cache.dtype
    if k.dtype != cache_dtype:
        k = k.to(cache_dtype)
    if v.dtype != cache_dtype:
        v = v.to(cache_dtype)

    self.k_cache[block_id, offset] = k
    self.v_cache[block_id, offset] = v
```

### 9.7 get_kv_cache 支持反量化

位置：`nanovllm/layers/attention.py`

```python
def get_kv_cache(self, context):
    ...
    for i in range(batch_size):
        row = block_table[i]
        valid_block_ids = row[row != -1]
        if valid_block_ids.numel() == 0:
            continue

        seq_len = int(seq_lens[i].item())
        seq_k = self.k_cache[valid_block_ids].reshape(
            -1, self.num_kv_heads, self.head_dim
        )[:seq_len]
        seq_v = self.v_cache[valid_block_ids].reshape(
            -1, self.num_kv_heads, self.head_dim
        )[:seq_len]

        if self.kv_cache_quant_mode == "int8_mock":
            seq_k_scale = self.k_scale_cache[valid_block_ids].reshape(
                -1, self.num_kv_heads, 1
            )[:seq_len]
            seq_v_scale = self.v_scale_cache[valid_block_ids].reshape(
                -1, self.num_kv_heads, 1
            )[:seq_len]
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
```

### 9.8 int8_mock 禁止走 flash-attn

位置：`nanovllm/layers/attention.py`

```python
def prefill_flashattn(self, q, k, v, context):
    if self.kv_cache_quant_mode != "none":
        raise RuntimeError(
            "int8_mock KV cache does not support flash-attn; "
            "use torch attention backend for quantization correctness tests"
        )
    ...
```

同时在测试或构造 engine 时，把所有层设成：

```python
for layer in runner.model.layers:
    layer.p_attn.forward_backend = "torch"
```

### 9.9 最小测试建议

新增：`pd_self/evaluation/test_kv_cache_int8_mock.py`

覆盖：

```text
1. quantize/dequantize 单元测试
2. int8_mock 下 kv_cache.dtype == torch.int8
3. scale_cache shape 正确
4. write_kv_cache 后 scale_cache 被写入
5. int8_mock + torch attention 能跑完小生成
6. int8_mock vs bf16 baseline 记录第一处分叉位置
```

第一版不要把 int8_mock 和 flash-attn 性能比较放在一起。int8_mock 是语义验证路径，不是高性能路径。

### 9.10 PD handoff 后续扩展

单机 int8_mock 跑通后，再扩展 PD：

```text
HandoffPayload 增加 quant_mode / scale metadata
KVConnector.save_kv() 导出 int8 kv_blocks + scale_blocks
KVConnector.load_kv() 恢复 int8 kv_blocks + scale_blocks
shared_memory backend 只负责搬 tensor，不负责量化数学
```

这和 vLLM / Mooncake 的边界更一致：

```text
Attention / ModelRunner: 负责量化、反量化、scale cache
KVConnector / Store: 负责搬运 KV block 和 scale metadata
```
