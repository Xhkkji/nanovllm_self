# nanovllm_self Infra Gap Analysis

这份文档总结当前 `nanovllm_self` 相对于官方 `nano-vllm` / `vLLM` 以及 AI infra 岗位常见要求，还缺哪些优化与功能。

## 当前已经具备的核心能力

你的 `nanovllm_self` 已经完成了原型级 inference engine 的核心骨架：

- 基本的 `prefill / decode` 两阶段
- `paged attention` 思路
- 全局 `kv cache`
- prefix cache 的基础版
- multi-seq 基础调度
- `sampler`、`temperature`、`ignore_eos`
- 基本 profiling
- 可以作为 agent backend 被调用

这说明它已经不是简单的 demo，而是一个真正的 LLM inference 原型。

---

## 一、最值得优先补的能力

### 1. 高性能 attention 内核替换

当前 `nanovllm_self` 的 attention 路径仍然大量依赖：

- Python 层循环
- `torch.matmul`
- `torch.einsum`
- 手动拼接 KV

而官方 `nano-vllm` 在 `attention.py` 中已经使用：

- `flash_attn_varlen_func`
- `flash_attn_with_kvcache`
- Triton `store_kvcache_kernel`

这部分是当前最大的性能差距之一。

建议优先级：`P0`

建议动作：

1. 先把 `store_kv_cache` 和 `get_kv_cache` 去 Python for-loop / 更矢量化
2. 再尝试接入 FlashAttention
3. 后续再考虑 Triton kernel

简历价值：非常高。

当前进展（简要）：

- 已做 `store_kv_cache` 矢量化
- `store_kv` profile 约从 `0.7104s` 降到 `0.5878s`
- `decode` 总时延约从 `10.6171s` 降到 `10.3511s`
- 总体收益存在，但当前更大的热点仍然是 `get_kv` 和 attention 主计算路径

---

### 2. decode-only CUDA Graph

官方 `nano-vllm` 的 `ModelRunner` 已经包含：

- warmup
- `capture_cudagraph()`
- decode 小 batch replay

这对低延迟 decode 很重要，也是成熟 serving runtime 很典型的能力。

当前 `nanovllm_self` 还没有这条路径。

建议优先级：`P0`

建议动作：

1. 先只做 decode-only CUDA Graph
2. 只支持几个固定 batch size 即可
3. 不需要一开始就做复杂 fallback

简历价值：高。

---

### 3. 系统化 benchmark / metrics

你现在已经有基础 profile 输出，这是很好的开始。

但从 AI infra 视角，更希望看到固定指标：

- TTFT（time to first token）
- ITL（inter-token latency）
- throughput
- prefill token/s
- decode token/s
- prefix cache 命中率
- KV cache 命中/复用情况

建议优先级：`P0`

建议动作：

1. 补 benchmark 脚本
2. 输出结构化结果（CSV / JSON）
3. 固定打印几项核心指标

简历价值：高。

---

### 4. 连续批处理逻辑增强

你已经有 scheduler 和 waiting/running 队列，但还更偏“教学版 continuous batching”。

建议进一步验证和增强：

- prefill 与 decode 混合调度
- 长短请求共存时的公平性
- token budget 分配策略
- 抢占和恢复逻辑

建议优先级：`P0`

建议动作：

1. 增加混合 workload 测试
2. 增加长短 prompt 并发 case
3. 统计不同请求类型的 latency

简历价值：高。

---

## 二、接下来应该补的能力

### 5. chunked prefill

`vLLM` 官方长期强调：

- continuous batching
- chunked prefill
- prefix caching

当前你的 `nanovllm_self` 更像：

- 一整个 prompt 一次性 prefill
- 然后再进入 decode

这在长 prompt 和高并发场景下不够接近真实 serving engine。

建议优先级：`P1`

建议动作：

1. 把大 prompt 拆块 prefill
2. 验证长 prompt 不阻塞短请求
3. 结合 scheduler 做更真实的 token budget 分配

简历价值：高。

---

### 6. 动态 KV cache block 估算

官方 `nano-vllm` 会在 warmup 后，根据显存使用动态估算可分配 KV blocks。

当前你的 `nanovllm_self` 更偏：

- 根据 config 固定分配

建议优先级：`P1`

建议动作：

1. warmup
2. 使用 `torch.cuda.mem_get_info()`
3. 动态计算 `num_kvcache_blocks`

简历价值：中高。

---

### 7. 更系统的 prefix cache 命中验证

你的 prefix cache 基础功能已经有了，但从 infra 角度还需要更系统的验证：

- 相同 system prompt 的多轮调用
- 不同长度公共前缀
- 命中率统计
- 命中后 TTFT 改善程度

建议优先级：`P1`

建议动作：

1. 单独做 prefix benchmark
2. 打印命中率和 latency 对比
3. 构造 agent 风格 prompt 验证 prefix reuse

简历价值：高。

---

### 8. 多次独立调用的状态隔离测试

如果 `nanovllm_self` 要作为稳定后端，必须确认：

- 多次连续调用 `generate()` 互不污染
- scheduler 不残留脏状态
- block_manager / kv cache 不误读旧状态

建议优先级：`P1`

建议动作：

1. 连续独立请求测试
2. 混合 prompt 长度测试
3. 无关 prompt 串行调用验证

简历价值：中高。

---

## 三、中期能力补充

### 9. Tensor Parallel / 多卡

官方 `nano-vllm` 已经有：

- `torch.distributed`
- NCCL
- `tensor_parallel_size`
- rank loop

而你的 `nanovllm_self` 仍然基本是单卡设计。

建议优先级：`P2`

建议动作：

1. 先设计 TP 切分思路
2. 先做 attention / linear 的 TP 原型
3. 再补多进程协同

简历价值：非常高。

---

### 10. Structured outputs / guided decoding

`vLLM` 已经支持 structured outputs。

这对于：

- JSON 输出
- tool calling
- agent 控制协议

都很有帮助。

建议优先级：`P2`

建议动作：

1. 先做最小 JSON / constrained output 原型
2. 再考虑和 agent 协议结合

简历价值：中高。

---

### 11. Quantization 支持

`vLLM` 支持多种量化路径：

- FP8
- INT8
- INT4
- GPTQ
- AWQ

而你的 `nanovllm_self` 目前还没有量化能力。

建议优先级：`P2`

建议动作：

1. 先支持一种最常见量化格式
2. 验证显存与吞吐收益

简历价值：高。

---

### 12. Speculative decoding

`vLLM` 官方支持 speculative decoding。

这属于更高级的低延迟优化项。

建议优先级：`P2`

建议动作：

1. 先理解 draft / target model 交互
2. 再做最小 speculative 原型

简历价值：高，但实现复杂度也高。

---

## 四、从 AI infra 岗位要求倒推，最关键的能力点

结合常见 AI infra / inference engineer 岗位要求，最常见关键词包括：

- CUDA / Triton
- distributed systems
- inference runtime
- GPU memory efficiency
- scheduling
- serving reliability
- observability / metrics
- large-scale inference serving

所以，如果你想让 `nanovllm_self` 更像一份能打动 AI infra 岗位的项目，最有含金量的是：

1. 高性能 attention kernel / Triton / FlashAttention
2. continuous batching + chunked prefill
3. CUDA Graph
4. 多卡 Tensor Parallel
5. 系统化 benchmark 和 metrics
6. KV cache / prefix cache 的工程化管理

---

## 五、推荐优先级路线图

### P0：最值得马上做

1. attention 热点优化
2. decode-only CUDA Graph
3. benchmark / metrics 系统化
4. 连续批处理逻辑增强

### P1：接下来做

5. chunked prefill
6. 动态 KV cache block 估算
7. prefix cache 命中验证
8. 状态隔离测试

### P2：中期能力

9. Tensor Parallel / 多卡
10. Structured outputs / guided decoding
11. Quantization
12. Speculative decoding

---

## 六、一句话总结

你的 `nanovllm_self` 已经有了“原型级 inference engine”的骨架。

下一步如果不考虑 agent，而是只从 AI infra / 官方 `nano-vllm` / `vLLM` 视角看，最值得做的是：

- 性能内核化
- 调度工程化
- 多卡化
- benchmark / metrics 化

也就是说，接下来最值得补的，不是再写更多 demo，而是把它收成一个更像真实 serving runtime 的系统。

---

## 七、TODO Checklist

下面给出一个不删除原分析、可直接执行的 TODO 清单。

### P0 Project List

1. 优化 attention 热点路径
2. 减少 Python 循环和小 tensor 开销
3. 评估并接入高性能 attention 内核
4. 加入 decode-only CUDA Graph
5. 补充 decode CUDA Graph replay 测试
6. 整理 benchmark 脚本
7. 输出关键性能指标
8. 做 continuous batching 混合 workload 测试
9. 验证长短请求混合下的调度公平性和稳定性

### P0 TODO

- [ ] 优化 attention 热点路径
- [ ] 减少 `store_kv_cache` / `get_kv_cache` 中的 Python 循环与小 tensor 开销
- [ ] 评估并接入更高性能 attention 内核（如 FlashAttention / Triton）
- [ ] 为 decode 路径加入 CUDA Graph 原型
- [ ] 补充 decode-only 小 batch CUDA Graph replay 测试
- [ ] 整理 benchmark 脚本
- [ ] 输出 TTFT / ITL / throughput 等关键指标
- [ ] 对 continuous batching 做混合 workload 测试
- [ ] 验证长短请求混合下的调度公平性与稳定性

### P1 TODO

- [ ] 实现 chunked prefill 原型
- [ ] 补动态 KV cache block 估算
- [ ] 增加 prefix cache 命中率统计
- [ ] 增加 prefix cache latency 对比实验
- [ ] 增加多次独立调用的状态隔离测试
- [ ] 验证 scheduler / block_manager / kv cache 在串行请求下无脏状态

### P2 TODO

- [ ] 设计 Tensor Parallel 原型
- [ ] 评估 structured outputs / guided decoding 的最小实现路径
- [ ] 选择并接入一种量化方案
- [ ] 调研 speculative decoding 并规划最小实验版本
