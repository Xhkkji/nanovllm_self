# AI Infra 推理优化岗位面试八股

这份笔记是给 `AI infra / LLM inference / 推理优化` 岗位准备的，侧重点是：

- 能把原理讲清楚
- 能把工程实现讲清楚
- 能把性能优化和指标讲清楚
- 能把你当前的 `nanovllm_self` 项目讲成一个完整故事

如果你想看功能路线图和缺口分析，先看：

- [NANOVLLM_SELF_INFRA_GAP_ANALYSIS.md](/home/xhk/nanovllm_self/NANOVLLM_SELF_INFRA_GAP_ANALYSIS.md)

这份文档更偏“面试题库 + 回答提纲”。

---

## 0. 公开资料索引

下面这些是这份笔记的主要公开来源，方便你回头查原文：

- vLLM 文档首页: https://docs.vllm.ai/
- vLLM 优化配置: https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM disaggregated prefill: https://docs.vllm.ai/en/v0.9.2/features/disagg_prefill.html
- FlashAttention 仓库: https://github.com/Dao-AILab/flash-attention
- FlashAttention 论文: https://arxiv.org/abs/2205.14135
- Mooncake 仓库: https://github.com/kvcache-ai/Mooncake
- Mooncake 论文: https://arxiv.org/abs/2407.00079
- AI Infra interview notes: https://github.com/ZonePG/cs-notes/blob/main/AI-System/04-AI-Infra-Interview.md
- AI Infra 面试汇总: https://www.nowcoder.com/discuss/891334000239734784
- AI Infra 面试收录: https://www.cnblogs.com/xmwblogs/p/19669357
- AI Infra 面试题整理: https://zhuanlan.zhihu.com/p/1894450748672161081
- AI Infra 秋招面经: https://zhuanlan.zhihu.com/p/2017740483217081305

---

## 1. 面试官主要在看什么

AI infra 推理优化岗，一般不是只问模型知识，而是看你能不能把下面几层串起来：

1. 模型计算图本身
2. Attention 和 KV cache 怎么省算力、省显存
3. 调度怎么做，prefill / decode 怎么切
4. CUDA / kernel / graph 怎么降开销
5. 分布式和 PD disaggregation 怎么做边界设计
6. 怎么 benchmark、怎么定位瓶颈、怎么证明优化有效

一句话：不是背概念，而是能解释“为什么慢、慢在哪里、怎么改、改完快多少”。

---

## 2. 高频八股题

### 2.1 Transformer / 推理基础

**Q1: 为什么 LLM 推理要分 prefill 和 decode？**

要点：

- Prefill 负责把 prompt 跑进模型，计算量大，但 token 间并行度高
- Decode 是自回归逐 token 生成，单步计算量小，但要频繁读 KV cache
- 两者瓶颈不同，所以调度、kernel、cache 设计也不同

**Q2: 为什么 decode 往往是 memory-bound？**

要点：

- 每步只生成 1 个 token，算术强度低
- 主要成本是读取历史 KV cache
- 计算量不一定大，但 HBM 带宽和访存模式很关键

**Q3: KV cache 解决什么问题？**

要点：

- 避免每次 decode 都重算历史 token 的 K/V
- 用显存换算力和时延
- 代价是显存占用随上下文长度线性增长

**Q4: KV cache 的显存怎么粗略估算？**

可以按这个思路答：

- `kv_memory ~= 2 * num_layers * batch * seq_len * num_kv_heads * head_dim * bytes_per_elem`
- `2` 是 K 和 V
- `num_kv_heads` 要注意 GQA / MQA，不要直接拿 `num_heads`
- 实际还要乘 block 对齐和碎片损耗

---

### 2.2 Attention / Kernel

**Q5: PagedAttention 解决什么问题？**

要点：

- 把连续 KV cache 改成 block/page 管理
- 避免大块连续显存分配和碎片化
- 让不同请求可以共享调度器和 cache 管理逻辑
- 可对照 vLLM 的 PagedAttention 设计理解: https://docs.vllm.ai/

**Q6: FlashAttention 为什么快？**

要点：

- 核心不是“少算”，而是减少 HBM 和 SRAM 之间的来回搬运
- 通过 tiling / fusion 降低 attention 的 memory traffic
- 对长序列收益明显

**Q7: FlashAttention 和普通 `torch.einsum` / `matmul` 的差别是什么？**

要点：

- 普通实现往往中间张量多、访存多
- FlashAttention 在 kernel 内融合 softmax、scale、mask、dropout 等步骤
- 更适合 GPU 上的真实推理场景

**Q8: 为什么 kernel 优化里经常强调 layout / contiguous / dtype？**

要点：

- layout 影响访存 coalescing
- contiguous 影响 kernel 是否要额外拷贝
- dtype 影响算力利用率和显存带宽
- 推理里常见瓶颈不是单纯 FLOPs，而是 memory traffic 和 launch overhead

---

### 2.3 调度 / batching

**Q9: Continuous batching 是什么？**

要点：

- 不等一批请求完全结束再换下一批
- 新请求可以动态插入运行队列
- 提高 GPU 利用率和吞吐

**Q10: Chunked prefill 解决什么问题？**

要点：

- 长 prompt 一次性 prefill 会占住 GPU，影响 decode 和短请求
- 把长 prefill 拆成多个 chunk，和 decode 混合调度
- 目标是更稳的 TTFT / ITL 平衡
- 可对照 vLLM 的优化文档理解: https://docs.vllm.ai/en/stable/configuration/optimization/

**Q11: 为什么 chunked prefill 常常先优先 decode？**

要点：

- decode 是在线交互里最敏感的路径
- 先保 decode，可以压尾延迟
- prefill 可以分片慢慢补

**Q12: 调度里最容易出错的点是什么？**

要点：

- batch size / token budget 的边界条件
- 长短 prompt 混合时的公平性
- prefill 没完成时的状态迁移
- block_table / context_lens / slot_mapping 一致性

---

### 2.4 CUDA Graph / 性能工程

**Q13: CUDA Graph 适合什么场景？**

要点：

- shape 稳定、重复执行的 steady-state 路径
- 特别适合 decode 这种每步形状固定的循环
- 能减少 kernel launch 和 CPU 侧调度开销

**Q14: CUDA Graph 的限制是什么？**

要点：

- capture 期间不能做很多 host 同步操作
- shape / 控制流不能频繁变化
- 适合固定 bucket，不适合完全动态的路径

**Q15: 为什么 graph 只能先做 decode steady-state？**

要点：

- decode 单步形状稳定，容易 capture
- prefill 往往长度变化大，捕获收益和复杂度都更差
- 所以工业上一般先做 decode graph

**Q16: 怎么判断优化是真的有效？**

要点：

- 不能只看单次运行时间
- 要看 warmup 后 steady-state
- 看 TTFT、ITL、throughput、p50/p99
- 最好有对拍，证明正确性没坏

---

### 2.5 分布式 / PD disaggregation

**Q17: 为什么要做 prefill / decode 分离？**

要点：

- Prefill 和 decode 的资源需求不同
- 分离后可以按阶段放到不同 GPU / 节点上
- 能更好控制尾延迟和资源利用率
- 可对照 vLLM disaggregated prefill: https://docs.vllm.ai/en/v0.9.2/features/disagg_prefill.html

**Q18: PD 分离时，handoff 边界要传什么？**

要点：

- 请求的元数据
- 已缓存的 KV 状态引用或快照
- block table / context length / slot mapping
- 采样相关状态

**Q19: PD 分离里最关键的系统问题是什么？**

要点：

- KV cache 怎么搬
- 搬运开销多大
- 传输边界怎么设计
- decode 端怎么恢复状态

**Q20: 为什么 Mooncake 这类系统会强调 KVCache-centric？**

要点：

- 因为真正昂贵的是 context state
- KV cache 本身就是 serving 的核心资产
- 调度、传输、存储都围绕 KV cache 展开
- 可对照 Mooncake 项目和论文: https://github.com/kvcache-ai/Mooncake

---

### 2.6 量化 / speculative / 结构化输出

**Q21: 量化为什么能加速推理？**

要点：

- 显存占用更小
- 带宽压力更低
- 某些硬件下还能提升吞吐
- 但要注意精度和 kernel 支持

**Q22: speculative decoding 的核心思想是什么？**

要点：

- 用小 draft model 先提案
- 大模型验证和修正
- 目标是减少 target model 的实际解码步数

**Q23: structured output / guided decoding 有什么价值？**

要点：

- 限制输出格式
- 更适合 JSON / tool calling / agent 协议
- 工程上会牵涉到 logits mask、约束状态机、解码策略

---

## 3. 结合你当前项目，面试时怎么讲

你现在最适合讲的故事，不是“我只写了一个 demo”，而是：

1. 我先搭了一个最小 inference runtime
2. 把 prefill / decode 路径拆开
3. 引入 KV cache 和 paged attention 思路
4. 再做 prefix cache、continuous batching、chunked prefill
5. 然后补 profiling 和 correctness 对拍
6. 最后推进 PD disaggregation 和 shared memory KV handoff

这条叙事是对的，因为它贴近真实 serving 系统的演进顺序。

### 3.1 你已经能讲的点

- `prefill / decode` 两阶段执行
- `paged attention` / `KV cache` 管理
- `prefix cache`
- `continuous batching`
- `chunked prefill`
- `CUDA Graph` / steady-state 方向
- `benchmark` 和 profile
- `PD disaggregation`
- `shared memory KV handoff`
- `evaluation` 对拍和 correctness regression

### 3.2 面试里可以直接说的优化逻辑

**如果问“你为什么做这些优化”**：

- 先解决正确性
- 再解决瓶颈
- 再做结构化 benchmark
- 再做阶段拆分和跨进程 handoff

**如果问“你怎么看待一个推理引擎”**：

- 它不是一个模型 forward
- 它是调度 + cache + kernel + runtime + 指标 的组合体

**如果问“最重要的指标是什么”**：

- 交互场景看 TTFT / ITL
- 批量场景看 throughput
- 长上下文看 KV cache 占用和命中

---

## 4. 你可以重点背的回答模板

### 4.1 PagedAttention

“PagedAttention 的核心是把 KV cache 从连续大块内存改成 page/block 管理。这样可以减少碎片，提升多请求场景下的 cache 利用率，也方便调度器做动态批处理和回收。”

### 4.2 Chunked prefill

“Chunked prefill 是把长 prompt 拆成多个 chunk，和 decode 请求一起调度。它的目标不是单纯提高吞吐，而是避免长 prefill 把 decode 卡住，改善 ITL 和尾延迟。”

### 4.3 CUDA Graph

“CUDA Graph 适合稳定的 steady-state decode。因为 decode 每步输入 shape 比较固定，可以把 launch 和 CPU 调度开销压掉。但 graph 对动态 shape 和 host 同步很敏感，所以一般先做固定 bucket，再做 fallback。”

### 4.4 PD disaggregation

“PD 分离本质上是把 prefill 和 decode 的资源域拆开，让它们独立扩缩、独立调度。难点不在拆进程，而在 KV handoff：要保证元数据、KV 状态、block table 和采样状态在边界处一致。”

### 4.5 Shared memory KV handoff

“如果同机部署，shared memory 是比磁盘更合理的中间层。它能保留 KV state 的低拷贝传递能力，同时避免文件落盘的额外延迟和管理复杂度。”

---

## 5. 你当前项目和主流框架的对齐点

### 5.1 和 vLLM 的对齐

vLLM 的公开文档里长期强调这些能力：

- PagedAttention
- continuous batching
- chunked prefill
- prefix caching
- CUDA/HIP graph
- tensor / pipeline parallelism
- disaggregated prefill

你的项目已经开始往这些方向收敛了，尤其是：

- KV cache 管理
- chunked prefill
- benchmark / profile
- PD handoff

### 5.2 和 Mooncake 的对齐

Mooncake 的核心思路是：

- prefill / decode 分离
- KVCache-centric
- 关注 KV transfer 和 SLO

你现在做的 PD、payload、shared memory，方向上就是在往这个框架靠。

---

## 6. 面试高频追问清单

下面这些问题，建议你至少都能说出一个清晰答案：

1. Prefill 和 decode 的瓶颈分别是什么？
2. KV cache 为什么会成为推理核心？
3. 为什么 PagedAttention 能缓解碎片化？
4. Chunked prefill 为什么能改善 tail latency？
5. CUDA Graph 为什么只适合固定形状路径？
6. FlashAttention 为什么比朴素 attention 快？
7. Continuous batching 和 static batching 有什么区别？
8. 怎么估算 KV cache 显存占用？
9. 为什么需要 prefix cache？
10. PD disaggregation 的边界应该传哪些状态？
11. Shared memory 为什么比磁盘更适合本地 handoff？
12. 怎么设计 benchmark 才能说明优化真的有效？
13. 怎么证明优化没把正确性搞坏？
14. 长短 prompt 混合时调度怎么保证公平？
15. decode 中怎么减少 Python / launch overhead？
16. 什么时候该做 TP，什么时候该做 PD？
17. 量化的收益和风险是什么？
18. speculative decoding 的收益来自哪里？
19. 结构化输出为什么会增加系统复杂度？
20. 如果只让你做一个 P0 优化，你会先做什么？

---

## 7. 最后冲刺建议

如果你面试前只剩 1-2 天，优先顺序建议是：

1. KV cache / PagedAttention / chunked prefill
2. decode bottleneck / CUDA Graph / FlashAttention
3. continuous batching / scheduler
4. benchmark / TTFT / ITL / throughput
5. PD disaggregation / KV handoff
6. quantization / speculative decoding / TP

---

## 8. 参考资料

### 官方 / 一手资料

- vLLM 首页与能力概览  
  https://docs.vllm.ai/

- vLLM Chunked Prefill / Optimization  
  https://docs.vllm.ai/en/stable/configuration/optimization/

- vLLM Disaggregated Prefill  
  https://docs.vllm.ai/en/v0.9.2/features/disagg_prefill.html

- FlashAttention README  
  https://github.com/Dao-AILab/flash-attention

- FlashAttention 论文  
  https://arxiv.org/abs/2205.14135

- Mooncake GitHub  
  https://github.com/kvcache-ai/Mooncake

- Mooncake 论文  
  https://arxiv.org/abs/2407.00079

### 面经 / 题库类资料

- AI Infra 面试汇总（牛客）  
  https://www.nowcoder.com/discuss/891334000239734784

- AI Infra 面试收录（博客园）  
  https://www.cnblogs.com/xmwblogs/p/19669357

- AI Infra 26 秋招面经（知乎专栏）  
  https://zhuanlan.zhihu.com/p/2017740483217081305

- 大模型推理优化面试题（知乎专栏）  
  https://zhuanlan.zhihu.com/p/1894450748672161081

- AI Infra interview notes（GitHub）  
  https://github.com/ZonePG/cs-notes/blob/main/AI-System/04-AI-Infra-Interview.md

---

## 9. 一句话结论

AI infra 推理优化岗面试，本质上是在看你能不能把：

- 模型计算
- 内存管理
- 调度系统
- GPU kernel
- 分布式传输
- 指标和 benchmark

串成一条完整链路。

你现在这个项目已经足够支撑这条叙事，后面只需要继续把性能、PD 和多进程边界做扎实。
