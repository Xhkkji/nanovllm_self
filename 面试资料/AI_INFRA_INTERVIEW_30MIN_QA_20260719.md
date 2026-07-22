# AI Infra 推理优化面试速记 + 自问自答

配套详细版：

- [AI_INFRA_INFERENCE_INTERVIEW_NOTES_20260719.md](/home/xhk/nanovllm_self/面试资料/AI_INFRA_INFERENCE_INTERVIEW_NOTES_20260719.md)

---

## 一、30 分钟速记版

### 1. 先背这条主线

LLM 推理优化可以按这条链路理解：

1. `prefill / decode` 分阶段
2. `KV cache` 解决重复计算
3. `PagedAttention` 解决 cache 管理和碎片
4. `continuous batching` 提高 GPU 利用率
5. `chunked prefill` 平衡长短请求
6. `FlashAttention / CUDA Graph` 降低 kernel 和 launch 开销
7. `PD disaggregation` 进一步拆资源域
8. `benchmark / profile / correctness` 证明优化有效

### 2. 核心概念

- Prefill：一次性吃进 prompt，算力密集
- Decode：逐 token 生成，访存密集
- KV cache：用显存换算力，避免重复算历史 token
- PagedAttention：block 化管理 KV，降低碎片和搬运成本
- Chunked prefill：长 prompt 拆块，避免堵住 decode
- CUDA Graph：把稳定 steady-state 路径的 launch 开销压掉
- PD 分离：prefill 和 decode 分开部署/调度

### 3. 高频指标

- TTFT：首 token 延迟
- ITL：token 间延迟
- Throughput：整体吞吐
- p50 / p99：尾部延迟
- KV cache 命中率
- Prefix cache 命中率

### 4. 面试时先说什么

- 先说瓶颈在哪
- 再说为什么这么设计
- 再说你怎么验证
- 最后说收益和边界条件

### 5. 你这个项目的最佳叙事

1. 先搭推理骨架
2. 做 prefill / decode
3. 上 KV cache / paged attention
4. 做 profiling 和 correctness 对拍
5. 做 chunked prefill / CUDA Graph
6. 推 PD 分离和 shared memory handoff

---

## 二、面试自问自答版

### 1. 为什么推理要分 prefill 和 decode？

prefill 计算量大、并行度高；decode 每步只出一个 token，主要在读历史 KV cache。两者瓶颈不同，所以优化手段也不同。

### 2. KV cache 解决了什么问题？

它避免每次 decode 都重算历史 token 的 K/V，是推理提速的核心手段之一，但代价是显存占用随上下文长度增长。

### 3. PagedAttention 为什么重要？

它把 KV cache 按 block/page 管理，减少碎片，方便动态调度，也更贴近真实 serving runtime。

### 4. 为什么 decode 往往是 memory-bound？

因为每步计算量小，但要反复访问历史 KV cache，瓶颈更多在显存带宽和访存模式。

### 5. FlashAttention 快在哪里？

它减少中间张量和 HBM 往返，把 attention 的多个步骤在 kernel 内融合，降低 memory traffic。

### 6. continuous batching 和静态 batch 的区别是什么？

静态 batch 要等一批结束再换下一批；continuous batching 可以动态插入新请求，GPU 利用率更高。

### 7. chunked prefill 的目的是什么？

把长 prompt 拆成多个 chunk，避免长 prefill 把短请求和 decode 卡住，改善 TTFT 和尾延迟。

### 8. CUDA Graph 为什么适合 decode？

decode 的 shape 和控制流相对稳定，适合 capture；这样可以减少 kernel launch 和 CPU 调度开销。

### 9. CUDA Graph 的限制是什么？

capture 期间不能做太多 host 同步操作，shape 也不能频繁变化，所以更适合固定 bucket。

### 10. PD 分离解决什么问题？

它把 prefill 和 decode 分到不同资源域，便于独立扩缩和优化资源利用率。

### 11. PD handoff 边界要传什么？

至少要有请求元数据、KV 状态引用或快照、block table、context length、采样状态。

### 12. 为什么 shared memory 比磁盘更适合本地 handoff？

因为它能保留低拷贝传递的优势，避免落盘的额外延迟和 IO 管理复杂度。

### 13. 怎么估算 KV cache 显存？

大致按 `2 * layers * batch * seq_len * kv_heads * head_dim * bytes` 估算，再考虑 block 对齐和碎片。

### 14. 怎么证明优化没把正确性搞坏？

必须做对拍：和 monolithic / HF / 官方实现比输出 token、边界 case、长短混合请求。

### 15. 面试官问“你最值得说的优化”怎么答？

答你做过的主线：KV cache、chunked prefill、benchmark、CUDA Graph、PD handoff。不要只报功能名，要说瓶颈和收益。

---

## 三、临场回答模板

### 模板 1：讲一个优化

“这个优化的目标是 ___。原始瓶颈在 ___，所以我把它改成 ___。改完以后我用 ___ 指标验证，结果是 ___。边界条件是 ___。”

### 模板 2：讲一个系统设计

“我会先把路径拆成 ___ / ___ / ___，再定义 handoff 边界，最后用 benchmark 和 correctness 去闭环验证。”

### 模板 3：讲项目

“我的项目不是只做一个模型 forward，而是做了一个小型 inference runtime：调度、cache、profiling、graph、PD handoff 都有覆盖。”

---

## 四、最后 10 分钟看什么

1. `prefill / decode` 区别
2. `KV cache` 和 `PagedAttention`
3. `chunked prefill`
4. `CUDA Graph`
5. `TTFT / ITL / throughput`
6. `PD disaggregation`
7. `shared memory KV handoff`

