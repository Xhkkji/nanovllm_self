# Agent-aware 调度面试整理

这份笔记用于解释当前 `nanovllm_self` 里 Agent-aware PD Pool 调度这条线：

- 这个实验为什么和真实 Agent Serving 场景有关
- 当前调度器到底做了什么
- 为什么能相对 round-robin 降低尾延迟
- 面试时哪些能说，哪些要明确边界

---

## 1. 真实 Agent Serving 的负载特点

Agent 和普通单轮问答不一样。普通问答通常是：

```text
user prompt -> LLM generate -> done
```

Agent 更像一个多步工作流：

```text
step 0: 规划任务，生成 plan
step 1: 根据 plan 做 tool reasoning
step 2: 读取工具结果，继续 refine / reason
step 3: 生成 final answer
```

因此 Agent Serving 对推理引擎的压力有几个特点：

1. **同一个 session 会产生多次 LLM 调用**

   一次用户任务可能拆成多个 step，每个 step 都是一次独立的模型请求。

2. **不同 step 的复杂度差异很大**

   例如 plan 可能输出较短，final answer 可能输出很长；代码分析、长文档总结、工具调用后的综合回答也可能带来很长的 prefill 或 decode。

3. **上下文会跨轮次增长**

   后续 step 通常会带上前面的 user task、plan、tool result、intermediate reasoning，因此 prompt 越来越长。

4. **真实系统会希望利用同 session prefix cache**

   同一个 session 的后续请求往往包含前一轮的大部分上下文。理想情况下，不应该每一轮都重新 prefill 完整历史，而是复用已经算过的 KV / prefix blocks，只计算新增 token。

当前这组 Agent-aware 调度实验主要验证的是第 1 点和第 2 点：**Agent 多步任务中的请求复杂度差异会造成 PD Pool 负载不均，负载感知调度可以降低尾延迟。**

它还没有把收益解释为 prefix cache 命中带来的收益。面试时要明确说：

```text
当前阶段验证的是 Agent-aware load balancing；
后续同 session prefix cache 是进一步让 session affinity 服务于 KV locality 的优化。
```

---

## 2. 为什么先不用急着做跨 session prefix cache

Agent 场景最先该关注的是 **同 session 跨轮次复用**，而不是跨 session 复用。

同 session 复用的模式很稳定：

```text
session A step 0:
  system + user task

session A step 1:
  system + user task + plan + tool result

session A step 2:
  system + user task + plan + tool result + reasoning
```

后一个 step 天然包含前一个 step 的大部分 prefix，所以只要用 `session_id` 就能找到复用关系。

跨 session 复用指的是不同用户 / 不同任务之间复用公共前缀，比如：

```text
system prompt
tool schema
agent instruction
固定模板
```

它当然有价值，但优先级低一些，原因是：

1. **命中率更不稳定**

   不同 session 的用户问题、历史消息、工具结果通常不同。除了 system prompt 和 tool schema，长前缀完全一致的概率不一定高。

2. **匹配逻辑更复杂**

   同 session 可以直接用 `session_id` 管理；跨 session 需要 token-level hash、prefix tree / radix tree、block hash 和严格的前缀一致性判断。

3. **缓存生命周期更难控制**

   同 session cache 可以随着会话结束释放；跨 session cache 需要全局 LRU / LFU、引用计数、显存压力控制。

4. **隔离和正确性要求更高**

   不同用户之间不能误复用私有上下文。跨 session cache 必须保证 token prefix 完全一致，不能只按模板名字或文本片段粗略判断。

所以合理路线是：

```text
阶段 1: Agent-aware load balancing
阶段 2: 同 session prefix cache
阶段 3: session affinity + prefix cache 联动
阶段 4: 跨 session 公共前缀缓存
```

---

## 3. 当前 Agent-aware 调度方法

当前调度是在 PD Pool 的 coordinator / driver 层做的，不修改 nano-vLLM 内部的原生 scheduler。

核心思想是：

```text
根据请求复杂度估计 + worker runtime feedback + session affinity，
给每条 Agent step 选择合适的 prefill worker 和 decode worker。
```

### 3.1 三种策略

当前对比了三种策略。

#### round_robin

固定轮询：

```text
request 0 -> P0, D0
request 1 -> P1, D1
request 2 -> P0, D0
request 3 -> P1, D1
```

它不看请求长度、不看 worker 是否忙、不看 session。优点是简单，缺点是在长短请求混合时很容易造成负载倾斜。

#### load_aware

负载感知调度：

```text
score(worker) = 虚拟排队时间 + worker feedback + 本次请求估计服务时间
选择 score 最小的 worker
```

它主要看：

- `input_tokens`：估算 prefill 代价
- `max_tokens`：估算 decode 代价
- worker 状态：队列深度、active decode 数、pending send / recv、busy 状态

这样可以避免把长 decode 请求继续塞给已经很忙的 D worker。

#### affinity_load_aware

在 load-aware 基础上加入 session affinity：

```text
如果同一个 session 之前路由到某个 decode worker，
后续请求优先回到这个 worker；
但如果该 worker 比最空闲 worker 多等太久，
就允许迁移到更空闲的 worker。
```

也就是说它不是死绑 session，而是在两个目标之间折中：

```text
负载均衡
+
会话局部性
```

当前 `affinity_max_extra_wait_s` 控制这个折中：

```text
preferred worker 额外等待 <= 阈值: 保持 affinity
preferred worker 额外等待 >  阈值: 迁移到更空闲 worker
```

### 3.2 当前为什么强调 session affinity

即使当前实验还没有依赖 prefix cache，session affinity 也有意义：

1. 它让同一个 Agent session 的连续 step 尽量回到同一组 worker，符合真实服务里的会话局部性设计。
2. 它为后续 prefix cache / KV locality 留出调度接口。
3. 它不是盲目粘住 worker，而是带有负载逃逸机制，避免单个 session 拖垮某个 decode worker。

后续补同 session prefix cache 后，session affinity 的解释会更完整：

```text
同 session 回到原 worker，不只是为了少迁移，
而是为了更容易命中该 worker 上保存的 prefix / KV blocks。
```

---

## 4. 实验设置

### 4.1 硬件和拓扑

真实 4 卡 A40，采用 2P2D PD Pool：

```text
Prefill workers:
  P0 -> GPU0
  P1 -> GPU2

Decode workers:
  D0 -> GPU1
  D1 -> GPU3
```

传输后端：

```text
KV_TRANSFER_BACKEND=sync_gpu
```

负载模式：

```text
LOAD_MODE=closed_loop
CONCURRENCY=4
MAX_OUTPUT_TOKENS_CAP=256
```

### 4.2 数据集构造

实验使用派生数据集：

```text
data/serving_benchmarks/agent_trace_qwen3_heavy2sessx8.jsonl
```

这个数据集基于已有 `agent_trace_qwen3_tokenized.jsonl` 构造，没有凭空写 prompt。它选了两个较重的 4-step Agent session：

```text
agent-000033
agent-000165
```

每个 session 重复 8 轮，总共 64 条请求。每轮 step 的输出上限大致是：

```text
plan          max_tokens = 51
tool_reason   max_tokens = 102
refine_reason max_tokens = 128
final_answer  max_tokens = 256
```

这样构造的目的，是模拟 Agent 场景里常见的 decode-heavy 多步任务：

- 前几个 step 输出中等
- final / refine step 输出很长
- 同 session 内 step 连续出现
- 不同步骤之间 decode 负载差异明显

这个场景比随机短请求更贴近 Agent Serving，因为 Agent 真实负载里确实会出现：

```text
短 plan + 中等 tool reasoning + 长 final answer
```

### 4.3 复现实验命令

```bash
cd /home/xhk/nanovllm_self

DATASET=/home/xhk/nanovllm_self/data/serving_benchmarks/agent_trace_qwen3_heavy2sessx8.jsonl \
PROFILE=agent_multi_step \
LIMIT=64 \
WARMUP=0 \
MAX_OUTPUT_TOKENS_CAP=256 \
PREFILL_GPUS=0,2 \
DECODE_GPUS=1,3 \
LOAD_MODE=closed_loop \
CONCURRENCY=4 \
KV_TRANSFER_BACKEND=sync_gpu \
RESULT_TAG=agent_heavy2sessx8_4gpu_cap256 \
REQUEST_TIMEOUT_S=900 \
STARTUP_TIMEOUT_S=240 \
WORKER_FEEDBACK=true \
WORKER_FEEDBACK_SCALE_S=1.0 \
PREFILL_INITIAL_BACKLOG_S=0,0 \
DECODE_INITIAL_BACKLOG_S=0,0 \
AFFINITY_MAX_EXTRA_WAIT_S=0.5 \
bash pd_self/multiprocess/scripts/run_agent_pd_pool_three_strategy_matrix.sh
```

结果路径：

```text
pd_self/multiprocess/result/agent_pd_pool_matrix/agent_heavy2sessx8_4gpu_cap256/agent_multi_step/strategy_compare_summary.json
```

---

## 5. 实验结果

三策略对比：

| strategy | tok/s | wall avg | wall p90 | 相对 round-robin |
| --- | ---: | ---: | ---: | ---: |
| round_robin | 34.39 | 14.81s | 26.44s | baseline |
| load_aware | 44.10 | 11.49s | 17.97s | p90 降低 32.0% |
| affinity_load_aware | 44.25 | 11.52s | 16.76s | p90 降低 36.6% |

关键观察：

```text
round_robin decode busy:
  D0 = 205.2s
  D1 = 409.8s

affinity_load_aware decode busy:
  D0 = 302.0s
  D1 = 313.0s
```

round-robin 虽然请求数量平均：

```text
D0 = 32 requests
D1 = 32 requests
```

但 decode 工作量不平均。一个 D worker 被更多长输出请求拖住，导致尾延迟升高。

`affinity_load_aware` 后，请求数和 decode 工作量都更均衡：

```text
D0 = 33 requests
D1 = 31 requests
```

同时 decode busy time 基本拉平，所以 p90 明显下降。

---

## 6. 为什么能有性能优势

核心原因不是“请求数量均衡”，而是“工作量均衡”。

round-robin 只能做到数量平均：

```text
每个 worker 分到差不多数量的请求
```

但 Agent 请求的服务时间差异很大：

```text
短 plan step       -> decode 很短
长 final answer    -> decode 很长
长 context request -> prefill 很重
tool reasoning     -> 可能输入/输出都偏重
```

所以 round-robin 可能出现：

```text
两个 worker 请求数一样，
但一个 worker 分到了更多长 decode 请求，
最终尾延迟被这个 worker 拖高。
```

load-aware / affinity-load-aware 改进点是：

1. 用 `input_tokens` 和 `max_tokens` 估计请求复杂度。
2. 从 worker_state 读取真实 runtime feedback。
3. 对 P worker 和 D worker 分别做路由。
4. 对同 session 请求保留 affinity，但 worker 过载时允许迁移。

因此它能把长 decode 请求更均匀地摊到不同 D worker 上，降低 decode 队列堆积，最终降低 p90。

可以用一句话总结：

```text
Agent-aware 调度解决的是 Agent 多步任务中请求复杂度不均导致的 PD Pool 负载倾斜问题。
```

---

## 7. 面试回答提纲

### Q1: 你的 Agent-aware 调度到底 aware 了什么？

回答：

它主要 aware 三类信息：

1. 请求复杂度：`input_tokens`、`max_tokens`、task type、step 信息。
2. Worker runtime feedback：队列深度、active decode、pending send / recv、busy 状态。
3. Session affinity：同一个 Agent session 的后续 step 尽量回到原 worker，但过载时允许迁移。

它不是只按 session 粘住 worker，而是在负载均衡和会话局部性之间做折中。

### Q2: 为什么 round-robin 不够？

回答：

round-robin 只能保证请求数量平均，不能保证工作量平均。Agent 场景里不同 step 的输出长度、输入长度差异很大，长 final answer 会显著占用 decode worker。结果可能是两个 worker 都分到 32 个请求，但一个 worker decode busy time 是另一个的两倍。

### Q3: 这次性能收益来自 prefix cache 吗？

回答：

不是。当前实验收益主要来自 load-aware routing 把 decode-heavy 请求摊平，降低 D worker 队列堆积。

更准确地说：

```text
当前阶段验证 Agent-aware load balancing；
后续同 session prefix cache 会让 session affinity 进一步服务于 KV locality。
```

### Q4: 既然真实 Agent 会有 prefix cache，现在实验还成立吗？

回答：

成立，但解释边界要清楚。

真实 Agent 里确实应该有同 session prefix cache，减少重复 prefill。当前实验先验证另一个独立问题：Agent step 复杂度差异会导致 PD Pool 负载倾斜，调度器能否缓解这个问题。

如果后续补上 prefix cache，收益构成会变成：

```text
load-aware balancing
+
session affinity 带来的 prefix / KV locality
```

而不是只靠负载均衡。

### Q5: 为什么先做同 session prefix cache，而不是跨 session prefix cache？

回答：

同 session 复用是 Agent 多轮任务的基本形态。后续 step 通常包含前一轮的大部分上下文，命中关系稳定，生命周期也清楚。

跨 session 复用更多是系统 prompt、tool schema、固定模板级别的优化，命中率不稳定，且需要 token-level hash、prefix tree、全局淘汰策略和严格隔离，复杂度更高。

所以优先级是：

```text
同 session prefix cache > session affinity + cache locality > 跨 session prefix cache
```

### Q6: affinity_load_aware 和 load_aware 的区别是什么？

回答：

load-aware 只看当前哪个 worker 更空闲；affinity-load-aware 还会记录 `session_id -> decode_worker_id`。同 session 的后续请求优先回到原 decode worker，但如果额外等待超过阈值，就迁移到更空闲 worker。

这个设计避免两个极端：

```text
完全不看 session -> cache locality 差
死绑 session -> 容易拖垮单个 worker
```

### Q7: 这个实验最关键的证据是什么？

回答：

不是只看 p90 数字，而是看 decode worker 负载是否被摊平。

实验中：

```text
round_robin:
  D0 busy = 205.2s
  D1 busy = 409.8s

affinity_load_aware:
  D0 busy = 302.0s
  D1 busy = 313.0s
```

这说明调度器确实缓解了 decode worker 负载倾斜，所以 p90 从 26.44s 降到 16.76s 是解释得通的。

---

## 8. 可以写进简历的表述

保守版本：

```text
针对 Agent 多步任务中请求复杂度差异导致的 PD Pool 负载倾斜问题，设计 Agent-aware 调度器，结合请求复杂度估计、Worker Runtime Feedback 与 Session Affinity，实现 Load-aware / Affinity-aware 的 Prefill/Decode Pool 动态路由；在 4 卡 A40 2P2D decode-heavy Agent Trace stress 实验中，相比 Round-robin 将端到端 p90 延迟降低 20%+。
```

更强版本：

```text
针对 Agent 多步任务中长短 step 混部导致的 Decode Worker 负载倾斜问题，设计 Agent-aware PD Pool 调度器，结合请求复杂度估计、Worker Runtime Feedback 与 Session Affinity，在负载均衡和会话局部性之间动态权衡；在 4 卡 A40 2P2D decode-heavy Agent Trace stress 实验中，相比 Round-robin 将端到端 p90 延迟降低约 36%，吞吐提升约 29%。
```

建议面试时优先用保守版本，追问细节时再拿出完整实验数据。

---

## 9. 当前边界和下一步

当前已经可以说：

- 实现了 4 卡 2P2D PD Pool 调度实验链路。
- 实现了 round-robin / load-aware / affinity-load-aware 三策略对比。
- 接入了 worker runtime feedback。
- 构造了更贴近 Agent decode-heavy 多步任务的 stress trace。
- 真实跑出了 p90 和吞吐收益。

当前不能夸大说：

- 不能说收益来自 prefix cache。
- 不能说已经完整实现真实 Agent 多轮 KV 复用。
- 不能说跨 session prefix cache 已经完成。

下一步最自然的是：

```text
补同 session prefix cache，
让同一个 Agent session 的后续 step 复用上一轮 prefix KV，
然后重新评估：
  no cache + round_robin
  no cache + affinity_load_aware
  cache + round_robin
  cache + affinity_load_aware
```

这样 Agent-aware 这条线会更完整：

```text
第一阶段：解决复杂度不均导致的负载倾斜
第二阶段：利用 session affinity 提高 prefix / KV locality
```
