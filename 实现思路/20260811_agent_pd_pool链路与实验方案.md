# Agent-aware PD Pool 链路与实验方案

## 当前目标

当前这条线不是单纯做一个 Agent Demo，而是验证：

Agent 请求的复杂度、会话 affinity、工具调用步骤等上层信息，能否反过来影响推理引擎层的物理调度，从而减少某些 decode worker 被长任务拖住、其他 worker 空闲的问题。

目前项目里已经有三层：

1. Agent trace 数据：模拟多步 Agent 请求。
2. Agent-aware coordinator：根据复杂度和 worker 反馈选择 prefill / decode worker。
3. nano-vLLM PD pool runtime：执行 prefill、NCCL KV handoff、decode。

## 数据输入怎么组织

默认数据集：

```text
data/serving_benchmarks/agent_trace_qwen3_tokenized.jsonl
```

每一行是一条可独立提交给引擎的请求，但它们保留了 Agent 任务关系，例如：

- `session_id`：同一个 Agent 会话。
- `step_id`：Agent 多步任务中的第几步。
- `task_type`：例如 plan、tool_reason、final 等。
- `input_tokens` / `target_output_tokens`：用于估算 prefill / decode 压力。
- `complexity_score`：调度器用于判断请求复杂度的轻量分数。

例子：

```text
agent-000000-step-0: session=agent-000000, step=0, task=plan
agent-000000-step-1: session=agent-000000, step=1, task=tool_reason
agent-000000-step-2: session=agent-000000, step=2, task=final
agent-000001-step-0: session=agent-000001, step=0, task=plan
```

`affinity_load_aware` 会尽量让同一个 `session_id` 的后续步骤继续走同一个 decode worker。这样做的动机是：真实 Agent serving 里，同一个会话常常有连续上下文和相似 decode 行为，decode 侧保留 locality 更容易扩展到 prefix cache / KV reuse / session cache。

## 4 卡 PD Pool 拓扑

当前 2P2D 默认拓扑：

```text
PREFILL_GPUS=0,2
DECODE_GPUS=1,3

global rank 0: P0, physical GPU0, local cuda:0
global rank 1: P1, physical GPU2, local cuda:1
global rank 2: D0, physical GPU1, local cuda:2
global rank 3: D1, physical GPU3, local cuda:3
```

注意：pool 模式下所有 worker 使用同一个：

```text
CUDA_VISIBLE_DEVICES=0,2,1,3
```

然后每个 worker 通过 `--local-cuda-device` 选择自己使用的本地 GPU。这样可以避免 PyTorch/NCCL 在 `destroy_process_group()` 收尾阶段把 `global_rank` 误当成本地 GPU ordinal。

## 单条请求的完整链路

1. benchmark driver 从 agent trace 里读取一条请求。
2. Agent-aware scheduler 估算请求复杂度：
   - prefill 压力主要看输入 token。
   - decode 压力主要看输出 token 上限。
   - affinity 策略会额外看 `session_id`。
3. scheduler 选择一个 prefill worker 和一个 decode worker。
4. driver 写 `request.json` 到目标 prefill worker 的 work dir。
5. prefill worker 执行 prefill，生成 KV。
6. prefill worker 把 payload metadata 写到目标 decode worker 的 work dir。
7. decode worker 看到 `payload_ready`，读取 metadata，写 `recv_ready`。
8. prefill worker 看到 `recv_ready` 后提交 NCCL `isend`。
9. decode worker 提交 NCCL `irecv`，接收 KV，并 restore 到本地 KV cache。
10. decode worker 执行 continuous decode，写 `decode_metrics.json` 和 `decode_done`。
11. benchmark driver 收集 metrics，最后写 summary 和策略对比。

文件握手顺序：

```text
request.json
  -> payload.pkl
  -> payload_ready
  -> recv_ready
  -> NCCL isend / irecv
  -> prefill_done
  -> decode_done
```

## 三种策略

### round_robin

基线策略。请求按序号轮流分配 worker，不看真实负载，也不看 Agent session。

优点是简单，缺点是复杂任务和短任务混在一起时，容易继续往已经拥塞的 worker 塞请求。

### load_aware

根据 driver 侧虚拟 backlog 和 worker_state 反馈选择负载更低的 worker。

当前反馈字段包括：

- prefill：`request_queue_depth`、`pending_sends`、`busy`
- decode：`active_decode_requests`、`pending_recvs`、`busy`

它体现的是推理引擎层的实时物理状态。

### affinity_load_aware

在 load-aware 基础上加入 Agent session affinity。

如果同一个 `session_id` 之前已经选过某个 decode worker，后续 step 会优先回到同一个 decode worker；如果没有历史记录，再按负载选择。这个策略用于模拟 Agent 多步任务在 serving 层常见的局部性。

## 如何运行 4 卡三策略矩阵

脚本：

```text
pd_self/multiprocess/scripts/run_agent_pd_pool_three_strategy_matrix.sh
```

推荐 smoke 命令：

```bash
cd /home/xhk/nanovllm_self

LIMIT=8 \
MAX_OUTPUT_TOKENS_CAP=8 \
PREFILL_GPUS=0,2 \
DECODE_GPUS=1,3 \
CONCURRENCY=2 \
MAX_ACTIVE_DECODE_REQUESTS=1 \
MAX_PENDING_SENDS=1 \
MAX_PENDING_RECVS=1 \
RESULT_TAG=agent_pd_pool_4gpu_smoke \
bash pd_self/multiprocess/scripts/run_agent_pd_pool_three_strategy_matrix.sh
```

更接近性能对比的命令：

```bash
cd /home/xhk/nanovllm_self

LIMIT=32 \
WARMUP=4 \
MAX_OUTPUT_TOKENS_CAP=32 \
PREFILL_GPUS=0,2 \
DECODE_GPUS=1,3 \
CONCURRENCY=4 \
MAX_ACTIVE_DECODE_REQUESTS=2 \
MAX_PENDING_SENDS=2 \
MAX_PENDING_RECVS=2 \
RESULT_TAG=agent_pd_pool_4gpu_perf \
bash pd_self/multiprocess/scripts/run_agent_pd_pool_three_strategy_matrix.sh
```

## 结果文件

每个策略会输出：

```text
pd_self/multiprocess/result/agent_pd_pool_matrix/<RESULT_TAG>/<PROFILE>/<strategy>/
  synthetic_metrics.jsonl
  synthetic_summary.json
  resource_cleanup.json
  work/
```

三策略对比输出：

```text
pd_self/multiprocess/result/agent_pd_pool_matrix/<RESULT_TAG>/<PROFILE>/
  strategy_compare_summary.json
  strategy_compare_summary.md
```

重点看：

- `pipeline_throughput_generated_tok_s`
- `wall_e2e_time_s.avg`
- `wall_e2e_time_s.p90`
- `affinity_hit_rate`
- `cross_route_rate`
- `prefill_worker_counts`
- `decode_worker_counts`
- `route_counts`

## 当前结论

当前 PD pool 已经不是固定 pair。已经验证过如下交叉路由：

```text
P1 -> D0
```

这说明外层 Agent-aware scheduler 可以独立选择 prefill pool 和 decode pool，后续可以继续扩展：

1. 更真实的负载矩阵。
2. 更长的 Agent trace。
3. KV block pool / session cache。
4. 与 TP/PP 的进一步组合。

## 2026-08-11 四卡三策略 smoke 结果

本轮命令参数：

```text
LIMIT=8
WARMUP=0
MAX_OUTPUT_TOKENS_CAP=8
PREFILL_GPUS=0,2
DECODE_GPUS=1,3
CONCURRENCY=2
MAX_ACTIVE_DECODE_REQUESTS=1
MAX_PENDING_SENDS=1
MAX_PENDING_RECVS=1
RESULT_TAG=agent_pd_pool_4gpu_smoke_20260811
```

结果目录：

```text
pd_self/multiprocess/result/agent_pd_pool_matrix/agent_pd_pool_4gpu_smoke_20260811/agent_multi_step/
```

三策略对比：

| strategy | reqs | tok/s | wall avg(s) | wall p90(s) | affinity | cross-route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| round_robin | 8 | 10.0355 | 1.5746 | 4.2615 | 0.00% | 0.00% |
| load_aware | 8 | 9.2358 | 1.6564 | 3.3542 | 0.00% | 50.00% |
| affinity_load_aware | 8 | 7.5880 | 2.0702 | 3.9167 | 62.50% | 37.50% |

这轮 smoke 的重点不是证明最终性能优势，而是验证链路完整性：

- 三个策略都完成了 8 条请求。
- 三个策略的 `resource_cleanup.json` 都通过。
- worker 日志没有 `Traceback`、`invalid device ordinal`、`destroy_process_group` 异常。
- `load_aware` 已经打出了 50% 的跨 P/D 路由，说明 pool 不是固定 pair。
- `affinity_load_aware` 的 affinity hit rate 是 62.5%，说明同一个 Agent session 的后续 step 会被调度器识别并倾向保持 decode locality。

当前 smoke 里 `load_aware / affinity_load_aware` 吞吐没有超过 `round_robin`，原因主要是：

1. 请求数太少，启动和首 token 抖动占比很高。
2. `max_output_tokens_cap=8`，decode 压力太短，调度收益不容易体现。
3. GPU0/1 同时有其他进程，性能抖动较大。

下一步要跑更有说服力的性能矩阵：

```text
LIMIT=32 或 64
WARMUP=4
MAX_OUTPUT_TOKENS_CAP=32 或 64
CONCURRENCY=4
MAX_ACTIVE_DECODE_REQUESTS=2
MAX_PENDING_SENDS=2
MAX_PENDING_RECVS=2
```

这组更适合观察复杂 Agent 请求下，load-aware 是否能降低 p90，并避免 decode worker 被长任务拖住。

## 2026-08-11 正式矩阵尝试与新卡点

本轮尝试参数：

```text
LIMIT=32
WARMUP=4
MAX_OUTPUT_TOKENS_CAP=32
PREFILL_GPUS=0,2
DECODE_GPUS=1,3
CONCURRENCY=4
MAX_ACTIVE_DECODE_REQUESTS=2
MAX_PENDING_SENDS=2
MAX_PENDING_RECVS=2
RESULT_TAG=agent_pd_pool_4gpu_perf_20260811
```

`round_robin` 已完整跑完：

```text
measured_requests=32
generated_tokens=992
pipeline_measure_wall_time_s=16.5628
pipeline_throughput_generated_tok_s=59.8932
wall_e2e_avg_s=2.0203
wall_e2e_p90_s=3.0991
```

结果文件：

```text
pd_self/multiprocess/result/agent_pd_pool_matrix/agent_pd_pool_4gpu_perf_20260811/agent_multi_step/round_robin/
```

`load_aware` 没有继续等待到超时，原因是它暴露了新的 runtime 稳定性问题：

```text
payload_ready=36
decode_done=34
missing:
  0008_agent-000003-step-0
  0010_agent-000003-step-2
```

当时 worker_state 显示：

```text
decode_0: active_decode_requests=1, pending_recvs=0, processed_bases=18
decode_1: active_decode_requests=1, pending_recvs=0, processed_bases=18
prefill_0: pending_sends=0, pending_handoffs=0
prefill_1: pending_sends=0, pending_handoffs=2
```

这说明问题不在 prefill 生产或 NCCL recv，而在 continuous decode 侧：

1. 两条请求已经进入 active decode。
2. 没有 pending recv。
3. GPU 利用率降到 0。
4. 但是 active request 没有写 `decode_done`。

因此下一步不继续压测矩阵，而是进入新的功能/稳定性修复：

```text
continuous decode active request watchdog + 终止条件诊断
```

最小化目标：

1. 在 `decode_worker_state.json` 里写出 active request 的 base/request_id/step 数/max_tokens。
2. 在 continuous decode loop 里增加 per-request step 计数和 elapsed time。
3. 如果某条 active request 超过 max_tokens 或超过 watchdog 时间仍未完成，写 `decode_error`，让 driver 明确失败，而不是整个矩阵一直挂住。
4. 检查 `DecodeEngine.step()` 返回 finished state 的条件，确认 max_tokens 终止是否在多 active request 下可靠触发。
