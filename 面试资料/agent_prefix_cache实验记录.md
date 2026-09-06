# Agent-aware P-local Prefix Cache 实验记录

## 1. 本次实现目标

最简版目标：

```text
P-local 同 session prefix cache
+ session 回 P 亲和
+ load-aware 兜底
```

不做：

```text
D -> P KV 回写
跨 P worker KV 迁移
跨 session prefix cache
只传 suffix delta KV
```

这个版本的定位是先把同 session 的 prefix 复用真正接入 PD Pool 调度链路，用最小改动验证 session locality 对 prefill 侧的收益。

## 2. 代码改动点

### 2.1 PrefillEngine 传递 session_id

`pd_self/prefill_engine.py`

- `run_prefill()` 增加 `session_ids`
- `add_request()` 增加 `session_id`
- `_build_sequence()` 创建 `Sequence(..., session_id=session_id)`

这样 prefill worker 里的 `Sequence` 才能被 `BlockManager` 按 session 查 prefix cache。

### 2.2 Prefill handoff 前保存 P-local prefix

`pd_self/prefill_engine.py`

PD handoff 请求不会走普通 FINISHED 分支，而是在 prefill 侧生成 payload 后被摘走。因此在：

```text
_make_payload()
-> 从 scheduler.running 移除
-> block_manager.deallocate(seq)
```

之前增加：

```python
block_manager.save_session_prefix_cache(seq.session_id, seq)
```

这样 deallocate 只释放当前 seq 引用，prefix cache 自己持有的引用会保留，本地 P worker 后续可以复用这些 KV blocks。

### 2.3 BlockManager 增加 prefix 命中观测字段

`nanovllm/engine/block_manager.py`

在 `allocate_with_prefill()` 里，当：

```text
session_prefix_cache[session_id] 存在
且当前 token_ids 以 cached token_ids 为前缀
```

则设置：

```text
prefix_cache_hit = true
prefix_cached_tokens = 命中的历史 token 数
prefix_new_tokens = 本轮需要新 prefill 的 token 数
prefix_cache_source = session_local
```

注意：`payload.num_cached_tokens` 表示本轮结束后 prompt 已经写入 KV 的长度，不等于历史 prefix 命中长度。因此必须单独记录 `prefix_cached_tokens`。

### 2.4 Pool 调度扩展成 P/D 双侧 affinity

`pd_self/multiprocess/evaluation/benchmark_agent_pd_pool.py`

原来主要是：

```text
session_id -> decode_worker
```

现在扩展为：

```text
session_id -> prefill_worker
session_id -> decode_worker
```

调度逻辑：

```text
P 侧：
  同 session 优先回保存 prefix cache 的 prefill worker；
  如果 preferred P 额外等待超过阈值，则迁移到更空闲 P。

D 侧：
  同 session 优先回原 decode worker；
  如果 preferred D 过载，则迁移到更空闲 D。
```

## 3. 实验设置

数据集：

```text
data/serving_benchmarks/agent_prefix_reuse_qwen3_4sessx8_pair_interleave.jsonl
```

数据特征：

```text
4 个 session
每个 session 8 个 step
后续 step 以前一个 step 的完整 prompt 为前缀
input_tokens: 406 到 1715
max_output_tokens_cap: 8
```

硬件：

```text
4 x A40
P0: GPU0 / rank0
D0: GPU1 / rank2
P1: GPU2 / rank1
D1: GPU3 / rank3
```

拓扑：

```text
GPU0 <-> GPU1: NV4
GPU2 <-> GPU3: NV4
GPU0/1 <-> GPU2/3: SYS
```

这说明当前机器没有 NVSwitch 全互联，而是两组 NVLink pair。跨 pair 的 P0->D1、P1->D0 会走 SYS 路径，传输效率和抖动都更差。

## 4. 关键结果

### Run1: closed_loop, concurrency=4, worker feedback enabled

命令核心参数：

```text
LIMIT=32
CONCURRENCY=4
MAX_OUTPUT_TOKENS_CAP=8
AFFINITY_MAX_EXTRA_WAIT_S=0.5
WORKER_FEEDBACK_SCALE_S=1.0
```

结果：

| strategy | tok/s | wall p90 | core p90 | prefix hit |
| --- | ---: | ---: | ---: | ---: |
| round_robin | 30.91 | 2.93s | 1.61s | 75.0% |
| load_aware | 27.99 | 3.67s | 1.59s | 75.0% |
| affinity_load_aware | 30.30 | 3.10s | 0.93s | 75.0% |

结论：

```text
affinity_load_aware 相比 round_robin，core_e2e p90 降低约 42.1%。
affinity_load_aware 相比 load_aware，core_e2e p90 降低约 41.4%。
```

这组结果能说明：在同 session 多轮请求中，P-local prefix cache 接入后，亲和调度能减少核心执行链路的尾延迟。

注意：这组 wall p90 没有改善，主要是 closed-loop 并发下文件控制面等待、worker 排队和部分跨 pair 路由带来的噪声覆盖了 core 链路收益。

### Run3: pair affinity, 避免跨 NVLink pair

命令核心参数：

```text
LIMIT=32
CONCURRENCY=4
MAX_OUTPUT_TOKENS_CAP=8
AFFINITY_MAX_EXTRA_WAIT_S=999
WORKER_FEEDBACK_SCALE_S=0.0
```

结果：

| strategy | tok/s | wall avg | wall p90 | core p90 | prefix hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| round_robin | 32.06 | 0.98s | 2.98s | 1.59s | 75.0% |
| load_aware | 29.99 | 1.03s | 2.94s | 1.61s | 75.0% |
| affinity_load_aware | 32.86 | 0.93s | 2.93s | 1.53s | 87.5% |

结论：

```text
prefix_cache_hit_rate 从 75.0% 提升到 87.5%。
平均命中 prefix tokens 从 682.7 提升到 768.0。
平均新增 prefill tokens 从 506.5 降到 330.0。
吞吐相比 round_robin 提升约 2.5%，wall avg 降低约 4.7%。
```

这组结果说明：在当前 A40 两组 NVLink pair 的拓扑下，调度器需要同时考虑 P-local prefix locality 和 P/D pair locality，否则跨 SYS 路径传输会抵消 prefix 收益。

## 5. 卡间 KV 传输带宽

Run3 统计：

| route | topology | count | avg GiB/s | p50 GiB/s |
| --- | --- | ---: | ---: | ---: |
| GPU0 -> GPU1 | NV4 | 16 | 18.73 | 21.73 |
| GPU2 -> GPU3 | NV4 | 16 | 21.00 | 25.44 |

Run3 没有跨 pair 路由：

```text
P0->D0: 16
P1->D1: 16
cross_route_rate: 0
```

Run2 中出现跨 pair：

```text
P0->D1 / GPU0->GPU3: SYS
P1->D0 / GPU2->GPU1: SYS
```

这类路径不走本地 NVLink pair，而要经过 CPU/PCIe/NUMA 互联，延迟和抖动都更明显。因此当前机器上做 PD Pool 调度时，不能把任意 P/D 组合当作等价链路。

## 6. 面试表述

可以这样说：

```text
我在 PD Pool 上补了最小版同 session P-local prefix cache：
调度器维护 session->prefill_worker 和 session->decode_worker 两个 affinity map。
同一个 Agent session 的后续 step 会优先回到保存 prefix KV blocks 的 prefill worker，
如果该 worker 过载，则通过 load-aware 兜底迁移。

在 4 卡 A40 2P2D sync_gpu 实验中，我构造了多 session 多 step 的 Agent prefix reuse trace。
结果显示，prefix hit rate 从 75% 提升到 87.5%，平均命中 prefix tokens 从约 683 提升到 768，
平均新增 prefill tokens 从约 507 降到 330。
在强调核心执行链路的配置下，affinity_load_aware 相比 round_robin 将 core_e2e p90 降低约 42%。

同时我发现 A40 机器不是 NVSwitch 全互联，而是 GPU0-GPU1、GPU2-GPU3 两组 NVLink pair。
如果调度产生 P0->D1 或 P1->D0 这类跨 pair 路由，KV handoff 会走 SYS 路径，可能抵消 prefix reuse 的收益。
因此最终策略需要同时考虑 prefix locality、worker load 和 P/D 物理链路 locality。
```

不能过度说：

```text
不能说当前已经实现 D->P KV 回写。
不能说当前已经实现跨 session prefix cache。
不能说当前 wall p90 稳定降低 20% 以上。
```

