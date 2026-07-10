# CUDA Graph Progress Summary (20260626)

## Scope

这份文档把 CUDA Graph 相关结果按递进关系整理到一起：

1. exact bucket 验证
2. same-instance graph reuse 验证
3. up-round bucket 验证
4. 更正式的 benchmark matrix

## 1. Exact Bucket Steady-State Benchmark

首轮先验证 exact bucket 路径：`graph_bucket = [1, 2, 4, 8]`

- 原始结果: `results/cuda_graph_exact_buckets.jsonl`
- 保留日志: `results/test_logs/benchmark_cuda_graph_exact_buckets.log`

### Results

| Prompt | Batch Size | Graph Off TTFT (ms) | Graph Off ITL (ms) | Graph Off Decode tok/s | Graph On TTFT (ms) | Graph On ITL (ms) | Graph On Decode tok/s | ITL Gain | Decode Gain | Throughput Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 24.79 | 17.55 | 57.89 | 32.97 | 6.13 | 165.85 | +65.1% | +186.5% | +169.9% |
| short | 2 | 29.69 | 18.36 | 110.63 | 30.42 | 6.33 | 320.86 | +65.5% | +190.0% | +176.4% |
| medium | 4 | 39.24 | 18.39 | 220.96 | 53.28 | 6.50 | 624.70 | +64.6% | +182.7% | +158.7% |
| medium | 8 | 56.48 | 18.42 | 441.24 | 57.88 | 6.52 | 1246.02 | +64.6% | +182.4% | +159.6% |

## 2. Same-Process Same-Instance Reuse

第二步验证 graph 是否会在同一进程、同一实例中复用，而不是重复 capture。

- 保留日志: `results/test_logs/test_cuda_graph_reuse_same_process.log`

### Results

对所有 exact bucket（`bs=1/2/4/8`），都观察到：

- `graph warmup count = 1`
- 第一轮包含 capture 成本
- 第二轮进入 replay steady-state

这一步说明当前实现已经具备：

- 同 bucket 首次 capture
- 后续同 bucket replay
- 同实例复用而不是反复重建

## 3. Up-Round Bucket Validation

第三步把 exact bucket 推进到 up-round bucket，验证非精确 batch size 的映射逻辑。

历史验证过的映射包括：

- `6 -> 8`
- `7 -> 8`
- `9 -> 16`
- `17 -> 32`

结论：

- 非 exact batch size 能正确路由到下一个 bucket
- graph capture 使用的是 rounded-up bucket
- replay 可以正常完成
- 输出能正确裁回 `real_bs`

## 4. Formal CUDA Graph Benchmark Matrix

最后用更正式的矩阵，把 exact bucket 和 up-round bucket 放到同一份文档里看 steady-state 表现。

### Results

| prompt | bs | gen | type | selected_bucket | baseline_itl_ms | graph_itl_ms | itl_gain | baseline_decode_tok/s | graph_decode_tok/s | decode_gain | baseline_throughput_tok/s | graph_throughput_tok/s | throughput_gain |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 64 | exact | 1 | 16.94 | 6.13 | +63.8% | 59.96 | 165.72 | +176.4% | 58.65 | 155.81 | +165.7% |
| short | 2 | 64 | exact | 2 | 21.79 | 6.38 | +70.7% | 93.26 | 318.36 | +241.4% | 90.78 | 296.68 | +226.8% |
| short | 3 | 64 | up_round | 4 | 20.43 | 6.48 | +68.3% | 149.16 | 470.36 | +215.3% | 144.39 | 425.32 | +194.6% |
| medium | 4 | 64 | exact | 4 | 17.82 | 6.43 | +63.9% | 228.09 | 632.21 | +177.2% | 220.60 | 574.11 | +160.2% |
| medium | 5 | 64 | up_round | 8 | 20.97 | 6.64 | +68.3% | 242.21 | 765.09 | +215.9% | 232.71 | 674.80 | +190.0% |
| medium | 8 | 64 | exact | 8 | 23.32 | 6.57 | +71.8% | 348.45 | 1236.26 | +254.8% | 332.71 | 1087.97 | +227.0% |
| medium | 9 | 64 | up_round | 16 | 17.96 | 6.88 | +61.7% | 509.18 | 1329.72 | +161.2% | 483.65 | 1167.49 | +141.4% |
| short | 17 | 32 | up_round | 32 | 17.89 | 7.22 | +59.6% | 981.01 | 2431.21 | +147.8% | 812.66 | 1733.43 | +113.3% |

### Notes

- exact bucket average gain: ITL `+67.6%`, decode `+212.4%`
- up-round bucket average gain: ITL `+64.5%`, decode `+185.0%`
- warmup runs are discarded before timing, so these are steady-state numbers

## 5. 当前结论

按递进关系看，CUDA Graph 这条线已经完成了：

1. exact bucket 验证
2. same-instance reuse 验证
3. up-round bucket 验证
4. 正式 benchmark matrix 汇总

因此当前可以比较明确地认为：

- `decode-only CUDA Graph` 已经是当前系统最有效的 decode 优化之一
- 主要收益体现在 `ITL` 和 `decode tok/s`
- 这条路径已经从“实验功能”推进到“可以作为主 serving 路径一部分”
