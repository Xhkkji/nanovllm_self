# Prefill Optimization Summary (20260626)

## Scope

这份文档整理 prefill 方向的阶段性工作，并按推进顺序收拢到一起：

1. prefill 正确性对齐
2. `torch prefill` vs `flash prefill` 性能对比
3. prefill 与 CUDA Graph 组合后的全栈结果

## 1. Prefill 正确性对齐

本阶段先解决了 prefill 两条路径的输出布局不一致问题。

### 关键修正

- `prefill_flashattn()` 返回的是 token-major:
  - `[total_q_tokens, num_heads, head_dim]`
- `prefill_torch()` 也统一到同样的 token-major 布局
- `qwen3.py` 的 prefill 路径移除多余的 `permute(1, 0, 2)`

### 结果

- `torch prefill + flash decode` 对拍通过
- `flash prefill + flash decode` 对拍通过
- 长前缀 `prefix prefill` 对拍通过

对应测试汇总日志：

- `results/test_logs/test_all_prefill_paths.log`

## 2. Prefill Backend Benchmark

固定 `decode=flashattn`，比较 `torch prefill` 和 `flash prefill`。

- 原始结果: `results/prefill_backend_benchmark.jsonl`
- 保留日志: `results/test_logs/benchmark_prefill_backends.log`

### Results

| prompt | bs | gen | torch_ttft_ms | flash_ttft_ms | ttft_gain | torch_prefill_tok/s | flash_prefill_tok/s | prefill_gain | torch_throughput_tok/s | flash_throughput_tok/s | throughput_gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 64 | 25.31 | 20.50 | +19.0% | 276.59 | 341.53 | +23.5% | 55.81 | 56.51 | +1.2% |
| short | 8 | 64 | 71.45 | 20.96 | +70.7% | 783.72 | 2671.14 | +240.8% | 406.90 | 428.27 | +5.3% |
| medium | 1 | 64 | 32.57 | 20.76 | +36.3% | 767.47 | 1203.96 | +56.9% | 47.61 | 56.21 | +18.1% |
| medium | 8 | 64 | 56.62 | 20.46 | +63.9% | 3532.05 | 9776.35 | +176.8% | 417.36 | 430.00 | +3.0% |
| long | 1 | 64 | 26.68 | 21.02 | +21.2% | 2136.55 | 2712.07 | +26.9% | 55.12 | 56.02 | +1.6% |
| long | 8 | 64 | 74.54 | 20.56 | +72.4% | 6117.20 | 22174.58 | +262.5% | 374.45 | 427.98 | +14.3% |

### Notes

- average TTFT gain: `+47.2%`
- average prefill throughput gain: `+131.2%`
- average end-to-end throughput gain: `+7.3%`
- 结论：`flash prefill` 明显改善 `TTFT` 和 `prefill tok/s`，但如果 decode 仍是主要瓶颈，端到端收益会被压缩

## 3. Prefill + Graph Full-Stack Matrix

固定 `decode=flashattn`，比较四种组合：

- `torch prefill + graph off`
- `flash prefill + graph off`
- `torch prefill + graph on`
- `flash prefill + graph on`

- 原始结果: `results/prefill_graph_matrix.jsonl`
- 保留日志: `results/test_logs/benchmark_prefill_graph_matrix.log`

### Results

| prompt | bs | gen | base_ttft | flash_ttft | graph_ttft | full_ttft | base_itl | graph_itl | full_itl | base_prefill_tok/s | flash_prefill_tok/s | full_prefill_tok/s | base_throughput | flash_throughput | graph_throughput | full_throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 64 | 24.79 | 25.98 | 26.43 | 21.79 | 17.60 | 6.15 | 6.12 | 282.34 | 269.45 | 321.22 | 56.44 | 51.45 | 154.73 | 157.03 |
| short | 8 | 64 | 54.85 | 20.71 | 55.45 | 22.09 | 18.50 | 6.58 | 6.58 | 1020.99 | 2703.92 | 2534.74 | 419.65 | 431.94 | 1088.73 | 1172.61 |
| medium | 1 | 64 | 25.10 | 20.61 | 34.48 | 27.89 | 17.83 | 6.17 | 6.22 | 995.87 | 1213.04 | 896.45 | 55.72 | 56.85 | 151.18 | 152.48 |
| medium | 8 | 64 | 56.39 | 20.41 | 58.01 | 21.52 | 18.46 | 6.58 | 6.59 | 3546.83 | 9797.57 | 9292.11 | 419.82 | 434.39 | 1083.21 | 1173.20 |
| long | 1 | 64 | 25.59 | 20.82 | 34.47 | 28.26 | 17.58 | 6.19 | 6.23 | 2227.83 | 2738.21 | 2017.02 | 56.49 | 56.57 | 150.90 | 152.09 |
| long | 8 | 64 | 56.85 | 26.67 | 76.82 | 21.48 | 18.47 | 6.70 | 6.58 | 8020.54 | 17100.51 | 21228.88 | 419.49 | 364.54 | 1026.38 | 1174.74 |

### Notes

- average full-stack TTFT gain vs base: `+29.1%`
- average full-stack ITL gain vs base: `+64.7%`
- average full-stack prefill throughput gain vs base: `+78.2%`
- average full-stack end-to-end throughput gain vs base: `+176.7%`

## 4. 当前结论

按推进顺序看，prefill 方向已经形成比较清晰的结论：

1. 先把 prefill 正确性对齐
2. 再确认 `flash prefill` 确实提升 `TTFT` 和 `prefill tok/s`
3. 最后确认它与 `CUDA Graph` 组合后，可以成为当前最优主路径的一部分

当前推荐 serving 路线：

- `flash prefill`
- `flash decode`
- `decode-only CUDA Graph`
