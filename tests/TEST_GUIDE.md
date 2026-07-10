# Test Guide

这份文档整理当前建议保留的测试基线、运行命令，以及对应的结果总结文档。

## 当前推荐回归

### 0. P0 混合工作负载统一回归

- 文件: `tests/test_p0_mixed_workload.py`
- 用途: 统一覆盖当前最重要的旧场景：
  - `torch prefill + flash decode`
  - `flash prefill + flash decode`
  - `mixed_len`
  - 长前缀 prefix prefill / prefix cache

运行命令:

```bash
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_p0_mixed_workload.py
```

### 1. torch prefill + flash decode 对拍

- 文件: `tests/test_flash_decode_compare_with_hf.py`
- 用途: 验证当前主链路在 `torch prefill + flash decode` 组合下与 HF greedy 对齐

运行命令:

```bash
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_flash_decode_compare_with_hf.py
```

### 2. flash prefill + flash decode 对拍

- 文件: `tests/test_flash_prefill_compare_with_hf.py`
- 用途: 验证 `flash prefill + flash decode` 路径与 HF greedy 对齐

运行命令:

```bash
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_flash_prefill_compare_with_hf.py
```

### 3. 长前缀 prefix prefill / prefix cache

- 文件: `tests/test_prefix_prefill_long.py`
- 用途: 验证真实长前缀场景下：
  - prefix cache 命中
  - 命中 token 数正确
  - 只计算 suffix
  - prefix prefill 结果与 HF / non-prefix 一致

运行命令:

```bash
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_prefix_prefill_long.py
```

## 当前推荐最小回归集合

如果只想做一次最小但可靠的回归，建议跑这三条：

```bash
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_p0_mixed_workload.py
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_flash_decode_compare_with_hf.py
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_flash_prefill_compare_with_hf.py
```

## 性能 / Benchmark 脚本

### 1. 通用运行时 benchmark

- 文件: `scripts/benchmark_runtime.py`
- 用途: 跑单个组合的 TTFT / ITL / throughput / prefill tok/s / decode tok/s

### 2. prefill backend 对比

- 文件: `scripts/benchmark_prefill_backends.py`
- 用途: 比较 `torch prefill` 与 `flash prefill`

### 3. P0 全栈矩阵

- 文件: `scripts/benchmark_prefill_graph_matrix.py`
- 用途: 固定 `decode=flashattn`，比较四种组合：
  - `torch prefill + graph off`
  - `flash prefill + graph off`
  - `torch prefill + graph on`
  - `flash prefill + graph on`

### 4. CUDA Graph 矩阵

- 文件: `scripts/benchmark_cuda_graph_matrix.py`
- 用途: 评估 exact bucket / up-round bucket 的 steady-state 表现

### 5. Graph 复用验证

- 文件: `scripts/test_cuda_graph_reuse_same_process.py`
- 用途: 验证同一进程内 graph capture 只发生一次，后续走 replay

## 结果总结文档

为了避免阶段性文档分散，当前推荐直接看这三份：

- `results/correctness_validation_summary_20260626.md`
- `results/prefill_optimization_summary_20260626.md`
- `results/cuda_graph_progress_summary_20260626.md`

它们分别对应：

1. 正确性验证
2. prefill 优化推进
3. CUDA Graph 优化推进

## 历史参考文件

下面这些保留作历史参考，但不再作为当前首选基线：

- `tests/test_compare_with_hf.py`
- `tests/test_prefix_prefill.py`
- `tests/SAMPLING_TEST_RESULTS.txt`
- `NANOVLLM_SELF_INFRA_GAP_ANALYSIS.md`

说明：

- `tests/test_compare_with_hf.py` 和 `tests/test_prefix_prefill.py` 更接近早期基线，当前主推荐以 flash 路径和长前缀专项测试为准
- `tests/SAMPLING_TEST_RESULTS.txt` 记录了采样链路的人工验证结论，暂未固化为正式 unittest

## 当前状态

当前已经明确验证通过的能力：

- 单 seq greedy
- 多 seq greedy
- torch prefill + flash decode
- flash prefill + flash decode
- 长前缀 prefix cache / prefix prefill
- CUDA Graph steady-state replay
- prefill / graph 的结构化 benchmark
