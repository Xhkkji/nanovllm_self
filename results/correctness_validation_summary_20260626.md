# Correctness Validation Summary (20260626)

## Scope

这份文档只整理“功能正确性”相关结论，聚焦当前已经验证通过、建议长期保留的基线。

## 当前推荐基线

### 1. torch prefill + flash decode

- 测试文件: `tests/test_flash_decode_compare_with_hf.py`
- 结论: 通过
- 覆盖:
  - 单 seq greedy
  - 多 seq greedy
  - `torch prefill + flash decode`
- 保留日志:
  - `results/test_logs/test_flash_decode_compare_with_hf.log`

### 2. flash prefill + flash decode

- 测试文件: `tests/test_flash_prefill_compare_with_hf.py`
- 结论: 通过
- 覆盖:
  - 单 seq greedy
  - 多 seq greedy
  - `flash prefill + flash decode`

### 3. 长前缀 prefix prefill / prefix cache

- 测试文件: `tests/test_prefix_prefill_long.py`
- 结论: 通过
- 覆盖:
  - 长共享前缀超过完整 block
  - `num_cached_tokens` 是否等于预期完整 block 数量
  - prefix prefill 是否只计算 suffix
  - next token 是否与 HF / non-prefix 一致
- 保留日志:
  - `results/test_logs/test_all_prefill_paths.log`

## 当前已经明确验证通过的路径

按递进关系整理，当前已经走通的是：

1. 基础 greedy 推理正确
2. `torch prefill + flash decode` 与 HF 对齐
3. `flash prefill + flash decode` 与 HF 对齐
4. 长前缀 prefix cache / prefix prefill 与 HF 对齐

这意味着当前主推理链路已经从“基本能跑”推进到“多条关键 serving 路径都已验证通过”。

## 当前推荐最小回归集合

```bash
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_flash_decode_compare_with_hf.py
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_flash_prefill_compare_with_hf.py
PYTHONPATH=/home/xhk/nanovllm_self python tests/test_prefix_prefill_long.py
```

## 历史参考

下面这些测试文件保留，但不再作为当前首选基线：

- `tests/test_compare_with_hf.py`
- `tests/test_prefix_prefill.py`
- `tests/SAMPLING_TEST_RESULTS.txt`

说明：

- `tests/test_compare_with_hf.py` 更接近早期主链路基线
- `tests/test_prefix_prefill.py` 更接近早期短 prefix 样例
- 当前更可靠的正确性基线，应以 flash 路径和长前缀 prefix 场景为准

## 结论

从正确性角度看，当前系统已经完成了这一阶段最关键的三件事：

1. 主 decode 路径对齐 HF
2. flash prefill 路径对齐 HF
3. prefix prefill / prefix cache 在真实长前缀场景下对齐 HF
