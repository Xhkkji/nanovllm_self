# 20260727 Dual-GPU PD Transfer Plan

## 1. 背景

当前 `nanovllm_self` 已经完成：

- 逻辑 PD：prefill runner 和 decode runner 分离
- `KVConnector`：producer / consumer 两侧抽象
- `KVStoreBackend`：dict / shared_memory 两种 KV 数据面
- `HandoffPayload` / `KVTransferMeta`：请求元数据和 KV 引用分离
- `int8_mock`：KV blocks + scale blocks 的 handoff 正确性
- shared memory 路径：`scale_storage_ref` 非空，prefill/decode scale 对齐，metadata 残留已清理

所以继续只做单进程单卡模拟，收益开始变低。下一步应该从“逻辑 PD”推进到“物理 PD”：

```text
prefill worker: 独立进程，独立 GPU
decode worker : 独立进程，独立 GPU
控制面       : 传 HandoffPayload / KVTransferMeta
数据面       : 先用 shared memory 传 KV blocks / scale blocks
```

## 2. 对齐 vLLM / Mooncake 的设计判断

vLLM 的 disaggregated prefilling 文档强调两点：

- prefill 和 decode 放在不同实例中，便于分别调 TTFT 和 ITL；
- KV 通过 connector 在 producer / consumer 之间转移，并且 production 级能力通常落在 connector / transfer backend。

Mooncake 的核心定位是 KVCache-centric disaggregated architecture：

- prefill 和 decoding clusters 分离；
- 把 KVCache 作为核心资源管理；
- 利用 CPU / DRAM / SSD / 传输引擎等构建解耦的数据面。

对应到当前项目，最自然的演进不是直接把所有逻辑推倒重写，而是保留已有抽象：

```text
KVConnector      ~= vLLM connector 控制面
KVStoreBackend   ~= KV transfer / lookup buffer / storage backend
SharedMemoryKVRef ~= 最小可序列化 KV reference
HandoffPayload   ~= prefill -> decode 的请求和 KV 引用元数据
```

因此下一步应该优先验证 worker 边界，而不是一上来做完整 CUDA IPC / RDMA。

## 3. 推荐路线

### P1-A: 双进程双 GPU shared-memory PD demo

目标：

```text
prefill worker 进程绑定 GPU0
decode worker 进程绑定 GPU1
payload 通过 pickle 文件交接
KV blocks / scale blocks 通过 shared memory 交接
decode worker 恢复 KV 后继续生成
```

这一步的价值：

- 证明 PD 不再只是同进程模拟；
- 证明 `pop_by_ref()` 不依赖 producer 进程中的 `_records`；
- 证明 shared memory ref 可以跨进程 attach；
- 给后续 P2P / CUDA IPC backend 提供 correctness baseline。

### P1-B: 单进程双 GPU P2P KV copy benchmark

目标：

```text
src tensor on cuda:0
dst tensor on cuda:1
dst.copy_(src, non_blocking=True)
CUDA Event 计时
对比 CPU/shared-memory 中转耗时
```

这一步只做 benchmark，不接主链路。它用于回答：

```text
GPU peer copy 是否可用？
相比 GPU -> CPU -> shared memory -> GPU 有多少潜在收益？
```

### P2: CUDA IPC / P2P transfer backend

在 P1-A 和 P1-B 都稳定后，再考虑：

```text
CudaIpcKVStoreBackend / P2PKVTransferBackend
CUDA IPC handle 生命周期
stream/event 同步
跨进程 GPU tensor attach / copy
异常清理
```

这一步更硬，但工程坑明显更多，不适合作为当前第一步。

## 4. 第一阶段验收标准

P1-A 最小验收：

```text
1. prefill worker 只初始化 prefill runner
2. decode worker 只初始化 decode runner
3. 两个进程通过 CUDA_VISIBLE_DEVICES 分别绑定不同物理 GPU
4. payload pickle 可被 decode worker 读取
5. meta.storage_ref 是 SharedMemoryKVRef
6. int8_mock 下 meta.scale_storage_ref 是 SharedMemoryKVRef
7. decode restore 后可以继续生成 token
8. consume=True 后 shared memory 被 unlink
```

注意：Python 3.11 的 `multiprocessing.shared_memory` 有 resource tracker。producer 进程如果创建 shm 后立刻退出，资源可能被清理。因此最小 demo 里 prefill worker 应保持存活，等 decode worker 写 `done_file` 后再退出。

## 5. 最简化代码样例

下面只是第一步样例，不是最终工程实现。目标是用最少代码证明：

```text
双进程
双 GPU
pickle metadata
shared memory KV + scale handoff
```

建议新建目录：

```text
pd_self/multiprocess_demo/
```

### 5.1 prefill_worker_demo.py

```python
import argparse
import os
import pickle
import time

from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from pd_self.kv_connector import KVConnector
from pd_self.kv_store import SharedMemoryKVStoreBackend
from pd_self.prefill_engine import PrefillEngine


def build_config(args):
    return Config(
        model_path=args.model_path,
        device="cuda:0",
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=512,
        block_size=256,
        num_blocks=64,
        kv_cache_quant_mode=args.kv_cache_quant_mode,
        kv_cache_scale_dtype="fp32",
        attention_compute_dtype="bf16",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/xhk/model/Qwen3-0.6B/")
    parser.add_argument("--prompt", default="What is KV cache?")
    parser.add_argument("--out", default="/tmp/nanovllm_pd_payload.pkl")
    parser.add_argument("--done-file", default="/tmp/nanovllm_pd_decode.done")
    parser.add_argument("--kv-cache-quant-mode", default="none", choices=["none", "int8_mock"])
    args = parser.parse_args()

    if os.path.exists(args.done_file):
        os.remove(args.done_file)

    config = build_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    runner = ModelRunner(config)
    backend = SharedMemoryKVStoreBackend()

    connector = KVConnector(
        config=config,
        role="producer",
        engine_id="prefill-worker-0",
        kv_store_backend=backend,
    )
    connector.register_model_runner(runner)

    engine = PrefillEngine(
        config=config,
        tokenizer=tokenizer,
        model_runner=runner,
        kv_connector=connector,
    )

    payload = engine.run_prefill(
        texts=[args.prompt],
        temperature=0.0,
        max_tokens=2,
        ignore_eos=True,
        start_seq_id=0,
    )[0]

    with open(args.out, "wb") as f:
        pickle.dump(payload, f)

    meta = payload.transfer_meta
    print("payload_written", args.out)
    print("storage_ref", type(meta.storage_ref).__name__ if meta and meta.storage_ref else None)
    print("scale_storage_ref", type(meta.scale_storage_ref).__name__ if meta and meta.scale_storage_ref else None)

    # Keep producer alive so Python resource_tracker does not clean shm before decode attaches.
    print("waiting_decode_done", args.done_file)
    while not os.path.exists(args.done_file):
        time.sleep(0.2)


if __name__ == "__main__":
    main()
```

### 5.2 decode_worker_demo.py

```python
import argparse
import os
import pickle

import torch
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from pd_self.decode_engine import DecodeEngine
from pd_self.kv_connector import KVConnector
from pd_self.kv_store import SharedMemoryKVStoreBackend


def build_config(args):
    return Config(
        model_path=args.model_path,
        device="cuda:0",
        max_num_seqs=4,
        max_num_batched_tokens=512,
        max_model_len=512,
        block_size=256,
        num_blocks=64,
        kv_cache_quant_mode=args.kv_cache_quant_mode,
        kv_cache_scale_dtype="fp32",
        attention_compute_dtype="bf16",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/xhk/model/Qwen3-0.6B/")
    parser.add_argument("--infile", default="/tmp/nanovllm_pd_payload.pkl")
    parser.add_argument("--done-file", default="/tmp/nanovllm_pd_decode.done")
    parser.add_argument("--kv-cache-quant-mode", default="none", choices=["none", "int8_mock"])
    args = parser.parse_args()

    config = build_config(args)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    runner = ModelRunner(config)

    # Fresh backend is intentional: pop_by_ref attaches by metadata ref, not by local _records.
    backend = SharedMemoryKVStoreBackend()
    connector = KVConnector(
        config=config,
        role="consumer",
        engine_id="decode-worker-0",
        kv_store_backend=backend,
    )
    connector.register_model_runner(runner)

    engine = DecodeEngine(
        config=config,
        tokenizer=tokenizer,
        model_runner=runner,
        kv_connector=connector,
    )

    with open(args.infile, "rb") as f:
        payload = pickle.load(f)

    with torch.inference_mode():
        results, restored = engine.restore_payloads([payload])
        print("restore_results", results)
        print("restored_count", len(restored))

        out = engine.step()
        print("decode_step_scheduled", len(out.scheduled))
        print("decode_step_tokens", out.token_ids)
        print("decode_finished_ids", out.finished_seq_ids)

    with open(args.done_file, "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
```

### 5.3 启动方式

终端 1：

```bash
cd /home/xhk/nanovllm_self
CUDA_VISIBLE_DEVICES=0 \
/home/xhk/miniconda3/envs/pytorch/bin/python \
pd_self/multiprocess_demo/prefill_worker_demo.py \
  --kv-cache-quant-mode int8_mock \
  --out /tmp/nanovllm_pd_payload.pkl \
  --done-file /tmp/nanovllm_pd_decode.done
```

终端 2：

```bash
cd /home/xhk/nanovllm_self
CUDA_VISIBLE_DEVICES=1 \
/home/xhk/miniconda3/envs/pytorch/bin/python \
pd_self/multiprocess_demo/decode_worker_demo.py \
  --kv-cache-quant-mode int8_mock \
  --infile /tmp/nanovllm_pd_payload.pkl \
  --done-file /tmp/nanovllm_pd_decode.done
```

这里两个进程内部都使用 `device="cuda:0"`，因为 `CUDA_VISIBLE_DEVICES` 已经把物理 GPU 做了重映射。

## 6. 为什么第一步不直接 CUDA IPC

直接 CUDA IPC 会同时引入：

- CUDA IPC handle 生命周期
- 跨进程 GPU memory ownership
- stream / event 同步
- PyTorch allocator 交互
- 异常退出清理

如果没有 shared-memory 双进程 baseline，后续很难判断错误来自：

```text
PD 状态机
payload 序列化
KV metadata
跨进程资源生命周期
GPU direct transfer
```

因此当前最小但有价值的第一步是：

```text
双进程双 GPU shared-memory PD demo
```

随后立刻补：

```text
单进程双 GPU P2P KV copy benchmark
```

这样既有功能正确性 baseline，又能尽快得到卡间直接传输的性能证据。

## 7. 后续简历表述目标

完成 P1-A / P1-B 后，可以更有底气地写：

```text
实现单机多 GPU Prefill-Decode Disaggregation 原型，Prefill Worker 与 Decode Worker 运行于独立 GPU，
通过可插拔 KV Transfer Backend 完成 KV Cache / scale metadata handoff，并对比 CPU shared-memory 中转
与 GPU P2P copy 的传输开销。
```

## 8. 参考

- vLLM Disaggregated Prefilling: https://docs.vllm.ai/en/latest/features/disagg_prefill/
- Mooncake GitHub: https://github.com/kvcache-ai/Mooncake
- Mooncake paper: https://arxiv.org/abs/2407.00079

## 9. 当前实测结果

已新增一键脚本：

```text
scripts/run_dual_gpu_pd_demo.sh
```

脚本行为：

```text
1. 清理旧 payload / done / logs / metrics
2. 后台启动 prefill worker，绑定 PREFILL_GPU，默认 GPU0
3. 等待 /tmp/nanovllm_pd_payload.pkl 生成
4. 启动 decode worker，绑定 DECODE_GPU，默认 GPU1
5. 等待 prefill worker 收到 done_file 后退出
6. 输出 prefill/decode 日志和 metrics 路径
```

运行命令：

```bash
cd /home/xhk/nanovllm_self
scripts/run_dual_gpu_pd_demo.sh
```

本次结果：

```text
prefill:
  CUDA_VISIBLE_DEVICES=0
  torch_current_device=0
  torch_device_name=NVIDIA A40
  storage_ref=SharedMemoryKVRef
  scale_storage_ref=SharedMemoryKVRef
  kv_nbytes=14680064
  scale_nbytes=458752

decode:
  CUDA_VISIBLE_DEVICES=1
  torch_current_device=0
  torch_device_name=NVIDIA A40
  restored_count=1
  decode_step_scheduled=1
  decode_step_tokens=[1558]
  decode_finished_ids=[0]
```

输出目录：

```text
pd_self/multiprocess/result/
```

metrics 文件：

```text
pd_self/multiprocess/result/dual_gpu_pd_prefill_metrics.json
pd_self/multiprocess/result/dual_gpu_pd_decode_metrics.json
```

log 文件：

```text
pd_self/multiprocess/result/dual_gpu_pd_prefill.log
pd_self/multiprocess/result/dual_gpu_pd_decode.log
```

关键耗时：

```text
prefill_time_s=1.3725
payload_write_time_s=0.00016
restore_time_s=0.00884
decode_step_time_s=1.6678
```

当前结论：

```text
P1-A 双进程双 GPU shared-memory PD demo 已跑通。
Prefill worker 与 Decode worker 分别绑定不同物理 GPU。
KV blocks 和 int8_mock scale blocks 可以通过 shared memory 跨进程 handoff。
Decode worker 能恢复 KV 并继续生成 token。
```

已知小尾巴：

```text
producer 退出时仍会打印 multiprocessing.resource_tracker warning。
原因是 decode worker 已经 unlink shared memory，producer 的 resource_tracker 退出时尝试重复清理。
该 warning 不影响当前正确性，后续可通过 resource_tracker.unregister 或统一资源 owner 设计收口。
```
