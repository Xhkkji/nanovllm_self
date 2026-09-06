import os
import sys
import json
from pathlib import Path

import torch
import torch.distributed as dist


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams


def main():
    # TP 测试入口：通过环境变量透传 TP_SIZE。
    # 这样同一个脚本可以直接用于 2 卡 / 4 卡测试：
    #   TP_SIZE=2 torchrun --nproc_per_node=2 ...
    #   TP_SIZE=4 torchrun --nproc_per_node=4 ...
    tp_size = int(os.environ.get("TP_SIZE", "2"))
    prompt = os.environ.get("TP_PROMPT", "What is a large language model?")
    max_tokens = int(os.environ.get("TP_MAX_TOKENS", "16"))
    output_path = os.environ.get("TP_RESULT_PATH", "")
    model_path = os.environ.get("MODEL_PATH", "/home/xhk/model/Qwen3-0.6B")
    
    # 显存统计：先清空 peak 计数，方便观察本次模型加载 + 生成的峰值。
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


    llm = LLM_self(
        model_path=model_path,
        tensor_parallel_size=tp_size,
        enable_profile=False,
    )
    encoded = llm.encoder(prompt)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)

    outputs = llm.generate(
        input_ids,
        sampling_params=[
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
        ],
    )

    # TP 验证收尾前先同步 CUDA。
    # all_reduce 等 NCCL collective 是异步进入 CUDA stream 的，直接销毁
    # process group 偶尔会让 watchdog 在退出阶段报未处理的异步错误。
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 两个 rank 都会生成结果。
    # 第一版可以只让 rank0 打印，避免重复输出。
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_stats = {
        "rank": rank,
        "world_size": world_size,
        "tp_size": tp_size,
        "cuda_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "memory_allocated_mb": (
            torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
        "max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
        "memory_reserved_mb": (
            torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
        "max_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        ),
    }

    gathered_stats = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered_stats, local_stats)
    else:
        gathered_stats = [local_stats]

    if rank == 0:
        output_token_ids = outputs[0]
        prompt_len = len(encoded["input_ids"])
        generated_token_ids = output_token_ids[prompt_len:]
        output_text = llm.decode(outputs)
        result = {
            "tp_size": tp_size,
            "prompt": prompt,
            "prompt_token_ids": encoded["input_ids"],
            "max_tokens": max_tokens,
            "output_token_ids": output_token_ids,
            "generated_token_ids": generated_token_ids,
            "output_text": output_text,
            "rank_memory": gathered_stats,
        }
        print(output_text)
        print("===== TP RESULT JSON =====")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # TP 测试收尾：torchrun 初始化了 NCCL process group。
    # 正常销毁可以避免 PyTorch 退出时提示 process group 未关闭。
    if dist.is_initialized():
        # 所有 rank 都完成输出统计后再销毁，避免某个 rank 提前退出，
        # 另一个 rank 仍在处理 collective 相关收尾。
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
