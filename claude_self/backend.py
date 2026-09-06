import torch
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams

class NanovllmBackend:
    """
    上层 agent 只调用 `generate_text(prompt)`
    
    注意：
    1. metadata 是给后续 Agent-aware 调度用的。
    2. 当前本地 backend 暂时不用 metadata，只是保持接口兼容。
    3. 后面切到 PDBackend / LangGraph wrapper 时，不需要改 Agent 主循环。
    """
    def __init__(self, model_path="/home/xhk/model/Qwen3-0.6B/", enable_profile=False):
        self.llm = LLM_self(
            model_path=model_path,
            enable_profile=enable_profile,
        )

    def generate_text(self, prompt: str, max_tokens: int = 256, metadata: dict | None = None) -> str:
        # Agent-aware 接口预留：
        # 本地单机 backend 暂时不使用 metadata。
        # metadata 里会包含 session_id / step_id / task_type 等调度语义。
        _ = metadata
        
        # 1. 编码
        encoded = self.llm.encoder(prompt)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)

        # 2. 当前建议先用 greedy，方便调 agent
        sampling_params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
        ]

        # 3. 调用 nanovllm_self
        output_token_ids = self.llm.generate(
            input_ids,
            sampling_params=sampling_params,
            request_metadata=metadata,
        )

        # 4. 只取新生成部分
        full_ids = output_token_ids[0]
        prompt_len = len(encoded["input_ids"])
        new_ids = full_ids[prompt_len:]

        text = self.llm.decode(new_ids)
        # 去掉字符串首尾的空白字符
        return text.strip()