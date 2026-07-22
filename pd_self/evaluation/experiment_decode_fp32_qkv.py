import gc
from types import MethodType

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams
from pd_self.coordinator import PDCoordinator


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"

PROMPTS = [
    "What is a large language model?",
    (
        "Explain how a transformer works in detail, including token embeddings, "
        "self-attention, multi-head attention, feed-forward layers, residual "
        "connections, and why positional information is needed."
    ),
]


def build_config():
    return Config(
        model_path=MODEL_PATH,
        device="cuda:0",
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        block_size=256,
        num_blocks=256,
    )


def find_first_diff(a, b):
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return limit
    return None


def run_monolithic_serial(tokenizer, max_tokens):
    llm = LLM_self(enable_profile=False)
    outputs = []
    try:
        with torch.no_grad():
            for prompt in PROMPTS:
                input_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"]
                sampling_params = [
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=max_tokens,
                        ignore_eos=True,
                    )
                ]
                outputs.append(llm.generate(input_ids, sampling_params=sampling_params)[0])
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return outputs


def patch_decode_qkv_fp32(model_runner):
    patched = []

    for layer in model_runner.model.layers:
        orig_forward = layer.forward

        def new_forward(self, x, positions, rotary_embedding, context):
            residual = x
            x = self.ln1(x)
            total_new_tokens = x.size(0)

            is_decode_step = (
                context is not None
                and context.max_seqlen_q == 1
                and context.cu_seqlens_q is not None
            )

            if is_decode_step:
                x_fp32 = x.float()
                q = F.linear(x_fp32, self.q_proj.weight.float()).view(
                    total_new_tokens, self.num_heads, self.head_dim
                )
                k = F.linear(x_fp32, self.k_proj.weight.float()).view(
                    total_new_tokens, self.num_kv_heads, self.head_dim
                )
                v = F.linear(x_fp32, self.v_proj.weight.float()).view(
                    total_new_tokens, self.num_kv_heads, self.head_dim
                )
            else:
                q = self.q_proj(x).view(total_new_tokens, self.num_heads, self.head_dim)
                k = self.k_proj(x).view(total_new_tokens, self.num_kv_heads, self.head_dim)
                v = self.v_proj(x).view(total_new_tokens, self.num_kv_heads, self.head_dim)

            q = self.q_norm(q)
            k = self.k_norm(k)
            q, k = rotary_embedding(positions, q, k)

            attn_output = self.p_attn(q, k, v, context)
            attn_output = attn_output.contiguous().view(
                total_new_tokens, self.num_heads * self.head_dim
            )
            attn_output = self.o_proj(attn_output.to(self.o_proj.weight.dtype))
            x = attn_output + residual

            residual = x
            x = self.ln2(x)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.silu(gate) * up
            x = self.down_proj(x)
            x = residual + x
            return x

        layer.forward = MethodType(new_forward, layer)
        patched.append((layer, orig_forward))

    return patched


def restore_patch(patched):
    for layer, orig_forward in patched:
        layer.forward = orig_forward


def run_pd(max_tokens, patch_fp32_decode_qkv=False):
    config = build_config()
    pd = PDCoordinator(config)
    patched = []

    try:
        if patch_fp32_decode_qkv:
            patched = patch_decode_qkv_fp32(pd.model_runner)

        with torch.no_grad():
            outputs = pd.generate(
                texts=PROMPTS,
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
            )
        return outputs
    finally:
        if patched:
            restore_patch(patched)
        del pd
        gc.collect()
        torch.cuda.empty_cache()


class DisableTF32:
    def __enter__(self):
        self.prev_matmul = torch.backends.cuda.matmul.allow_tf32
        self.prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, exc_type, exc, tb):
        torch.backends.cuda.matmul.allow_tf32 = self.prev_matmul
        torch.backends.cudnn.allow_tf32 = self.prev_cudnn


def summarize_case(name, baseline_outputs, test_outputs, prompt_lens):
    print(f"=== {name} ===")
    for i, (ref, cur, prompt_len) in enumerate(zip(baseline_outputs, test_outputs, prompt_lens)):
        diff = find_first_diff(ref, cur)
        if diff is None:
            print(f"seq{i}: exact_match")
            continue

        gen_step = diff - prompt_len
        print(
            f"seq{i}: first_diff_abs={diff} first_diff_gen_step={gen_step} "
            f"ref_token={ref[diff]} cur_token={cur[diff]}"
        )


def main():
    max_tokens = 32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_lens = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in PROMPTS
    ]

    monolithic_outputs = run_monolithic_serial(tokenizer, max_tokens=max_tokens)
    baseline_pd_outputs = run_pd(max_tokens=max_tokens, patch_fp32_decode_qkv=False)
    fp32_pd_outputs = run_pd(max_tokens=max_tokens, patch_fp32_decode_qkv=True)
    with DisableTF32():
        fp32_no_tf32_pd_outputs = run_pd(max_tokens=max_tokens, patch_fp32_decode_qkv=True)

    print("prompt_lens:", prompt_lens)
    summarize_case("baseline_pd_vs_monolithic", monolithic_outputs, baseline_pd_outputs, prompt_lens)
    summarize_case("fp32_decode_qkv_pd_vs_monolithic", monolithic_outputs, fp32_pd_outputs, prompt_lens)
    summarize_case(
        "fp32_decode_qkv_no_tf32_pd_vs_monolithic",
        monolithic_outputs,
        fp32_no_tf32_pd_outputs,
        prompt_lens,
    )


if __name__ == "__main__":
    main()
