import argparse
import contextlib
import io

from benchmark_runtime import make_inputs, set_backends, set_cuda_graph
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify CUDA Graph capture is reused on the second run in the same process."
    )
    parser.add_argument("--prompt", choices=["short", "medium", "long"], default="short")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=0)
    return parser.parse_args()


def count_capture_messages(log_text, batch_size):
    needle = f"graph warmup for bs={batch_size}.."
    return log_text.count(needle)


def run_once(llm, prompt_name, batch_size, gen_len):
    inputs = make_inputs(llm.tokenizer, prompt_name, batch_size)
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=gen_len, ignore_eos=True)
        for _ in range(batch_size)
    ]
    result = llm.generate(inputs, sampling_params=sampling_params, return_metrics=True)
    return result["metrics"]


def main():
    args = parse_args()
    llm = LLM_self(enable_profile=False)
    set_backends(llm, "torch", "flashattn")
    set_cuda_graph(llm, True)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for _ in range(args.warmup_runs):
            run_once(llm, args.prompt, args.batch_size, args.gen_len)
        first = run_once(llm, args.prompt, args.batch_size, args.gen_len)
        second = run_once(llm, args.prompt, args.batch_size, args.gen_len)

    log_text = buffer.getvalue()
    capture_count = count_capture_messages(log_text, args.batch_size)

    print("=== Reuse Check ===")
    print(f"prompt={args.prompt} bs={args.batch_size} gen={args.gen_len}")
    print(f"graph warmup count: {capture_count}")
    print(
        "first : "
        f"TTFT(ms)={first['ttft_ms']:.2f} "
        f"ITL(ms)={first['itl_ms']:.2f} "
        f"decode_tok/s={first['decode_tok_s']:.2f}"
    )
    print(
        "second: "
        f"TTFT(ms)={second['ttft_ms']:.2f} "
        f"ITL(ms)={second['itl_ms']:.2f} "
        f"decode_tok/s={second['decode_tok_s']:.2f}"
    )
    print()
    print("=== Captured Stdout ===")
    print(log_text, end="")

    if capture_count != 1:
        raise SystemExit(
            f"Expected graph warmup to appear exactly once in the same process, got {capture_count}."
        )


if __name__ == "__main__":
    main()
