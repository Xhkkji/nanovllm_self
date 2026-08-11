import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEFAULT_INPUT = "data/serving_benchmarks/sharegpt_qwen3_tokenized_5k.jsonl"
DEFAULT_OUTPUT = "data/serving_benchmarks/agent_trace_qwen3_tokenized.jsonl"
DEFAULT_SUMMARY = "data/serving_benchmarks/agent_trace_qwen3_summary.json"


TASK_PLANS = {
    "simple_agent": [
        ("plan", 1.0, 0.30),
        ("final_answer", 1.2, 1.00),
    ],
    "tool_agent": [
        ("plan", 1.0, 0.25),
        ("tool_reason", 2.2, 0.50),
        ("final_answer", 2.6, 1.00),
    ],
    "deep_research_agent": [
        ("plan", 1.0, 0.20),
        ("tool_reason", 2.4, 0.40),
        ("refine_reason", 3.2, 0.50),
        ("final_answer", 3.6, 1.00),
    ],
    "code_agent": [
        ("analyze_code", 1.2, 0.30),
        ("tool_reason", 2.0, 0.50),
        ("final_code", 2.4, 1.00),
    ],
}


STEP_PROMPTS = {
    "plan": "You are an agent planner. Break down the user task and decide what information is needed.",
    "tool_reason": "You are an agent reasoning over tool observations. Use the observations to update the solution.",
    "refine_reason": "You are an agent refining a multi-step solution after an intermediate observation.",
    "final_answer": "You are an agent writing the final answer after planning and tool reasoning.",
    "analyze_code": "You are a coding agent. Analyze the task and identify the implementation steps.",
    "final_code": "You are a coding agent. Produce the final code-oriented answer.",
}


FILLER_TEXT = (
    "\nObservation: synthetic tool result with relevant facts, partial evidence, "
    "intermediate state, retrieved context, and prior reasoning notes."
)


def iter_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return
    if stripped[0] == "[":
        for row in json.loads(stripped):
            yield row
        return

    # Agent trace 构造：历史 tokenized 数据理论上是 jsonl，但个别 prompt 里可能保留
    # 真实换行，导致“一条 JSON 对象跨多行”。这里用 JSONDecoder 从文本流里连续解对象，
    # 比 splitlines() 更稳，也不会影响标准 jsonl 的读取。
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        row, pos = decoder.raw_decode(text, pos)
        yield row


def cap(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def pick_task_plan(row, index: int) -> tuple[str, list[tuple[str, float, float]]]:
    # Agent trace 构造：这里不依赖真实 Agent SDK，而是根据已有 serving profile
    # 生成几类典型 Agent 任务。这样可以快速制造“多步骤、长上下文、工具结果膨胀”
    # 的 workload，用来验证推理 runtime 的调度策略。
    prompt = str(row.get("prompt", "")).lower()
    profile = row.get("profile", "")
    if any(keyword in prompt for keyword in ("code", "python", "java", "debug", "function")):
        return "code_agent", TASK_PLANS["code_agent"]
    if profile in ("long_in_long_out", "long_in_short_out"):
        return "deep_research_agent", TASK_PLANS["deep_research_agent"]
    if profile in ("short_in_long_out", "mixed_chat") or index % 3 == 0:
        return "tool_agent", TASK_PLANS["tool_agent"]
    return "simple_agent", TASK_PLANS["simple_agent"]


def target_input_tokens(base_input_tokens: int, multiplier: float, args) -> int:
    # Agent trace 构造：后续 step 会拼入工具结果/历史摘要，所以输入长度通常逐步变大。
    # 用 multiplier 控制膨胀比例，再用上下限避免生成过短或过长的极端样本。
    target = int(base_input_tokens * multiplier)
    return cap(target, args.min_step_input_tokens, args.max_step_input_tokens)


def target_output_tokens(base_output_tokens: int, ratio: float, args) -> int:
    # Agent trace 构造：plan 通常短，final answer 通常接近原始回答长度。
    # ratio 让每个 step 的 decode 压力有差异，便于调度器识别 decode-heavy 请求。
    target = int(base_output_tokens * ratio)
    return cap(target, args.min_step_output_tokens, args.max_step_output_tokens)


def make_prompt(tokenizer, row, task_kind: str, step_name: str, step_id: int, target_tokens: int) -> tuple[str, int]:
    original_prompt = str(row.get("prompt", "")).strip()
    prefix = STEP_PROMPTS.get(step_name, STEP_PROMPTS["tool_reason"])
    prompt = (
        f"{prefix}\n"
        f"Task kind: {task_kind}\n"
        f"Step id: {step_id}\n"
        f"User task:\n{original_prompt}\n"
    )

    # Agent trace 构造：这里真的用 tokenizer 把 prompt 补到目标 token 数附近。
    # 这样 benchmark 里模型实际看到的上下文长度，和 jsonl 记录的 input_tokens 基本一致；
    # 否则只改 input_tokens 元数据，prefill 压力不会真实变化。
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    while len(token_ids) < target_tokens:
        prompt += FILLER_TEXT
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    if len(token_ids) > target_tokens:
        token_ids = token_ids[:target_tokens]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

    return prompt, len(token_ids)


def build_agent_rows(source_rows, tokenizer, args):
    rows = []
    skipped = defaultdict(int)
    for program_index, row in enumerate(source_rows):
        if args.limit_programs and program_index >= args.limit_programs:
            break

        base_input_tokens = int(row.get("input_tokens", 0))
        base_output_tokens = int(row.get("max_tokens", row.get("output_len", 0)) or 0)
        if base_input_tokens <= 0 or base_output_tokens <= 0:
            skipped["missing_token_length"] += 1
            continue

        task_kind, plan = pick_task_plan(row, program_index)
        program_id = f"agent-{program_index:06d}"
        num_steps = len(plan)

        for step_id, (step_name, input_multiplier, output_ratio) in enumerate(plan):
            target_in = target_input_tokens(base_input_tokens, input_multiplier, args)
            target_out = target_output_tokens(base_output_tokens, output_ratio, args)
            prompt, actual_input_tokens = make_prompt(
                tokenizer,
                row,
                task_kind,
                step_name,
                step_id,
                target_in,
            )

            # Agent trace 数据协议：每一行仍然是一条普通 LLM 请求，但 program_id/session_id
            # 把多行串成一个 Agent 任务。调度器据此做 session affinity，PD worker 不需要知道
            # Agent 的内部语义，只需要照常执行 request.json。
            rows.append(
                {
                    "id": f"{program_id}-step-{step_id}",
                    "program_id": program_id,
                    "session_id": program_id,
                    "source_id": row.get("source_id", row.get("id", program_index)),
                    "source_request_id": row.get("id"),
                    "source": "sharegpt_agent_trace",
                    "profile": "agent_multi_step",
                    "task_kind": task_kind,
                    "task_type": step_name,
                    "step_id": step_id,
                    "num_steps": num_steps,
                    "prompt": prompt,
                    "input_tokens": actual_input_tokens,
                    "output_len": target_out,
                    "max_tokens": target_out,
                    "total_tokens": actual_input_tokens + target_out,
                    "model_path": args.model_path,
                    "tokenizer": "qwen3",
                }
            )
    return rows, skipped


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def summarize(rows, skipped, args):
    by_task_kind = defaultdict(list)
    by_task_type = defaultdict(list)
    for row in rows:
        by_task_kind[row["task_kind"]].append(row)
        by_task_type[row["task_type"]].append(row)

    def group_summary(items):
        input_tokens = [item["input_tokens"] for item in items]
        output_tokens = [item["max_tokens"] for item in items]
        total_tokens = [item["total_tokens"] for item in items]
        return {
            "count": len(items),
            "input_tokens": {
                "min": min(input_tokens),
                "p50": percentile(input_tokens, 0.50),
                "p90": percentile(input_tokens, 0.90),
                "max": max(input_tokens),
            },
            "output_tokens": {
                "min": min(output_tokens),
                "p50": percentile(output_tokens, 0.50),
                "p90": percentile(output_tokens, 0.90),
                "max": max(output_tokens),
            },
            "total_tokens": {
                "min": min(total_tokens),
                "p50": percentile(total_tokens, 0.50),
                "p90": percentile(total_tokens, 0.90),
                "max": max(total_tokens),
            },
        }

    program_ids = {row["program_id"] for row in rows}
    return {
        "input": args.input,
        "output": args.output,
        "model_path": args.model_path,
        "num_programs": len(program_ids),
        "num_requests": len(rows),
        "limit_programs": args.limit_programs,
        "max_step_input_tokens": args.max_step_input_tokens,
        "max_step_output_tokens": args.max_step_output_tokens,
        "skipped": dict(skipped),
        "by_task_kind": {
            name: group_summary(items)
            for name, items in sorted(by_task_kind.items())
        },
        "by_task_type": {
            name: group_summary(items)
            for name, items in sorted(by_task_type.items())
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a synthetic multi-step Agent serving trace from tokenized ShareGPT rows."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--limit-programs", type=int, default=200)
    parser.add_argument("--min-step-input-tokens", type=int, default=32)
    parser.add_argument("--max-step-input-tokens", type=int, default=1536)
    parser.add_argument("--min-step-output-tokens", type=int, default=16)
    parser.add_argument("--max-step-output-tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    source_rows = list(iter_jsonl(Path(args.input)))
    rows, skipped = build_agent_rows(source_rows, tokenizer, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, skipped, args)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(rows)} agent-step rows to {output_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
