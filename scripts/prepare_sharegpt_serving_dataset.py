import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEFAULT_OUTPUT = "data/serving_benchmarks/sharegpt_qwen3_tokenized.jsonl"
DEFAULT_SUMMARY = "data/serving_benchmarks/sharegpt_qwen3_summary.json"


def iter_sharegpt_rows(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    if text[0] == "[":
        for row in json.loads(text):
            yield row
        return

    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def normalize_role(role):
    role = str(role).lower()
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant"):
        return "assistant"
    return role


def extract_first_turn(row):
    conversations = row.get("conversations") or row.get("conversation") or []
    if len(conversations) < 2:
        return None

    for idx in range(len(conversations) - 1):
        user_msg = conversations[idx]
        assistant_msg = conversations[idx + 1]
        if normalize_role(user_msg.get("from", user_msg.get("role"))) != "user":
            continue
        if normalize_role(assistant_msg.get("from", assistant_msg.get("role"))) != "assistant":
            continue

        prompt = str(user_msg.get("value", user_msg.get("content", ""))).strip()
        answer = str(assistant_msg.get("value", assistant_msg.get("content", ""))).strip()
        if prompt and answer:
            return prompt, answer
    return None


def classify_profile(input_tokens, output_tokens, short_in, long_in, short_out, long_out):
    if input_tokens <= short_in and output_tokens <= short_out:
        return "short_in_short_out"
    if input_tokens <= short_in and output_tokens >= long_out:
        return "short_in_long_out"
    if input_tokens >= long_in and output_tokens <= short_out:
        return "long_in_short_out"
    if input_tokens >= long_in and output_tokens >= long_out:
        return "long_in_long_out"
    return "mixed_chat"


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"]].append(row)

    summary = {"num_requests": len(rows), "profiles": {}}
    for profile, items in sorted(groups.items()):
        input_tokens = [item["input_tokens"] for item in items]
        output_tokens = [item["output_len"] for item in items]
        total_tokens = [item["total_tokens"] for item in items]
        summary["profiles"][profile] = {
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
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT conversations to nano-vLLM serving benchmark JSONL."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-input-tokens", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=512)
    parser.add_argument("--short-input-threshold", type=int, default=256)
    parser.add_argument("--long-input-threshold", type=int, default=1024)
    parser.add_argument("--short-output-threshold", type=int, default=128)
    parser.add_argument("--long-output-threshold", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    rows = []
    skipped = defaultdict(int)
    for source_idx, source_row in enumerate(iter_sharegpt_rows(input_path)):
        turn = extract_first_turn(source_row)
        if turn is None:
            skipped["no_user_assistant_turn"] += 1
            continue

        prompt, answer = turn
        input_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        output_tokens = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
        output_tokens = min(output_tokens, args.max_output_tokens_cap)
        total_tokens = input_tokens + output_tokens

        if input_tokens < args.min_input_tokens:
            skipped["too_short"] += 1
            continue
        if args.max_total_tokens and total_tokens > args.max_total_tokens:
            skipped["too_long"] += 1
            continue

        profile = classify_profile(
            input_tokens,
            output_tokens,
            args.short_input_threshold,
            args.long_input_threshold,
            args.short_output_threshold,
            args.long_output_threshold,
        )
        rows.append(
            {
                "id": f"sharegpt-{len(rows):06d}",
                "source_id": source_row.get("id", source_idx),
                "source": "sharegpt",
                "profile": profile,
                "prompt": prompt,
                "input_tokens": input_tokens,
                "output_len": output_tokens,
                "max_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_path": args.model_path,
                "tokenizer": "qwen3",
            }
        )
        if args.limit and len(rows) >= args.limit:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "model_path": args.model_path,
            "skipped": dict(skipped),
            "max_total_tokens": args.max_total_tokens,
            "max_output_tokens_cap": args.max_output_tokens_cap,
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")
    if skipped:
        print(f"skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
