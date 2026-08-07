import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


MODEL_PATH = "/home/xhk/model/Qwen3-0.6B/"
DEFAULT_INPUT = "data/serving_benchmarks/synthetic_serving_lengths.jsonl"
DEFAULT_OUTPUT = "data/serving_benchmarks/synthetic_serving_qwen3_tokenized.jsonl"
DEFAULT_SUMMARY = "data/serving_benchmarks/synthetic_serving_qwen3_summary.json"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(values, pct: float):
    if not values:
        return None
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"]].append(row)

    summary = {
        "num_requests": len(rows),
        "profiles": {},
    }

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
            "fits_512": sum(item["fits_512"] for item in items),
            "fits_2048": sum(item["fits_2048"] for item in items),
        }

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize and annotate synthetic serving benchmark prompts."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--sample-per-profile", type=int, default=0)
    parser.add_argument(
        "--sample-output",
        default="data/serving_benchmarks/synthetic_serving_qwen3_tokenized_sample.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    rows = []
    for row in load_jsonl(input_path):
        input_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        input_tokens = len(input_ids)
        output_len = int(row["output_len"])
        total_tokens = input_tokens + output_len

        rows.append(
            {
                **row,
                "model_path": args.model_path,
                "tokenizer": "qwen3",
                "input_tokens": input_tokens,
                "max_tokens": output_len,
                "total_tokens": total_tokens,
                "fits_512": total_tokens <= 512,
                "fits_2048": total_tokens <= 2048,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary["input"] = str(input_path)
    summary["output"] = str(output_path)
    summary["model_path"] = args.model_path
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.sample_per_profile > 0:
        seen = defaultdict(int)
        sample_path = Path(args.sample_output)
        with sample_path.open("w", encoding="utf-8") as f:
            for row in rows:
                profile = row["profile"]
                if seen[profile] >= args.sample_per_profile:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen[profile] += 1

    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")
    if args.sample_per_profile > 0:
        print(f"wrote sample to {args.sample_output}")


if __name__ == "__main__":
    main()
