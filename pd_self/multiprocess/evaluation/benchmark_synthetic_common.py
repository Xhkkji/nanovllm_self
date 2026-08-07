import json
import math
from pathlib import Path


DEFAULT_DATASET = "data/serving_benchmarks/synthetic_serving_qwen3_tokenized.jsonl"
DEFAULT_RESULT_DIR = "pd_self/multiprocess/result"


def profile_dir_name(profile):
    if not profile:
        return "all"
    safe = []
    for ch in str(profile):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def benchmark_result_dir(mode, profile):
    return Path(DEFAULT_RESULT_DIR) / mode / profile_dir_name(profile)


def default_metrics_path(mode, profile):
    return str(benchmark_result_dir(mode, profile) / "synthetic_metrics.jsonl")


def default_summary_path(mode, profile):
    return str(benchmark_result_dir(mode, profile) / "synthetic_summary.json")


def default_work_dir(mode, profile):
    return str(benchmark_result_dir(mode, profile) / "work")


def default_compare_path(profile, pd_mode="persistent_pd"):
    return str(
        Path(DEFAULT_RESULT_DIR)
        / "compare"
        / profile_dir_name(profile)
        / f"{pd_mode}_vs_single_summary.json"
    )


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def select_requests(path, limit, profile=None, max_total_tokens=2048):
    rows = []
    for row in load_jsonl(path):
        if profile and row.get("profile") != profile:
            continue
        if max_total_tokens and row.get("total_tokens", 0) > max_total_tokens:
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[lower]
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(rows, time_keys):
    summary = {
        "num_requests": len(rows),
        "profiles": sorted({row.get("profile") for row in rows if row.get("profile")}),
        "generated_tokens": sum(row.get("generated_tokens", 0) for row in rows),
        "time": {},
    }
    for key in time_keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        summary["time"][key] = {
            "avg": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "max": max(values),
        }

    e2e_values = [row["core_e2e_time_s"] for row in rows if row.get("core_e2e_time_s")]
    total_generated = summary["generated_tokens"]
    total_e2e = sum(e2e_values)
    summary["throughput_generated_tok_s"] = (
        total_generated / total_e2e if total_e2e > 0 else 0.0
    )
    return summary


def cap_max_tokens(row, cap):
    max_tokens = int(row.get("max_tokens", row.get("output_len", 1)))
    if cap and cap > 0:
        max_tokens = min(max_tokens, cap)
    return max(1, max_tokens)
