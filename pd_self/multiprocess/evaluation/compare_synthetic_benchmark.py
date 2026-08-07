import argparse
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT_DIR)

from benchmark_synthetic_common import (
    default_compare_path,
    default_summary_path,
    write_json,
)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_avg(summary, key):
    return summary.get("time", {}).get(key, {}).get("avg")


def ratio(numerator, denominator):
    if denominator in (None, 0):
        return None
    return numerator / denominator


def parse_args():
    parser = argparse.ArgumentParser(description="Compare single-GPU and PD benchmark summaries.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--pd-mode", default="persistent_pd", choices=["persistent_pd", "pipeline_pd"])
    parser.add_argument("--single", default=None)
    parser.add_argument("--pd", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.single = args.single or default_summary_path("single_gpu", args.profile)
    args.pd = args.pd or default_summary_path(args.pd_mode, args.profile)
    args.output = args.output or default_compare_path(args.profile, args.pd_mode)
    single = read_json(args.single)
    pd = read_json(args.pd)

    single_core = get_avg(single, "core_e2e_time_s")
    pd_core = get_avg(pd, "core_e2e_time_s")
    single_wall = get_avg(single, "wall_e2e_time_s")
    pd_wall = get_avg(pd, "wall_e2e_time_s")

    comparison = {
        "single_summary": args.single,
        "pd_summary": args.pd,
        "profile": args.profile,
        "pd_mode": args.pd_mode,
        "num_requests_single": single.get("num_requests"),
        "num_requests_pd": pd.get("num_requests"),
        "avg_core_e2e_single_s": single_core,
        "avg_core_e2e_pd_s": pd_core,
        "avg_core_e2e_pd_over_single": ratio(pd_core, single_core),
        "avg_wall_e2e_single_s": single_wall,
        "avg_wall_e2e_pd_s": pd_wall,
        "avg_wall_e2e_pd_over_single": ratio(pd_wall, single_wall),
        "single_throughput_generated_tok_s": single.get("throughput_generated_tok_s"),
        "pd_throughput_generated_tok_s": pd.get("throughput_generated_tok_s"),
        "pd_restore_avg_s": get_avg(pd, "restore_time_s"),
        "pd_prefill_avg_s": get_avg(pd, "prefill_time_s"),
        "pd_decode_avg_s": get_avg(pd, "decode_time_s"),
        "pd_pipeline_measure_wall_time_s": pd.get("pipeline_measure_wall_time_s"),
        "pd_pipeline_throughput_generated_tok_s": pd.get("pipeline_throughput_generated_tok_s"),
    }
    write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("comparison_written", args.output)


if __name__ == "__main__":
    main()
