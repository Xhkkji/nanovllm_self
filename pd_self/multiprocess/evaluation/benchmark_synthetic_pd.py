import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT_DIR)

from benchmark_synthetic_common import (
    DEFAULT_DATASET,
    DEFAULT_RESULT_DIR,
    cap_max_tokens,
    select_requests,
    summarize,
    write_json,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-GPU PD synthetic serving benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=16)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument("--python-bin", default="/home/xhk/miniconda3/envs/pytorch/bin/python")
    parser.add_argument("--work-dir", default=f"{DEFAULT_RESULT_DIR}/pd_synthetic_work")
    parser.add_argument("--output", default=f"{DEFAULT_RESULT_DIR}/pd_synthetic_metrics.jsonl")
    parser.add_argument("--summary-output", default=f"{DEFAULT_RESULT_DIR}/pd_synthetic_summary.json")
    parser.add_argument("--timeout-s", type=int, default=300)
    return parser.parse_args()


def wait_for_payload(path, proc, timeout_s, log_path):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if Path(path).exists():
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"prefill worker exited before payload was written; see {log_path}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for payload: {path}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_one(args, row, idx, work_dir):
    request_id = row.get("id", f"synth-{idx:04d}")
    max_tokens = max(2, cap_max_tokens(row, args.max_output_tokens_cap))
    request = {
        **row,
        "id": request_id,
        "max_tokens": max_tokens,
    }

    request_json = work_dir / f"{idx:04d}_{request_id}.json"
    payload_path = work_dir / f"{idx:04d}_{request_id}.payload.pkl"
    done_path = work_dir / f"{idx:04d}_{request_id}.done"
    prefill_log = work_dir / f"{idx:04d}_{request_id}.prefill.log"
    decode_log = work_dir / f"{idx:04d}_{request_id}.decode.log"
    prefill_metrics_path = work_dir / f"{idx:04d}_{request_id}.prefill_metrics.json"
    decode_metrics_path = work_dir / f"{idx:04d}_{request_id}.decode_metrics.json"

    request_json.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    for path in (payload_path, done_path, prefill_metrics_path, decode_metrics_path):
        if path.exists():
            path.unlink()

    prefill_env = os.environ.copy()
    prefill_env["CUDA_VISIBLE_DEVICES"] = args.prefill_gpu
    decode_env = os.environ.copy()
    decode_env["CUDA_VISIBLE_DEVICES"] = args.decode_gpu

    prefill_cmd = [
        args.python_bin,
        "pd_self/multiprocess/prefill_worker.py",
        "--request-json",
        str(request_json),
        "--kv-cache-quant-mode",
        args.kv_cache_quant_mode,
        "--out",
        str(payload_path),
        "--done-file",
        str(done_path),
        "--metrics-out",
        str(prefill_metrics_path),
    ]
    decode_cmd = [
        args.python_bin,
        "pd_self/multiprocess/decode_worker.py",
        "--kv-cache-quant-mode",
        args.kv_cache_quant_mode,
        "--infile",
        str(payload_path),
        "--done-file",
        str(done_path),
        "--metrics-out",
        str(decode_metrics_path),
        "--run-to-finish",
    ]

    wall_t0 = perf_counter()
    prefill_proc = None
    try:
        with prefill_log.open("w", encoding="utf-8") as f:
            prefill_proc = subprocess.Popen(
                prefill_cmd,
                cwd=ROOT_DIR,
                env=prefill_env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        wait_for_payload(payload_path, prefill_proc, args.timeout_s, prefill_log)
        with decode_log.open("w", encoding="utf-8") as f:
            subprocess.run(
                decode_cmd,
                cwd=ROOT_DIR,
                env=decode_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout_s,
            )
        prefill_proc.wait(timeout=args.timeout_s)
    finally:
        if prefill_proc is not None and prefill_proc.poll() is None:
            prefill_proc.kill()
            prefill_proc.wait(timeout=10)

    wall_e2e_time_s = perf_counter() - wall_t0
    prefill_metrics = read_json(prefill_metrics_path)
    decode_metrics = read_json(decode_metrics_path)

    prefill_time_s = float(prefill_metrics.get("prefill_time_s", 0.0))
    payload_write_time_s = float(prefill_metrics.get("payload_write_time_s", 0.0))
    payload_read_time_s = float(decode_metrics.get("payload_read_time_s", 0.0))
    restore_time_s = float(decode_metrics.get("restore_time_s", 0.0))
    decode_time_s = float(decode_metrics.get("decode_step_time_s", 0.0))
    core_e2e_time_s = (
        prefill_time_s
        + payload_write_time_s
        + payload_read_time_s
        + restore_time_s
        + decode_time_s
    )
    generated_tokens = int(decode_metrics.get("generated_tokens", 0))

    return {
        "mode": "pd_shared_memory",
        "request_index": idx,
        "request_id": request_id,
        "profile": row.get("profile"),
        "input_tokens_dataset": row.get("input_tokens"),
        "max_tokens": max_tokens,
        "target_output_tokens": row.get("max_tokens", row.get("output_len")),
        "total_tokens_dataset": row.get("total_tokens"),
        "kv_cache_quant_mode": args.kv_cache_quant_mode,
        "prefill_gpu": args.prefill_gpu,
        "decode_gpu": args.decode_gpu,
        "prefill_time_s": prefill_time_s,
        "payload_write_time_s": payload_write_time_s,
        "payload_read_time_s": payload_read_time_s,
        "restore_time_s": restore_time_s,
        "decode_time_s": decode_time_s,
        "core_e2e_time_s": core_e2e_time_s,
        "wall_e2e_time_s": wall_e2e_time_s,
        "prefill_model_init_time_s": prefill_metrics.get("model_init_time_s"),
        "decode_model_init_time_s": decode_metrics.get("model_init_time_s"),
        "generated_tokens": generated_tokens,
        "throughput_generated_tok_s": generated_tokens / core_e2e_time_s
        if core_e2e_time_s > 0
        else 0.0,
        "num_kv_blocks": prefill_metrics.get("num_kv_blocks", 0),
        "kv_nbytes": prefill_metrics.get("kv_nbytes", 0),
        "scale_nbytes": prefill_metrics.get("scale_nbytes", 0),
        "prefill_log": str(prefill_log),
        "decode_log": str(decode_log),
        "prefill_metrics": str(prefill_metrics_path),
        "decode_metrics": str(decode_metrics_path),
    }


def main():
    args = parse_args()
    requests = select_requests(
        args.dataset,
        limit=args.limit,
        profile=args.profile,
        max_total_tokens=args.max_total_tokens,
    )
    if not requests:
        raise RuntimeError("no benchmark requests selected")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in enumerate(requests):
        metrics = run_one(args, row, idx, work_dir)
        rows.append(metrics)
        print(
            f"[pd] {idx + 1}/{len(requests)} id={metrics['request_id']} "
            f"profile={metrics['profile']} generated={metrics['generated_tokens']} "
            f"core={metrics['core_e2e_time_s']:.4f}s wall={metrics['wall_e2e_time_s']:.4f}s "
            f"restore={metrics['restore_time_s']:.4f}s"
        )

    summary = summarize(
        rows,
        [
            "prefill_time_s",
            "payload_write_time_s",
            "payload_read_time_s",
            "restore_time_s",
            "decode_time_s",
            "core_e2e_time_s",
            "wall_e2e_time_s",
        ],
    )
    summary.update(
        {
            "mode": "pd_shared_memory",
            "dataset": args.dataset,
            "limit": args.limit,
            "profile": args.profile,
            "max_total_tokens": args.max_total_tokens,
            "max_output_tokens_cap": args.max_output_tokens_cap,
            "prefill_gpu": args.prefill_gpu,
            "decode_gpu": args.decode_gpu,
            "kv_cache_quant_mode": args.kv_cache_quant_mode,
        }
    )
    write_jsonl(args.output, rows)
    write_json(args.summary_output, summary)
    print("metrics_written", args.output)
    print("summary_written", args.summary_output)


if __name__ == "__main__":
    main()
