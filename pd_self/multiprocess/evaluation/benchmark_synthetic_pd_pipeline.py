import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter

# Pipeline PD benchmark driver。
#
# 与 benchmark_synthetic_pd_persistent.py 的串行模式不同，这个 driver 会：
# 1. 启动同一套 persistent prefill/decode worker。
# 2. prefill worker 使用 --no-wait-decode-done，产出 payload 后立刻处理下一条请求。
# 3. driver 先投递 warmup 请求并等待完成，不纳入 summary。
# 4. driver 再一次性投递 measure 请求，形成 pipeline 批处理。
# 5. decode worker 持续扫描 prefill_done 并逐条 decode。
# 6. driver 等所有 measure decode_done 后统一汇总。
#
# 这个版本的核心指标：
# - per-request core_e2e：单条请求各阶段耗时之和。
# - per-request wall_e2e：从 request.json 写入到 decode_done 被 driver 观察到。
# - pipeline_total_wall_time_s：从第一条请求提交到最后一条请求完成。
# - pipeline_throughput_generated_tok_s：measure 请求生成 token / pipeline measure wall。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
for path in (CURRENT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT_DIR)

from benchmark_synthetic_common import (
    DEFAULT_DATASET,
    cap_max_tokens,
    default_metrics_path,
    default_summary_path,
    default_work_dir,
    select_requests,
    summarize,
    write_json,
    write_jsonl,
)


def result_mode(args):
    if getattr(args, "kv_transfer_backend", "shared_memory") == "shared_memory":
        return "pipeline_pd"
    return f"pipeline_pd_{args.kv_transfer_backend}"


def atomic_write_json(path: Path, obj) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def clean_work_dir(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    for path in work_dir.iterdir():
        if path.is_file():
            path.unlink()


def request_base(idx, request_id):
    safe_id = str(request_id).replace("/", "_")
    return f"{idx:04d}_{safe_id}"


def build_paths(work_dir: Path, base: str):
    return {
        "request": work_dir / f"{base}.request.json",
        "prefill_metrics": work_dir / f"{base}.prefill_metrics.json",
        "payload_ready": work_dir / f"{base}.payload_ready",
        "recv_ready": work_dir / f"{base}.recv_ready",
        "prefill_done": work_dir / f"{base}.prefill_done",
        "prefill_error": work_dir / f"{base}.prefill_error.json",
        "decode_metrics": work_dir / f"{base}.decode_metrics.json",
        "decode_done": work_dir / f"{base}.decode_done",
        "decode_error": work_dir / f"{base}.decode_error.json",
    }


def wait_for_file(path: Path, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def raise_if_worker_error(paths_by_base):
    for paths in paths_by_base.values():
        for key in ("prefill_error", "decode_error"):
            error_path = paths[key]
            if error_path.exists():
                raise RuntimeError(
                    f"worker error: {error_path}\n{error_path.read_text(encoding='utf-8')}"
                )


def wait_for_all_decode_done(paths_by_base, submit_times, timeout_s):
    deadline = time.time() + timeout_s
    completed_at = {}
    while time.time() < deadline:
        raise_if_worker_error(paths_by_base)
        for base, paths in paths_by_base.items():
            if base in completed_at:
                continue
            if paths["decode_done"].exists():
                completed_at[base] = perf_counter()
        if len(completed_at) == len(paths_by_base):
            return completed_at
        time.sleep(0.05)
    missing = sorted(set(paths_by_base) - set(completed_at))
    raise TimeoutError(f"timed out waiting decode_done for {missing[:5]}")


def start_worker(args, role, work_dir: Path, log_path: Path):
    env = os.environ.copy()
    if role == "prefill":
        env["CUDA_VISIBLE_DEVICES"] = args.prefill_gpu
        script = "pd_self/multiprocess/persistent_prefill_worker.py"
        extra_args = [
            "--no-wait-decode-done",
            "--max-pending-sends",
            str(args.max_pending_sends),
        ]
    elif role == "decode":
        env["CUDA_VISIBLE_DEVICES"] = args.decode_gpu
        script = "pd_self/multiprocess/persistent_decode_worker.py"
        extra_args = [
            "--decode-mode",
            args.decode_mode,
            "--max-active-decode-requests",
            str(args.max_active_decode_requests),
            "--max-pending-recvs",
            str(args.max_pending_recvs),
        ]
    else:
        raise ValueError(role)

    cmd = [
        args.python_bin,
        script,
        "--work-dir",
        str(work_dir),
        "--kv-cache-quant-mode",
        args.kv_cache_quant_mode,
        "--kv-transfer-backend",
        args.kv_transfer_backend,
        "--nccl-port",
        str(args.nccl_port),
        "--poll-interval-s",
        str(args.poll_interval_s),
        *extra_args,
    ]
    with log_path.open("w", encoding="utf-8") as f:
        return subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def submit_requests(args, entries, work_dir: Path):
    paths_by_base = {}
    metadata_by_base = {}
    submit_times = {}
    first_submit_t = None

    for idx, row, phase, measure_index in entries:
        request_id = row.get("id", f"synth-{idx:04d}")
        base = request_base(idx, request_id)
        paths = build_paths(work_dir, base)
        max_tokens = max(2, cap_max_tokens(row, args.max_output_tokens_cap))
        request = {
            **row,
            "id": request_id,
            "max_tokens": max_tokens,
        }

        now = perf_counter()
        first_submit_t = now if first_submit_t is None else first_submit_t
        submit_times[base] = now
        paths_by_base[base] = paths
        metadata_by_base[base] = {
            "phase": phase,
            "request_index": idx,
            "measure_index": measure_index,
            "request_id": request_id,
            "profile": row.get("profile"),
            "input_tokens_dataset": row.get("input_tokens"),
            "max_tokens": max_tokens,
            "target_output_tokens": row.get("max_tokens", row.get("output_len")),
            "total_tokens_dataset": row.get("total_tokens"),
        }
        atomic_write_json(paths["request"], request)

    return paths_by_base, metadata_by_base, submit_times, first_submit_t


def build_metrics_rows(args, paths_by_base, metadata_by_base, submit_times, completed_at):
    rows = []
    for base in sorted(paths_by_base):
        paths = paths_by_base[base]
        meta = metadata_by_base[base]
        prefill_metrics = read_json(paths["prefill_metrics"])
        decode_metrics = read_json(paths["decode_metrics"])

        prefill_time_s = float(prefill_metrics.get("prefill_time_s", 0.0))
        payload_write_time_s = float(prefill_metrics.get("payload_write_time_s", 0.0))
        transfer_wait_time_s = float(prefill_metrics.get("transfer_wait_time_s", 0.0))
        send_submit_time_s = float(prefill_metrics.get("send_submit_time_s", 0.0))
        send_complete_latency_s = float(
            prefill_metrics.get("send_complete_latency_s", 0.0)
        )
        send_finalize_wait_time_s = float(
            prefill_metrics.get("send_finalize_wait_time_s", 0.0)
        )
        transfer_time_s = float(prefill_metrics.get("transfer_time_s", 0.0))
        payload_read_time_s = float(decode_metrics.get("payload_read_time_s", 0.0))
        restore_time_s = float(decode_metrics.get("restore_time_s", 0.0))
        decode_time_s = float(decode_metrics.get("decode_time_s", 0.0))
        decode_compute_time_s = float(
            decode_metrics.get("decode_compute_time_s", decode_time_s)
        )
        core_e2e_time_s = (
            prefill_time_s
            + payload_write_time_s
            + transfer_wait_time_s
            + transfer_time_s
            + payload_read_time_s
            + restore_time_s
            + decode_time_s
        )
        generated_tokens = int(decode_metrics.get("generated_tokens", 0))
        wall_e2e_time_s = completed_at[base] - submit_times[base]

        rows.append(
            {
                "mode": f"pd_{args.kv_transfer_backend}_pipeline",
                **meta,
                "kv_cache_quant_mode": args.kv_cache_quant_mode,
                "kv_transfer_backend": args.kv_transfer_backend,
                "prefill_gpu": args.prefill_gpu,
                "decode_gpu": args.decode_gpu,
                "prefill_time_s": prefill_time_s,
                "payload_write_time_s": payload_write_time_s,
                "transfer_wait_time_s": transfer_wait_time_s,
                "send_submit_time_s": send_submit_time_s,
                "send_complete_latency_s": send_complete_latency_s,
                "send_finalize_wait_time_s": send_finalize_wait_time_s,
                "transfer_time_s": transfer_time_s,
                "payload_read_time_s": payload_read_time_s,
                "restore_time_s": restore_time_s,
                "decode_time_s": decode_time_s,
                "decode_compute_time_s": decode_compute_time_s,
                "core_e2e_time_s": core_e2e_time_s,
                "wall_e2e_time_s": wall_e2e_time_s,
                "generated_tokens": generated_tokens,
                "throughput_generated_tok_s": generated_tokens / core_e2e_time_s
                if core_e2e_time_s > 0
                else 0.0,
                "num_kv_blocks": prefill_metrics.get("num_kv_blocks", 0),
                "kv_nbytes": prefill_metrics.get("kv_nbytes", 0),
                "scale_nbytes": prefill_metrics.get("scale_nbytes", 0),
                "prefill_metrics": str(paths["prefill_metrics"]),
                "decode_metrics": str(paths["decode_metrics"]),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline dual-GPU PD synthetic benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=16)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument("--kv-transfer-backend", default="shared_memory", choices=["shared_memory", "sync_gpu"])
    parser.add_argument("--python-bin", default="/home/xhk/miniconda3/envs/pytorch/bin/python")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--nccl-port",
        type=int,
        default=29577,
        help="Local TCP port used by the two sync_gpu workers to initialize NCCL.",
    )
    parser.add_argument(
        "--decode-mode",
        default="continuous",
        choices=["run_to_finish", "continuous"],
        help="Decode worker mode used by pipeline benchmark.",
    )
    parser.add_argument("--max-active-decode-requests", type=int, default=4)
    parser.add_argument("--max-pending-sends", type=int, default=4)
    parser.add_argument("--max-pending-recvs", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kv_transfer_backend == "sync_gpu" and args.decode_mode != "continuous":
        raise ValueError(
            "sync_gpu requires --decode-mode continuous; "
            "run_to_finish still waits for prefill_done and can deadlock."
        )

    mode = result_mode(args)
    args.work_dir = args.work_dir or default_work_dir(mode, args.profile)
    args.output = args.output or default_metrics_path(mode, args.profile)
    args.summary_output = args.summary_output or default_summary_path(mode, args.profile)
    requests = select_requests(
        args.dataset,
        limit=args.limit + args.warmup,
        profile=args.profile,
        max_total_tokens=args.max_total_tokens,
    )
    if not requests:
        raise RuntimeError("no benchmark requests selected")

    work_dir = Path(args.work_dir)
    clean_work_dir(work_dir)

    prefill_log = work_dir / "pipeline_prefill.log"
    decode_log = work_dir / "pipeline_decode.log"
    prefill_proc = None
    decode_proc = None
    try:
        prefill_proc = start_worker(args, "prefill", work_dir, prefill_log)
        decode_proc = start_worker(args, "decode", work_dir, decode_log)
        wait_for_file(work_dir / "prefill_worker.ready.json", args.startup_timeout_s)
        wait_for_file(work_dir / "decode_worker.ready.json", args.startup_timeout_s)
        prefill_ready = read_json(work_dir / "prefill_worker.ready.json")
        decode_ready = read_json(work_dir / "decode_worker.ready.json")

        warmup_entries = [
            (idx, row, "warmup", None)
            for idx, row in enumerate(requests[: args.warmup])
        ]
        measure_entries = [
            (args.warmup + idx, row, "measure", idx)
            for idx, row in enumerate(requests[args.warmup :])
        ]

        all_rows = []
        warmup_first_submit_t = None
        warmup_last_done_t = None
        if warmup_entries:
            (
                warmup_paths,
                warmup_metadata,
                warmup_submit_times,
                warmup_first_submit_t,
            ) = submit_requests(args, warmup_entries, work_dir)
            warmup_completed_at = wait_for_all_decode_done(
                warmup_paths,
                warmup_submit_times,
                timeout_s=args.request_timeout_s,
            )
            warmup_last_done_t = max(warmup_completed_at.values())
            all_rows.extend(
                build_metrics_rows(
                    args,
                    warmup_paths,
                    warmup_metadata,
                    warmup_submit_times,
                    warmup_completed_at,
                )
            )

        (
            measure_paths,
            measure_metadata,
            measure_submit_times,
            measure_first_submit_t,
        ) = submit_requests(args, measure_entries, work_dir)
        measure_completed_at = wait_for_all_decode_done(
            measure_paths,
            measure_submit_times,
            timeout_s=args.request_timeout_s,
        )
        measure_last_done_t = max(measure_completed_at.values())
        all_rows.extend(
            build_metrics_rows(
                args,
                measure_paths,
                measure_metadata,
                measure_submit_times,
                measure_completed_at,
            )
        )
        rows = sorted(all_rows, key=lambda row: row["request_index"])
        for row in rows:
            print(
                f"[pipeline-pd][{row['phase']}] {row['request_index'] + 1}/{len(rows)} "
                f"id={row['request_id']} profile={row['profile']} generated={row['generated_tokens']} "
                f"core={row['core_e2e_time_s']:.4f}s wall={row['wall_e2e_time_s']:.4f}s "
                f"restore={row['restore_time_s']:.4f}s"
            )

        measured_rows = [row for row in rows if row.get("phase") == "measure"]
        summary = summarize(
            measured_rows,
            [
                "prefill_time_s",
                "payload_write_time_s",
                "transfer_wait_time_s",
                "send_submit_time_s",
                "send_complete_latency_s",
                "send_finalize_wait_time_s",
                "transfer_time_s",
                "payload_read_time_s",
                "restore_time_s",
                "decode_time_s",
                "decode_compute_time_s",
                "core_e2e_time_s",
                "wall_e2e_time_s",
            ],
        )
        total_generated = sum(row.get("generated_tokens", 0) for row in measured_rows)
        first_submit_candidates = [
            t for t in (warmup_first_submit_t, measure_first_submit_t) if t is not None
        ]
        first_submit_t = min(first_submit_candidates) if first_submit_candidates else None
        last_done_candidates = [
            t for t in (warmup_last_done_t, measure_last_done_t) if t is not None
        ]
        last_done_t = max(last_done_candidates) if last_done_candidates else None
        pipeline_total_wall_time_s = (
            last_done_t - first_submit_t
            if first_submit_t is not None and last_done_t is not None
            else 0.0
        )
        pipeline_measure_wall_time_s = (
            measure_last_done_t - measure_first_submit_t
            if measure_first_submit_t is not None
            else 0.0
        )
        summary.update(
            {
                "mode": f"pd_{args.kv_transfer_backend}_pipeline",
                "dataset": args.dataset,
                "limit": args.limit,
                "warmup": args.warmup,
                "total_selected_requests": len(requests),
                "measured_requests": len(measured_rows),
                "profile": args.profile,
                "max_total_tokens": args.max_total_tokens,
                "max_output_tokens_cap": args.max_output_tokens_cap,
                "prefill_gpu": args.prefill_gpu,
                "decode_gpu": args.decode_gpu,
                "kv_cache_quant_mode": args.kv_cache_quant_mode,
                "kv_transfer_backend": args.kv_transfer_backend,
                "nccl_port": args.nccl_port,
                "decode_mode": args.decode_mode,
                "max_active_decode_requests": args.max_active_decode_requests,
                "max_pending_sends": args.max_pending_sends,
                "max_pending_recvs": args.max_pending_recvs,
                "prefill_worker_init": prefill_ready,
                "decode_worker_init": decode_ready,
                "pipeline_total_wall_time_s": pipeline_total_wall_time_s,
                "pipeline_measure_wall_time_s": pipeline_measure_wall_time_s,
                "pipeline_throughput_generated_tok_s": (
                    total_generated / pipeline_measure_wall_time_s
                    if pipeline_measure_wall_time_s > 0
                    else 0.0
                ),
                "prefill_log": str(prefill_log),
                "decode_log": str(decode_log),
                "result_dir": os.path.dirname(args.summary_output),
            }
        )
        write_jsonl(args.output, rows)
        write_json(args.summary_output, summary)
        print("metrics_written", args.output)
        print("summary_written", args.summary_output)
    finally:
        (work_dir / "shutdown").write_text("shutdown\n", encoding="utf-8")
        for proc in (prefill_proc, decode_proc):
            if proc is None:
                continue
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
