import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter

# Legacy persistent PD benchmark driver。
#
# 该文件已经被当前主线 benchmark_synthetic_pd_pipeline.py 替代，仅保留为串行常驻 worker 的早期实验归档。
# 这个文件不直接跑模型，而是负责“控制面编排”：
# 1. 读取 synthetic dataset，选出 warmup + measure 请求。
# 2. 启动 persistent_prefill_worker.py 到 prefill GPU。
# 3. 启动 persistent_decode_worker.py 到 decode GPU。
# 4. 等两个 worker 写 ready 文件，确认模型只加载一次。
# 5. 串行写入 *.request.json，让 prefill/decode worker 通过文件协议处理请求。
# 6. 等每条请求的 *.decode_done，读取 prefill/decode metrics。
# 7. warmup 请求写入 metrics jsonl，但不进入 summary。
# 8. 最后写 shutdown 文件，让两个 worker 正常退出。
#
# 这版是“串行常驻 worker”：
#   request i 完成 decode 后才发 request i+1。
# 它的目标是先去掉每条请求重复启动进程/加载模型的噪声。
# 下一版如果要体现 PD overlap，可以把 driver 改成 pipeline：
#   prefill request i+1 的同时 decode request i。
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[4]
EVAL_DIR = ROOT_DIR / "pd_self" / "multiprocess" / "evaluation"
for path in (CURRENT_DIR, EVAL_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
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


def atomic_write_json(path: Path, obj) -> None:
    """原子写 request/metrics 控制文件，避免 worker 轮询时读到半截 JSON。"""
    # request/metrics 控制文件都用原子写，避免 worker 轮询时读到半截 JSON。
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def read_json(path):
    """读取 worker ready 或逐请求 metrics JSON。"""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def wait_for_file(path: Path, timeout_s: float, error_paths=None):
    """等待 ready/decode_done 等文件出现，并同步检查 worker error 文件。"""
    # 通用等待函数：
    # - 等 ready 文件：判断 worker 是否初始化完成。
    # - 等 decode_done：判断一条请求是否完成。
    # - 同时检查 error_paths：worker 失败时不要一直等到 timeout。
    deadline = time.time() + timeout_s
    error_paths = error_paths or []
    while time.time() < deadline:
        for error_path in error_paths:
            if error_path.exists():
                raise RuntimeError(f"worker error: {error_path}\n{error_path.read_text(encoding='utf-8')}")
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def clean_work_dir(work_dir: Path):
    """清理本次 persistent benchmark 的 work_dir 普通文件，避免上次残留影响本次运行。"""
    # 每次 benchmark 前清掉上一次留下的控制文件。
    # 只删除当前 work_dir 下的普通文件，不递归删目录，避免误伤 result 下其他输出。
    work_dir.mkdir(parents=True, exist_ok=True)
    for path in work_dir.iterdir():
        if path.is_file():
            path.unlink()


def request_base(idx, request_id):
    """生成一条请求的文件名前缀，确保相同 request_id 多次出现时仍不冲突。"""
    # request_id 可能来自 dataset；这里加 idx 前缀，保证同一个 id 多次出现也不冲突。
    safe_id = str(request_id).replace("/", "_")
    return f"{idx:04d}_{safe_id}"


def build_paths(work_dir: Path, base: str):
    """根据 base 构造 driver/prefill/decode 三方共享的文件协议路径集合。"""
    # driver、prefill worker、decode worker 共享同一套文件命名协议。
    # 只要 base 一致，就能定位同一条请求的 request/payload/metrics/done。
    return {
        "request": work_dir / f"{base}.request.json",
        "prefill_metrics": work_dir / f"{base}.prefill_metrics.json",
        "prefill_done": work_dir / f"{base}.prefill_done",
        "prefill_error": work_dir / f"{base}.prefill_error.json",
        "decode_metrics": work_dir / f"{base}.decode_metrics.json",
        "decode_done": work_dir / f"{base}.decode_done",
        "decode_error": work_dir / f"{base}.decode_error.json",
    }


def start_worker(args, role, work_dir: Path, log_path: Path):
    """启动一个常驻 prefill 或 decode worker，并把 stdout/stderr 写入对应日志文件。"""
    # 启动常驻 worker。
    # 注意：子进程内部 config.device 都是 cuda:0；
    # 这里通过 CUDA_VISIBLE_DEVICES 把 cuda:0 映射到指定物理 GPU。
    env = os.environ.copy()
    if role == "prefill":
        env["CUDA_VISIBLE_DEVICES"] = args.prefill_gpu
        script = "pd_self/multiprocess/persistent_prefill_worker.py"
    elif role == "decode":
        env["CUDA_VISIBLE_DEVICES"] = args.decode_gpu
        script = "pd_self/multiprocess/persistent_decode_worker.py"
    else:
        raise ValueError(role)

    cmd = [
        args.python_bin,
        script,
        "--work-dir",
        str(work_dir),
        "--kv-cache-quant-mode",
        args.kv_cache_quant_mode,
        "--poll-interval-s",
        str(args.poll_interval_s),
    ]
    with log_path.open("w", encoding="utf-8") as f:
        # stdout/stderr 全部写日志文件。
        # 这样 benchmark 控制台只输出每条请求的核心指标，详细模型加载日志留在 work_dir。
        return subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def run_one(args, row, idx, work_dir: Path, phase: str, measure_index: int | None):
    """串行提交单条请求给常驻 PD worker，等待完成后汇总 prefill/decode 指标。"""
    # driver 侧处理一条请求的完整生命周期：
    # 1. 根据 dataset row 生成 request JSON。
    # 2. 写入 *.request.json，prefill worker 会轮询到它。
    # 3. 等待 *.decode_done，表示 decode worker 已完成。
    # 4. 读取 prefill/decode metrics，合成单条请求的统一指标。
    request_id = row.get("id", f"synth-{idx:04d}")
    base = request_base(idx, request_id)
    paths = build_paths(work_dir, base)
    max_tokens = max(2, cap_max_tokens(row, args.max_output_tokens_cap))
    request = {
        **row,
        "id": request_id,
        "max_tokens": max_tokens,
    }

    wall_t0 = perf_counter()
    # 这是请求进入系统的时间点。wall_e2e 包含文件轮询等待，
    # 但不包含 worker 启动和模型初始化。
    atomic_write_json(paths["request"], request)
    wait_for_file(
        paths["decode_done"],
        timeout_s=args.request_timeout_s,
        error_paths=[paths["prefill_error"], paths["decode_error"]],
    )
    wall_e2e_time_s = perf_counter() - wall_t0

    # prefill/decode metrics 分开记录，再在 driver 侧拼出统一口径。
    prefill_metrics = read_json(paths["prefill_metrics"])
    decode_metrics = read_json(paths["decode_metrics"])

    prefill_time_s = float(prefill_metrics.get("prefill_time_s", 0.0))
    payload_write_time_s = float(prefill_metrics.get("payload_write_time_s", 0.0))
    payload_read_time_s = float(decode_metrics.get("payload_read_time_s", 0.0))
    restore_time_s = float(decode_metrics.get("restore_time_s", 0.0))
    decode_time_s = float(decode_metrics.get("decode_time_s", 0.0))
    core_e2e_time_s = (
        # core_e2e 是“核心执行成本”：
        # prefill + payload 写读 + restore + decode。
        # 它不包含 worker 初始化，也基本不包含外部脚本启动成本。
        prefill_time_s
        + payload_write_time_s
        + payload_read_time_s
        + restore_time_s
        + decode_time_s
    )
    generated_tokens = int(decode_metrics.get("generated_tokens", 0))

    return {
        "mode": "pd_shared_memory_persistent",
        "phase": phase,
        "request_index": idx,
        "measure_index": measure_index,
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


def parse_args():
    """解析常驻串行 PD benchmark 参数，包括 warmup、GPU、结果路径和超时设置。"""
    parser = argparse.ArgumentParser(description="Persistent dual-GPU PD synthetic benchmark.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens-cap", type=int, default=16)
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--kv-cache-quant-mode", default="int8_mock", choices=["none", "int8_mock"])
    parser.add_argument("--python-bin", default="/home/xhk/miniconda3/envs/pytorch/bin/python")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    return parser.parse_args()


def main():
    """常驻串行 PD benchmark 主入口：启动 worker、逐条提交请求、写 summary 并关闭 worker。"""
    args = parse_args()
    args.work_dir = args.work_dir or default_work_dir("persistent_pd", args.profile)
    args.output = args.output or default_metrics_path("persistent_pd", args.profile)
    args.summary_output = args.summary_output or default_summary_path("persistent_pd", args.profile)
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

    prefill_log = work_dir / "persistent_prefill.log"
    decode_log = work_dir / "persistent_decode.log"
    prefill_proc = None
    decode_proc = None
    try:
        # 先启动两个常驻 worker，然后等 ready 文件。
        # ready 文件里会记录各自的 model_init_time_s，便于确认模型只加载一次。
        prefill_proc = start_worker(args, "prefill", work_dir, prefill_log)
        decode_proc = start_worker(args, "decode", work_dir, decode_log)
        wait_for_file(work_dir / "prefill_worker.ready.json", args.startup_timeout_s)
        wait_for_file(work_dir / "decode_worker.ready.json", args.startup_timeout_s)

        prefill_ready = read_json(work_dir / "prefill_worker.ready.json")
        decode_ready = read_json(work_dir / "decode_worker.ready.json")
        rows = []
        for idx, row in enumerate(requests):
            # 第一版串行发请求：这会牺牲 PD overlap，但最容易验证链路正确性和指标稳定性。
            phase = "warmup" if idx < args.warmup else "measure"
            measure_index = idx - args.warmup if phase == "measure" else None
            metrics = run_one(args, row, idx, work_dir, phase, measure_index)
            rows.append(metrics)
            print(
                f"[persistent-pd][{phase}] {idx + 1}/{len(requests)} id={metrics['request_id']} "
                f"profile={metrics['profile']} generated={metrics['generated_tokens']} "
                f"core={metrics['core_e2e_time_s']:.4f}s wall={metrics['wall_e2e_time_s']:.4f}s "
                f"restore={metrics['restore_time_s']:.4f}s"
            )

        measured_rows = [row for row in rows if row.get("phase") == "measure"]
        summary = summarize(
            measured_rows,
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
                # summary 中保留 worker init 信息，但不把 init 时间计入每条请求 core_e2e。
                "mode": "pd_shared_memory_persistent",
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
                "prefill_worker_init": prefill_ready,
                "decode_worker_init": decode_ready,
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
        # 无论 benchmark 正常结束还是中途异常，都写 shutdown。
        # worker 主循环看到 shutdown 文件后会退出；如果 20 秒内不退出，再强制 kill。
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
