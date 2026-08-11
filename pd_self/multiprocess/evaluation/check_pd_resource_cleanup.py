import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]


def resolve_path(path: str) -> Path:
    """把相对路径解析到仓库根目录，方便脚本从任意目录执行。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_DIR / p


def read_json(path: Path) -> dict:
    """读取 JSON 文件；资源检查里主要用于读取 worker_state 和 error 文件。"""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_worker_processes() -> list[dict]:
    """扫描当前用户下仍在运行的 PD worker 进程，检查 benchmark 是否正常收尾。"""
    result = subprocess.run(
        ["ps", "-u", os.environ.get("USER", ""), "-o", "pid=,ppid=,cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        if "pd_self/multiprocess/persistent_" not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "cmd": parts[2],
            }
        )
    return rows


def collect_error_files(work_dir: Path) -> list[dict]:
    """收集 work_dir 下的 prefill/decode error 文件，帮助定位异常请求。"""
    rows = []
    if not work_dir.exists():
        return rows
    for path in sorted(work_dir.rglob("*.error.json")):
        try:
            payload = read_json(path)
        except Exception as exc:
            payload = {"read_error": repr(exc)}
        rows.append(
            {
                "path": str(path),
                "payload": payload,
            }
        )
    return rows


def collect_state_files(work_dir: Path) -> list[dict]:
    """读取 worker_state 文件，确认 worker 退出前是否仍有 active/pending 状态。"""
    rows = []
    if not work_dir.exists():
        return rows
    for path in sorted(work_dir.rglob("*worker_state.json")):
        try:
            state = read_json(path)
        except Exception as exc:
            state = {"read_error": repr(exc)}
        rows.append(
            {
                "path": str(path),
                "state": state,
            }
        )
    return rows


def collect_control_files(work_dir: Path) -> dict:
    """统计 work_dir 中剩余控制文件数量，用于判断是否有未完成请求残留。"""
    suffixes = [
        ".request.json",
        ".payload.pkl",
        ".payload_ready",
        ".recv_ready",
        ".prefill_done",
        ".decode_done",
        ".prefill_metrics.json",
        ".decode_metrics.json",
    ]
    counts = {suffix: 0 for suffix in suffixes}
    if not work_dir.exists():
        return counts
    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        for suffix in suffixes:
            if path.name.endswith(suffix):
                counts[suffix] += 1
    return counts


def collect_shm_candidates() -> list[dict]:
    """保守扫描 /dev/shm 下的 Python shared_memory 候选项，只报告不删除。"""
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        return []
    rows = []
    for path in sorted(shm_dir.glob("psm_*")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        rows.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return rows


def state_has_pending_work(state: dict) -> bool:
    """判断单个 worker_state 是否显示仍有 active/pending 工作。"""
    keys = (
        "busy",
        "request_queue_depth",
        "pending_handoffs",
        "pending_sends",
        "active_decode_requests",
        "pending_recvs",
    )
    for key in keys:
        value = state.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def parse_args():
    """解析资源清理检查参数。"""
    parser = argparse.ArgumentParser(
        description="Check whether PD benchmark workers and temporary resources were cleaned up."
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fail-on-shm-candidates",
        action="store_true",
        help="Treat any /dev/shm/psm_* candidate as failure. Default only reports them.",
    )
    return parser.parse_args()


def main():
    """执行资源清理检查并写出 JSON 报告。"""
    args = parse_args()
    work_dir = resolve_path(args.work_dir)
    worker_processes = collect_worker_processes()
    error_files = collect_error_files(work_dir)
    state_files = collect_state_files(work_dir)
    control_file_counts = collect_control_files(work_dir)
    shm_candidates = collect_shm_candidates()
    pending_states = [
        item for item in state_files if state_has_pending_work(item.get("state", {}))
    ]

    passed = (
        not worker_processes
        and not error_files
        and not pending_states
        and (not shm_candidates or not args.fail_on_shm_candidates)
    )
    summary = {
        "work_dir": str(work_dir),
        "passed": passed,
        "worker_processes": worker_processes,
        "num_error_files": len(error_files),
        "error_files": error_files[:20],
        "num_state_files": len(state_files),
        "pending_state_files": pending_states[:20],
        "control_file_counts": control_file_counts,
        "num_shm_candidates": len(shm_candidates),
        "shm_candidates": shm_candidates[:50],
        "fail_on_shm_candidates": args.fail_on_shm_candidates,
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("resource_check_written", output_path)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
