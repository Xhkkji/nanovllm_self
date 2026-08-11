import hashlib
import os
import time
from pathlib import Path
from multiprocessing.connection import Client, Listener


def control_socket_path(work_dir: Path) -> str:
    """根据 work_dir 生成控制面 Unix socket 路径，路径过长时映射到 /tmp 短路径。"""
    sock_path = str(Path(work_dir) / "pd_control.sock")
    if len(sock_path) < 100:
        return sock_path

    # AF_UNIX socket 路径通常有 108 字节左右的系统限制。
    # benchmark 结果目录较深时，sync_gpu 控制面会因为路径过长启动失败。
    # 这里仅把控制面 socket 映射到 /tmp 短路径，payload/metrics/log 仍保持在原 work_dir。
    digest = hashlib.sha1(str(work_dir).encode("utf-8")).hexdigest()[:16]
    return f"/tmp/nanovllm_pd_{digest}.sock"


def make_control_listener(sock_path: str) -> Listener:
    """
    decode 侧创建控制面监听 socket。

    这个 socket 只传很轻的同步/异步控制消息，例如 payload_ready、recv_ready。
    真正的 KV 数据不走这里，而是走 shared_memory 或 sync_gpu 后端。
    如果上次异常退出残留了 socket 文件，这里先删掉，避免 Listener bind 失败。
    """
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass

    return Listener(sock_path, family="AF_UNIX")


def connect_control(sock_path: str, timeout_s: float = 120.0, poll_interval_s: float = 0.05):
    """
    prefill 侧连接 decode 的控制 socket。

    persistent worker 启动时 decode 可能还在加载模型/初始化 NCCL，因此 prefill
    不能只尝试一次连接。这里用轻量 retry 等待控制面可用，超时后让脚本显式失败。
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return Client(sock_path, family="AF_UNIX")
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(poll_interval_s)

    raise TimeoutError(f"timed out connecting control socket: {sock_path}")
