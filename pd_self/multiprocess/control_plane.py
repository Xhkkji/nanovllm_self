import os
import time
from pathlib import Path
from multiprocessing.connection import Client, Listener


def control_socket_path(work_dir: Path) -> str:
    return str(Path(work_dir) / "pd_control.sock")


def make_control_listener(sock_path: str) -> Listener:
    """
    decode 侧创建监听 socket。
    如果上次异常退出残留了 socket 文件，先删掉。
    """
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass

    return Listener(sock_path, family="AF_UNIX")


def connect_control(sock_path: str, timeout_s: float = 120.0, poll_interval_s: float = 0.05):
    """
    prefill 侧连接 decode 的控制 socket。
    decode worker 可能还在加载，所以这里做轻量 retry。
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            return Client(sock_path, family="AF_UNIX")
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(poll_interval_s)

    raise TimeoutError(f"timed out connecting control socket: {sock_path}")
