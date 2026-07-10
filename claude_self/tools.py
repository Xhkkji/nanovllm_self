import subprocess
from pathlib import Path
import shlex

ALLOWED_COMMANDS = {
    "pwd",
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "echo",
    "find",
    "grep",
    "rg",
}
# {
#     "pwd",
#     "ls",
#     "cat",
#     "head",
#     "tail",
#     "wc",
#     "echo",
#     "find",
#     "grep",
#     "rg",
#     "git",
#     "python",
#     "pytest",
# }

# 明显危险的关键字，作为额外保护
BLOCKED_KEYWORDS = {
    "rm",
    "mv",
    "cp",
    "sudo",
    "chmod",
    "chown",
    "kill",
    "pkill",
    "reboot",
    "shutdown",
    "mkfs",
    "dd",
    "mount",
    "umount",
    "curl",
    "wget",
    "ssh",
    "scp",
    "git reset",
    "git clean",
}


def read_file(path: str, max_chars: int = 4000) -> str:
    p = Path(path)

    if not p.exists():
        return f"[read_file] file not found: {path}"
    if not p.is_file():
        return f"[read_file] not a file: {path}"
    
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[read_file] faild: {e}"
    
    return text[:max_chars]

def run_shell(cmd: str, timeout: int = 10, max_chars: int = 4000, allowed_root: str = "/home/xhk/nanovllm_self") -> str:
    """
    安全版 shell 工具：
    1. 只允许白名单命令
    2. 拒绝危险关键字
    3. 限制工作目录
    4. 限制超时与输出长度
    """
    cmd = cmd.strip()
    if not cmd:
        return f"[run_shell] empty command"

    # 危险关键字拦截
    lowered = cmd.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lowered:
            return f"[run_shell] blocked dangerous command: {keyword}"

    # 用shlex安全拆分命令
    # 例子
    # cmd = "ls -la /home/user"
    # parts = shlex.split(cmd)
    # print(parts)  # ['ls', '-la', '/home/user']

    try:
        parts = shlex.split(cmd)
    except Exception as e:
        return f"[run_shell] failed to parse command: {e}"
    if not parts:
        return f"[run_shell] empty command"
    
    # 只检查主命令
    base_cmd = parts[0]
    if base_cmd not in ALLOWED_COMMANDS:
        return f"[run_shell] command not allowed: {base_cmd}"
    
    # 限制工作目录
    workdir = Path(allowed_root)
    if not workdir.exists() or not workdir.is_dir():
        return f"[run_shell] invalid allowed_root: {allowed_root}"
    

    try:
        result = subprocess.run(
            parts,
            # shell=True,  # 危险！
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir)
        )
        output = (result.stdout or "") + (result.stderr or "")
        return output[:max_chars]
    except Exception as e:
        return f"[run_shell] failed: {e}"

TOOLS = {
    "read_file": read_file,
    "run_shell": run_shell,
}
