import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAUDE_SELF_DIR = os.path.join(ROOT_DIR, "claude_self")
for path in (ROOT_DIR, CLAUDE_SELF_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent import Agent  # noqa: E402
from scheduled_backend import ScheduledBackend  # noqa: E402


class FakeBackend:
    """
    Agent 异步/PD 分离调度测试用的假后端。

    这个测试不加载真实模型，也不占用 GPU。它只模拟两次 LLM 调用：
    1. 第一次返回 ACTION，让 LangGraph 进入 tool_node；
    2. 第二次返回 FINAL，让 LangGraph 正常收尾。

    这样可以把 Agent 编排层和底层推理引擎解耦开：
    - 如果这个测试失败，说明 Agent 状态流转、工具执行或 metadata 有问题；
    - 如果这个测试通过，而真实模型失败，再去排查 NanovllmBackend / PDPoolBackend。
    """

    def __init__(self):
        self.calls = []

    def generate_text(self, prompt: str, max_tokens: int = 128, metadata: dict | None = None) -> str:
        """
        模拟 backend.generate_text() 接口。

        注意接口参数必须和真实后端保持一致，尤其是 metadata。
        Agent-aware 调度依赖 metadata 里的 session_id / step_id / task_type。
        """
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "metadata": metadata or {},
            }
        )

        if len(self.calls) == 1:
            return (
                "ACTION: read_file\n"
                "ACTION_INPUT: /home/xhk/nanovllm_self/main.py"
            )

        return "FINAL: 已经读取 main.py，文件主要用于启动本地 nanovllm 推理测试。"


def main():
    """
    验证最小 Agent 工具闭环：

        user
          -> llm 输出 ACTION
          -> tool_node 执行 read_file
          -> llm 根据工具结果输出 FINAL
          -> final_node 返回结果

    同时验证 ScheduledBackend 能拿到每一步 Agent metadata。
    """
    fake_backend = FakeBackend()

    # Agent-aware 调度包装器。
    # 这里仍然使用 ScheduledBackend，是为了验证每个 Agent step 都会进入调度器。
    # inner_backend 用 FakeBackend，避免测试依赖 GPU / 模型权重。
    backend = ScheduledBackend(
        inner_backend=fake_backend,
        scheduler="affinity_load_aware",
        num_workers=2,
        initial_backlog_s="0,0",
    )

    agent = Agent(
        backend=backend,
        max_steps=3,
        session_id="fake-agent-test",
        max_tokens=64,
    )

    result = agent.run("请先使用工具读取项目入口文件，然后总结它的作用。")

    message_roles = [msg["role"] for msg in agent.messages]
    metadata_rows = [call["metadata"] for call in fake_backend.calls]

    print("===== RESULT =====")
    print(result)

    print("\n===== BACKEND CALL METADATA =====")
    for index, metadata in enumerate(metadata_rows):
        print(
            index,
            metadata.get("session_id"),
            metadata.get("step_id"),
            metadata.get("task_type"),
        )

    print("\n===== ROUTE HISTORY =====")
    for item in backend.route_history:
        print(item)

    print("\n===== MESSAGE ROLES =====")
    print(message_roles)

    assert result.startswith("已经读取")
    assert len(fake_backend.calls) == 2
    assert len(backend.route_history) == 2
    assert message_roles == ["user", "assistant", "tool", "assistant"]
    assert metadata_rows[0]["session_id"] == "fake-agent-test"
    assert metadata_rows[0]["step_id"] == 0
    assert metadata_rows[0]["task_type"] == "plan"
    assert metadata_rows[1]["session_id"] == "fake-agent-test"
    assert metadata_rows[1]["step_id"] == 1
    assert metadata_rows[1]["task_type"] == "tool_reason"

    print("\nPASS: agent llm -> tool -> llm -> final")


if __name__ == "__main__":
    main()
