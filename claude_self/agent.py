from prompt import build_prompt
from parser import parse_agent_output
from tools import TOOLS
from backend import NanovllmBackend
from memory import trim_messages
import re

class Agent:
    """
    最小 agent 主循环：
    1. 保存消息历史
    2. 拼 prompt
    3. 调模型
    4. 解析输出
    5. 如果要工具，就执行工具
    6. 再继续下一轮
    """

    def __init__(self, backend, max_steps: int = 5):
        self.backend = backend
        self.max_steps = max_steps
        self.messages = []
        # 记录已经执行过的工具调用，避免重复 action 死循环
        self.executed_actions = set()
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
        })
    
    def step(self):
        prompt_messages = trim_messages(
            self.messages,
            max_context_chars=8000,
            max_tool_chars=3000,
        )
        prompt = build_prompt(prompt_messages)
        output = self.backend.generate_text(prompt)
        parsed = parse_agent_output(output)
        return output, parsed

    def _extract_file_path(self, text: str):
        match = re.search(r"(/[\w\-. /]+\.py)\b", text)
        if match:
            return match.group(1).strip()
        return None
    
    def run(self, user_input: str) -> str:
        self.add_message("user", user_input)

        lowered = user_input.lower()
        file_path = self._extract_file_path(user_input)
        if file_path and any(word in lowered for word in ["read", "summarize", "inspect", "open"]):
            tool_result = TOOLS["read_file"](file_path)
            self.executed_actions.add(("read_file", file_path))
            self.add_message(
                "tool",
                f"TOOL_RESULT for read_file:\nINPUT: {file_path}\nOUTPUT:\n{tool_result}"
            )

        for _ in range(self.max_steps):
            raw_output, parsed = self.step()

            if parsed["type"] == "final":
                self.add_message("assistant", parsed["content"])
                return parsed["content"]
            
            if parsed["type"] == "action":
                action = parsed["action"]
                action_input = parsed["action_input"]

                # 保留模型原始输出，方便后面调试
                self.add_message("assistant", raw_output)

                if action not in TOOLS:
                    self.add_message("tool", f"TOOL_RESULT for {action}:\nINPUT: {action_input}\nERROR:\nunknown tool")
                    continue
                
                # 用 action + input 作为唯一键
                action_key = (action, action_input)

                # 如果重复调用同一个工具和同一个输入，不再重复执行
                if action_key in self.executed_actions:
                    self.add_message(
                        "tool",
                        f"TOOL_RESULT for {action}:\nINPUT: {action_input}\nERROR:\nrepeated action blocked"
                    )
                    continue

                # 第一次执行该工具
                self.executed_actions.add(action_key)
                tool_result = TOOLS[action](action_input)

                 # 更明确地告诉模型这是工具结果
                self.add_message(
                    "tool",
                    f"TOOL_RESULT for {action}:\nINPUT: {action_input}\nOUTPUT:\n{tool_result}"
                )
                continue
                
        final_text = "Reached max steps without FINAL answer."
        self.add_message("assistant", final_text)
        return final_text
