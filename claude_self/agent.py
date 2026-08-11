from langgraph.graph import StateGraph, START, END

from prompt import build_prompt
from parser import parse_agent_output
from tools import TOOLS
from backend import NanovllmBackend
from memory import trim_messages

from typing import Any, TypedDict
import re


class AgentState(TypedDict):
    # LangGraph 的共享状态。每个 node 读这个 state，并返回更新后的 state。
    messages: list[dict[str, str]]
    session_id: str
    step_id: int
    max_steps: int
    max_tokens: int
    executed_actions: list[tuple[str, str]]
    final_answer: str | None
    last_raw_output: str | None
    last_parsed: dict[str, Any] | None


class Agent:
    """
    最小 agent 主循环：
    1. 保存消息历史
    2. 拼 prompt
    3. 调模型
    4. 解析输出
    5. 如果要工具，就执行工具
    6. 再继续下一轮
    
    20260811
    LangGraph 版本 Agent。
    主线只有一条：
    LangGraph 负责 Agent 编排；
    backend.generate_text() 负责模型调用；
    metadata 负责把 Agent step 信息传给后端调度器。
    """

    def __init__(
        self,
        backend,
        max_steps: int = 5,
        session_id: str | None = None,
        max_tokens: int = 128,
    ):
        self.backend = backend
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        # 记录已经执行过的工具调用，避免重复 action 死循环
        # 新版本使用state["executed_actions"]
        # self.executed_actions = set()
        # Agent-aware 调度：一个 session_id 对应一个完整 Agent 任务。
        # 同一个任务里的 plan/tool_reason/final_answer 都使用同一个 session_id，
        # 后续 scheduler 可以用它做 PD pair affinity / prefix locality。
        self.session_id = session_id or "agent-session-000000"
        self.graph = self._build_graph()
        
    def _extract_file_path(self, text: str):
        """
        从一段文本中提取一个看起来像是Python文件路径的子字符串
        """
        match = re.search(r"(/[\w\-. /]+\.py)\b", text)
        if match:
            return match.group(1).strip()
        return None
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
        })
        
    def _infer_task_type(self, state: AgentState) -> str:
        """
        Agent-aware 调度：根据当前消息历史粗略判断这次 LLM 调用属于哪类 Agent step。

        第一版不要复杂化：
        1. 如果刚收到 tool 结果，下一次模型调用就是 tool_reason。
        2. 如果还没有 assistant 输出，第一次模型调用就是 plan。
        3. 其他情况先归为 final_answer / mixed_agent。

        后面如果接 LangGraph，可以由 graph node 直接传 task_type。
        """
        messages = state["messages"]
        
        # 如果最后一条消息是工具结果，
        # 说明本次模型调用的任务是根据工具结果继续推理。
        if messages and messages[-1]["role"] == "tool":
            return "tool_reason"

        # 如果历史中还没有 assistant 消息，
        # 说明这是当前 Agent session 的第一次模型调用
        has_assistant = any(msg["role"] == "assistant" for msg in messages)
        if not has_assistant:
            return "plan"
        
        return "final_answer"
    
    def _build_step_metadata(self, state: AgentState, prompt: str) -> dict:
        """
        构造一次 Agent LLM 调用的元数据。
        这部分是 Agent 编排层和推理调度层之间的接口。
        当前调用链：
            Agent
            |
            | session_id / step_id / task_type / max_tokens
            v
            ScheduledBackend
            |
            v
            Agent Scheduler
        后续如果接入真实 PD Pool，
        这些字段可以继续写入 request.json，
        交给 Prefill / Decode Worker 使用。
        """
        task_type = self._infer_task_type(state)
        session_id = state["session_id"]
        step_id = state["step_id"]
        max_tokens = state["max_tokens"]
         
        return {
            # 当前请求的唯一标识。
            "id": f"{session_id}-step-{step_id}",
            # Agent session 标识。
            # 同一个 Agent 任务的多次 LLM 调用共享这个值。
            "program_id": session_id,
            "session_id": session_id,
            # 当前是该 Agent 任务的第几次模型调用。
            "step_id": step_id,
            # 当前 Agent step 类型。
            "task_type": task_type,
            # 给调度器使用的输出长度估计。
            "max_tokens": max_tokens,
            # 当前版本 ScheduledBackend 暂时不使用 prompt，
            # 但保留字段，方便后续真实 PD 请求传递。
            "prompt": prompt,
        }
    
    def _bootstrap_node(self, state: AgentState) -> AgentState:
        """
        复用你现有的文件读取快捷逻辑。

        这一步不是模型推理，只是把明显的 read_file 请求先变成 tool message，
        让后续 LLM step 直接基于工具结果回答。
        """
        messages = list(state["messages"])
        if not messages:
            return state

        user_input = messages[0]["content"]
        lowered = user_input.lower()
        file_path = self._extract_file_path(user_input)

        should_read = (
            file_path
            and any(word in lowered for word in ["read", "summarize", "inspect", "open"])
        )
        action_key = ("read_file", file_path)

        if should_read and action_key not in state["executed_actions"]:
            tool_result = TOOLS["read_file"](file_path)
            messages.append({
                "role": "tool",
                "content": (
                    f"TOOL_RESULT for read_file:\n"
                    f"INPUT: {file_path}\n"
                    f"OUTPUT:\n{tool_result}"
                ),
            })
            return {
                **state,
                "messages": messages,
                "executed_actions": [*state["executed_actions"], action_key],
            }

        return state

    def _llm_node(self, state: AgentState) -> AgentState:
        """
        LangGraph 中唯一的模型调用节点。
        
        工作：工位长（_llm_node）从传送带上取下当前的“半成品”（state）。
        这个 state 是一个包含所有历史信息的大盒子：
        messages：完整的对话历史（用户问题、模型上一轮的输出、工具返回的天气数据）。
        step_id：当前是第 2 步。
        max_tokens：本轮最多生成 1024 个 token。

        关键点：
        不在 Agent 里关心具体推理系统。
        只调用 backend.generate_text(prompt, max_tokens, metadata)。
        """
        prompt_messages = trim_messages(
            state["messages"],
            max_context_chars=8000,
            max_tool_chars=3000,
        )
        
        prompt = build_prompt(prompt_messages)
        metadata = self._build_step_metadata(state, prompt)
        
        raw_output = self.backend.generate_text(
            prompt,
            max_tokens=state["max_tokens"],
            metadata={
                **metadata,
                "prompt": prompt,
            },
        )
        parsed = parse_agent_output(raw_output)
        
        return {
            **state,
            "step_id": state["step_id"] + 1,
            "last_raw_output": raw_output,
            "last_parsed": parsed,
        }
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """
        工具执行节点。

        复用你已有的 TOOLS，不自己新造工具系统。
        后续迁移 LangChain tools 时，也只替换这里。
        """
        parsed = state["last_parsed"]
        messages = list(state["messages"])

        if parsed is None or parsed.get("type") != "action":
            return state

        action = parsed["action"]
        action_input = parsed["action_input"]
        action_key = (action, action_input)

        messages.append({
            "role": "assistant",
            "content": state["last_raw_output"] or "",
        })

        if action not in TOOLS:
            messages.append({
                "role": "tool",
                "content": (
                    f"TOOL_RESULT for {action}:\n"
                    f"INPUT: {action_input}\n"
                    f"ERROR:\nunknown tool"
                ),
            })
            return {**state, "messages": messages}

        if action_key in state["executed_actions"]:
            messages.append({
                "role": "tool",
                "content": (
                    f"TOOL_RESULT for {action}:\n"
                    f"INPUT: {action_input}\n"
                    f"ERROR:\nrepeated action blocked"
                ),
            })
            return {**state, "messages": messages}

        tool_result = TOOLS[action](action_input)
        messages.append({
            "role": "tool",
            "content": (
                f"TOOL_RESULT for {action}:\n"
                f"INPUT: {action_input}\n"
                f"OUTPUT:\n{tool_result}"
            ),
        })

        return {
            **state,
            "messages": messages,
            "executed_actions": [*state["executed_actions"], action_key],
        }
    
    def _final_node(self, state: AgentState) -> AgentState:
        """
        FINAL 收尾节点。
        """
        parsed = state["last_parsed"]

        if parsed and parsed.get("type") == "final":
            final_text = parsed["content"]
        else:
            final_text = "Reached max steps without FINAL answer."

        messages = [
            *state["messages"],
            {
                "role": "assistant",
                "content": final_text,
            },
        ]

        return {
            **state,
            "messages": messages,
            "final_answer": final_text,
        }
        
    def _route_after_llm(self, state: AgentState) -> str:
        """
        LangGraph 条件边。

        LLM 输出 FINAL -> final
        LLM 输出 ACTION -> tool
        达到 max_steps -> final
        """
        parsed = state["last_parsed"]

        if state["step_id"] >= state["max_steps"]:
            return "final"

        if parsed and parsed.get("type") == "action":
            return "tool"

        return "final"
    
    def _build_graph(self):
        """
        构造 LangGraph。

        图结构：
        START -> bootstrap -> llm -> tool -> llm -> ... -> final -> END
        """
        graph = StateGraph(AgentState)

        graph.add_node("bootstrap", self._bootstrap_node)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("final", self._final_node)

        graph.add_edge(START, "bootstrap")
        graph.add_edge("bootstrap", "llm")

        graph.add_conditional_edges(
            "llm",
            self._route_after_llm,
            {
                "tool": "tool",
                "final": "final",
            },
        )

        graph.add_edge("tool", "llm")
        graph.add_edge("final", END)

        return graph.compile()

    def run(self, user_input: str) -> str:
        """
        对外接口保持不变：main.py 仍然可以 agent.run(user_input)。
        """
        init_state: AgentState = {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            "session_id": self.session_id,
            "step_id": 0,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "executed_actions": [],
            "final_answer": None,
            "last_raw_output": None,
            "last_parsed": None,
        }

        final_state = self.graph.invoke(init_state)
        # 保存最终状态中的消息历史。
        # main.py 会通过 agent.messages 打印完整调用过程。
        self.messages = final_state["messages"]
        return final_state["final_answer"] or "No final answer."
    
    # def step(self):
    #     prompt_messages = trim_messages(
    #         self.messages,
    #         max_context_chars=8000,
    #         max_tool_chars=3000,
    #     )
    #     prompt = build_prompt(prompt_messages)
        
    #     # 第一版先固定每个 Agent step 的最大输出长度。
    #     # 后续可以根据 task_type 调整：
    #     # plan 短一点，tool_reason 中等，final_answer 长一点。
    #     max_tokens = self.max_tokens
        
    #     # Agent-aware 调度：每一次模型调用都生成一个 step metadata。
    #     # 这就是 Agent 层和推理调度层之间的桥。
    #     metadata = self._build_step_metadata(prompt, max_tokens)
        
    #     output = self.backend.generate_text(
    #         prompt,
    #         max_tokens=max_tokens,
    #         metadata=metadata,
    #     )
        
    #     parsed = parse_agent_output(output)
        
    #     # Agent-aware 调度：一次 LLM 调用结束后 step_id 自增。
    #     # 注意工具调用不增加 step_id，只有调用模型才算一个 Agent step。
    #     self.step_id += 1
        
    #     return output, parsed
    
    # def run(self, user_input: str) -> str:
    #     """
    #     对外接口保持不变：main.py 仍然可以 agent.run(user_input)。
    #     """
    #     self.add_message("user", user_input)

    #     lowered = user_input.lower()
    #     file_path = self._extract_file_path(user_input)
    #     if file_path and any(word in lowered for word in ["read", "summarize", "inspect", "open"]):
    #         tool_result = TOOLS["read_file"](file_path)
    #         self.executed_actions.add(("read_file", file_path))
    #         self.add_message(
    #             "tool",
    #             f"TOOL_RESULT for read_file:\nINPUT: {file_path}\nOUTPUT:\n{tool_result}"
    #         )

    #     for _ in range(self.max_steps):
    #         raw_output, parsed = self.step()

    #         if parsed["type"] == "final":
    #             self.add_message("assistant", parsed["content"])
    #             return parsed["content"]
            
    #         if parsed["type"] == "action":
    #             action = parsed["action"]
    #             action_input = parsed["action_input"]

    #             # 保留模型原始输出，方便后面调试
    #             self.add_message("assistant", raw_output)

    #             if action not in TOOLS:
    #                 self.add_message("tool", f"TOOL_RESULT for {action}:\nINPUT: {action_input}\nERROR:\nunknown tool")
    #                 continue
                
    #             # 用 action + input 作为唯一键
    #             action_key = (action, action_input)

    #             # 如果重复调用同一个工具和同一个输入，不再重复执行
    #             if action_key in self.executed_actions:
    #                 self.add_message(
    #                     "tool",
    #                     f"TOOL_RESULT for {action}:\nINPUT: {action_input}\nERROR:\nrepeated action blocked"
    #                 )
    #                 continue

    #             # 第一次执行该工具
    #             self.executed_actions.add(action_key)
    #             tool_result = TOOLS[action](action_input)

    #              # 更明确地告诉模型这是工具结果
    #             self.add_message(
    #                 "tool",
    #                 f"TOOL_RESULT for {action}:\nINPUT: {action_input}\nOUTPUT:\n{tool_result}"
    #             )
    #             continue
                
    #     final_text = "Reached max steps without FINAL answer."
    #     self.add_message("assistant", final_text)
    #     return final_text
