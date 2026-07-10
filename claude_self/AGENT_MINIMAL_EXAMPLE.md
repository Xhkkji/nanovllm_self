# Claude Self 最小 Agent 框架样例

这份文档给出一个“最小可跑”的 agent 组织方式，目标是先把 `nanovllm_self` 当作推理内核用起来，不修改现有 `nanovllm` 主逻辑。

## 面向复杂系统的目录设计

如果后面 `claude_self` 确实要往复杂系统上靠，我仍然建议：

- 继续放在 `nanovllm_self` 仓库内
- 继续和 `nanovllm/` 并列
- 但把它当成“未来可以独立拆出去的子系统”来设计

也就是说，当前更适合的是“单仓双子系统”，而不是一开始就拆成两个 repo。

推荐的中期目录结构可以长这样：

```text
nanovllm_self/
├── nanovllm/                 # 推理内核
├── claude_self/              # agent 系统
│   ├── AGENT_MINIMAL_EXAMPLE.md
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── nanovllm_backend.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── loop.py
│   │   ├── parser.py
│   │   └── state.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompt.py
│   │   └── builders.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py
│   │   ├── file_tools.py
│   │   └── shell_tools.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── history.py
│   │   └── summarizer.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── session.py
│   ├── demos/
│   │   └── demo_simple.py
│   └── tests/
│       ├── test_parser.py
│       ├── test_tools.py
│       └── test_agent_loop.py
├── tests/
└── main.py
```

这个结构的关键点不是“文件多”，而是职责边界清楚：

- `nanovllm/` 只做推理
- `claude_self/backend/` 只做后端适配
- `claude_self/agent/` 只做 agent 循环和状态管理
- `claude_self/tools/` 只做工具系统
- `claude_self/prompts/` 只做 prompt 组织
- `claude_self/memory/` 只做上下文和摘要
- `claude_self/runtime/` 只做运行时配置、日志、会话

### 为什么现在不建议单独拆仓

你当前还在高频联调：

- prompt
- 推理接口
- sampling
- decode
- 工具调用
- agent loop

这个阶段如果把 agent 独立成另一个仓库，成本会很高：

- 改接口更麻烦
- 调试链更长
- 两边版本同步更累
- 做实验时来回跳工程

所以当前阶段更适合：

- 仓库内并列
- 代码上分层
- 接口上收口

### 真正要守住的边界

最重要的不是“目录是不是并列”，而是 `claude_self` 不要直接依赖 `nanovllm` 的内部细节。

推荐依赖方向：

```text
claude_self
  -> backend adapter
  -> nanovllm.llm / LLM_self 的公开接口
  -> encode / generate / decode
```

尽量避免变成：

```text
claude_self
  -> scheduler
  -> Sequence
  -> Context
  -> block_manager
  -> model_runner
```

前者后面容易维护，也容易拆分。

后者后面会越来越难改。

### 建议的演进节奏

可以按下面这个节奏长：

1. 第一阶段：最小可跑
2. 第二阶段：拆出 `backend / agent / tools`
3. 第三阶段：增加 `memory / runtime / tests`
4. 第四阶段：如果后端不止 `nanovllm_self`，再考虑独立成单独项目

换句话说，现在先按“同仓独立子系统”设计，是最稳的。

## 目录建议

建议先把 agent 逻辑单独放在 `claude_self/` 目录下，和 `nanovllm/` 并列。

一个最小结构可以是：

```text
claude_self/
├── AGENT_MINIMAL_EXAMPLE.md
├── backend.py
├── simple_agent.py
├── tools.py
├── prompt.py
├── parser.py
└── demo.py
```

如果你想更简单，第一版也可以只保留：

```text
claude_self/
├── backend.py
├── simple_agent.py
└── tools.py
```

把 `prompt` 和 `parser` 暂时写进 `simple_agent.py`。

---

## 整体思路

建议把整个系统拆成三层：

1. `nanovllm_self`：只负责推理
2. `backend`：把推理器包装成“文本输入 -> 文本输出”接口
3. `agent`：负责 prompt 拼装、工具调用、循环控制

也就是说：

- `nanovllm_self` 不关心 agent
- `agent` 不关心底层 paged attention / scheduler / context
- 中间用一个很薄的 `backend` 连接

---

## 一、backend.py

这一层负责把 `LLM_self` 包成一个简单接口：

```python
import torch
from nanovllm.llm import LLM_self
from nanovllm.sampling_params import SamplingParams


class NanoLLMBackend:
    def __init__(self, model_path="/home/xhk/model/Qwen3-0.6B/"):
        self.llm = LLM_self(model=model_path)

    def generate_text(self, prompt: str, max_tokens: int = 256) -> str:
        # 1. 编码
        encoded = self.llm.engine.encoder(prompt)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)

        # 2. 当前建议先用 greedy，方便调 agent
        sampling_params = [
            SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            )
        ]

        # 3. 调用 nanovllm_self
        output_token_ids = self.llm.engine.generate(
            input_ids,
            sampling_params=sampling_params,
        )

        # 4. 只取新生成部分
        full_ids = output_token_ids[0]
        prompt_len = len(encoded["input_ids"])
        new_ids = full_ids[prompt_len:]

        text = self.llm.engine.decode(new_ids)
        return text.strip()
```

### 说明

这一层的核心目标只有一个：

- 上层 agent 只调用 `generate_text(prompt)`

这样以后你即使换 sampler、换模型、换 prompt template，agent 层都不用动太多。

---

## 二、tools.py

第一版工具不要太多，先做两个最基本的：

- `read_file`
- `run_shell`

```python
import subprocess
from pathlib import Path


def read_file(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if not p.exists():
        return f"[read_file] 文件不存在: {path}"
    if not p.is_file():
        return f"[read_file] 不是普通文件: {path}"

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"[read_file] 读取失败: {e}"

    return text[:max_chars]


def run_shell(cmd: str, timeout: int = 10, max_chars: int = 4000) -> str:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return output[:max_chars]
    except Exception as e:
        return f"[run_shell] 执行失败: {e}"


TOOLS = {
    "read_file": read_file,
    "run_shell": run_shell,
}
```

### 说明

第一版重点是“工具闭环跑通”，不是功能多。

所以建议：

- 每个工具都限制输出长度
- 不要一开始就接太多工具
- 先保证 agent 能稳定执行 1 到 2 个工具

---

## 三、parser.py

先不要上复杂 JSON schema，直接用最简单的文本协议。

```python
def parse_agent_output(text: str):
    text = text.strip()

    if text.startswith("FINAL:"):
        return {
            "type": "final",
            "content": text[len("FINAL:"):].strip()
        }

    lines = text.splitlines()
    action = None
    action_input = None

    for line in lines:
        if line.startswith("ACTION:"):
            action = line[len("ACTION:"):].strip()
        elif line.startswith("ACTION_INPUT:"):
            action_input = line[len("ACTION_INPUT:"):].strip()

    if action is not None:
        return {
            "type": "action",
            "action": action,
            "action_input": action_input or ""
        }

    # 如果格式没对上，先退化成最终回答
    return {
        "type": "final",
        "content": text
    }
```

### 说明

第一版推荐只支持两种模型输出：

```text
ACTION: read_file
ACTION_INPUT: /home/xhk/nanovllm_self/main.py
```

或者：

```text
FINAL: 这是最终答案
```

这样最容易调试。

---

## 四、prompt.py

先做最简单的纯文本 prompt 拼接。

```python
SYSTEM_PROMPT = """你是一个本地代码助手。
你可以回答问题，也可以请求使用工具。

如果你需要调用工具，请严格输出以下格式：
ACTION: <tool_name>
ACTION_INPUT: <input>

如果你已经可以直接回答，请严格输出以下格式：
FINAL: <你的回答>

可用工具：
1. read_file
2. run_shell

不要输出多余解释。
"""


def build_prompt(messages):
    parts = [SYSTEM_PROMPT, ""]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{role.upper()}: {content}")

    parts.append("ASSISTANT:")
    return "\n".join(parts)
```

### 说明

这里的消息先统一用这种结构：

```python
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
{"role": "tool", "content": "..."}
```

这样足够简单，也方便以后扩展。

---

## 五、simple_agent.py

这是 agent 的主循环。

```python
from prompt import build_prompt
from parser import parse_agent_output
from tools import TOOLS


class SimpleAgent:
    def __init__(self, backend, max_steps: int = 5):
        self.backend = backend
        self.max_steps = max_steps
        self.messages = []

    def add_user_message(self, text: str):
        self.messages.append({
            "role": "user",
            "content": text,
        })

    def add_assistant_message(self, text: str):
        self.messages.append({
            "role": "assistant",
            "content": text,
        })

    def add_tool_message(self, text: str):
        self.messages.append({
            "role": "tool",
            "content": text,
        })

    def step(self):
        prompt = build_prompt(self.messages)
        output = self.backend.generate_text(prompt, max_tokens=256)
        parsed = parse_agent_output(output)
        return output, parsed

    def run(self, user_input: str) -> str:
        self.add_user_message(user_input)

        for _ in range(self.max_steps):
            raw_output, parsed = self.step()

            if parsed["type"] == "final":
                self.add_assistant_message(parsed["content"])
                return parsed["content"]

            if parsed["type"] == "action":
                action = parsed["action"]
                action_input = parsed["action_input"]

                if action not in TOOLS:
                    error_text = f"[tool_error] 未知工具: {action}"
                    self.add_tool_message(error_text)
                    continue

                tool_result = TOOLS[action](action_input)
                self.add_assistant_message(raw_output)
                self.add_tool_message(tool_result)
                continue

        final_text = "达到最大步骤数，任务结束。"
        self.add_assistant_message(final_text)
        return final_text
```

### 说明

这个循环的本质是：

1. 把历史拼成 prompt
2. 让模型决定是直接回答还是调用工具
3. 如果调用工具，就执行工具
4. 把工具结果写回历史
5. 继续下一轮

这是最基础的 agent 闭环。

---

## 六、demo.py

一个最小入口示例：

```python
from backend import NanoLLMBackend
from simple_agent import SimpleAgent


def main():
    backend = NanoLLMBackend()
    agent = SimpleAgent(backend, max_steps=5)

    user_input = "请查看 /home/xhk/nanovllm_self/main.py，并告诉我这个程序主流程在做什么。"
    result = agent.run(user_input)

    print("===== FINAL RESULT =====")
    print(result)


if __name__ == "__main__":
    main()
```

---

## 七、推荐的第一版工作顺序

建议你按下面顺序做：

1. 先写 `backend.py`
2. 再写 `tools.py`
3. 再写 `simple_agent.py`
4. 先手动构造一个 prompt，看看模型能不能稳定输出 `FINAL:`
5. 再让模型尝试输出 `ACTION:`
6. 最后接上完整 agent loop

这样定位问题最清楚。

---

## 八、第一版容易踩的坑

### 1. 不要先开采样

建议第一版固定：

```python
temperature = 0.0
```

因为 agent 输出格式很脆弱，采样一开，很容易把：

- `ACTION:`
- `ACTION_INPUT:`
- `FINAL:`

这些结构打乱。

### 2. 工具输出不要太长

否则上下文会很快膨胀。

### 3. 一定限制最大轮数

例如：

```python
max_steps = 5
```

不然模型可能会无休止地调用工具。

### 4. 先不要追求“像 Claude Code 一样复杂”

第一版只要做到：

- 能看文件
- 能跑命令
- 能基于结果继续思考
- 能输出最终答案

就已经是一个完整的最小 agent 了。

---

## 九、后续可以怎么升级

当最小版跑通以后，可以逐步加：

1. 多轮对话上下文
2. 更清晰的工具输出格式
3. `THOUGHT / ACTION / OBSERVATION / FINAL` 协议
4. 更多工具
5. 文件写入能力
6. 流式输出
7. 更正式的 parser

但建议先把“最小闭环”做稳。

---

## 十、一句话总结

你现在最合适的做法是：

- `nanovllm_self` 继续只做推理内核
- `claude_self/` 单独负责 agent 逻辑
- 用一个很薄的 `backend` 把两边连起来

这样结构最清楚，也最方便你后面继续演进。
