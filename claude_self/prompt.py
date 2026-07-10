SYSTEM_PROMPT = """You are a local coding assistant.

You may respond in only one of two ways:

Way 1:
Start your response with FINAL:
Then give the final answer in one short paragraph.

Way 2:
If you need a tool, write exactly two lines:
ACTION: tool_name
ACTION_INPUT: tool_input

Rules:
1. Never output both FINAL and ACTION in the same response.
2. Never output ACTION: None.
3. Never explain your reasoning.
4. Never output placeholder text.
5. Never copy instruction words like tool_name, tool_input, or final answer.
6. If the user asks to read, inspect, open, or summarize a file path, you must call read_file first.
7. Do not answer file-related questions from memory.
8. If a TOOL_RESULT message is present, it is the true result of your previous tool call.
9. After receiving TOOL_RESULT, give a FINAL answer unless another different tool is truly necessary.
10. Do not repeat the same ACTION with the same ACTION_INPUT.

Available tools:
- read_file
- run_shell
"""

def build_prompt(messages):
    """
    把消息历史拼成一个纯文本 prompt。
    当前先不用复杂 chat template，最简单最稳。
    """
    parts = [SYSTEM_PROMPT, ""]

    # messages = [
#     {"role": "user", "content": "What is an llm?"},]
    
    # 比如：
#     messages = [
#     {"role": "user", "content": "Read /home/xhk/nanovllm_self/main.py"},
#     {"role": "assistant", "content": "ACTION: read_file\nACTION_INPUT: /home/xhk/nanovllm_self/main.py"},
#     {"role": "tool", "content": "import torch\nfrom nanovllm.llm import LLM_self\n..."},
# ]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"{role.upper()}: {content}")
    
    parts.append("ASSISTANT:")
    return "\n".join(parts)

# 返回值：
#     You are a local coding assistant.

# You must follow these rules:
# 1. If you can answer directly, output exactly:
# FINAL: <your answer>

# 2. If you need a tool, output exactly:
# ACTION: <tool_name>
# ACTION_INPUT: <input>

# 3. Do not output anything else.
# 4. Keep your answer short and structured.

# Available tools:
# - read_file
# - run_shell

# USER: What is an llm?
# ASSISTANT:（这里表示轮到模型回答了）