

def shorten_text(text: str, max_chars: int) -> str:
    """
    保留文本前后，中间省略。
    对代码文件比只保留开头更有用。
    """
    if len(text) <= max_chars:
        return text
    
    half = max_chars // 2

    return (
        text[:half]
        + "\n\n...[truncated]...\n\n"
        + text[-half:]
    )

def normalize_message_for_prompt(msg: dict, max_tool_chars: int = 3000) -> dict:
    """
    给 prompt 使用的 message 版本。
    不改变原始 messages，只生成裁剪后的副本。
    """
    role = msg["role"]
    content = msg["content"]

    # tool 内容最容易撑爆上下文，优先截断
    if role == "tool":
        content = shorten_text(content, max_tool_chars)
    return {
        "role": role,
        "content": content,
    }

def _tool_key(content: str):
    if not content.startswith("TOOL_RESULT for "):
        return None

    lines = content.splitlines()
    action = None
    action_input = None

    for line in lines:
        if line.startswith("TOOL_RESULT for "):
            action = line[len("TOOL_RESULT for "):].strip().rstrip(":")
        elif line.startswith("INPUT:"):
            action_input = line[len("INPUT:"):].strip()

    if action is None:
        return None
    return (action, action_input or "")


def trim_messages(messages: list[dict], max_context_chars: int = 8000, max_tool_chars: int = 3000) -> list[dict]:
    kept = []
    total = 0
    seen_tool_keys = set()

    for msg in reversed(messages):
        normalized_msg = normalize_message_for_prompt(msg, max_tool_chars=max_tool_chars)

        if normalized_msg["role"] == "tool":
            key = _tool_key(normalized_msg["content"])
            if key is not None:
                if key in seen_tool_keys:
                    continue
                seen_tool_keys.add(key)

        msg_len = len(f"{normalized_msg['role'].upper()}: {normalized_msg['content']}\n")
        if total + msg_len > max_context_chars:
            break

        kept.append(normalized_msg)
        total += msg_len

    kept.reverse()
    return kept
