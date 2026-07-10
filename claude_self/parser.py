

def clean_final_content(text: str) -> str:
    """
    从 FINAL: 后面提取干净答案。
    小模型经常在 FINAL 后继续输出 Wait/Okay/But 等废话，这里做最小截断。
    """
    content = text.strip()

    stop_markers = [
        "\nWait,",
        "\nOkay,",
        "\nBut ",
        "\nHowever,",
        "\nACTION:",
        "\nACTION_INPUT:",
        "\nUSER:",
        "\nASSISTANT:",
        "\nTOOL:",
    ]

    for marker in stop_markers:
        idx = content.find(marker)
        if idx != -1:
            content = content[:idx].strip()

    return content


def parse_agent_output(text: str):
    """
    解析模型输出。

    支持两种协议：
    1. FINAL: ...
    2. ACTION: ...
       ACTION_INPUT: ...
    """
    text = text.strip()

    # 只要出现FINAL，优先认为这是最终答案
    final_idx = text.find("FINAL:")
    if final_idx != -1:
        final_content = text[final_idx + len("FINAL:"):].strip()
        final_content = clean_final_content(final_content)
        return {
            "type": "final",
            "content": final_content,
        }

    
    # text = "第一行\n第二行\n第三行"
    # lines = text.splitlines()
    # print(lines)  # ['第一行', '第二行', '第三行']
    lines = text.splitlines()

    # ACTION 协议（执行动作）
    # 模型决定调用某个工具/函数：
    # text
    # ACTION: get_weather
    # ACTION_INPUT: {"city": "北京"}

    # 解析后：
    # {
    # "type": "action",
    # "action": "get_weather",
    # "action_input": "{\"city\": \"北京\"}"
    # }
    action = None
    action_input = None

    for line in lines:
        if line.startswith("ACTION:"):
            action = line[len("ACTION:"):].strip()
        elif line.startswith("ACTION_INPUT:"):
            action_input = line[len("ACTION_INPUT:"):].strip()
    
    invalid_actions = {"", "none", "null", "nil"}
    if action is not None:
        if action.lower() in invalid_actions:
            return {
                "type": "final",
                "content": text
            }

        return {
            "type": "action",
            "action": action,
            "action_input":action_input or "",
        }

    # 如果模型没有严格遵守格式，先退化成最终输出
    # FINAL 协议（最终答案）
    # 模型输出最终结果给用户：
    # text
    # FINAL: 北京的天气是晴天，温度25°C
    return {
        "type": "final",
        "content": clean_final_content(text),
    }

