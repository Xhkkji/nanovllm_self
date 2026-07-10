import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend import NanovllmBackend
from agent import Agent


def main():
    backend = NanovllmBackend(enable_profile=False)
    agent = Agent(backend, max_steps=5)

    # 第一阶段先测 FINAL 协议
    # user_input = "What is an llm? Answer in one sentence."
    user_input = "Read /home/xhk/nanovllm_self/main.py and summarize it."
    result = agent.run(user_input)

    print("===== FINAL RESULT =====")
    print(result)
    print("\n===== MESSAGE HISTORY =====")
    for msg in agent.messages:
        print(f"{msg['role'].upper()}: {msg['content']}")


if __name__ == "__main__":
    main()