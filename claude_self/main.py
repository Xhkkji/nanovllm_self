import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend import NanovllmBackend
from agent import Agent
from scheduled_backend import ScheduledBackend


def parse_args():
    """解析 Claude self demo 的最小运行参数。"""
    parser = argparse.ArgumentParser(description="Run the minimal claude_self demo.")
    parser.add_argument(
        "--prompt",
        default="Read /home/xhk/nanovllm_self/main.py and summarize it.",
        help="User input passed to the demo agent.",
    )
    parser.add_argument(
        "--session-id",
        default="demo-agent-000001",
        help="Agent session id used for route tracking.",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--scheduler", default="affinity_load_aware")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--initial-backlog-s", default="0,0")
    
    parser.add_argument(
        "--backend",
        choices=["local", "scheduled", "pd_pool"],
        default="scheduled",
        help="Backend used by the agent.",
    )
    parser.add_argument("--prefill-work-dirs", default="")
    parser.add_argument("--decode-work-dirs", default="")
    parser.add_argument("--prefill-global-ranks", default="")
    parser.add_argument("--decode-global-ranks", default="")
    
    return parser.parse_args()

def parse_csv_str(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_int(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    args = parse_args()

    if args.backend == "local":
        backend = NanovllmBackend(enable_profile=False)

    elif args.backend == "scheduled":
        local_backend = NanovllmBackend(enable_profile=False)
        backend = ScheduledBackend(
            inner_backend=local_backend,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            initial_backlog_s=args.initial_backlog_s,
        )

    elif args.backend == "pd_pool":
        from pd_pool_backend import PDPoolBackend

        backend = PDPoolBackend(
            prefill_work_dirs=parse_csv_str(args.prefill_work_dirs),
            decode_work_dirs=parse_csv_str(args.decode_work_dirs),
            prefill_global_ranks=parse_csv_int(args.prefill_global_ranks),
            decode_global_ranks=parse_csv_int(args.decode_global_ranks),
            scheduler=args.scheduler,
            initial_backlog_s=args.initial_backlog_s,
        )

    agent = Agent(
        backend,
        max_steps=args.max_steps,
        session_id=args.session_id,
        max_tokens=args.max_tokens,
    )

    result = agent.run(args.prompt)

    print("===== FINAL RESULT =====")
    print(result)

    print("\n===== ROUTE HISTORY =====")
    if hasattr(backend, "route_history"):
        for item in backend.route_history:
            print(item)

    print("\n===== MESSAGE HISTORY =====")
    for msg in agent.messages:
        print(f"{msg['role'].upper()}: {msg['content']}")


if __name__ == "__main__":
    main()
