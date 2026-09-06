from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimization scaffold entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("candidates", help="List candidate definitions.")
    subparsers.add_parser("benchmark", help="Run benchmark cases.")
    subparsers.add_parser("traffic", help="Generate traffic datasets.")
    subparsers.add_parser("loop", help="Run the optimization loop.")
    subparsers.add_parser("report", help="Summarize benchmark results.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        f"Optimization scaffold only. Command '{args.command}' is not implemented yet."
    )


if __name__ == "__main__":
    main()
