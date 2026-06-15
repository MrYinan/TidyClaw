#!/usr/bin/env python3
"""Return the current household service robot status for OpenClaw tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.robot_tool_state import build_robot_status, print_json  # noqa: E402
from scripts.runtime_config import apply_runtime_environment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Return current robot cleaner status.")
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply_runtime_environment(override_existing=True)
    result = build_robot_status()
    print_json(result, compact=args.format == "compact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
