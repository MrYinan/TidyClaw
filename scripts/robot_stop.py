#!/usr/bin/env python3
"""Request a safe stop for the household service robot mission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.robot_tool_state import MEMORY_DIR, StateManager, build_robot_report, build_robot_status, print_json, runner_process_summary  # noqa: E402


def safe_reason(value: str) -> str:
    text = str(value or "user_stop").strip()
    if not text or any(char in text for char in "\r\n\t"):
        return "user_stop"
    return text[:200]


def request_stop(reason: str, *, memory_dir: Path = MEMORY_DIR) -> dict[str, Any]:
    manager = StateManager(memory_dir)
    state = manager.stop_mission(reason=safe_reason(reason))
    status_payload = build_robot_status(memory_dir)
    report_payload = build_robot_report(memory_dir)
    runner = runner_process_summary()
    return {
        "status": "success",
        "result_type": "robot_cleaner_stop_requested",
        "schema": "robot_cleaner_stop_v1",
        "reason": safe_reason(reason),
        "stop_state": "stop_pending" if runner.get("alive") else "stopped",
        "message": "mission disabled; runner will stop after its current safe check if it is running",
        "runner": runner,
        "state": {
            "mission_enabled": state.mission.get("enabled"),
            "mission_mode": state.mission.get("mode"),
            "patrol_enabled": state.patrol.get("enabled"),
            "patrol_mode": state.patrol.get("mode"),
            "step_count": state.patrol.get("step_count"),
            "room_complete": state.room.get("room_complete"),
        },
        "status_after_stop": status_payload,
        "report": report_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Request a safe stop for robot cleaner.")
    parser.add_argument("--reason", default="user_stop")
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = request_stop(args.reason)
    print_json(result, compact=args.format == "compact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
