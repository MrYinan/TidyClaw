#!/usr/bin/env python3
"""Prepare one bounded decision turn for OpenClaw tool control.

This script refreshes online perception and then builds
``robot_cleaner_decision_context_v1``. It is the backend for
``robot_cleaner_prepare_decision_turn`` and intentionally reuses the existing
stable skill scripts instead of duplicating vision or YOLO logic here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_option import (
    MEMORY_DIR,
    ScriptResult,
    json_print,
    now_iso,
    run_observe_refresh,
    run_script,
    script_result_payload,
)
from scripts.local_costmap import LocalCostmap
from scripts.option_state_sync import ensure_tool_mission_active


JsonDict = dict[str, Any]
DECISION_CONTEXT_SCRIPT = REPO_ROOT / "scripts" / "decision_context_builder.py"
DEFAULT_CONTEXT_PATH = MEMORY_DIR / "decision-context.json"
DEFAULT_PERCEPTION_PATH = MEMORY_DIR / "yolo-current-rgbd.json"
TRACE_PATH = MEMORY_DIR / "decision-turn-trace.jsonl"


def append_trace(event: JsonDict, *, trace_path: Path = TRACE_PATH) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), "time": now_iso(), **event}
    with trace_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def read_json_file(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def summarize_local_costmap(costmap: JsonDict) -> JsonDict:
    action_safety = costmap.get("action_safety") if isinstance(costmap.get("action_safety"), dict) else {}
    moveahead = action_safety.get("MoveAhead") if isinstance(action_safety.get("MoveAhead"), dict) else {}
    return {
        "status": costmap.get("status"),
        "result_type": costmap.get("result_type"),
        "last_updated_at": costmap.get("last_updated_at"),
        "holding_object": costmap.get("holding_object"),
        "held_object_labels": costmap.get("held_object_labels"),
        "front_clearance_m": costmap.get("front_clearance_m"),
        "left_clearance_m": costmap.get("left_clearance_m"),
        "right_clearance_m": costmap.get("right_clearance_m"),
        "moveahead_safe": moveahead.get("safe"),
        "moveahead_reason": moveahead.get("reason"),
        "moveahead_confidence": moveahead.get("confidence"),
        "moveahead_observed_ratio": moveahead.get("observed_ratio"),
        "blocked_actions": costmap.get("blocked_actions") or [],
    }


def update_local_costmap_from_refresh(refresh: JsonDict) -> JsonDict:
    vision = refresh.get("vision") if isinstance(refresh.get("vision"), dict) else {}
    perception = refresh.get("perception") if isinstance(refresh.get("perception"), dict) else {}
    vision_data = vision.get("data") if isinstance(vision.get("data"), dict) else {}
    perception_data = perception.get("data") if isinstance(perception.get("data"), dict) else {}
    service_state = read_json_file(MEMORY_DIR / "service-task-state.json")
    patrol_state = read_json_file(MEMORY_DIR / "patrol-state.json")
    holding_object = bool(service_state.get("holding_object") or perception_data.get("holding_object_context"))
    held_labels = service_state.get("held_object_labels")
    if not isinstance(held_labels, list):
        held_labels = service_state.get("held_object_label") or perception_data.get("held_object_labels") or []
    if isinstance(held_labels, str):
        held_labels = [held_labels]
    step = patrol_state.get("step_count")
    if not isinstance(step, int):
        step = service_state.get("step_count") if isinstance(service_state.get("step_count"), int) else None
    costmap = LocalCostmap(MEMORY_DIR).update(
        vision=vision_data,
        analysis=perception_data,
        holding_object=holding_object,
        held_object_labels=[str(item) for item in held_labels if str(item).strip()],
        step=step,
        persist=True,
    )
    return summarize_local_costmap(costmap)


def build_context_command_args(args: argparse.Namespace, output_path: Path) -> list[str]:
    return [
        "--task-mode",
        str(args.task_mode),
        "--output",
        str(output_path),
        "--max-candidates",
        str(max(1, int(args.max_candidates))),
        "--max-options",
        str(max(1, int(args.max_options))),
        "--format",
        "compact",
    ]


def summarize_refresh(refresh: JsonDict) -> JsonDict:
    vision = refresh.get("vision") if isinstance(refresh.get("vision"), dict) else {}
    perception = refresh.get("perception") if isinstance(refresh.get("perception"), dict) else {}
    vision_data = vision.get("data") if isinstance(vision.get("data"), dict) else {}
    perception_data = perception.get("data") if isinstance(perception.get("data"), dict) else {}
    return {
        "status": refresh.get("status"),
        "result_type": refresh.get("result_type"),
        "vision_status": vision_data.get("status"),
        "vision_result_type": vision_data.get("result_type"),
        "image_path": vision_data.get("image_path"),
        "depth_path": vision_data.get("depth_path"),
        "perception_status": perception_data.get("status"),
        "perception_result_type": perception_data.get("result_type"),
        "perception_backend": perception_data.get("perception_backend"),
        "candidate_count": perception_data.get("candidate_count"),
        "pickup_target_detected": perception_data.get("pickup_target_detected"),
        "place_receptacle_detected": perception_data.get("place_receptacle_detected"),
        "frontier_exists": perception_data.get("frontier_exists"),
        "recommended_action": perception_data.get("recommended_action"),
        "perception_written": refresh.get("perception_written"),
    }


def build_decision_context(args: argparse.Namespace, output_path: Path) -> ScriptResult:
    return run_script(
        DECISION_CONTEXT_SCRIPT,
        build_context_command_args(args, output_path),
        timeout_seconds=max(1, int(args.timeout)),
    )


def prepare_decision_turn(args: argparse.Namespace) -> JsonDict:
    output_path = resolve_workspace_path(args.output)
    attempts: list[JsonDict] = []
    refresh: JsonDict = {}
    mission_activation = ensure_tool_mission_active(memory_dir=MEMORY_DIR, mode="SERVICE")

    for attempt_index in range(1, max(0, int(args.observe_retries)) + 2):
        refresh = run_observe_refresh(timeout_seconds=max(1, int(args.timeout)))
        attempts.append(summarize_refresh(refresh))
        if refresh.get("status") == "success":
            break

    if refresh.get("status") != "success":
        result = {
            "status": "error",
            "result_type": "decision_turn_prepare_failed",
            "stage": "observe_refresh",
            "mission_activation": mission_activation,
            "attempts": attempts,
            "required_next": "retry_prepare_decision_turn_or_stop",
        }
        append_trace({"event": "decision_turn_prepare_failed", "result": result})
        return result

    local_costmap: JsonDict
    try:
        local_costmap = update_local_costmap_from_refresh(refresh)
    except Exception as exc:
        local_costmap = {
            "status": "error",
            "result_type": "local_costmap_update_failed",
            "message": str(exc),
        }

    context_result = build_decision_context(args, output_path)
    context = context_result.data if isinstance(context_result.data, dict) else {}
    if context_result.returncode != 0 or context.get("status") != "success":
        result = {
            "status": "error",
            "result_type": "decision_turn_prepare_failed",
            "stage": "build_decision_context",
            "mission_activation": mission_activation,
            "observe_refresh": attempts[-1] if attempts else {},
            "local_costmap": local_costmap,
            "context_builder": script_result_payload(context_result),
            "required_next": "inspect_decision_context_builder",
        }
        append_trace({"event": "decision_turn_prepare_failed", "result": result})
        return result

    option_set = context.get("option_set") if isinstance(context.get("option_set"), dict) else {}
    result = {
        "status": "success",
        "result_type": "decision_turn_prepared",
        "schema": "robot_cleaner_decision_turn_v1",
        "prepared_at": now_iso(),
        "mission_activation": mission_activation,
        "context_path": display_path(output_path),
        "perception_path": display_path(DEFAULT_PERCEPTION_PATH),
        "observe_refresh": attempts[-1] if attempts else {},
        "local_costmap": local_costmap,
        "decision_context": context,
        "option_set": option_set,
        "model_decision_required": True,
        "rule_baseline_option_id": option_set.get("rule_baseline_option_id"),
        "option_count": len(option_set.get("options") or []) if isinstance(option_set.get("options"), list) else 0,
        "consistency_warnings": context.get("consistency_warnings") or [],
    }
    append_trace(
        {
            "event": "decision_turn_prepared",
            "status": "success",
            "model_decision_required": True,
            "rule_baseline_option_id": result.get("rule_baseline_option_id"),
            "option_count": result.get("option_count"),
            "context_path": result.get("context_path"),
            "mission_activation": mission_activation,
            "local_costmap": result.get("local_costmap"),
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one OpenClaw robot decision turn.")
    parser.add_argument("--task-mode", choices=("auto", "tidy", "clean"), default="tidy")
    parser.add_argument("--output", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per backend script in seconds.")
    parser.add_argument(
        "--observe-retries",
        type=int,
        default=1,
        help="Retry observe/perception this many times before failing the turn.",
    )
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_decision_turn(args)
    json_print(result, compact=args.format == "compact")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
