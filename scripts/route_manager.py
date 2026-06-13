#!/usr/bin/env python3
"""Committed route-step manager for frontier exploration.

This module bridges the gap between a long-lived active frontier goal and the
LLM-facing one-step option contract.  It is intentionally stateful only in
``memory/room-state.json``; map writing remains owned by the existing
action-odometry/navigation memory stack.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.map_backend import load_map_backend
    from scripts.position_map_core import (
        HEADING_ORDER,
        heading_between,
        left_heading,
        manhattan_distance,
        neighbor_for_action,
        parse_cell,
        right_heading,
    )
except ImportError:  # pragma: no cover - direct script execution
    from map_backend import load_map_backend
    from position_map_core import (
        HEADING_ORDER,
        heading_between,
        left_heading,
        manhattan_distance,
        neighbor_for_action,
        parse_cell,
        right_heading,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"

ACTIVE_ROUTE_SCHEMA = "robot_cleaner_active_route_v1"
ROUTE_STEP_SCHEMA = "robot_cleaner_route_step_v1"
ROUTE_UPDATE_SCHEMA = "robot_cleaner_route_update_v1"

TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATION_ACTIONS = {"RotateLeft", "RotateRight"}
ROUTE_BLOCK_PATIENCE = 4

JsonDict = dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def clean_empty(value: Any) -> Any:
    if isinstance(value, dict):
        result: JsonDict = {}
        for key, item in value.items():
            cleaned = clean_empty(item)
            if cleaned in (None, "", [], {}):
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := clean_empty(item)) not in (None, "", [], {})]
    return value


def load_json(path: Path, default: JsonDict | None = None) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return dict(default or {})
    return data if isinstance(data, dict) else dict(default or {})


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_cell_safe(value: Any) -> tuple[int, int] | None:
    try:
        return parse_cell(value)
    except Exception:
        return None


def normalize_cell(value: Any) -> str:
    parsed = parse_cell_safe(value)
    if parsed is None:
        return ""
    return f"{parsed[0]},{parsed[1]}"


def encode_cell(cell: str) -> str:
    text = normalize_cell(cell)
    if not text:
        return "unknown"
    x, z = parse_cell(text)
    return f"x{'m' + str(abs(x)) if x < 0 else x}_z{'m' + str(abs(z)) if z < 0 else z}"


def route_id_for(goal_cell: str, path: Sequence[str]) -> str:
    digest = hashlib.sha1("|".join([goal_cell, *path]).encode("utf-8")).hexdigest()[:8]
    return f"route-frontier-{encode_cell(goal_cell)}-{digest}"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def current_pose(position_status: JsonDict, room: JsonDict) -> tuple[str, str]:
    pose = as_dict(position_status.get("pose"))
    cell = normalize_cell(pose.get("cell") or position_status.get("last_cell") or room.get("last_cell") or "0,0")
    heading = str(pose.get("heading") or position_status.get("last_heading") or room.get("last_heading") or "north")
    if heading not in HEADING_ORDER:
        heading = "north"
    return cell or "0,0", heading


def action_safety_record(costmap: JsonDict, action: str) -> JsonDict:
    return as_dict(as_dict(costmap.get("action_safety")).get(action))


def action_executable(costmap: JsonDict, action: str) -> tuple[bool, str]:
    record = action_safety_record(costmap, action)
    if not record:
        return False, "missing_action_safety"
    if record.get("safe") is not True:
        return False, str(record.get("reason") or "costmap_blocked")
    return True, str(record.get("reason") or "clear_swept_volume")


def opposite_heading(heading: str) -> str:
    if heading not in HEADING_ORDER:
        return "south"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) + 2) % 4]


def adjacent(source: str, target: str) -> bool:
    try:
        return manhattan_distance(source, target) == 1
    except Exception:
        return False


def trim_path_to_current(path: Sequence[Any], *, current_cell: str, goal_cell: str) -> list[str]:
    normalized = [normalize_cell(item) for item in path]
    normalized = [item for item in normalized if item]
    if not normalized:
        return [current_cell, goal_cell] if current_cell != goal_cell and adjacent(current_cell, goal_cell) else [current_cell]
    if current_cell in normalized:
        index = normalized.index(current_cell)
        trimmed = normalized[index:]
    else:
        trimmed = [current_cell, *normalized]
    if trimmed[-1] != goal_cell and goal_cell:
        trimmed.append(goal_cell)
    deduped: list[str] = []
    for cell in trimmed:
        if deduped and deduped[-1] == cell:
            continue
        deduped.append(cell)
    return deduped or [current_cell]


def next_cell_from_path(path: Sequence[str], current_cell: str) -> str:
    if len(path) >= 2 and path[0] == current_cell:
        return path[1]
    if len(path) >= 2 and current_cell in path:
        index = path.index(current_cell)
        if index + 1 < len(path):
            return path[index + 1]
    return ""


def route_step_for_next_cell(
    *,
    route_id: str,
    step_index: int,
    current_cell: str,
    current_heading: str,
    next_cell: str,
    goal_cell: str,
    path: Sequence[str],
) -> JsonDict:
    desired_heading = heading_between(current_cell, next_cell)
    if desired_heading is None:
        return {
            "schema": ROUTE_STEP_SCHEMA,
            "route_id": route_id,
            "step_index": step_index,
            "status": "blocked",
            "reason": "next_cell_not_adjacent",
            "current_cell": current_cell,
            "current_heading": current_heading,
            "next_cell": next_cell,
            "goal_cell": goal_cell,
            "path_remaining": list(path),
        }

    if desired_heading == current_heading:
        action = "MoveAhead"
        target_cell = next_cell
        progress_effect = "advance_to_next_cell"
        turns_remaining = 0
    elif desired_heading == left_heading(current_heading):
        action = "RotateLeft"
        target_cell = current_cell
        progress_effect = "turn_toward_next_cell"
        turns_remaining = 1
    elif desired_heading == right_heading(current_heading):
        action = "RotateRight"
        target_cell = current_cell
        progress_effect = "turn_toward_next_cell"
        turns_remaining = 1
    else:
        action = "RotateLeft"
        target_cell = current_cell
        progress_effect = "turnaround_toward_next_cell"
        turns_remaining = 2

    return clean_empty(
        {
            "schema": ROUTE_STEP_SCHEMA,
            "route_id": route_id,
            "step_index": step_index,
            "status": "active",
            "action": action,
            "current_cell": current_cell,
            "current_heading": current_heading,
            "target_cell": target_cell,
            "next_cell": next_cell,
            "goal_cell": goal_cell,
            "desired_heading": desired_heading,
            "heading_after_action": (
                left_heading(current_heading)
                if action == "RotateLeft"
                else right_heading(current_heading)
                if action == "RotateRight"
                else current_heading
            ),
            "turns_remaining": turns_remaining,
            "progress_effect": progress_effect,
            "path_remaining": list(path),
            "one_step_only": True,
        }
    )


def existing_route_step_index(previous: JsonDict, *, same_route: bool, current_cell: str) -> int:
    if not same_route:
        return 0
    previous_step = as_dict(previous.get("route_step"))
    previous_current = str(previous_step.get("current_cell") or previous.get("current_cell") or "")
    try:
        index = int(previous_step.get("step_index", previous.get("step_index", 0)) or 0)
    except (TypeError, ValueError):
        index = 0
    if previous_current and previous_current != current_cell:
        return index + 1
    return index


def build_active_route(
    *,
    active_goal: JsonDict,
    previous_route: JsonDict,
    current_cell: str,
    current_heading: str,
    costmap: JsonDict,
) -> JsonDict:
    goal_cell = normalize_cell(active_goal.get("cell"))
    if not goal_cell:
        return inactive_route("missing_active_goal_cell", previous_route=previous_route)
    if current_cell == goal_cell:
        return inactive_route("goal_cell_reached", previous_route=previous_route, goal_cell=goal_cell)

    original_path = [normalize_cell(item) for item in as_list(active_goal.get("path"))]
    original_path = [item for item in original_path if item]
    path = trim_path_to_current(original_path, current_cell=current_cell, goal_cell=goal_cell)
    if len(path) < 2:
        return inactive_route("active_goal_path_unavailable", previous_route=previous_route, goal_cell=goal_cell, path=path)

    next_cell = next_cell_from_path(path, current_cell)
    if not next_cell:
        return inactive_route("active_goal_next_cell_unavailable", previous_route=previous_route, goal_cell=goal_cell, path=path)

    candidate_route_id = route_id_for(goal_cell, original_path or path)
    same_route = str(previous_route.get("goal_cell") or "") == goal_cell and str(previous_route.get("route_id") or "") == candidate_route_id
    route_id = str(previous_route.get("route_id") or candidate_route_id) if same_route else candidate_route_id
    step_index = existing_route_step_index(previous_route, same_route=same_route, current_cell=current_cell)
    route_step = route_step_for_next_cell(
        route_id=route_id,
        step_index=step_index,
        current_cell=current_cell,
        current_heading=current_heading,
        next_cell=next_cell,
        goal_cell=goal_cell,
        path=path,
    )
    action = str(route_step.get("action") or "")
    executable, safety_reason = action_executable(costmap, action) if action else (False, "missing_route_step_action")
    if not executable:
        fail_count = safe_int(previous_route.get("fail_count"), 0) + 1 if same_route else 1
        return clean_empty(
            {
                "schema": ACTIVE_ROUTE_SCHEMA,
                "status": "blocked",
                "route_id": route_id,
                "goal_cell": goal_cell,
                "goal_type": active_goal.get("mode") or "frontier_cluster",
                "current_cell": current_cell,
                "current_heading": current_heading,
                "next_cell": next_cell,
                "desired_heading": route_step.get("desired_heading"),
                "next_action": action,
                "path": path,
                "route_step": route_step,
                "fail_count": fail_count,
                "blocked_reason": safety_reason,
                "requires_replan": fail_count >= ROUTE_BLOCK_PATIENCE,
                "updated_at": now_iso(),
            }
        )

    try:
        distance_to_goal = manhattan_distance(current_cell, goal_cell)
    except Exception:
        distance_to_goal = None
    return clean_empty(
        {
            "schema": ACTIVE_ROUTE_SCHEMA,
            "status": "active",
            "route_id": route_id,
            "goal_cell": goal_cell,
            "goal_type": active_goal.get("mode") or "frontier_cluster",
            "cluster_cells": as_list(active_goal.get("cluster_cells")),
            "current_cell": current_cell,
            "current_heading": current_heading,
            "next_cell": next_cell,
            "next_action": action,
            "desired_heading": route_step.get("desired_heading"),
            "path": path,
            "path_length": max(0, len(path) - 1),
            "distance_to_goal": distance_to_goal,
            "step_index": step_index,
            "route_step": route_step,
            "safety_reason": safety_reason,
            "source": "active_frontier_goal",
            "updated_at": now_iso(),
        }
    )


def inactive_route(
    reason: str,
    *,
    previous_route: JsonDict,
    goal_cell: str = "",
    path: Sequence[str] | None = None,
) -> JsonDict:
    return clean_empty(
        {
            "schema": ACTIVE_ROUTE_SCHEMA,
            "status": "inactive",
            "route_id": previous_route.get("route_id"),
            "goal_cell": goal_cell or previous_route.get("goal_cell"),
            "path": list(path or []),
            "reason": reason,
            "updated_at": now_iso(),
        }
    )


def update_active_route(
    *,
    memory_dir: Path | str | None = None,
    persist: bool = True,
) -> JsonDict:
    memory = Path(memory_dir) if memory_dir else MEMORY_DIR
    room_path = memory / "room-state.json"
    costmap_path = memory / "navigation-costmap.json"
    snapshot = load_map_backend(memory).load_snapshot()
    position_status = snapshot.to_position_status()
    room = load_json(room_path)
    costmap = load_json(costmap_path)
    current_cell, current_heading = current_pose(position_status, room)
    active_goal = as_dict(room.get("active_frontier_goal") or position_status.get("active_frontier_goal"))
    previous_route = as_dict(room.get("active_route") or position_status.get("active_route"))

    if not active_goal or active_goal.get("status") not in {None, "active"}:
        active_route = inactive_route("no_active_frontier_goal", previous_route=previous_route)
    else:
        active_route = build_active_route(
            active_goal=active_goal,
            previous_route=previous_route,
            current_cell=current_cell,
            current_heading=current_heading,
            costmap=costmap,
        )

    if persist:
        room["active_route"] = active_route
        atomic_write_json(room_path, room)

    return clean_empty(
        {
            "schema": ROUTE_UPDATE_SCHEMA,
            "status": "success",
            "result_type": "active_route_updated",
            "generated_at": now_iso(),
            "active_route": active_route,
            "current_pose": {"cell": current_cell, "heading": current_heading},
        }
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Update committed active route for frontier exploration.")
    parser.add_argument("--memory-dir", default=str(MEMORY_DIR))
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    args = parser.parse_args()
    result = update_active_route(memory_dir=Path(args.memory_dir), persist=not args.no_persist)
    if args.format == "compact":
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
