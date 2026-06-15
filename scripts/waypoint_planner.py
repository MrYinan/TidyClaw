#!/usr/bin/env python3
"""Waypoint-level planner for inspection coverage patrol.

This is the second layer in the Roboclaws-style navigation contract:

MapBackend -> inspection_waypoints -> coverage_waypoint_state -> waypoint_planner

The planner accepts a durable inspection waypoint id and produces a route plus
one safe next action.  It does not choose which waypoint should be visited; that
remains a goal-level decision.  It also does not execute movement; execution is
owned by execute_option/move-robot.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.coverage_waypoint_state import (
        clear_active_waypoint_goal,
        load_room_state,
        mark_waypoint_blocked,
        normalize_coverage_waypoint_state,
        save_room_state,
        set_active_waypoint_goal,
    )
    from scripts.global_planner import AStarGlobalPlanner
    from scripts.inspection_waypoints import build_inspection_waypoints, waypoint_by_id
    from scripts.map_backend import load_map_backend
    from scripts.map_backend.base import JsonDict, as_dict, as_list
    from scripts.route_manager import route_id_for, route_step_for_next_cell
    from scripts.runtime_config import apply_runtime_environment, restore_runtime_environment
except ImportError:  # pragma: no cover - direct script execution
    from coverage_waypoint_state import (
        clear_active_waypoint_goal,
        load_room_state,
        mark_waypoint_blocked,
        normalize_coverage_waypoint_state,
        save_room_state,
        set_active_waypoint_goal,
    )
    from global_planner import AStarGlobalPlanner
    from inspection_waypoints import build_inspection_waypoints, waypoint_by_id
    from map_backend import load_map_backend
    from map_backend.base import JsonDict, as_dict, as_list
    from route_manager import route_id_for, route_step_for_next_cell
    from runtime_config import apply_runtime_environment, restore_runtime_environment


WAYPOINT_PLAN_SCHEMA = "robot_cleaner_waypoint_plan_v1"
ACTIVE_WAYPOINT_ROUTE_SCHEMA = "robot_cleaner_active_waypoint_route_v1"
WAYPOINT_ROUTE_STEP_SOURCE = "inspection_waypoint_planner"

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[1] / "memory"
TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATION_ACTIONS = {"RotateLeft", "RotateRight"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> JsonDict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        replaced = False
        last_error: PermissionError | None = None
        for attempt in range(20):
            try:
                os.replace(tmp_name, path)
                replaced = True
                break
            except PermissionError as exc:
                last_error = exc
                if attempt >= 19:
                    break
                time.sleep(min(0.5, 0.05 * (attempt + 1)))
        if not replaced:
            try:
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
            except PermissionError:
                if last_error is not None:
                    raise last_error
                raise
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def current_pose(snapshot, room: Mapping[str, Any]) -> tuple[str, str]:
    pose = as_dict(snapshot.pose)
    cell = str(pose.get("cell") or snapshot.to_position_status().get("last_cell") or room.get("last_cell") or "0,0")
    heading = str(pose.get("heading") or snapshot.to_position_status().get("last_heading") or room.get("last_heading") or "north")
    return cell, heading


def waypoint_public_payload(waypoint: Mapping[str, Any]) -> JsonDict:
    keys = (
        "waypoint_id",
        "cell",
        "x_cell",
        "z_cell",
        "label",
        "purpose",
        "waypoint_source",
        "coverage_radius_cells",
        "coverage_estimate",
        "component_id",
        "component_index",
        "component_size",
        "covered_cell_count",
        "world_position",
    )
    return {key: waypoint[key] for key in keys if key in waypoint}


def action_safety_record(costmap: Mapping[str, Any], action: str) -> JsonDict:
    return as_dict(as_dict(costmap.get("action_safety")).get(action))


def min_observed_ratio(record: Mapping[str, Any]) -> float:
    try:
        return float(record.get("min_observed_ratio"))
    except (TypeError, ValueError):
        return 0.0


def observed_ratio(record: Mapping[str, Any]) -> float:
    try:
        return float(record.get("observed_ratio"))
    except (TypeError, ValueError):
        return 0.0


def validate_action_safety(costmap: Mapping[str, Any], action: str) -> JsonDict:
    if not action:
        return {"safe": False, "reason": "missing_action"}
    record = action_safety_record(costmap, action)
    if not record:
        return {"safe": False, "reason": "missing_action_safety", "action": action}
    if record.get("safe") is not True:
        return {
            "safe": False,
            "reason": str(record.get("reason") or "costmap_blocked"),
            "action": action,
            "record": dict(record),
        }
    if action in TRANSLATION_ACTIONS and observed_ratio(record) < min_observed_ratio(record):
        return {
            "safe": False,
            "reason": "low_observed_swept_volume",
            "action": action,
            "observed_ratio": observed_ratio(record),
            "min_observed_ratio": min_observed_ratio(record),
            "record": dict(record),
        }
    return {
        "safe": True,
        "reason": str(record.get("reason") or "clear_swept_volume"),
        "action": action,
        "record": dict(record),
    }


def _route_status_from_safety(status: str, safety: Mapping[str, Any]) -> str:
    if status != "success":
        return "no_path"
    return "active" if safety.get("safe") is True else "blocked"


def _route_block_reason(plan: Mapping[str, Any], safety: Mapping[str, Any]) -> str:
    if plan.get("status") != "success":
        return str(plan.get("replan_reason") or "no_path_to_waypoint")
    return str(safety.get("reason") or "action_not_safe")


def build_waypoint_route(
    *,
    waypoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    current_cell: str,
    current_heading: str,
    costmap: Mapping[str, Any],
    previous_route: Mapping[str, Any] | None = None,
) -> JsonDict:
    waypoint_id = str(waypoint.get("waypoint_id") or "")
    target_cell = str(waypoint.get("cell") or plan.get("requested_target_cell") or "")
    path = [str(item) for item in as_list(plan.get("path")) if str(item or "").strip()]
    route_id = str(as_dict(previous_route).get("route_id") or route_id_for(target_cell, path or [current_cell, target_cell]))
    step_index = safe_int(as_dict(previous_route).get("step_index"), 0)
    if as_dict(previous_route).get("current_cell") and as_dict(previous_route).get("current_cell") != current_cell:
        step_index += 1

    if current_cell == target_cell:
        return {
            "schema": ACTIVE_WAYPOINT_ROUTE_SCHEMA,
            "status": "reached",
            "route_id": route_id,
            "waypoint_id": waypoint_id,
            "goal_cell": target_cell,
            "current_cell": current_cell,
            "current_heading": current_heading,
            "path": [current_cell],
            "reason": "waypoint_cell_reached",
            "requires_observe": True,
            "updated_at": now_iso(),
        }
    if plan.get("status") != "success" or len(path) < 2:
        return clean_empty(
            {
                "schema": ACTIVE_WAYPOINT_ROUTE_SCHEMA,
                "status": "no_path",
                "route_id": route_id,
                "waypoint_id": waypoint_id,
                "goal_cell": target_cell,
                "current_cell": current_cell,
                "current_heading": current_heading,
                "path": path,
                "planner_status": plan.get("status"),
                "reason": str(plan.get("replan_reason") or "no_path_to_waypoint"),
                "updated_at": now_iso(),
            }
        )

    next_cell = str(plan.get("next_cell") or (path[1] if len(path) > 1 else ""))
    route_step = route_step_for_next_cell(
        route_id=route_id,
        step_index=step_index,
        current_cell=current_cell,
        current_heading=current_heading,
        next_cell=next_cell,
        goal_cell=target_cell,
        path=path,
    )
    action = str(route_step.get("action") or plan.get("next_action") or "")
    safety = validate_action_safety(costmap, action)
    status = _route_status_from_safety(str(plan.get("status") or ""), safety)
    return clean_empty(
        {
            "schema": ACTIVE_WAYPOINT_ROUTE_SCHEMA,
            "status": status,
            "route_id": route_id,
            "waypoint_id": waypoint_id,
            "goal_cell": target_cell,
            "current_cell": current_cell,
            "current_heading": current_heading,
            "next_cell": next_cell,
            "next_action": action,
            "path": path,
            "path_length": max(0, len(path) - 1),
            "route_cost": plan.get("route_cost"),
            "route_step": {
                **dict(route_step),
                "source": WAYPOINT_ROUTE_STEP_SOURCE,
            },
            "action_safety": safety,
            "blocked_reason": None if safety.get("safe") is True else _route_block_reason(plan, safety),
            "planner": plan.get("planner"),
            "planner_status": plan.get("status"),
            "updated_at": now_iso(),
        }
    )


def _load_waypoint_context(memory: Path) -> tuple[Any, JsonDict, JsonDict, JsonDict, dict[str, JsonDict]]:
    snapshot = load_map_backend(memory).load_snapshot()
    waypoint_set = build_inspection_waypoints(snapshot)
    room = load_room_state(memory)
    cell, _ = current_pose(snapshot, room)
    coverage_state = normalize_coverage_waypoint_state(
        waypoint_set,
        as_dict(room.get("coverage_waypoints")),
        current_cell=cell,
    )
    waypoints = waypoint_by_id(waypoint_set)
    return snapshot, waypoint_set, room, coverage_state, waypoints


def _persist_coverage(memory: Path, room: JsonDict, coverage_state: Mapping[str, Any]) -> None:
    room["coverage_waypoints"] = dict(coverage_state)
    save_room_state(room, memory)


def plan_to_inspection_waypoint(
    waypoint_id: str,
    *,
    memory_dir: Path | str | None = None,
    persist: bool = True,
) -> JsonDict:
    """Activate and plan one step toward an inspection waypoint."""

    memory = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
    snapshot, waypoint_set, room, coverage_state, waypoints = _load_waypoint_context(memory)
    waypoint_key = str(waypoint_id or "").strip()
    if waypoint_key not in waypoints:
        return {
            "schema": WAYPOINT_PLAN_SCHEMA,
            "status": "invalid_waypoint",
            "result_type": "waypoint_plan_failed",
            "waypoint_id": waypoint_key,
            "reason": "unknown_waypoint_id",
            "available_waypoint_ids": list(waypoints.keys())[:12],
        }
    try:
        coverage_state = set_active_waypoint_goal(coverage_state, waypoint_key)
    except ValueError as exc:
        return {
            "schema": WAYPOINT_PLAN_SCHEMA,
            "status": "invalid_waypoint",
            "result_type": "waypoint_plan_failed",
            "waypoint_id": waypoint_key,
            "reason": str(exc),
        }
    result = _plan_active_waypoint(
        memory=memory,
        snapshot=snapshot,
        room=room,
        coverage_state=coverage_state,
        waypoint=waypoints[waypoint_key],
    )
    if persist:
        coverage_state = dict(result.get("coverage_waypoints") or coverage_state)
        _persist_coverage(memory, room, coverage_state)
        _persist_last_waypoint_plan(memory, result)
    return result


def continue_active_waypoint_goal(
    *,
    memory_dir: Path | str | None = None,
    persist: bool = True,
) -> JsonDict:
    """Plan one additional step toward the active inspection waypoint goal."""

    memory = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
    snapshot, waypoint_set, room, coverage_state, waypoints = _load_waypoint_context(memory)
    active = as_dict(coverage_state.get("active_waypoint_goal"))
    waypoint_key = str(active.get("waypoint_id") or "").strip()
    if not waypoint_key:
        return {
            "schema": WAYPOINT_PLAN_SCHEMA,
            "status": "no_active_waypoint_goal",
            "result_type": "waypoint_plan_inactive",
            "required_next": "choose_inspection_waypoint",
            "coverage_waypoints": coverage_state,
        }
    waypoint = waypoints.get(waypoint_key)
    if waypoint is None:
        coverage_state = clear_active_waypoint_goal(coverage_state, reason="active_waypoint_missing_from_map")
        if persist:
            _persist_coverage(memory, room, coverage_state)
        return {
            "schema": WAYPOINT_PLAN_SCHEMA,
            "status": "invalid_waypoint",
            "result_type": "waypoint_plan_failed",
            "waypoint_id": waypoint_key,
            "reason": "active_waypoint_missing_from_current_map",
            "coverage_waypoints": coverage_state,
        }
    result = _plan_active_waypoint(
        memory=memory,
        snapshot=snapshot,
        room=room,
        coverage_state=coverage_state,
        waypoint=waypoint,
    )
    if persist:
        coverage_state = dict(result.get("coverage_waypoints") or coverage_state)
        _persist_coverage(memory, room, coverage_state)
        _persist_last_waypoint_plan(memory, result)
    return result


def _plan_active_waypoint(
    *,
    memory: Path,
    snapshot,
    room: JsonDict,
    coverage_state: Mapping[str, Any],
    waypoint: Mapping[str, Any],
) -> JsonDict:
    current_cell, current_heading = current_pose(snapshot, room)
    target_cell = str(waypoint.get("cell") or "").strip()
    waypoint_id = str(waypoint.get("waypoint_id") or "")
    if not target_cell:
        next_state = mark_waypoint_blocked(
            coverage_state,
            waypoint_id,
            reason="waypoint_missing_target_cell",
        )
        return {
            "schema": WAYPOINT_PLAN_SCHEMA,
            "status": "blocked",
            "result_type": "waypoint_plan_failed",
            "waypoint_id": waypoint_id,
            "reason": "waypoint_missing_target_cell",
            "coverage_waypoints": next_state,
        }
    costmap = read_json(memory / "navigation-costmap.json")
    planner = AStarGlobalPlanner(memory)
    plan = planner.plan_to_goal(
        current_cell=current_cell,
        current_heading=current_heading,
        target_cell=target_cell,
        target_reason="inspection_waypoint",
        goal_type="inspection_waypoint",
        position_status=snapshot.to_position_status(),
        semantic_status={"cells": {}},
        drive_model="nonholonomic",
    )
    previous_route = as_dict(coverage_state.get("active_waypoint_route"))
    route = build_waypoint_route(
        waypoint=waypoint,
        plan=plan,
        current_cell=current_cell,
        current_heading=current_heading,
        costmap=costmap,
        previous_route=previous_route,
    )
    next_state = dict(coverage_state)
    status = str(route.get("status") or "")
    if status in {"no_path"}:
        next_state = mark_waypoint_blocked(next_state, waypoint_id, reason=str(route.get("reason") or "no_path"))
    elif status == "blocked":
        next_state = dict(next_state)
        route["fail_count"] = safe_int(previous_route.get("fail_count"), 0) + 1 if previous_route else 1
        active = as_dict(next_state.get("active_waypoint_goal"))
        active["status"] = "blocked"
        active["blocked_reason"] = route.get("blocked_reason")
        next_state["active_waypoint_goal"] = active
    elif status == "reached":
        active = as_dict(next_state.get("active_waypoint_goal"))
        active["status"] = "reached"
        active["reached_at"] = now_iso()
        active["requires_observe"] = True
        next_state["active_waypoint_goal"] = active
    else:
        active = as_dict(next_state.get("active_waypoint_goal"))
        active["status"] = "active"
        active["cell"] = target_cell
        active["route_id"] = route.get("route_id")
        active["path"] = route.get("path")
        active["next_action"] = route.get("next_action")
        active["next_cell"] = route.get("next_cell")
        next_state["active_waypoint_goal"] = active
    next_state["active_waypoint_route"] = route
    next_state["updated_at"] = now_iso()
    return clean_empty(
        {
            "schema": WAYPOINT_PLAN_SCHEMA,
            "status": status,
            "result_type": "waypoint_plan_ready" if status == "active" else "waypoint_plan_status",
            "waypoint": waypoint_public_payload(waypoint),
            "current_pose": {"cell": current_cell, "heading": current_heading},
            "route": route,
            "planner_result": {
                "status": plan.get("status"),
                "planner": plan.get("planner"),
                "selected_goal_cell": plan.get("selected_goal_cell"),
                "path": plan.get("path"),
                "route_cost": plan.get("route_cost"),
                "expanded_node_count": plan.get("expanded_node_count"),
                "replan_reason": plan.get("replan_reason"),
            },
            "next_action": route.get("next_action"),
            "next_cell": route.get("next_cell"),
            "required_next": (
                "execute_waypoint_route_step"
                if status == "active"
                else "observe_at_waypoint"
                if status == "reached"
                else "recover_or_choose_new_waypoint"
            ),
            "coverage_waypoints": next_state,
        }
    )


def _persist_last_waypoint_plan(memory: Path, result: Mapping[str, Any]) -> None:
    path = memory / "waypoint-plan.json"
    atomic_write_json(path, dict(result))


def last_waypoint_plan(memory_dir: Path | str | None = None) -> JsonDict:
    memory = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
    return read_json(memory / "waypoint-plan.json")


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Plan one step toward an inspection waypoint.")
    parser.add_argument("command", choices=["plan", "continue", "last"])
    parser.add_argument("--waypoint-id", default="")
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--format", choices=["json", "pretty"], default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime_environment = apply_runtime_environment(override_existing=True)
    previous_runtime_env = (
        runtime_environment.get("previous_env") if isinstance(runtime_environment.get("previous_env"), dict) else {}
    )
    memory = Path(args.memory_dir) if args.memory_dir else None
    try:
        if args.command == "plan":
            if not args.waypoint_id:
                raise SystemExit("--waypoint-id is required for plan")
            result = plan_to_inspection_waypoint(
                args.waypoint_id,
                memory_dir=memory,
                persist=not args.no_persist,
            )
        elif args.command == "continue":
            result = continue_active_waypoint_goal(memory_dir=memory, persist=not args.no_persist)
        else:
            result = last_waypoint_plan(memory_dir=memory)
        if isinstance(result, dict):
            result.setdefault("runtime_environment", runtime_environment)
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.format == "pretty" else None))
        return 0
    finally:
        restore_runtime_environment(previous_runtime_env)


if __name__ == "__main__":
    raise SystemExit(main())
