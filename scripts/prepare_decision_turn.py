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
from scripts.exploration_goal_manager import update_exploration_goals
from scripts.coverage_waypoint_state import (
    load_room_state,
    mark_waypoint_observed,
    normalize_coverage_waypoint_state,
    save_room_state,
)
from scripts.inspection_waypoints import build_inspection_waypoints, waypoint_by_id
from scripts.local_costmap import LocalCostmap
from scripts.map_backend import BackendUnavailableError, load_map_backend
from scripts.authoritative_map_sync import sync_authoritative_room_state
from scripts.option_state_sync import ensure_tool_mission_active
from scripts.route_manager import update_active_route
from scripts.runtime_config import apply_runtime_environment, restore_runtime_environment


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
        "perception_mode": refresh.get("perception_mode"),
        "vision_status": vision_data.get("status"),
        "vision_result_type": vision_data.get("result_type"),
        "vision_elapsed_ms": vision.get("elapsed_ms"),
        "image_path": vision_data.get("image_path"),
        "depth_path": vision_data.get("depth_path"),
        "perception_status": perception_data.get("status"),
        "perception_result_type": perception_data.get("result_type"),
        "perception_elapsed_ms": perception.get("elapsed_ms"),
        "perception_backend": perception_data.get("perception_backend"),
        "candidate_count": perception_data.get("candidate_count"),
        "pickup_target_detected": perception_data.get("pickup_target_detected"),
        "place_receptacle_detected": perception_data.get("place_receptacle_detected"),
        "frontier_exists": perception_data.get("frontier_exists"),
        "recommended_action": perception_data.get("recommended_action"),
        "perception_written": refresh.get("perception_written"),
    }


def current_map_cell(memory_dir: Path) -> str:
    try:
        snapshot = load_map_backend(memory_dir).load_snapshot()
    except BackendUnavailableError:
        position = read_json_file(memory_dir / "position-map.json")
        pose = as_dict(position.get("pose"))
        return str(pose.get("cell") or position.get("last_cell") or "").strip()
    pose = as_dict(snapshot.pose)
    position = snapshot.to_position_status()
    return str(pose.get("cell") or position.get("last_cell") or "").strip()


def decide_perception_mode(args: argparse.Namespace, *, memory_dir: Path = MEMORY_DIR) -> JsonDict:
    requested = str(getattr(args, "perception_mode", "auto") or "auto").strip().lower()
    if requested in {"full", "full_task", "full-task"}:
        return {
            "mode": "full",
            "source": "cli",
            "reason": "perception_mode_forced_full",
        }
    if requested in {"navigation_only", "navigation-only", "nav_only", "nav-only"}:
        return {
            "mode": "navigation_only",
            "source": "cli",
            "reason": "perception_mode_forced_navigation_only",
        }

    service_state = read_json_file(memory_dir / "service-task-state.json")
    if bool(service_state.get("holding_object")):
        return {
            "mode": "full",
            "source": "auto",
            "reason": "holding_object_requires_place_perception",
        }
    phase = str(service_state.get("phase") or "").strip()
    if phase in {"SEARCH_RECEPTACLE", "LOCK_RECEPTACLE", "ALIGN_RECEPTACLE", "PLACE_OBJECT"}:
        return {
            "mode": "full",
            "source": "auto",
            "reason": "place_phase_requires_surface_perception",
        }

    room = read_json_file(memory_dir / "room-state.json")
    coverage = as_dict(room.get("coverage_waypoints"))
    active = as_dict(coverage.get("active_waypoint_goal"))
    waypoint_id = str(active.get("waypoint_id") or "").strip()
    target_cell = str(active.get("cell") or "").strip()
    status = str(active.get("status") or "").strip().lower()
    if not waypoint_id or status in {"blocked", "failed", "observed"}:
        return {
            "mode": "full",
            "source": "auto",
            "reason": "no_active_waypoint_goal_requires_task_observe",
        }
    current_cell = current_map_cell(memory_dir)
    if current_cell and target_cell and current_cell == target_cell:
        return {
            "mode": "full",
            "source": "auto",
            "reason": "active_waypoint_reached_requires_task_observe",
            "active_waypoint_goal": {
                "waypoint_id": waypoint_id,
                "cell": target_cell,
                "status": status or active.get("status"),
            },
            "current_cell": current_cell,
        }
    return {
        "mode": "navigation_only",
        "source": "auto",
        "reason": "continue_active_waypoint_route_uses_depth_costmap_only",
        "active_waypoint_goal": {
            "waypoint_id": waypoint_id,
            "cell": target_cell,
            "status": status or active.get("status"),
        },
        "current_cell": current_cell,
    }


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def bounded_items(value: Any, *, limit: int) -> list[Any]:
    return as_list(value)[: max(0, int(limit))]


def compact_candidate(candidate: Any) -> JsonDict:
    data = as_dict(candidate)
    compact = {
        key: data.get(key)
        for key in (
            "candidate_id",
            "track_id",
            "label",
            "object_label",
            "target_label",
            "surface_label",
            "surface_type",
            "confidence",
            "distance_m",
            "bearing_deg",
            "location_hint",
            "pickup_now",
            "place_now",
            "place_precheck_ready",
            "final_place_ready",
            "precheck_ok",
            "needs_alignment",
            "needs_approach",
            "interaction_point",
            "required_next",
            "reason",
        )
        if key in data
    }
    placement_points = as_list(data.get("placement_points"))
    if placement_points:
        compact["placement_points_sample"] = placement_points[:8]
        compact["placement_points_count"] = len(placement_points)
    return compact


def compact_decision_context(context: JsonDict, *, context_path: Path) -> JsonDict:
    """Return the bounded public state that is safe to write into OpenClaw chat.

    The full context remains on disk at ``context_path``. Tool responses should
    not embed the full file because it contains large intermediate perception
    and mapping fields that make OpenClaw session writes slow and fragile.
    """

    option_set = as_dict(context.get("option_set"))
    navigation = as_dict(context.get("navigation"))
    exploration = as_dict(context.get("exploration"))
    explore_plan = as_dict(context.get("explore_plan"))
    coverage_waypoints = as_dict(context.get("coverage_waypoints"))
    inspection_waypoints = as_dict(context.get("inspection_waypoints"))
    worklist = as_dict(context.get("worklist"))
    current_view = as_dict(worklist.get("current_view"))
    perception = as_dict(context.get("perception"))

    return {
        "status": context.get("status"),
        "schema": context.get("schema"),
        "generated_at": context.get("generated_at"),
        "full_context_path": display_path(context_path),
        "task": context.get("task"),
        "mission": {
            key: context.get(key)
            for key in (
                "task_mode",
                "phase",
                "holding_object",
                "held_object",
                "room_complete",
                "done_reason",
            )
            if key in context
        },
        "perception": {
            key: perception.get(key)
            for key in (
                "status",
                "result_type",
                "perception_backend",
                "source_info",
                "pickup_target_detected",
                "place_receptacle_detected",
                "candidate_count",
                "source_time",
            )
            if key in perception
        },
        "navigation": {
            "map_backend": navigation.get("map_backend"),
            "current_pose": navigation.get("current_pose") or exploration.get("current_pose"),
            "coverage": navigation.get("coverage") or exploration.get("coverage"),
            "active_frontier_goal": navigation.get("active_frontier_goal")
            or exploration.get("active_frontier_goal"),
            "active_route": navigation.get("active_route") or exploration.get("active_route"),
            "local_costmap": navigation.get("local_costmap"),
        },
        "exploration": {
            key: exploration.get(key)
            for key in (
                "recent_path",
                "revisit_counts",
                "frontier_candidates",
                "loop_warning",
                "avoid_actions",
                "low_confidence_moves",
                "camera_posture",
                "coverage_patrol",
            )
            if key in exploration
        },
        "explore_plan": {
            key: explore_plan.get(key)
            for key in (
                "mode",
                "status",
                "reason",
                "active_frontier_goal",
                "active_route",
                "selected_frontier",
                "selected_option_id",
                "suppressed_frontier_cells",
                "required_next",
            )
            if key in explore_plan
        },
        "coverage_waypoints": {
            key: coverage_waypoints.get(key)
            for key in (
                "schema",
                "waypoint_source",
                "map_backend",
                "required_waypoint_count",
                "observed_waypoint_count",
                "blocked_waypoint_count",
                "pending_waypoint_count",
                "sweep_coverage_rate",
                "active_waypoint_goal",
                "next_unobserved_waypoints",
            )
            if key in coverage_waypoints
        },
        "inspection_waypoints": {
            key: inspection_waypoints.get(key)
            for key in (
                "schema",
                "source",
                "map_backend",
                "required_waypoint_count",
                "coverage_radius_cells",
            )
            if key in inspection_waypoints
        },
        "worklist": {
            "held_object": worklist.get("held_object"),
            "pickup_candidate_count": len(as_list(current_view.get("pickup_candidates"))),
            "place_candidate_count": len(as_list(current_view.get("place_candidates"))),
            "pickup_candidates": [
                compact_candidate(item) for item in bounded_items(current_view.get("pickup_candidates"), limit=4)
            ],
            "place_candidates": [
                compact_candidate(item) for item in bounded_items(current_view.get("place_candidates"), limit=4)
            ],
        },
        "option_set": option_set,
        "done_readiness": context.get("done_readiness"),
        "consistency_warnings": context.get("consistency_warnings") or [],
    }


def observe_reached_active_waypoint(refresh: JsonDict) -> JsonDict:
    """Mark the active inspection waypoint observed after a successful prepare observe."""

    if refresh.get("status") != "success":
        return {
            "status": "skipped",
            "result_type": "waypoint_observation_not_recorded",
            "reason": "observe_refresh_failed",
        }
    try:
        snapshot = load_map_backend(MEMORY_DIR).load_snapshot()
    except BackendUnavailableError as exc:
        return {
            "status": "error",
            "result_type": "waypoint_observation_record_failed",
            "reason": "map_backend_unavailable",
            "message": str(exc),
        }

    room = load_room_state(MEMORY_DIR)
    waypoint_set = build_inspection_waypoints(snapshot)
    pose = as_dict(snapshot.pose)
    current_cell = str(
        pose.get("cell")
        or snapshot.to_position_status().get("last_cell")
        or room.get("last_cell")
        or ""
    ).strip()
    coverage = normalize_coverage_waypoint_state(
        waypoint_set,
        as_dict(room.get("coverage_waypoints")),
        current_cell=current_cell,
    )
    active = as_dict(coverage.get("active_waypoint_goal"))
    waypoint_id = str(active.get("waypoint_id") or "").strip()
    if not waypoint_id:
        return {
            "status": "skipped",
            "result_type": "waypoint_observation_not_recorded",
            "reason": "no_active_waypoint_goal",
            "coverage_waypoints": {
                "sweep_coverage_rate": coverage.get("sweep_coverage_rate"),
                "pending_waypoint_count": coverage.get("pending_waypoint_count"),
                "observed_waypoint_count": coverage.get("observed_waypoint_count"),
                "required_waypoint_count": coverage.get("required_waypoint_count"),
            },
        }

    waypoint = waypoint_by_id(waypoint_set).get(waypoint_id) or {}
    target_cell = str(active.get("cell") or waypoint.get("cell") or "").strip()
    if not current_cell or not target_cell or current_cell != target_cell:
        return {
            "status": "skipped",
            "result_type": "waypoint_observation_not_recorded",
            "reason": "active_waypoint_not_reached",
            "active_waypoint_goal": {
                "waypoint_id": waypoint_id,
                "cell": target_cell,
                "status": active.get("status"),
            },
            "current_cell": current_cell,
        }

    perception = as_dict(refresh.get("perception"))
    perception_data = as_dict(perception.get("data"))
    vision = as_dict(refresh.get("vision"))
    vision_data = as_dict(vision.get("data"))
    observation_id = str(
        perception_data.get("observation_id")
        or vision_data.get("image_path")
        or perception_data.get("image_path")
        or refresh.get("perception_written")
        or now_iso()
    )
    updated = mark_waypoint_observed(
        coverage,
        waypoint_id,
        observation_id=observation_id,
    )
    room["coverage_waypoints"] = updated
    save_room_state(room, MEMORY_DIR)
    return {
        "status": "success",
        "result_type": "active_waypoint_observed",
        "waypoint_id": waypoint_id,
        "cell": target_cell,
        "current_cell": current_cell,
        "observation_id": observation_id,
        "sweep_coverage_rate": updated.get("sweep_coverage_rate"),
        "observed_waypoint_count": updated.get("observed_waypoint_count"),
        "blocked_waypoint_count": updated.get("blocked_waypoint_count"),
        "pending_waypoint_count": updated.get("pending_waypoint_count"),
        "required_waypoint_count": updated.get("required_waypoint_count"),
    }


def build_decision_context(args: argparse.Namespace, output_path: Path) -> ScriptResult:
    return run_script(
        DECISION_CONTEXT_SCRIPT,
        build_context_command_args(args, output_path),
        timeout_seconds=max(1, int(args.context_timeout)),
    )


def prepare_decision_turn(args: argparse.Namespace) -> JsonDict:
    output_path = resolve_workspace_path(args.output)
    runtime_environment = apply_runtime_environment(override_existing=True)
    previous_runtime_env = runtime_environment.get("previous_env") if isinstance(runtime_environment.get("previous_env"), dict) else {}
    try:
        return _prepare_decision_turn(args, output_path=output_path, runtime_environment=runtime_environment)
    finally:
        restore_runtime_environment(previous_runtime_env)


def _prepare_decision_turn(
    args: argparse.Namespace,
    *,
    output_path: Path,
    runtime_environment: JsonDict,
) -> JsonDict:
    started_at = time.time()
    attempts: list[JsonDict] = []
    refresh: JsonDict = {}
    mission_activation = ensure_tool_mission_active(memory_dir=MEMORY_DIR, mode="SERVICE")
    perception_mode = decide_perception_mode(args, memory_dir=MEMORY_DIR)

    for attempt_index in range(1, max(0, int(args.observe_retries)) + 2):
        refresh = run_observe_refresh(
            timeout_seconds=max(1, int(args.timeout)),
            vision_timeout_seconds=max(1, int(args.vision_timeout)),
            yolo_timeout_seconds=max(1, int(args.yolo_timeout)),
            perception_mode=str(perception_mode.get("mode") or "full"),
        )
        attempts.append(summarize_refresh(refresh))
        if refresh.get("status") == "success":
            break

    if refresh.get("status") != "success":
        result = {
            "status": "error",
            "result_type": "decision_turn_prepare_failed",
            "stage": "observe_refresh",
            "runtime_environment": runtime_environment,
            "mission_activation": mission_activation,
            "perception_mode": perception_mode,
            "attempts": attempts,
            "elapsed_ms": round((time.time() - started_at) * 1000, 1),
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

    try:
        exploration_goals = update_exploration_goals(memory_dir=MEMORY_DIR, persist=True)
    except Exception as exc:
        exploration_goals = {
            "status": "error",
            "result_type": "exploration_goal_update_failed",
            "message": str(exc),
        }

    try:
        active_route = update_active_route(memory_dir=MEMORY_DIR, persist=True)
    except Exception as exc:
        active_route = {
            "status": "error",
            "result_type": "active_route_update_failed",
            "message": str(exc),
        }

    try:
        authoritative_map_sync = sync_authoritative_room_state(MEMORY_DIR)
    except Exception as exc:
        authoritative_map_sync = {
            "status": "error",
            "result_type": "authoritative_map_sync_failed",
            "message": str(exc),
        }

    try:
        waypoint_observation = observe_reached_active_waypoint(refresh)
    except Exception as exc:
        waypoint_observation = {
            "status": "error",
            "result_type": "waypoint_observation_record_failed",
            "message": str(exc),
        }

    context_result = build_decision_context(args, output_path)
    context = context_result.data if isinstance(context_result.data, dict) else {}
    if context_result.returncode != 0 or context.get("status") != "success":
        result = {
            "status": "error",
            "result_type": "decision_turn_prepare_failed",
            "stage": "build_decision_context",
            "runtime_environment": runtime_environment,
            "mission_activation": mission_activation,
            "perception_mode": perception_mode,
            "observe_refresh": attempts[-1] if attempts else {},
            "local_costmap": local_costmap,
            "exploration_goals": exploration_goals,
            "active_route": active_route,
            "authoritative_map_sync": authoritative_map_sync,
            "waypoint_observation": waypoint_observation,
            "context_builder": script_result_payload(context_result),
            "elapsed_ms": round((time.time() - started_at) * 1000, 1),
            "required_next": "inspect_decision_context_builder",
        }
        append_trace({"event": "decision_turn_prepare_failed", "result": result})
        return result

    option_set = context.get("option_set") if isinstance(context.get("option_set"), dict) else {}
    decision_context_public = compact_decision_context(context, context_path=output_path)
    result = {
        "status": "success",
        "result_type": "decision_turn_prepared",
        "schema": "robot_cleaner_decision_turn_v1",
        "prepared_at": now_iso(),
        "runtime_environment": runtime_environment,
        "mission_activation": mission_activation,
        "perception_mode": perception_mode,
        "context_path": display_path(output_path),
        "perception_path": display_path(DEFAULT_PERCEPTION_PATH),
        "observe_refresh": attempts[-1] if attempts else {},
        "local_costmap": local_costmap,
        "exploration_goals": exploration_goals,
        "active_route": active_route,
        "authoritative_map_sync": authoritative_map_sync,
        "waypoint_observation": waypoint_observation,
        "context_builder": script_result_payload(context_result),
        "decision_context": decision_context_public,
        "full_decision_context_path": display_path(output_path),
        "full_decision_context_written": True,
        "option_set": option_set,
        "model_decision_required": True,
        "rule_baseline_option_id": option_set.get("rule_baseline_option_id"),
        "option_count": len(option_set.get("options") or []) if isinstance(option_set.get("options"), list) else 0,
        "consistency_warnings": context.get("consistency_warnings") or [],
        "elapsed_ms": round((time.time() - started_at) * 1000, 1),
    }
    append_trace(
        {
            "event": "decision_turn_prepared",
            "status": "success",
            "model_decision_required": True,
            "rule_baseline_option_id": result.get("rule_baseline_option_id"),
            "option_count": result.get("option_count"),
            "elapsed_ms": result.get("elapsed_ms"),
            "context_path": result.get("context_path"),
            "observe_refresh": result.get("observe_refresh"),
            "context_builder": result.get("context_builder"),
            "runtime_environment": runtime_environment,
            "mission_activation": mission_activation,
            "perception_mode": perception_mode,
            "local_costmap": result.get("local_costmap"),
            "exploration_goals": result.get("exploration_goals"),
            "active_route": result.get("active_route"),
            "authoritative_map_sync": result.get("authoritative_map_sync"),
            "waypoint_observation": result.get("waypoint_observation"),
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one OpenClaw robot decision turn.")
    parser.add_argument("--task-mode", choices=("auto", "tidy", "clean"), default="tidy")
    parser.add_argument("--output", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per backend script in seconds.")
    parser.add_argument("--vision-timeout", type=int, default=30, help="Timeout for get-vision in seconds.")
    parser.add_argument("--yolo-timeout", type=int, default=60, help="Timeout for YOLO service analysis in seconds.")
    parser.add_argument(
        "--context-timeout",
        type=int,
        default=45,
        help="Timeout for decision_context_builder.py in seconds.",
    )
    parser.add_argument(
        "--observe-retries",
        type=int,
        default=0,
        help="Retry observe/perception this many times before failing the turn.",
    )
    parser.add_argument(
        "--perception-mode",
        choices=("auto", "full", "navigation_only"),
        default="auto",
        help=(
            "auto uses full task perception for pick/place/waypoint observe, "
            "and depth-only navigation perception while continuing an active waypoint."
        ),
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
