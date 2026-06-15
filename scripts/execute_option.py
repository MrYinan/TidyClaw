#!/usr/bin/env python3
"""Execute one validated option from robot_cleaner_decision_context_v1.

This is the bridge between an OpenClaw/LLM option selection and the existing
script-based robot skills. It does not let the model run arbitrary commands:
the selected option id must exist in the decision context, pass local gates,
and map to one known executor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_CONTEXT_PATH = MEMORY_DIR / "decision-context.json"
TRACE_PATH = MEMORY_DIR / "option-execution-trace.jsonl"
PLACE_PRECHECK_CACHE_PATH = MEMORY_DIR / "place-precheck-cache.json"

GET_VISION_SCRIPT = REPO_ROOT / "skills" / "get-vision" / "scripts" / "get_vision.py"
YOLO_SCRIPT = REPO_ROOT / "skills" / "perceive-scene-yolo" / "scripts" / "perceive_scene_yolo.py"
MOVE_SCRIPT = REPO_ROOT / "skills" / "move-robot" / "scripts" / "move_robot.py"
CLEAN_SCRIPT = REPO_ROOT / "skills" / "clean-garbage" / "scripts" / "clean_garbage.py"
PICK_SCRIPT = REPO_ROOT / "skills" / "pick-object" / "scripts" / "pick_object.py"
PLACE_SCRIPT = REPO_ROOT / "skills" / "place-object" / "scripts" / "place_object.py"

DECISION_CONTEXT_SCHEMA = "robot_cleaner_decision_context_v1"
PLACE_PRECHECK_CACHE_SCHEMA = "robot_cleaner_place_precheck_cache_v1"
PHYSICAL_KINDS = {
    "move_action",
    "explore_inspection_waypoint",
    "continue_active_waypoint_goal",
    "explore_frontier_cluster",
    "explore_route_step",
    "explore_frontier",
    "explore_waypoint",
    "recovery_action",
    "service_action",
    "clean_action",
}
NAVIGATION_ONLY_WAYPOINT_KINDS = {"explore_inspection_waypoint", "continue_active_waypoint_goal"}
SURFACE_REGION_SOURCES = {
    "pointcloud_plane",
    "pointcloud_plane_completion",
    "pointcloud_plane_grid_completion",
    "depth_region_geometry",
}
MOVE_ACTIONS = {
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
}
TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
DEFAULT_FORWARD_MIN_OBSERVED_RATIO = 0.30
DEFAULT_LATERAL_MIN_OBSERVED_RATIO = 0.50
DEFAULT_REAR_MIN_OBSERVED_RATIO = 0.60


JsonDict = dict[str, Any]


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.option_state_sync import sync_option_result
from scripts.runtime_config import apply_runtime_environment, restore_runtime_environment
from scripts.waypoint_planner import continue_active_waypoint_goal, plan_to_inspection_waypoint


@dataclass
class ScriptResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    data: JsonDict
    elapsed_ms: float = 0.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_print(data: JsonDict, *, compact: bool = False) -> None:
    if compact:
        print(json.dumps(data, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def load_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {
            "status": "error",
            "result_type": "error_json_load_failed",
            "path": str(path),
            "message": str(exc),
        }
    return data if isinstance(data, dict) else {"status": "error", "result_type": "error_json_root_not_object"}


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def context_stale_error(context: JsonDict) -> JsonDict | None:
    lifecycle = as_dict(context.get("context_lifecycle"))
    if lifecycle.get("stale_after_option_execution") is not True and lifecycle.get("valid_for_execution") is not False:
        return None
    return {
        "type": "decision_context_stale_after_option_execution",
        "required_next": "robot_cleaner_prepare_decision_turn",
        "message": "This decision context was already used for an option execution; prepare a fresh turn before selecting another option.",
        "executed_option_id": lifecycle.get("executed_option_id"),
        "executed_at": lifecycle.get("executed_at"),
        "execution_result_type": lifecycle.get("execution_result_type"),
    }


def mark_context_stale_after_execution(
    context_path: Path,
    *,
    option_id: str,
    execution: JsonDict,
) -> JsonDict:
    try:
        context = load_json(context_path)
        if context.get("schema") != DECISION_CONTEXT_SCHEMA:
            return {
                "status": "skipped",
                "result_type": "context_stale_mark_skipped_invalid_schema",
                "context_path": str(context_path),
            }
        lifecycle = as_dict(context.get("context_lifecycle"))
        lifecycle.update(
            {
                "valid_for_execution": False,
                "stale_after_option_execution": True,
                "stale_reason": "option_executed",
                "executed_option_id": option_id,
                "executed_at": now_iso(),
                "execution_status": execution.get("status"),
                "execution_result_type": execution.get("result_type"),
                "required_next": "robot_cleaner_prepare_decision_turn",
            }
        )
        context["context_lifecycle"] = lifecycle
        atomic_write_json(context_path, context)
        return {
            "status": "success",
            "result_type": "context_marked_stale_after_execution",
            "context_path": str(context_path),
            "required_next": "robot_cleaner_prepare_decision_turn",
        }
    except Exception as exc:
        return {
            "status": "error",
            "result_type": "error_context_stale_mark_failed",
            "context_path": str(context_path),
            "message": str(exc),
        }


def parse_json_output(text: str) -> JsonDict:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        pass
    for line in reversed([line.strip() for line in stripped.splitlines() if line.strip()]):
        try:
            data = json.loads(line)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            continue
    return {}


def hidden_startupinfo() -> Any:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def run_script(script: Path, args: Sequence[str], *, timeout_seconds: int) -> ScriptResult:
    command = [sys.executable, str(script), *args]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started_at = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
            check=False,
            startupinfo=hidden_startupinfo(),
            creationflags=hidden_creationflags(),
        )
        data = parse_json_output(completed.stdout) or parse_json_output(completed.stderr)
        return ScriptResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            data=data,
            elapsed_ms=round((time.time() - started_at) * 1000, 1),
        )
    except subprocess.TimeoutExpired as exc:
        return ScriptResult(
            command=command,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            data={
                "status": "error",
                "result_type": "error_script_timeout",
                "message": f"script timed out after {timeout_seconds}s",
            },
            elapsed_ms=round((time.time() - started_at) * 1000, 1),
        )


def short_command(command: Sequence[str]) -> list[str]:
    result = []
    for item in command:
        text = str(item)
        if len(text) > 240:
            text = text[:237] + "..."
        result.append(text)
    return result


def script_result_payload(result: ScriptResult) -> JsonDict:
    return {
        "command": short_command(result.command),
        "returncode": result.returncode,
        "data": result.data,
        "elapsed_ms": result.elapsed_ms,
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
    }


def append_trace(event: JsonDict, *, trace_path: Path = TRACE_PATH) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), "time": now_iso(), **event}
    with trace_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def option_list(context: JsonDict) -> list[JsonDict]:
    options = context.get("option_set", {}).get("options", [])
    return [dict(item) for item in options if isinstance(item, dict)]


def find_option(context: JsonDict, option_id: str) -> JsonDict | None:
    for option in option_list(context):
        if str(option.get("option_id") or "") == option_id:
            return option
    return None


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


def number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def resolved_move_action_for_option(context: JsonDict, option: JsonDict) -> str:
    action = str(option.get("action") or "").strip()
    if action:
        return action
    step_option_id = str(option.get("resolved_step_option_id") or "").strip()
    if not step_option_id:
        return ""
    step_option = find_option(context, step_option_id)
    if not step_option:
        return ""
    return str(step_option.get("action") or "").strip()


def waypoint_id_for_option(option: JsonDict) -> str:
    target = as_dict(option.get("waypoint_target"))
    waypoint_id = str(target.get("waypoint_id") or "").strip()
    if waypoint_id:
        return waypoint_id
    option_id = str(option.get("option_id") or "")
    prefix = "explore:inspection_waypoint:"
    if option_id.startswith(prefix):
        return option_id[len(prefix) :].strip()
    active = as_dict(option.get("active_waypoint_goal"))
    return str(active.get("waypoint_id") or "").strip()


def move_action_safety_record(context: JsonDict, action: str) -> JsonDict:
    navigation = as_dict(context.get("navigation"))
    return as_dict(as_dict(as_dict(navigation.get("local_costmap")).get("action_safety")).get(action))


def default_min_observed_ratio_for_action(action: str) -> float:
    if action == "MoveBack":
        return DEFAULT_REAR_MIN_OBSERVED_RATIO
    if action in {"MoveLeft", "MoveRight"}:
        return DEFAULT_LATERAL_MIN_OBSERVED_RATIO
    if action == "MoveAhead":
        return DEFAULT_FORWARD_MIN_OBSERVED_RATIO
    return 0.0


def required_min_observed_ratio_for_action(action: str, record: JsonDict) -> float:
    observed_min = number_or_none(record.get("min_observed_ratio"))
    default_min = default_min_observed_ratio_for_action(action)
    if observed_min is None:
        return default_min
    return max(float(observed_min), float(default_min))


def option_allows_bootstrap_moveahead(option: JsonDict, action: str) -> bool:
    if action != "MoveAhead":
        return False
    if str(option.get("safety_source") or "") == "perception-navigation-bootstrap":
        return True
    step_source = str(as_dict(option.get("exploration_effect")).get("safety_source") or "")
    return step_source == "perception-navigation-bootstrap"


def option_uses_committed_route_costmap(option: JsonDict, action: str, action_safety: JsonDict) -> bool:
    return (
        action == "MoveAhead"
        and option.get("kind") in {"explore_route_step", "explore_frontier_cluster"}
        and action_safety.get("safe") is True
        and str(option.get("safety_source") or "") in {"active-route-costmap", "navigation-costmap"}
        and (
            option.get("kind") == "explore_frontier_cluster"
            or as_dict(option.get("route_step")).get("status", "active") == "active"
        )
    )


def place_candidate_executor_ready(candidate: JsonDict) -> bool:
    actionability = as_dict(candidate.get("actionability"))
    executor_checks = as_dict(candidate.get("executor_checks"))
    if executor_checks.get("precheck_ok") is True:
        return True
    if actionability.get("needs_alignment") is True or actionability.get("needs_approach") is True:
        return False
    if actionability.get("reachable") is False:
        return False
    return actionability.get("place_now") is True or actionability.get("final_place_ready") is True


def has_interaction_point(candidate: JsonDict) -> bool:
    point = as_dict(candidate.get("interaction_point"))
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y)


def has_placement_contract(candidate: JsonDict) -> bool:
    if as_list(candidate.get("placement_points")):
        return True
    contract = as_dict(candidate.get("placement_safety_contract"))
    if contract.get("version") == "plane_local_grid_v1":
        return True
    completion = as_dict(candidate.get("free_space_completion"))
    return completion.get("mode") == "plane_local_2d_grid"


def place_candidate_precheck_ready(candidate: JsonDict) -> bool:
    actionability = as_dict(candidate.get("actionability"))
    executor_checks = as_dict(candidate.get("executor_checks"))
    if executor_checks.get("precheck_ok") is True:
        return False
    if str(candidate.get("surface_candidate_source") or "") not in SURFACE_REGION_SOURCES:
        return False
    if actionability.get("reachable") is False:
        return False
    if actionability.get("blocked") is True or actionability.get("failed_recently") is True:
        return False
    if actionability.get("visual_place_ready") is not True:
        return False
    if actionability.get("affordance_ready") is False:
        return False
    return bool(has_interaction_point(candidate) and has_placement_contract(candidate))


def required_next_for_place_candidate(candidate: JsonDict) -> str:
    actionability = as_dict(candidate.get("actionability"))
    if actionability.get("needs_alignment") is True:
        return "align_receptacle_then_observe"
    if actionability.get("needs_approach") is True:
        return "approach_receptacle_then_observe"
    return "observe:refresh"


def selected_option_id(args: argparse.Namespace) -> str:
    if args.option_id:
        return str(args.option_id).strip()
    raw = str(args.selection_json or "").strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("selected_option_id") or data.get("option_id") or "").strip()


def validate_context(context: JsonDict) -> list[JsonDict]:
    errors: list[JsonDict] = []
    if context.get("schema") != DECISION_CONTEXT_SCHEMA:
        errors.append(
            {
                "type": "invalid_context_schema",
                "expected": DECISION_CONTEXT_SCHEMA,
                "actual": context.get("schema"),
            }
        )
    if context.get("forbidden_private_fields_absent") is False:
        errors.append({"type": "forbidden_private_fields_present"})
    stale = context_stale_error(context)
    if stale:
        errors.append(stale)
    return errors


def context_map_backend(context: JsonDict) -> str:
    backend = as_dict(as_dict(context.get("navigation")).get("map_backend")).get("backend")
    return str(backend or "").strip()


def runtime_map_backend(runtime_environment: JsonDict) -> str:
    backend = as_dict(runtime_environment.get("map_backend")).get("backend")
    return str(backend or "").strip()


def validate_context_runtime_backend(context: JsonDict, runtime_environment: JsonDict) -> JsonDict | None:
    context_backend = context_map_backend(context)
    runtime_backend = runtime_map_backend(runtime_environment)
    if not context_backend or not runtime_backend or context_backend == runtime_backend:
        return None
    return {
        "type": "decision_context_map_backend_mismatch",
        "context_backend": context_backend,
        "runtime_backend": runtime_backend,
        "required_next": "robot_cleaner_prepare_decision_turn",
        "message": "Decision context was prepared with a different map backend than the executor is using.",
    }


def warning_blocks_physical_action(warning: JsonDict) -> bool:
    return warning.get("blocking") is not False


def navigation_only_perception_allows_option(perception: JsonDict, option: JsonDict) -> bool:
    """Allow waypoint navigation with depth/costmap perception, but not task actions."""
    if str(option.get("kind") or "") not in NAVIGATION_ONLY_WAYPOINT_KINDS:
        return False
    mode = str(perception.get("perception_mode") or "").strip().lower()
    result_type = str(perception.get("result_type") or "").strip()
    return (
        mode in {"navigation_only", "navigation-only", "nav_only", "nav-only"}
        and result_type == "navigation_only_observed"
        and perception.get("status") == "success"
        and perception.get("online_safe") is True
    )


def validate_resolved_move_safety(context: JsonDict, action: str, option: JsonDict) -> list[JsonDict]:
    errors: list[JsonDict] = []
    if action not in MOVE_ACTIONS:
        errors.append({"type": "invalid_move_action", "action": action})
        return errors
    action_safety = move_action_safety_record(context, action)
    if action_safety.get("safe") is False:
        errors.append(
            {
                "type": "move_action_blocked_by_costmap",
                "action": action,
                "reason": action_safety.get("reason"),
            }
        )
    if action in TRANSLATION_ACTIONS:
        if not action_safety:
            if not option_allows_bootstrap_moveahead(option, action):
                errors.append(
                    {
                        "type": "move_action_missing_costmap_safety",
                        "action": action,
                        "required_next": "observe_or_rotate_before_translation",
                    }
                )
        else:
            observed_ratio = number_or_none(action_safety.get("observed_ratio"))
            min_observed_ratio = required_min_observed_ratio_for_action(action, action_safety)
            if (
                observed_ratio is not None
                and min_observed_ratio is not None
                and float(observed_ratio) < float(min_observed_ratio)
            ):
                errors.append(
                    {
                        "type": "move_action_low_observed_ratio",
                        "action": action,
                        "observed_ratio": round(float(observed_ratio), 4),
                        "min_observed_ratio": round(float(min_observed_ratio), 4),
                        "required_next": "observe_or_rotate_before_translation",
                    }
                )
    if (
        action == "MoveAhead"
        and as_dict(context.get("perception")).get("obstacle_ahead") is True
        and not option_uses_committed_route_costmap(option, action, action_safety)
    ):
        errors.append({"type": "moveahead_blocked_by_perception"})
    return errors


def validate_move_action_fields(context: JsonDict, action: str) -> list[JsonDict]:
    return validate_resolved_move_safety(context, action, {"action": action})


def validate_option(
    context: JsonDict,
    option: JsonDict,
    *,
    allow_stale_context: bool = False,
) -> list[JsonDict]:
    errors: list[JsonDict] = []
    option_id = str(option.get("option_id") or "")
    if not option_id:
        errors.append({"type": "missing_option_id"})
    if option.get("executable_now") is False:
        errors.append({"type": "option_not_executable_now", "option_id": option_id})

    physical = bool(option.get("physical_action")) or str(option.get("kind") or "") in PHYSICAL_KINDS
    if physical:
        perception = as_dict(context.get("perception"))
        if (
            perception.get("structured_perception_available") is False
            and not navigation_only_perception_allows_option(perception, option)
        ):
            errors.append(
                {
                    "type": "structured_perception_unavailable",
                    "required_next": "observe:refresh",
                }
            )
        warnings = [
            item
            for item in as_list(context.get("consistency_warnings"))
            if isinstance(item, dict) and warning_blocks_physical_action(item)
        ]
        if warnings and not allow_stale_context:
            errors.append(
                {
                    "type": "context_consistency_warning",
                    "required_next": "observe:refresh",
                    "warnings": warnings,
                }
            )
    if option.get("kind") == "move_action":
        action = resolved_move_action_for_option(context, option)
        errors.extend(validate_resolved_move_safety(context, action, option))
    if option.get("kind") == "explore_inspection_waypoint":
        waypoint_id = waypoint_id_for_option(option)
        if not waypoint_id:
            errors.append({"type": "inspection_waypoint_missing_waypoint_id"})
        if not str(option.get("option_id") or "").startswith("explore:inspection_waypoint:"):
            errors.append({"type": "invalid_inspection_waypoint_option_id", "option_id": option_id})
    if option.get("kind") == "continue_active_waypoint_goal":
        active_goal = as_dict(option.get("active_waypoint_goal"))
        if option_id != "continue:active_waypoint_goal":
            errors.append({"type": "invalid_continue_active_waypoint_option_id", "option_id": option_id})
        if not str(active_goal.get("waypoint_id") or "").strip():
            errors.append({"type": "continue_active_waypoint_missing_goal"})
    if option.get("kind") == "explore_frontier_cluster":
        action = resolved_move_action_for_option(context, option)
        target = as_dict(option.get("frontier_cluster_target"))
        route_step = as_dict(option.get("route_step"))
        if not target.get("cell"):
            errors.append({"type": "explore_frontier_cluster_missing_target_cell"})
        if not str(option.get("option_id") or "").startswith("explore:frontier_cluster:"):
            errors.append({"type": "invalid_explore_frontier_cluster_option_id", "option_id": option_id})
        if route_step and not route_step.get("goal_cell"):
            errors.append({"type": "explore_frontier_cluster_route_step_missing_goal_cell"})
        errors.extend(validate_resolved_move_safety(context, action, option))
    if option.get("kind") == "explore_frontier":
        action = resolved_move_action_for_option(context, option)
        frontier_target = as_dict(option.get("frontier_target"))
        if not frontier_target.get("cell"):
            errors.append({"type": "explore_frontier_missing_target_cell"})
        if not str(option.get("option_id") or "").startswith("explore:frontier:"):
            errors.append({"type": "invalid_explore_frontier_option_id", "option_id": option_id})
        errors.extend(validate_resolved_move_safety(context, action, option))
    if option.get("kind") == "explore_route_step":
        action = resolved_move_action_for_option(context, option)
        route_step = as_dict(option.get("route_step"))
        if not route_step.get("route_id"):
            errors.append({"type": "explore_route_step_missing_route_id"})
        if not route_step.get("goal_cell"):
            errors.append({"type": "explore_route_step_missing_goal_cell"})
        if not str(option.get("option_id") or "").startswith("explore:route_step:"):
            errors.append({"type": "invalid_explore_route_step_option_id", "option_id": option_id})
        errors.extend(validate_resolved_move_safety(context, action, option))
    if option.get("kind") == "explore_waypoint":
        action = resolved_move_action_for_option(context, option)
        waypoint_target = as_dict(option.get("waypoint_target"))
        if not waypoint_target.get("cell"):
            errors.append({"type": "explore_waypoint_missing_target_cell"})
        if not str(option.get("option_id") or "").startswith("explore:waypoint:"):
            errors.append({"type": "invalid_explore_waypoint_option_id", "option_id": option_id})
        errors.extend(validate_resolved_move_safety(context, action, option))
    if option.get("kind") == "recovery_action":
        action = resolved_move_action_for_option(context, option)
        if not str(option.get("option_id") or "").startswith("recover:"):
            errors.append({"type": "invalid_recovery_option_id", "option_id": option_id})
        if action not in MOVE_ACTIONS:
            errors.append({"type": "invalid_recovery_action", "action": action})
        errors.extend(validate_resolved_move_safety(context, action, option))
    if option.get("kind") == "service_action":
        action = str(option.get("action") or "")
        held = bool(as_dict(as_dict(context.get("worklist")).get("held_object")).get("holding_object"))
        if action == "pick-object" and held:
            errors.append({"type": "cannot_pick_while_holding_object"})
        if action == "place-object" and not held:
            errors.append({"type": "cannot_place_without_holding_object"})
        if action not in {"pick-object", "place-object"}:
            errors.append({"type": "invalid_service_action", "action": action})
        else:
            candidate = find_candidate_for_option(context, option)
            if not candidate:
                errors.append(
                    {
                        "type": "service_candidate_not_found",
                        "option_id": option_id,
                        "candidate_ref": option.get("candidate_ref"),
                    }
                )
            else:
                actionability = as_dict(candidate.get("actionability"))
                executor_checks = as_dict(candidate.get("executor_checks"))
                if action == "pick-object" and not (
                    actionability.get("pickup_now") is True and actionability.get("reachable") is True
                ):
                    errors.append(
                        {
                            "type": "candidate_not_pickup_ready",
                            "required_next": "align_pickup_target_or_observe",
                            "actionability": actionability,
                        }
                    )
                if action == "place-object" and not place_candidate_executor_ready(candidate):
                    errors.append(
                        {
                            "type": "candidate_not_place_executor_ready",
                            "required_next": required_next_for_place_candidate(candidate),
                            "actionability": actionability,
                            "executor_checks": executor_checks,
                        }
                    )
    if option.get("kind") == "place_precheck":
        held = bool(as_dict(as_dict(context.get("worklist")).get("held_object")).get("holding_object"))
        candidate = find_candidate_for_option(context, option)
        if not held:
            errors.append({"type": "cannot_place_precheck_without_holding_object"})
        if not candidate:
            errors.append(
                {
                    "type": "service_candidate_not_found",
                    "option_id": option_id,
                    "candidate_ref": option.get("candidate_ref"),
                }
            )
        elif not place_candidate_precheck_ready(candidate):
            errors.append(
                {
                    "type": "candidate_not_place_precheck_ready",
                    "required_next": required_next_for_place_candidate(candidate),
                    "actionability": as_dict(candidate.get("actionability")),
                    "executor_checks": as_dict(candidate.get("executor_checks")),
                }
            )
    return errors


def candidate_matches(candidate: JsonDict, ref: JsonDict) -> bool:
    if not ref:
        return False
    for key in ("candidate_id", "track_id", "candidate_signature", "surface_candidate_id"):
        expected = ref.get(key)
        if expected and str(candidate.get(key) or "") == str(expected):
            return True
    return False


def iter_worklist_candidates(context: JsonDict) -> list[JsonDict]:
    worklist = as_dict(context.get("worklist"))
    current_view = as_dict(worklist.get("current_view"))
    candidates: list[JsonDict] = []
    for key in (
        "pickup_candidates",
        "receptacle_candidates",
        "surface_candidates",
        "cleanable_candidates",
        "obstacles",
    ):
        candidates.extend(dict(item) for item in as_list(current_view.get(key)) if isinstance(item, dict))
    memory = as_dict(worklist.get("memory"))
    for key in ("pickup_targets", "receptacles"):
        candidates.extend(dict(item) for item in as_list(memory.get(key)) if isinstance(item, dict))
    return candidates


def find_candidate_for_option(context: JsonDict, option: JsonDict) -> JsonDict | None:
    ref = as_dict(option.get("candidate_ref"))
    for candidate in iter_worklist_candidates(context):
        if candidate_matches(candidate, ref):
            return candidate
    return None


def candidate_executor_payload(candidate: JsonDict, *, role: str) -> JsonDict:
    geometry = as_dict(candidate.get("geometry") or candidate.get("last_observation"))
    bbox = as_dict(candidate.get("bbox")) or as_dict(geometry.get("bbox"))
    actionability = as_dict(candidate.get("actionability"))
    payload: JsonDict = {
        "schema_version": 1,
        "role": role,
        "id": candidate.get("candidate_id") or candidate.get("track_id"),
        "track_id": candidate.get("track_id"),
        "candidate_signature": candidate.get("candidate_signature"),
        "surface_candidate_id": candidate.get("surface_candidate_id"),
        "surface_region_id": candidate.get("surface_region_id"),
        "surface_candidate_source": candidate.get("surface_candidate_source"),
        "label": candidate.get("label"),
        "raw_label": candidate.get("raw_label") or candidate.get("label"),
        "task_semantic_class": candidate.get("task_semantic_class") or candidate.get("task_class"),
        "region_type": candidate.get("region_type"),
        "parent_object": candidate.get("parent_object"),
        "parent_label": candidate.get("parent_label"),
        "confidence": candidate.get("confidence"),
        "position_hint": candidate.get("position_hint") or geometry.get("position_hint"),
        "surface_hint": candidate.get("surface_hint") or geometry.get("surface_hint"),
        "source": candidate.get("source"),
        "bbox": bbox,
        "geometry": geometry,
        "executor_bridge": "execute_option_v1",
    }
    for key in (
        "interaction_point",
        "center_3d",
        "parent_bbox",
        "region_bbox",
        "geometry_checks",
        "occupancy_checks",
        "memory_checks",
        "executor_checks",
        "free_space_completion",
        "placement_safety_contract",
    ):
        value = candidate.get(key)
        if isinstance(value, dict):
            payload[key] = dict(value)
    for key in (
        "visible_occupants",
        "placement_avoidance_candidates",
        "placement_points",
        "blocked_by",
        "affordance",
        "rejection_reasons",
    ):
        value = candidate.get(key)
        if isinstance(value, list):
            copied = [dict(item) if isinstance(item, dict) else item for item in value]
            payload[key] = copied[:16] if key == "placement_avoidance_candidates" else copied[:8]
    if isinstance(payload.get("placement_points"), list):
        payload["placement_point_count"] = len(payload["placement_points"])
    if bbox and all(key in bbox for key in ("x", "y", "w", "h")):
        try:
            x = float(bbox.get("x") or 0)
            y = float(bbox.get("y") or 0)
            w = float(bbox.get("w") or 0)
            h = float(bbox.get("h") or 0)
            payload.setdefault("center", {"x": round(x + w * 0.5, 3), "y": round(y + h * 0.5, 3)})
            if role == "place" and "interaction_point" not in payload:
                payload["interaction_point"] = {
                    "x": round(x + w * 0.5, 3),
                    "y": round(y + h * 0.88, 3),
                }
            elif role != "place" and "interaction_point" not in payload:
                payload["interaction_point"] = dict(payload["center"])
        except (TypeError, ValueError):
            pass
    for source_key, target_key in (
        ("bearing_deg", "bearing_deg"),
        ("distance_m", "distance"),
        ("ground_distance_m", "ground_distance"),
        ("height_m", "height"),
        ("area_ratio", "area_ratio"),
        ("bottom_y_ratio", "bottom_y_ratio"),
    ):
        if source_key in geometry:
            payload[target_key] = geometry.get(source_key)
    for key in (
        "reachable",
        "blocked",
        "pickup_now",
        "place_now",
        "visual_place_ready",
        "affordance_ready",
        "final_place_ready",
        "failed_recently",
        "needs_alignment",
        "needs_approach",
        "is_floor_level",
        "is_support_surface",
        "visual_box_ambiguous",
        "broad_front_receptacle",
        "front_edge_receptacle",
    ):
        if key in actionability:
            payload[key] = actionability.get(key)
        elif key in candidate:
            payload[key] = candidate.get(key)
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def safe_precheck_result(data: JsonDict) -> JsonDict:
    allowed = (
        "status",
        "result_type",
        "message",
        "precheck_supported",
        "precheck_ok",
        "precheck_reason",
        "suggested_recovery",
        "executor_action_hint",
        "visual_receptacle_grounding_passed",
        "visual_receptacle_grounding_result_type",
        "interactable_current_pose",
        "interactable_pose_count",
        "interactable_distance_bucket",
        "interactable_angle_bucket",
        "placement_attempt_count",
        "placement_attempt_modes",
        "placement_point_source",
        "placement_clearance_contract_applied",
        "placement_execution_mode",
        "placement_target_required",
        "placement_target_resolution_error_m",
        "placement_target_resolution_tolerance_m",
        "failed_candidate_id",
        "candidate_id",
        "http_status",
    )
    return {key: data.get(key) for key in allowed if data.get(key) not in (None, "", [], {})}


def write_place_precheck_cache(
    context: JsonDict,
    option: JsonDict,
    candidate: JsonDict,
    payload: JsonDict,
    result: ScriptResult,
) -> None:
    sources = as_dict(context.get("sources"))
    perception_source = as_dict(sources.get("perception"))
    cache = {
        "schema": PLACE_PRECHECK_CACHE_SCHEMA,
        "cached_at": now_iso(),
        "context_generated_at": context.get("generated_at"),
        "perception_source_last_modified": perception_source.get("last_modified"),
        "selected_option_id": option.get("option_id"),
        "candidate_ref": option.get("candidate_ref"),
        "candidate_payload_ref": {
            "id": payload.get("id"),
            "surface_candidate_id": payload.get("surface_candidate_id"),
            "surface_candidate_source": payload.get("surface_candidate_source"),
            "label": payload.get("label"),
        },
        "result": safe_precheck_result(result.data),
    }
    PLACE_PRECHECK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLACE_PRECHECK_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_observe_refresh(
    *,
    timeout_seconds: int,
    vision_timeout_seconds: int | None = None,
    yolo_timeout_seconds: int | None = None,
    perception_mode: str = "full",
) -> JsonDict:
    vision_timeout = max(1, int(vision_timeout_seconds or timeout_seconds))
    yolo_timeout = max(1, int(yolo_timeout_seconds or timeout_seconds))
    vision = run_script(GET_VISION_SCRIPT, [], timeout_seconds=vision_timeout)
    if vision.returncode != 0 or vision.data.get("status") != "success":
        return {
            "status": "error",
            "result_type": "observe_refresh_vision_failed",
            "vision": script_result_payload(vision),
        }
    image_path = str(vision.data.get("image_path") or "")
    if not image_path:
        return {
            "status": "error",
            "result_type": "observe_refresh_missing_image_path",
            "vision": script_result_payload(vision),
        }
    camera = vision.data.get("camera") if isinstance(vision.data.get("camera"), dict) else {}
    mode = str(perception_mode or "full").strip().lower()
    if mode in {"navigation_only", "navigation-only", "nav_only", "nav-only"}:
        depth_path = str(vision.data.get("depth_path") or "")
        navigation_data = {
            "status": "success",
            "result_type": "navigation_only_observed",
            "perception_mode": "navigation_only",
            "perception_backend": "depth_local_costmap",
            "online_safe": True,
            "image_path": image_path,
            "depth_path": depth_path,
            "camera": camera,
            "candidate_count": 0,
            "pickup_target_detected": False,
            "place_receptacle_detected": False,
            "direct_pickup_detected": False,
            "direct_place_detected": False,
            "floor_trash_detected": False,
            "frontier_exists": None,
            "recommended_action": None,
            "reason": "Skipped semantic YOLO/pointcloud task perception while continuing an active waypoint route.",
        }
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        (MEMORY_DIR / "yolo-current-rgbd.json").write_text(
            json.dumps(navigation_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        analysis = ScriptResult(
            command=["navigation-only-perception"],
            returncode=0,
            stdout=json.dumps(navigation_data, ensure_ascii=False),
            stderr="",
            data=navigation_data,
        )
        return {
            "status": "success",
            "result_type": "observe_refresh_executed",
            "physical_action_executed": False,
            "perception_mode": "navigation_only",
            "vision": script_result_payload(vision),
            "perception": script_result_payload(analysis),
            "perception_written": str(MEMORY_DIR / "yolo-current-rgbd.json"),
        }
    yolo_args = ["--image", image_path, "--timeout", str(yolo_timeout)]
    depth_path = str(vision.data.get("depth_path") or "")
    if depth_path:
        yolo_args.extend(["--depth", depth_path])
    if camera:
        yolo_args.extend(["--camera-json", json.dumps(camera, ensure_ascii=False, separators=(",", ":"))])
    analysis = run_script(YOLO_SCRIPT, yolo_args, timeout_seconds=yolo_timeout + 10)
    if analysis.data:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        (MEMORY_DIR / "yolo-current-rgbd.json").write_text(
            json.dumps(analysis.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "success" if analysis.returncode == 0 and analysis.data.get("status") == "success" else "error",
        "result_type": "observe_refresh_executed",
        "physical_action_executed": False,
        "perception_mode": "full",
        "vision": script_result_payload(vision),
        "perception": script_result_payload(analysis),
        "perception_written": str(MEMORY_DIR / "yolo-current-rgbd.json") if analysis.data else "",
    }


def run_move_step(
    context: JsonDict,
    option: JsonDict,
    *,
    action: str,
    timeout_seconds: int,
    result_type: str,
    extra: JsonDict | None = None,
) -> JsonDict:
    result = run_script(MOVE_SCRIPT, ["--action", action], timeout_seconds=timeout_seconds)
    ok = (
        result.returncode == 0
        and result.data.get("status") != "error"
        and result.data.get("lastActionSuccess", True) is not False
    )
    state_sync_option = dict(option)
    state_sync_option["kind"] = "move_action"
    state_sync_option["action"] = action
    state_sync = sync_option_result(
        context=context,
        option=state_sync_option,
        candidate=None,
        execution=result.data,
        success=ok,
        memory_dir=MEMORY_DIR,
    )
    payload = {
        "status": "success" if ok else "error",
        "result_type": result_type,
        "option_id": option.get("option_id"),
        "physical_action_executed": True,
        "resolved_action": action,
        "execution": script_result_payload(result),
        "state_sync": state_sync,
    }
    payload.update(extra or {})
    return payload


def run_waypoint_plan_step(
    context: JsonDict,
    option: JsonDict,
    *,
    plan: JsonDict,
    timeout_seconds: int,
    result_type: str,
) -> JsonDict:
    status = str(plan.get("status") or "")
    route = as_dict(plan.get("route"))
    action = str(plan.get("next_action") or route.get("next_action") or "").strip()
    if status == "active" and action:
        return run_move_step(
            context,
            option,
            action=action,
            timeout_seconds=timeout_seconds,
            result_type=result_type,
            extra={
                "waypoint_plan": plan,
                "waypoint_target": as_dict(option.get("waypoint_target")),
                "active_waypoint_goal": as_dict(option.get("active_waypoint_goal")),
                "route_step": as_dict(route.get("route_step")),
                "active_route": route,
                "one_step_only": True,
            },
        )
    if status == "reached":
        return {
            "status": "success",
            "result_type": "option_inspection_waypoint_reached",
            "option_id": option.get("option_id"),
            "physical_action_executed": False,
            "required_next": "observe:refresh",
            "message": "Inspection waypoint cell is reached; refresh perception to mark/inspect this waypoint.",
            "waypoint_plan": plan,
        }
    return {
        "status": "error",
        "result_type": "option_inspection_waypoint_plan_unavailable",
        "option_id": option.get("option_id"),
        "physical_action_executed": False,
        "required_next": plan.get("required_next") or "recover_or_choose_new_waypoint",
        "waypoint_plan": plan,
    }


def run_selected_option(
    context: JsonDict,
    option: JsonDict,
    *,
    timeout_seconds: int,
    dry_run: bool,
    strict_visual_grounding: bool,
) -> JsonDict:
    option_id = str(option.get("option_id") or "")
    kind = str(option.get("kind") or "")
    action = str(option.get("action") or "")

    if dry_run:
        return {
            "status": "success",
            "result_type": "option_dry_run_validated",
            "option_id": option_id,
            "option": option,
            "physical_action_executed": False,
        }

    if option_id == "observe:refresh" or kind == "perception":
        return run_observe_refresh(timeout_seconds=timeout_seconds)
    if option_id == "done:probe" or kind == "completion_probe":
        return {
            "status": "success",
            "result_type": "done_probe_result",
            "option_id": option_id,
            "physical_action_executed": False,
            "done_readiness": context.get("done_readiness"),
        }
    if kind == "move_action":
        return run_move_step(
            context,
            option,
            action=resolved_move_action_for_option(context, option),
            timeout_seconds=timeout_seconds,
            result_type="option_move_executed",
        )
    if kind == "explore_inspection_waypoint":
        waypoint_id = waypoint_id_for_option(option)
        plan = plan_to_inspection_waypoint(
            waypoint_id,
            memory_dir=MEMORY_DIR,
            persist=True,
        )
        return run_waypoint_plan_step(
            context,
            option,
            plan=plan,
            timeout_seconds=timeout_seconds,
            result_type="option_explore_inspection_waypoint_step_executed",
        )
    if kind == "continue_active_waypoint_goal":
        plan = continue_active_waypoint_goal(
            memory_dir=MEMORY_DIR,
            persist=True,
        )
        return run_waypoint_plan_step(
            context,
            option,
            plan=plan,
            timeout_seconds=timeout_seconds,
            result_type="option_continue_active_waypoint_step_executed",
        )
    if kind == "explore_frontier_cluster":
        resolved_action = resolved_move_action_for_option(context, option)
        return run_move_step(
            context,
            option,
            action=resolved_action,
            timeout_seconds=timeout_seconds,
            result_type="option_explore_frontier_cluster_step_executed",
            extra={
                "frontier_cluster_target": as_dict(option.get("frontier_cluster_target")),
                "route_step": as_dict(option.get("route_step")),
                "active_route": as_dict(option.get("active_route")),
                "resolved_step_option_id": option.get("resolved_step_option_id"),
                "one_step_only": True,
            },
        )
    if kind == "explore_frontier":
        resolved_action = resolved_move_action_for_option(context, option)
        return run_move_step(
            context,
            option,
            action=resolved_action,
            timeout_seconds=timeout_seconds,
            result_type="option_explore_frontier_step_executed",
            extra={
                "frontier_target": as_dict(option.get("frontier_target")),
                "resolved_step_option_id": option.get("resolved_step_option_id"),
                "one_step_only": True,
            },
        )
    if kind == "explore_route_step":
        resolved_action = resolved_move_action_for_option(context, option)
        return run_move_step(
            context,
            option,
            action=resolved_action,
            timeout_seconds=timeout_seconds,
            result_type="option_explore_route_step_executed",
            extra={
                "route_step": as_dict(option.get("route_step")),
                "active_route": as_dict(option.get("active_route")),
                "resolved_step_option_id": option.get("resolved_step_option_id"),
                "one_step_only": True,
            },
        )
    if kind == "explore_waypoint":
        resolved_action = resolved_move_action_for_option(context, option)
        return run_move_step(
            context,
            option,
            action=resolved_action,
            timeout_seconds=timeout_seconds,
            result_type="option_explore_waypoint_step_executed",
            extra={
                "waypoint_target": as_dict(option.get("waypoint_target")),
                "resolved_step_option_id": option.get("resolved_step_option_id"),
                "one_step_only": True,
            },
        )
    if kind == "recovery_action":
        resolved_action = resolved_move_action_for_option(context, option)
        return run_move_step(
            context,
            option,
            action=resolved_action,
            timeout_seconds=timeout_seconds,
            result_type="option_recovery_step_executed",
            extra={
                "dead_end_context": as_dict(option.get("dead_end_context")),
                "one_step_only": True,
            },
        )
    if kind == "clean_action":
        result = run_script(CLEAN_SCRIPT, [], timeout_seconds=timeout_seconds)
        ok = result.returncode == 0 and result.data.get("status") != "error"
        state_sync = sync_option_result(
            context=context,
            option=option,
            candidate=None,
            execution=result.data,
            success=ok,
            memory_dir=MEMORY_DIR,
        )
        return {
            "status": "success" if ok else "error",
            "result_type": "option_clean_executed",
            "option_id": option_id,
            "physical_action_executed": True,
            "execution": script_result_payload(result),
            "state_sync": state_sync,
        }
    if kind == "place_precheck":
        candidate = find_candidate_for_option(context, option)
        if not candidate:
            return {
                "status": "error",
                "result_type": "error_option_candidate_not_found",
                "option_id": option_id,
                "candidate_ref": option.get("candidate_ref"),
            }
        payload = candidate_executor_payload(candidate, role="place")
        args = [
            "--precheck-only",
            "--candidate-json",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ]
        if strict_visual_grounding:
            args.insert(1, "--strict-visual-grounding")
        result = run_script(PLACE_SCRIPT, args, timeout_seconds=timeout_seconds)
        write_place_precheck_cache(context, option, candidate, payload, result)
        ok = result.returncode == 0 and result.data.get("precheck_ok") is True
        return {
            "status": "success" if ok else "error",
            "result_type": "option_place_precheck_executed",
            "option_id": option_id,
            "physical_action_executed": False,
            "candidate_payload": payload,
            "precheck_cache_path": str(PLACE_PRECHECK_CACHE_PATH),
            "execution": script_result_payload(result),
        }
    if kind == "service_action":
        candidate = find_candidate_for_option(context, option)
        if not candidate:
            return {
                "status": "error",
                "result_type": "error_option_candidate_not_found",
                "option_id": option_id,
                "candidate_ref": option.get("candidate_ref"),
            }
        role = "pickup" if action == "pick-object" else "place"
        payload = candidate_executor_payload(candidate, role=role)
        script = PICK_SCRIPT if action == "pick-object" else PLACE_SCRIPT
        args = ["--candidate-json", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))]
        if strict_visual_grounding:
            args.insert(0, "--strict-visual-grounding")
        result = run_script(script, args, timeout_seconds=timeout_seconds)
        expected = "pickup_executed" if action == "pick-object" else "place_executed"
        ok = result.returncode == 0 and (
            result.data.get("status") == "success"
            or result.data.get("result_type") == expected
            or result.data.get("lastActionSuccess") is True
        )
        state_sync = sync_option_result(
            context=context,
            option=option,
            candidate=candidate,
            execution=result.data,
            success=ok,
            memory_dir=MEMORY_DIR,
        )
        return {
            "status": "success" if ok else "error",
            "result_type": f"option_{role}_executed",
            "option_id": option_id,
            "physical_action_executed": True,
            "candidate_payload": payload,
            "execution": script_result_payload(result),
            "state_sync": state_sync,
        }
    return {
        "status": "error",
        "result_type": "error_unsupported_option_kind",
        "option_id": option_id,
        "kind": kind,
        "action": action,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one selected option_id from decision context.")
    parser.add_argument("--option-id", default="", help="Selected option_id from option_set.options.")
    parser.add_argument(
        "--selection-json",
        default="",
        help="Optional JSON containing selected_option_id, as returned by an LLM/OpenClaw agent.",
    )
    parser.add_argument(
        "--context",
        default=str(DEFAULT_CONTEXT_PATH),
        help="Path to robot_cleaner_decision_context_v1 JSON.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Executor script timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without executing scripts.")
    parser.add_argument(
        "--allow-stale-context",
        action="store_true",
        help="Allow physical actions even when consistency_warnings are present.",
    )
    parser.add_argument(
        "--no-strict-visual-grounding",
        action="store_true",
        help="Do not pass --strict-visual-grounding to pick/place.",
    )
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_environment = apply_runtime_environment(override_existing=True)
    previous_runtime_env = (
        runtime_environment.get("previous_env") if isinstance(runtime_environment.get("previous_env"), dict) else {}
    )
    option_id = selected_option_id(args)
    try:
        if not option_id:
            result = {
                "status": "error",
                "result_type": "error_missing_option_id",
                "message": "Pass --option-id or --selection-json with selected_option_id.",
                "runtime_environment": runtime_environment,
            }
            json_print(result, compact=args.format == "compact")
            return 2

        context_path = Path(args.context)
        if not context_path.is_absolute():
            context_path = REPO_ROOT / context_path
        context = load_json(context_path)
        context_errors = validate_context(context)
        backend_mismatch = validate_context_runtime_backend(context, runtime_environment)
        if backend_mismatch:
            context_errors.append(backend_mismatch)
        option = find_option(context, option_id) if not context_errors else None
        if option is None and not context_errors:
            context_errors.append(
                {
                    "type": "selected_option_not_found",
                    "selected_option_id": option_id,
                    "available_option_ids": [str(item.get("option_id") or "") for item in option_list(context)],
                }
            )
        option_errors = (
            validate_option(context, option or {}, allow_stale_context=bool(args.allow_stale_context))
            if option is not None
            else []
        )
        if context_errors or option_errors:
            result = {
                "status": "error",
                "result_type": "error_option_validation_failed",
                "selected_option_id": option_id,
                "context_path": str(context_path),
                "context_errors": context_errors,
                "option_errors": option_errors,
                "runtime_environment": runtime_environment,
            }
            append_trace({"event": "option_validation_failed", "result": result})
            json_print(result, compact=args.format == "compact")
            return 1

        execution = run_selected_option(
            context,
            option,
            timeout_seconds=max(1, int(args.timeout)),
            dry_run=bool(args.dry_run),
            strict_visual_grounding=not bool(args.no_strict_visual_grounding),
        )
        context_lifecycle_update: JsonDict = {}
        if not bool(args.dry_run) and execution.get("result_type") != "done_probe_result":
            context_lifecycle_update = mark_context_stale_after_execution(
                context_path,
                option_id=option_id,
                execution=execution,
            )
        result = {
            "status": execution.get("status", "error"),
            "result_type": "selected_option_handled",
            "selected_option_id": option_id,
            "context_path": str(context_path),
            "runtime_environment": runtime_environment,
            "option": option,
            "execution": execution,
            "context_lifecycle_update": context_lifecycle_update,
            "trace_path": str(TRACE_PATH),
        }
        append_trace({"event": "selected_option_handled", "result": result})
        json_print(result, compact=args.format == "compact")
        return 0 if result["status"] == "success" else 1
    finally:
        restore_runtime_environment(previous_runtime_env)


if __name__ == "__main__":
    raise SystemExit(main())
