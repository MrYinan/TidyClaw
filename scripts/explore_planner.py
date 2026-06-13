#!/usr/bin/env python3
"""Loop-aware exploration planning for the public decision context.

This module is intentionally read-only.  It consumes the existing position map,
local costmap, and exploration summary, then emits compact planning facts for
the LLM-facing decision context.  The executor remains authoritative: any
planner output must still become an option and pass execute_option validation
before a physical action is run.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scripts.position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        HEADING_VECTORS,
        format_cell,
        heading_between,
        left_heading,
        manhattan_distance,
        neighbor_for_action,
        parse_cell,
        right_heading,
    )
except ImportError:  # pragma: no cover - direct script execution
    from position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        HEADING_VECTORS,
        format_cell,
        heading_between,
        left_heading,
        manhattan_distance,
        neighbor_for_action,
        parse_cell,
        right_heading,
    )

try:
    from scripts.map_backend import load_map_backend
except ImportError:  # pragma: no cover - direct script execution
    from map_backend import load_map_backend


JsonDict = dict[str, Any]

EXPLORE_PLAN_SCHEMA = "robot_cleaner_explore_plan_v1"
TRANSLATION_ACTIONS = ("MoveAhead", "MoveLeft", "MoveRight", "MoveBack")
ROTATION_ACTIONS = ("RotateLeft", "RotateRight")
LATERAL_ACTIONS = {"MoveLeft", "MoveRight"}
BLOCKED_CELL_STATES = {CELL_OCCUPIED, CELL_INFLATED}
MAX_WAYPOINT_CANDIDATES = 4
MAX_SUPPRESSED_FRONTIERS = 8
MAX_SAME_LATERAL_WAYPOINT_STREAK = 2


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
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := clean_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := clean_empty(item)) not in (None, "", [], {})]
    return value


def number_or_none(value: Any, *, digits: int = 4) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def safe_parse_cell(cell: Any) -> tuple[int, int] | None:
    try:
        return parse_cell(cell)
    except Exception:
        return None


def safe_neighbor_for_action(cell: str, heading: str, action: str) -> str:
    try:
        return neighbor_for_action(cell, heading, action)
    except Exception:
        return str(cell)


def canonical_action(value: Any) -> str:
    text = str(value or "").strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    compact = text.replace("_", "").replace("-", "").lower()
    mapping = {
        "moveahead": "MoveAhead",
        "forward": "MoveAhead",
        "moveback": "MoveBack",
        "back": "MoveBack",
        "moveleft": "MoveLeft",
        "left": "MoveLeft",
        "moveright": "MoveRight",
        "right": "MoveRight",
        "rotateleft": "RotateLeft",
        "turnleft": "RotateLeft",
        "rotateright": "RotateRight",
        "turnright": "RotateRight",
        "lookup": "LookUp",
        "lookdown": "LookDown",
    }
    return mapping.get(compact, text)


def action_safety_record(navigation_costmap: JsonDict, action: str) -> JsonDict:
    return as_dict(as_dict(navigation_costmap.get("action_safety")).get(action))


def action_is_safe(navigation_costmap: JsonDict, action: str) -> bool | None:
    record = action_safety_record(navigation_costmap, action)
    if not record:
        return None
    value = record.get("safe")
    return value if isinstance(value, bool) else None


def cell_record(position_map: JsonDict, cell: str) -> JsonDict:
    return as_dict(as_dict(position_map.get("cells")).get(cell))


def cell_state(position_map: JsonDict, cell: str) -> str:
    return str(cell_record(position_map, cell).get("state") or "unknown")


def cell_visited(position_map: JsonDict, cell: str) -> bool:
    return cell_record(position_map, cell).get("visited") is True


def edge_is_blocked(position_map: JsonDict, source: str, target: str) -> bool:
    edges = as_dict(position_map.get("edges"))
    blocked = {str(item) for item in as_list(edges.get("blocked_edges"))}
    blocked.update(str(item) for item in as_list(position_map.get("blocked_edges")))
    return f"{source}->{target}" in blocked or f"{target}->{source}" in blocked


def recent_cells_from_exploration(exploration: JsonDict, *, limit: int = 8) -> list[str]:
    cells = [
        str(as_dict(item).get("cell") or "").strip()
        for item in as_list(exploration.get("recent_path"))
        if str(as_dict(item).get("cell") or "").strip()
    ]
    return cells[-max(1, int(limit)) :]


def frontier_cells(position_map: JsonDict, exploration: JsonDict) -> list[str]:
    cells: list[str] = []
    for raw in as_list(exploration.get("frontier_candidates")):
        cell = str(as_dict(raw).get("cell") or "").strip()
        if cell:
            cells.append(cell)
    for raw in as_list(position_map.get("frontiers")) or as_list(position_map.get("frontier_cells")):
        cell = str(raw or "").strip()
        if cell:
            cells.append(cell)
    result: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        if cell in seen or safe_parse_cell(cell) is None:
            continue
        seen.add(cell)
        result.append(cell)
    return result


def nearest_frontier_distance(cell: str, frontiers: Iterable[str]) -> int | None:
    distances: list[int] = []
    for frontier in frontiers:
        if safe_parse_cell(frontier) is None:
            continue
        try:
            distances.append(manhattan_distance(cell, frontier))
        except Exception:
            continue
    return min(distances) if distances else None


def distance_delta(current: int | None, target: int | None) -> int | None:
    if current is None or target is None:
        return None
    return int(current) - int(target)


def latest_rotation_undo(recent_actions: Sequence[Any], action: str) -> bool:
    canonical = [canonical_action(item) for item in as_list(recent_actions)]
    canonical = [item for item in canonical if item]
    if not canonical:
        return False
    last = canonical[-1]
    return (last == "RotateLeft" and action == "RotateRight") or (
        last == "RotateRight" and action == "RotateLeft"
    )


def same_action_streak(recent_actions: Sequence[Any], action: str) -> int:
    streak = 0
    for raw in reversed(as_list(recent_actions)):
        if canonical_action(raw) != action:
            break
        streak += 1
    return streak


def rotation_loop_active(exploration: JsonDict) -> bool:
    loop = as_dict(exploration.get("loop_warning"))
    reason = str(loop.get("reason") or "")
    return bool(
        loop.get("active") is True
        and (
            loop.get("severity") == "strong"
            or "rotation_oscillation" in reason
            or "many_rotations" in reason
        )
    )


def action_heading_after(current_heading: str, action: str) -> str:
    if action == "RotateLeft":
        return left_heading(current_heading)
    if action == "RotateRight":
        return right_heading(current_heading)
    return current_heading


def direction_from_delta(source: str, target: str) -> str:
    try:
        adjacent = heading_between(source, target)
    except Exception:
        adjacent = None
    if adjacent:
        return adjacent
    parsed_source = safe_parse_cell(source)
    parsed_target = safe_parse_cell(target)
    if parsed_source is None or parsed_target is None:
        return ""
    sx, sz = parsed_source
    tx, tz = parsed_target
    dx = tx - sx
    dz = tz - sz
    if abs(dx) >= abs(dz) and dx != 0:
        return "east" if dx > 0 else "west"
    if dz != 0:
        return "north" if dz > 0 else "south"
    return ""


def build_waypoint_candidates(
    *,
    position_map: JsonDict,
    navigation_costmap: JsonDict,
    exploration: JsonDict,
    current_cell: str,
    current_heading: str,
) -> list[JsonDict]:
    frontiers = frontier_cells(position_map, exploration)
    frontier_set = set(frontiers)
    recent_cells = set(recent_cells_from_exploration(exploration, limit=8))
    action_effects = as_dict(exploration.get("action_effects"))
    active_goal = as_dict(position_map.get("active_frontier_goal"))
    active_route = as_dict(position_map.get("active_route"))
    route_blocked = active_route.get("status") == "blocked"
    active_goal_cell = str(active_goal.get("cell") or "").strip()
    active_next_action = canonical_action(active_goal.get("next_action"))
    active_next_cell = str(active_goal.get("next_cell") or "").strip()
    coverage_patrol = as_dict(position_map.get("coverage_patrol"))
    coverage_active = coverage_patrol.get("active") is True
    coverage_target = str(coverage_patrol.get("target_cell") or "").strip()
    loop_active = rotation_loop_active(exploration)
    recent_actions = as_list(position_map.get("recent_actions"))
    current_nearest = nearest_frontier_distance(current_cell, frontiers)
    current_active_distance = (
        manhattan_distance(current_cell, active_goal_cell)
        if active_goal_cell and safe_parse_cell(active_goal_cell) is not None
        else None
    )
    current_coverage_distance = (
        manhattan_distance(current_cell, coverage_target)
        if coverage_target and safe_parse_cell(coverage_target) is not None
        else None
    )

    candidates: list[JsonDict] = []
    for action in TRANSLATION_ACTIONS:
        safety = action_safety_record(navigation_costmap, action)
        if safety.get("safe") is not True:
            continue
        if action in LATERAL_ACTIONS:
            lateral_streak = same_action_streak(recent_actions, action)
            if route_blocked:
                continue
            if lateral_streak >= MAX_SAME_LATERAL_WAYPOINT_STREAK:
                continue
        target_cell = safe_neighbor_for_action(current_cell, current_heading, action)
        if safe_parse_cell(target_cell) is None:
            continue
        target_state = cell_state(position_map, target_cell)
        if target_state in BLOCKED_CELL_STATES:
            continue
        if edge_is_blocked(position_map, current_cell, target_cell):
            continue

        effect = as_dict(action_effects.get(action))
        target_nearest = nearest_frontier_distance(target_cell, frontiers)
        nearest_delta = distance_delta(current_nearest, target_nearest)
        active_delta = distance_delta(
            current_active_distance,
            manhattan_distance(target_cell, active_goal_cell)
            if active_goal_cell and safe_parse_cell(active_goal_cell) is not None
            else None,
        )
        coverage_delta = distance_delta(
            current_coverage_distance,
            manhattan_distance(target_cell, coverage_target)
            if coverage_target and safe_parse_cell(coverage_target) is not None
            else None,
        )
        enters_frontier = target_cell in frontier_set or effect.get("enters_frontier") is True
        target_visited = cell_visited(position_map, target_cell) or effect.get("enters_visited_cell") is True
        target_recent = target_cell in recent_cells or effect.get("repeats_recent_path") is True
        follows_active_goal = bool(
            active_goal_cell
            and (
                action == active_next_action
                or target_cell == active_next_cell
                or (active_delta is not None and active_delta > 0)
            )
        )
        follows_coverage_patrol = bool(
            coverage_active
            and coverage_target
            and (target_cell == coverage_target or (coverage_delta is not None and coverage_delta > 0))
        )
        toward_frontier = bool(
            enters_frontier
            or effect.get("toward_frontier") is True
            or (nearest_delta is not None and nearest_delta > 0)
            or follows_active_goal
        )

        if loop_active:
            purpose = "break_rotation_loop_via_safe_translation"
        elif follows_active_goal:
            purpose = "advance_active_frontier_goal"
        elif enters_frontier:
            purpose = "enter_frontier"
        elif toward_frontier:
            purpose = "move_toward_frontier"
        elif follows_coverage_patrol:
            purpose = "coverage_patrol_step"
        elif target_state == CELL_FREE and not target_visited:
            purpose = "enter_unvisited_known_free_cell"
        else:
            continue

        score = 0.0
        if loop_active:
            score += 8.0
        if follows_active_goal:
            score += 5.0
        if follows_coverage_patrol:
            score += 3.2
        if action == "MoveAhead":
            score += 2.0
        if enters_frontier:
            score += 4.0
        if toward_frontier:
            score += 2.5
        if target_state == CELL_FREE and not target_visited:
            score += 1.5
        if nearest_delta is not None:
            score += max(-2.0, min(3.0, float(nearest_delta)))
        if active_delta is not None:
            score += max(-1.5, min(3.5, float(active_delta)))
        if coverage_delta is not None:
            score += max(-1.0, min(2.0, float(coverage_delta)))
        if target_recent:
            score -= 1.2
        if target_visited:
            score -= 0.7
        if action == "MoveBack":
            score -= 1.5
        if action in {"MoveLeft", "MoveRight"}:
            score -= 0.6

        reasons = [purpose]
        if loop_active:
            reasons.append("rotation_loop_active")
        if follows_active_goal:
            reasons.append("advances_active_frontier_goal")
        if follows_coverage_patrol:
            reasons.append("advances_coverage_patrol_target")
        if enters_frontier:
            reasons.append("target_cell_is_frontier")
        if toward_frontier and not enters_frontier:
            reasons.append("reduces_distance_to_frontier")
        if target_recent:
            reasons.append("target_cell_recently_visited")
        elif target_visited:
            reasons.append("target_cell_visited_but_safe_escape")

        candidates.append(
            clean_empty(
                {
                    "cell": target_cell,
                    "action": action,
                    "score": round(score, 3),
                    "purpose": purpose,
                    "target_state": target_state,
                    "target_visited": target_visited,
                    "target_recent": target_recent,
                    "nearest_frontier_distance_delta": nearest_delta,
                    "active_frontier_goal_cell": active_goal_cell,
                    "active_frontier_distance_delta": active_delta,
                    "coverage_patrol_target_cell": coverage_target,
                    "coverage_patrol_distance_delta": coverage_delta,
                    "frontier_cell": target_cell if target_cell in frontier_set else "",
                    "safety": {
                        "reason": safety.get("reason"),
                        "observed_ratio": number_or_none(safety.get("observed_ratio")),
                        "min_observed_ratio": number_or_none(safety.get("min_observed_ratio")),
                    },
                    "reasons": reasons,
                }
            )
        )

    candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("cell"))))
    return candidates[:MAX_WAYPOINT_CANDIDATES]


def build_suppressed_frontiers(
    *,
    position_map: JsonDict,
    exploration: JsonDict,
    waypoint_candidates: Sequence[JsonDict],
    current_cell: str,
    current_heading: str,
    recent_actions: Sequence[Any],
) -> list[JsonDict]:
    if not rotation_loop_active(exploration):
        return []

    has_waypoint_escape = bool(waypoint_candidates)
    action_effects = as_dict(exploration.get("action_effects"))
    suppressed: list[JsonDict] = []
    for raw in as_list(exploration.get("frontier_candidates")):
        candidate = as_dict(raw)
        cell = str(candidate.get("cell") or "").strip()
        action = canonical_action(candidate.get("first_action_hint"))
        if not cell or action not in ROTATION_ACTIONS:
            continue

        reasons: list[str] = []
        if has_waypoint_escape:
            reasons.append("safe_translation_waypoint_available")
        if latest_rotation_undo(recent_actions, action):
            reasons.append("immediate_rotation_undo")
        effect = as_dict(action_effects.get(action))
        if effect.get("effect") == "faces_blocked_known_cell":
            reasons.append("rotation_faces_known_blocked_cell")
        if effect.get("faces_unvisited_area") is False and not effect.get("toward_frontier"):
            reasons.append("rotation_does_not_face_useful_unvisited_area")

        if not reasons:
            continue
        suppressed.append(
            clean_empty(
                {
                    "cell": cell,
                    "action": action,
                    "reason": ";".join(reasons),
                    "severity": "block" if has_waypoint_escape else "advisory",
                    "frontier_score": candidate.get("score"),
                    "first_step_target_cell": candidate.get("first_step_target_cell"),
                    "route_risk": candidate.get("route_risk"),
                    "facing_cell": effect.get("facing_cell"),
                    "facing_cell_state": effect.get("facing_cell_state"),
                }
            )
        )
        if len(suppressed) >= MAX_SUPPRESSED_FRONTIERS:
            break
    return suppressed


def compact_frontier_rankings(exploration: JsonDict, suppressed: Sequence[JsonDict]) -> list[JsonDict]:
    suppressed_cells = {str(item.get("cell")) for item in suppressed if item.get("cell")}
    rankings: list[JsonDict] = []
    for raw in as_list(exploration.get("frontier_candidates"))[:8]:
        item = as_dict(raw)
        cell = str(item.get("cell") or "").strip()
        if not cell:
            continue
        rankings.append(
            clean_empty(
                {
                    "cell": cell,
                    "score": item.get("score"),
                    "distance_steps": item.get("distance_steps"),
                    "first_action_hint": item.get("first_action_hint"),
                    "route_risk": item.get("route_risk"),
                    "planner_status": "suppressed_rotation_loop_risk" if cell in suppressed_cells else "candidate",
                }
            )
        )
    return rankings


def build_recovery_actions_for_plan(
    *,
    mode: str,
    active_route: JsonDict,
    navigation_costmap: JsonDict,
) -> list[JsonDict]:
    actions: list[JsonDict] = []
    if mode == "route_blocked_recovery":
        blocked_action = canonical_action(active_route.get("next_action"))
        blocked_reason = str(active_route.get("blocked_reason") or "route_blocked")
        for action in ("RotateLeft", "RotateRight"):
            safety = action_safety_record(navigation_costmap, action)
            if safety.get("safe") is not True:
                continue
            actions.append(
                clean_empty(
                    {
                        "action": action,
                        "reason": "scan_for_alternative_route_after_active_route_blocked",
                        "trigger": "route_blocked_recovery",
                        "blocked_route_goal": active_route.get("goal_cell"),
                        "blocked_route_action": blocked_action,
                        "blocked_reason": blocked_reason,
                    }
                )
            )
        if not actions:
            for action in ("LookUp", "LookDown"):
                safety = action_safety_record(navigation_costmap, action)
                if safety and safety.get("safe") is not True:
                    continue
                actions.append(
                    {
                        "action": action,
                        "reason": "refresh_depth_before_replanning_blocked_route",
                        "trigger": "route_blocked_recovery",
                        "blocked_route_goal": active_route.get("goal_cell"),
                    }
                )
                break
        return actions

    if mode == "rotation_loop_scan_limited":
        return [
            {
                "action": "LookUp",
                "reason": "refresh_depth_with_higher_camera_pitch_before_repeating_rotation_loop",
                "trigger": "rotation_loop_scan_limited",
            }
        ]
    return []


def build_explore_plan(
    *,
    position_map: JsonDict,
    navigation_costmap: JsonDict | None = None,
    exploration: JsonDict | None = None,
) -> JsonDict:
    exploration = as_dict(exploration)
    navigation_costmap = as_dict(navigation_costmap)
    pose = as_dict(exploration.get("current_pose")) or as_dict(position_map.get("pose"))
    current_cell = str(pose.get("cell") or "0,0")
    current_heading = str(pose.get("heading") or "north")
    if safe_parse_cell(current_cell) is None:
        current_cell = "0,0"
    if current_heading not in HEADING_VECTORS:
        current_heading = "north"

    recent_actions = as_list(position_map.get("recent_actions"))
    active_goal = as_dict(position_map.get("active_frontier_goal"))
    active_route = as_dict(position_map.get("active_route"))
    route_blocked = active_route.get("status") == "blocked"
    coverage_patrol = as_dict(position_map.get("coverage_patrol"))
    waypoint_candidates = build_waypoint_candidates(
        position_map=position_map,
        navigation_costmap=navigation_costmap,
        exploration=exploration,
        current_cell=current_cell,
        current_heading=current_heading,
    )
    suppressed = build_suppressed_frontiers(
        position_map=position_map,
        exploration=exploration,
        waypoint_candidates=waypoint_candidates,
        current_cell=current_cell,
        current_heading=current_heading,
        recent_actions=recent_actions,
    )
    loop_active = rotation_loop_active(exploration)
    if route_blocked:
        mode = "route_blocked_recovery"
    elif active_route.get("status") == "active" and as_dict(active_route.get("route_step")):
        mode = "committed_route"
    elif loop_active and waypoint_candidates:
        mode = "break_rotation_loop"
    elif loop_active and suppressed:
        mode = "rotation_loop_scan_limited"
    elif waypoint_candidates:
        mode = "normal"
    else:
        mode = "no_safe_translation_waypoint"

    return clean_empty(
        {
            "schema": EXPLORE_PLAN_SCHEMA,
            "generated_at": now_iso(),
            "current_pose": {"cell": current_cell, "heading": current_heading},
            "mode": mode,
            "loop_active": loop_active,
            "active_frontier_goal": active_goal,
            "active_route": active_route,
            "coverage_patrol": coverage_patrol,
            "policy": {
                "goal": "prefer_goal_level_exploration_over_raw_motor_moves",
                "route_policy": (
                    "when active_route.status is active and explore:route_step exists, "
                    "continue that committed route step before choosing waypoint/frontier alternatives"
                ),
                "waypoint_policy": (
                    "when break_rotation_loop is active and explore:waypoint options exist, "
                    "use them before rotation-only frontier options"
                ),
                "frontier_suppression": (
                    "frontiers whose first step only repeats a rotation loop are hidden from primary options "
                    "when a safe translation waypoint is available"
                ),
            },
            "waypoint_candidates": waypoint_candidates,
            "recovery_actions": build_recovery_actions_for_plan(
                mode=mode,
                active_route=active_route,
                navigation_costmap=navigation_costmap,
            ),
            "suppressed_frontiers": suppressed,
            "suppressed_frontier_cells": [item.get("cell") for item in suppressed if item.get("cell")],
            "frontier_rankings": compact_frontier_rankings(exploration, suppressed),
        }
    )


def load_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    import argparse

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build loop-aware exploration planning summary.")
    parser.add_argument("--memory-dir", default=str(root / "memory"))
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    args = parser.parse_args()
    memory = Path(args.memory_dir)
    position_map = load_map_backend(memory).load_snapshot().to_position_status()
    costmap = load_json(memory / "navigation-costmap.json")
    exploration = load_json(memory / "decision-context.json").get("exploration", {})
    plan = build_explore_plan(
        position_map=position_map,
        navigation_costmap=costmap,
        exploration=as_dict(exploration),
    )
    if args.format == "compact":
        print(json.dumps(plan, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
