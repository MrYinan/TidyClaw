#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a bounded exploration summary for LLM decision turns.

The decision context should tell the model whether it is looping without
exposing the full position map. This module consumes the existing
position-map/navigation-costmap/global-plan JSON objects and returns a compact,
read-only summary: recent path, revisit pressure, frontier candidates,
loop warnings, and actions that are legal but poor exploration choices.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scripts.position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        HEADING_VECTORS,
        cell_distance,
        format_cell,
        four_neighbors,
        heading_between,
        left_heading,
        manhattan_distance,
        neighbor_for_action,
        opposite_heading,
        parse_cell,
        right_heading,
    )
except ImportError:  # pragma: no cover - direct script execution
    from position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        HEADING_VECTORS,
        cell_distance,
        format_cell,
        four_neighbors,
        heading_between,
        left_heading,
        manhattan_distance,
        neighbor_for_action,
        opposite_heading,
        parse_cell,
        right_heading,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = REPO_ROOT / "memory"

EXPLORATION_CONTEXT_SCHEMA = "robot_cleaner_exploration_context_v1"
TRANSLATION_ACTIONS = ("MoveAhead", "MoveLeft", "MoveRight", "MoveBack")
ROTATION_ACTIONS = ("RotateLeft", "RotateRight")
MOVE_ACTIONS = (*TRANSLATION_ACTIONS, *ROTATION_ACTIONS)
RECENT_PATH_LIMIT = 14
RECENT_ACTION_LIMIT = 18
MAX_AVOID_ACTIONS = 8

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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def number_or_none(value: Any, *, digits: int = 4) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    return None


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
        return [clean_empty(item) for item in value if clean_empty(item) not in (None, "", [], {})]
    return value


def canonical_action(value: Any) -> str:
    text = str(value or "").strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    compact = text.replace("_", "").replace("-", "").lower()
    mapping = {
        "moveahead": "MoveAhead",
        "forward": "MoveAhead",
        "moveforward": "MoveAhead",
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
    }
    return mapping.get(compact, text)


def safe_parse_cell(value: Any) -> tuple[int, int] | None:
    try:
        return parse_cell(value)
    except Exception:
        return None


def safe_neighbor_for_action(cell: str, heading: str, action: str) -> str | None:
    try:
        return neighbor_for_action(cell, heading, action)
    except Exception:
        return None


def edge_is_blocked(position_map: JsonDict, source: str, target: str) -> bool:
    edges = as_dict(position_map.get("edges"))
    blocked = {str(item) for item in as_list(edges.get("blocked_edges"))}
    blocked.update(str(item) for item in as_list(position_map.get("blocked_edges")))
    return f"{source}->{target}" in blocked or f"{target}->{source}" in blocked


def _previous_pose(cell: str, heading: str, action: str) -> tuple[str, str]:
    if action == "MoveAhead":
        return neighbor_for_action(cell, heading, "MoveBack"), heading
    if action == "MoveBack":
        return neighbor_for_action(cell, heading, "MoveAhead"), heading
    if action == "MoveLeft":
        return neighbor_for_action(cell, heading, "MoveRight"), heading
    if action == "MoveRight":
        return neighbor_for_action(cell, heading, "MoveLeft"), heading
    if action == "RotateLeft":
        return cell, right_heading(heading)
    if action == "RotateRight":
        return cell, left_heading(heading)
    return cell, heading


def reconstruct_recent_path(
    *,
    current_cell: str,
    current_heading: str,
    recent_actions: Sequence[Any],
    limit: int = RECENT_PATH_LIMIT,
) -> list[JsonDict]:
    """Reconstruct a compact pose trail from action odometry.

    position-map currently stores recent actions but not a full pose trail. The
    reconstruction is intentionally approximate; it is only for loop awareness,
    not for executor safety.
    """

    actions = [canonical_action(item) for item in as_list(recent_actions)]
    actions = [action for action in actions if action in MOVE_ACTIONS][-max(1, int(limit)) :]
    cell = str(current_cell or "0,0")
    heading = str(current_heading or "north")
    if safe_parse_cell(cell) is None:
        cell = "0,0"
    if heading not in HEADING_VECTORS:
        heading = "north"

    reversed_states: list[JsonDict] = []
    for action in reversed(actions):
        reversed_states.append({"cell": cell, "heading": heading, "arrived_by": action})
        try:
            cell, heading = _previous_pose(cell, heading, action)
        except Exception:
            break
    reversed_states.append({"cell": cell, "heading": heading, "arrived_by": "start"})
    return list(reversed(reversed_states))[-(limit + 1) :]


def visited_or_seen_count(cells: JsonDict, cell: str) -> int:
    rec = as_dict(cells.get(cell))
    for key in ("visit_count", "visited_count", "seen_count"):
        value = rec.get(key)
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 1 if rec.get("visited") else 0


def build_revisit_counts(position_map: JsonDict, recent_path: Sequence[JsonDict]) -> JsonDict:
    cells = as_dict(position_map.get("cells"))
    path_cells = [str(item.get("cell")) for item in recent_path if item.get("cell")]
    counts = Counter(path_cells)
    repeated = [
        {
            "cell": cell,
            "recent_count": count,
            "known_seen_count": visited_or_seen_count(cells, cell),
        }
        for cell, count in counts.items()
        if count > 1
    ]
    repeated.sort(key=lambda item: (-int(item["recent_count"]), str(item["cell"])))
    current_cell = path_cells[-1] if path_cells else ""
    return clean_empty(
        {
            "recent_window": len(path_cells),
            "recent_unique_cell_count": len(set(path_cells)),
            "current_cell": current_cell,
            "current_cell_recent_count": counts.get(current_cell, 0),
            "repeated_recent_cells": repeated[:8],
        }
    )


def _recent_area(path_cells: Sequence[str]) -> JsonDict:
    coords = [safe_parse_cell(cell) for cell in path_cells]
    valid = [item for item in coords if item is not None]
    if not valid:
        return {}
    xs = [x for x, _ in valid]
    zs = [z for _, z in valid]
    return {
        "min_cell": format_cell(min(xs), min(zs)),
        "max_cell": format_cell(max(xs), max(zs)),
        "width_cells": max(xs) - min(xs) + 1,
        "height_cells": max(zs) - min(zs) + 1,
    }


def build_loop_warning(
    *,
    recent_path: Sequence[JsonDict],
    recent_actions: Sequence[Any],
) -> JsonDict:
    path_cells = [str(item.get("cell")) for item in recent_path if item.get("cell")]
    if not path_cells:
        return {"active": False, "severity": "none", "reason": "no_recent_path"}
    counts = Counter(path_cells)
    unique_count = len(counts)
    window = len(path_cells)
    current_count = counts[path_cells[-1]]
    max_count = max(counts.values()) if counts else 0
    action_window = [canonical_action(item) for item in as_list(recent_actions)[-RECENT_ACTION_LIMIT:]]
    rotation_count = sum(1 for action in action_window if action in ROTATION_ACTIONS)
    translation_count = sum(1 for action in action_window if action in TRANSLATION_ACTIONS)
    unique_ratio = float(unique_count) / float(max(1, window))

    alternating_rotations = 0
    for prev, current in zip(action_window, action_window[1:]):
        if {prev, current} == {"RotateLeft", "RotateRight"}:
            alternating_rotations += 1

    reasons: list[str] = []
    severity = "none"
    active = False
    if window >= 6 and (current_count >= 4 or max_count >= 4):
        active = True
        severity = "strong"
        reasons.append("same_cell_revisited_four_or_more_times_in_recent_path")
    if window >= 8 and unique_ratio <= 0.55:
        active = True
        if severity == "none":
            severity = "advisory"
        reasons.append("recent_path_has_low_unique_cell_ratio")
    if len(action_window) >= 6 and rotation_count >= 5 and translation_count <= 3:
        active = True
        if severity == "none":
            severity = "advisory"
        reasons.append("many_rotations_with_little_translation")
    if alternating_rotations >= 2:
        active = True
        if severity == "none":
            severity = "advisory"
        reasons.append("left_right_rotation_oscillation")

    return clean_empty(
        {
            "active": active,
            "severity": severity,
            "reason": ";".join(reasons) if reasons else "recent_path_has_sufficient_progress",
            "recent_window": window,
            "recent_unique_cell_count": unique_count,
            "current_cell_recent_count": current_count,
            "max_recent_cell_count": max_count,
            "recent_unique_ratio": round(unique_ratio, 3),
            "rotation_count": rotation_count,
            "translation_count": translation_count,
            "recent_area": _recent_area(path_cells),
        }
    )


def direction_from_delta(source: str, target: str) -> str | None:
    adjacent = heading_between(source, target)
    if adjacent:
        return adjacent
    parsed_source = safe_parse_cell(source)
    parsed_target = safe_parse_cell(target)
    if parsed_source is None or parsed_target is None:
        return None
    sx, sz = parsed_source
    tx, tz = parsed_target
    dx = tx - sx
    dz = tz - sz
    if abs(dx) >= abs(dz) and dx != 0:
        return "east" if dx > 0 else "west"
    if dz != 0:
        return "north" if dz > 0 else "south"
    return None


def first_action_toward_direction(current_heading: str, desired_heading: str | None) -> str | None:
    if not desired_heading:
        return None
    if desired_heading == current_heading:
        return "MoveAhead"
    if desired_heading == left_heading(current_heading):
        return "RotateLeft"
    if desired_heading == right_heading(current_heading):
        return "RotateRight"
    return "RotateRight"


def frontier_first_action_hint(
    *,
    current_heading: str,
    direction: str | None,
    planner_selected: bool,
    planner_next_action: str,
) -> str | None:
    """Return the first exploration step exposed to the LLM.

    Frontier exploration should be target-level. When a frontier lies to the
    side, the first step should rotate and re-observe rather than sidestep into
    an unknown cell. Sideways motion remains available as a low-level fallback
    option, but explore:* options prefer rotate/forward steps.
    """

    target_action = first_action_toward_direction(current_heading, direction)
    planner_action = canonical_action(planner_next_action)
    if planner_selected and planner_action in {"MoveAhead", "RotateLeft", "RotateRight"}:
        return planner_action
    return target_action


def _frontier_cluster_sizes(frontiers: Sequence[str]) -> dict[str, int]:
    remaining = {str(item) for item in frontiers if str(item).strip()}
    sizes: dict[str, int] = {}
    while remaining:
        root = remaining.pop()
        stack = [root]
        cluster = [root]
        while stack:
            current = stack.pop()
            for neighbor in four_neighbors(current):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    cluster.append(neighbor)
        for cell in cluster:
            sizes[cell] = len(cluster)
    return sizes


def unknown_neighbor_count(cells: JsonDict, cell: str) -> int:
    count = 0
    for neighbor in four_neighbors(cell):
        rec = as_dict(cells.get(neighbor))
        if not rec or str(rec.get("state") or "unknown") == "unknown":
            count += 1
    return count


def _heading_projection(current_cell: str, current_heading: str, target: str) -> float:
    source = safe_parse_cell(current_cell)
    dest = safe_parse_cell(target)
    if source is None or dest is None:
        return 0.0
    sx, sz = source
    tx, tz = dest
    dx, dz = HEADING_VECTORS.get(current_heading, (0, 1))
    return float((tx - sx) * dx + (tz - sz) * dz)


def build_frontier_candidates(
    *,
    position_map: JsonDict,
    global_plan: JsonDict,
    current_cell: str,
    current_heading: str,
    recent_path: Sequence[JsonDict],
    limit: int,
) -> list[JsonDict]:
    cells = as_dict(position_map.get("cells"))
    raw_frontiers = as_list(position_map.get("frontiers")) or as_list(position_map.get("frontier_cells"))
    frontiers = list(dict.fromkeys(str(item) for item in raw_frontiers if str(item).strip()))
    if not frontiers:
        return []

    cluster_sizes = _frontier_cluster_sizes(frontiers)
    recent_cells = {str(item.get("cell")) for item in recent_path[-8:] if item.get("cell")}
    selected_goal = str(global_plan.get("selected_goal_cell") or "")
    planner_next_action = str(global_plan.get("next_action") or "")
    candidates: list[JsonDict] = []
    for frontier in frontiers:
        if safe_parse_cell(frontier) is None:
            continue
        direction = direction_from_delta(current_cell, frontier)
        distance = manhattan_distance(current_cell, frontier)
        unknown_gain = unknown_neighbor_count(cells, frontier)
        cluster = int(cluster_sizes.get(frontier, 1) or 1)
        projection = _heading_projection(current_cell, current_heading, frontier)
        behind_penalty = abs(min(0.0, projection))
        recent_penalty = 1.0 if frontier in recent_cells else 0.0
        seen_penalty = 0.35 * float(visited_or_seen_count(cells, frontier))
        planner_selected = frontier == selected_goal
        score = (
            10.0
            - 0.65 * float(distance)
            + 0.9 * float(unknown_gain)
            + 0.35 * min(8.0, float(cluster))
            - 1.4 * behind_penalty
            - 2.5 * recent_penalty
            - seen_penalty
            + (1.2 if planner_selected else 0.0)
        )
        reasons: list[str] = []
        if unknown_gain >= 2:
            reasons.append("high_unknown_neighbor_gain")
        if cluster >= 3:
            reasons.append("frontier_cluster")
        if projection < 0:
            reasons.append("behind_current_heading")
        if frontier in recent_cells:
            reasons.append("on_recent_path")
        if planner_selected:
            reasons.append("matches_existing_global_plan")
        first_action = frontier_first_action_hint(
            current_heading=current_heading,
            direction=direction,
            planner_selected=planner_selected,
            planner_next_action=planner_next_action,
        )
        candidates.append(
            clean_empty(
                {
                    "cell": frontier,
                    "distance_steps": distance,
                    "direction_from_current": direction,
                    "first_action_hint": first_action,
                    "unknown_neighbor_count": unknown_gain,
                    "cluster_size": cluster,
                    "planner_selected": planner_selected,
                    "score": round(score, 3),
                    "reasons": reasons,
                    "first_action_policy": "rotate_or_forward_only",
                }
            )
        )
    candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("distance_steps", 9999)), str(item.get("cell"))))
    return candidates[: max(1, int(limit))]


def _target_cell_record(position_map: JsonDict, cell: str) -> JsonDict:
    return as_dict(as_dict(position_map.get("cells")).get(cell))


def build_avoid_actions(
    *,
    position_map: JsonDict,
    navigation_costmap: JsonDict,
    current_cell: str,
    current_heading: str,
    recent_path: Sequence[JsonDict],
    recent_actions: Sequence[Any],
) -> list[JsonDict]:
    avoid: list[JsonDict] = []
    recent_cells = [str(item.get("cell")) for item in recent_path if item.get("cell")]
    recent_last = set(recent_cells[-6:])
    action_safety = as_dict(navigation_costmap.get("action_safety"))
    frontier_set = {str(item) for item in as_list(position_map.get("frontiers")) if str(item).strip()}
    for action in TRANSLATION_ACTIONS:
        target = safe_neighbor_for_action(current_cell, current_heading, action)
        if not target:
            continue
        target_rec = _target_cell_record(position_map, target)
        target_state = str(target_rec.get("state") or "unknown")
        reasons: list[str] = []
        severity = "warning"
        safety = as_dict(action_safety.get(action))
        if safety.get("safe") is False:
            reasons.append("local_costmap_marks_action_unsafe")
            severity = "block"
        if target_state in {CELL_OCCUPIED, CELL_INFLATED}:
            reasons.append(f"target_cell_{target_state}")
            severity = "block"
        if edge_is_blocked(position_map, current_cell, target):
            reasons.append("blocked_edge_in_position_map")
            severity = "block"
        if target in recent_last:
            reasons.append("returns_to_recent_path")
        elif target_rec.get("visited") is True and target not in frontier_set:
            reasons.append("enters_already_visited_non_frontier_cell")
        if not reasons:
            continue
        avoid.append(
            clean_empty(
                {
                    "action": action,
                    "target_cell": target,
                    "severity": severity,
                    "reasons": reasons,
                    "target_state": target_state,
                    "known_seen_count": visited_or_seen_count(as_dict(position_map.get("cells")), target),
                }
            )
        )

    action_window = [canonical_action(item) for item in as_list(recent_actions)[-4:]]
    last_action = action_window[-1] if action_window else ""
    if last_action == "RotateLeft":
        avoid.append(
            {
                "action": "RotateRight",
                "severity": "advisory",
                "reasons": ["immediate_rotation_undo"],
            }
        )
    elif last_action == "RotateRight":
        avoid.append(
            {
                "action": "RotateLeft",
                "severity": "advisory",
                "reasons": ["immediate_rotation_undo"],
            }
        )
    return avoid[:MAX_AVOID_ACTIONS]


def heading_after_action(current_heading: str, action: str) -> str:
    if action == "RotateLeft":
        return left_heading(current_heading)
    if action == "RotateRight":
        return right_heading(current_heading)
    return current_heading


def _cell_state(cells: JsonDict, cell: str) -> str:
    rec = as_dict(cells.get(cell))
    return str(rec.get("state") or "unknown")


def _cell_visited(cells: JsonDict, cell: str) -> bool:
    return as_dict(cells.get(cell)).get("visited") is True


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


def best_frontier_cell(frontier_candidates: Sequence[JsonDict]) -> str:
    for candidate in frontier_candidates:
        cell = str(as_dict(candidate).get("cell") or "").strip()
        if cell:
            return cell
    return ""


def frontier_delta(current_distance: int | None, target_distance: int | None) -> int | None:
    if current_distance is None or target_distance is None:
        return None
    return int(current_distance) - int(target_distance)


def build_translation_effect(
    *,
    action: str,
    position_map: JsonDict,
    current_cell: str,
    current_heading: str,
    recent_path: Sequence[JsonDict],
    frontier_candidates: Sequence[JsonDict],
) -> JsonDict:
    cells = as_dict(position_map.get("cells"))
    frontier_set = {str(item) for item in as_list(position_map.get("frontiers")) if str(item).strip()}
    recent_cells = {str(item.get("cell")) for item in recent_path[-6:] if item.get("cell")}
    target = safe_neighbor_for_action(current_cell, current_heading, action)
    if not target:
        return {"action": action, "effect_type": "translation", "effect": "unknown"}

    target_state = _cell_state(cells, target)
    target_visited = _cell_visited(cells, target)
    enters_frontier = target in frontier_set
    enters_visited_cell = bool(target_visited)
    repeats_recent_path = target in recent_cells
    best = best_frontier_cell(frontier_candidates)
    best_distance_before = manhattan_distance(current_cell, best) if best and safe_parse_cell(best) else None
    best_distance_after = manhattan_distance(target, best) if best and safe_parse_cell(best) else None
    nearest_before = nearest_frontier_distance(current_cell, frontier_set)
    nearest_after = nearest_frontier_distance(target, frontier_set)
    best_delta = frontier_delta(best_distance_before, best_distance_after)
    nearest_delta = frontier_delta(nearest_before, nearest_after)
    toward_frontier = enters_frontier or bool(
        (best_delta is not None and best_delta > 0)
        or (nearest_delta is not None and nearest_delta > 0)
    )

    if target_state in {CELL_OCCUPIED, CELL_INFLATED}:
        effect = "blocked_known_cell"
    elif enters_frontier:
        effect = "enters_frontier"
    elif target_state == "unknown":
        effect = "enters_unknown_cell"
    elif toward_frontier:
        effect = "toward_frontier"
    elif enters_visited_cell:
        effect = "enters_visited_cell"
    else:
        effect = "local_translation"

    return clean_empty(
        {
            "action": action,
            "effect_type": "translation",
            "effect": effect,
            "target_cell": target,
            "target_state": target_state,
            "enters_visited_cell": enters_visited_cell,
            "enters_frontier": enters_frontier,
            "toward_frontier": toward_frontier,
            "repeats_recent_path": repeats_recent_path,
            "best_frontier_cell": best,
            "best_frontier_distance_delta": best_delta,
            "nearest_frontier_distance_delta": nearest_delta,
        }
    )


def build_rotation_effect(
    *,
    action: str,
    position_map: JsonDict,
    current_cell: str,
    current_heading: str,
    frontier_candidates: Sequence[JsonDict],
) -> JsonDict:
    cells = as_dict(position_map.get("cells"))
    frontier_set = {str(item) for item in as_list(position_map.get("frontiers")) if str(item).strip()}
    new_heading = heading_after_action(current_heading, action)
    facing_cell = safe_neighbor_for_action(current_cell, new_heading, "MoveAhead") or current_cell
    facing_state = _cell_state(cells, facing_cell)
    facing_visited = _cell_visited(cells, facing_cell)
    facing_frontier = facing_cell in frontier_set
    faces_unvisited_area = facing_frontier or (not facing_visited and facing_state not in {CELL_OCCUPIED, CELL_INFLATED})
    best = best_frontier_cell(frontier_candidates)
    best_direction = direction_from_delta(current_cell, best) if best else None
    toward_frontier = bool(best_direction and new_heading == best_direction)

    if facing_state in {CELL_OCCUPIED, CELL_INFLATED}:
        effect = "faces_blocked_known_cell"
    elif facing_frontier:
        effect = "faces_frontier"
    elif faces_unvisited_area:
        effect = "faces_unvisited_area"
    elif toward_frontier:
        effect = "turns_toward_frontier"
    else:
        effect = "faces_visited_area"

    return clean_empty(
        {
            "action": action,
            "effect_type": "rotation",
            "effect": effect,
            "new_heading": new_heading,
            "facing_cell": facing_cell,
            "facing_cell_state": facing_state,
            "facing_cell_visited": facing_visited,
            "facing_frontier": facing_frontier,
            "faces_unvisited_area": faces_unvisited_area,
            "toward_frontier": toward_frontier,
            "best_frontier_cell": best,
            "best_frontier_direction": best_direction,
        }
    )


def build_action_effects(
    *,
    position_map: JsonDict,
    current_cell: str,
    current_heading: str,
    recent_path: Sequence[JsonDict],
    frontier_candidates: Sequence[JsonDict],
) -> dict[str, JsonDict]:
    effects: dict[str, JsonDict] = {}
    for action in TRANSLATION_ACTIONS:
        effects[action] = build_translation_effect(
            action=action,
            position_map=position_map,
            current_cell=current_cell,
            current_heading=current_heading,
            recent_path=recent_path,
            frontier_candidates=frontier_candidates,
        )
    for action in ROTATION_ACTIONS:
        effects[action] = build_rotation_effect(
            action=action,
            position_map=position_map,
            current_cell=current_cell,
            current_heading=current_heading,
            frontier_candidates=frontier_candidates,
        )
    return clean_empty(effects)


def build_exploration_context(
    *,
    position_map: JsonDict,
    navigation_costmap: JsonDict | None = None,
    global_plan: JsonDict | None = None,
    max_frontier_candidates: int = 6,
    recent_path_limit: int = RECENT_PATH_LIMIT,
) -> JsonDict:
    pose = as_dict(position_map.get("pose"))
    current_cell = str(pose.get("cell") or position_map.get("last_cell") or "0,0")
    current_heading = str(pose.get("heading") or position_map.get("last_heading") or "north")
    if safe_parse_cell(current_cell) is None:
        current_cell = "0,0"
    if current_heading not in HEADING_VECTORS:
        current_heading = "north"

    recent_actions = as_list(position_map.get("recent_actions"))
    recent_path = reconstruct_recent_path(
        current_cell=current_cell,
        current_heading=current_heading,
        recent_actions=recent_actions,
        limit=max(4, int(recent_path_limit)),
    )
    revisit_counts = build_revisit_counts(position_map, recent_path)
    loop_warning = build_loop_warning(recent_path=recent_path, recent_actions=recent_actions)
    frontier_candidates = build_frontier_candidates(
        position_map=position_map,
        global_plan=as_dict(global_plan),
        current_cell=current_cell,
        current_heading=current_heading,
        recent_path=recent_path,
        limit=max_frontier_candidates,
    )
    avoid_actions = build_avoid_actions(
        position_map=position_map,
        navigation_costmap=as_dict(navigation_costmap),
        current_cell=current_cell,
        current_heading=current_heading,
        recent_path=recent_path,
        recent_actions=recent_actions,
    )
    action_effects = build_action_effects(
        position_map=position_map,
        current_cell=current_cell,
        current_heading=current_heading,
        recent_path=recent_path,
        frontier_candidates=frontier_candidates,
    )

    stats = as_dict(position_map.get("stats"))
    return clean_empty(
        {
            "schema": EXPLORATION_CONTEXT_SCHEMA,
            "generated_at": now_iso(),
            "current_pose": {
                "cell": current_cell,
                "heading": current_heading,
                "pose_confidence": number_or_none(pose.get("pose_confidence")),
                "position_uncertainty_cells": number_or_none(pose.get("position_uncertainty_cells")),
                "heading_confidence": number_or_none(pose.get("heading_confidence")),
            },
            "map_progress": {
                "visited_cell_count": number_or_none(stats.get("visited_cell_count")),
                "frontier_count": len(as_list(position_map.get("frontiers"))),
                "collision_count": number_or_none(stats.get("collision_count")),
            },
            "recent_path": recent_path,
            "revisit_counts": revisit_counts,
            "frontier_candidates": frontier_candidates,
            "loop_warning": loop_warning,
            "avoid_actions": avoid_actions,
            "action_effects": action_effects,
        }
    )


def load_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def json_print(data: JsonDict, *, compact: bool = False) -> None:
    if compact:
        print(json.dumps(data, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a compact exploration context from robot memory.")
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--max-frontiers", type=int, default=6)
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    memory_dir = Path(args.memory_dir)
    result = build_exploration_context(
        position_map=load_json(memory_dir / "position-map.json"),
        navigation_costmap=load_json(memory_dir / "navigation-costmap.json"),
        global_plan=load_json(memory_dir / "global-plan.json"),
        max_frontier_candidates=max(1, int(args.max_frontiers)),
    )
    json_print(result, compact=args.format == "compact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
