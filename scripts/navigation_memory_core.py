#!/usr/bin/env python3
"""
Navigation memory for robot-cleaner.

This module bridges the persistent Action-Odometry Occupancy Grid, semantic
layer, bounded A* global planner, and online-safe RGB-D local costmap.  It
retains room-state compatibility for interrupted patrols while replacing the
old forward-only frontier heuristic with collision-aware cardinal actions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from scripts.position_map_core import PositionMap
    from scripts.semantic_mapping_core import SemanticMap
    from scripts.global_planner import (
        DRIVE_MODEL_HOLONOMIC,
        DRIVE_MODEL_NONHOLONOMIC,
        AStarGlobalPlanner,
        normalize_drive_model,
    )
except ImportError:
    from position_map_core import PositionMap
    from semantic_mapping_core import SemanticMap
    from global_planner import (
        DRIVE_MODEL_HOLONOMIC,
        DRIVE_MODEL_NONHOLONOMIC,
        AStarGlobalPlanner,
        normalize_drive_model,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
ROOM_STATE_PATH = MEMORY_DIR / "room-state.json"

DEFAULT_CELL_SIZE = 0.25
# FloorPlan1's reachable kitchen area is larger than the first debug strip.
# Keep this conservative so "coverage=1.00" means broad coverage, not merely
# enough steps in a corridor-like local area.
DEFAULT_TARGET_CELLS = 120
MOVE_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
LOOK_ACTIONS = {"LookUp", "LookDown"}
HEADING_ORDER = ["north", "east", "south", "west"]
HEADING_VECTORS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}


JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        replaced = False
        last_replace_error: Optional[PermissionError] = None
        for attempt in range(20):
            try:
                os.replace(tmp_name, path)
                replaced = True
                break
            except PermissionError as exc:
                last_replace_error = exc
                if attempt >= 19:
                    break
                time.sleep(min(0.5, 0.05 * (attempt + 1)))
        if not replaced:
            try:
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
            except PermissionError:
                if last_replace_error is not None:
                    raise last_replace_error
                raise
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except PermissionError:
                pass


def unique_extend(existing: Iterable[str], incoming: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in list(existing) + list(incoming):
        if item is None:
            continue
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def default_room_state(room_name: str = "current_room") -> JsonDict:
    return {
        "schema_version": 1,
        "room_name": room_name,
        "explored_steps": 0,
        "targets_found": [],
        "targets_cleaned": [],
        "room_complete": False,
        "cell_size": DEFAULT_CELL_SIZE,
        "visited_cells": [],
        "visited_cell_counts": {},
        "known_open_edges": [],
        "blocked_edges": [],
        "hard_blocked_edges": [],
        "coverage_estimate": 0.0,
        "frontier_cells": [],
        "known_frontier_cells": [],
        "collision_count": 0,
        "oscillation_count": 0,
        "stagnation_count": 0,
        "backtrack_count": 0,
        "turn_streak_count": 0,
        "last_new_cell": None,
        "recent_navigation_actions": [],
        "last_navigation_decision": {},
        "last_frontier_target": None,
        "active_frontier_goal": None,
        "frontier_cooldowns": {},
        "frontier_history": [],
        "frontier_oscillation_count": 0,
        "last_cell": None,
        "last_heading": None,
        "navigation_last_update": None,
        "last_update": None,
    }


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def parse_json_arg(value: Optional[str]) -> JsonDict:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def normalize_rotation_y(rotation_y: float) -> float:
    value = rotation_y % 360.0
    if value < 0:
        value += 360.0
    return value


def heading_from_rotation(rotation_y: float) -> str:
    value = normalize_rotation_y(rotation_y)
    index = int(round(value / 90.0)) % 4
    return HEADING_ORDER[index]


def left_heading(heading: str) -> str:
    index = HEADING_ORDER.index(heading)
    return HEADING_ORDER[(index - 1) % 4]


def right_heading(heading: str) -> str:
    index = HEADING_ORDER.index(heading)
    return HEADING_ORDER[(index + 1) % 4]


def cell_from_position(position: JsonDict, cell_size: float = DEFAULT_CELL_SIZE) -> str:
    x = float(position.get("x", 0.0) or 0.0)
    z = float(position.get("z", 0.0) or 0.0)
    ix = int(round(x / cell_size))
    iz = int(round(z / cell_size))
    return f"{ix},{iz}"


def parse_cell(cell: str) -> Tuple[int, int]:
    left, right = str(cell).split(",", 1)
    return int(left), int(right)


def neighbor_cell(cell: str, heading: str) -> str:
    x, z = parse_cell(cell)
    dx, dz = HEADING_VECTORS[heading]
    return f"{x + dx},{z + dz}"


def edge_key(source: str, target: str) -> str:
    return f"{source}->{target}"


def edge_pair_key(source: str, target: str) -> str:
    left = str(source).strip()
    right = str(target).strip()
    if left <= right:
        return f"{left}|{right}"
    return f"{right}|{left}"


def parse_edge(edge: str) -> Optional[Tuple[str, str]]:
    value = str(edge)
    if "->" not in value:
        return None
    source, target = value.split("->", 1)
    source = source.strip()
    target = target.strip()
    if not source or not target:
        return None
    return source, target


def reverse_edge_key(edge: str) -> str:
    parsed = parse_edge(edge)
    if not parsed:
        return str(edge)
    source, target = parsed
    return edge_key(target, source)


def opposite_heading(heading: str) -> str:
    index = HEADING_ORDER.index(heading)
    return HEADING_ORDER[(index + 2) % 4]


def heading_between(source: str, target: str) -> Optional[str]:
    sx, sz = parse_cell(source)
    tx, tz = parse_cell(target)
    dx = tx - sx
    dz = tz - sz
    for heading, vector in HEADING_VECTORS.items():
        if vector == (dx, dz):
            return heading
    return None


def manhattan_distance(left: str, right: str) -> int:
    lx, lz = parse_cell(left)
    rx, rz = parse_cell(right)
    return abs(lx - rx) + abs(lz - rz)


def has_explicit_pose(data: JsonDict, *, prefix: str = "") -> bool:
    """Return true only when a payload actually carries simulator pose.

    V2 online payloads intentionally omit pose. The navigation memory therefore
    must not assume missing pose means the real origin; it falls back to
    action-feedback odometry stored in room-state.json.
    """
    position_key = f"{prefix}position" if prefix else "position"
    rotation_key = f"{prefix}rotation" if prefix else "rotation"
    return isinstance(data.get(position_key), dict) and isinstance(data.get(rotation_key), dict)


def pose_from_vision(vision: JsonDict, *, fallback_cell: str = "0,0", fallback_heading: str = "north") -> Tuple[str, str]:
    position = vision.get("position") if isinstance(vision.get("position"), dict) else None
    rotation = vision.get("rotation") if isinstance(vision.get("rotation"), dict) else None
    if position is None or rotation is None:
        return fallback_cell, fallback_heading
    cell = cell_from_position(position)
    heading = heading_from_rotation(float(rotation.get("y", 0.0) or 0.0))
    return cell, heading


def pose_from_action_feedback(
    *,
    room: JsonDict,
    action: str,
    success: bool,
    action_result: JsonDict,
    fallback_vision: JsonDict,
) -> Tuple[str, str, str, str]:
    """Infer before/after pose from online-safe action feedback.

    If an offline/debug caller explicitly includes pose, this function can use
    it. Otherwise it updates a coarse relative grid from previous room-state and
    action success. This keeps navigation memory usable under the V2 RGB-only
    online contract.
    """
    room_cell = str(room.get("last_cell") or "0,0")
    room_heading = str(room.get("last_heading") or "north")
    if room_heading not in HEADING_ORDER:
        room_heading = "north"

    before_position = action_result.get("position_before")
    before_rotation = action_result.get("rotation_before")
    after_position = action_result.get("position_after")
    after_rotation = action_result.get("rotation_after")

    if isinstance(before_position, dict) and isinstance(before_rotation, dict):
        before_cell = cell_from_position(before_position)
        before_heading = heading_from_rotation(float(before_rotation.get("y", 0.0) or 0.0))
    else:
        before_cell, before_heading = pose_from_vision(
            fallback_vision,
            fallback_cell=room_cell,
            fallback_heading=room_heading,
        )

    if isinstance(after_position, dict) and isinstance(after_rotation, dict):
        after_cell = cell_from_position(after_position)
        after_heading = heading_from_rotation(float(after_rotation.get("y", 0.0) or 0.0))
        return before_cell, before_heading, after_cell, after_heading

    after_cell = before_cell
    after_heading = before_heading
    if success:
        if action == "RotateLeft":
            after_heading = left_heading(before_heading)
        elif action == "RotateRight":
            after_heading = right_heading(before_heading)
        elif action == "MoveAhead":
            after_cell = neighbor_cell(before_cell, before_heading)
        elif action == "MoveBack":
            after_cell = neighbor_cell(before_cell, opposite_heading(before_heading))
        elif action == "MoveLeft":
            after_cell = neighbor_cell(before_cell, left_heading(before_heading))
        elif action == "MoveRight":
            after_cell = neighbor_cell(before_cell, right_heading(before_heading))
    return before_cell, before_heading, after_cell, after_heading


def boolish(value: Any) -> bool:
    return bool(value)


def local_costmap_action_record(analysis: JsonDict, action: str) -> Optional[JsonDict]:
    costmap = analysis.get("local_costmap") if isinstance(analysis, dict) else None
    if not isinstance(costmap, dict) or str(costmap.get("status") or "") != "success":
        return None
    rec = (costmap.get("action_safety") or {}).get(str(action))
    return dict(rec) if isinstance(rec, dict) else None


@dataclass(frozen=True)
class MotionSafety:
    """Depth-primary motion safety decision for one low-level action.

    ``known`` means RGB-D local costmap has enough evidence to override legacy
    YOLO/RGB navigation hints.  When ``known`` is false, callers may fall back
    to semantic hints such as open_directions/obstacle_ahead.
    """

    action: str
    known: bool
    safe: Optional[bool]
    confidence: float = 0.0
    observed_ratio: float = 0.0
    reason: str = "no_local_costmap"
    record: Optional[JsonDict] = None


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _local_costmap_safe_min_confidence(action: str) -> float:
    action = str(action)
    if action == "MoveAhead":
        return env_float("ROBOT_LOCAL_COSTMAP_MOVEAHEAD_SAFE_MIN_CONFIDENCE", 0.50)
    if action == "MoveBack":
        return env_float("ROBOT_LOCAL_COSTMAP_MOVEBACK_SAFE_MIN_CONFIDENCE", 0.50)
    if action in {"MoveLeft", "MoveRight"}:
        return env_float("ROBOT_LOCAL_COSTMAP_LATERAL_SAFE_MIN_CONFIDENCE", 0.55)
    if action in ROTATE_ACTIONS:
        return env_float("ROBOT_LOCAL_COSTMAP_ROTATE_SAFE_MIN_CONFIDENCE", 0.20)
    if action in LOOK_ACTIONS:
        return env_float("ROBOT_LOCAL_COSTMAP_LOOK_SAFE_MIN_CONFIDENCE", 0.0)
    return env_float("ROBOT_LOCAL_COSTMAP_SAFE_MIN_CONFIDENCE", 0.50)


def _local_costmap_min_observed_ratio(action: str, rec: JsonDict) -> float:
    supplied = rec.get("min_observed_ratio")
    if supplied is not None:
        base = _float_value(supplied, 0.0)
    elif action == "MoveBack":
        base = 0.60
    elif action == "MoveAhead":
        base = 0.15
    elif action in {"MoveLeft", "MoveRight"}:
        base = 0.55
    else:
        base = 0.0
    if action == "MoveBack":
        base = max(base, env_float("ROBOT_MOVE_BACK_MIN_OBSERVED_RATIO", 0.60))
    return max(0.0, min(1.0, base))


def local_costmap_motion_safety(analysis: JsonDict, action: str) -> MotionSafety:
    """Return the authoritative local RGB-D safety state when available.

    This deliberately separates geometric safety from semantic navigation
    hints.  YOLO's obstacle_ahead/open_directions can still guide exploration
    when depth is absent, but a confident local costmap decision owns hard
    motion veto/allow decisions.
    """

    action = str(action)
    rec = local_costmap_action_record(analysis, action)
    if not rec:
        return MotionSafety(action=action, known=False, safe=None)

    confidence = _float_value(rec.get("confidence"), 0.0)
    # Old tests and early local-costmap payloads did not always include
    # observed_ratio.  A positive safe record with sufficient confidence is
    # still useful; modern payloads provide the stricter per-action ratio.
    observed_ratio = _float_value(rec.get("observed_ratio"), 1.0)
    reason = str(rec.get("reason") or "local_costmap")

    unsafe_min_confidence = env_float("ROBOT_LOCAL_COSTMAP_UNSAFE_MIN_CONFIDENCE", 0.20)
    if rec.get("safe") is False:
        return MotionSafety(
            action=action,
            known=confidence >= unsafe_min_confidence,
            safe=False,
            confidence=confidence,
            observed_ratio=observed_ratio,
            reason=reason,
            record=rec,
        )

    if rec.get("safe") is True:
        min_confidence = _local_costmap_safe_min_confidence(action)
        min_observed = _local_costmap_min_observed_ratio(action, rec)
        known = confidence >= min_confidence and observed_ratio >= min_observed
        return MotionSafety(
            action=action,
            known=known,
            safe=True,
            confidence=confidence,
            observed_ratio=observed_ratio,
            reason=reason if known else f"low_evidence:{reason}",
            record=rec,
        )

    return MotionSafety(
        action=action,
        known=False,
        safe=None,
        confidence=confidence,
        observed_ratio=observed_ratio,
        reason=reason,
        record=rec,
    )


def local_costmap_known_safe(analysis: JsonDict, action: str) -> bool:
    safety = local_costmap_motion_safety(analysis, action)
    return bool(safety.known and safety.safe is True)


def local_costmap_known_unsafe(analysis: JsonDict, action: str, *, min_confidence: float = 0.20) -> bool:
    safety = local_costmap_motion_safety(analysis, action)
    return bool(safety.safe is False and safety.confidence >= float(min_confidence) and safety.known)


def local_costmap_front_corridor(analysis: JsonDict) -> JsonDict:
    costmap = analysis.get("local_costmap") if isinstance(analysis, dict) else None
    if not isinstance(costmap, dict) or str(costmap.get("status") or "") != "success":
        return {}
    corridor = costmap.get("front_corridor")
    return dict(corridor) if isinstance(corridor, dict) else {}


def local_costmap_safe_alternative(analysis: JsonDict, preferred: Sequence[str]) -> Optional[str]:
    for action in preferred:
        safety = local_costmap_motion_safety(analysis, action)
        if safety.known and safety.safe is True:
            return str(action)
    return None


@dataclass
class NavigationRecommendation:
    action: str
    reason: str
    target_cell: Optional[str]
    source: str = "navigation_memory"
    target_track_id: Optional[str] = None
    goal_type: Optional[str] = None
    target_reason: Optional[str] = None
    planner: Optional[str] = None
    plan_status: Optional[str] = None
    path: List[str] = field(default_factory=list)
    next_cell: Optional[str] = None
    route_cost: Optional[float] = None
    requested_target_cell: Optional[str] = None
    selected_goal_cell: Optional[str] = None
    selected_goal_kind: Optional[str] = None
    semantic_score: Optional[float] = None
    replan_reason: Optional[str] = None

    def as_dict(self) -> JsonDict:
        return {
            "action": self.action,
            "reason": self.reason,
            "target_cell": self.target_cell,
            "source": self.source,
            "target_track_id": self.target_track_id,
            "goal_type": self.goal_type,
            "target_reason": self.target_reason,
            "planner": self.planner,
            "plan_status": self.plan_status,
            "path": list(self.path),
            "next_cell": self.next_cell,
            "route_cost": self.route_cost,
            "requested_target_cell": self.requested_target_cell,
            "selected_goal_cell": self.selected_goal_cell,
            "selected_goal_kind": self.selected_goal_kind,
            "semantic_score": self.semantic_score,
            "replan_reason": self.replan_reason,
        }


class NavigationMemory:
    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        *,
        target_cells: int = DEFAULT_TARGET_CELLS,
    ) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.room_path = self.memory_dir / "room-state.json"
        self.target_cells = max(1, int(target_cells))
        self.position_map = PositionMap(self.memory_dir, target_cells=self.target_cells)
        self.semantic_map = SemanticMap(self.memory_dir)
        self.global_planner = AStarGlobalPlanner(self.memory_dir)
        self.drive_model = normalize_drive_model(os.getenv("ROBOT_NAV_DRIVE_MODEL", DRIVE_MODEL_NONHOLONOMIC))
        self.frontier_sticky_max_steps = max(1, env_int("ROBOT_FRONTIER_STICKY_MAX_STEPS", 8))
        self.frontier_progress_patience = max(1, env_int("ROBOT_FRONTIER_PROGRESS_PATIENCE", 3))
        self.frontier_cooldown_steps = max(1, env_int("ROBOT_FRONTIER_COOLDOWN_STEPS", 8))
        self.forward_bias_when_pose_uncertain = env_bool("ROBOT_NAV_FORWARD_BIAS_WHEN_POSE_UNCERTAIN", True)
        self.forward_bias_uncertainty_cells = max(
            0.0,
            env_float("ROBOT_NAV_FORWARD_BIAS_UNCERTAINTY_CELLS", 1.50),
        )
        self.forward_bias_after_place = env_bool("ROBOT_NAV_FORWARD_BIAS_AFTER_PLACE", True)
        self.local_forward_progress_enabled = env_bool("ROBOT_NAV_LOCAL_FORWARD_PROGRESS_ENABLED", True)
        self.local_forward_progress_frontier_distance = max(
            1,
            env_int("ROBOT_NAV_LOCAL_FORWARD_PROGRESS_FRONTIER_DISTANCE", 4),
        )
        self.local_side_probe_after_forward_blocked = env_bool(
            "ROBOT_NAV_LOCAL_SIDE_PROBE_AFTER_FORWARD_BLOCKED",
            True,
        )
        self.local_front_corner_bypass_enabled = env_bool(
            "ROBOT_NAV_LOCAL_FRONT_CORNER_BYPASS_ENABLED",
            True,
        )
        self.local_side_probe_forward_streak = max(
            1,
            env_int("ROBOT_NAV_LOCAL_SIDE_PROBE_FORWARD_STREAK", 3),
        )
        self.local_side_probe_max_visit_count = max(
            0,
            env_int("ROBOT_NAV_LOCAL_SIDE_PROBE_MAX_VISIT_COUNT", 2),
        )
        self.local_side_explore_enabled = env_bool("ROBOT_NAV_LOCAL_SIDE_EXPLORE_ENABLED", True)
        self.local_side_explore_max_visit_count = max(
            0,
            env_int("ROBOT_NAV_LOCAL_SIDE_EXPLORE_MAX_VISIT_COUNT", 0),
        )
        self.local_side_explore_frontier_distance = max(
            1,
            env_int("ROBOT_NAV_LOCAL_SIDE_EXPLORE_FRONTIER_DISTANCE", 2),
        )
        self.frontier_backtrack_guard_enabled = env_bool("ROBOT_NAV_FRONTIER_BACKTRACK_GUARD_ENABLED", True)
        self.frontier_backtrack_guard_forward_streak = max(
            1,
            env_int("ROBOT_NAV_FRONTIER_BACKTRACK_GUARD_FORWARD_STREAK", 3),
        )
        self.frontier_backtrack_filter_enabled = env_bool("ROBOT_NAV_FRONTIER_BACKTRACK_FILTER_ENABLED", True)
        self.frontier_reached_cooldown_steps = max(1, env_int("ROBOT_FRONTIER_REACHED_COOLDOWN_STEPS", 24))
        self.global_astar_pose_gate_enabled = env_bool("ROBOT_NAV_GLOBAL_ASTAR_POSE_GATE_ENABLED", True)
        self.global_astar_max_uncertainty_cells = max(
            0.0,
            env_float("ROBOT_NAV_GLOBAL_ASTAR_MAX_UNCERTAINTY_CELLS", 3.0),
        )
        self.collision_recovery_turn_steps = max(
            1,
            env_int("ROBOT_NAV_COLLISION_RECOVERY_TURN_STEPS", 3),
        )
        self.recovery_scan_escape_turns = max(
            1,
            env_int("ROBOT_NAV_RECOVERY_SCAN_ESCAPE_TURNS", 4),
        )

    def load_room(self) -> JsonDict:
        room = load_json(self.room_path, default_room_state())
        room = self.normalize_room(room)

        try:
            position_status = self.position_map.status()
            room = self._merge_position_compat(room, position_status)
        except Exception as exc:
            room["position_map_status"] = "error"
            room["position_map_error"] = str(exc)
            room["frontier_cells"] = self._global_frontier_cells(room)
            room["known_frontier_cells"] = list(room["frontier_cells"])

        reachable_frontiers = self._global_frontier_cells(room)
        if reachable_frontiers:
            room["frontier_cells"] = reachable_frontiers
        elif not room.get("frontier_cells"):
            room["frontier_cells"] = reachable_frontiers

        room["known_frontier_cells"] = list(room.get("frontier_cells", []))
        self._update_coverage(room)
        return room

    def save_room(self, room: JsonDict) -> None:
        atomic_write_json(self.room_path, self.normalize_room(room))
    
    def _merge_position_compat(self, room: JsonDict, position_status: JsonDict) -> JsonDict:
        compat = position_status.get("room_state_compat")
        if not isinstance(compat, dict):
            compat = {
                key: position_status.get(key)
                for key in [
                    "cell_size",
                    "last_cell",
                    "last_heading",
                    "visited_cells",
                    "visited_cell_counts",
                    "known_open_edges",
                    "blocked_edges",
                    "hard_blocked_edges",
                    "frontier_cells",
                    "known_frontier_cells",
                    "coverage_estimate",
                    "collision_count",
                    "position_map_status",
                    "pose_confidence",
                    "position_uncertainty_cells",
                    "heading_confidence",
                    "pose_trust",
                    "occupancy_summary",
                ]
                if key in position_status
            }

        for key, value in compat.items():
            if value is not None:
                room[key] = value

        room["position_map_status"] = "active"
        room["pose_confidence"] = position_status.get("pose_confidence", compat.get("pose_confidence"))
        room["position_uncertainty_cells"] = position_status.get(
            "position_uncertainty_cells",
            compat.get("position_uncertainty_cells"),
        )
        room["heading_confidence"] = position_status.get("heading_confidence", compat.get("heading_confidence"))
        room["pose_trust"] = position_status.get("pose_trust", compat.get("pose_trust", {}))
        room["occupancy_summary"] = position_status.get("occupancy_summary", compat.get("occupancy_summary", {}))
        return room

    def normalize_room(self, room: JsonDict) -> JsonDict:
        base = default_room_state(str(room.get("room_name") or "current_room"))
        base.update(room)
        base["cell_size"] = float(base.get("cell_size", DEFAULT_CELL_SIZE) or DEFAULT_CELL_SIZE)
        base["visited_cells"] = unique_extend([], base.get("visited_cells", []))
        raw_counts = base.get("visited_cell_counts", {})
        if not isinstance(raw_counts, dict):
            raw_counts = {}
        base["visited_cell_counts"] = {
            str(key): max(0, int(value))
            for key, value in raw_counts.items()
            if str(key).strip()
        }
        base["known_open_edges"] = unique_extend([], base.get("known_open_edges", []))
        base["hard_blocked_edges"] = unique_extend([], base.get("hard_blocked_edges", []))
        base["blocked_edges"] = unique_extend(base.get("blocked_edges", []), base["hard_blocked_edges"])
        blocked_set = set(base["blocked_edges"])
        base["known_open_edges"] = [edge for edge in base["known_open_edges"] if edge not in blocked_set]
        base["frontier_cells"] = unique_extend([], base.get("frontier_cells", []))
        base["known_frontier_cells"] = unique_extend([], base.get("known_frontier_cells", []))
        base["collision_count"] = max(0, int(base.get("collision_count", 0) or 0))
        base["oscillation_count"] = max(0, int(base.get("oscillation_count", 0) or 0))
        base["stagnation_count"] = max(0, int(base.get("stagnation_count", 0) or 0))
        base["backtrack_count"] = max(0, int(base.get("backtrack_count", 0) or 0))
        base["turn_streak_count"] = max(0, int(base.get("turn_streak_count", 0) or 0))
        try:
            coverage = float(base.get("coverage_estimate", 0.0) or 0.0)
        except (TypeError, ValueError):
            coverage = 0.0
        base["coverage_estimate"] = max(0.0, min(1.0, coverage))
        recent_actions = base.get("recent_navigation_actions", [])
        if not isinstance(recent_actions, list):
            recent_actions = []
        base["recent_navigation_actions"] = [
            str(item).strip()
            for item in recent_actions
            if str(item).strip()
        ][-16:]
        if not isinstance(base.get("last_navigation_decision"), dict):
            base["last_navigation_decision"] = {}
        if base.get("last_frontier_target") is not None:
            base["last_frontier_target"] = str(base.get("last_frontier_target"))
        if not isinstance(base.get("active_frontier_goal"), dict):
            base["active_frontier_goal"] = None
        if not isinstance(base.get("frontier_cooldowns"), dict):
            base["frontier_cooldowns"] = {}
        if not isinstance(base.get("frontier_history"), list):
            base["frontier_history"] = []
        base["frontier_history"] = [
            item
            for item in base.get("frontier_history", [])
            if isinstance(item, dict)
        ][-40:]
        base["frontier_oscillation_count"] = max(0, int(base.get("frontier_oscillation_count", 0) or 0))
        if base.get("last_new_cell") is not None:
            base["last_new_cell"] = str(base.get("last_new_cell"))
        if base.get("last_cell") is not None:
            base["last_cell"] = str(base.get("last_cell"))
        if base.get("last_heading") is not None:
            base["last_heading"] = str(base.get("last_heading"))
        return base

    def reset(self, *, room_name: str = "current_room") -> JsonDict:
        room = self.load_room()
        fresh = default_room_state(room_name)
        for key in [
            "schema_version",
            "explored_steps",
            "targets_found",
            "targets_cleaned",
            "room_complete",
            "last_update",
        ]:
            if key in room:
                fresh[key] = room[key]
        fresh["room_name"] = room_name or str(room.get("room_name") or "current_room")

        try:
            position_status = self.position_map.reset(room_name=fresh["room_name"])
            fresh = self._merge_position_compat(fresh, position_status)
        except Exception as exc:
            fresh["position_map_status"] = "error"
            fresh["position_map_error"] = str(exc)

        try:
            semantic_status = self.semantic_map.reset(room_name=fresh["room_name"])
            fresh["semantic_map_status"] = semantic_status.get("status")
            fresh["semantic_map_summary"] = semantic_status.get("stats", {})
        except Exception as exc:
            fresh["semantic_map_status"] = "error"
            fresh["semantic_map_error"] = str(exc)

        try:
            self.global_planner.reset()
        except Exception as exc:
            fresh["global_planner_status"] = "error"
            fresh["global_planner_error"] = str(exc)

        fresh["navigation_last_update"] = now_iso()
        self.save_room(fresh)
        return self.status()

    def observe(
        self,
        *,
        vision: JsonDict,
        analysis: Optional[JsonDict] = None,
        persist: bool = True,
    ) -> JsonDict:
        room = self.load_room()
        analysis = analysis or {}

        try:
            position_status = self.position_map.observe(
                vision=vision,
                analysis=analysis,
                persist=persist,
            )
            room = self._merge_position_compat(room, position_status)
        except Exception as exc:
            cell, heading = pose_from_vision(
                vision,
                fallback_cell=str(room.get("last_cell") or "0,0"),
                fallback_heading=str(room.get("last_heading") or "north"),
            )
            self._ensure_cell(room, cell)
            room["last_cell"] = cell
            room["last_heading"] = heading
            self._record_observed_open_edges(room, cell, heading, analysis)
            room["position_map_status"] = "error"
            room["position_map_error"] = str(exc)

        reachable_frontiers = self._global_frontier_cells(room)
        if reachable_frontiers:
            room["frontier_cells"] = reachable_frontiers
        elif not room.get("frontier_cells"):
            room["frontier_cells"] = reachable_frontiers
        room["known_frontier_cells"] = list(room["frontier_cells"])
        self._update_coverage(room)
        room["navigation_last_update"] = now_iso()

        if persist:
            self.save_room(room)
            return self.status()
        return self._status_from_room(room)
    
    def observe_semantics(
        self,
        *,
        analysis: JsonDict,
        step: int = 0,
        persist: bool = True,
    ) -> JsonDict:
        """Fuse current perception + object-memory tracks onto position-map cells."""
        nav_status = self.status()
        result = self.semantic_map.update_from_observation(
            analysis or {},
            navigation_status=nav_status,
            step=int(step),
            persist=persist,
        )
        room = self.load_room()
        room["semantic_map_status"] = result.get("status")
        room["semantic_map_summary"] = result.get("stats", {})
        room["navigation_last_update"] = now_iso()
        if persist:
            self.save_room(room)
        return result

    def record_step(
        self,
        *,
        action: str,
        success: bool,
        vision: JsonDict,
        analysis: Optional[JsonDict] = None,
        action_result: Optional[JsonDict] = None,
        failure_reason: Optional[str] = None,
        recommendation: Optional[JsonDict] = None,
    ) -> JsonDict:
        room = self.load_room()
        action_result = action_result or {}
        analysis = analysis or {}

        try:
            position_status = self.position_map.record_action(
                action=action,
                success=success,
                vision=vision,
                analysis=analysis,
                action_result=action_result,
                failure_reason=failure_reason,
                persist=True,
            )
            room = self._merge_position_compat(room, position_status)
        except Exception as exc:
            before_cell, before_heading, after_cell, after_heading = pose_from_action_feedback(
                room=room,
                action=action,
                success=success,
                action_result=action_result,
                fallback_vision=vision,
            )

            self._ensure_cell(room, before_cell)
            self._record_observed_open_edges(room, before_cell, before_heading, analysis)
            new_cell = False
            if success and action in TRANSLATION_ACTIONS:
                new_cell = self._visit_cell(room, after_cell)
                self._add_known_open_edge(
                    room,
                    before_cell,
                    after_cell,
                    override_blocked=True,
                    override_hard=True,
                )
            elif success:
                new_cell = self._ensure_cell(room, after_cell)

            room["last_cell"] = after_cell if success else before_cell
            room["last_heading"] = after_heading if success else before_heading

            if not success and action in TRANSLATION_ACTIONS:
                target = after_cell if after_cell != before_cell else (
                    neighbor_cell(before_cell, before_heading) if action == "MoveAhead" else
                    neighbor_cell(before_cell, opposite_heading(before_heading)) if action == "MoveBack" else
                    neighbor_cell(before_cell, left_heading(before_heading)) if action == "MoveLeft" else
                    neighbor_cell(before_cell, right_heading(before_heading))
                )
                self._mark_blocked_edge(room, before_cell, target, hard=True)
                room["collision_count"] = int(room.get("collision_count", 0)) + 1

            if new_cell:
                room["stagnation_count"] = 0
                room["last_new_cell"] = after_cell
            else:
                room["stagnation_count"] = int(room.get("stagnation_count", 0)) + 1

            room["position_map_status"] = "error"
            room["position_map_error"] = str(exc)

        if action == "MoveBack":
            room["backtrack_count"] = int(room.get("backtrack_count", 0)) + 1
        elif success and action in {"MoveAhead", "MoveLeft", "MoveRight"}:
            room["backtrack_count"] = max(0, int(room.get("backtrack_count", 0)) - 1)

        if action in ROTATE_ACTIONS and success:
            room["turn_streak_count"] = int(room.get("turn_streak_count", 0)) + 1
        elif action in TRANSLATION_ACTIONS and success:
            room["turn_streak_count"] = 0

        if self._is_rotation_oscillation(room.get("recent_navigation_actions", []), action):
            room["oscillation_count"] = int(room.get("oscillation_count", 0)) + 1

        recent = list(room.get("recent_navigation_actions", []))
        recent.append(action if success else f"{action}:failed:{failure_reason or 'unknown'}")
        room["recent_navigation_actions"] = recent[-16:]

        if isinstance(recommendation, dict) and recommendation:
            room["last_navigation_decision"] = dict(recommendation)

        self._update_frontier_goal_feedback(
            room,
            action=action,
            success=success,
            failure_reason=failure_reason,
            recommendation=recommendation,
        )

        reachable_frontiers = self._global_frontier_cells(room)
        if reachable_frontiers:
            room["frontier_cells"] = reachable_frontiers
        elif not room.get("frontier_cells"):
            room["frontier_cells"] = reachable_frontiers
        room["known_frontier_cells"] = list(room["frontier_cells"])
        self._update_coverage(room)
        room["navigation_last_update"] = now_iso()
        self.save_room(room)
        return self.status()
    
    def recommend(
        self,
        *,
        vision: JsonDict,
        analysis: JsonDict,
        recent_actions: Optional[Sequence[str]] = None,
        recent_failed_action: Optional[str] = None,
        target_cell: Optional[str] = None,
        target_reason: Optional[str] = None,
        target_track_id: Optional[str] = None,
        goal_type: Optional[str] = None,
        target_heading: Optional[str] = None,
        persist: bool = True,
    ) -> JsonDict:
        room = self.load_room()
        try:
            position_status = self.position_map.observe(
                vision=vision,
                analysis=analysis,
                persist=persist,
            )
            room = self._merge_position_compat(room, position_status)
        except Exception as exc:
            cell, heading = pose_from_vision(
                vision,
                fallback_cell=str(room.get("last_cell") or "0,0"),
                fallback_heading=str(room.get("last_heading") or "north"),
            )
            self._ensure_cell(room, cell)
            room["last_cell"] = cell
            room["last_heading"] = heading
            self._record_observed_open_edges(room, cell, heading, analysis)
            room["position_map_status"] = "error"
            room["position_map_error"] = str(exc)

        cell = str(room.get("last_cell") or "0,0")
        heading = str(room.get("last_heading") or "north")
        reachable_frontiers = self._global_frontier_cells(room)
        if reachable_frontiers:
            room["frontier_cells"] = reachable_frontiers
        elif not room.get("frontier_cells"):
            room["frontier_cells"] = reachable_frontiers
        room["known_frontier_cells"] = list(room["frontier_cells"])

        recommendation = self._recommend_action(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=list(recent_actions or []),
            recent_failed_action=recent_failed_action,
            target_cell=target_cell,
            target_reason=target_reason,
            target_track_id=target_track_id,
            goal_type=goal_type,
            target_heading=target_heading,
        )

        room["last_navigation_decision"] = recommendation.as_dict()
        self._update_coverage(room)
        room["navigation_last_update"] = now_iso()
        if persist:
            self.save_room(room)
            result = self.status()
        else:
            result = self._status_from_room(room)
        result["recommendation"] = recommendation.as_dict()
        return result

    def status(self) -> JsonDict:
        room = self.load_room()
        return self._status_from_room(room)

    def _status_from_room(self, room: JsonDict) -> JsonDict:
        try:
            semantic_status = self.semantic_map.status()
            semantic_summary = semantic_status.get("stats", {})
        except Exception as exc:
            semantic_status = {"status": "error", "message": str(exc)}
            semantic_summary = {}
        try:
            last_global_plan = self.global_planner.last_plan()
        except Exception as exc:
            last_global_plan = {"status": "error", "message": str(exc), "planner": "astar"}
        return {
            "status": "success",
            "result_type": "navigation_memory_status",
            "room_name": room.get("room_name"),
            "cell_size": room.get("cell_size"),
            "last_cell": room.get("last_cell"),
            "last_heading": room.get("last_heading"),
            "visited_cell_count": len(room.get("visited_cells", [])),
            "visited_cells": room.get("visited_cells", []),
            "frontier_cells": room.get("frontier_cells", []),
            "reachable_frontier_count": len(room.get("frontier_cells", [])),
            "known_open_edges": room.get("known_open_edges", []),
            "known_frontier_cells": room.get("known_frontier_cells", []),
            "blocked_edges": room.get("blocked_edges", []),
            "hard_blocked_edges": room.get("hard_blocked_edges", []),
            "coverage_estimate": room.get("coverage_estimate", 0.0),
            "collision_count": room.get("collision_count", 0),
            "oscillation_count": room.get("oscillation_count", 0),
            "stagnation_count": room.get("stagnation_count", 0),
            "backtrack_count": room.get("backtrack_count", 0),
            "turn_streak_count": room.get("turn_streak_count", 0),
            "last_new_cell": room.get("last_new_cell"),
            "recent_navigation_actions": room.get("recent_navigation_actions", []),
            "last_navigation_decision": room.get("last_navigation_decision", {}),
            "drive_model": self.drive_model,
            "last_frontier_target": room.get("last_frontier_target"),
            "active_frontier_goal": room.get("active_frontier_goal"),
            "frontier_cooldowns": room.get("frontier_cooldowns", {}),
            "frontier_oscillation_count": room.get("frontier_oscillation_count", 0),
            "navigation_last_update": room.get("navigation_last_update"),
            "position_map_status": room.get("position_map_status"),
            "pose_confidence": room.get("pose_confidence"),
            "position_uncertainty_cells": room.get("position_uncertainty_cells"),
            "heading_confidence": room.get("heading_confidence"),
            "pose_trust": room.get("pose_trust", {}),
            "occupancy_summary": room.get("occupancy_summary", {}),
            "coverage_model": "fixed_target_cells_with_reachable_frontier_completion",
            "semantic_map_status": semantic_status.get("status"),
            "semantic_map_summary": semantic_summary,
            "semantic_frontier_scores": semantic_status.get("frontier_scores", {}),
            "last_global_plan": last_global_plan,
        }

    def _ensure_cell(self, room: JsonDict, cell: str) -> bool:
        was_new = cell not in set(room.get("visited_cells", []))
        room["visited_cells"] = unique_extend(room.get("visited_cells", []), [cell])
        counts = room.get("visited_cell_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        counts.setdefault(cell, 0)
        room["visited_cell_counts"] = counts
        return was_new

    def _visit_cell(self, room: JsonDict, cell: str) -> bool:
        was_new = self._ensure_cell(room, cell)
        counts = room.get("visited_cell_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        counts[cell] = int(counts.get(cell, 0)) + 1
        room["visited_cell_counts"] = counts
        return was_new

    def _heading_by_direction(self, heading: str) -> Dict[str, str]:
        return {
            "forward": heading,
            "left": left_heading(heading),
            "right": right_heading(heading),
        }

    def _record_observed_open_edges(
        self,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
    ) -> None:
        if not isinstance(analysis, dict):
            return
        open_directions = set(analysis.get("open_directions", []) or [])
        forward_target = neighbor_cell(cell, heading)
        forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
        if forward_safety.known and forward_safety.safe is True:
            self._observe_safe_edge(
                room,
                cell,
                forward_target,
                action="MoveAhead",
                safety=forward_safety,
            )
            return
        if forward_safety.known and forward_safety.safe is False:
            self._mark_blocked_edge(room, cell, forward_target)
            return
        if "forward" in open_directions and not boolish(analysis.get("obstacle_ahead", False)):
            target = neighbor_cell(cell, heading)
            self._add_known_open_edge(room, cell, target)

    def _add_known_open_edge(
        self,
        room: JsonDict,
        source: str,
        target: str,
        *,
        override_blocked: bool = False,
        override_hard: bool = False,
    ) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        if self._edge_blocked(room, source, target):
            if not override_blocked:
                return
            self._clear_blocked_edge(room, source, target, clear_hard=override_hard)
            if self._edge_blocked(room, source, target):
                return
        room["known_open_edges"] = unique_extend(room.get("known_open_edges", []), [direct, reverse])

    def _mark_blocked_edge(self, room: JsonDict, source: str, target: str, *, hard: bool = False) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        room["blocked_edges"] = unique_extend(room.get("blocked_edges", []), [direct, reverse])
        if hard:
            room["hard_blocked_edges"] = unique_extend(room.get("hard_blocked_edges", []), [direct, reverse])
        known = [
            edge
            for edge in room.get("known_open_edges", [])
            if edge not in {direct, reverse}
        ]
        room["known_open_edges"] = known
        room["frontier_cells"] = [
            cell
            for cell in room.get("frontier_cells", []) or []
            if str(cell) != str(target)
        ]
        room["known_frontier_cells"] = [
            cell
            for cell in room.get("known_frontier_cells", []) or []
            if str(cell) != str(target)
        ]

    def _clear_blocked_edge(self, room: JsonDict, source: str, target: str, *, clear_hard: bool = False) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        if not clear_hard:
            hard_blocked = set(room.get("hard_blocked_edges", []) or [])
            if direct in hard_blocked or reverse in hard_blocked:
                return
        room["blocked_edges"] = [
            edge
            for edge in room.get("blocked_edges", []) or []
            if edge not in {direct, reverse}
        ]
        if clear_hard:
            room["hard_blocked_edges"] = [
                edge
                for edge in room.get("hard_blocked_edges", []) or []
                if edge not in {direct, reverse}
            ]

    def _edge_hard_blocked(self, room: JsonDict, source: str, target: str) -> bool:
        hard = set(room.get("hard_blocked_edges", []) or [])
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        return direct in hard or reverse in hard

    def _observe_safe_edge(
        self,
        room: JsonDict,
        source: str,
        target: str,
        *,
        action: str,
        safety: MotionSafety,
    ) -> bool:
        """Apply dynamic costmap clearing for a currently safe swept volume.

        A confident RGB-D local costmap observation can clear stale/soft
        blocked edges.  Edges created by actual translation collisions remain
        hard-blocked until the robot physically traverses them successfully.
        """
        if source == target:
            return False
        if not (safety.known and safety.safe is True):
            return False
        if self._edge_hard_blocked(room, source, target):
            return False
        was_blocked = self._edge_blocked(room, source, target)
        self._add_known_open_edge(
            room,
            source,
            target,
            override_blocked=True,
            override_hard=False,
        )
        if was_blocked and not self._edge_blocked(room, source, target):
            history = room.setdefault("frontier_history", [])
            if isinstance(history, list):
                history.append(
                    {
                        "time": now_iso(),
                        "step": self._navigation_step(room),
                        "event": "soft_blocked_edge_cleared",
                        "edge": edge_pair_key(source, target),
                        "action": action,
                        "confidence": round(float(safety.confidence), 4),
                        "observed_ratio": round(float(safety.observed_ratio), 4),
                        "reason": safety.reason,
                    }
                )
                room["frontier_history"] = history[-40:]
        return not self._edge_blocked(room, source, target)

    def _compute_local_frontier_cells(self, room: JsonDict, cell: str, heading: str, analysis: JsonDict) -> List[str]:
        open_directions = set(analysis.get("open_directions", []) or [])
        visited = set(room.get("visited_cells", []))
        frontiers: List[str] = []
        for direction, candidate_heading in self._heading_by_direction(heading).items():
            target = neighbor_cell(cell, candidate_heading)
            action = "MoveAhead" if direction == "forward" else "MoveLeft" if direction == "left" else "MoveRight"
            safety = local_costmap_motion_safety(analysis, action)
            if safety.known:
                if safety.safe is not True:
                    continue
            elif direction not in open_directions:
                continue
            if self._edge_blocked(room, cell, target):
                continue
            if target not in visited:
                frontiers.append(target)
        return unique_extend([], frontiers)

    def _reachable_cells_from_open_edges(self, room: JsonDict, start: str) -> set[str]:
        """Return the connected known-open component for the current pose."""
        start_cell = str(start or "").strip()
        if not start_cell:
            return set()
        blocked_pairs = set()
        for edge in room.get("blocked_edges", []) or []:
            parsed = parse_edge(str(edge))
            if parsed:
                blocked_pairs.add(edge_pair_key(parsed[0], parsed[1]))

        adjacency: Dict[str, List[str]] = {}
        for edge in room.get("known_open_edges", []) or []:
            parsed = parse_edge(str(edge))
            if not parsed:
                continue
            source, target = parsed
            if edge_pair_key(source, target) in blocked_pairs:
                continue
            adjacency.setdefault(source, []).append(target)
            adjacency.setdefault(target, []).append(source)

        reachable = {start_cell}
        stack = [start_cell]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, []):
                if neighbor in reachable:
                    continue
                reachable.add(neighbor)
                stack.append(neighbor)
        return reachable

    def _global_frontier_cells(self, room: JsonDict) -> List[str]:
        visited = set(room.get("visited_cells", []))
        current_cell = str(room.get("last_cell") or "").strip()
        reachable_sources = self._reachable_cells_from_open_edges(room, current_cell) if current_cell else set(visited)
        if not reachable_sources:
            reachable_sources = set(visited)
        frontiers: List[str] = []
        for edge in room.get("known_open_edges", []) or []:
            parsed = parse_edge(str(edge))
            if not parsed:
                continue
            source, target = parsed
            if source not in reachable_sources:
                continue
            if target in visited:
                continue
            if self._edge_blocked(room, source, target):
                continue
            frontiers.append(target)
        return unique_extend([], frontiers)

    def _update_coverage(self, room: JsonDict) -> None:
        visited = len(room.get("visited_cells", []))
        blocked = len(room.get("blocked_edges", []))
        # Conservative heuristic: blocked edges count a little because they
        # represent explored boundaries, but visited cells dominate coverage.
        coverage = min(1.0, (visited + 0.25 * blocked) / float(self.target_cells))
        room["coverage_estimate"] = round(coverage, 3)

    def _edge_blocked(self, room: JsonDict, source: str, target: str) -> bool:
        blocked = set(room.get("blocked_edges", []))
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        return direct in blocked or reverse in blocked

    def _pose_requires_relocalization(self, room: JsonDict) -> bool:
        if not self.global_astar_pose_gate_enabled:
            return False
        pose_trust = room.get("pose_trust") if isinstance(room.get("pose_trust"), dict) else {}
        try:
            uncertainty = float(room.get("position_uncertainty_cells", 0.0) or 0.0)
        except (TypeError, ValueError):
            uncertainty = 0.0
        level = str(pose_trust.get("level") or "").lower()
        return bool(
            level == "low"
            or bool(pose_trust.get("relocalization_recommended")) and uncertainty >= self.global_astar_max_uncertainty_cells
            or uncertainty >= self.global_astar_max_uncertainty_cells
        )

    def _recent_translation_failure(self, recent_actions: Sequence[str], recent_failed_action: Optional[str]) -> bool:
        if self._base_action(recent_failed_action) in TRANSLATION_ACTIONS:
            return True
        if not recent_actions:
            return False
        last = str(recent_actions[-1])
        return "failed" in last and self._base_action(last) in TRANSLATION_ACTIONS

    def _stable_scan_turn(self, preferred: str, analysis: JsonDict, recent_actions: Sequence[str]) -> str:
        preferred = preferred if preferred in ROTATE_ACTIONS else "RotateLeft"
        last = self._base_action(recent_actions[-1]) if recent_actions else None
        if last in ROTATE_ACTIONS and not local_costmap_known_unsafe(analysis, last):
            return str(last)
        if not local_costmap_known_unsafe(analysis, preferred):
            return preferred
        opposite = "RotateRight" if preferred == "RotateLeft" else "RotateLeft"
        if not local_costmap_known_unsafe(analysis, opposite):
            return opposite
        return "LookDown"

    def _local_relocalization_recovery(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        reason: str,
    ) -> NavigationRecommendation:
        forward_cell = neighbor_cell(cell, heading)
        forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
        if forward_safety.known and forward_safety.safe is True:
            self._observe_safe_edge(
                room,
                cell,
                forward_cell,
                action="MoveAhead",
                safety=forward_safety,
            )
            if not self._edge_blocked(room, cell, forward_cell):
                return NavigationRecommendation(
                    "MoveAhead",
                    f"{reason}_local_recovery_forward_clear:"
                    f"confidence={forward_safety.confidence:.2f};observed={forward_safety.observed_ratio:.2f}",
                    forward_cell,
                    source="local_recovery",
                )
        forward_blocked = bool(
            (forward_safety.known and forward_safety.safe is False)
            or self._edge_blocked(room, cell, forward_cell)
            or boolish(analysis.get("obstacle_ahead", False))
        )
        back_safety = local_costmap_motion_safety(analysis, "MoveBack")
        back_safe = bool(back_safety.known and back_safety.safe is True and not self._recent_moveback_loop(recent_actions))
        if forward_blocked and back_safe:
            return NavigationRecommendation(
                "MoveBack",
                f"{reason}_local_recovery_backoff:forward_reason={forward_safety.reason}",
                None,
                source="local_recovery",
            )

        left_target = neighbor_cell(cell, left_heading(heading))
        right_target = neighbor_cell(cell, right_heading(heading))
        left_blocked = self._edge_blocked(room, cell, left_target) or local_costmap_known_unsafe(analysis, "RotateLeft")
        right_blocked = self._edge_blocked(room, cell, right_target) or local_costmap_known_unsafe(analysis, "RotateRight")
        preferred = "RotateRight" if left_blocked and not right_blocked else "RotateLeft"
        action = self._stable_scan_turn(preferred, analysis, recent_actions)
        target = right_target if action == "RotateRight" else left_target if action == "RotateLeft" else cell
        return NavigationRecommendation(
            action,
            f"{reason}_local_recovery_scan:forward_blocked={forward_blocked};pose_gate={self._pose_requires_relocalization(room)}",
            target,
            source="local_recovery",
        )

    def _turn_action_toward(self, heading: str, desired_heading: str, recent_actions: Sequence[str]) -> str:
        if desired_heading == heading:
            return "MoveAhead"
        if desired_heading == left_heading(heading):
            return self._stable_turn("RotateLeft", recent_actions)
        if desired_heading == right_heading(heading):
            return self._stable_turn("RotateRight", recent_actions)
        # Facing the opposite direction: keep rotating in one stable direction
        # instead of alternating left/right around the same cell.
        return self._stable_turn("RotateRight", recent_actions)

    def _is_rotation_oscillation(self, recent_actions: Sequence[str], action: str) -> bool:
        if action not in ROTATE_ACTIONS or len(recent_actions) < 2:
            return False
        last = self._base_action(recent_actions[-1])
        previous = self._base_action(recent_actions[-2])
        if action == "RotateLeft":
            return last == "RotateRight" and previous == "RotateLeft"
        if action == "RotateRight":
            return last == "RotateLeft" and previous == "RotateRight"
        return False

    def _base_action(self, action: Optional[str]) -> Optional[str]:
        if not action:
            return None
        value = str(action)
        for known in ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "clean-garbage"]:
            if value.startswith(known):
                return known
        return value

    def _recent_moveback_loop(self, recent_actions: Sequence[str]) -> bool:
        bases = [self._base_action(action) for action in recent_actions[-6:]]
        text = ",".join(str(item) for item in bases)
        return (
            "MoveAhead,MoveBack" in text
            or "MoveBack,MoveAhead" in text
            or bases.count("MoveBack") >= 2
        )

    def _recent_lateral_oscillation(self, recent_actions: Sequence[str]) -> bool:
        bases = [self._base_action(action) for action in recent_actions[-6:]]
        lateral = [action for action in bases if action in {"MoveLeft", "MoveRight"}]
        if len(lateral) < 4:
            return False
        last_four = bases[-4:]
        if not all(action in {"MoveLeft", "MoveRight"} for action in last_four):
            return False
        return all(left != right for left, right in zip(last_four, last_four[1:]))

    def _recent_action_streak(self, recent_actions: Sequence[str], action: str) -> int:
        streak = 0
        for recent in reversed(list(recent_actions)):
            if self._base_action(recent) != action:
                break
            streak += 1
        return streak

    def _recent_forward_progress_count(self, recent_actions: Sequence[str], *, window: int = 8) -> int:
        bases = [self._base_action(action) for action in list(recent_actions)[-max(1, int(window)):]]
        if "MoveBack" in bases:
            return 0
        return sum(1 for action in bases if action == "MoveAhead")

    def _cell_visit_count(self, room: JsonDict, cell: str) -> int:
        counts = room.get("visited_cell_counts") if isinstance(room.get("visited_cell_counts"), dict) else {}
        try:
            return int(counts.get(str(cell), 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _local_side_probe_after_forward_block(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
    ) -> Optional[NavigationRecommendation]:
        if not self.local_side_probe_after_forward_blocked:
            return None
        forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
        forward_blocked = bool(forward_safety.known and forward_safety.safe is False)
        if not forward_blocked and not self._edge_blocked(room, cell, neighbor_cell(cell, heading)):
            return None
        forward_streak = self._recent_action_streak(recent_actions, "MoveAhead")
        forward_progress_count = self._recent_forward_progress_count(recent_actions)
        if (
            forward_streak < self.local_side_probe_forward_streak
            and forward_progress_count < self.local_side_probe_forward_streak
        ):
            return None
        if self._recent_lateral_oscillation(recent_actions):
            return None

        side_specs = [
            ("MoveLeft", "RotateLeft", neighbor_cell(cell, left_heading(heading)), "left"),
            ("MoveRight", "RotateRight", neighbor_cell(cell, right_heading(heading)), "right"),
        ]
        last_action = self._base_action(recent_actions[-1]) if recent_actions else None
        candidates: List[Tuple[int, int, int, str, str, str]] = []
        for index, (lateral_action, turn_action, target, side) in enumerate(side_specs):
            action = lateral_action if self.drive_model == DRIVE_MODEL_HOLONOMIC else turn_action
            if recent_failed_action == action:
                continue
            if self._edge_blocked(room, cell, target):
                continue
            lateral_safety = local_costmap_motion_safety(analysis, lateral_action)
            if lateral_safety.known and lateral_safety.safe is False:
                continue
            if self.drive_model == DRIVE_MODEL_HOLONOMIC:
                if not (lateral_safety.known and lateral_safety.safe is True):
                    continue
            elif local_costmap_known_unsafe(analysis, turn_action):
                continue
            visit_count = self._cell_visit_count(room, target)
            if visit_count > self.local_side_probe_max_visit_count:
                continue
            opposite_recent_penalty = 1 if (
                (last_action == "MoveLeft" and action == "MoveRight")
                or (last_action == "MoveRight" and action == "MoveLeft")
                or (last_action == "RotateLeft" and action == "RotateRight")
                or (last_action == "RotateRight" and action == "RotateLeft")
            ) else 0
            candidates.append((visit_count, opposite_recent_penalty, index, action, target, side))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        visit_count, _, _, action, target, side = candidates[0]
        return NavigationRecommendation(
            action,
            (
                "local_side_probe_after_forward_blocked:"
                f"forward_streak={forward_streak};forward_progress={forward_progress_count};"
                f"side={side};visit_count={visit_count};"
                f"drive_model={self.drive_model};forward_reason={forward_safety.reason}"
            ),
            target,
            source="local_bootstrap",
        )

    def _local_front_corner_bypass(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_failed_action: Optional[str],
    ) -> Optional[NavigationRecommendation]:
        """Bypass a one-sided near-front obstacle using a visible side lane.

        This is the discrete-action equivalent of a local planner steering
        around a handle/chair leg that clips one side of the forward swept
        volume.  It must be backed by RGB-D lane evidence; otherwise the older
        recovery and frontier logic remains responsible.
        """
        if not self.local_front_corner_bypass_enabled:
            return None
        forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
        if not (forward_safety.known and forward_safety.safe is False):
            return None
        corridor = local_costmap_front_corridor(analysis)
        if not bool(corridor.get("asymmetric_bypass_available")):
            return None
        side = str(corridor.get("preferred_bypass_side") or "")
        if side not in {"left", "right"}:
            return None
        lanes = corridor.get("lanes") if isinstance(corridor.get("lanes"), dict) else {}
        lane = lanes.get(side) if isinstance(lanes.get(side), dict) else {}
        if not bool(lane.get("safe")):
            return None

        target = neighbor_cell(cell, left_heading(heading) if side == "left" else right_heading(heading))
        if self._edge_blocked(room, cell, target) or self._edge_hard_blocked(room, cell, target):
            return None

        lateral_action = "MoveLeft" if side == "left" else "MoveRight"
        turn_action = "RotateLeft" if side == "left" else "RotateRight"
        lateral_safety = local_costmap_motion_safety(analysis, lateral_action)
        if lateral_safety.safe is True and lateral_safety.confidence >= 0.20:
            action = lateral_action
        else:
            if local_costmap_known_unsafe(analysis, turn_action):
                return None
            action = turn_action
        if self._base_action(recent_failed_action) == action:
            return None

        return NavigationRecommendation(
            action,
            (
                "local_front_corner_bypass:"
                f"blocked_side={corridor.get('dominant_blocker_side')};"
                f"bypass={side};lane_observed={lane.get('observed_ratio')};"
                f"lane_reason={lane.get('reason')};forward_reason={forward_safety.reason};"
                f"lateral_reason={lateral_safety.reason};"
                f"drive_model={self.drive_model}"
            ),
            target,
            source="local_bootstrap",
        )

    def _local_side_exploration_probe(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
    ) -> Optional[NavigationRecommendation]:
        """Use nearby side openings before sending A* to an old global frontier.

        In the default non-holonomic mode this returns a rotation, not a blind
        side translation. The next frame must validate that side with RGB-D
        before any forward motion happens.
        """
        if not self.local_side_explore_enabled:
            return None
        if self._recent_lateral_oscillation(recent_actions):
            return None

        visited_cells = {str(item) for item in (room.get("visited_cells", []) or [])}
        frontiers = list(room.get("frontier_cells", []) or [])
        open_directions = set(str(item) for item in (analysis.get("open_directions", []) or []))
        last_action = self._base_action(recent_actions[-1]) if recent_actions else None

        side_specs = [
            ("MoveLeft", "RotateLeft", neighbor_cell(cell, left_heading(heading)), "left"),
            ("MoveRight", "RotateRight", neighbor_cell(cell, right_heading(heading)), "right"),
        ]
        candidates: List[Tuple[int, int, int, int, int, str, str, str]] = []
        for index, (lateral_action, turn_action, target, side) in enumerate(side_specs):
            action = lateral_action if self.drive_model == DRIVE_MODEL_HOLONOMIC else turn_action
            if recent_failed_action == action:
                continue
            if self._edge_blocked(room, cell, target) or self._edge_hard_blocked(room, cell, target):
                continue

            lateral_safety = local_costmap_motion_safety(analysis, lateral_action)
            if lateral_safety.known and lateral_safety.safe is False:
                continue
            if self.drive_model == DRIVE_MODEL_HOLONOMIC:
                if not (lateral_safety.known and lateral_safety.safe is True):
                    continue
            else:
                if local_costmap_known_unsafe(analysis, turn_action):
                    continue
                if side not in open_directions and not lateral_safety.known:
                    continue

            visit_count = self._cell_visit_count(room, target)
            nearby_frontier_score = self._frontier_near_cell_score(
                target_cell=target,
                frontiers=frontiers,
                max_distance=self.local_side_explore_frontier_distance,
            )
            unvisited = target not in visited_cells
            if not unvisited and nearby_frontier_score <= 0:
                continue
            if visit_count > self.local_side_explore_max_visit_count and nearby_frontier_score <= 0:
                continue

            opposite_recent_penalty = 1 if (
                (last_action == "RotateLeft" and action == "RotateRight")
                or (last_action == "RotateRight" and action == "RotateLeft")
                or (last_action == "MoveLeft" and action == "MoveRight")
                or (last_action == "MoveRight" and action == "MoveLeft")
            ) else 0
            candidates.append(
                (
                    0 if unvisited else 1,
                    -nearby_frontier_score,
                    visit_count,
                    opposite_recent_penalty,
                    index,
                    action,
                    target,
                    side,
                )
            )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:5])
        _, frontier_sort, visit_count, _, _, action, target, side = candidates[0]
        return NavigationRecommendation(
            action,
            (
                "local_side_explore_before_global_frontier:"
                f"side={side};visit_count={visit_count};"
                f"nearby_frontiers={-frontier_sort};drive_model={self.drive_model}"
            ),
            target,
            source="local_bootstrap",
        )

    def _frontier_ahead_score(
        self,
        *,
        cell: str,
        heading: str,
        frontiers: Sequence[Any],
        max_distance: int,
    ) -> int:
        """Count frontier cells in a short forward cone.

        This is the local planner's information-gain cue. It prevents a
        slightly higher semantic frontier off to the side from stealing control
        when the current RGB-D view shows a safe corridor leading toward
        unknown space.
        """
        try:
            cx, cz = parse_cell(cell)
        except Exception:
            cx, cz = 0, 0
        fdx, fdz = HEADING_VECTORS.get(heading, (0, 1))
        score = 0
        for raw in frontiers or []:
            try:
                fx, fz = parse_cell(str(raw))
            except Exception:
                continue
            dx = fx - cx
            dz = fz - cz
            forward_distance = dx * fdx + dz * fdz
            lateral_distance = abs(dx * fdz - dz * fdx)
            if 1 <= forward_distance <= int(max_distance) and lateral_distance <= max(1, forward_distance):
                score += 1
        return score

    def _frontier_projection(self, *, cell: str, heading: str, frontier: Any) -> Tuple[int, int]:
        """Return heading-relative forward and lateral distance for a frontier."""
        try:
            cx, cz = parse_cell(str(cell))
            fx, fz = parse_cell(str(frontier))
        except Exception:
            return 0, 999
        fdx, fdz = HEADING_VECTORS.get(str(heading), (0, 1))
        dx = fx - cx
        dz = fz - cz
        forward_distance = dx * fdx + dz * fdz
        lateral_distance = abs(dx * fdz - dz * fdx)
        return int(forward_distance), int(lateral_distance)

    def _frontiers_not_behind(self, *, cell: str, heading: str, frontiers: Sequence[Any]) -> List[str]:
        """Prefer frontier goals in the current half-plane during free exploration."""
        preferred: List[str] = []
        for raw in frontiers or []:
            frontier = str(raw or "").strip()
            if not frontier:
                continue
            forward_distance, _ = self._frontier_projection(cell=cell, heading=heading, frontier=frontier)
            if forward_distance >= 0:
                preferred.append(frontier)
        return unique_extend([], preferred)

    def _frontier_near_cell_score(
        self,
        *,
        target_cell: str,
        frontiers: Sequence[Any],
        max_distance: int,
    ) -> int:
        try:
            tx, tz = parse_cell(str(target_cell))
        except Exception:
            return 0
        score = 0
        for raw in frontiers or []:
            try:
                fx, fz = parse_cell(str(raw))
            except Exception:
                continue
            if abs(fx - tx) + abs(fz - tz) <= int(max_distance):
                score += 1
        return score

    def _navigation_step(self, room: JsonDict) -> int:
        try:
            return int(room.get("explored_steps", 0) or 0)
        except (TypeError, ValueError):
            return len(room.get("recent_navigation_actions", []) or [])

    def _active_frontier_cooldowns(self, room: JsonDict, step: int) -> JsonDict:
        raw = room.get("frontier_cooldowns") if isinstance(room.get("frontier_cooldowns"), dict) else {}
        active: JsonDict = {}
        for cell, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            try:
                until_step = int(entry.get("until_step", 0) or 0)
            except (TypeError, ValueError):
                until_step = 0
            if until_step >= step:
                active[str(cell)] = dict(entry)
        room["frontier_cooldowns"] = active
        return active

    def _append_frontier_history(self, room: JsonDict, event: str, **payload: Any) -> None:
        history = room.setdefault("frontier_history", [])
        if not isinstance(history, list):
            history = []
            room["frontier_history"] = history
        entry = {"time": now_iso(), "step": self._navigation_step(room), "event": event}
        entry.update(payload)
        history.append(entry)
        room["frontier_history"] = history[-40:]

    def _cooldown_frontier(self, room: JsonDict, cell: Optional[str], *, reason: str, steps: Optional[int] = None) -> None:
        frontier = str(cell or "").strip()
        if not frontier:
            return
        step = self._navigation_step(room)
        cooldowns = self._active_frontier_cooldowns(room, step)
        previous = cooldowns.get(frontier) if isinstance(cooldowns.get(frontier), dict) else {}
        failure_count = int(previous.get("failure_count", 0) or 0) + 1
        duration = max(1, int(steps if steps is not None else self.frontier_cooldown_steps))
        cooldowns[frontier] = {
            "cell": frontier,
            "reason": reason,
            "until_step": step + duration,
            "failure_count": failure_count,
            "last_update": now_iso(),
        }
        room["frontier_cooldowns"] = cooldowns
        active = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else None
        if active and str(active.get("cell") or "") == frontier:
            room["active_frontier_goal"] = None
        self._append_frontier_history(room, "frontier_cooldown", cell=frontier, reason=reason, until_step=step + duration)

    def _cooldown_frontier_cluster(
        self,
        room: JsonDict,
        cell: Optional[str],
        *,
        reason: str,
        radius: int = 1,
        steps: Optional[int] = None,
    ) -> None:
        center = str(cell or "").strip()
        if not center:
            return
        try:
            candidates = [
                str(frontier)
                for frontier in (room.get("frontier_cells", []) or [])
                if manhattan_distance(center, str(frontier)) <= max(0, int(radius))
            ]
        except Exception:
            candidates = []
        for frontier in unique_extend([center], candidates):
            self._cooldown_frontier(room, frontier, reason=reason, steps=steps)

    def _preferred_frontier(self, room: JsonDict, *, cell: str, frontiers: Sequence[str]) -> Optional[str]:
        active = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else None
        if not active:
            return None
        goal = str(active.get("cell") or "").strip()
        if not goal:
            room["active_frontier_goal"] = None
            return None
        step = self._navigation_step(room)
        cooldowns = self._active_frontier_cooldowns(room, step)
        if goal in cooldowns:
            room["active_frontier_goal"] = None
            return None
        if cell == goal:
            room["active_frontier_goal"] = None
            self._cooldown_frontier_cluster(
                room,
                goal,
                reason="frontier_reached",
                radius=1,
                steps=self.frontier_reached_cooldown_steps,
            )
            self._append_frontier_history(room, "frontier_reached", cell=goal)
            return None
        try:
            started_step = int(active.get("started_step", step) or step)
        except (TypeError, ValueError):
            started_step = step
        if step - started_step > self.frontier_sticky_max_steps:
            self._cooldown_frontier(room, goal, reason="frontier_sticky_timeout")
            return None
        return goal

    def _remember_frontier_goal(self, room: JsonDict, plan: JsonDict, *, current_cell: str) -> None:
        selected = str(plan.get("selected_goal_cell") or plan.get("requested_target_cell") or "").strip()
        if not selected:
            return
        step = self._navigation_step(room)
        previous = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else {}
        same_goal = str(previous.get("cell") or "") == selected
        try:
            previous_stale = int(previous.get("stale_count", 0) or 0) if same_goal else 0
        except (TypeError, ValueError):
            previous_stale = 0
        room["active_frontier_goal"] = {
            "cell": selected,
            "started_step": int(previous.get("started_step", step) if same_goal else step),
            "last_selected_step": step,
            "started_from_cell": previous.get("started_from_cell", current_cell) if same_goal else current_cell,
            "last_distance": manhattan_distance(current_cell, selected),
            "stale_count": previous_stale,
            "path": list(plan.get("path") or []),
            "next_action": plan.get("next_action"),
            "reason": plan.get("replan_reason") or plan.get("target_reason") or "semantic_frontier",
        }
        room["last_frontier_target"] = selected

    def _update_frontier_goal_feedback(
        self,
        room: JsonDict,
        *,
        action: str,
        success: bool,
        failure_reason: Optional[str],
        recommendation: Optional[JsonDict],
    ) -> None:
        recommendation = recommendation if isinstance(recommendation, dict) else {}
        selected = str(
            recommendation.get("selected_goal_cell")
            or recommendation.get("requested_target_cell")
            or recommendation.get("target_cell")
            or ""
        ).strip()
        if selected and action in TRANSLATION_ACTIONS and not success:
            self._cooldown_frontier(
                room,
                selected,
                reason=f"translation_failed:{action}:{failure_reason or 'unknown'}",
            )

        if self._recent_lateral_oscillation(room.get("recent_navigation_actions", [])):
            room["frontier_oscillation_count"] = int(room.get("frontier_oscillation_count", 0) or 0) + 1
            self._cooldown_frontier(room, selected or room.get("last_frontier_target"), reason="lateral_oscillation")

        active = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else None
        if not active:
            return
        goal = str(active.get("cell") or "").strip()
        if not goal:
            room["active_frontier_goal"] = None
            return
        current_cell = str(room.get("last_cell") or "")
        if current_cell == goal:
            room["active_frontier_goal"] = None
            self._cooldown_frontier_cluster(
                room,
                goal,
                reason="frontier_reached",
                radius=1,
                steps=self.frontier_reached_cooldown_steps,
            )
            self._append_frontier_history(room, "frontier_reached", cell=goal)
            return
        if not current_cell:
            return
        distance = manhattan_distance(current_cell, goal)
        try:
            previous_distance = int(active.get("last_distance", distance) or distance)
        except (TypeError, ValueError):
            previous_distance = distance
        stale = int(active.get("stale_count", 0) or 0)
        if distance < previous_distance:
            stale = 0
        elif success and action in TRANSLATION_ACTIONS:
            stale += 1
        active["last_distance"] = distance
        active["stale_count"] = stale
        active["last_feedback_step"] = self._navigation_step(room)
        room["active_frontier_goal"] = active
        if stale >= self.frontier_progress_patience:
            self._cooldown_frontier(room, goal, reason=f"frontier_no_progress:{stale}")

    def _stable_turn(
        self,
        preferred: str,
        recent_actions: Sequence[str],
        fallback: str = "RotateLeft",
    ) -> str:
        if preferred not in ROTATE_ACTIONS:
            return preferred
        bases = [self._base_action(action) for action in recent_actions[-4:]]
        last = bases[-1] if bases else None
        previous = bases[-2] if len(bases) >= 2 else None
        opposite = "RotateRight" if preferred == "RotateLeft" else "RotateLeft"
        if last == opposite and previous == preferred:
            return str(last)
        return preferred if preferred in ROTATE_ACTIONS else fallback

    def _recommendation_from_plan(
        self,
        *,
        plan: JsonDict,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
        source: str,
        target_track_id: Optional[str] = None,
        goal_type: Optional[str] = None,
        target_reason: Optional[str] = None,
    ) -> NavigationRecommendation:
        action = str(plan.get("next_action") or "")
        next_cell = plan.get("next_cell")
        reason_base = str(target_reason or plan.get("target_reason") or source)
        if str(plan.get("status")) != "success" or action not in MOVE_ACTIONS:
            return NavigationRecommendation(
                self._stable_turn("RotateLeft", recent_actions),
                f"{reason_base}_astar_no_path_scan",
                str(plan.get("requested_target_cell") or plan.get("selected_goal_cell") or "") or None,
                source=source,
                target_track_id=target_track_id,
                goal_type=goal_type,
                target_reason=reason_base,
                planner="astar",
                plan_status=str(plan.get("status") or "no_path"),
                path=list(plan.get("path") or []),
                next_cell=str(next_cell) if next_cell else None,
                route_cost=plan.get("route_cost"),
                requested_target_cell=plan.get("requested_target_cell"),
                selected_goal_cell=plan.get("selected_goal_cell"),
                selected_goal_kind=plan.get("selected_goal_kind"),
                semantic_score=plan.get("semantic_score"),
                replan_reason=plan.get("replan_reason") or "astar_no_path_scan",
            )

        if action in TRANSLATION_ACTIONS:
            open_directions = set(analysis.get("open_directions", []) or [])
            obstacle_ahead = boolish(analysis.get("obstacle_ahead", False))
            forward_cell = neighbor_cell(cell, heading)
            direction_by_action = {
                "MoveAhead": "forward",
                "MoveLeft": "left",
                "MoveRight": "right",
            }
            visual_direction = direction_by_action.get(action)
            local_safety = local_costmap_motion_safety(analysis, action)
            local_blocked = bool(local_safety.known and local_safety.safe is False)
            local_safe = bool(local_safety.known and local_safety.safe is True)
            if local_safe and next_cell is not None:
                self._observe_safe_edge(
                    room,
                    cell,
                    str(next_cell),
                    action=action,
                    safety=local_safety,
                )
            edge_blocked = bool(next_cell is not None and self._edge_blocked(room, cell, str(next_cell)))
            visual_blocked = bool(
                not local_safety.known
                and
                action == "MoveAhead"
                and (
                    obstacle_ahead
                    or (visual_direction and visual_direction not in open_directions)
                )
            )
            if action == "MoveBack" and not local_safe:
                local_blocked = True
            if local_blocked or edge_blocked or visual_blocked or recent_failed_action == action:
                selected_frontier = str(plan.get("selected_goal_cell") or plan.get("requested_target_cell") or "").strip()
                if source == "semantic_frontier" and selected_frontier:
                    self._cooldown_frontier(
                        room,
                        selected_frontier,
                        reason=f"astar_first_step_vetoed_by_local_costmap:{action}",
                        steps=max(2, self.frontier_cooldown_steps // 2),
                    )
                forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
                forward_safe_by_local = bool(forward_safety.known and forward_safety.safe is True)
                forward_safe_by_visual = bool(
                    not forward_safety.known
                    and "forward" in open_directions
                    and not obstacle_ahead
                )
                forward_safe = bool(
                    (forward_safe_by_local or forward_safe_by_visual)
                    and not self._edge_blocked(room, cell, forward_cell)
                    and recent_failed_action != "MoveAhead"
                )
                if self.drive_model == DRIVE_MODEL_NONHOLONOMIC:
                    alternatives = ["RotateLeft", "RotateRight", "LookDown", "LookUp"]
                else:
                    alternatives = ["MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "MoveBack"]
                if forward_safe:
                    alternatives.insert(0, "MoveAhead")
                alternative = local_costmap_safe_alternative(
                    analysis,
                    alternatives,
                )
                fallback = alternative or self._stable_turn("RotateLeft", recent_actions)
                return NavigationRecommendation(
                    fallback,
                    f"{reason_base}_astar_local_costmap_rejected_{action.lower()}",
                    str(plan.get("requested_target_cell") or plan.get("selected_goal_cell") or "") or None,
                    source=source,
                    target_track_id=target_track_id,
                    goal_type=goal_type,
                    target_reason=reason_base,
                    planner="astar",
                    plan_status=str(plan.get("status")),
                    path=list(plan.get("path") or []),
                    next_cell=str(next_cell) if next_cell else None,
                    route_cost=plan.get("route_cost"),
                    requested_target_cell=plan.get("requested_target_cell"),
                    selected_goal_cell=plan.get("selected_goal_cell"),
                    selected_goal_kind=plan.get("selected_goal_kind"),
                    semantic_score=plan.get("semantic_score"),
                    replan_reason=f"local_costmap_rejected_astar_{action.lower()}",
                )
        elif action in ROTATE_ACTIONS and local_costmap_known_unsafe(analysis, action):
            selected_frontier = str(plan.get("selected_goal_cell") or plan.get("requested_target_cell") or "").strip()
            if source == "semantic_frontier" and selected_frontier:
                self._cooldown_frontier(
                    room,
                    selected_frontier,
                    reason=f"astar_first_step_vetoed_by_local_costmap:{action}",
                    steps=max(2, self.frontier_cooldown_steps // 2),
                )
            opposite = "RotateRight" if action == "RotateLeft" else "RotateLeft"
            fallback = (
                opposite
                if not local_costmap_known_unsafe(analysis, opposite)
                else local_costmap_safe_alternative(
                    analysis,
                    ["LookDown", "LookUp", "MoveBack"]
                    if self.drive_model == DRIVE_MODEL_NONHOLONOMIC
                    else ["MoveLeft", "MoveRight", "LookDown", "LookUp", "MoveBack"],
                )
                or "LookDown"
            )
            return NavigationRecommendation(
                fallback,
                f"{reason_base}_astar_local_costmap_rejected_{action.lower()}",
                str(plan.get("requested_target_cell") or plan.get("selected_goal_cell") or "") or None,
                source=source,
                target_track_id=target_track_id,
                goal_type=goal_type,
                target_reason=reason_base,
                planner="astar",
                plan_status=str(plan.get("status")),
                path=list(plan.get("path") or []),
                next_cell=str(next_cell) if next_cell else None,
                route_cost=plan.get("route_cost"),
                requested_target_cell=plan.get("requested_target_cell"),
                selected_goal_cell=plan.get("selected_goal_cell"),
                selected_goal_kind=plan.get("selected_goal_kind"),
                semantic_score=plan.get("semantic_score"),
                replan_reason=f"local_costmap_rejected_astar_{action.lower()}",
            )

        if len(plan.get("path") or []) <= 1:
            suffix = "goal_scan"
        elif action == "MoveAhead":
            suffix = "path_forward"
        elif action in {"MoveLeft", "MoveRight", "MoveBack"}:
            suffix = "path_translate"
        else:
            suffix = "path_turn"
        return NavigationRecommendation(
            action,
            f"{reason_base}_astar_{suffix}",
            str(plan.get("requested_target_cell") or plan.get("selected_goal_cell") or "") or None,
            source=source,
            target_track_id=target_track_id,
            goal_type=goal_type,
            target_reason=reason_base,
            planner="astar",
            plan_status=str(plan.get("status")),
            path=list(plan.get("path") or []),
            next_cell=str(next_cell) if next_cell else None,
            route_cost=plan.get("route_cost"),
            requested_target_cell=plan.get("requested_target_cell"),
            selected_goal_cell=plan.get("selected_goal_cell"),
            selected_goal_kind=plan.get("selected_goal_kind"),
            semantic_score=plan.get("semantic_score"),
            replan_reason=plan.get("replan_reason"),
        )

    def _recommend_global_frontier(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
        goal_type: Optional[str] = None,
        allow_backtrack: bool = True,
    ) -> Optional[NavigationRecommendation]:
        try:
            position_status = self.position_map.status()
            semantic_status = self.semantic_map.status()
            step = self._navigation_step(room)
            cooldowns = self._active_frontier_cooldowns(room, step)
            preferred = self._preferred_frontier(
                room,
                cell=cell,
                frontiers=room.get("frontier_cells", []),
            )
            search_frontiers = unique_extend(room.get("frontier_cells", []), [preferred] if preferred else [])
            if not allow_backtrack and self.frontier_backtrack_filter_enabled:
                local_frontiers = self._frontiers_not_behind(
                    cell=cell,
                    heading=heading,
                    frontiers=search_frontiers,
                )
                if local_frontiers:
                    search_frontiers = local_frontiers
                    if preferred and preferred not in set(local_frontiers):
                        preferred = None
            if preferred:
                sticky_plan = self.global_planner.plan_to_best_frontier(
                    current_cell=cell,
                    current_heading=heading,
                    frontier_cells=[preferred],
                    goal_type=goal_type,
                    target_reason="semantic_frontier",
                    position_status=position_status,
                    semantic_status=semantic_status,
                    blocked_edges=room.get("blocked_edges", []),
                    preferred_frontier=preferred,
                    frontier_cooldowns=cooldowns,
                    allow_backtrack=True,
                    drive_model=self.drive_model,
                )
                if str(sticky_plan.get("status")) == "success":
                    self._remember_frontier_goal(room, sticky_plan, current_cell=cell)
                    return self._recommendation_from_plan(
                        plan=sticky_plan,
                        room=room,
                        cell=cell,
                        heading=heading,
                        analysis=analysis,
                        recent_actions=recent_actions,
                        recent_failed_action=recent_failed_action,
                        source="semantic_frontier",
                        goal_type=goal_type,
                        target_reason="semantic_frontier",
                    )
                self._cooldown_frontier(room, preferred, reason="sticky_frontier_no_reachable_path")
                cooldowns = self._active_frontier_cooldowns(room, step)
                preferred = None
            plan = self.global_planner.plan_to_best_frontier(
                current_cell=cell,
                current_heading=heading,
                frontier_cells=search_frontiers,
                goal_type=goal_type,
                target_reason="semantic_frontier",
                position_status=position_status,
                semantic_status=semantic_status,
                blocked_edges=room.get("blocked_edges", []),
                preferred_frontier=preferred,
                frontier_cooldowns=cooldowns,
                allow_backtrack=allow_backtrack,
                drive_model=self.drive_model,
            )
        except Exception:
            return None
        if str(plan.get("status")) != "success":
            return None
        self._remember_frontier_goal(room, plan, current_cell=cell)
        return self._recommendation_from_plan(
            plan=plan,
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=recent_actions,
            recent_failed_action=recent_failed_action,
            source="semantic_frontier",
            goal_type=goal_type,
            target_reason="semantic_frontier",
        )

    def _recommend_object_target(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
        target_cell: str,
        target_reason: Optional[str],
        target_track_id: Optional[str],
        goal_type: Optional[str],
        target_heading: Optional[str],
    ) -> NavigationRecommendation:
        base_reason = str(target_reason or "object_memory_target")
        forward_cell = neighbor_cell(cell, heading)
        if str(target_cell) == forward_cell:
            forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
            if forward_safety.known and forward_safety.safe is True:
                self._observe_safe_edge(
                    room,
                    cell,
                    forward_cell,
                    action="MoveAhead",
                    safety=forward_safety,
                )
            if forward_safety.known and forward_safety.safe is False:
                alternatives = (
                    ["RotateLeft", "RotateRight", "LookDown", "LookUp"]
                    if self.drive_model == DRIVE_MODEL_NONHOLONOMIC
                    else ["MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookDown", "LookUp"]
                )
                fallback = local_costmap_safe_alternative(
                    analysis,
                    alternatives,
                )
                if fallback:
                    return NavigationRecommendation(
                        fallback,
                        f"{base_reason}_astar_local_costmap_rejected_moveahead",
                        str(target_cell),
                        source="object_memory",
                        target_track_id=target_track_id,
                        goal_type=goal_type,
                        target_reason=base_reason,
                        planner="astar",
                        plan_status="local_costmap_veto",
                        path=[cell, str(target_cell)],
                        next_cell=str(target_cell),
                        requested_target_cell=str(target_cell),
                        selected_goal_cell=str(target_cell),
                        selected_goal_kind="requested_goal",
                        replan_reason="local_costmap_rejected_astar_moveahead",
                    )
        try:
            plan = self.global_planner.plan_to_goal(
                current_cell=cell,
                current_heading=heading,
                target_cell=str(target_cell),
                target_heading=target_heading,
                target_reason=base_reason,
                target_track_id=target_track_id,
                goal_type=goal_type,
                position_status=self.position_map.status(),
                semantic_status=self.semantic_map.status(),
                blocked_edges=room.get("blocked_edges", []),
                drive_model=self.drive_model,
            )
        except Exception as exc:
            plan = {
                "status": "error",
                "planner": "astar",
                "requested_target_cell": str(target_cell),
                "replan_reason": f"astar_exception:{exc}",
            }
        return self._recommendation_from_plan(
            plan=plan,
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=recent_actions,
            recent_failed_action=recent_failed_action,
            source="object_memory",
            target_track_id=target_track_id,
            goal_type=goal_type,
            target_reason=base_reason,
        )

    def _recommend_action(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
        target_cell: Optional[str] = None,
        target_reason: Optional[str] = None,
        target_track_id: Optional[str] = None,
        goal_type: Optional[str] = None,
        target_heading: Optional[str] = None,
    ) -> NavigationRecommendation:
        open_directions = set(analysis.get("open_directions", []) or [])
        obstacle_ahead = boolish(analysis.get("obstacle_ahead", False))
        recent_sequence = list(room.get("recent_navigation_actions", [])) + list(recent_actions)
        backtrack_count = int(room.get("backtrack_count", 0) or 0)
        stagnation = int(room.get("stagnation_count", 0) or 0)
        forward_cell = neighbor_cell(cell, heading)
        left_cell = neighbor_cell(cell, left_heading(heading))
        right_cell = neighbor_cell(cell, right_heading(heading))
        forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
        forward_safe_by_local = bool(forward_safety.known and forward_safety.safe is True)
        forward_safe_by_visual = bool(
            not forward_safety.known
            and "forward" in open_directions
            and not obstacle_ahead
        )
        if forward_safe_by_local:
            self._observe_safe_edge(
                room,
                cell,
                forward_cell,
                action="MoveAhead",
                safety=forward_safety,
            )
        forward_edge_blocked = self._edge_blocked(room, cell, forward_cell)
        forward_safe = (
            (forward_safe_by_local or forward_safe_by_visual)
            and not forward_edge_blocked
            and recent_failed_action != "MoveAhead"
        )

        # Goal-directed navigation always uses bounded A*. The old direct
        # Manhattan-heading heuristic was intentionally removed.
        if target_cell:
            return self._recommend_object_target(
                room=room,
                cell=cell,
                heading=heading,
                analysis=analysis,
                recent_actions=recent_sequence,
                recent_failed_action=recent_failed_action,
                target_cell=str(target_cell),
                target_reason=target_reason,
                target_track_id=target_track_id,
                goal_type=goal_type,
                target_heading=target_heading,
            )

        visited_cells = {str(item) for item in (room.get("visited_cells", []) or [])}
        visited_counts = room.get("visited_cell_counts") if isinstance(room.get("visited_cell_counts"), dict) else {}
        try:
            forward_visit_count = int(visited_counts.get(forward_cell, 0) or 0)
        except (TypeError, ValueError):
            forward_visit_count = 0
        frontier_ahead_score = self._frontier_ahead_score(
            cell=cell,
            heading=heading,
            frontiers=room.get("frontier_cells", []),
            max_distance=self.local_forward_progress_frontier_distance,
        )
        recent_failed_base = self._base_action(recent_failed_action)
        if (
            self.local_forward_progress_enabled
            and forward_safe
            and not self._recent_moveback_loop(recent_sequence)
        ):
            if recent_failed_base in {"MoveLeft", "MoveRight"}:
                return NavigationRecommendation(
                    "MoveAhead",
                    "local_forward_progress_after_lateral_failure",
                    forward_cell,
                    source="local_bootstrap",
                )
            if forward_cell not in visited_cells:
                return NavigationRecommendation(
                    "MoveAhead",
                    "local_forward_progress_unvisited_safe_cell",
                    forward_cell,
                    source="local_bootstrap",
                )
            if frontier_ahead_score > 0 and forward_visit_count <= 1:
                return NavigationRecommendation(
                    "MoveAhead",
                    f"local_forward_progress_toward_frontier_cluster:{frontier_ahead_score}",
                    forward_cell,
                    source="local_bootstrap",
                )

        turn_streak = int(room.get("turn_streak_count", 0) or 0)
        if (
            forward_safe_by_local
            and not self._edge_hard_blocked(room, cell, forward_cell)
            and recent_failed_action != "MoveAhead"
            and turn_streak >= self.recovery_scan_escape_turns
            and not self._recent_moveback_loop(recent_sequence)
        ):
            self._add_known_open_edge(
                room,
                cell,
                forward_cell,
                override_blocked=True,
                override_hard=False,
            )
            if not self._edge_blocked(room, cell, forward_cell):
                return NavigationRecommendation(
                    "MoveAhead",
                    f"recovery_scan_escape_forward_clear:turn_streak={turn_streak};"
                    f"confidence={forward_safety.confidence:.2f};observed={forward_safety.observed_ratio:.2f}",
                    forward_cell,
                    source="local_bootstrap",
                )

        pose_trust = room.get("pose_trust") if isinstance(room.get("pose_trust"), dict) else {}
        try:
            uncertainty_cells = float(room.get("position_uncertainty_cells", 0.0) or 0.0)
        except (TypeError, ValueError):
            uncertainty_cells = 0.0
        pose_uncertain = bool(
            self.forward_bias_when_pose_uncertain
            and (
                bool(pose_trust.get("relocalization_recommended"))
                or uncertainty_cells >= self.forward_bias_uncertainty_cells
            )
        )
        if forward_safe and pose_uncertain and not self._recent_moveback_loop(recent_sequence):
            return NavigationRecommendation(
                "MoveAhead",
                "local_forward_bias_pose_uncertain_before_global_frontier",
                forward_cell,
                source="local_bootstrap",
            )
        last_recent_action = self._base_action(recent_sequence[-1]) if recent_sequence else None
        if (
            forward_safe
            and self.forward_bias_after_place
            and last_recent_action == "place-object"
            and not self._recent_moveback_loop(recent_sequence)
        ):
            return NavigationRecommendation(
                "MoveAhead",
                "local_forward_bias_after_place_before_global_frontier",
                forward_cell,
                source="local_bootstrap",
            )

        forward_streak = self._recent_action_streak(recent_sequence, "MoveAhead")
        forward_progress_count = self._recent_forward_progress_count(recent_sequence)
        active_frontier = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else None
        if active_frontier and str(active_frontier.get("cell") or "").strip():
            sticky_frontier = self._recommend_global_frontier(
                room=room,
                cell=cell,
                heading=heading,
                analysis=analysis,
                recent_actions=recent_sequence,
                recent_failed_action=recent_failed_action,
                goal_type=goal_type,
                allow_backtrack=True,
            )
            if sticky_frontier is not None:
                return sticky_frontier

        corner_bypass = self._local_front_corner_bypass(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_failed_action=recent_failed_action,
        )
        if corner_bypass is not None:
            return corner_bypass

        side_probe = self._local_side_probe_after_forward_block(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=recent_sequence,
            recent_failed_action=recent_failed_action,
        )
        if side_probe is not None:
            return side_probe

        pose_recovery_needed = self._pose_requires_relocalization(room)
        translation_recovery_needed = self._recent_translation_failure(recent_sequence, recent_failed_action)
        forward_blocked_for_recovery = bool(
            (forward_safety.known and forward_safety.safe is False)
            or self._edge_blocked(room, cell, forward_cell)
            or (obstacle_ahead and not forward_safe_by_local)
        )
        if translation_recovery_needed or (
            pose_recovery_needed
            and forward_blocked_for_recovery
            and turn_streak < self.collision_recovery_turn_steps
        ):
            return self._local_relocalization_recovery(
                room=room,
                cell=cell,
                heading=heading,
                analysis=analysis,
                recent_actions=recent_sequence,
                reason="pose_relocalization_required" if pose_recovery_needed else "translation_failure_recovery",
            )

        side_explore = self._local_side_exploration_probe(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=recent_sequence,
            recent_failed_action=recent_failed_action,
        )
        if side_explore is not None:
            return side_explore

        allow_frontier_backtrack = True
        if (
            self.frontier_backtrack_guard_enabled
            and (
                forward_streak >= self.frontier_backtrack_guard_forward_streak
                or forward_progress_count >= self.frontier_backtrack_guard_forward_streak
            )
            and stagnation < self.frontier_progress_patience
            and not self._recent_moveback_loop(recent_sequence)
        ):
            allow_frontier_backtrack = False

        # Exploration also prefers an A* route to a semantic frontier.
        frontier = self._recommend_global_frontier(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=recent_sequence,
            recent_failed_action=recent_failed_action,
            goal_type=goal_type,
            allow_backtrack=allow_frontier_backtrack,
        )
        if frontier is not None:
            return frontier

        # Bootstrap fallback only: when the map has no reachable frontier yet,
        # use current online-safe visual openness to gather more map evidence.
        side_options: List[Tuple[str, str, str]] = []
        left_safety = local_costmap_motion_safety(analysis, "MoveLeft")
        right_safety = local_costmap_motion_safety(analysis, "MoveRight")
        if self.drive_model == DRIVE_MODEL_NONHOLONOMIC:
            if (
                not (left_safety.known and left_safety.safe is False)
                and not local_costmap_known_unsafe(analysis, "RotateLeft")
                and ("left" in open_directions or left_safety.known)
            ):
                side_options.append(("RotateLeft", left_cell, "left"))
            if (
                not (right_safety.known and right_safety.safe is False)
                and not local_costmap_known_unsafe(analysis, "RotateRight")
                and ("right" in open_directions or right_safety.known)
            ):
                side_options.append(("RotateRight", right_cell, "right"))
        elif (left_safety.known and left_safety.safe is True) or (not left_safety.known and "left" in open_directions):
            side_options.append(("MoveLeft", left_cell, "left"))
        elif left_safety.known and left_safety.safe is False and not local_costmap_known_unsafe(analysis, "RotateLeft"):
            side_options.append(("RotateLeft", left_cell, "left"))
        if self.drive_model != DRIVE_MODEL_NONHOLONOMIC and ((right_safety.known and right_safety.safe is True) or (not right_safety.known and "right" in open_directions)):
            side_options.append(("MoveRight", right_cell, "right"))
        elif self.drive_model != DRIVE_MODEL_NONHOLONOMIC and right_safety.known and right_safety.safe is False and not local_costmap_known_unsafe(analysis, "RotateRight"):
            side_options.append(("RotateRight", right_cell, "right"))
        if forward_safe and not self._recent_moveback_loop(recent_sequence):
            return NavigationRecommendation("MoveAhead", "local_observation_bootstrap_forward", forward_cell, source="local_bootstrap")
        if side_options:
            preferred, target, side = side_options[0]
            action = self._stable_turn(preferred, recent_sequence) if preferred in ROTATE_ACTIONS else preferred
            return NavigationRecommendation(action, f"local_observation_bootstrap_scan_{side}", target, source="local_bootstrap")
        if obstacle_ahead or self._edge_blocked(room, cell, forward_cell):
            back_rec = local_costmap_action_record(analysis, "MoveBack")
            back_safe = bool(
                back_rec
                and back_rec.get("safe") is True
                and float(back_rec.get("observed_ratio", 0.0) or 0.0) >= 0.60
            )
            if back_safe and not self._recent_moveback_loop(recent_sequence) and backtrack_count == 0 and stagnation < 4:
                return NavigationRecommendation("MoveBack", "local_observation_bootstrap_backoff", None, source="local_bootstrap")
        return NavigationRecommendation(self._stable_turn("RotateLeft", recent_sequence), "local_observation_bootstrap_rotate", left_cell, source="local_bootstrap")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage robot-cleaner navigation memory.")
    parser.add_argument("--memory-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show", help="Show navigation memory status.")
    sub.add_parser("status", help="Alias for show.")

    reset = sub.add_parser("reset", help="Reset navigation memory fields in room-state.json.")
    reset.add_argument("--room", default="current_room")

    observe = sub.add_parser("observe", help="Record the current observed pose/cell.")
    observe.add_argument("--vision-json", required=True)
    observe.add_argument("--analysis-json", default="{}")

    recommend = sub.add_parser("recommend", help="Return a navigation-memory exploration recommendation.")
    recommend.add_argument("--vision-json", required=True)
    recommend.add_argument("--analysis-json", required=True)
    recommend.add_argument("--recent-actions-json", default="[]")
    recommend.add_argument("--recent-failed-action", default=None)

    record = sub.add_parser("record-step", help="Record one navigation/action result.")
    record.add_argument("--action", required=True)
    record.add_argument("--success", action="store_true")
    record.add_argument("--failed", action="store_true")
    record.add_argument("--failure-reason", default=None)
    record.add_argument("--vision-json", required=True)
    record.add_argument("--analysis-json", default="{}")
    record.add_argument("--action-result-json", default="{}")

    return parser


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    memory_dir = Path(args.memory_dir) if args.memory_dir else None
    manager = NavigationMemory(memory_dir)

    try:
        if args.command in {"show", "status"}:
            result = manager.status()
        elif args.command == "reset":
            result = manager.reset(room_name=args.room)
        elif args.command == "observe":
            result = manager.observe(
                vision=parse_json_arg(args.vision_json),
                analysis=parse_json_arg(args.analysis_json),
            )
        elif args.command == "recommend":
            recent_actions = json.loads(args.recent_actions_json)
            if not isinstance(recent_actions, list):
                raise ValueError("--recent-actions-json must be a JSON list")
            result = manager.recommend(
                vision=parse_json_arg(args.vision_json),
                analysis=parse_json_arg(args.analysis_json),
                recent_actions=[str(item) for item in recent_actions],
                recent_failed_action=args.recent_failed_action,
            )
        elif args.command == "record-step":
            if args.success and args.failed:
                raise ValueError("--success and --failed are mutually exclusive")
            success = not bool(args.failed)
            result = manager.record_step(
                action=args.action,
                success=success,
                failure_reason=args.failure_reason,
                vision=parse_json_arg(args.vision_json),
                analysis=parse_json_arg(args.analysis_json),
                action_result=parse_json_arg(args.action_result_json),
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print_json({"status": "error", "result_type": "navigation_memory_error", "message": str(exc)})
        return 1

    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
