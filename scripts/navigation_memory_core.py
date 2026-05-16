#!/usr/bin/env python3
"""
Lightweight navigation memory for robot-cleaner.

This module provides a small frontier-like memory layer for the AI2-THOR
single-room patrol runner. It is not a full SLAM system. It records coarse
grid cells, blocked forward edges, collision/oscillation counters, and a
conservative exploration recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
ROOM_STATE_PATH = MEMORY_DIR / "room-state.json"

DEFAULT_CELL_SIZE = 0.25
# FloorPlan1's reachable kitchen area is larger than the first debug strip.
# Keep this conservative so "coverage=1.00" means broad coverage, not merely
# enough steps in a corridor-like local area.
DEFAULT_TARGET_CELLS = 120
MOVE_ACTIONS = {"MoveAhead", "MoveBack", "RotateLeft", "RotateRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
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
        for attempt in range(8):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt >= 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


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
    return before_cell, before_heading, after_cell, after_heading


def boolish(value: Any) -> bool:
    return bool(value)


@dataclass
class NavigationRecommendation:
    action: str
    reason: str
    target_cell: Optional[str]
    source: str = "navigation_memory"

    def as_dict(self) -> JsonDict:
        return {
            "action": self.action,
            "reason": self.reason,
            "target_cell": self.target_cell,
            "source": self.source,
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

    def load_room(self) -> JsonDict:
        room = load_json(self.room_path, default_room_state())
        room = self.normalize_room(room)
        room["frontier_cells"] = self._global_frontier_cells(room)
        room["known_frontier_cells"] = list(room["frontier_cells"])
        self._update_coverage(room)
        return room

    def save_room(self, room: JsonDict) -> None:
        atomic_write_json(self.room_path, self.normalize_room(room))

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
        base["blocked_edges"] = unique_extend([], base.get("blocked_edges", []))
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
        cell, heading = pose_from_vision(
            vision,
            fallback_cell=str(room.get("last_cell") or "0,0"),
            fallback_heading=str(room.get("last_heading") or "north"),
        )
        self._ensure_cell(room, cell)
        room["last_cell"] = cell
        room["last_heading"] = heading
        self._record_observed_open_edges(room, cell, heading, analysis or {})
        room["frontier_cells"] = self._global_frontier_cells(room)
        room["known_frontier_cells"] = list(room["frontier_cells"])
        self._update_coverage(room)
        room["navigation_last_update"] = now_iso()
        if persist:
            self.save_room(room)
            return self.status()
        return self._status_from_room(room)

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
        if success and action in {"MoveAhead", "MoveBack"}:
            new_cell = self._visit_cell(room, after_cell)
            self._add_known_open_edge(room, before_cell, after_cell)
        elif success:
            new_cell = self._ensure_cell(room, after_cell)
        room["last_cell"] = after_cell if success else before_cell
        room["last_heading"] = after_heading if success else before_heading

        if new_cell:
            room["stagnation_count"] = 0
            room["last_new_cell"] = after_cell
        else:
            room["stagnation_count"] = int(room.get("stagnation_count", 0)) + 1

        if action == "MoveBack":
            room["backtrack_count"] = int(room.get("backtrack_count", 0)) + 1
        elif new_cell and action == "MoveAhead":
            room["backtrack_count"] = 0
        else:
            room["backtrack_count"] = max(0, int(room.get("backtrack_count", 0)) - 1)

        if action in ROTATE_ACTIONS and success:
            room["turn_streak_count"] = int(room.get("turn_streak_count", 0)) + 1
        else:
            room["turn_streak_count"] = 0

        if action in {"MoveAhead", "MoveBack"} and not success:
            target_heading = before_heading
            if action == "MoveBack":
                target_heading = opposite_heading(before_heading)
            target = neighbor_cell(before_cell, target_heading)
            self._mark_blocked_edge(room, before_cell, target)
            room["collision_count"] = int(room.get("collision_count", 0)) + 1

        if self._is_rotation_oscillation(room.get("recent_navigation_actions", []), action):
            room["oscillation_count"] = int(room.get("oscillation_count", 0)) + 1

        recent = list(room.get("recent_navigation_actions", []))
        recent.append(action if success else f"{action}:failed:{failure_reason or 'unknown'}")
        room["recent_navigation_actions"] = recent[-16:]
        room["frontier_cells"] = self._global_frontier_cells(room)
        room["known_frontier_cells"] = list(room["frontier_cells"])
        if isinstance(recommendation, dict) and recommendation:
            room["last_navigation_decision"] = dict(recommendation)
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
        persist: bool = True,
    ) -> JsonDict:
        room = self.load_room()
        cell, heading = pose_from_vision(
            vision,
            fallback_cell=str(room.get("last_cell") or "0,0"),
            fallback_heading=str(room.get("last_heading") or "north"),
        )
        self._ensure_cell(room, cell)
        room["last_cell"] = cell
        room["last_heading"] = heading
        self._record_observed_open_edges(room, cell, heading, analysis)
        room["frontier_cells"] = self._global_frontier_cells(room)
        room["known_frontier_cells"] = list(room["frontier_cells"])

        recommendation = self._recommend_action(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=list(recent_actions or []),
            recent_failed_action=recent_failed_action,
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
            "known_open_edges": room.get("known_open_edges", []),
            "known_frontier_cells": room.get("known_frontier_cells", []),
            "blocked_edges": room.get("blocked_edges", []),
            "coverage_estimate": room.get("coverage_estimate", 0.0),
            "collision_count": room.get("collision_count", 0),
            "oscillation_count": room.get("oscillation_count", 0),
            "stagnation_count": room.get("stagnation_count", 0),
            "backtrack_count": room.get("backtrack_count", 0),
            "turn_streak_count": room.get("turn_streak_count", 0),
            "last_new_cell": room.get("last_new_cell"),
            "recent_navigation_actions": room.get("recent_navigation_actions", []),
            "last_navigation_decision": room.get("last_navigation_decision", {}),
            "last_frontier_target": room.get("last_frontier_target"),
            "navigation_last_update": room.get("navigation_last_update"),
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
        if "forward" in open_directions and not boolish(analysis.get("obstacle_ahead", False)):
            target = neighbor_cell(cell, heading)
            self._add_known_open_edge(room, cell, target)

    def _add_known_open_edge(self, room: JsonDict, source: str, target: str) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        if self._edge_blocked(room, source, target):
            return
        room["known_open_edges"] = unique_extend(room.get("known_open_edges", []), [direct, reverse])

    def _mark_blocked_edge(self, room: JsonDict, source: str, target: str) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        room["blocked_edges"] = unique_extend(room.get("blocked_edges", []), [direct, reverse])
        known = [
            edge
            for edge in room.get("known_open_edges", [])
            if edge not in {direct, reverse}
        ]
        room["known_open_edges"] = known

    def _compute_local_frontier_cells(self, room: JsonDict, cell: str, heading: str, analysis: JsonDict) -> List[str]:
        open_directions = set(analysis.get("open_directions", []) or [])
        visited = set(room.get("visited_cells", []))
        frontiers: List[str] = []
        for direction, candidate_heading in self._heading_by_direction(heading).items():
            if direction not in open_directions:
                continue
            target = neighbor_cell(cell, candidate_heading)
            if self._edge_blocked(room, cell, target):
                continue
            if target not in visited:
                frontiers.append(target)
        return unique_extend([], frontiers)

    def _global_frontier_cells(self, room: JsonDict) -> List[str]:
        visited = set(room.get("visited_cells", []))
        frontiers: List[str] = []
        for edge in room.get("known_open_edges", []) or []:
            parsed = parse_edge(str(edge))
            if not parsed:
                continue
            source, target = parsed
            if source not in visited:
                continue
            if target in visited:
                continue
            if self._edge_blocked(room, source, target):
                continue
            frontiers.append(target)
        return unique_extend([], frontiers)

    def _graph_neighbors(self, room: JsonDict, cell: str) -> List[str]:
        neighbors: List[str] = []
        for edge in room.get("known_open_edges", []) or []:
            parsed = parse_edge(str(edge))
            if not parsed:
                continue
            source, target = parsed
            if source != cell:
                continue
            if self._edge_blocked(room, source, target):
                continue
            neighbors.append(target)
        return unique_extend([], neighbors)

    def _shortest_path_to_frontier(self, room: JsonDict, start: str) -> Optional[List[str]]:
        frontier_cells = self._global_frontier_cells(room)
        if not frontier_cells:
            return None
        frontier_set = set(frontier_cells)
        visited_cells = set(room.get("visited_cells", []))
        queue = deque([[start]])
        seen = {start}

        best_path: Optional[List[str]] = None
        while queue:
            path = queue.popleft()
            current = path[-1]
            for target in self._graph_neighbors(room, current):
                if target in seen:
                    continue
                next_path = path + [target]
                if target in frontier_set:
                    if best_path is None:
                        best_path = next_path
                    elif len(next_path) < len(best_path):
                        best_path = next_path
                    elif len(next_path) == len(best_path):
                        current_best = best_path[-1]
                        if manhattan_distance(start, target) < manhattan_distance(start, current_best):
                            best_path = next_path
                    continue
                if target in visited_cells:
                    seen.add(target)
                    queue.append(next_path)
        return best_path

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
        for known in ["MoveAhead", "MoveBack", "RotateLeft", "RotateRight", "clean-garbage"]:
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

    def _recommend_global_frontier(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
    ) -> Optional[NavigationRecommendation]:
        open_directions = set(analysis.get("open_directions", []) or [])
        obstacle_ahead = boolish(analysis.get("obstacle_ahead", False))
        visited = set(room.get("visited_cells", []))
        turn_streak = int(room.get("turn_streak_count", 0) or 0)

        for _ in range(8):
            path = self._shortest_path_to_frontier(room, cell)
            if not path or len(path) < 2:
                return None

            next_cell = path[1]
            desired_heading = heading_between(cell, next_cell)
            if not desired_heading:
                return None

            action = self._turn_action_toward(heading, desired_heading, recent_actions)
            target_frontier = path[-1]

            if (
                action in ROTATE_ACTIONS
                and next_cell not in visited
                and room.get("last_frontier_target") == target_frontier
                and turn_streak >= 8
            ):
                self._mark_blocked_edge(room, cell, next_cell)
                room["frontier_cells"] = self._global_frontier_cells(room)
                room["known_frontier_cells"] = list(room["frontier_cells"])
                continue

            room["last_frontier_target"] = target_frontier
            if action == "MoveAhead":
                if self._edge_blocked(room, cell, next_cell):
                    continue
                if recent_failed_action == "MoveAhead":
                    return None
                if "forward" not in open_directions or obstacle_ahead:
                    self._mark_blocked_edge(room, cell, next_cell)
                    room["frontier_cells"] = self._global_frontier_cells(room)
                    room["known_frontier_cells"] = list(room["frontier_cells"])
                    continue
                return NavigationRecommendation(
                    "MoveAhead",
                    "global_frontier_path_forward",
                    target_frontier,
                )

            return NavigationRecommendation(
                action,
                "global_frontier_path_turn",
                target_frontier,
            )

        return None

    def _recommend_action(
        self,
        *,
        room: JsonDict,
        cell: str,
        heading: str,
        analysis: JsonDict,
        recent_actions: Sequence[str],
        recent_failed_action: Optional[str],
    ) -> NavigationRecommendation:
        open_directions = set(analysis.get("open_directions", []) or [])
        obstacle_ahead = boolish(analysis.get("obstacle_ahead", False))
        visited = set(room.get("visited_cells", []))
        counts = room.get("visited_cell_counts", {})
        recent_sequence = list(room.get("recent_navigation_actions", [])) + list(recent_actions)
        stagnation = int(room.get("stagnation_count", 0) or 0)
        backtrack_count = int(room.get("backtrack_count", 0) or 0)
        turn_streak = int(room.get("turn_streak_count", 0) or 0)

        forward_cell = neighbor_cell(cell, heading)
        left_cell = neighbor_cell(cell, left_heading(heading))
        right_cell = neighbor_cell(cell, right_heading(heading))

        forward_blocked = self._edge_blocked(room, cell, forward_cell)
        forward_safe = (
            "forward" in open_directions
            and not obstacle_ahead
            and not forward_blocked
            and recent_failed_action != "MoveAhead"
        )

        side_options: List[Tuple[str, str, str]] = []
        if "left" in open_directions:
            side_options.append(("RotateLeft", left_cell, "left"))
        if "right" in open_directions:
            side_options.append(("RotateRight", right_cell, "right"))

        if forward_safe and forward_cell not in visited:
            return NavigationRecommendation("MoveAhead", "frontier_forward_unvisited", forward_cell)

        for action, target, side in side_options:
            if (
                turn_streak < 8
                and target not in visited
                and not self._edge_blocked(room, cell, target)
            ):
                stable_action = self._stable_turn(action, recent_sequence)
                return NavigationRecommendation(stable_action, f"frontier_{side}_unvisited", target)

        global_frontier = self._recommend_global_frontier(
            room=room,
            cell=cell,
            heading=heading,
            analysis=analysis,
            recent_actions=recent_sequence,
            recent_failed_action=recent_failed_action,
        )
        if global_frontier is not None:
            return global_frontier

        if (
            side_options
            and turn_streak < 8
            and (stagnation >= 4 or int(counts.get(forward_cell, 0) or 0) >= 3)
        ):
            action, target, side = min(
                side_options,
                key=lambda item: int(counts.get(item[1], 0) or 0),
            )
            stable_action = self._stable_turn(action, recent_sequence)
            return NavigationRecommendation(stable_action, f"stagnation_side_scan_{side}", target)

        if forward_safe and not self._recent_moveback_loop(recent_sequence):
            return NavigationRecommendation("MoveAhead", "forward_open_revisit_allowed", forward_cell)

        if side_options:
            action, target, side = min(
                side_options,
                key=lambda item: int(counts.get(item[1], 0) or 0),
            )
            stable_action = self._stable_turn(action, recent_sequence)
            return NavigationRecommendation(stable_action, f"side_scan_low_visit_{side}", target)

        if obstacle_ahead or forward_blocked:
            if (
                not self._recent_moveback_loop(recent_sequence)
                and backtrack_count == 0
                and stagnation < 4
            ):
                return NavigationRecommendation("MoveBack", "blocked_front_conservative_backoff", None)
            return NavigationRecommendation(
                self._stable_turn("RotateLeft", recent_sequence),
                "blocked_front_scan_no_backtrack",
                left_cell,
            )

        return NavigationRecommendation(
            self._stable_turn("RotateLeft", recent_sequence),
            "no_frontier_conservative_scan",
            left_cell,
        )


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
