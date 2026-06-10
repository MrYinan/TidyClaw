#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Map Core for OpenClaw robot-cleaner.

This module upgrades the old lightweight navigation memory into an
Action-Odometry Occupancy Grid:

1. Action odometry:
   MoveAhead / MoveBack / MoveLeft / MoveRight update a coarse grid pose;
   RotateLeft / RotateRight update heading; LookUp / LookDown preserve map pose.

2. Position map:
   Each cell stores unknown/free/occupied/inflated_occupied, visit count,
   collision count, evidence, and timestamps.

3. Pose uncertainty:
   The map tracks pose_confidence and position_uncertainty_cells so object
   memory can know whether projected object cells are reliable.

4. Occupancy update:
   Action feedback and optional RGB-D/depth observations update local free /
   occupied cells.

5. Frontier extraction:
   Frontiers are free cells adjacent to unknown cells, not merely unvisited
   neighbors.

This file is intentionally online-safe. It does not read AI2-THOR raw metadata.
It consumes only sanitized vision/action feedback already available to the
runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
POSITION_MAP_PATH = MEMORY_DIR / "position-map.json"
ROOM_STATE_PATH = MEMORY_DIR / "room-state.json"

POSITION_MAP_SCHEMA_VERSION = 1

DEFAULT_CELL_SIZE_M = 0.25
DEFAULT_TARGET_CELLS = 120

CELL_UNKNOWN = "unknown"
CELL_FREE = "free"
CELL_OCCUPIED = "occupied"
CELL_INFLATED = "inflated_occupied"

HEADING_ORDER = ["north", "east", "south", "west"]
HEADING_VECTORS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
HEADING_THETA_DEG = {
    "north": 0.0,
    "east": 90.0,
    "south": 180.0,
    "west": 270.0,
}

MOVE_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}


JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return dict(default)
    if not text.strip():
        return dict(default)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(data, dict):
        return dict(default)
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
        last_error: Optional[PermissionError] = None
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
            except PermissionError:
                pass


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def unique_extend(existing: Iterable[str], incoming: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in list(existing or []) + list(incoming or []):
        if item is None:
            continue
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_cell(cell: Any) -> Tuple[int, int]:
    left, right = str(cell or "0,0").split(",", 1)
    return int(left), int(right)


def format_cell(x: int, z: int) -> str:
    return f"{int(x)},{int(z)}"


def edge_key(source: str, target: str) -> str:
    return f"{source}->{target}"


def edge_pair_key(source: str, target: str) -> str:
    left = str(source).strip()
    right = str(target).strip()
    if left <= right:
        return f"{left}|{right}"
    return f"{right}|{left}"


def parse_edge(edge: Any) -> Optional[Tuple[str, str]]:
    value = str(edge or "")
    if "->" not in value:
        return None
    source, target = value.split("->", 1)
    source = source.strip()
    target = target.strip()
    if not source or not target:
        return None
    return source, target


def heading_from_rotation(rotation_y: float) -> str:
    value = float(rotation_y or 0.0) % 360.0
    index = int(round(value / 90.0)) % 4
    return HEADING_ORDER[index]


def left_heading(heading: str) -> str:
    if heading not in HEADING_ORDER:
        heading = "north"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) - 1) % 4]


def right_heading(heading: str) -> str:
    if heading not in HEADING_ORDER:
        heading = "north"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) + 1) % 4]


def opposite_heading(heading: str) -> str:
    if heading not in HEADING_ORDER:
        heading = "north"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) + 2) % 4]


def heading_between(source: str, target: str) -> Optional[str]:
    sx, sz = parse_cell(source)
    tx, tz = parse_cell(target)
    dx = tx - sx
    dz = tz - sz
    for heading, vector in HEADING_VECTORS.items():
        if vector == (dx, dz):
            return heading
    return None


def neighbor_cell(cell: str, heading: str) -> str:
    x, z = parse_cell(cell)
    dx, dz = HEADING_VECTORS.get(heading, (0, 1))
    return format_cell(x + dx, z + dz)


def neighbor_for_action(cell: str, heading: str, action: str) -> str:
    if action == "MoveAhead":
        return neighbor_cell(cell, heading)
    if action == "MoveBack":
        return neighbor_cell(cell, opposite_heading(heading))
    if action == "MoveLeft":
        return neighbor_cell(cell, left_heading(heading))
    if action == "MoveRight":
        return neighbor_cell(cell, right_heading(heading))
    return str(cell)


@dataclass(frozen=True)
class MotionSafety:
    action: str
    known: bool
    safe: Optional[bool]
    confidence: float = 0.0
    observed_ratio: float = 0.0
    reason: str = "no_local_costmap"


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


def local_costmap_action_record(analysis: JsonDict, action: str) -> Optional[JsonDict]:
    costmap = analysis.get("local_costmap") if isinstance(analysis, dict) else None
    if not isinstance(costmap, dict) or str(costmap.get("status") or "") != "success":
        return None
    rec = (costmap.get("action_safety") or {}).get(str(action))
    return dict(rec) if isinstance(rec, dict) else None


def local_costmap_motion_safety(analysis: JsonDict, action: str) -> MotionSafety:
    action = str(action)
    rec = local_costmap_action_record(analysis, action)
    if not rec:
        return MotionSafety(action=action, known=False, safe=None)

    confidence = _float_value(rec.get("confidence"), 0.0)
    # Old/local test payloads may omit observed_ratio; modern costmap records
    # include it and then the per-action minimum below becomes authoritative.
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
        )
    if rec.get("safe") is True:
        min_confidence = _local_costmap_safe_min_confidence(action)
        min_observed = _local_costmap_min_observed_ratio(action, rec)
        return MotionSafety(
            action=action,
            known=confidence >= min_confidence and observed_ratio >= min_observed,
            safe=True,
            confidence=confidence,
            observed_ratio=observed_ratio,
            reason=reason,
        )
    return MotionSafety(
        action=action,
        known=False,
        safe=None,
        confidence=confidence,
        observed_ratio=observed_ratio,
        reason=reason,
    )


def four_neighbors(cell: str) -> List[str]:
    return [neighbor_cell(cell, heading) for heading in HEADING_ORDER]


def cell_distance(left: str, right: str) -> float:
    lx, lz = parse_cell(left)
    rx, rz = parse_cell(right)
    return math.hypot(float(lx - rx), float(lz - rz))


def manhattan_distance(left: str, right: str) -> int:
    lx, lz = parse_cell(left)
    rx, rz = parse_cell(right)
    return abs(lx - rx) + abs(lz - rz)


def cell_from_position(position: JsonDict, cell_size_m: float = DEFAULT_CELL_SIZE_M) -> str:
    x = float(position.get("x", 0.0) or 0.0)
    z = float(position.get("z", 0.0) or 0.0)
    return format_cell(round(x / cell_size_m), round(z / cell_size_m))


def pose_from_vision(vision: JsonDict, fallback_cell: str, fallback_heading: str, cell_size_m: float) -> Tuple[str, str, str]:
    """Return cell, heading, source.

    V2 online payloads normally omit true simulator pose, so fallback action
    odometry is the normal path. If a debug/eval payload explicitly carries
    position+rotation, we can consume it, but this should not be relied on by
    online policy.
    """
    position = vision.get("position") if isinstance(vision.get("position"), dict) else None
    rotation = vision.get("rotation") if isinstance(vision.get("rotation"), dict) else None
    if position is not None and rotation is not None:
        cell = cell_from_position(position, cell_size_m=cell_size_m)
        heading = heading_from_rotation(float(rotation.get("y", 0.0) or 0.0))
        return cell, heading, "explicit_pose_debug"
    return str(fallback_cell or "0,0"), str(fallback_heading or "north"), "action_odometry_fallback"


def bresenham_cells(start: str, end: str) -> List[str]:
    x0, z0 = parse_cell(start)
    x1, z1 = parse_cell(end)
    dx = abs(x1 - x0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz
    x, z = x0, z0
    out: List[str] = []
    while True:
        out.append(format_cell(x, z))
        if x == x1 and z == z1:
            break
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x += sx
        if e2 < dx:
            err += dx
            z += sz
    return out


@dataclass
class PoseState:
    cell: str = "0,0"
    x_cell: int = 0
    z_cell: int = 0
    heading: str = "north"
    theta_deg: float = 0.0
    pose_confidence: float = 1.0
    position_uncertainty_cells: float = 0.0
    heading_confidence: float = 1.0
    update_source: str = "initial"


@dataclass
class PositionCell:
    cell: str
    x: int
    z: int
    state: str = CELL_UNKNOWN
    visited: bool = False
    seen_count: int = 0
    collision_count: int = 0
    last_seen_step: Optional[int] = None
    last_visit_step: Optional[int] = None
    occupancy_confidence: float = 0.0
    free_evidence: int = 0
    occupied_evidence: int = 0
    inflated_from: List[str] = field(default_factory=list)


def default_position_map(room_name: str = "current_room") -> JsonDict:
    return {
        "schema_version": POSITION_MAP_SCHEMA_VERSION,
        "room_name": room_name or "current_room",
        "map_frame": {
            "coordinate_mode": "action_odometry_grid",
            "cell_size_m": DEFAULT_CELL_SIZE_M,
            "origin_cell": "0,0",
            "axis": {
                "x_cell_positive": "east",
                "z_cell_positive": "north",
            },
        },
        "pose": asdict(PoseState()),
        "cells": {
            "0,0": asdict(
                PositionCell(
                    cell="0,0",
                    x=0,
                    z=0,
                    state=CELL_FREE,
                    visited=True,
                    seen_count=1,
                    last_seen_step=0,
                    last_visit_step=0,
                    occupancy_confidence=1.0,
                    free_evidence=1,
                )
            )
        },
        "edges": {
            "known_open_edges": [],
            "blocked_edges": [],
            "hard_blocked_edges": [],
        },
        "frontiers": [],
        "recent_actions": [],
        "stats": {
            "visited_cell_count": 1,
            "free_cell_count": 1,
            "occupied_cell_count": 0,
            "inflated_cell_count": 0,
            "unknown_frontier_count": 0,
            "collision_count": 0,
            "last_updated_step": 0,
            "last_updated_at": now_iso(),
        },
    }


class PositionMap:
    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        *,
        target_cells: int = DEFAULT_TARGET_CELLS,
    ) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.path = self.memory_dir / "position-map.json"
        self.room_state_path = self.memory_dir / "room-state.json"
        self.target_cells = max(1, int(target_cells))

        self.depth_sample_stride = env_int("ROBOT_POSITION_MAP_DEPTH_STRIDE", 24)
        self.depth_max_m = env_float("ROBOT_POSITION_MAP_DEPTH_MAX_M", 3.0)
        self.depth_min_m = env_float("ROBOT_POSITION_MAP_DEPTH_MIN_M", 0.15)
        self.floor_height_max_m = env_float("ROBOT_POSITION_MAP_FLOOR_HEIGHT_MAX_M", 0.15)
        self.obstacle_height_min_m = env_float("ROBOT_POSITION_MAP_OBSTACLE_HEIGHT_MIN_M", 0.12)
        self.obstacle_height_max_m = env_float("ROBOT_POSITION_MAP_OBSTACLE_HEIGHT_MAX_M", 1.60)
        self.normal_inflation_radius_m = env_float("ROBOT_POSITION_MAP_ROBOT_RADIUS_M", 0.18)
        self.holding_inflation_radius_m = env_float("ROBOT_POSITION_MAP_HOLDING_RADIUS_M", 0.28)
        self.raycast_free = str(os.getenv("ROBOT_POSITION_MAP_RAYCAST_FREE", "1")).strip() not in {"0", "false", "False"}
        self.depth_cell_obstacle_min_points = max(1, env_int("ROBOT_POSITION_MAP_DEPTH_CELL_OBSTACLE_MIN_POINTS", 4))
        self.depth_cell_obstacle_min_ratio = clamp(env_float("ROBOT_POSITION_MAP_DEPTH_CELL_OBSTACLE_MIN_RATIO", 0.55), 0.0, 1.0)
        self.depth_occupied_threshold = max(1, env_int("ROBOT_POSITION_MAP_DEPTH_OCCUPIED_EVIDENCE_THRESHOLD", 3))
        self.depth_visited_occupied_threshold = max(
            self.depth_occupied_threshold,
            env_int("ROBOT_POSITION_MAP_DEPTH_VISITED_OCCUPIED_EVIDENCE_THRESHOLD", 8),
        )
        self.depth_occupied_free_ratio = max(0.0, env_float("ROBOT_POSITION_MAP_DEPTH_OCCUPIED_FREE_RATIO", 0.35))

    # ------------------------------------------------------------------
    # Load / save / normalize
    # ------------------------------------------------------------------

    def load(self) -> JsonDict:
        data = load_json(self.path, default_position_map())
        return self.normalize(data)

    def save(self, data: JsonDict) -> None:
        atomic_write_json(self.path, self.normalize(data))

    def reset(self, *, room_name: str = "current_room") -> JsonDict:
        data = default_position_map(room_name)
        self._refresh_frontiers_and_stats(data)
        self.save(data)
        return self.status()

    def normalize(self, data: JsonDict) -> JsonDict:
        base = default_position_map(str(data.get("room_name") or "current_room"))
        base.update(data if isinstance(data, dict) else {})

        if not isinstance(base.get("map_frame"), dict):
            base["map_frame"] = {}
        base["map_frame"].setdefault("coordinate_mode", "action_odometry_grid")
        base["map_frame"].setdefault("cell_size_m", DEFAULT_CELL_SIZE_M)
        try:
            base["map_frame"]["cell_size_m"] = float(base["map_frame"].get("cell_size_m", DEFAULT_CELL_SIZE_M) or DEFAULT_CELL_SIZE_M)
        except (TypeError, ValueError):
            base["map_frame"]["cell_size_m"] = DEFAULT_CELL_SIZE_M
        base["map_frame"].setdefault("origin_cell", "0,0")
        base["map_frame"].setdefault("axis", {"x_cell_positive": "east", "z_cell_positive": "north"})

        pose = base.get("pose") if isinstance(base.get("pose"), dict) else {}
        cell = str(pose.get("cell") or "0,0")
        try:
            x, z = parse_cell(cell)
        except Exception:
            x, z, cell = 0, 0, "0,0"
        heading = str(pose.get("heading") or "north")
        if heading not in HEADING_ORDER:
            heading = "north"
        base["pose"] = {
            "cell": cell,
            "x_cell": int(pose.get("x_cell", x) or x),
            "z_cell": int(pose.get("z_cell", z) or z),
            "heading": heading,
            "theta_deg": float(HEADING_THETA_DEG.get(heading, 0.0)),
            "pose_confidence": clamp(float(pose.get("pose_confidence", 1.0) or 1.0), 0.0, 1.0),
            "position_uncertainty_cells": max(0.0, float(pose.get("position_uncertainty_cells", 0.0) or 0.0)),
            "heading_confidence": clamp(float(pose.get("heading_confidence", 1.0) or 1.0), 0.0, 1.0),
            "update_source": str(pose.get("update_source") or "initial"),
        }

        cells = base.get("cells") if isinstance(base.get("cells"), dict) else {}
        normalized_cells: Dict[str, Any] = {}
        for key, raw in cells.items():
            cell_key = str(key)
            try:
                x, z = parse_cell(cell_key)
            except Exception:
                continue
            rec = raw if isinstance(raw, dict) else {}
            state = str(rec.get("state") or CELL_UNKNOWN)
            if state not in {CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED, CELL_INFLATED}:
                state = CELL_UNKNOWN
            normalized = {
                "cell": cell_key,
                "x": x,
                "z": z,
                "state": state,
                "visited": bool(rec.get("visited", False)),
                "seen_count": max(0, int(rec.get("seen_count", 0) or 0)),
                "collision_count": max(0, int(rec.get("collision_count", 0) or 0)),
                "last_seen_step": rec.get("last_seen_step"),
                "last_visit_step": rec.get("last_visit_step"),
                "occupancy_confidence": clamp(float(rec.get("occupancy_confidence", 0.0) or 0.0), 0.0, 1.0),
                "free_evidence": max(0, int(rec.get("free_evidence", 0) or 0)),
                "occupied_evidence": max(0, int(rec.get("occupied_evidence", 0) or 0)),
                "inflated_from": [str(item) for item in rec.get("inflated_from", []) or [] if str(item).strip()],
            }
            if (
                normalized["visited"]
                and normalized["collision_count"] == 0
                and normalized["free_evidence"] >= normalized["occupied_evidence"]
            ):
                normalized["state"] = CELL_FREE
            normalized_cells[cell_key] = normalized
        base["cells"] = normalized_cells

        self.ensure_cell(base, base["pose"]["cell"])
        self.mark_free(base, base["pose"]["cell"], evidence="pose_current", visited=True, step=base.get("stats", {}).get("last_updated_step"))

        edges = base.get("edges") if isinstance(base.get("edges"), dict) else {}
        hard_blocked_edges = unique_extend([], edges.get("hard_blocked_edges", []))
        blocked_edges = unique_extend(edges.get("blocked_edges", []), hard_blocked_edges)
        blocked_set = set(blocked_edges)
        base["edges"] = {
            "known_open_edges": [
                edge
                for edge in unique_extend([], edges.get("known_open_edges", []))
                if edge not in blocked_set
            ],
            "blocked_edges": blocked_edges,
            "hard_blocked_edges": hard_blocked_edges,
        }

        if not isinstance(base.get("recent_actions"), list):
            base["recent_actions"] = []
        base["recent_actions"] = [str(item) for item in base.get("recent_actions", [])][-24:]

        self._refresh_frontiers_and_stats(base)
        return base

    # ------------------------------------------------------------------
    # Cell / edge operations
    # ------------------------------------------------------------------

    def cell_size_m(self, data: JsonDict) -> float:
        frame = data.get("map_frame") if isinstance(data.get("map_frame"), dict) else {}
        try:
            return float(frame.get("cell_size_m", DEFAULT_CELL_SIZE_M) or DEFAULT_CELL_SIZE_M)
        except (TypeError, ValueError):
            return DEFAULT_CELL_SIZE_M

    def ensure_cell(self, data: JsonDict, cell: str) -> JsonDict:
        cells = data.setdefault("cells", {})
        if cell not in cells:
            x, z = parse_cell(cell)
            cells[cell] = asdict(PositionCell(cell=cell, x=x, z=z))
        return cells[cell]

    def mark_free(
        self,
        data: JsonDict,
        cell: str,
        *,
        evidence: str = "unknown",
        visited: bool = False,
        step: Optional[int] = None,
        weight: int = 1,
    ) -> None:
        rec = self.ensure_cell(data, cell)
        rec["free_evidence"] = int(rec.get("free_evidence", 0) or 0) + max(1, int(weight))
        # Pose/action evidence is authoritative for the robot footprint: the
        # cell containing the robot cannot simultaneously be a static obstacle.
        if visited and int(rec.get("collision_count", 0) or 0) == 0:
            rec["state"] = CELL_FREE
        elif (
            int(rec.get("occupied_evidence", 0) or 0) == 0
            or rec.get("state") in {CELL_UNKNOWN, CELL_INFLATED}
            or int(rec.get("free_evidence", 0) or 0) >= int(rec.get("occupied_evidence", 0) or 0) * 2
        ):
            rec["state"] = CELL_FREE
        rec["occupancy_confidence"] = clamp(0.35 + 0.08 * int(rec.get("free_evidence", 0) or 0), 0.0, 1.0)
        if visited:
            rec["visited"] = True
            rec["seen_count"] = int(rec.get("seen_count", 0) or 0) + 1
            rec["last_visit_step"] = step
        rec["last_seen_step"] = step if step is not None else rec.get("last_seen_step")

    def mark_occupied(
        self,
        data: JsonDict,
        cell: str,
        *,
        evidence: str = "unknown",
        step: Optional[int] = None,
        collision: bool = False,
        weight: int = 1,
    ) -> None:
        rec = self.ensure_cell(data, cell)
        rec["occupied_evidence"] = int(rec.get("occupied_evidence", 0) or 0) + max(1, int(weight))
        depth_evidence = str(evidence or "").startswith("depth_")
        pose_cell = str(data.get("pose", {}).get("cell") or "")
        if collision:
            rec["state"] = CELL_OCCUPIED
        elif depth_evidence and cell == pose_cell:
            rec["state"] = CELL_FREE
        elif depth_evidence:
            threshold = (
                self.depth_visited_occupied_threshold
                if bool(rec.get("visited", False)) and int(rec.get("collision_count", 0) or 0) == 0
                else self.depth_occupied_threshold
            )
            occupied = int(rec.get("occupied_evidence", 0) or 0)
            free = int(rec.get("free_evidence", 0) or 0)
            if occupied >= threshold and occupied >= max(1, int(math.ceil(free * self.depth_occupied_free_ratio))):
                rec["state"] = CELL_OCCUPIED
            elif rec.get("state") in {CELL_UNKNOWN, CELL_INFLATED}:
                rec["state"] = CELL_UNKNOWN
        else:
            rec["state"] = CELL_OCCUPIED
        rec["occupancy_confidence"] = clamp(0.45 + 0.12 * int(rec.get("occupied_evidence", 0) or 0), 0.0, 1.0)
        rec["last_seen_step"] = step if step is not None else rec.get("last_seen_step")
        if collision:
            rec["collision_count"] = int(rec.get("collision_count", 0) or 0) + 1

    def add_open_edge(
        self,
        data: JsonDict,
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
        edges = data.setdefault("edges", {})
        blocked = set(edges.setdefault("blocked_edges", []))
        hard_blocked = set(edges.setdefault("hard_blocked_edges", []))
        if direct in hard_blocked or reverse in hard_blocked:
            if not (override_blocked and override_hard):
                return
        if direct in blocked or reverse in blocked:
            if not override_blocked:
                return
            self.clear_blocked_edge(data, source, target, clear_hard=override_hard)
            blocked = set(data.setdefault("edges", {}).setdefault("blocked_edges", []))
        if direct in blocked or reverse in blocked:
            return
        data["edges"]["known_open_edges"] = unique_extend(
            data["edges"].get("known_open_edges", []),
            [direct, reverse],
        )

    def add_blocked_edge(self, data: JsonDict, source: str, target: str, *, hard: bool = False) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        edges = data.setdefault("edges", {})
        edges["blocked_edges"] = unique_extend(edges.get("blocked_edges", []), [direct, reverse])
        if hard:
            edges["hard_blocked_edges"] = unique_extend(edges.get("hard_blocked_edges", []), [direct, reverse])
        edges["known_open_edges"] = [
            edge
            for edge in edges.get("known_open_edges", []) or []
            if edge not in {direct, reverse}
        ]

    def clear_blocked_edge(self, data: JsonDict, source: str, target: str, *, clear_hard: bool = False) -> None:
        if source == target:
            return
        direct = edge_key(source, target)
        reverse = edge_key(target, source)
        edges = data.setdefault("edges", {})
        if not clear_hard:
            hard_blocked = set(edges.get("hard_blocked_edges", []) or [])
            if direct in hard_blocked or reverse in hard_blocked:
                return
        edges["blocked_edges"] = [
            edge
            for edge in edges.get("blocked_edges", []) or []
            if edge not in {direct, reverse}
        ]
        if clear_hard:
            edges["hard_blocked_edges"] = [
                edge
                for edge in edges.get("hard_blocked_edges", []) or []
                if edge not in {direct, reverse}
            ]

    def edge_blocked(self, data: JsonDict, source: str, target: str) -> bool:
        blocked = set(data.get("edges", {}).get("blocked_edges", []) or [])
        return edge_key(source, target) in blocked or edge_key(target, source) in blocked

    def edge_hard_blocked(self, data: JsonDict, source: str, target: str) -> bool:
        hard = set(data.get("edges", {}).get("hard_blocked_edges", []) or [])
        return edge_key(source, target) in hard or edge_key(target, source) in hard

    def observe_safe_edge(
        self,
        data: JsonDict,
        source: str,
        target: str,
        *,
        action: str,
        safety: MotionSafety,
        step: Optional[int],
    ) -> None:
        """Clear stale soft blocks using current RGB-D swept-volume evidence."""
        if source == target:
            return
        if not (safety.known and safety.safe is True):
            return
        if self.edge_hard_blocked(data, source, target):
            return
        was_blocked = self.edge_blocked(data, source, target)
        self.add_open_edge(data, source, target, override_blocked=True, override_hard=False)
        if self.edge_blocked(data, source, target):
            return
        self.mark_free(
            data,
            target,
            evidence=f"local_costmap_safe:{action}:{safety.reason}",
            visited=False,
            step=step,
        )
        if was_blocked:
            stats = data.setdefault("stats", {})
            cleared = stats.setdefault("soft_blocked_edges_cleared", [])
            if isinstance(cleared, list):
                cleared.append(
                    {
                        "edge": edge_pair_key(source, target),
                        "action": action,
                        "confidence": round(float(safety.confidence), 4),
                        "observed_ratio": round(float(safety.observed_ratio), 4),
                        "reason": safety.reason,
                        "step": step,
                        "time": now_iso(),
                    }
                )
                stats["soft_blocked_edges_cleared"] = cleared[-24:]

    # ------------------------------------------------------------------
    # Observation / action update
    # ------------------------------------------------------------------

    def observe(
        self,
        *,
        vision: JsonDict,
        analysis: Optional[JsonDict] = None,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> JsonDict:
        data = self.load()
        pose = data["pose"]
        cell_size = self.cell_size_m(data)
        cell, heading, source = pose_from_vision(
            vision or {},
            fallback_cell=str(pose.get("cell") or "0,0"),
            fallback_heading=str(pose.get("heading") or "north"),
            cell_size_m=cell_size,
        )
        self._set_pose(data, cell=cell, heading=heading, source=source, step=step)
        self.mark_free(data, cell, evidence="observe_pose", visited=True, step=step)
        self._record_open_directions(data, cell, heading, analysis or {})
        self.update_from_depth(data, vision=vision or {}, analysis=analysis or {}, step=step)
        self.inflate_obstacles(data, holding_object=bool((analysis or {}).get("holding_object", False)))
        self._recover_pose_confidence_from_local_observation(data, analysis or {})
        self._refresh_frontiers_and_stats(data, step=step)
        if persist:
            self.save(data)
        return self.status_from_data(data)

    def record_action(
        self,
        *,
        action: str,
        success: bool,
        vision: JsonDict,
        analysis: Optional[JsonDict] = None,
        action_result: Optional[JsonDict] = None,
        failure_reason: Optional[str] = None,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> JsonDict:
        data = self.load()
        analysis = analysis or {}
        action_result = action_result or {}
        pose = data["pose"]
        before_cell = str(pose.get("cell") or "0,0")
        before_heading = str(pose.get("heading") or "north")
        after_cell = before_cell
        after_heading = before_heading

        explicit_after_position = (
            action_result.get("after_position")
            if isinstance(action_result.get("after_position"), dict)
            else action_result.get("position_after")
            if isinstance(action_result.get("position_after"), dict)
            else None
        )
        explicit_after_rotation = (
            action_result.get("after_rotation")
            if isinstance(action_result.get("after_rotation"), dict)
            else action_result.get("rotation_after")
            if isinstance(action_result.get("rotation_after"), dict)
            else None
        )
        if success and explicit_after_position and explicit_after_rotation:
            after_cell = cell_from_position(explicit_after_position, self.cell_size_m(data))
            after_heading = heading_from_rotation(float(explicit_after_rotation.get("y", 0.0) or 0.0))
        elif success:
            if action == "RotateLeft":
                after_heading = left_heading(before_heading)
            elif action == "RotateRight":
                after_heading = right_heading(before_heading)
            elif action in TRANSLATION_ACTIONS:
                after_cell = neighbor_for_action(before_cell, before_heading, action)

        self.mark_free(data, before_cell, evidence="action_before", visited=True, step=step)

        if success:
            self._set_pose(data, cell=after_cell, heading=after_heading, source="action_odometry", step=step)
            self._degrade_pose_confidence(data, action=action, success=True)
            self.mark_free(data, after_cell, evidence=f"action_success:{action}", visited=True, step=step)
            if action in TRANSLATION_ACTIONS and before_cell != after_cell:
                self.add_open_edge(data, before_cell, after_cell, override_blocked=True, override_hard=True)
        else:
            self._degrade_pose_confidence(data, action=action, success=False)
            if action in TRANSLATION_ACTIONS:
                target_cell = neighbor_for_action(before_cell, before_heading, action)
                self.mark_occupied(data, target_cell, evidence=f"action_failed:{action}:{failure_reason or 'unknown'}", step=step, collision=True)
                self.add_blocked_edge(data, before_cell, target_cell, hard=True)
                stats = data.setdefault("stats", {})
                stats["collision_count"] = int(stats.get("collision_count", 0) or 0) + 1

        self._record_open_directions(data, str(data["pose"].get("cell") or before_cell), str(data["pose"].get("heading") or before_heading), analysis)
        self.update_from_depth(data, vision=vision or {}, analysis=analysis, step=step)
        self.inflate_obstacles(data, holding_object=bool(analysis.get("holding_object", False)))
        self._recover_pose_confidence_from_local_observation(data, analysis)
        recent = list(data.get("recent_actions", []) or [])
        recent.append(action if success else f"{action}:failed:{failure_reason or 'unknown'}")
        data["recent_actions"] = recent[-24:]
        self._refresh_frontiers_and_stats(data, step=step)
        if persist:
            self.save(data)
        return self.status_from_data(data)

    def _set_pose(self, data: JsonDict, *, cell: str, heading: str, source: str, step: Optional[int]) -> None:
        x, z = parse_cell(cell)
        if heading not in HEADING_ORDER:
            heading = "north"
        pose = data.setdefault("pose", {})
        pose["cell"] = cell
        pose["x_cell"] = x
        pose["z_cell"] = z
        pose["heading"] = heading
        pose["theta_deg"] = float(HEADING_THETA_DEG.get(heading, 0.0))
        pose["update_source"] = source

        if source == "explicit_pose_debug":
            self._degrade_pose_confidence(data, action="explicit_pose_debug", success=True)

    def _degrade_pose_confidence(self, data: JsonDict, *, action: str, success: bool) -> None:
        pose = data.setdefault("pose", {})
        pose_conf = float(pose.get("pose_confidence", 1.0) or 1.0)
        heading_conf = float(pose.get("heading_confidence", 1.0) or 1.0)
        uncertainty = float(pose.get("position_uncertainty_cells", 0.0) or 0.0)

        if not success:
            pose_conf *= 0.92
            heading_conf *= 0.96
            uncertainty += 0.75
        elif action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
            pose_conf *= 0.985
            uncertainty += 0.05
        elif action in ROTATE_ACTIONS:
            heading_conf *= 0.99
            uncertainty += 0.02
        elif action == "explicit_pose_debug":
            pose_conf = max(pose_conf, 0.98)
            heading_conf = max(heading_conf, 0.98)
            uncertainty = min(uncertainty, 0.25)
        else:
            pose_conf *= 0.995

        pose["pose_confidence"] = clamp(pose_conf, 0.20, 1.0)
        pose["heading_confidence"] = clamp(heading_conf, 0.20, 1.0)
        pose["position_uncertainty_cells"] = clamp(uncertainty, 0.0, 6.0)

    def _recover_pose_confidence_from_local_observation(self, data: JsonDict, analysis: JsonDict) -> None:
        """Let stable RGB-D observations counteract unbounded odometry decay.

        This is not a full SLAM relocalizer.  It is the lightweight equivalent
        of a mature stack's localization update: when the current local costmap
        has broad depth support, the pose estimate becomes slightly less stale
        instead of degrading forever during long patrols.
        """
        costmap = analysis.get("local_costmap") if isinstance(analysis, dict) else None
        if not isinstance(costmap, dict) or str(costmap.get("status") or "") != "success":
            return
        try:
            observed = int(costmap.get("observed_cell_count", 0) or 0)
        except (TypeError, ValueError):
            observed = 0
        min_observed = max(1, env_int("ROBOT_POSITION_MAP_RELOCALIZE_MIN_OBSERVED_CELLS", 120))
        if observed < min_observed:
            return

        action_records = costmap.get("action_safety") if isinstance(costmap.get("action_safety"), dict) else {}
        known_motion = 0
        for action in ("MoveAhead", "RotateLeft", "RotateRight"):
            rec = action_records.get(action) if isinstance(action_records.get(action), dict) else None
            if not rec:
                continue
            if rec.get("safe") in {True, False}:
                known_motion += 1
        if known_motion < 2:
            return

        pose = data.setdefault("pose", {})
        pose_conf = clamp(float(pose.get("pose_confidence", 1.0) or 1.0), 0.20, 1.0)
        heading_conf = clamp(float(pose.get("heading_confidence", 1.0) or 1.0), 0.20, 1.0)
        uncertainty = clamp(float(pose.get("position_uncertainty_cells", 0.0) or 0.0), 0.0, 6.0)

        pose_gain = env_float("ROBOT_POSITION_MAP_RELOCALIZE_POSE_GAIN", 0.012)
        heading_gain = env_float("ROBOT_POSITION_MAP_RELOCALIZE_HEADING_GAIN", 0.016)
        uncertainty_drop = env_float("ROBOT_POSITION_MAP_RELOCALIZE_UNCERTAINTY_DROP", 0.12)
        pose["pose_confidence"] = clamp(pose_conf + pose_gain, 0.20, 1.0)
        pose["heading_confidence"] = clamp(heading_conf + heading_gain, 0.20, 1.0)
        pose["position_uncertainty_cells"] = clamp(uncertainty - uncertainty_drop, 0.0, 6.0)

    def _record_open_directions(self, data: JsonDict, cell: str, heading: str, analysis: JsonDict) -> None:
        if not isinstance(analysis, dict):
            return
        open_directions = set(analysis.get("open_directions", []) or [])
        obstacle_ahead = bool(analysis.get("obstacle_ahead", False))

        forward_target = neighbor_cell(cell, heading)
        forward_safety = local_costmap_motion_safety(analysis, "MoveAhead")
        if forward_safety.known and forward_safety.safe is True:
            self.observe_safe_edge(
                data,
                cell,
                forward_target,
                action="MoveAhead",
                safety=forward_safety,
                step=analysis.get("step"),
            )
        elif forward_safety.known and forward_safety.safe is False:
            self.add_blocked_edge(data, cell, forward_target)
        elif "forward" in open_directions and not obstacle_ahead and not self.edge_blocked(data, cell, forward_target):
            self.mark_free(data, forward_target, evidence="open_direction_forward", visited=False, step=analysis.get("step"))
            self.add_open_edge(data, cell, forward_target)

        left_target = neighbor_cell(cell, left_heading(heading))
        left_safety = local_costmap_motion_safety(analysis, "MoveLeft")
        if left_safety.known and left_safety.safe is True:
            self.observe_safe_edge(
                data,
                cell,
                left_target,
                action="MoveLeft",
                safety=left_safety,
                step=analysis.get("step"),
            )
        elif left_safety.known and left_safety.safe is False:
            self.add_blocked_edge(data, cell, left_target)
        elif "left" in open_directions and not self.edge_blocked(data, cell, left_target):
            self.mark_free(data, left_target, evidence="open_direction_left", visited=False, step=analysis.get("step"))
            self.add_open_edge(data, cell, left_target)

        right_target = neighbor_cell(cell, right_heading(heading))
        right_safety = local_costmap_motion_safety(analysis, "MoveRight")
        if right_safety.known and right_safety.safe is True:
            self.observe_safe_edge(
                data,
                cell,
                right_target,
                action="MoveRight",
                safety=right_safety,
                step=analysis.get("step"),
            )
        elif right_safety.known and right_safety.safe is False:
            self.add_blocked_edge(data, cell, right_target)
        elif "right" in open_directions and not self.edge_blocked(data, cell, right_target):
            self.mark_free(data, right_target, evidence="open_direction_right", visited=False, step=analysis.get("step"))
            self.add_open_edge(data, cell, right_target)

        if obstacle_ahead and not forward_safety.known:
            self.add_blocked_edge(data, cell, forward_target)

    # ------------------------------------------------------------------
    # Depth occupancy update
    # ------------------------------------------------------------------

    def update_from_depth(self, data: JsonDict, *, vision: JsonDict, analysis: Optional[JsonDict], step: Optional[int]) -> JsonDict:
        depth = self._load_depth_array(vision)
        if depth is None:
            return {
                "status": "skipped",
                "reason": "no_depth",
            }
        camera = self._camera_params(vision)
        if not camera:
            return {
                "status": "skipped",
                "reason": "no_camera",
            }

        try:
            import numpy as np  # type: ignore
        except Exception:
            return {
                "status": "skipped",
                "reason": "numpy_unavailable",
            }

        arr = np.asarray(depth)
        if arr.ndim != 2:
            return {
                "status": "skipped",
                "reason": f"bad_depth_shape:{arr.shape}",
            }

        h, w = arr.shape
        fx = float(camera.get("fx") or camera.get("focal_x") or max(1.0, w / 2.0))
        fy = float(camera.get("fy") or camera.get("focal_y") or max(1.0, h / 2.0))
        cx = float(camera.get("cx") or (w / 2.0))
        cy = float(camera.get("cy") or (h / 2.0))
        camera_height = float(camera.get("camera_height_m", 0.901) or 0.901)
        camera_horizon_deg = float(camera.get("camera_horizon_deg", 0.0) or 0.0)
        camera_pitch = math.radians(camera_horizon_deg)
        cos_pitch = math.cos(camera_pitch)
        sin_pitch = math.sin(camera_pitch)

        pose_cell = str(data.get("pose", {}).get("cell") or "0,0")
        pose_heading = str(data.get("pose", {}).get("heading") or "north")
        cell_size = self.cell_size_m(data)

        ray_free_cells: set[str] = set()
        floor_like_hits: Dict[str, int] = {}
        obstacle_hits: Dict[str, int] = {}

        stride = max(4, int(self.depth_sample_stride))
        for v in range(0, h, stride):
            for u in range(0, w, stride):
                z_forward = float(arr[v, u])
                if not math.isfinite(z_forward):
                    continue
                if z_forward < self.depth_min_m or z_forward > self.depth_max_m:
                    continue

                # Pinhole camera approximation corrected for the current
                # camera horizon.  AI2-THOR depth is expressed along the camera
                # ray, while the position map needs ground-plane forward
                # distance and world-relative obstacle height.
                x_right = (float(u) - cx) * z_forward / fx
                y_up_camera = -(float(v) - cy) * z_forward / fy
                ground_forward = z_forward * cos_pitch + y_up_camera * sin_pitch
                world_height_est = camera_height + y_up_camera * cos_pitch - z_forward * sin_pitch

                target_cell = self._local_metric_to_global_cell(
                    pose_cell=pose_cell,
                    pose_heading=pose_heading,
                    x_right_m=x_right,
                    z_forward_m=ground_forward,
                    cell_size_m=cell_size,
                )

                if self.raycast_free:
                    for free_cell in bresenham_cells(pose_cell, target_cell)[:-1]:
                        ray_free_cells.add(free_cell)

                # Conservative obstacle marking. This is intentionally rough:
                # semantic/local costmap can refine it later.  Aggregate per
                # grid cell first; a single projected tabletop point should not
                # permanently occupy the global navigation map.
                if self.obstacle_height_min_m <= world_height_est <= self.obstacle_height_max_m:
                    obstacle_hits[target_cell] = int(obstacle_hits.get(target_cell, 0) or 0) + 1
                elif world_height_est <= self.floor_height_max_m:
                    floor_like_hits[target_cell] = int(floor_like_hits.get(target_cell, 0) or 0) + 1

        marked_free = 0
        marked_occupied = 0

        for free_cell in ray_free_cells:
            self.mark_free(data, free_cell, evidence="depth_ray", visited=False, step=step)
            marked_free += 1

        for floor_cell, hits in floor_like_hits.items():
            self.mark_free(data, floor_cell, evidence="depth_floor_like_cell", visited=False, step=step)
            marked_free += 1

        for occ_cell, hits in obstacle_hits.items():
            floor_hits = int(floor_like_hits.get(occ_cell, 0) or 0)
            total_hits = hits + floor_hits
            obstacle_ratio = float(hits) / float(max(1, total_hits))
            if hits < self.depth_cell_obstacle_min_points or obstacle_ratio < self.depth_cell_obstacle_min_ratio:
                continue
            self.mark_occupied(data, occ_cell, evidence="depth_obstacle_cell", step=step)
            marked_occupied += 1

        return {
            "status": "success",
            "marked_free": marked_free,
            "marked_occupied": marked_occupied,
            "depth_obstacle_cell_count": len(obstacle_hits),
            "depth_floor_cell_count": len(floor_like_hits),
        }

    def _load_depth_array(self, vision: JsonDict) -> Any:
        if not isinstance(vision, dict):
            return None

        depth = vision.get("depth_frame", None)
        if depth is not None and not isinstance(depth, dict):
            return depth

        # Sanitized get-vision payloads expose ``depth`` as metadata
        # (encoding/shape/unit) and ``depth_path`` as the local .npy file.  Do
        # not accidentally feed the metadata dictionary to numpy.asarray.
        inline_depth = vision.get("depth", None)
        if inline_depth is not None and not isinstance(inline_depth, dict):
            return inline_depth

        depth_path = vision.get("depth_path") or vision.get("depth_npy_path")
        if not depth_path:
            return None

        try:
            import numpy as np  # type: ignore
        except Exception:
            return None

        try:
            return np.load(str(depth_path))
        except Exception:
            return None

    def _camera_params(self, vision: JsonDict) -> JsonDict:
        if not isinstance(vision, dict):
            return {}
        for key in ("camera_json", "camera", "camera_params", "camera_intrinsics"):
            value = vision.get(key)
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, str) and value.strip():
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        # Some get-vision outputs flatten these fields.
        keys = {"fx", "fy", "cx", "cy", "width", "height", "fov_deg", "camera_height_m", "camera_horizon_deg"}
        if any(key in vision for key in keys):
            return {key: vision.get(key) for key in keys if key in vision}
        return {}

    def _local_metric_to_global_cell(
        self,
        *,
        pose_cell: str,
        pose_heading: str,
        x_right_m: float,
        z_forward_m: float,
        cell_size_m: float,
    ) -> str:
        # Local camera/robot coordinates:
        # z_forward = robot forward, x_right = robot right.
        forward_cells = int(round(z_forward_m / max(1e-6, cell_size_m)))
        right_cells = int(round(x_right_m / max(1e-6, cell_size_m)))

        cx, cz = parse_cell(pose_cell)
        fdx, fdz = HEADING_VECTORS.get(pose_heading, (0, 1))
        right_heading_name = right_heading(pose_heading)
        rdx, rdz = HEADING_VECTORS.get(right_heading_name, (1, 0))

        gx = cx + fdx * forward_cells + rdx * right_cells
        gz = cz + fdz * forward_cells + rdz * right_cells
        return format_cell(gx, gz)

    # ------------------------------------------------------------------
    # Inflation / frontier / status
    # ------------------------------------------------------------------

    def inflate_obstacles(self, data: JsonDict, *, holding_object: bool = False) -> None:
        cells = data.setdefault("cells", {})
        # Clear previous inflated cells that have no direct evidence.
        for key, rec in list(cells.items()):
            if rec.get("state") == CELL_INFLATED:
                rec["state"] = CELL_UNKNOWN
                rec["inflated_from"] = []

        radius_m = self.holding_inflation_radius_m if holding_object else self.normal_inflation_radius_m
        cell_size = self.cell_size_m(data)
        radius_cells = max(0, int(math.ceil(radius_m / max(1e-6, cell_size))))
        if radius_cells <= 0:
            return

        occupied = [
            key
            for key, rec in cells.items()
            if rec.get("state") == CELL_OCCUPIED
        ]
        for occ in occupied:
            ox, oz = parse_cell(occ)
            for dx in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    if math.hypot(dx, dz) > radius_cells:
                        continue
                    cell = format_cell(ox + dx, oz + dz)
                    if cell == occ:
                        continue
                    rec = self.ensure_cell(data, cell)
                    if rec.get("state") in {CELL_OCCUPIED, CELL_FREE} and rec.get("visited"):
                        # Do not erase visited free cells with inflation only.
                        continue
                    if rec.get("state") != CELL_OCCUPIED:
                        rec["state"] = CELL_INFLATED
                        rec["inflated_from"] = unique_extend(rec.get("inflated_from", []), [occ])
                        rec["occupancy_confidence"] = max(float(rec.get("occupancy_confidence", 0.0) or 0.0), 0.35)

    def extract_frontiers(self, data: JsonDict) -> List[str]:
        cells = data.setdefault("cells", {})
        frontiers: List[str] = []
        for cell, rec in cells.items():
            if rec.get("state") != CELL_FREE:
                continue
            for nb in four_neighbors(cell):
                nb_rec = cells.get(nb)
                if nb_rec is None or nb_rec.get("state", CELL_UNKNOWN) == CELL_UNKNOWN:
                    frontiers.append(nb)
        return unique_extend([], frontiers)

    def _refresh_frontiers_and_stats(self, data: JsonDict, step: Optional[int] = None) -> None:
        frontiers = self.extract_frontiers(data)
        data["frontiers"] = frontiers

        cells = data.get("cells", {})
        visited = [key for key, rec in cells.items() if rec.get("visited")]
        free = [key for key, rec in cells.items() if rec.get("state") == CELL_FREE]
        occupied = [key for key, rec in cells.items() if rec.get("state") == CELL_OCCUPIED]
        inflated = [key for key, rec in cells.items() if rec.get("state") == CELL_INFLATED]
        collision_count = sum(int(rec.get("collision_count", 0) or 0) for rec in cells.values())

        stats = data.setdefault("stats", {})
        stats["visited_cell_count"] = len(visited)
        stats["free_cell_count"] = len(free)
        stats["occupied_cell_count"] = len(occupied)
        stats["inflated_cell_count"] = len(inflated)
        stats["unknown_frontier_count"] = len(frontiers)
        stats["collision_count"] = collision_count
        if step is not None:
            stats["last_updated_step"] = step
        stats["last_updated_at"] = now_iso()

    def coverage_estimate(self, data: JsonDict) -> float:
        visited_count = int(data.get("stats", {}).get("visited_cell_count", 0) or 0)
        return clamp(float(visited_count) / float(max(1, self.target_cells)), 0.0, 1.0)

    def occupancy_summary(self, data: JsonDict) -> JsonDict:
        stats = data.get("stats", {})
        return {
            "free_cell_count": stats.get("free_cell_count", 0),
            "occupied_cell_count": stats.get("occupied_cell_count", 0),
            "inflated_cell_count": stats.get("inflated_cell_count", 0),
            "unknown_frontier_count": stats.get("unknown_frontier_count", 0),
            "collision_count": stats.get("collision_count", 0),
        }

    def pose_trust_summary(self, data: JsonDict) -> JsonDict:
        """Summarize how much downstream modules should trust exact map cells.

        Action odometry without relocalization is intentionally conservative: the
        longer the run, the less safe it is to treat an old remembered cell as an
        exact camera viewpoint. Downstream planners should degrade gracefully from
        exact-viewpoint navigation to coarse-region / visual reacquisition.
        """
        pose = data.get("pose") if isinstance(data.get("pose"), dict) else {}
        try:
            confidence = float(pose.get("pose_confidence", 1.0) or 1.0)
        except (TypeError, ValueError):
            confidence = 1.0
        try:
            uncertainty = float(pose.get("position_uncertainty_cells", 0.0) or 0.0)
        except (TypeError, ValueError):
            uncertainty = 0.0
        try:
            heading_confidence = float(pose.get("heading_confidence", 1.0) or 1.0)
        except (TypeError, ValueError):
            heading_confidence = 1.0

        low_conf = env_float("ROBOT_POSITION_MAP_LOW_POSE_CONFIDENCE", 0.45)
        high_uncertainty = env_float("ROBOT_POSITION_MAP_HIGH_UNCERTAINTY_CELLS", 3.0)
        medium_conf = env_float("ROBOT_POSITION_MAP_MEDIUM_POSE_CONFIDENCE", 0.68)
        medium_uncertainty = env_float("ROBOT_POSITION_MAP_MEDIUM_UNCERTAINTY_CELLS", 1.75)
        if confidence < low_conf or uncertainty >= high_uncertainty:
            level = "low"
        elif confidence < medium_conf or uncertainty >= medium_uncertainty:
            level = "medium"
        else:
            level = "high"
        return {
            "level": level,
            "pose_confidence": round(clamp(confidence, 0.0, 1.0), 4),
            "position_uncertainty_cells": round(max(0.0, uncertainty), 4),
            "heading_confidence": round(clamp(heading_confidence, 0.0, 1.0), 4),
            "precise_viewpoint_allowed": level == "high",
            "region_only": level == "low",
            "relocalization_recommended": level != "high",
        }

    def export_room_state_compat(self, data: JsonDict) -> JsonDict:
        cells = data.get("cells", {})
        visited_cells = unique_extend([], [key for key, rec in cells.items() if rec.get("visited")])
        visited_counts = {
            key: int(rec.get("seen_count", 1) or 1)
            for key, rec in cells.items()
            if rec.get("visited")
        }
        edges = data.get("edges", {})
        pose = data.get("pose", {})
        frontiers = unique_extend([], data.get("frontiers", []))
        return {
            "cell_size": self.cell_size_m(data),
            "last_cell": pose.get("cell"),
            "last_heading": pose.get("heading"),
            "visited_cells": visited_cells,
            "visited_cell_counts": visited_counts,
            "known_open_edges": unique_extend([], edges.get("known_open_edges", [])),
            "blocked_edges": unique_extend([], edges.get("blocked_edges", [])),
            "hard_blocked_edges": unique_extend([], edges.get("hard_blocked_edges", [])),
            "frontier_cells": frontiers,
            "known_frontier_cells": list(frontiers),
            "coverage_estimate": self.coverage_estimate(data),
            "collision_count": data.get("stats", {}).get("collision_count", 0),
            "position_map_status": "active",
            "pose_confidence": pose.get("pose_confidence", 1.0),
            "position_uncertainty_cells": pose.get("position_uncertainty_cells", 0.0),
            "heading_confidence": pose.get("heading_confidence", 1.0),
            "pose_trust": self.pose_trust_summary(data),
            "occupancy_summary": self.occupancy_summary(data),
        }

    def status(self) -> JsonDict:
        return self.status_from_data(self.load())

    def status_from_data(self, data: JsonDict) -> JsonDict:
        compat = self.export_room_state_compat(data)
        pose = data.get("pose", {})
        return {
            "status": "success",
            "result_type": "position_map_status",
            "room_name": data.get("room_name"),
            "map_frame": data.get("map_frame", {}),
            "pose": pose,
            "last_cell": pose.get("cell"),
            "last_heading": pose.get("heading"),
            "frontier_cells": data.get("frontiers", []),
            "cells": data.get("cells", {}),
            "edges": data.get("edges", {}),
            "stats": data.get("stats", {}),
            "occupancy_summary": self.occupancy_summary(data),
            "pose_trust": self.pose_trust_summary(data),
            "room_state_compat": compat,
            **compat,
        }


def parse_json_arg(value: Optional[str]) -> JsonDict:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage robot-cleaner position map.")
    parser.add_argument("--memory-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show", help="Show position-map status.")
    sub.add_parser("status", help="Alias for show.")

    reset = sub.add_parser("reset", help="Reset position map.")
    reset.add_argument("--room", default="current_room")

    observe = sub.add_parser("observe", help="Observe current pose/depth.")
    observe.add_argument("--vision-json", required=True)
    observe.add_argument("--analysis-json", default="{}")
    observe.add_argument("--step", type=int, default=None)

    record = sub.add_parser("record-action", help="Record one action result.")
    record.add_argument("--action", required=True)
    record.add_argument("--success", action="store_true")
    record.add_argument("--failed", action="store_true")
    record.add_argument("--failure-reason", default=None)
    record.add_argument("--vision-json", required=True)
    record.add_argument("--analysis-json", default="{}")
    record.add_argument("--action-result-json", default="{}")
    record.add_argument("--step", type=int, default=None)

    return parser


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    memory_dir = Path(args.memory_dir) if args.memory_dir else None
    manager = PositionMap(memory_dir)

    try:
        if args.command in {"show", "status"}:
            result = manager.status()
        elif args.command == "reset":
            result = manager.reset(room_name=args.room)
        elif args.command == "observe":
            result = manager.observe(
                vision=parse_json_arg(args.vision_json),
                analysis=parse_json_arg(args.analysis_json),
                step=args.step,
            )
        elif args.command == "record-action":
            if args.success and args.failed:
                raise ValueError("--success and --failed are mutually exclusive")
            success = not bool(args.failed)
            result = manager.record_action(
                action=args.action,
                success=success,
                failure_reason=args.failure_reason,
                vision=parse_json_arg(args.vision_json),
                analysis=parse_json_arg(args.analysis_json),
                action_result=parse_json_arg(args.action_result_json),
                step=args.step,
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print_json({"status": "error", "result_type": "position_map_error", "message": str(exc)})
        return 1

    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
