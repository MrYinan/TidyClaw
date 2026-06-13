#!/usr/bin/env python3
"""Action-odometry map backend backed by existing memory JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from scripts.map_backend.base import (
        JsonDict,
        MapSnapshot,
        as_dict,
        now_iso,
        parse_cell,
        safe_float,
        safe_int,
        unique_strings,
    )
    from scripts.position_map_core import DEFAULT_CELL_SIZE_M, PositionMap
except ImportError:  # pragma: no cover - direct script execution
    from map_backend.base import (
        JsonDict,
        MapSnapshot,
        as_dict,
        now_iso,
        parse_cell,
        safe_float,
        safe_int,
        unique_strings,
    )
    from position_map_core import DEFAULT_CELL_SIZE_M, PositionMap


def read_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def file_info(path: Path) -> JsonDict:
    info: JsonDict = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        info["loaded"] = False
        return info
    info.update(
        {
            "loaded": True,
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }
    )
    return info


def normalized_pose(position_status: JsonDict) -> JsonDict:
    pose = dict(as_dict(position_status.get("pose")))
    cell = str(pose.get("cell") or position_status.get("last_cell") or "0,0")
    parsed = parse_cell(cell) or (0, 0)
    heading = str(pose.get("heading") or position_status.get("last_heading") or "north")
    pose["cell"] = f"{parsed[0]},{parsed[1]}"
    pose.setdefault("x_cell", parsed[0])
    pose.setdefault("z_cell", parsed[1])
    pose.setdefault("heading", heading)
    pose.setdefault("pose_confidence", safe_float(position_status.get("pose_confidence"), 1.0))
    pose.setdefault("confidence", pose.get("pose_confidence"))
    pose.setdefault(
        "position_uncertainty_cells",
        safe_float(position_status.get("position_uncertainty_cells"), 0.0),
    )
    pose.setdefault("heading_confidence", safe_float(position_status.get("heading_confidence"), 1.0))
    pose.setdefault("update_source", "action_odometry")
    return pose


def normalized_map_frame(position_status: JsonDict, room_state: JsonDict) -> JsonDict:
    frame = dict(as_dict(position_status.get("map_frame")))
    frame.setdefault("coordinate_mode", "action_odometry_grid")
    frame.setdefault("cell_size_m", room_state.get("cell_size") or DEFAULT_CELL_SIZE_M)
    frame.setdefault("origin_cell", "0,0")
    frame.setdefault(
        "axis",
        {
            "x_cell_positive": "east",
            "z_cell_positive": "north",
        },
    )
    return frame


def normalized_edges(position_status: JsonDict, room_state: JsonDict) -> JsonDict:
    edges = as_dict(position_status.get("edges"))
    return {
        "known_open_edges": unique_strings(edges.get("known_open_edges"), room_state.get("known_open_edges")),
        "blocked_edges": unique_strings(edges.get("blocked_edges"), room_state.get("blocked_edges")),
        "hard_blocked_edges": unique_strings(
            edges.get("hard_blocked_edges"),
            room_state.get("hard_blocked_edges"),
        ),
    }


def normalized_frontiers(position_status: JsonDict, room_state: JsonDict) -> list[str]:
    return unique_strings(
        position_status.get("frontiers"),
        position_status.get("frontier_cells"),
        room_state.get("frontier_cells"),
        room_state.get("known_frontier_cells"),
    )


def normalized_coverage(position_status: JsonDict, room_state: JsonDict, frontiers: list[str]) -> JsonDict:
    stats = as_dict(position_status.get("stats"))
    occupancy = as_dict(position_status.get("occupancy_summary"))
    return {
        "coverage_estimate": safe_float(
            room_state.get("coverage_estimate", position_status.get("coverage_estimate")),
            0.0,
        ),
        "visited_cell_count": safe_int(
            stats.get("visited_cell_count", occupancy.get("visited_cell_count", len(room_state.get("visited_cells") or []))),
            0,
        ),
        "free_cell_count": safe_int(stats.get("free_cell_count", occupancy.get("free_cell_count")), 0),
        "occupied_cell_count": safe_int(
            stats.get("occupied_cell_count", occupancy.get("occupied_cell_count")),
            0,
        ),
        "inflated_cell_count": safe_int(
            stats.get("inflated_cell_count", occupancy.get("inflated_cell_count")),
            0,
        ),
        "unknown_frontier_count": safe_int(
            stats.get("unknown_frontier_count", occupancy.get("unknown_frontier_count", len(frontiers))),
            len(frontiers),
        ),
        "collision_count": safe_int(
            stats.get("collision_count", room_state.get("collision_count", occupancy.get("collision_count"))),
            0,
        ),
    }


class ActionOdometryMapBackend:
    """Default read-only backend using the existing action-odometry map."""

    backend_name = "action_odometry"

    def __init__(self, memory_dir: Path | str | None = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else Path(__file__).resolve().parents[2] / "memory"
        self.position_map_path = self.memory_dir / "position-map.json"
        self.room_state_path = self.memory_dir / "room-state.json"
        self.position_map = PositionMap(self.memory_dir)

    def load_snapshot(self) -> MapSnapshot:
        raw_position = read_json(self.position_map_path)
        position_status = self.position_map.status()
        if raw_position:
            # Keep public decision behavior stable: existing decision code read
            # position-map.json directly, so explicit raw fields must win over
            # status-time defaults such as recomputed frontier cells.
            for key in (
                "room_name",
                "map_frame",
                "pose",
            ):
                if key in raw_position:
                    position_status[key] = raw_position[key]
            position_status["cells"] = raw_position.get("cells") or {}
            position_status["edges"] = raw_position.get("edges") or {}
            raw_frontiers = raw_position.get("frontiers", raw_position.get("frontier_cells", []))
            position_status["frontiers"] = list(raw_frontiers or [])
            position_status["frontier_cells"] = list(raw_frontiers or [])
            position_status["stats"] = raw_position.get("stats") or {}
            position_status["recent_actions"] = list(raw_position.get("recent_actions") or [])
        room_state = read_json(self.room_state_path)
        frontiers = normalized_frontiers(position_status, room_state)
        pose = normalized_pose(position_status)
        map_frame = normalized_map_frame(position_status, room_state)
        edges = normalized_edges(position_status, room_state)
        coverage = normalized_coverage(position_status, room_state, frontiers)
        active_frontier_goal = room_state.get("active_frontier_goal")
        if not isinstance(active_frontier_goal, dict):
            active_frontier_goal = None
        active_route = room_state.get("active_route")
        if not isinstance(active_route, dict):
            active_route = None
        frontier_history = room_state.get("frontier_history")
        if not isinstance(frontier_history, list):
            frontier_history = []
        frontier_history = [item for item in frontier_history if isinstance(item, dict)]
        frontier_cooldowns = room_state.get("frontier_cooldowns")
        if not isinstance(frontier_cooldowns, dict):
            frontier_cooldowns = {}
        return MapSnapshot(
            backend=self.backend_name,
            generated_at=now_iso(),
            map_frame=map_frame,
            pose=pose,
            coverage=coverage,
            frontiers=frontiers,
            edges=edges,
            active_frontier_goal=active_frontier_goal,
            active_route=active_route,
            frontier_history=frontier_history,
            frontier_cooldowns=frontier_cooldowns,
            position_status=position_status,
            room_state=room_state,
            source_paths={
                "position_map": str(self.position_map_path),
                "room_state": str(self.room_state_path),
            },
            source_info={
                "position_map": file_info(self.position_map_path),
                "room_state": file_info(self.room_state_path),
            },
        )
