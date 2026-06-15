#!/usr/bin/env python3
"""AI2-THOR ground-truth map backend for simulator mapping stages.

This backend is deliberately offline/simulator-specific.  It consumes
``/eval/map`` or a cached ``ai2thor-groundtruth-map.json`` payload and projects
AI2-THOR reachable positions into the same MapSnapshot contract used by the
online action-odometry backend.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from scripts.map_backend.action_odometry import file_info, read_json
    from scripts.map_backend.base import (
        BackendUnavailableError,
        JsonDict,
        MapSnapshot,
        as_dict,
        as_list,
        now_iso,
        safe_float,
        unique_strings,
    )
    from scripts.position_map_core import (
        DEFAULT_CELL_SIZE_M,
        CELL_FREE,
        edge_key,
        format_cell,
        four_neighbors,
        heading_from_rotation,
        parse_cell,
    )
except ImportError:  # pragma: no cover - direct script execution
    from map_backend.action_odometry import file_info, read_json
    from map_backend.base import (
        BackendUnavailableError,
        JsonDict,
        MapSnapshot,
        as_dict,
        as_list,
        now_iso,
        safe_float,
        unique_strings,
    )
    from position_map_core import (
        DEFAULT_CELL_SIZE_M,
        CELL_FREE,
        edge_key,
        format_cell,
        four_neighbors,
        heading_from_rotation,
        parse_cell,
    )


GROUNDTRUTH_CACHE_NAME = "ai2thor-groundtruth-map.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:5000/eval/map"


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


def normalize_source_mode(value: Any) -> str:
    mode = str(value or os.getenv("ROBOT_AI2THOR_MAP_SOURCE", "auto")).strip().lower()
    if mode in {"cache", "cached", "file"}:
        return "cache"
    if mode in {"live", "server", "endpoint", "http"}:
        return "live"
    return "auto"


def backend_endpoint() -> str:
    explicit = os.getenv("ROBOT_AI2THOR_MAP_URL")
    if explicit:
        return explicit
    base = os.getenv("ROBOT_BACKEND_URL")
    if base:
        return base.rstrip("/") + "/eval/map"
    return DEFAULT_ENDPOINT


def fetch_json(url: str, *, timeout_s: float) -> JsonDict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=max(0.2, float(timeout_s))) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BackendUnavailableError(f"AI2-THOR ground-truth endpoint unavailable: {url}: {exc}") from exc
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError(f"AI2-THOR ground-truth endpoint returned invalid JSON: {url}") from exc
    if not isinstance(data, dict):
        raise BackendUnavailableError(f"AI2-THOR ground-truth endpoint returned non-object JSON: {url}")
    return data


def cell_from_world(position: JsonDict, cell_size_m: float) -> str:
    x = round(float(position.get("x", 0.0) or 0.0) / cell_size_m)
    z = round(float(position.get("z", 0.0) or 0.0) / cell_size_m)
    return format_cell(int(x), int(z))


def reachable_cell_records(positions: list[Any], *, cell_size_m: float) -> dict[str, JsonDict]:
    cells: dict[str, JsonDict] = {}
    for item in positions:
        pos = as_dict(item)
        if not pos:
            continue
        try:
            cell = cell_from_world(pos, cell_size_m)
            x_cell, z_cell = parse_cell(cell)
            world = {
                "x": round(float(pos.get("x", 0.0) or 0.0), 4),
                "y": round(float(pos.get("y", 0.0) or 0.0), 4),
                "z": round(float(pos.get("z", 0.0) or 0.0), 4),
            }
        except (TypeError, ValueError):
            continue
        previous = cells.get(cell)
        if previous is not None:
            previous.setdefault("world_positions", []).append(world)
            continue
        cells[cell] = {
            "cell": cell,
            "x": x_cell,
            "z": z_cell,
            "state": CELL_FREE,
            "visited": False,
            "seen_count": 0,
            "collision_count": 0,
            "occupancy_confidence": 1.0,
            "free_evidence": 1,
            "occupied_evidence": 0,
            "source": "ai2thor_reachable_position",
            "world_position": world,
            "world_positions": [world],
        }
    return cells


def existing_visited_cells(position_map: JsonDict, room_state: JsonDict) -> set[str]:
    visited: set[str] = set()
    # In groundtruth mode, legacy position-map/action-odometry cells are not an
    # authoritative trajectory. Only room-state visited cells that were written
    # by this backend are reused across turns.
    if str(room_state.get("map_backend") or "") != "ai2thor_groundtruth":
        return visited
    for cell in as_list(room_state.get("visited_cells")):
        text = str(cell or "").strip()
        if text:
            visited.add(text)
    return visited


def build_frontiers(cells: dict[str, JsonDict], visited: set[str]) -> list[str]:
    if not cells:
        return []
    frontier_set: set[str] = set()
    for cell in visited:
        if cell not in cells:
            continue
        for neighbor in four_neighbors(cell):
            if neighbor in cells and neighbor not in visited:
                frontier_set.add(neighbor)
    if frontier_set:
        return sorted(frontier_set, key=lambda item: parse_cell(item))
    # Cold start: expose the whole known traversable map as coverage frontier.
    return sorted(cells.keys(), key=lambda item: parse_cell(item))[:32]


def known_open_edges(cells: dict[str, JsonDict]) -> list[str]:
    cell_set = set(cells)
    edges: list[str] = []
    for cell in sorted(cell_set, key=lambda item: parse_cell(item)):
        for neighbor in four_neighbors(cell):
            if neighbor not in cell_set:
                continue
            edges.append(edge_key(cell, neighbor))
    return edges


def normalize_payload(raw: JsonDict) -> JsonDict:
    if isinstance(raw.get("groundtruth_map"), dict):
        return as_dict(raw.get("groundtruth_map"))
    if isinstance(raw.get("ai2thor_groundtruth"), dict):
        return as_dict(raw.get("ai2thor_groundtruth"))
    return raw


class AI2ThorGroundTruthMapBackend:
    """MapBackend backed by AI2-THOR reachable positions."""

    backend_name = "ai2thor_groundtruth"

    def __init__(
        self,
        memory_dir: Path | str | None = None,
        *,
        endpoint: str | None = None,
        source_mode: str | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else Path(__file__).resolve().parents[2] / "memory"
        self.cache_path = self.memory_dir / GROUNDTRUTH_CACHE_NAME
        self.position_map_path = self.memory_dir / "position-map.json"
        self.room_state_path = self.memory_dir / "room-state.json"
        self.endpoint = endpoint or backend_endpoint()
        self.source_mode = normalize_source_mode(source_mode)
        self.timeout_s = env_float("ROBOT_AI2THOR_MAP_TIMEOUT_S", 3.0)

    def _load_raw_payload(self) -> tuple[JsonDict, JsonDict]:
        cache_info = file_info(self.cache_path)
        if self.source_mode in {"auto", "cache"} and self.cache_path.exists():
            raw = read_json(self.cache_path)
            if raw:
                return raw, {"source": "cache", "cache": cache_info, "endpoint": self.endpoint}
        if self.source_mode in {"auto", "live"}:
            raw = fetch_json(self.endpoint, timeout_s=self.timeout_s)
            return raw, {"source": "live", "cache": cache_info, "endpoint": self.endpoint}
        raise BackendUnavailableError(
            f"AI2-THOR ground-truth map source unavailable: source_mode={self.source_mode}, cache={self.cache_path}"
        )

    def load_snapshot(self) -> MapSnapshot:
        raw, source = self._load_raw_payload()
        payload = normalize_payload(raw)
        if str(payload.get("status") or "success") != "success":
            raise BackendUnavailableError(
                f"AI2-THOR ground-truth map payload is not successful: {payload.get('result_type')}"
            )

        position_map = read_json(self.position_map_path)
        room_state = read_json(self.room_state_path)
        cell_size_m = safe_float(payload.get("grid_size_m"), DEFAULT_CELL_SIZE_M) or DEFAULT_CELL_SIZE_M
        cells = reachable_cell_records(as_list(payload.get("reachable_positions")), cell_size_m=cell_size_m)
        if not cells:
            raise BackendUnavailableError("AI2-THOR ground-truth map has no reachable positions.")

        robot = as_dict(payload.get("robot"))
        robot_position = as_dict(robot.get("position"))
        robot_rotation = as_dict(robot.get("rotation"))
        pose_cell = cell_from_world(robot_position, cell_size_m) if robot_position else "0,0"
        if pose_cell not in cells:
            x_cell, z_cell = parse_cell(pose_cell)
            cells[pose_cell] = {
                "cell": pose_cell,
                "x": x_cell,
                "z": z_cell,
                "state": CELL_FREE,
                "visited": True,
                "seen_count": 1,
                "collision_count": 0,
                "occupancy_confidence": 1.0,
                "free_evidence": 1,
                "occupied_evidence": 0,
                "source": "ai2thor_agent_pose",
            }

        heading = heading_from_rotation(float(robot_rotation.get("y", 0.0) or 0.0))
        x_cell, z_cell = parse_cell(pose_cell)
        visited = existing_visited_cells(position_map, room_state)
        visited.add(pose_cell)
        for cell, rec in cells.items():
            if cell in visited:
                rec["visited"] = True
                rec["seen_count"] = max(1, int(rec.get("seen_count", 0) or 0))
        visited_in_map = {cell for cell in visited if cell in cells}
        frontiers = build_frontiers(cells, visited_in_map)
        open_edges = known_open_edges(cells)
        room_is_groundtruth = str(room_state.get("map_backend") or "") == self.backend_name
        blocked_edges = unique_strings(room_state.get("blocked_edges")) if room_is_groundtruth else []
        hard_blocked = unique_strings(room_state.get("hard_blocked_edges")) if room_is_groundtruth else []
        blocked_set = set(blocked_edges) | set(hard_blocked)
        open_edges = [edge for edge in open_edges if edge not in blocked_set]
        free_count = len(cells)
        visited_count = len(visited_in_map)
        coverage_estimate = round(visited_count / float(max(1, free_count)), 4)
        map_frame = {
            "coordinate_mode": "ai2thor_groundtruth_grid",
            "cell_size_m": cell_size_m,
            "origin_cell": "0,0",
            "axis": {"x_cell_positive": "east", "z_cell_positive": "north"},
            "source": "ai2thor_reachable_positions",
        }
        pose = {
            "cell": pose_cell,
            "x_cell": x_cell,
            "z_cell": z_cell,
            "heading": heading,
            "theta_deg": float(robot_rotation.get("y", 0.0) or 0.0) % 360.0,
            "pose_confidence": 1.0,
            "confidence": 1.0,
            "position_uncertainty_cells": 0.0,
            "heading_confidence": 1.0,
            "update_source": "ai2thor_groundtruth",
            "camera_horizon": robot.get("cameraHorizon"),
        }
        coverage = {
            "coverage_estimate": coverage_estimate,
            "visited_cell_count": visited_count,
            "free_cell_count": free_count,
            "occupied_cell_count": 0,
            "inflated_cell_count": 0,
            "unknown_frontier_count": len(frontiers),
            "collision_count": int(as_dict(position_map.get("stats")).get("collision_count", room_state.get("collision_count", 0)) or 0),
        }
        position_status = {
            "status": "success",
            "result_type": "position_map_status",
            "room_name": payload.get("scene") or position_map.get("room_name") or "ai2thor_scene",
            "map_frame": map_frame,
            "pose": pose,
            "last_cell": pose_cell,
            "last_heading": heading,
            "cells": cells,
            "edges": {
                "known_open_edges": open_edges,
                "blocked_edges": blocked_edges,
                "hard_blocked_edges": hard_blocked,
            },
            "frontiers": frontiers,
            "frontier_cells": frontiers,
            "coverage_estimate": coverage_estimate,
            "stats": coverage,
            "recent_actions": as_list(position_map.get("recent_actions"))[-24:],
            "map_contract": {
                "backend": self.backend_name,
                "source": "ai2thor_reachable_positions",
                "online_safe": False,
                "usage_scope": "offline_mapping_or_simulator_backend_only",
            },
        }
        merged_room = dict(room_state)
        merged_room.setdefault("room_name", position_status["room_name"])
        merged_room["last_cell"] = pose_cell
        merged_room["last_heading"] = heading
        merged_room["frontier_cells"] = frontiers
        merged_room["known_frontier_cells"] = frontiers
        merged_room["coverage_estimate"] = coverage_estimate
        merged_room["visited_cells"] = sorted(visited_in_map, key=lambda item: parse_cell(item))
        return MapSnapshot(
            backend=self.backend_name,
            generated_at=now_iso(),
            map_frame=map_frame,
            pose=pose,
            coverage=coverage,
            frontiers=frontiers,
            edges=position_status["edges"],
            active_frontier_goal=as_dict(room_state.get("active_frontier_goal")) or None,
            active_route=as_dict(room_state.get("active_route")) or None,
            frontier_history=[item for item in as_list(room_state.get("frontier_history")) if isinstance(item, dict)],
            frontier_cooldowns=as_dict(room_state.get("frontier_cooldowns")),
            position_status=position_status,
            room_state=merged_room,
            source_paths={
                "groundtruth_cache": str(self.cache_path),
                "position_map_overlay": str(self.position_map_path),
                "room_state_overlay": str(self.room_state_path),
            },
            source_info={
                "groundtruth": source,
                "position_map_overlay": file_info(self.position_map_path),
                "room_state_overlay": file_info(self.room_state_path),
            },
        )
