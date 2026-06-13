#!/usr/bin/env python3
"""Read-only map backend contract for public robot decision state."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


JsonDict = dict[str, Any]
MAP_SNAPSHOT_SCHEMA = "robot_cleaner_map_snapshot_v1"


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_cell(value: Any) -> tuple[int, int] | None:
    try:
        left, right = str(value or "").split(",", 1)
        return int(left), int(right)
    except (TypeError, ValueError):
        return None


def unique_strings(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in as_list(value):
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
    return result


class BackendUnavailableError(RuntimeError):
    """Raised when a configured map backend cannot be constructed."""


class MapBackend(Protocol):
    """Read-only map backend contract.

    Backends may read simulator, SLAM, or action-odometry state.  They must
    return the same MapSnapshot shape so decision code is backend-replaceable.
    """

    backend_name: str
    memory_dir: Path

    def load_snapshot(self) -> "MapSnapshot":
        ...


@dataclass(frozen=True)
class MapSnapshot:
    """Normalized navigation-map projection for agent-facing decision code."""

    backend: str
    generated_at: str
    map_frame: JsonDict = field(default_factory=dict)
    pose: JsonDict = field(default_factory=dict)
    coverage: JsonDict = field(default_factory=dict)
    frontiers: list[str] = field(default_factory=list)
    edges: JsonDict = field(default_factory=dict)
    active_frontier_goal: JsonDict | None = None
    active_route: JsonDict | None = None
    frontier_history: list[JsonDict] = field(default_factory=list)
    frontier_cooldowns: JsonDict = field(default_factory=dict)
    position_status: JsonDict = field(default_factory=dict)
    room_state: JsonDict = field(default_factory=dict)
    source_paths: JsonDict = field(default_factory=dict)
    source_info: JsonDict = field(default_factory=dict)

    @property
    def schema(self) -> str:
        return MAP_SNAPSHOT_SCHEMA

    def to_dict(self, *, include_raw: bool = False) -> JsonDict:
        payload: JsonDict = {
            "schema": self.schema,
            "backend": self.backend,
            "generated_at": self.generated_at,
            "map_frame": copy.deepcopy(self.map_frame),
            "pose": copy.deepcopy(self.pose),
            "coverage": copy.deepcopy(self.coverage),
            "frontiers": list(self.frontiers),
            "edges": copy.deepcopy(self.edges),
            "active_frontier_goal": copy.deepcopy(self.active_frontier_goal),
            "active_route": copy.deepcopy(self.active_route),
            "frontier_history": copy.deepcopy(self.frontier_history),
            "frontier_cooldowns": copy.deepcopy(self.frontier_cooldowns),
            "source_paths": copy.deepcopy(self.source_paths),
            "source_info": copy.deepcopy(self.source_info),
        }
        if include_raw:
            payload["position_status"] = self.to_position_status()
            payload["room_state"] = self.to_navigation_room_state()
        return payload

    def public_summary(self) -> JsonDict:
        return {
            "schema": self.schema,
            "backend": self.backend,
            "generated_at": self.generated_at,
            "cell_size_m": self.map_frame.get("cell_size_m"),
            "coordinate_mode": self.map_frame.get("coordinate_mode"),
            "pose_confidence": self.pose.get("pose_confidence", self.pose.get("confidence")),
            "position_uncertainty_cells": self.pose.get("position_uncertainty_cells"),
            "heading_confidence": self.pose.get("heading_confidence"),
            "frontier_count": len(self.frontiers),
            "coverage_estimate": self.coverage.get("coverage_estimate"),
        }

    def source_loaded_info(self) -> JsonDict:
        return {
            "loaded": True,
            "backend": self.backend,
            "schema": self.schema,
            "source_paths": copy.deepcopy(self.source_paths),
            "sources": copy.deepcopy(self.source_info),
        }

    def to_position_status(self) -> JsonDict:
        status = copy.deepcopy(self.position_status)
        status.setdefault("status", "success")
        status.setdefault("result_type", "position_map_status")
        status["map_backend"] = self.backend
        status["map_snapshot_schema"] = self.schema
        status.setdefault("map_frame", copy.deepcopy(self.map_frame))
        status.setdefault("pose", copy.deepcopy(self.pose))
        status.setdefault("last_cell", self.pose.get("cell"))
        status.setdefault("last_heading", self.pose.get("heading"))
        status.setdefault("frontiers", list(self.frontiers))
        status.setdefault("frontier_cells", list(self.frontiers))
        status.setdefault("edges", copy.deepcopy(self.edges))
        status.setdefault("coverage_estimate", self.coverage.get("coverage_estimate"))
        if "stats" not in status:
            status["stats"] = {
                key: value
                for key, value in self.coverage.items()
                if key.endswith("_count") or key == "coverage_estimate"
            }
        status.setdefault("active_frontier_goal", copy.deepcopy(self.active_frontier_goal))
        status.setdefault("active_route", copy.deepcopy(self.active_route))
        status.setdefault("frontier_history", copy.deepcopy(self.frontier_history))
        status.setdefault("frontier_cooldowns", copy.deepcopy(self.frontier_cooldowns))
        coverage_patrol = self.room_state.get("coverage_patrol") if isinstance(self.room_state, dict) else None
        if isinstance(coverage_patrol, dict):
            status.setdefault("coverage_patrol", copy.deepcopy(coverage_patrol))
        return status

    def to_navigation_room_state(self) -> JsonDict:
        room = copy.deepcopy(self.room_state)
        compat = as_dict(self.position_status.get("room_state_compat"))
        for key, value in compat.items():
            room.setdefault(key, copy.deepcopy(value))
        room.setdefault("last_cell", self.pose.get("cell"))
        room.setdefault("last_heading", self.pose.get("heading"))
        room.setdefault("frontier_cells", list(self.frontiers))
        room.setdefault("known_frontier_cells", list(self.frontiers))
        room.setdefault("coverage_estimate", self.coverage.get("coverage_estimate"))
        room.setdefault("active_frontier_goal", copy.deepcopy(self.active_frontier_goal))
        room.setdefault("active_route", copy.deepcopy(self.active_route))
        room.setdefault("frontier_history", copy.deepcopy(self.frontier_history))
        room.setdefault("frontier_cooldowns", copy.deepcopy(self.frontier_cooldowns))
        edges = as_dict(self.edges)
        room.setdefault("known_open_edges", unique_strings(edges.get("known_open_edges")))
        room.setdefault("blocked_edges", unique_strings(edges.get("blocked_edges")))
        room.setdefault("hard_blocked_edges", unique_strings(edges.get("hard_blocked_edges")))
        return room
