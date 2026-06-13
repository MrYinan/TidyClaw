#!/usr/bin/env python3
"""MapBackend for a persisted map bundle.

The bundle backend lets tidy execution consume a frozen or curated map product
instead of rebuilding navigation state from scattered runtime memory files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from scripts.map_backend.base import (
        BackendUnavailableError,
        JsonDict,
        MapSnapshot,
        as_dict,
        as_list,
        now_iso,
    )
    from scripts.position_map_core import CELL_FREE, edge_key, four_neighbors, parse_cell
except ImportError:  # pragma: no cover - direct script execution
    from map_backend.base import (
        BackendUnavailableError,
        JsonDict,
        MapSnapshot,
        as_dict,
        as_list,
        now_iso,
    )
    from position_map_core import CELL_FREE, edge_key, four_neighbors, parse_cell


DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[2] / "memory" / "maps" / "current"


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
    info.update({"loaded": True, "size_bytes": int(stat.st_size), "mtime": float(stat.st_mtime)})
    return info


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


def sorted_cells(cells: list[str]) -> list[str]:
    def key(cell: str) -> tuple[int, int, str]:
        try:
            x, z = parse_cell(cell)
            return x, z, cell
        except Exception:
            return 0, 0, cell

    return sorted(cells, key=key)


def resolve_bundle_snapshot_path(memory_dir: Path, configured: str | None = None) -> Path:
    raw = configured or os.getenv("ROBOT_MAP_BUNDLE_PATH")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = memory_dir / "maps" / "current"
    if path.is_dir() or path.suffix.lower() != ".json":
        return path / "map-snapshot.json"
    return path


def live_pose(live_position: JsonDict, live_room: JsonDict, fallback_pose: JsonDict) -> JsonDict:
    pose = dict(as_dict(live_position.get("pose")) or fallback_pose)
    cell = str(pose.get("cell") or live_position.get("last_cell") or live_room.get("last_cell") or fallback_pose.get("cell") or "0,0")
    try:
        x, z = parse_cell(cell)
    except Exception:
        x, z, cell = 0, 0, "0,0"
    heading = str(pose.get("heading") or live_position.get("last_heading") or live_room.get("last_heading") or fallback_pose.get("heading") or "north")
    pose["cell"] = cell
    pose["x_cell"] = int(pose.get("x_cell", x) or x)
    pose["z_cell"] = int(pose.get("z_cell", z) or z)
    pose["heading"] = heading
    pose.setdefault("pose_confidence", fallback_pose.get("pose_confidence", fallback_pose.get("confidence", 1.0)))
    pose.setdefault("confidence", pose.get("pose_confidence"))
    pose.setdefault("position_uncertainty_cells", fallback_pose.get("position_uncertainty_cells", 0.0))
    pose.setdefault("heading_confidence", fallback_pose.get("heading_confidence", 1.0))
    pose["update_source"] = str(pose.get("update_source") or "live_position_overlay")
    return pose


def overlay_live_cells(bundle_cells: JsonDict, live_position: JsonDict, pose: JsonDict) -> JsonDict:
    cells = {str(cell): dict(as_dict(record)) for cell, record in as_dict(bundle_cells).items()}
    live_cells = as_dict(live_position.get("cells"))
    for cell, live_record in live_cells.items():
        text = str(cell or "").strip()
        if not text:
            continue
        live = as_dict(live_record)
        record = cells.setdefault(
            text,
            {
                "cell": text,
                "state": CELL_FREE,
                "occupancy_confidence": 0.5,
                "source": "live_position_overlay",
            },
        )
        for key in ("visited", "seen_count", "collision_count", "last_seen_step", "last_visit_step"):
            if key in live:
                record[key] = live[key]
        if live.get("visited") is True:
            record["visited"] = True
            record["state"] = CELL_FREE
            record["seen_count"] = max(1, int(record.get("seen_count", 0) or 0))
        if int(live.get("collision_count", 0) or 0) > 0:
            record["collision_count"] = int(live.get("collision_count", 0) or 0)
        cells[text] = record

    pose_cell = str(pose.get("cell") or "0,0")
    pose_record = cells.setdefault(
        pose_cell,
        {
            "cell": pose_cell,
            "state": CELL_FREE,
            "occupancy_confidence": 0.5,
            "source": "live_pose_overlay",
        },
    )
    pose_record["visited"] = True
    pose_record["seen_count"] = max(1, int(pose_record.get("seen_count", 0) or 0))
    pose_record["state"] = CELL_FREE
    return cells


def recompute_coverage_frontiers(cells: JsonDict, fallback_frontiers: list[str]) -> list[str]:
    visited = {str(cell) for cell, rec in cells.items() if as_dict(rec).get("visited") is True}
    if not visited:
        return list(fallback_frontiers)
    frontier_set: set[str] = set()
    for cell in visited:
        for neighbor in four_neighbors(cell):
            rec = as_dict(cells.get(neighbor))
            if rec and str(rec.get("state") or CELL_FREE) == CELL_FREE and rec.get("visited") is not True:
                frontier_set.add(neighbor)
    return sorted_cells(list(frontier_set))


def recompute_known_open_edges(cells: JsonDict, existing_edges: JsonDict) -> list[str]:
    cell_set = {
        str(cell)
        for cell, rec in cells.items()
        if str(as_dict(rec).get("state") or CELL_FREE) == CELL_FREE
    }
    blocked = set(unique_strings(existing_edges.get("blocked_edges"), existing_edges.get("hard_blocked_edges")))
    edges: list[str] = []
    for cell in sorted_cells(list(cell_set)):
        for neighbor in four_neighbors(cell):
            if neighbor not in cell_set:
                continue
            key = edge_key(cell, neighbor)
            if key not in blocked:
                edges.append(key)
    return edges


def snapshot_from_payload(
    payload: JsonDict,
    *,
    source_path: Path,
    memory_dir: Path | None = None,
) -> MapSnapshot:
    if str(payload.get("schema") or "") != "robot_cleaner_map_snapshot_v1":
        raise BackendUnavailableError(f"Map bundle snapshot has unsupported schema: {payload.get('schema')}")
    memory = Path(memory_dir) if memory_dir else source_path.parents[2]
    live_position = read_json(memory / "position-map.json")
    live_room = read_json(memory / "room-state.json")
    position_status = as_dict(payload.get("position_status"))
    room_state = as_dict(payload.get("room_state"))
    if not position_status:
        position_status = {
            "status": "success",
            "result_type": "position_map_status",
            "map_frame": as_dict(payload.get("map_frame")),
            "pose": as_dict(payload.get("pose")),
            "cells": {},
            "edges": as_dict(payload.get("edges")),
            "frontiers": as_list(payload.get("frontiers")),
            "frontier_cells": as_list(payload.get("frontiers")),
            "stats": as_dict(payload.get("coverage")),
        }
    if not room_state:
        room_state = {
            "last_cell": as_dict(payload.get("pose")).get("cell"),
            "last_heading": as_dict(payload.get("pose")).get("heading"),
            "frontier_cells": as_list(payload.get("frontiers")),
            "coverage_estimate": as_dict(payload.get("coverage")).get("coverage_estimate"),
        }
    live_active_goal = as_dict(live_room.get("active_frontier_goal")) or None
    live_active_route = as_dict(live_room.get("active_route")) or None
    if live_active_goal is None:
        live_active_route = None
    live_dynamic = {
        "active_frontier_goal": live_active_goal,
        "active_route": live_active_route,
        "frontier_history": [item for item in as_list(live_room.get("frontier_history")) if isinstance(item, dict)],
        "frontier_cooldowns": as_dict(live_room.get("frontier_cooldowns")),
        "coverage_patrol": as_dict(live_room.get("coverage_patrol")) or None,
    }
    pose = live_pose(live_position, live_room, as_dict(payload.get("pose")))
    cells = overlay_live_cells(as_dict(position_status.get("cells")), live_position, pose)
    fallback_frontiers = [str(item) for item in as_list(payload.get("frontiers")) if str(item).strip()]
    frontiers = recompute_coverage_frontiers(cells, fallback_frontiers)
    edges = as_dict(payload.get("edges"))
    live_edges = as_dict(live_position.get("edges"))
    blocked_edges = unique_strings(edges.get("blocked_edges"), live_edges.get("blocked_edges"), live_room.get("blocked_edges"))
    hard_blocked = unique_strings(edges.get("hard_blocked_edges"), live_edges.get("hard_blocked_edges"), live_room.get("hard_blocked_edges"))
    edges = {
        "known_open_edges": recompute_known_open_edges(cells, {"blocked_edges": blocked_edges, "hard_blocked_edges": hard_blocked}),
        "blocked_edges": blocked_edges,
        "hard_blocked_edges": hard_blocked,
    }
    visited_count = sum(1 for rec in cells.values() if as_dict(rec).get("visited") is True)
    free_count = sum(1 for rec in cells.values() if str(as_dict(rec).get("state") or CELL_FREE) == CELL_FREE)
    coverage = dict(as_dict(payload.get("coverage")))
    coverage.update(
        {
            "coverage_estimate": round(visited_count / float(max(1, free_count)), 4),
            "visited_cell_count": visited_count,
            "free_cell_count": free_count,
            "unknown_frontier_count": len(frontiers),
        }
    )
    position_status = dict(position_status)
    position_status.update(
        {
            "pose": pose,
            "last_cell": pose.get("cell"),
            "last_heading": pose.get("heading"),
            "cells": cells,
            "edges": edges,
            "frontiers": frontiers,
            "frontier_cells": frontiers,
            "coverage_estimate": coverage["coverage_estimate"],
            "stats": {**as_dict(position_status.get("stats")), **coverage},
            "recent_actions": as_list(live_position.get("recent_actions")) or as_list(position_status.get("recent_actions")),
            "active_frontier_goal": live_dynamic["active_frontier_goal"],
            "active_route": live_dynamic["active_route"],
            "frontier_history": live_dynamic["frontier_history"],
            "frontier_cooldowns": live_dynamic["frontier_cooldowns"],
            "map_bundle_overlay": {
                "live_position_path": str(memory / "position-map.json"),
                "live_room_path": str(memory / "room-state.json"),
                "pose_source": pose.get("update_source"),
                "static_map_source": str(source_path),
            },
        }
    )
    room_state = dict(room_state)
    room_state.update(
        {
            "last_cell": pose.get("cell"),
            "last_heading": pose.get("heading"),
            "frontier_cells": frontiers,
            "known_frontier_cells": frontiers,
            "coverage_estimate": coverage["coverage_estimate"],
            "visited_cells": sorted_cells(
                [str(cell) for cell, rec in cells.items() if as_dict(rec).get("visited") is True]
            ),
        }
    )
    room_state["active_frontier_goal"] = live_dynamic["active_frontier_goal"]
    room_state["active_route"] = live_dynamic["active_route"]
    room_state["frontier_history"] = live_dynamic["frontier_history"]
    room_state["frontier_cooldowns"] = live_dynamic["frontier_cooldowns"]
    if live_dynamic["coverage_patrol"] is not None:
        room_state["coverage_patrol"] = live_dynamic["coverage_patrol"]
    elif "coverage_patrol" in room_state:
        room_state.pop("coverage_patrol", None)
    source_info = as_dict(payload.get("source_info"))
    source_info["map_bundle"] = file_info(source_path)
    source_info["live_position_overlay"] = file_info(memory / "position-map.json")
    source_info["live_room_overlay"] = file_info(memory / "room-state.json")
    source_paths = as_dict(payload.get("source_paths"))
    source_paths["map_bundle"] = str(source_path)
    source_paths["live_position_overlay"] = str(memory / "position-map.json")
    source_paths["live_room_overlay"] = str(memory / "room-state.json")
    return MapSnapshot(
        backend="map_bundle",
        generated_at=now_iso(),
        map_frame=as_dict(payload.get("map_frame")),
        pose=pose,
        coverage=coverage,
        frontiers=frontiers,
        edges=edges,
        active_frontier_goal=live_dynamic["active_frontier_goal"],
        active_route=live_dynamic["active_route"],
        frontier_history=live_dynamic["frontier_history"],
        frontier_cooldowns=live_dynamic["frontier_cooldowns"],
        position_status=position_status,
        room_state=room_state,
        source_paths=source_paths,
        source_info=source_info,
    )


class MapBundleBackend:
    """Read a persisted map bundle from disk."""

    backend_name = "map_bundle"

    def __init__(self, memory_dir: Path | str | None = None, *, bundle_path: str | Path | None = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else Path(__file__).resolve().parents[2] / "memory"
        self.snapshot_path = resolve_bundle_snapshot_path(self.memory_dir, str(bundle_path) if bundle_path else None)

    def load_snapshot(self) -> MapSnapshot:
        payload = read_json(self.snapshot_path)
        if not payload:
            raise BackendUnavailableError(f"Map bundle snapshot is missing or invalid: {self.snapshot_path}")
        return snapshot_from_payload(payload, source_path=self.snapshot_path, memory_dir=self.memory_dir)
