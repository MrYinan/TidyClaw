#!/usr/bin/env python3
"""Persist the configured MapBackend snapshot as the public navigation state.

Legacy action-odometry files may still exist for fallback/debug, but OpenClaw
agent-facing room-state fields should reflect the configured authoritative map
backend, currently AI2-THOR groundtruth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from scripts.map_backend import BackendUnavailableError, load_map_backend
    from scripts.map_backend.base import MapSnapshot, as_dict, as_list
except ImportError:  # pragma: no cover - direct script execution
    from map_backend import BackendUnavailableError, load_map_backend
    from map_backend.base import MapSnapshot, as_dict, as_list


JsonDict = dict[str, Any]


def read_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _room_patch_from_snapshot(snapshot: MapSnapshot) -> JsonDict:
    room_projection = snapshot.to_navigation_room_state()
    pose = as_dict(snapshot.pose)
    coverage = as_dict(snapshot.coverage)
    edges = as_dict(snapshot.edges)
    return {
        "map_backend": snapshot.backend,
        "map_snapshot_schema": snapshot.schema,
        "map_frame": as_dict(snapshot.map_frame),
        "last_cell": pose.get("cell"),
        "last_heading": pose.get("heading"),
        "pose": pose,
        "pose_confidence": pose.get("pose_confidence", pose.get("confidence")),
        "position_uncertainty_cells": pose.get("position_uncertainty_cells"),
        "heading_confidence": pose.get("heading_confidence"),
        "coverage_estimate": coverage.get("coverage_estimate"),
        "visited_cells": as_list(room_projection.get("visited_cells")),
        "frontier_cells": list(snapshot.frontiers),
        "known_frontier_cells": list(snapshot.frontiers),
        "known_open_edges": as_list(room_projection.get("known_open_edges")),
        "blocked_edges": as_list(edges.get("blocked_edges")),
        "hard_blocked_edges": as_list(edges.get("hard_blocked_edges")),
        "active_frontier_goal": as_dict(snapshot.active_frontier_goal),
        "active_route": as_dict(snapshot.active_route),
        "frontier_history": as_list(snapshot.frontier_history),
        "frontier_cooldowns": as_dict(snapshot.frontier_cooldowns),
        "authoritative_navigation": {
            "backend": snapshot.backend,
            "generated_at": snapshot.generated_at,
            "coordinate_mode": as_dict(snapshot.map_frame).get("coordinate_mode"),
            "last_cell": pose.get("cell"),
            "last_heading": pose.get("heading"),
            "coverage_estimate": coverage.get("coverage_estimate"),
            "frontier_count": len(snapshot.frontiers),
            "visited_cell_count": coverage.get("visited_cell_count"),
            "source": "map_backend_snapshot",
        },
    }


def _unique_strings(*values: Any) -> list[str]:
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


def _last_blocked_edges(room: JsonDict) -> list[str]:
    last_blocked = as_dict(room.get("last_blocked_edge"))
    return _unique_strings(last_blocked.get("edge"), last_blocked.get("reverse_edge"))


def _merge_public_blocked_edges(room: JsonDict, patch: JsonDict) -> None:
    durable_hard_edges = _unique_strings(
        room.get("hard_blocked_edges"),
        patch.get("hard_blocked_edges"),
        _last_blocked_edges(room),
    )
    durable_blocked_edges = _unique_strings(
        room.get("blocked_edges"),
        patch.get("blocked_edges"),
        durable_hard_edges,
    )
    patch["blocked_edges"] = durable_blocked_edges
    patch["hard_blocked_edges"] = durable_hard_edges


def sync_authoritative_room_state(
    memory_dir: Path | str,
    *,
    odometry_debug: JsonDict | None = None,
) -> JsonDict:
    """Overwrite public room-state navigation fields from MapBackend snapshot."""

    memory = Path(memory_dir)
    try:
        snapshot = load_map_backend(memory).load_snapshot()
    except BackendUnavailableError as exc:
        return {
            "status": "error",
            "result_type": "authoritative_map_sync_failed",
            "message": str(exc),
        }

    room_path = memory / "room-state.json"
    room = read_json(room_path)
    patch = _room_patch_from_snapshot(snapshot)
    _merge_public_blocked_edges(room, patch)
    room.update({key: value for key, value in patch.items() if value is not None})
    room["position_map_status"] = "debug_fallback"
    room["position_map_note"] = (
        "position-map/action-odometry is retained only as fallback/debug; "
        "public navigation uses map_backend snapshot."
    )
    if odometry_debug:
        debug = as_dict(room.get("debug"))
        debug["action_odometry"] = odometry_debug
        room["debug"] = debug

    atomic_write_json(room_path, room)

    pose = as_dict(snapshot.pose)
    coverage = as_dict(snapshot.coverage)
    return {
        "status": "success",
        "result_type": "authoritative_map_room_state_synced",
        "map_backend": snapshot.backend,
        "coordinate_mode": as_dict(snapshot.map_frame).get("coordinate_mode"),
        "last_cell": pose.get("cell"),
        "last_heading": pose.get("heading"),
        "coverage_estimate": coverage.get("coverage_estimate"),
        "visited_cell_count": coverage.get("visited_cell_count"),
        "frontier_count": len(snapshot.frontiers),
        "room_state_path": str(room_path),
    }
