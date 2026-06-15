#!/usr/bin/env python3
"""Persistent coverage waypoint state for room patrol.

The state lives under ``room-state.json`` as ``coverage_waypoints``.  It is the
coverage checklist that makes inspection waypoint patrol stable across turns:
required waypoints come from the map, while observed/blocked/active status is
runtime state.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from scripts.inspection_waypoints import INSPECTION_WAYPOINTS_SCHEMA, manhattan, waypoint_by_id
    from scripts.map_backend.base import JsonDict, as_dict, as_list
except ImportError:  # pragma: no cover - direct script execution
    from inspection_waypoints import INSPECTION_WAYPOINTS_SCHEMA, manhattan, waypoint_by_id
    from map_backend.base import JsonDict, as_dict, as_list


COVERAGE_WAYPOINT_STATE_SCHEMA = "robot_cleaner_coverage_waypoint_state_v1"
DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[1] / "memory"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> JsonDict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        last_error: PermissionError | None = None
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
            except OSError:
                pass


def unique_strings(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    if values is None:
        return result
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _waypoint_public_payload(item: Mapping[str, Any]) -> JsonDict:
    keys = (
        "waypoint_id",
        "cell",
        "x_cell",
        "z_cell",
        "label",
        "purpose",
        "waypoint_source",
        "coverage_radius_cells",
        "coverage_estimate",
        "component_id",
        "component_index",
        "component_size",
        "covered_cell_count",
        "map_visited",
        "world_position",
    )
    return {key: item[key] for key in keys if key in item}


def _status_template(waypoint: Mapping[str, Any]) -> JsonDict:
    return {
        "waypoint_id": str(waypoint.get("waypoint_id") or ""),
        "cell": str(waypoint.get("cell") or ""),
        "status": "pending",
        "attempt_count": 0,
        "last_distance_cells": None,
    }


def normalize_coverage_waypoint_state(
    waypoint_set: Mapping[str, Any],
    existing: Mapping[str, Any] | None = None,
    *,
    current_cell: str | None = None,
) -> JsonDict:
    if str(waypoint_set.get("schema") or "") != INSPECTION_WAYPOINTS_SCHEMA:
        raise ValueError(f"waypoint_set must use schema {INSPECTION_WAYPOINTS_SCHEMA}")

    existing_state = as_dict(existing)
    waypoints = waypoint_by_id(waypoint_set)
    required_ids = [
        waypoint_id
        for waypoint_id in unique_strings(waypoint_set.get("required_waypoint_ids") or waypoints.keys())
        if waypoint_id in waypoints
    ]
    required_set = set(required_ids)
    observed_ids = [item for item in unique_strings(existing_state.get("observed_waypoint_ids")) if item in required_set]
    blocked_ids = [
        item
        for item in unique_strings(existing_state.get("blocked_waypoint_ids"))
        if item in required_set and item not in observed_ids
    ]
    waypoint_status = {
        waypoint_id: _status_template(waypoints[waypoint_id])
        for waypoint_id in required_ids
    }
    previous_status = as_dict(existing_state.get("waypoint_status"))
    for waypoint_id, previous in previous_status.items():
        if waypoint_id not in waypoint_status or not isinstance(previous, Mapping):
            continue
        waypoint_status[waypoint_id].update(
            {
                key: previous[key]
                for key in (
                    "attempt_count",
                    "selected_at",
                    "last_selected_at",
                    "observed_at",
                    "blocked_at",
                    "blocked_reason",
                    "last_observation_id",
                    "last_distance_cells",
                    "last_result",
                )
                if key in previous
            }
        )
    for waypoint_id in observed_ids:
        waypoint_status[waypoint_id]["status"] = "observed"
    for waypoint_id in blocked_ids:
        waypoint_status[waypoint_id]["status"] = "blocked"
    if current_cell:
        for waypoint_id in required_ids:
            cell = str(waypoints[waypoint_id].get("cell") or "")
            waypoint_status[waypoint_id]["last_distance_cells"] = manhattan(current_cell, cell) if cell else None

    active_goal = as_dict(existing_state.get("active_waypoint_goal"))
    active_id = str(active_goal.get("waypoint_id") or "").strip()
    if active_id not in required_set or active_id in observed_ids or active_id in blocked_ids:
        active_goal = None
    else:
        active_goal = dict(active_goal)
        active_goal.setdefault("status", "active")
        active_goal.setdefault("cell", waypoints[active_id].get("cell"))
        active_goal.setdefault("waypoint_source", waypoints[active_id].get("waypoint_source"))

    completed_count = len(set(observed_ids) | set(blocked_ids))
    required_count = len(required_ids)
    pending_ids = [
        waypoint_id
        for waypoint_id in required_ids
        if waypoint_id not in observed_ids and waypoint_id not in blocked_ids
    ]
    next_pending = sorted(
        (
            dict(_waypoint_public_payload(waypoints[waypoint_id]), **waypoint_status[waypoint_id])
            for waypoint_id in pending_ids
        ),
        key=lambda item: (
            item.get("last_distance_cells") is None,
            item.get("last_distance_cells") if item.get("last_distance_cells") is not None else 10**9,
            str(item.get("waypoint_id") or ""),
        ),
    )
    return {
        "schema": COVERAGE_WAYPOINT_STATE_SCHEMA,
        "updated_at": now_iso(),
        "waypoint_set_schema": waypoint_set.get("schema"),
        "waypoint_source": waypoint_set.get("source"),
        "map_backend": waypoint_set.get("map_backend"),
        "coverage_radius_cells": waypoint_set.get("coverage_radius_cells"),
        "required_waypoint_ids": required_ids,
        "required_waypoint_count": required_count,
        "required_waypoints": [_waypoint_public_payload(waypoints[waypoint_id]) for waypoint_id in required_ids],
        "observed_waypoint_ids": observed_ids,
        "observed_waypoint_count": len(observed_ids),
        "blocked_waypoint_ids": blocked_ids,
        "blocked_waypoint_count": len(blocked_ids),
        "pending_waypoint_ids": pending_ids,
        "pending_waypoint_count": len(pending_ids),
        "completed_waypoint_count": completed_count,
        "sweep_coverage_rate": round(completed_count / float(max(1, required_count)), 6),
        "active_waypoint_goal": active_goal,
        "waypoint_status": waypoint_status,
        "next_unobserved_waypoints": next_pending[:8],
    }


def mark_waypoint_observed(
    state: Mapping[str, Any],
    waypoint_id: str,
    *,
    observation_id: str | None = None,
    observed_at: str | None = None,
) -> JsonDict:
    updated = dict(as_dict(state))
    required = set(unique_strings(updated.get("required_waypoint_ids")))
    waypoint = str(waypoint_id or "").strip()
    if waypoint not in required:
        raise ValueError(f"Unknown coverage waypoint_id: {waypoint}")
    observed = unique_strings(list(updated.get("observed_waypoint_ids") or []) + [waypoint])
    blocked = [item for item in unique_strings(updated.get("blocked_waypoint_ids")) if item != waypoint]
    status = {key: dict(as_dict(value)) for key, value in as_dict(updated.get("waypoint_status")).items()}
    item = status.setdefault(waypoint, {"waypoint_id": waypoint})
    item.update(
        {
            "status": "observed",
            "observed_at": observed_at or now_iso(),
            "last_result": "observed",
        }
    )
    if observation_id:
        item["last_observation_id"] = observation_id
    active = as_dict(updated.get("active_waypoint_goal"))
    updated["active_waypoint_goal"] = None if str(active.get("waypoint_id") or "") == waypoint else active or None
    updated["observed_waypoint_ids"] = observed
    updated["blocked_waypoint_ids"] = blocked
    updated["waypoint_status"] = status
    return _recompute_counts(updated)


def mark_waypoint_blocked(
    state: Mapping[str, Any],
    waypoint_id: str,
    *,
    reason: str = "blocked",
    blocked_at: str | None = None,
) -> JsonDict:
    updated = dict(as_dict(state))
    required = set(unique_strings(updated.get("required_waypoint_ids")))
    waypoint = str(waypoint_id or "").strip()
    if waypoint not in required:
        raise ValueError(f"Unknown coverage waypoint_id: {waypoint}")
    if waypoint not in set(unique_strings(updated.get("observed_waypoint_ids"))):
        updated["blocked_waypoint_ids"] = unique_strings(list(updated.get("blocked_waypoint_ids") or []) + [waypoint])
    status = {key: dict(as_dict(value)) for key, value in as_dict(updated.get("waypoint_status")).items()}
    item = status.setdefault(waypoint, {"waypoint_id": waypoint})
    item.update(
        {
            "status": "blocked",
            "blocked_at": blocked_at or now_iso(),
            "blocked_reason": reason,
            "last_result": "blocked",
        }
    )
    active = as_dict(updated.get("active_waypoint_goal"))
    updated["active_waypoint_goal"] = None if str(active.get("waypoint_id") or "") == waypoint else active or None
    updated["waypoint_status"] = status
    return _recompute_counts(updated)


def set_active_waypoint_goal(
    state: Mapping[str, Any],
    waypoint_id: str,
    *,
    selected_at: str | None = None,
) -> JsonDict:
    updated = dict(as_dict(state))
    waypoint = str(waypoint_id or "").strip()
    waypoints = {
        str(item.get("waypoint_id") or ""): dict(item)
        for item in as_list(updated.get("required_waypoints"))
        if isinstance(item, Mapping)
    }
    if waypoint not in set(unique_strings(updated.get("required_waypoint_ids"))) or waypoint not in waypoints:
        raise ValueError(f"Unknown coverage waypoint_id: {waypoint}")
    if waypoint in set(unique_strings(updated.get("observed_waypoint_ids"))):
        raise ValueError(f"Cannot activate already observed waypoint_id: {waypoint}")
    if waypoint in set(unique_strings(updated.get("blocked_waypoint_ids"))):
        raise ValueError(f"Cannot activate blocked waypoint_id: {waypoint}")
    when = selected_at or now_iso()
    target = waypoints[waypoint]
    active = {
        "schema": "robot_cleaner_active_waypoint_goal_v1",
        "status": "active",
        "waypoint_id": waypoint,
        "cell": target.get("cell"),
        "selected_at": when,
        "waypoint_source": target.get("waypoint_source"),
        "purpose": target.get("purpose"),
    }
    status = {key: dict(as_dict(value)) for key, value in as_dict(updated.get("waypoint_status")).items()}
    item = status.setdefault(waypoint, {"waypoint_id": waypoint, "cell": target.get("cell")})
    item["status"] = "active"
    item["selected_at"] = item.get("selected_at") or when
    item["last_selected_at"] = when
    item["attempt_count"] = int(item.get("attempt_count") or 0) + 1
    updated["active_waypoint_goal"] = active
    updated["waypoint_status"] = status
    updated["updated_at"] = now_iso()
    return updated


def clear_active_waypoint_goal(state: Mapping[str, Any], *, reason: str = "cleared") -> JsonDict:
    updated = dict(as_dict(state))
    active = as_dict(updated.get("active_waypoint_goal"))
    waypoint = str(active.get("waypoint_id") or "")
    if waypoint:
        status = {key: dict(as_dict(value)) for key, value in as_dict(updated.get("waypoint_status")).items()}
        item = status.setdefault(waypoint, {"waypoint_id": waypoint})
        if item.get("status") == "active":
            item["status"] = "pending"
        item["last_result"] = reason
        updated["waypoint_status"] = status
    updated["active_waypoint_goal"] = None
    updated["updated_at"] = now_iso()
    return updated


def _recompute_counts(state: Mapping[str, Any]) -> JsonDict:
    updated = dict(as_dict(state))
    required = unique_strings(updated.get("required_waypoint_ids"))
    observed = [item for item in unique_strings(updated.get("observed_waypoint_ids")) if item in set(required)]
    blocked = [
        item
        for item in unique_strings(updated.get("blocked_waypoint_ids"))
        if item in set(required) and item not in set(observed)
    ]
    pending = [item for item in required if item not in set(observed) and item not in set(blocked)]
    completed = len(set(observed) | set(blocked))
    updated.update(
        {
            "updated_at": now_iso(),
            "observed_waypoint_ids": observed,
            "observed_waypoint_count": len(observed),
            "blocked_waypoint_ids": blocked,
            "blocked_waypoint_count": len(blocked),
            "pending_waypoint_ids": pending,
            "pending_waypoint_count": len(pending),
            "completed_waypoint_count": completed,
            "required_waypoint_count": len(required),
            "sweep_coverage_rate": round(completed / float(max(1, len(required))), 6),
        }
    )
    return updated


def room_state_path(memory_dir: Path | str | None = None) -> Path:
    return (Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR) / "room-state.json"


def load_room_state(memory_dir: Path | str | None = None) -> JsonDict:
    return read_json(room_state_path(memory_dir))


def save_room_state(room: Mapping[str, Any], memory_dir: Path | str | None = None) -> None:
    atomic_write_json(room_state_path(memory_dir), dict(room))


def sync_room_coverage_waypoints(
    waypoint_set: Mapping[str, Any],
    *,
    memory_dir: Path | str | None = None,
    current_cell: str | None = None,
    persist: bool = True,
) -> JsonDict:
    room = load_room_state(memory_dir)
    coverage = normalize_coverage_waypoint_state(
        waypoint_set,
        as_dict(room.get("coverage_waypoints")),
        current_cell=current_cell,
    )
    room["coverage_waypoints"] = coverage
    if persist:
        save_room_state(room, memory_dir)
    return coverage
