#!/usr/bin/env python3
"""
State manager core for the household service robot memory files.

This module is intentionally independent from the OpenClaw agent runtime.
It gives the project one programmatic place to update:

- memory/patrol-state.json
- memory/mission-state.json
- memory/room-state.json

The manager supports old and new room-state shapes, normalizes them into a
single flat schema, writes atomically, and exposes a small CLI for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


VALID_MODES = {
    "IDLE",
    "BOOT_SCAN",
    "EXPLORE",
    "SERVICE",
    "CLEAN",
    "VERIFY_CLEAN",
    "RECOVER",
    "ROOM_COMPLETE",
    "MISSION_REPORT",
    "DONE",
}
TERMINAL_MODES = {"RECOVER", "ROOM_COMPLETE", "MISSION_REPORT", "DONE"}

DEFAULT_MISSION = "room_patrol_service"
DEFAULT_ROOM = "current_room"
DEFAULT_MAX_STEPS = 200
DEFAULT_NAVIGATION_TARGET_CELLS = 120


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
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
                replaced = True
            except PermissionError:
                if last_replace_error is not None:
                    raise last_replace_error
                raise
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except PermissionError:
                # On Windows, antivirus/indexers can hold a just-written temp
                # file briefly. Leaving a dot-prefixed temp behind is safer
                # than failing the patrol loop after the real write already
                # failed or succeeded.
                pass


def default_patrol(max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "enabled": False,
        "mode": "IDLE",
        "step_count": 0,
        "max_steps": int(max_steps),
        "failed_attempts": 0,
        "last_action": None,
        "consecutive_no_target": 0,
        "consecutive_repeated_view": 0,
        "last_update": None,
    }


def default_mission(max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "mission": DEFAULT_MISSION,
        "enabled": False,
        "mode": "IDLE",
        "current_room": DEFAULT_ROOM,
        "total_steps_completed": 0,
        "max_steps": int(max_steps),
        "garbage_detected": [],
        "garbage_cleaned": [],
        "objects_detected": [],
        "objects_placed": [],
        "service_tasks_completed": [],
        "summary_needed": False,
        "final_summary": "",
        "last_result": None,
        "last_update": None,
    }


def default_room(room_name: str = DEFAULT_ROOM) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "room_name": room_name,
        "explored_steps": 0,
        "targets_found": [],
        "targets_cleaned": [],
        "objects_placed": [],
        "service_tasks_completed": [],
        "room_complete": False,
        "cell_size": 0.25,
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


@dataclass
class StateSnapshot:
    patrol: Dict[str, Any]
    mission: Dict[str, Any]
    room: Dict[str, Any]

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "patrol": self.patrol,
            "mission": self.mission,
            "room": self.room,
        }


class StateManager:
    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else Path(__file__).resolve().parents[1] / "memory"
        self.patrol_path = self.memory_dir / "patrol-state.json"
        self.mission_path = self.memory_dir / "mission-state.json"
        self.room_path = self.memory_dir / "room-state.json"

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load_state(self) -> StateSnapshot:
        #三个json文件数据修正
        patrol = self._normalize_patrol(load_json(self.patrol_path, default_patrol()))
        mission = self._normalize_mission(load_json(self.mission_path, default_mission()))
        room = self._normalize_room(load_json(self.room_path, default_room()))
        return StateSnapshot(patrol=patrol, mission=mission, room=room)

    def save_state(self, state: StateSnapshot) -> None:
        atomic_write_json(self.patrol_path, state.patrol)
        atomic_write_json(self.mission_path, state.mission)
        atomic_write_json(self.room_path, state.room)

    def normalize_and_save(self) -> StateSnapshot:
        state = self.load_state()
        self.save_state(state)
        return state

    # ------------------------------------------------------------------
    # Main state transitions
    # ------------------------------------------------------------------

    def start_mission(
        self,
        *,
        room_name: str = DEFAULT_ROOM,
        max_steps: int = DEFAULT_MAX_STEPS,
        reset_counters: bool = True,
    ) -> StateSnapshot:
        ts = now_iso()
        if reset_counters:
            patrol = default_patrol(max_steps)
            mission = default_mission(max_steps)
            room = default_room(room_name)
        else:
            state = self.load_state()
            patrol, mission, room = state.patrol, state.mission, state.room

        patrol.update(
            {
                "enabled": True,
                "mode": "BOOT_SCAN",
                "max_steps": int(max_steps),
                "last_update": ts,
            }
        )
        mission.update(
            {
                "mission": DEFAULT_MISSION,
                "enabled": True,
                "mode": "BOOT_SCAN",
                "current_room": room_name,
                "max_steps": int(max_steps),
                "summary_needed": True,
                "final_summary": "",
                "last_result": None,
                "last_update": ts,
            }
        )
        room.update(
            {
                "room_name": room_name,
                "room_complete": False,
                "last_update": ts,
            }
        )
        state = StateSnapshot(patrol=patrol, mission=mission, room=room)
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def record_step(
        self,
        *,
        action: str,
        mode: str = "EXPLORE",
        detected: Optional[Iterable[str]] = None,
        cleaned: Optional[Iterable[str]] = None,
        placed: Optional[Iterable[str]] = None,
        service_completed: Optional[Iterable[str]] = None,
        success: bool = True,
        failure_reason: Optional[str] = None,
        no_target: bool = False,
        repeated_view: bool = False,
    ) -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()

        if self._should_ignore_step_after_stop(state):
            state.patrol["last_update"] = ts
            state.mission["last_update"] = ts
            state.room["last_update"] = ts
            if not str(state.mission.get("last_result") or "").startswith("stopped:"):
                state.mission["last_result"] = state.mission.get("last_result") or "ignored_step_after_stop"
            self.save_state(state)
            return state

        self._ensure_active_state(state, mode=mode, ts=ts)

        state.patrol["step_count"] = int(state.patrol.get("step_count", 0)) + 1
        state.patrol["last_action"] = action if success else f"{action} (failed: {failure_reason or 'unknown'})"
        state.patrol["mode"] = mode
        state.mission["mode"] = mode

        if success:
            state.patrol["failed_attempts"] = int(state.patrol.get("failed_attempts", 0))
        else:
            state.patrol["failed_attempts"] = int(state.patrol.get("failed_attempts", 0)) + 1
            state.mission["last_result"] = f"failed: {failure_reason or 'unknown'}"

        state.patrol["consecutive_no_target"] = (
            int(state.patrol.get("consecutive_no_target", 0)) + 1 if no_target else 0
        )
        state.patrol["consecutive_repeated_view"] = (
            int(state.patrol.get("consecutive_repeated_view", 0)) + 1 if repeated_view else 0
        )

        if detected:
            state.mission["garbage_detected"] = unique_extend(
                state.mission.get("garbage_detected", []), detected
            )
            state.mission["objects_detected"] = unique_extend(
                state.mission.get("objects_detected", []), detected
            )
            state.room["targets_found"] = unique_extend(state.room.get("targets_found", []), detected)

        if cleaned:
            state.mission["garbage_cleaned"] = unique_extend(
                state.mission.get("garbage_cleaned", []), cleaned
            )
            state.room["targets_cleaned"] = unique_extend(state.room.get("targets_cleaned", []), cleaned)

        if placed:
            state.mission["objects_placed"] = unique_extend(
                state.mission.get("objects_placed", []), placed
            )
            state.room["objects_placed"] = unique_extend(state.room.get("objects_placed", []), placed)

        if service_completed:
            state.mission["service_tasks_completed"] = unique_extend(
                state.mission.get("service_tasks_completed", []), service_completed
            )
            state.room["service_tasks_completed"] = unique_extend(
                state.room.get("service_tasks_completed", []), service_completed
            )
            state.mission["last_result"] = f"service_completed: {', '.join(service_completed)}"

        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def record_detection(self, labels: Iterable[str]) -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        state.mission["garbage_detected"] = unique_extend(state.mission.get("garbage_detected", []), labels)
        state.mission["objects_detected"] = unique_extend(state.mission.get("objects_detected", []), labels)
        state.room["targets_found"] = unique_extend(state.room.get("targets_found", []), labels)
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def record_clean_success(self, label: str) -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        labels = [label]
        state.mission["garbage_cleaned"] = unique_extend(state.mission.get("garbage_cleaned", []), labels)
        state.room["targets_cleaned"] = unique_extend(state.room.get("targets_cleaned", []), labels)
        state.mission["last_result"] = f"cleaned: {label}"
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def record_failure(self, reason: str, *, action: Optional[str] = None) -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        state.patrol["failed_attempts"] = int(state.patrol.get("failed_attempts", 0)) + 1
        if action:
            state.patrol["last_action"] = f"{action} (failed: {reason})"
        state.mission["last_result"] = f"failed: {reason}"
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def mark_room_complete(self, *, summary: str = "") -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        state.patrol["mode"] = "MISSION_REPORT"
        state.mission["mode"] = "MISSION_REPORT"
        state.mission["summary_needed"] = True
        state.mission["final_summary"] = summary
        state.mission["last_result"] = "room_complete"
        state.room["room_complete"] = True
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def stop_mission(self, *, reason: str = "user_stop") -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        state.patrol["enabled"] = False
        state.patrol["mode"] = "DONE"
        state.patrol["last_update"] = ts

        state.mission["enabled"] = False
        state.mission["mode"] = "DONE"
        state.mission["summary_needed"] = True
        state.mission["last_result"] = f"stopped: {reason}"
        state.mission["final_summary"] = f"mission_stopped:{reason}"
        state.mission["last_update"] = ts

        state.room["last_update"] = ts
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def mark_recover_failed(self, *, reason: str = "recover_failed") -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        state.patrol["enabled"] = False
        state.patrol["mode"] = "RECOVER"
        state.patrol["last_update"] = ts

        state.mission["enabled"] = False
        state.mission["mode"] = "RECOVER"
        state.mission["summary_needed"] = True
        state.mission["last_result"] = f"recover_failed: {reason}"
        state.mission["final_summary"] = f"recover_failed:{reason}"
        state.mission["last_update"] = ts

        state.room["room_complete"] = False
        state.room["last_update"] = ts
        self._sync_common_fields(state, ts=ts, source="patrol")
        self.save_state(state)
        return state

    def finalize_report(
        self,
        *,
        reason: str = "report_delivered",
        report_message: str = "",
        archive: bool = True,
    ) -> Dict[str, Any]:
        state = self.load_state()
        ts = now_iso()
        archive_path: Optional[Path] = None

        archive_entry = {
            "schema_version": 1,
            "archived_at": ts,
            "reason": reason,
            "report_message": report_message,
            "patrol": state.patrol,
            "mission": state.mission,
            "room": state.room,
        }
        if archive:
            archive_path = self._append_mission_archive(archive_entry)

        room_name = str(
            state.mission.get("current_room")
            or state.room.get("room_name")
            or DEFAULT_ROOM
        )
        max_steps = int(
            state.patrol.get("max_steps")
            or state.mission.get("max_steps")
            or DEFAULT_MAX_STEPS
        )
        reset_state = StateSnapshot(
            patrol=default_patrol(max_steps),
            mission=default_mission(max_steps),
            room=default_room(room_name),
        )
        reset_state.patrol["last_update"] = ts
        reset_state.mission["last_update"] = ts
        reset_state.room["last_update"] = ts
        self.save_state(reset_state)

        return {
            "status": "success",
            "result_type": "state_finalized_to_idle",
            "reason": reason,
            "archived": bool(archive),
            "archive_path": str(archive_path) if archive_path else None,
            "state": reset_state.as_dict(),
        }

    def sync_counts(self, *, source: str = "patrol") -> StateSnapshot:
        state = self.load_state()
        ts = now_iso()
        self._sync_common_fields(state, ts=ts, source=source)
        self.save_state(state)
        return state

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_state(self) -> Dict[str, Any]:
        state = self.load_state()
        issues: List[str] = []
        warnings: List[str] = []

        patrol_steps = int(state.patrol.get("step_count", 0))
        mission_steps = int(state.mission.get("total_steps_completed", 0))
        room_steps = int(state.room.get("explored_steps", 0))

        if not (patrol_steps == mission_steps == room_steps):
            issues.append(
                "step count mismatch: "
                f"patrol={patrol_steps}, mission={mission_steps}, room={room_steps}"
            )

        patrol_max = int(state.patrol.get("max_steps", DEFAULT_MAX_STEPS))
        mission_max = int(state.mission.get("max_steps", DEFAULT_MAX_STEPS))
        if patrol_max != mission_max:
            issues.append(f"max_steps mismatch: patrol={patrol_max}, mission={mission_max}")

        patrol_mode = state.patrol.get("mode")
        mission_mode = state.mission.get("mode")
        if patrol_mode not in VALID_MODES:
            issues.append(f"invalid patrol mode: {patrol_mode}")
        if mission_mode not in VALID_MODES:
            issues.append(f"invalid mission mode: {mission_mode}")
        if patrol_mode != mission_mode:
            warnings.append(f"mode differs: patrol={patrol_mode}, mission={mission_mode}")

        if state.room.get("room_complete") and patrol_mode not in {"ROOM_COMPLETE", "MISSION_REPORT", "DONE"}:
            warnings.append("room_complete=true but patrol mode is not a terminal/reporting mode")

        if patrol_steps >= patrol_max and not state.room.get("room_complete"):
            warnings.append("step_count reached max_steps but room_complete is false")

        return {
            "valid": not issues,
            "issues": issues,
            "warnings": warnings,
            "state": state.as_dict(),
        }

    def should_continue(self) -> Dict[str, Any]:
        state = self.load_state()
        reasons: List[str] = []

        if not state.mission.get("enabled", False):
            reasons.append("mission disabled")#任务被手动关闭
        if not state.patrol.get("enabled", False):
            reasons.append("patrol disabled")# 巡逻功能关闭
        if state.room.get("room_complete", False):
            reasons.append("room already complete")
        if int(state.patrol.get("step_count", 0)) >= int(state.patrol.get("max_steps", DEFAULT_MAX_STEPS)):
            reasons.append("max_steps reached")
        if state.patrol.get("mode") in {"ROOM_COMPLETE", "MISSION_REPORT", "DONE"}:#进入终止模式
            reasons.append(f"terminal patrol mode: {state.patrol.get('mode')}")
        if state.mission.get("mode") in {"ROOM_COMPLETE", "MISSION_REPORT", "DONE"}:
            reasons.append(f"terminal mission mode: {state.mission.get('mode')}")

        return {
            "continue": not reasons,
            "reasons": reasons,
            "state": state.as_dict(),
        }

    def _should_ignore_step_after_stop(self, state: StateSnapshot) -> bool:
        patrol_mode = str(state.patrol.get("mode") or "")
        mission_mode = str(state.mission.get("mode") or "")
        if patrol_mode in TERMINAL_MODES or mission_mode in TERMINAL_MODES:
            return True
        if not bool(state.patrol.get("enabled", False)) or not bool(state.mission.get("enabled", False)):
            return True
        return False

    # ------------------------------------------------------------------
    # Normalization internals
    # ------------------------------------------------------------------

    def _normalize_patrol(self, data: Dict[str, Any]) -> Dict[str, Any]:
        base = default_patrol(int(data.get("max_steps", DEFAULT_MAX_STEPS)))
        base.update(data)
        base["schema_version"] = 1
        base["enabled"] = bool(base.get("enabled", False))
        base["mode"] = str(base.get("mode") or "IDLE")
        base["step_count"] = max(0, int(base.get("step_count", 0)))
        base["max_steps"] = max(1, int(base.get("max_steps", DEFAULT_MAX_STEPS)))
        base["failed_attempts"] = max(0, int(base.get("failed_attempts", 0)))
        base["consecutive_no_target"] = max(0, int(base.get("consecutive_no_target", 0)))
        base["consecutive_repeated_view"] = max(0, int(base.get("consecutive_repeated_view", 0)))
        return base

    def _normalize_mission(self, data: Dict[str, Any]) -> Dict[str, Any]:
        base = default_mission(int(data.get("max_steps", DEFAULT_MAX_STEPS)))
        base.update(data)
        base["schema_version"] = 1
        base["mission"] = str(base.get("mission") or DEFAULT_MISSION)
        base["enabled"] = bool(base.get("enabled", False))
        base["mode"] = str(base.get("mode") or "IDLE")
        base["current_room"] = base.get("current_room") or DEFAULT_ROOM
        base["total_steps_completed"] = max(0, int(base.get("total_steps_completed", 0)))
        base["max_steps"] = max(1, int(base.get("max_steps", DEFAULT_MAX_STEPS)))
        base["garbage_detected"] = unique_extend([], base.get("garbage_detected", []))
        base["garbage_cleaned"] = unique_extend([], base.get("garbage_cleaned", []))
        objects_detected = base.get("objects_detected", [])
        if not objects_detected:
            objects_detected = base.get("garbage_detected", [])
        base["objects_detected"] = unique_extend([], objects_detected)
        base["objects_placed"] = unique_extend([], base.get("objects_placed", []))
        base["service_tasks_completed"] = unique_extend([], base.get("service_tasks_completed", []))
        base["summary_needed"] = bool(base.get("summary_needed", False))
        base["final_summary"] = str(base.get("final_summary") or "")
        return base

    def _normalize_room(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Backward compatibility with the older nested shape:
        # {"current_room": {"visited_steps": ..., "trash_found": ...}}
        if isinstance(data.get("current_room"), dict):
            old = data["current_room"]
            room_name = old.get("room_name") or data.get("room_name") or DEFAULT_ROOM
            flat = default_room(str(room_name))
            flat.update(
                {
                    "explored_steps": old.get("visited_steps", old.get("explored_steps", 0)),
                    "targets_found": old.get("trash_found", old.get("targets_found", [])),
                    "targets_cleaned": old.get("trash_cleaned", old.get("targets_cleaned", [])),
                    "room_complete": old.get("room_complete", False),
                    "last_update": data.get("last_update", old.get("last_update")),
                }
            )
            data = flat

        base = default_room(str(data.get("room_name") or DEFAULT_ROOM))
        base.update(data)
        base.pop("current_room", None)
        base["schema_version"] = 1
        base["room_name"] = base.get("room_name") or DEFAULT_ROOM
        base["explored_steps"] = max(0, int(base.get("explored_steps", 0)))
        base["targets_found"] = unique_extend([], base.get("targets_found", []))
        base["targets_cleaned"] = unique_extend([], base.get("targets_cleaned", []))
        base["objects_placed"] = unique_extend([], base.get("objects_placed", []))
        base["service_tasks_completed"] = unique_extend([], base.get("service_tasks_completed", []))
        base["room_complete"] = bool(base.get("room_complete", False))
        base["cell_size"] = float(base.get("cell_size", 0.25) or 0.25)
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
        visited_count = len(base.get("visited_cells", []))
        blocked_count = len(base.get("blocked_edges", []))
        coverage = min(
            1.0,
            (visited_count + 0.25 * blocked_count) / float(DEFAULT_NAVIGATION_TARGET_CELLS),
        )
        base["coverage_estimate"] = round(coverage, 3)
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

    def _ensure_active_state(self, state: StateSnapshot, *, mode: str, ts: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}")
        state.patrol["enabled"] = True
        state.mission["enabled"] = True
        state.mission["mission"] = state.mission.get("mission") or DEFAULT_MISSION
        state.mission["current_room"] = state.mission.get("current_room") or state.room.get("room_name") or DEFAULT_ROOM
        state.room["room_name"] = state.room.get("room_name") or state.mission["current_room"]
        state.patrol["last_update"] = ts
        state.mission["last_update"] = ts
        state.room["last_update"] = ts

    def _sync_common_fields(self, state: StateSnapshot, *, ts: str, source: str) -> None:
        if source not in {"patrol", "mission", "room", "max"}:
            raise ValueError("source must be one of: patrol, mission, room, max")

        if source == "patrol":
            steps = int(state.patrol.get("step_count", 0))
        elif source == "mission":
            steps = int(state.mission.get("total_steps_completed", 0))
        elif source == "room":
            steps = int(state.room.get("explored_steps", 0))
        else:
            steps = max(
                int(state.patrol.get("step_count", 0)),
                int(state.mission.get("total_steps_completed", 0)),
                int(state.room.get("explored_steps", 0)),
            )

        max_steps = int(state.patrol.get("max_steps", state.mission.get("max_steps", DEFAULT_MAX_STEPS)))
        room_name = state.mission.get("current_room") or state.room.get("room_name") or DEFAULT_ROOM

        state.patrol["step_count"] = steps
        state.patrol["max_steps"] = max_steps
        state.patrol["last_update"] = ts

        state.mission["total_steps_completed"] = steps
        state.mission["max_steps"] = max_steps
        state.mission["current_room"] = room_name
        state.mission["last_update"] = ts

        state.room["explored_steps"] = steps
        state.room["room_name"] = room_name
        state.room["last_update"] = ts

        state.mission["garbage_detected"] = unique_extend(
            state.mission.get("garbage_detected", []),
            state.room.get("targets_found", []),
        )
        state.mission["objects_detected"] = unique_extend(
            state.mission.get("objects_detected", []),
            state.room.get("targets_found", []),
        )
        state.room["targets_found"] = unique_extend(
            state.room.get("targets_found", []),
            unique_extend(state.mission.get("garbage_detected", []), state.mission.get("objects_detected", [])),
        )
        state.mission["garbage_cleaned"] = unique_extend(
            state.mission.get("garbage_cleaned", []),
            state.room.get("targets_cleaned", []),
        )
        state.room["targets_cleaned"] = unique_extend(
            state.room.get("targets_cleaned", []),
            state.mission.get("garbage_cleaned", []),
        )
        state.mission["objects_placed"] = unique_extend(
            state.mission.get("objects_placed", []),
            state.room.get("objects_placed", []),
        )
        state.room["objects_placed"] = unique_extend(
            state.room.get("objects_placed", []),
            state.mission.get("objects_placed", []),
        )
        state.mission["service_tasks_completed"] = unique_extend(
            state.mission.get("service_tasks_completed", []),
            state.room.get("service_tasks_completed", []),
        )
        state.room["service_tasks_completed"] = unique_extend(
            state.room.get("service_tasks_completed", []),
            state.mission.get("service_tasks_completed", []),
        )

    def _append_mission_archive(self, entry: Dict[str, Any]) -> Path:
        archive_path = self.memory_dir / "mission-archive.jsonl"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        return archive_path


def parse_labels(value: Optional[str]) -> List[str]:
    if not value:
        return []
    labels: List[str] = []
    for part in value.split(","):
        item = part.strip()
        if item:
            labels.append(item)
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage household service robot memory state files.")
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Directory containing patrol-state.json, mission-state.json and room-state.json.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show", help="Print normalized state without writing.")
    sub.add_parser("validate", help="Validate state consistency.")
    sub.add_parser("should-continue", help="Check whether the patrol should continue.")

    sync = sub.add_parser("sync", help="Normalize and synchronize the three JSON files.")
    sync.add_argument("--source", choices=["patrol", "mission", "room", "max"], default="patrol")

    start = sub.add_parser("start-mission", help="Start or reset a room patrol mission.")
    start.add_argument("--room", default=DEFAULT_ROOM)
    start.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    start.add_argument("--no-reset", action="store_true", help="Keep counters and only mark mission active.")

    step = sub.add_parser("record-step", help="Record one completed physical-action step.")
    step.add_argument("--action", required=True)
    step.add_argument("--mode", default="EXPLORE", choices=sorted(VALID_MODES))
    step.add_argument("--detected", default="", help="Comma-separated labels detected this step.")
    step.add_argument("--cleaned", default="", help="Comma-separated labels cleaned this step.")
    step.add_argument("--placed", default="", help="Comma-separated labels placed this step.")
    step.add_argument(
        "--service-completed",
        default="",
        help="Comma-separated pickup/place service subgoals completed this step.",
    )
    step.add_argument("--failed", action="store_true")
    step.add_argument("--failure-reason", default=None)
    step.add_argument("--no-target", action="store_true")
    step.add_argument("--repeated-view", action="store_true")

    detect = sub.add_parser("record-detection", help="Record detected target labels.")
    detect.add_argument("labels", nargs="+")

    clean = sub.add_parser("record-clean", help="Record one successfully cleaned target label.")
    clean.add_argument("label")

    failure = sub.add_parser("record-failure", help="Record one failure without increasing step_count.")
    failure.add_argument("reason")
    failure.add_argument("--action", default=None)

    complete = sub.add_parser("mark-room-complete", help="Mark the current room complete.")
    complete.add_argument("--summary", default="")

    stop = sub.add_parser("stop-mission", help="Stop the current patrol mission without marking room complete.")
    stop.add_argument("--reason", default="user_stop")

    recover = sub.add_parser("mark-recover-failed", help="Mark patrol as blocked in RECOVER without completing the room.")
    recover.add_argument("--reason", default="recover_failed")

    finalize = sub.add_parser(
        "finalize-report",
        help="Archive the completed/reporting mission and reset current state to IDLE.",
    )
    finalize.add_argument("--reason", default="report_delivered")
    finalize.add_argument("--report-message", default="")
    finalize.add_argument("--no-archive", action="store_true")

    return parser


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manager = StateManager(Path(args.memory_dir) if args.memory_dir else None)

    try:
        if args.command == "show":
            print_json(manager.load_state().as_dict())
        elif args.command == "validate":
            result = manager.validate_state()
            print_json(result)
            return 0 if result["valid"] else 2
        elif args.command == "should-continue":
            print_json(manager.should_continue())
        elif args.command == "sync":
            print_json(manager.sync_counts(source=args.source).as_dict())
        elif args.command == "start-mission":
            print_json(
                manager.start_mission(
                    room_name=args.room,
                    max_steps=args.max_steps,
                    reset_counters=not args.no_reset,
                ).as_dict()
            )
        elif args.command == "record-step":
            print_json(
                manager.record_step(
                    action=args.action,
                    mode=args.mode,
                    detected=parse_labels(args.detected),
                    cleaned=parse_labels(args.cleaned),
                    placed=parse_labels(args.placed),
                    service_completed=parse_labels(args.service_completed),
                    success=not args.failed,
                    failure_reason=args.failure_reason,
                    no_target=args.no_target,
                    repeated_view=args.repeated_view,
                ).as_dict()
            )
        elif args.command == "record-detection":
            print_json(manager.record_detection(args.labels).as_dict())
        elif args.command == "record-clean":
            print_json(manager.record_clean_success(args.label).as_dict())
        elif args.command == "record-failure":
            print_json(manager.record_failure(args.reason, action=args.action).as_dict())
        elif args.command == "mark-room-complete":
            print_json(manager.mark_room_complete(summary=args.summary).as_dict())
        elif args.command == "stop-mission":
            print_json(manager.stop_mission(reason=args.reason).as_dict())
        elif args.command == "mark-recover-failed":
            print_json(manager.mark_recover_failed(reason=args.reason).as_dict())
        elif args.command == "finalize-report":
            print_json(
                manager.finalize_report(
                    reason=args.reason,
                    report_message=args.report_message,
                    archive=not args.no_archive,
                )
            )
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print_json({"status": "error", "message": str(exc)})
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
