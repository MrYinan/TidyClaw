#!/usr/bin/env python3
"""Shared state helpers for Robot Cleaner OpenClaw tools."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.state_manager_core import StateManager  # noqa: E402


JsonDict = dict[str, Any]
MEMORY_DIR = REPO_ROOT / "memory"
PID_FILE = MEMORY_DIR / "patrol-runner.pid.json"
PROCESS_LOG = MEMORY_DIR / "patrol-runner-process.log"
DECISION_CONTEXT_PATH = MEMORY_DIR / "decision-context.json"
SERVICE_TASK_STATE_PATH = MEMORY_DIR / "service-task-state.json"
YOLO_CURRENT_PATH = MEMORY_DIR / "yolo-current-rgbd.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def print_json(data: JsonDict, *, compact: bool = False) -> None:
    if compact:
        print(json.dumps(data, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def read_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        return {"status": "error", "result_type": "json_read_failed", "path": str(path), "message": str(exc)}
    return data if isinstance(data, dict) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def short_list(values: Iterable[Any], *, limit: int = 12) -> list[Any]:
    items = list(values)
    return items[-limit:] if len(items) > limit else items


def hidden_startupinfo() -> Any:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def load_pid_info() -> JsonDict:
    return read_json(PID_FILE)


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
                startupinfo=hidden_startupinfo(),
                creationflags=hidden_creationflags(),
            )
        except Exception:
            return False
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def runner_process_summary() -> JsonDict:
    pid_info = load_pid_info()
    pid: int | None = None
    try:
        pid = int(pid_info.get("pid")) if pid_info.get("pid") is not None else None
    except (TypeError, ValueError):
        pid = None
    alive = process_alive(pid)
    return {
        "pid": pid,
        "alive": alive,
        "started_at": pid_info.get("started_at"),
        "pid_file": str(PID_FILE),
        "log_path": str(PROCESS_LOG),
        "pid_info_present": bool(pid_info),
    }


def load_state(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    manager = StateManager(memory_dir)
    return manager.load_state().as_dict()


def decision_context_summary(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    path = memory_dir / "decision-context.json"
    context = read_json(path)
    option_set = context.get("option_set") if isinstance(context.get("option_set"), dict) else {}
    options = option_set.get("options") if isinstance(option_set.get("options"), list) else []
    return {
        "exists": bool(context),
        "path": str(path),
        "status": context.get("status"),
        "schema": context.get("schema"),
        "generated_at": context.get("generated_at"),
        "rule_baseline_option_id": option_set.get("rule_baseline_option_id"),
        "model_decision_required": bool(context) and bool(option_set),
        "option_count": len(options),
        "option_ids": [str(item.get("option_id")) for item in options[:12] if isinstance(item, dict)],
        "consistency_warning_count": len(as_list(context.get("consistency_warnings"))),
    }


def perception_summary(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    path = memory_dir / "yolo-current-rgbd.json"
    perception = read_json(path)
    return {
        "exists": bool(perception),
        "path": str(path),
        "status": perception.get("status"),
        "result_type": perception.get("result_type"),
        "perception_backend": perception.get("perception_backend"),
        "candidate_count": perception.get("candidate_count"),
        "pickup_target_detected": perception.get("pickup_target_detected"),
        "place_receptacle_detected": perception.get("place_receptacle_detected"),
        "frontier_exists": perception.get("frontier_exists"),
        "recommended_action": perception.get("recommended_action"),
    }


def service_summary(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    path = memory_dir / "service-task-state.json"
    service = read_json(path)
    completed = [item for item in as_list(service.get("completed_subgoals")) if isinstance(item, dict)]
    return {
        "exists": bool(service),
        "path": str(path),
        "phase": service.get("phase"),
        "holding_object": bool(service.get("holding_object")),
        "held_object_label": service.get("held_object_label"),
        "held_object_track_id": service.get("held_object_track_id"),
        "held_object_labels": as_list(service.get("held_object_labels")),
        "pickup_surface_policy": service.get("pickup_surface_policy"),
        "interaction_grounding": service.get("interaction_grounding"),
        "last_reason": service.get("last_reason"),
        "completed_subgoal_count": len(completed),
        "completed_subgoals_tail": completed[-5:],
    }


def build_robot_status(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    state = load_state(memory_dir)
    patrol = state.get("patrol", {}) if isinstance(state.get("patrol"), dict) else {}
    mission = state.get("mission", {}) if isinstance(state.get("mission"), dict) else {}
    room = state.get("room", {}) if isinstance(state.get("room"), dict) else {}
    service = service_summary(memory_dir)
    runner = runner_process_summary()

    placed = unique_strings(as_list(room.get("objects_placed")) or as_list(mission.get("objects_placed")))
    service_completed = unique_strings(
        as_list(room.get("service_tasks_completed")) or as_list(mission.get("service_tasks_completed"))
    )
    cleaned = unique_strings(as_list(room.get("targets_cleaned")) or as_list(mission.get("garbage_cleaned")))
    detected = unique_strings(
        as_list(mission.get("objects_detected")) or as_list(room.get("targets_found")) or as_list(mission.get("garbage_detected"))
    )

    step_count = int(patrol.get("step_count") or mission.get("total_steps_completed") or room.get("explored_steps") or 0)
    max_steps = int(patrol.get("max_steps") or mission.get("max_steps") or 0)
    room_complete = bool(room.get("room_complete"))

    return {
        "status": "success",
        "result_type": "robot_cleaner_status",
        "schema": "robot_cleaner_status_v1",
        "generated_at": now_iso(),
        "runner": runner,
        "task": {
            "mission_enabled": bool(mission.get("enabled")),
            "mission_mode": mission.get("mode"),
            "patrol_enabled": bool(patrol.get("enabled")),
            "patrol_mode": patrol.get("mode"),
            "room": mission.get("current_room") or room.get("room_name"),
            "room_complete": room_complete,
            "step_count": step_count,
            "max_steps": max_steps,
            "last_action": patrol.get("last_action"),
            "last_result": mission.get("last_result"),
            "final_summary": mission.get("final_summary"),
        },
        "service": service,
        "progress": {
            "objects_placed_count": len(placed),
            "objects_placed": placed,
            "service_tasks_completed_count": len(service_completed),
            "service_tasks_completed": service_completed,
            "completed_subgoal_count": service.get("completed_subgoal_count", 0),
            "targets_cleaned_count": len(cleaned),
            "targets_cleaned": cleaned,
            "visual_candidates_count": len(detected),
            "visual_candidates": short_list(detected, limit=12),
        },
        "navigation": {
            "last_cell": room.get("last_cell"),
            "last_heading": room.get("last_heading"),
            "coverage_estimate": room.get("coverage_estimate"),
            "frontier_count": len(as_list(room.get("frontier_cells")) or as_list(room.get("known_frontier_cells"))),
            "collision_count": int(room.get("collision_count", 0) or 0),
            "oscillation_count": int(room.get("oscillation_count", 0) or 0),
            "pose_confidence": room.get("pose_confidence"),
            "position_uncertainty_cells": room.get("position_uncertainty_cells"),
            "heading_confidence": room.get("heading_confidence"),
        },
        "perception": perception_summary(memory_dir),
        "decision_context": decision_context_summary(memory_dir),
        "raw_state_paths": {
            "mission": str(memory_dir / "mission-state.json"),
            "room": str(memory_dir / "room-state.json"),
            "patrol": str(memory_dir / "patrol-state.json"),
            "service_task": str(memory_dir / "service-task-state.json"),
        },
    }


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "unknown"


def build_report_message(status_payload: JsonDict) -> tuple[str, str, bool]:
    task = status_payload.get("task", {}) if isinstance(status_payload.get("task"), dict) else {}
    progress = status_payload.get("progress", {}) if isinstance(status_payload.get("progress"), dict) else {}
    service = status_payload.get("service", {}) if isinstance(status_payload.get("service"), dict) else {}
    navigation = status_payload.get("navigation", {}) if isinstance(status_payload.get("navigation"), dict) else {}
    runner = status_payload.get("runner", {}) if isinstance(status_payload.get("runner"), dict) else {}

    step_count = task.get("step_count", 0)
    max_steps = task.get("max_steps", 0)
    placed_count = progress.get("objects_placed_count", 0)
    completed_count = progress.get("service_tasks_completed_count", 0)
    cleaned_count = progress.get("targets_cleaned_count", 0)
    candidate_count = progress.get("visual_candidates_count", 0)
    coverage = format_percent(navigation.get("coverage_estimate"))
    frontier_count = navigation.get("frontier_count", 0)
    holding = bool(service.get("holding_object"))
    phase = service.get("phase") or "unknown"

    if bool(task.get("room_complete")):
        return (
            "room_complete",
            (
                f"当前房间整理任务已完成。已放置 {placed_count} 个物体，"
                f"已完成 {completed_count} 个整理子任务，已清理 {cleaned_count} 个地面目标，"
                f"剩余视觉候选约 {candidate_count} 个，覆盖率约 {coverage}，frontier {frontier_count}。"
            ),
            True,
        )

    if task.get("mission_mode") == "RECOVER" or task.get("patrol_mode") == "RECOVER":
        return (
            "recover",
            f"当前整理任务进入 RECOVER 状态。阶段={phase}，步数={step_count}/{max_steps}，原因={task.get('final_summary') or task.get('last_result') or 'unknown'}。",
            True,
        )

    if task.get("mission_mode") == "DONE" or task.get("patrol_mode") == "DONE":
        return (
            "stopped",
            f"当前整理任务已停止。阶段={phase}，步数={step_count}/{max_steps}，已放置 {placed_count} 个物体，已完成 {completed_count} 个整理子任务。",
            True,
        )

    if bool(runner.get("alive")):
        return (
            "running",
            (
                f"当前房间整理任务正在运行。阶段={phase}，holding={holding}，步数={step_count}/{max_steps}，"
                f"已放置 {placed_count} 个物体，已完成 {completed_count} 个整理子任务，覆盖率约 {coverage}。"
            ),
            False,
        )

    if completed_count or placed_count:
        return (
            "subgoal_progress",
            (
                f"当前没有后台 runner 进程，但已有整理进度。阶段={phase}，步数={step_count}/{max_steps}，"
                f"已放置 {placed_count} 个物体，已完成 {completed_count} 个整理子任务。"
            ),
            False,
        )

    return (
        "not_running",
        f"当前没有运行中的 robot cleaner 后台任务。最近阶段={phase}，步数={step_count}/{max_steps}，holding={holding}。",
        False,
    )


def build_robot_report(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    status_payload = build_robot_status(memory_dir)
    report_type, message, report_ready = build_report_message(status_payload)
    return {
        "status": "success",
        "result_type": "robot_cleaner_report",
        "schema": "robot_cleaner_report_v1",
        "generated_at": now_iso(),
        "report_ready": report_ready,
        "report_type": report_type,
        "user_message": message,
        "summary": {
            "task": status_payload.get("task"),
            "service": status_payload.get("service"),
            "progress": status_payload.get("progress"),
            "navigation": status_payload.get("navigation"),
            "runner": status_payload.get("runner"),
        },
    }
