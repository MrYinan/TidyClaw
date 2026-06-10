#!/usr/bin/env python3
"""State synchronization for options executed outside patrol_runner.

The option executor calls robot skills directly. Those skills intentionally only
talk to the backend and return structured feedback, while patrol_runner used to
own the domain-state commit. This module keeps that commit logic reusable and
small so future MCP wrappers can call the same state layer.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
SERVICE_TASK_STATE_NAME = "service-task-state.json"
PLACE_PRECHECK_CACHE_NAME = "place-precheck-cache.json"
SERVICE_INITIAL_PHASE = "SEARCH_PICKUP_TARGET"
HELD_FOOD_LABELS = {"apple", "banana", "lettuce", "orange", "potato", "tomato"}
RECENTLY_PLACED_SUPPRESSION_STEPS = 12


JsonDict = dict[str, Any]


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


def load_json(path: Path, default: JsonDict | None = None) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return dict(default or {})
    return data if isinstance(data, dict) else dict(default or {})


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_service_label(label: Any) -> str:
    value = str(label or "").strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def service_label_tokens(*values: Any) -> list[str]:
    labels: list[str] = []
    for value in values:
        parts = value if isinstance(value, (list, tuple, set)) else str(value or "").split(",")
        for part in parts:
            raw = str(part or "").strip()
            normalized = normalize_service_label(raw)
            for token in (normalized, raw):
                if token and token not in labels:
                    labels.append(token)
    return labels


def held_object_family_for_labels(labels: Iterable[Any]) -> str:
    tokens = {normalize_service_label(label) for label in labels if normalize_service_label(label)}
    if tokens & HELD_FOOD_LABELS:
        return "food"
    if tokens:
        return "pickup_target"
    return "unknown"


def default_service_task_state() -> JsonDict:
    return {
        "schema_version": 1,
        "task_style": "alfred_like_pick_place_metadata_hidden_executor",
        "interaction_grounding": "metadata-hidden",
        "pickup_surface_policy": "floor-only",
        "pickup_target_labels": [],
        "phase": SERVICE_INITIAL_PHASE,
        "subgoal_index": 0,
        "target_label": None,
        "target_raw_label": None,
        "target_signature": None,
        "target_track_id": None,
        "target_last_seen_step": None,
        "target_lost_scan_count": 0,
        "receptacle_label": None,
        "receptacle_raw_label": None,
        "receptacle_signature": None,
        "receptacle_track_id": None,
        "receptacle_last_seen_step": None,
        "receptacle_lost_scan_count": 0,
        "holding_object": False,
        "phase_attempts": 0,
        "target_attempts": 0,
        "receptacle_attempts": 0,
        "receptacle_alignment_streak": 0,
        "receptacle_action_hint": None,
        "receptacle_action_hint_until_step": None,
        "receptacle_last_position_hint": None,
        "surface_place_status": None,
        "surface_search_attempts": 0,
        "no_ready_surface_steps": 0,
        "last_surface_search_action": None,
        "last_surface_search_reason": None,
        "held_object_label": None,
        "held_object_raw_label": None,
        "held_object_family": "unknown",
        "held_object_labels": [],
        "held_object_track_id": None,
        "completed_subgoals": [],
        "recently_placed_labels": {},
        "recently_placed_tracks": {},
        "last_reason": "initialized",
        "last_update": now_iso(),
        "history": [],
    }


def normalize_service_state(data: JsonDict) -> JsonDict:
    state = default_service_task_state()
    if isinstance(data, dict):
        state.update(data)
    if not isinstance(state.get("history"), list):
        state["history"] = []
    if not isinstance(state.get("completed_subgoals"), list):
        state["completed_subgoals"] = []
    if not isinstance(state.get("recently_placed_labels"), dict):
        state["recently_placed_labels"] = {}
    if not isinstance(state.get("recently_placed_tracks"), dict):
        state["recently_placed_tracks"] = {}
    if not isinstance(state.get("held_object_labels"), list):
        state["held_object_labels"] = service_label_tokens(
            state.get("held_object_label"),
            state.get("held_object_raw_label"),
        )
    if not state.get("held_object_family"):
        state["held_object_family"] = held_object_family_for_labels(state.get("held_object_labels") or [])
    return state


def load_service_state(memory_dir: Path = MEMORY_DIR) -> JsonDict:
    return normalize_service_state(load_json(Path(memory_dir) / SERVICE_TASK_STATE_NAME, default_service_task_state()))


def save_service_state(state: JsonDict, memory_dir: Path = MEMORY_DIR) -> None:
    state["last_update"] = now_iso()
    atomic_write_json(Path(memory_dir) / SERVICE_TASK_STATE_NAME, normalize_service_state(state))


def current_step_count(memory_dir: Path = MEMORY_DIR) -> int:
    patrol = load_json(Path(memory_dir) / "patrol-state.json", {})
    try:
        return max(0, int(patrol.get("step_count", 0) or 0))
    except (TypeError, ValueError):
        return 0


def candidate_label(candidate: JsonDict) -> str:
    return str(candidate.get("raw_label") or candidate.get("label") or "").strip()


def candidate_track_id(candidate: JsonDict) -> str:
    for key in ("object_memory_track_id", "track_id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


def short_candidate(candidate: JsonDict | None) -> JsonDict:
    item = as_dict(candidate)
    if not item:
        return {}
    return {
        key: value
        for key, value in {
            "id": item.get("candidate_id") or item.get("id"),
            "surface_candidate_id": item.get("surface_candidate_id"),
            "label": item.get("label"),
            "raw_label": item.get("raw_label"),
            "task_semantic_class": item.get("task_semantic_class") or item.get("task_class"),
            "position_hint": item.get("position_hint"),
            "surface_hint": item.get("surface_hint"),
        }.items()
        if value not in (None, "", [], {})
    }


def append_service_history(state: JsonDict, event: str, *, step: int, **payload: Any) -> None:
    history = state.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        state["history"] = history
    entry = {
        "time": now_iso(),
        "step": int(step),
        "event": event,
    }
    entry.update(payload)
    history.append(entry)
    state["history"] = history[-80:]


def set_service_phase(
    state: JsonDict,
    phase: str,
    *,
    reason: str,
    candidate: JsonDict | None,
    step: int,
    holding_object: bool,
    increment_subgoal: bool = False,
) -> None:
    old_phase = str(state.get("phase") or SERVICE_INITIAL_PHASE)
    if old_phase != phase:
        state["phase_attempts"] = 0
        if increment_subgoal:
            try:
                state["subgoal_index"] = int(state.get("subgoal_index", 0) or 0) + 1
            except (TypeError, ValueError):
                state["subgoal_index"] = 1
    state["phase"] = phase
    state["last_reason"] = reason
    state["holding_object"] = bool(holding_object)
    append_service_history(
        state,
        "phase_transition",
        step=step,
        old_phase=old_phase,
        phase=phase,
        reason=reason,
        candidate=short_candidate(candidate),
    )


def set_held_object_context(state: JsonDict, candidate: JsonDict, *, track_id: str = "") -> None:
    raw_label = candidate.get("raw_label") or candidate.get("label")
    label = candidate.get("label") or raw_label
    tokens = service_label_tokens(label, raw_label)
    state["held_object_labels"] = tokens
    state["held_object_label"] = normalize_service_label(label or (tokens[0] if tokens else "held_object"))
    state["held_object_raw_label"] = str(raw_label or label or (tokens[0] if tokens else "held_object")).strip()
    state["held_object_family"] = held_object_family_for_labels(tokens)
    state["held_object_track_id"] = track_id or candidate_track_id(candidate) or state.get("target_track_id")
    state["holding_object"] = True


def clear_held_object_context(state: JsonDict) -> None:
    state["held_object_label"] = None
    state["held_object_raw_label"] = None
    state["held_object_family"] = "unknown"
    state["held_object_labels"] = []
    state["held_object_track_id"] = None
    state["holding_object"] = False


def reset_pick_place_locks(state: JsonDict) -> None:
    for key in (
        "target_label",
        "target_raw_label",
        "target_signature",
        "target_track_id",
        "target_last_seen_step",
        "receptacle_label",
        "receptacle_raw_label",
        "receptacle_signature",
        "receptacle_track_id",
        "receptacle_last_seen_step",
        "receptacle_action_hint",
        "receptacle_action_hint_until_step",
        "receptacle_last_position_hint",
    ):
        state[key] = None
    state["target_attempts"] = 0
    state["receptacle_attempts"] = 0
    state["receptacle_alignment_streak"] = 0


def suppress_recently_placed_track(state: JsonDict, *, track_id: str, label: str, step: int) -> None:
    track_text = str(track_id or "").strip()
    if not track_text:
        return
    tracks = state.setdefault("recently_placed_tracks", {})
    if not isinstance(tracks, dict):
        tracks = {}
        state["recently_placed_tracks"] = tracks
    tracks[track_text] = {
        "track_id": track_text,
        "label": normalize_service_label(label) or None,
        "until_step": int(step) + RECENTLY_PLACED_SUPPRESSION_STEPS,
    }


def failure_reason_from_execution(execution: JsonDict, default: str) -> str:
    return str(
        execution.get("error_message")
        or execution.get("message")
        or execution.get("result_type")
        or default
    )


def record_state_manager_step(
    *,
    memory_dir: Path,
    action: str,
    mode: str,
    success: bool,
    failure_reason: str | None = None,
    detected: Iterable[str] | None = None,
    cleaned: Iterable[str] | None = None,
    placed: Iterable[str] | None = None,
    service_completed: Iterable[str] | None = None,
) -> JsonDict:
    try:
        from scripts.state_manager_core import StateManager

        state = StateManager(memory_dir).record_step(
            action=action,
            mode=mode,
            detected=detected,
            cleaned=cleaned,
            placed=placed,
            service_completed=service_completed,
            success=success,
            failure_reason=failure_reason,
        )
        return {
            "status": "success",
            "result_type": "state_manager_step_recorded",
            "step_count": state.patrol.get("step_count"),
            "mode": state.patrol.get("mode"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "result_type": "error_state_manager_step_failed",
            "message": str(exc),
        }


def update_object_memory_picked(memory_dir: Path, *, candidate: JsonDict, step: int, state: JsonDict) -> JsonDict:
    try:
        from scripts.object_memory_core import ObjectMemory

        picked = ObjectMemory(memory_dir).mark_picked(
            track_id=str(state.get("target_track_id") or "") or None,
            candidate=candidate,
            step=step,
        )
        if isinstance(picked, dict) and picked.get("track_id"):
            state["held_object_track_id"] = picked.get("track_id")
        return {
            "status": "success",
            "result_type": "object_memory_pickup_marked",
            "event": picked,
        }
    except Exception as exc:
        return {
            "status": "error",
            "result_type": "error_object_memory_pickup_failed",
            "message": str(exc),
        }


def update_object_memory_placed(
    memory_dir: Path,
    *,
    held_track_id: str,
    receptacle_track_id: str,
    candidate: JsonDict,
    step: int,
) -> JsonDict:
    try:
        from scripts.object_memory_core import ObjectMemory

        result = ObjectMemory(memory_dir).mark_placed(
            held_track_id=held_track_id or None,
            receptacle_track_id=receptacle_track_id or None,
            receptacle_candidate=candidate,
            step=step,
        )
        return {
            "status": "success",
            "result_type": "object_memory_place_marked",
            "events": as_list(as_dict(result).get("events")),
        }
    except Exception as exc:
        return {
            "status": "error",
            "result_type": "error_object_memory_place_failed",
            "message": str(exc),
        }


def remove_place_precheck_cache(memory_dir: Path) -> bool:
    path = Path(memory_dir) / PLACE_PRECHECK_CACHE_NAME
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def sync_pickup_success(
    *,
    memory_dir: Path,
    context: JsonDict,
    candidate: JsonDict,
    execution: JsonDict,
) -> JsonDict:
    step = current_step_count(memory_dir)
    state = load_service_state(memory_dir)
    holding_after = bool(execution.get("holding_object", True))
    label = candidate_label(candidate) or "held_object"
    object_memory = {}

    if holding_after:
        set_held_object_context(state, candidate)
        object_memory = update_object_memory_picked(memory_dir, candidate=candidate, step=step, state=state)
    else:
        clear_held_object_context(state)

    state["receptacle_alignment_streak"] = 0
    set_service_phase(
        state,
        "VERIFY_HOLDING",
        reason="pickup_executed",
        candidate=candidate,
        step=step,
        holding_object=holding_after,
    )
    if holding_after:
        set_service_phase(
            state,
            "SEARCH_RECEPTACLE",
            reason="pickup_verified_inventory_holding",
            candidate=candidate,
            step=step,
            holding_object=True,
            increment_subgoal=True,
        )
    save_service_state(state, memory_dir)

    state_manager = record_state_manager_step(
        memory_dir=memory_dir,
        action="pick-object",
        mode="SERVICE",
        success=holding_after,
        failure_reason=None if holding_after else "pickup_success_but_inventory_empty",
        detected=[label],
    )
    return {
        "status": "success" if holding_after else "warning",
        "result_type": "pickup_state_synchronized",
        "service_task_state_path": str(Path(memory_dir) / SERVICE_TASK_STATE_NAME),
        "holding_object": holding_after,
        "held_object_track_id": state.get("held_object_track_id"),
        "object_memory": object_memory,
        "state_manager": state_manager,
        "context_generated_at": context.get("generated_at"),
    }


def sync_place_success(
    *,
    memory_dir: Path,
    context: JsonDict,
    candidate: JsonDict,
    execution: JsonDict,
) -> JsonDict:
    step = current_step_count(memory_dir)
    state = load_service_state(memory_dir)
    holding_after = bool(execution.get("holding_object", False))
    placed_label = str(
        state.get("target_raw_label")
        or state.get("held_object_raw_label")
        or state.get("target_label")
        or state.get("held_object_label")
        or "held_object"
    )
    receptacle_label = str(
        candidate.get("raw_label")
        or candidate.get("label")
        or state.get("receptacle_raw_label")
        or state.get("receptacle_label")
        or "receptacle"
    )
    held_track_id = str(state.get("held_object_track_id") or state.get("target_track_id") or "").strip()
    receptacle_track_id = str(state.get("receptacle_track_id") or candidate_track_id(candidate) or "").strip()

    state["holding_object"] = holding_after
    state["receptacle_alignment_streak"] = 0
    set_service_phase(
        state,
        "VERIFY_TASK_DONE",
        reason="place_executed",
        candidate=candidate,
        step=step,
        holding_object=holding_after,
    )

    if holding_after:
        save_service_state(state, memory_dir)
        state_manager = record_state_manager_step(
            memory_dir=memory_dir,
            action="place-object",
            mode="SERVICE",
            success=False,
            failure_reason="place_success_but_inventory_still_holding",
        )
        return {
            "status": "warning",
            "result_type": "place_state_not_finalized_inventory_still_holding",
            "service_task_state_path": str(Path(memory_dir) / SERVICE_TASK_STATE_NAME),
            "holding_object": True,
            "state_manager": state_manager,
            "context_generated_at": context.get("generated_at"),
        }

    object_memory = update_object_memory_placed(
        memory_dir,
        held_track_id=held_track_id,
        receptacle_track_id=receptacle_track_id,
        candidate=candidate,
        step=step,
    )
    clear_held_object_context(state)
    completion = f"{placed_label}->{receptacle_label}"
    completed = state.setdefault("completed_subgoals", [])
    if not isinstance(completed, list):
        completed = []
    completed.append(
        {
            "step": step,
            "time": now_iso(),
            "object": placed_label,
            "receptacle": receptacle_label,
            "summary": completion,
        }
    )
    state["completed_subgoals"] = completed[-80:]
    suppress_recently_placed_track(state, track_id=held_track_id, label=placed_label, step=step)
    reset_pick_place_locks(state)
    set_service_phase(
        state,
        SERVICE_INITIAL_PHASE,
        reason="place_verified_inventory_empty_continue_patrol",
        candidate=candidate,
        step=step,
        holding_object=False,
        increment_subgoal=True,
    )
    save_service_state(state, memory_dir)
    cache_removed = remove_place_precheck_cache(memory_dir)

    state_manager = record_state_manager_step(
        memory_dir=memory_dir,
        action="place-object",
        mode="SERVICE",
        success=True,
        placed=[placed_label],
        service_completed=[completion],
    )
    return {
        "status": "success",
        "result_type": "place_state_synchronized",
        "service_task_state_path": str(Path(memory_dir) / SERVICE_TASK_STATE_NAME),
        "holding_object": False,
        "placed_object": placed_label,
        "receptacle": receptacle_label,
        "completion": completion,
        "precheck_cache_removed": cache_removed,
        "object_memory": object_memory,
        "state_manager": state_manager,
        "context_generated_at": context.get("generated_at"),
    }


def sync_service_failure(
    *,
    memory_dir: Path,
    action: str,
    candidate: JsonDict,
    execution: JsonDict,
) -> JsonDict:
    step = current_step_count(memory_dir)
    state = load_service_state(memory_dir)
    reason = failure_reason_from_execution(execution, f"{action}_failed")
    append_service_history(
        state,
        "service_action_failed",
        step=step,
        action=action,
        reason=reason,
        candidate=short_candidate(candidate),
    )
    state["last_reason"] = reason
    save_service_state(state, memory_dir)
    state_manager = record_state_manager_step(
        memory_dir=memory_dir,
        action=action,
        mode="SERVICE",
        success=False,
        failure_reason=reason,
    )
    return {
        "status": "success",
        "result_type": "service_failure_state_recorded",
        "reason": reason,
        "state_manager": state_manager,
    }


def sync_clean_result(*, memory_dir: Path, execution: JsonDict, success: bool) -> JsonDict:
    label = str(execution.get("label") or execution.get("target_label") or execution.get("object_label") or "").strip()
    state_manager = record_state_manager_step(
        memory_dir=memory_dir,
        action="clean-garbage",
        mode="CLEAN",
        success=success,
        failure_reason=None if success else failure_reason_from_execution(execution, "clean_failed"),
        cleaned=[label] if success and label else None,
    )
    return {
        "status": "success" if state_manager.get("status") == "success" else "warning",
        "result_type": "clean_state_synchronized",
        "state_manager": state_manager,
    }


def sync_move_result(*, memory_dir: Path, action: str, execution: JsonDict, success: bool) -> JsonDict:
    state_manager = record_state_manager_step(
        memory_dir=memory_dir,
        action=action,
        mode="SERVICE",
        success=success,
        failure_reason=None if success else failure_reason_from_execution(execution, "move_failed"),
    )
    return {
        "status": "success" if state_manager.get("status") == "success" else "warning",
        "result_type": "move_state_synchronized",
        "state_manager": state_manager,
    }


def sync_option_result(
    *,
    context: JsonDict,
    option: JsonDict,
    candidate: JsonDict | None,
    execution: JsonDict,
    success: bool,
    memory_dir: Path = MEMORY_DIR,
) -> JsonDict:
    kind = str(option.get("kind") or "")
    action = str(option.get("action") or "")
    memory_dir = Path(memory_dir)

    if kind == "move_action":
        return sync_move_result(memory_dir=memory_dir, action=action, execution=execution, success=success)
    if kind == "clean_action":
        return sync_clean_result(memory_dir=memory_dir, execution=execution, success=success)
    if kind != "service_action":
        return {
            "status": "skipped",
            "result_type": "option_state_sync_not_required",
            "kind": kind,
        }

    candidate_dict = as_dict(candidate)
    if action == "pick-object":
        if success:
            return sync_pickup_success(
                memory_dir=memory_dir,
                context=context,
                candidate=candidate_dict,
                execution=execution,
            )
        return sync_service_failure(
            memory_dir=memory_dir,
            action=action,
            candidate=candidate_dict,
            execution=execution,
        )
    if action == "place-object":
        if success:
            return sync_place_success(
                memory_dir=memory_dir,
                context=context,
                candidate=candidate_dict,
                execution=execution,
            )
        return sync_service_failure(
            memory_dir=memory_dir,
            action=action,
            candidate=candidate_dict,
            execution=execution,
        )
    return {
        "status": "skipped",
        "result_type": "unsupported_service_action_state_sync",
        "action": action,
    }
