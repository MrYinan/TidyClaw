#!/usr/bin/env python3
"""
Programmatic patrol runner for the household service robot workspace.

This runner keeps OpenClaw as the task/skill framework, but moves the long
execution loop out of the chat context. Each step still follows the project
contract:

get-vision -> perceive-scene-yolo/analyze-scene-opencv -> decide
-> one physical action -> verify -> update state-manager.

Default behavior runs one short segment. Use --continuous for a long patrol.
In tidy mode, a successful pickup/place is a service subgoal, not room completion.
机器人是【一步一记录】，每执行一个动作（一步），就立刻把状态写入 JSON 文件
分段的作用：
每段结束必做 3 次安全检查（你代码里的 run () 函数）：
    任务完成 / 步数满了吗？
    达到最大分段数了吗？
    要不要继续跑？
    → 随时能停，不会无限卡死。
3. 好控制：支持 “断续运行”
    分段可以实现：
    跑 1 段就停（默认行为）
    连续跑 N 段（加--continuous）
    限制最多跑 10 段（加--max-segments 10）
    如果不分段，只能一直跑停不下来，根本没法调试。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"

sys.path.insert(0, str(REPO_ROOT))
from scripts.state_manager_core import DEFAULT_MAX_STEPS, DEFAULT_ROOM, StateManager, StateSnapshot  # noqa: E402
from scripts.navigation_memory_core import NavigationMemory  # noqa: E402
from scripts.perception_action_validator import (  # noqa: E402
    DEFAULT_BACKEND_BASE_URL,
    short_backend_candidate,
    validate_clean_target_with_backend,
)


MOVE_ACTIONS = {"MoveAhead", "MoveBack", "RotateLeft", "RotateRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
OPPOSITE_ROTATION = {"RotateLeft": "RotateRight", "RotateRight": "RotateLeft"}
NON_RETRYABLE_CLEAN_ERRORS = {
    "error_no_target_in_front",
    "error_no_cleanable_target",
    "error_not_floor_level",
}
PLACE_RETARGET_ERRORS = {
    "error_no_receptacle",
    "error_no_receptacle_in_front",
    "error_receptacle_not_centered",
    "error_target_not_receptacle",
    "error_visual_receptacle_not_grounded",
    "error_visual_receptacle_instance_mismatch",
    "error_visual_receptacle_ambiguous",
    "error_place_failed",
    "error_place_inventory_not_empty",
    "error_place_no_reachable_point",
    "error_place_object_missing_after_action",
    "error_place_position_unreachable",
    "error_place_receptacle_mismatch",
}
PICKUP_RETARGET_ERRORS = {
    "error_no_pickup_target_in_front",
    "error_visual_pickup_target_not_grounded",
    "error_visual_pickup_instance_mismatch",
    "error_visual_pickup_ambiguous",
    "error_target_not_pickupable",
}


SERVICE_TASK_STATE_PATH = MEMORY_DIR / "service-task-state.json"
SERVICE_INITIAL_PHASE = "SEARCH_PICKUP_TARGET"
SERVICE_DONE_PHASE = "TASK_DONE"
SERVICE_PICKUP_PHASES = {
    "SEARCH_PICKUP_TARGET",
    "LOCK_PICKUP_TARGET",
    "ALIGN_PICKUP_TARGET",
    "PICK_OBJECT",
    "VERIFY_HOLDING",
}
SERVICE_PLACE_PHASES = {
    "SEARCH_RECEPTACLE",
    "LOCK_RECEPTACLE",
    "ALIGN_RECEPTACLE",
    "PLACE_OBJECT",
    "VERIFY_TASK_DONE",
}


GET_VISION_SCRIPT = REPO_ROOT / "skills" / "get-vision" / "scripts" / "get_vision.py"
OPENCV_ANALYZE_SCRIPT = REPO_ROOT / "skills" / "analyze-scene-opencv" / "scripts" / "analyze_scene_opencv.py"
YOLO_ANALYZE_SCRIPT = REPO_ROOT / "skills" / "perceive-scene-yolo" / "scripts" / "perceive_scene_yolo.py"
MOVE_SCRIPT = REPO_ROOT / "skills" / "move-robot" / "scripts" / "move_robot.py"
CLEAN_SCRIPT = REPO_ROOT / "skills" / "clean-garbage" / "scripts" / "clean_garbage.py"
PICK_SCRIPT = REPO_ROOT / "skills" / "pick-object" / "scripts" / "pick_object.py"
PLACE_SCRIPT = REPO_ROOT / "skills" / "place-object" / "scripts" / "place_object.py"
YOLO_SERVICE_URL = os.getenv("ROBOT_YOLO_SERVICE_URL", "http://127.0.0.1:5055/analyze")

PERCEPTION_BACKENDS = {
    "opencv": OPENCV_ANALYZE_SCRIPT,
    "yolo": YOLO_ANALYZE_SCRIPT,
}


JsonDict = Dict[str, Any]


@dataclass
class ScriptResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    data: JsonDict

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and isinstance(self.data, dict)


@dataclass
class Decision:
    kind: str
    action: str
    mode: str
    reason: str
    candidate: Optional[JsonDict] = None


@dataclass
class StepOutcome:
    step_recorded: bool
    stop_requested: bool
    action: Optional[str]
    success: bool
    reason: str


@dataclass
class RunnerConfig:
    start: bool
    no_reset: bool
    room: str
    max_steps: Optional[int]
    segment_steps: int
    continuous: bool
    sleep_seconds: float
    dry_run: bool
    status: bool
    max_segments: Optional[int]
    timeout_seconds: int
    clean_validation: str
    perception_backend: str
    task_mode: str
    interaction_grounding: str
    pickup_surface_policy: str
    pickup_target_labels: Tuple[str, ...]
    verbose: bool
    quiet: bool


def json_print(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def hidden_startupinfo() -> Optional[subprocess.STARTUPINFO]:
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


def parse_json_output(text: str) -> JsonDict:
    stripped = text.strip()
    if not stripped:
        return {}

    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        pass

    for line in reversed([line.strip() for line in stripped.splitlines() if line.strip()]):
        try:
            data = json.loads(line)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(stripped[start : end + 1])
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            pass

    return {}


def run_script(script: Path, args: Sequence[str], timeout_seconds: int) -> ScriptResult:
    command = [sys.executable, str(script), *args]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
            check=False,
            startupinfo=hidden_startupinfo(),
            creationflags=hidden_creationflags(),
        )
        data = parse_json_output(completed.stdout)
        if not data and completed.stderr:
            data = parse_json_output(completed.stderr)
        return ScriptResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            data=data,
        )
    except subprocess.TimeoutExpired as exc:
        return ScriptResult(
            command=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            data={
                "status": "error",
                "result_type": "error_script_timeout",
                "message": f"script timed out after {timeout_seconds}s",
            },
        )


def run_yolo_service(image_path: str, timeout_seconds: int) -> ScriptResult:
    payload = json.dumps({"image": image_path}, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(
        YOLO_SERVICE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=max(1, timeout_seconds)) as response:
            text = response.read().decode("utf-8", errors="replace")
            data = parse_json_output(text)
            return ScriptResult(
                command=["yolo-service", YOLO_SERVICE_URL],
                returncode=0 if response.status < 400 else 1,
                stdout=text,
                stderr="",
                data=data,
            )
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        data = parse_json_output(text)
        if not data:
            data = {
                "status": "error",
                "result_type": "error_yolo_service_http",
                "message": text or str(exc),
                "http_status": exc.code,
            }
        return ScriptResult(
            command=["yolo-service", YOLO_SERVICE_URL],
            returncode=1,
            stdout=text,
            stderr=str(exc),
            data=data,
        )
    except (OSError, TimeoutError, urlerror.URLError) as exc:
        return ScriptResult(
            command=["yolo-service", YOLO_SERVICE_URL],
            returncode=1,
            stdout="",
            stderr=str(exc),
            data={
                "status": "error",
                "result_type": "error_yolo_service_unavailable",
                "message": str(exc),
            },
        )


def truthy(value: Any) -> bool:
    return bool(value)


def get_nested_number(data: JsonDict, *keys: str, default: float = 0.0) -> float:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    try:
        return float(node)
    except (TypeError, ValueError):
        return default


def candidate_key(candidate: JsonDict) -> str:
    center = candidate.get("center") if isinstance(candidate.get("center"), dict) else {}
    bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
    cx_bucket = int(get_nested_number({"center": center}, "center", "x") // 48)
    cy_bucket = int(get_nested_number({"center": center}, "center", "y") // 48)
    area_bucket = int(float(candidate.get("area", 0.0)) // 250)
    return "|".join(
        [
            str(candidate.get("label", "unknown")),
            str(candidate.get("position_hint", "unknown")),
            str(cx_bucket),
            str(cy_bucket),
            str(area_bucket),
            str(int(float(bbox.get("w", 0) or 0) // 12)),
            str(int(float(bbox.get("h", 0) or 0) // 12)),
        ]
    )


def candidate_family_key(candidate: JsonDict) -> str:
    return "|".join(
        [
            "family",
            str(candidate.get("task_semantic_class", "unknown")),
            str(candidate.get("raw_label") or candidate.get("label") or "unknown").lower(),
            str(candidate.get("position_hint") or "unknown"),
        ]
    )


def detected_labels(analysis: JsonDict) -> List[str]:
    labels: List[str] = []
    candidates = []
    candidates.extend(analysis.get("trash_candidates", []) or [])
    candidates.extend(analysis.get("service_candidates", []) or [])
    candidates.extend(analysis.get("receptacle_candidates", []) or [])
    for candidate in candidates:
        if isinstance(candidate, dict):
            label = str(candidate.get("raw_label") or candidate.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def has_service_target(analysis: JsonDict) -> bool:
    if bool(analysis.get("pickup_target_detected", False)):
        return True
    if bool(analysis.get("place_receptacle_detected", False)):
        return True
    for key in ("service_candidates", "receptacle_candidates"):
        candidates = analysis.get(key, []) or []
        if isinstance(candidates, list) and any(isinstance(item, dict) for item in candidates):
            return True
    return False


def view_signature(vision: JsonDict, analysis: JsonDict) -> str:
    """Build a repeated-view signature without simulator pose.

    V1 used AI2-THOR position/rotation/cameraHorizon here. V2 keeps repeated
    view detection online-safe by relying on RGB-derived analysis only.
    """
    occupancy = analysis.get("occupancy") if isinstance(analysis.get("occupancy"), dict) else {}

    candidates = []
    signature_candidates = []
    signature_candidates.extend(analysis.get("trash_candidates", []) or [])
    signature_candidates.extend(analysis.get("service_candidates", []) or [])
    signature_candidates.extend(analysis.get("receptacle_candidates", []) or [])
    for candidate in signature_candidates:
        if not isinstance(candidate, dict):
            continue
        center = candidate.get("center") if isinstance(candidate.get("center"), dict) else {}
        candidates.append(
            [
                candidate.get("raw_label") or candidate.get("label"),
                candidate.get("task_semantic_class"),
                candidate.get("position_hint"),
                int(float(center.get("x", 0) or 0) // 48),
                int(float(center.get("y", 0) or 0) // 48),
            ]
        )

    signature = {
        "observation_contract": vision.get("observation_contract"),
        "occupancy": {key: round(float(value), 2) for key, value in occupancy.items()},
        "candidates": sorted(candidates),
        "obstacle_ahead": bool(analysis.get("obstacle_ahead", False)),
        "open_directions": sorted(list(analysis.get("open_directions", []) or [])),
    }
    return compact_json(signature)


def base_action(action: Optional[str]) -> Optional[str]:
    if not action:
        return None
    action = str(action)
    for known in [
        "clean-garbage",
        "pick-object",
        "place-object",
        "MoveAhead",
        "MoveBack",
        "RotateLeft",
        "RotateRight",
    ]:
        if action.startswith(known):
            return known
    return action


class PatrolRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.manager = StateManager(MEMORY_DIR)
        self.navigation = NavigationMemory(MEMORY_DIR)
        self.recent_actions: List[str] = []
        self.last_view_signature: Optional[str] = None
        self.suppressed_until_step: Dict[str, int] = {}
        self.clean_failures: Dict[str, int] = {}
        self.alignment_attempts: Dict[str, int] = {}
        self.segment_history: List[JsonDict] = []
        self.current_segment_stats = self._new_segment_stats()
        self.recent_results: List[Tuple[str, bool]] = []
        self.consecutive_action_failures = 0
        self.pending_navigation_recommendation: Optional[JsonDict] = None
        self.holding_object = False
        self.service_failures: Dict[str, int] = {}
        self.service_state = self.load_service_task_state()
        self.pending_service_completions: List[str] = []
        self.pending_placed_objects: List[str] = []

        try:
            state = self.manager.load_state()
            last_action = base_action(state.patrol.get("last_action"))
            if last_action:
                self.recent_actions.append(last_action)
        except Exception:
            pass

    def default_service_task_state(self) -> JsonDict:
        return {
            "schema_version": 1,
            "task_style": "alfred_like_pick_place_metadata_hidden_executor",
            "interaction_grounding": self.config.interaction_grounding,
            "pickup_surface_policy": self.config.pickup_surface_policy,
            "pickup_target_labels": list(self.config.pickup_target_labels),
            "phase": SERVICE_INITIAL_PHASE,
            "subgoal_index": 0,
            "target_label": None,
            "target_raw_label": None,
            "target_signature": None,
            "target_last_seen_step": None,
            "target_lost_scan_count": 0,
            "receptacle_label": None,
            "receptacle_raw_label": None,
            "receptacle_signature": None,
            "receptacle_last_seen_step": None,
            "receptacle_lost_scan_count": 0,
            "holding_object": False,
            "phase_attempts": 0,
            "target_attempts": 0,
            "receptacle_attempts": 0,
            "receptacle_alignment_streak": 0,
            "completed_subgoals": [],
            "recently_placed_labels": {},
            "last_reason": "initialized",
            "last_update": datetime.now().astimezone().isoformat(timespec="seconds"),
            "history": [],
        }

    def load_service_task_state(self) -> JsonDict:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return self.default_service_task_state()
        if not SERVICE_TASK_STATE_PATH.exists():
            return self.default_service_task_state()
        try:
            data = json.loads(SERVICE_TASK_STATE_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            return self.default_service_task_state()
        if not isinstance(data, dict):
            return self.default_service_task_state()
        state = self.default_service_task_state()
        state.update(data)
        if not isinstance(state.get("history"), list):
            state["history"] = []
        if not isinstance(state.get("completed_subgoals"), list):
            state["completed_subgoals"] = []
        if not isinstance(state.get("recently_placed_labels"), dict):
            state["recently_placed_labels"] = {}
        return state

    def save_service_task_state(self) -> None:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return
        if self.config.dry_run:
            return
        self.service_state["last_update"] = datetime.now().astimezone().isoformat(timespec="seconds")
        SERVICE_TASK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.service_state, ensure_ascii=False, indent=2) + "\n"
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            tmp_path = SERVICE_TASK_STATE_PATH.with_name(
                f".{SERVICE_TASK_STATE_PATH.name}.{os.getpid()}.{attempt}.tmp"
            )
            try:
                tmp_path.write_text(payload, encoding="utf-8")
                os.replace(str(tmp_path), str(SERVICE_TASK_STATE_PATH))
                return
            except OSError as exc:
                last_error = exc
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))

        try:
            with open(str(SERVICE_TASK_STATE_PATH), "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
            last_error = None
        except OSError as exc:
            last_error = exc

        if last_error is not None:
            self.emit(
                "service_state_save_failed",
                {
                    "path": str(SERVICE_TASK_STATE_PATH),
                    "error": str(last_error),
                    "error_type": type(last_error).__name__,
                },
            )

    def reset_service_task_state(self, *, reason: str) -> None:
        self.service_state = self.default_service_task_state()
        self.service_state["last_reason"] = reason
        self.service_state["holding_object"] = bool(self.holding_object)
        self.append_service_history("reset", reason=reason)
        self.save_service_task_state()
        self.emit("service_task_reset", {"phase": self.service_state.get("phase"), "reason": reason})

    def append_service_history(self, event: str, **payload: Any) -> None:
        history = self.service_state.setdefault("history", [])
        if not isinstance(history, list):
            history = []
            self.service_state["history"] = history
        entry = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": self.current_step_count(),
            "event": event,
        }
        entry.update(payload)
        history.append(entry)
        self.service_state["history"] = history[-80:]

    def set_service_phase(
        self,
        phase: str,
        *,
        reason: str,
        candidate: Optional[JsonDict] = None,
        increment_subgoal: bool = False,
    ) -> None:
        old_phase = str(self.service_state.get("phase") or SERVICE_INITIAL_PHASE)
        if old_phase != phase:
            self.service_state["phase_attempts"] = 0
            if increment_subgoal:
                self.service_state["subgoal_index"] = int(self.service_state.get("subgoal_index", 0) or 0) + 1
        self.service_state["phase"] = phase
        self.service_state["last_reason"] = reason
        self.service_state["holding_object"] = bool(self.holding_object)
        self.append_service_history(
            "phase_transition",
            old_phase=old_phase,
            phase=phase,
            reason=reason,
            candidate=self.short_candidate(candidate),
        )
        self.save_service_task_state()
        if old_phase != phase:
            self.emit(
                "service_phase",
                {
                    "from": old_phase,
                    "to": phase,
                    "subgoal_index": self.service_state.get("subgoal_index"),
                    "reason": reason,
                    "candidate": self.short_candidate(candidate),
                },
            )

    def service_phase(self) -> str:
        return str(self.service_state.get("phase") or SERVICE_INITIAL_PHASE)

    def candidate_label(self, candidate: Optional[JsonDict]) -> str:
        if not isinstance(candidate, dict):
            return ""
        return str(candidate.get("raw_label") or candidate.get("label") or "").strip()

    def service_lock_matches(self, candidate: JsonDict, *, role: str) -> bool:
        label_key = "target_raw_label" if role == "pickup" else "receptacle_raw_label"
        fallback_key = "target_label" if role == "pickup" else "receptacle_label"
        locked_label = str(self.service_state.get(label_key) or self.service_state.get(fallback_key) or "").strip()
        if not locked_label:
            return False
        candidate_label = self.candidate_label(candidate)
        return candidate_label.lower() == locked_label.lower()

    def lock_service_candidate(self, candidate: JsonDict, *, role: str, reason: str) -> None:
        signature = candidate_key(candidate)
        label = str(candidate.get("label") or "")
        raw_label = str(candidate.get("raw_label") or label)
        step = self.current_step_count()
        if role == "pickup":
            self.service_state["target_label"] = label
            self.service_state["target_raw_label"] = raw_label
            self.service_state["target_signature"] = signature
            self.service_state["target_last_seen_step"] = step
            self.service_state["target_lost_scan_count"] = 0
            self.service_state["target_attempts"] = int(self.service_state.get("target_attempts", 0) or 0) + 1
            phase = "LOCK_PICKUP_TARGET"
        else:
            self.service_state["receptacle_label"] = label
            self.service_state["receptacle_raw_label"] = raw_label
            self.service_state["receptacle_signature"] = signature
            self.service_state["receptacle_last_seen_step"] = step
            self.service_state["receptacle_lost_scan_count"] = 0
            self.service_state["receptacle_attempts"] = int(self.service_state.get("receptacle_attempts", 0) or 0) + 1
            phase = "LOCK_RECEPTACLE"
        self.set_service_phase(phase, reason=reason, candidate=candidate)

    def clear_service_lock(self, *, role: str, reason: str) -> None:
        if role == "pickup":
            self.service_state["target_label"] = None
            self.service_state["target_raw_label"] = None
            self.service_state["target_signature"] = None
            self.service_state["target_last_seen_step"] = None
            self.service_state["target_lost_scan_count"] = 0
            self.service_state["target_attempts"] = 0
            next_phase = SERVICE_INITIAL_PHASE
        else:
            self.service_state["receptacle_label"] = None
            self.service_state["receptacle_raw_label"] = None
            self.service_state["receptacle_signature"] = None
            self.service_state["receptacle_last_seen_step"] = None
            self.service_state["receptacle_lost_scan_count"] = 0
            self.service_state["receptacle_attempts"] = 0
            next_phase = "SEARCH_RECEPTACLE"
        self.set_service_phase(next_phase, reason=reason)

    def max_locked_scan_steps(self, *, role: str) -> int:
        env_name = "ROBOT_MAX_LOCKED_PICKUP_SCAN_STEPS" if role == "pickup" else "ROBOT_MAX_LOCKED_PLACE_SCAN_STEPS"
        default = "3" if role == "pickup" else "4"
        try:
            return max(0, int(os.getenv(env_name, default)))
        except (TypeError, ValueError):
            return int(default)

    def locked_lost_steps(self, *, role: str) -> int:
        key = "target_last_seen_step" if role == "pickup" else "receptacle_last_seen_step"
        last_seen = self.service_state.get(key)
        try:
            return max(0, self.current_step_count() - int(last_seen))
        except (TypeError, ValueError):
            return self.max_locked_scan_steps(role=role) + 1

    def increment_locked_scan_count(self, *, role: str) -> int:
        key = "target_lost_scan_count" if role == "pickup" else "receptacle_lost_scan_count"
        try:
            count = int(self.service_state.get(key, 0) or 0) + 1
        except (TypeError, ValueError):
            count = 1
        self.service_state[key] = count
        self.save_service_task_state()
        return count

    def recently_placed_label_blocked(self, candidate: JsonDict) -> bool:
        raw_label = str(candidate.get("raw_label") or candidate.get("label") or "").strip().lower()
        if not raw_label:
            return False
        blocked = self.service_state.get("recently_placed_labels")
        if not isinstance(blocked, dict):
            return False
        until_step = int(blocked.get(raw_label, -1) or -1)
        if until_step < self.current_step_count():
            blocked.pop(raw_label, None)
            self.service_state["recently_placed_labels"] = blocked
            self.save_service_task_state()
            return False
        return True

    def suppress_recently_placed_label(self, label: str, *, steps: int = 12) -> None:
        normalized = str(label or "").strip().lower()
        if not normalized:
            return
        blocked = self.service_state.setdefault("recently_placed_labels", {})
        if not isinstance(blocked, dict):
            blocked = {}
            self.service_state["recently_placed_labels"] = blocked
        blocked[normalized] = self.current_step_count() + max(1, int(steps))
        self.save_service_task_state()

    def pickup_candidate_is_actionable_or_promising(self, candidate: JsonDict) -> bool:
        """Keep tidy mode from chasing visible-but-not-executable pickup boxes."""
        if not self.pickup_task_filter_allowed(candidate):
            return False
        if not truthy(candidate.get("reachable")):
            return False

        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        position_hint = str(candidate.get("position_hint") or "")
        surface_hint = str(candidate.get("surface_hint") or "")
        floor_like = surface_hint == "floor" or (
            truthy(candidate.get("is_floor_level")) and bottom_y_ratio >= 0.78
        )

        if truthy(candidate.get("pickup_now")):
            return True

        if floor_like and confidence >= 0.45:
            return True

        if position_hint == "front-center" and confidence >= 0.55 and area_ratio >= 0.0015:
            return bottom_y_ratio >= 0.78

        if (
            self.config.interaction_grounding == "metadata-hidden"
            and self.config.pickup_surface_policy == "any-surface"
            and surface_hint == "surface_or_elevated"
            and confidence >= 0.70
            and area_ratio >= 0.0015
            and bottom_y_ratio >= 0.35
        ):
            return True

        return False

    def visible_service_candidates(self, analysis: JsonDict, *, task_class: str) -> List[JsonDict]:
        pools: List[Any] = []
        if task_class == "pickup_target":
            pools.append(analysis.get("best_pickup_candidate"))
        elif task_class == "place_receptacle":
            pools.append(analysis.get("best_receptacle_candidate"))
        pools.extend(analysis.get("service_candidates", []) or [])
        pools.extend(analysis.get("receptacle_candidates", []) or [])

        candidates: List[JsonDict] = []
        seen_keys = set()
        for candidate in pools:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("task_semantic_class") != task_class:
                continue
            if self.is_suppressed(candidate):
                continue
            if not truthy(candidate.get("reachable")):
                continue
            if task_class == "pickup_target" and not self.pickup_candidate_is_actionable_or_promising(candidate):
                continue
            if task_class == "pickup_target" and self.recently_placed_label_blocked(candidate):
                continue
            if task_class == "place_receptacle" and self.receptacle_visual_box_ambiguous(candidate):
                continue
            key = candidate_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)
        candidates.sort(key=lambda item: self.service_candidate_score(item, task_class=task_class), reverse=True)
        return candidates

    def service_candidate_score(self, candidate: JsonDict, *, task_class: str) -> float:
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        cx_ratio = float(geometry.get("cx_ratio", 0.5) or 0.5)
        center_bonus = max(0.0, 0.5 - abs(cx_ratio - 0.5))
        value = confidence * 2.0 + min(area_ratio, 0.20) + center_bonus
        if task_class == "pickup_target" and truthy(candidate.get("pickup_now")):
            value += 5.0
        if task_class == "place_receptacle" and truthy(candidate.get("place_now")):
            value += 5.0
        if str(candidate.get("position_hint") or "") == "front-center":
            value += 0.8
        if task_class == "pickup_target" and self.pickup_surface_allowed(candidate):
            value += 0.4
        if task_class == "pickup_target" and truthy(candidate.get("needs_alignment")):
            value -= 0.6
        elif truthy(candidate.get("needs_alignment")):
            value += 0.2
        return value

    def candidate_center_offset(self, candidate: JsonDict) -> float:
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        try:
            return float(geometry.get("cx_ratio", 0.5) or 0.5) - 0.5
        except (TypeError, ValueError):
            return 0.0

    def pickup_label_allowed(self, candidate: JsonDict) -> bool:
        allowed = tuple(label.strip().lower() for label in self.config.pickup_target_labels if label.strip())
        if not allowed:
            return True
        label = str(candidate.get("label") or "").strip().lower()
        raw_label = str(candidate.get("raw_label") or "").strip().lower()
        return label in allowed or raw_label in allowed

    def pickup_surface_allowed(self, candidate: JsonDict) -> bool:
        if truthy(candidate.get("is_support_surface")):
            return False
        policy = str(self.config.pickup_surface_policy or "floor-only").strip().lower()
        if policy == "any-surface":
            return True

        surface_hint = str(candidate.get("surface_hint") or "")
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        try:
            floor_min_bottom = float(os.getenv("ROBOT_PICKUP_FLOOR_MIN_BOTTOM_RATIO", "0.78"))
        except (TypeError, ValueError):
            floor_min_bottom = 0.78
        if surface_hint == "floor":
            return bool(truthy(candidate.get("is_floor_level")) and bottom_y_ratio >= floor_min_bottom)
        return bool(
            truthy(candidate.get("is_floor_level"))
            and surface_hint != "surface_or_elevated"
            and bottom_y_ratio >= floor_min_bottom
        )

    def pickup_task_filter_allowed(self, candidate: JsonDict) -> bool:
        return self.pickup_label_allowed(candidate) and self.pickup_surface_allowed(candidate)

    def receptacle_front_edge_candidate(self, candidate: JsonDict) -> bool:
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return False
        if str(candidate.get("position_hint") or "") != "front-center":
            return False
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        try:
            cx_ratio = float(geometry.get("cx_ratio", 0.5) or 0.5)
        except (TypeError, ValueError):
            cx_ratio = 0.5
        try:
            center_y_ratio = float(geometry.get("cy_ratio", candidate.get("center_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            center_y_ratio = 0.0
        try:
            bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            bottom_y_ratio = 0.0
        try:
            area_ratio = float(geometry.get("area_ratio", candidate.get("area_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            area_ratio = 0.0
        try:
            width_ratio = float(geometry.get("width_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            width_ratio = 0.0
        if width_ratio <= 0.0:
            bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
            try:
                width_ratio = float(bbox.get("w", 0.0) or 0.0) / max(
                    1.0,
                    float(os.getenv("ROBOT_VIEW_WIDTH", "600")),
                )
            except (TypeError, ValueError):
                width_ratio = 0.0

        try:
            min_bottom = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MIN_BOTTOM_RATIO", "0.88"))
        except (TypeError, ValueError):
            min_bottom = 0.88
        try:
            min_center_y = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MIN_CENTER_Y_RATIO", "0.58"))
        except (TypeError, ValueError):
            min_center_y = 0.58
        try:
            max_area = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MAX_AREA_RATIO", "0.72"))
        except (TypeError, ValueError):
            max_area = 0.72
        try:
            max_width = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MAX_WIDTH_RATIO", "1.01"))
        except (TypeError, ValueError):
            max_width = 1.01

        return bool(
            abs(cx_ratio - 0.5) <= 0.10
            and bottom_y_ratio >= min_bottom
            and center_y_ratio >= min_center_y
            and area_ratio <= max_area
            and width_ratio <= max_width
        )

    def receptacle_visual_box_ambiguous(self, candidate: JsonDict) -> bool:
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return False
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        center_offset = abs(self.candidate_center_offset(candidate))
        try:
            area_ratio = float(geometry.get("area_ratio", candidate.get("area_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            area_ratio = 0.0
        try:
            bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            bottom_y_ratio = 0.0
        try:
            center_y_ratio = float(geometry.get("cy_ratio", candidate.get("center_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            center_y_ratio = 0.0
        try:
            width_ratio = float(geometry.get("width_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            width_ratio = 0.0
        if width_ratio <= 0.0:
            bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
            try:
                width_ratio = float(bbox.get("w", 0.0) or 0.0) / max(
                    1.0,
                    float(os.getenv("ROBOT_VIEW_WIDTH", "600")),
                )
            except (TypeError, ValueError):
                width_ratio = 0.0

        try:
            max_area = float(os.getenv("ROBOT_PLACE_MAX_AREA", "0.45"))
        except (TypeError, ValueError):
            max_area = 0.45
        try:
            max_width = float(os.getenv("ROBOT_PLACE_MAX_WIDTH", "0.995"))
        except (TypeError, ValueError):
            max_width = 0.995
        candidate_broad_front = truthy(candidate.get("broad_front_receptacle"))
        front_edge_receptacle = self.receptacle_front_edge_candidate(candidate)

        broad_front_receptacle = bool(
            candidate_broad_front
            or front_edge_receptacle
            or (
                str(candidate.get("position_hint") or "") == "front-center"
                and center_offset <= 0.10
                and bottom_y_ratio >= 0.66
                and area_ratio <= max_area
            )
        )
        if broad_front_receptacle:
            return False
        if truthy(candidate.get("visual_box_ambiguous")):
            return True
        return bool(area_ratio > max_area or width_ratio > max_width)

    def locked_or_best_service_candidate(self, analysis: JsonDict, *, task_class: str, role: str) -> Optional[JsonDict]:
        candidates = self.visible_service_candidates(analysis, task_class=task_class)
        for candidate in candidates:
            if self.service_lock_matches(candidate, role=role):
                return candidate
        return candidates[0] if candidates else None

    def service_pick_ready(self, candidate: JsonDict) -> bool:
        if not self.pickup_task_filter_allowed(candidate):
            return False
        if not truthy(candidate.get("reachable")):
            return False
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        surface_hint = str(candidate.get("surface_hint") or "")
        position_hint = str(candidate.get("position_hint") or "")
        center_offset = abs(self.candidate_center_offset(candidate))

        if truthy(candidate.get("needs_alignment")):
            return False
        if truthy(candidate.get("support_context_blocked")):
            return False

        try:
            pickup_min_conf = float(os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55"))
        except (TypeError, ValueError):
            pickup_min_conf = 0.55

        if truthy(candidate.get("pickup_now")):
            try:
                pickup_now_max_offset = float(os.getenv("ROBOT_PICKUP_NOW_MAX_CENTER_OFFSET", "0.22"))
            except (TypeError, ValueError):
                pickup_now_max_offset = 0.22
            return bool(
                position_hint == "front-center"
                and center_offset <= pickup_now_max_offset
                and confidence >= pickup_min_conf
            )

        try:
            pick_ready_max_offset = float(os.getenv("ROBOT_PICK_READY_MAX_CENTER_OFFSET", "0.14"))
        except (TypeError, ValueError):
            pick_ready_max_offset = 0.14
        if position_hint != "front-center" or center_offset > pick_ready_max_offset:
            return False

        try:
            small_floor_min_area = float(os.getenv("ROBOT_PICKUP_SMALL_FLOOR_MIN_AREA", "0.0007"))
        except (TypeError, ValueError):
            small_floor_min_area = 0.0007
        try:
            small_floor_bottom_ratio = float(os.getenv("ROBOT_PICKUP_SMALL_FLOOR_BOTTOM_RATIO", "0.88"))
        except (TypeError, ValueError):
            small_floor_bottom_ratio = 0.88
        if (
            surface_hint == "floor"
            and truthy(candidate.get("is_floor_level"))
            and bottom_y_ratio >= small_floor_bottom_ratio
            and area_ratio >= small_floor_min_area
            and confidence >= pickup_min_conf
        ):
            return True

        if self.config.interaction_grounding == "metadata-hidden" and surface_hint == "surface_or_elevated":
            return bool(
                self.config.pickup_surface_policy == "any-surface"
                and confidence >= 0.75
                and area_ratio >= 0.003
                and bottom_y_ratio >= 0.35
            )

        if surface_hint != "floor" and bottom_y_ratio < 0.78:
            return False
        return confidence >= pickup_min_conf and area_ratio >= 0.0015

    def service_place_ready(self, candidate: JsonDict) -> bool:
        if str(candidate.get("position_hint") or "") != "front-center":
            return False
        if not truthy(candidate.get("reachable")):
            return False
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        center_offset = abs(self.candidate_center_offset(candidate))
        if self.receptacle_visual_box_ambiguous(candidate):
            return False
        if self.config.interaction_grounding == "metadata-hidden":
            return bool(
                truthy(candidate.get("is_support_surface"))
                and center_offset <= 0.10
                and bottom_y_ratio >= 0.68
                and confidence >= 0.72
                and area_ratio >= 0.025
            )
        if truthy(candidate.get("place_now")) and center_offset <= 0.10:
            return True
        if truthy(candidate.get("is_support_surface")) and confidence >= 0.65 and center_offset <= 0.10:
            return True
        return confidence >= 0.70 and area_ratio >= 0.015

    def max_receptacle_align_attempts(self) -> int:
        try:
            return max(1, int(os.getenv("ROBOT_MAX_RECEPTACLE_ALIGN_ATTEMPTS", "4")))
        except (TypeError, ValueError):
            return 4

    def service_place_probe_ready(self, candidate: JsonDict) -> bool:
        """Allow one backend-grounded place probe after visual alignment stalls."""
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return False
        if not truthy(candidate.get("reachable")):
            return False
        if not truthy(candidate.get("is_support_surface")):
            return False
        if self.receptacle_visual_box_ambiguous(candidate):
            return False

        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        center_offset = abs(self.candidate_center_offset(candidate))
        position_hint = str(candidate.get("position_hint") or "")
        front_edge_like = bool(
            truthy(candidate.get("front_edge_receptacle"))
            or truthy(candidate.get("broad_front_receptacle"))
            or truthy(candidate.get("place_now"))
        )
        if truthy(candidate.get("needs_approach")) and not front_edge_like:
            return False

        try:
            max_center_offset = float(os.getenv("ROBOT_PLACE_PROBE_MAX_CENTER_OFFSET", "0.18"))
        except (TypeError, ValueError):
            max_center_offset = 0.18
        try:
            min_bottom = float(
                os.getenv(
                    "ROBOT_PLACE_PROBE_MIN_BOTTOM_RATIO",
                    "0.55" if front_edge_like else "0.66",
                )
            )
        except (TypeError, ValueError):
            min_bottom = 0.55 if front_edge_like else 0.66
        try:
            min_confidence = float(
                os.getenv(
                    "ROBOT_PLACE_PROBE_MIN_CONF",
                    "0.70" if front_edge_like else "0.72",
                )
            )
        except (TypeError, ValueError):
            min_confidence = 0.70 if front_edge_like else 0.72
        try:
            min_area = float(os.getenv("ROBOT_PLACE_PROBE_MIN_AREA", "0.018"))
        except (TypeError, ValueError):
            min_area = 0.018

        return bool(
            position_hint in {"front-center", "front-left", "front-right"}
            and center_offset <= max_center_offset
            and bottom_y_ratio >= min_bottom
            and confidence >= min_confidence
            and area_ratio >= min_area
        )

    def suppress_service_candidate(self, candidate: JsonDict, *, reason: str, steps: int = 6) -> None:
        keys = [candidate_key(candidate), candidate_family_key(candidate)]
        until_step = self.current_step_count() + max(1, int(steps))
        for key in keys:
            self.suppressed_until_step[key] = until_step
        self.emit(
            "candidate_suppressed",
            {
                "candidate": self.short_candidate(candidate),
                "result_type": reason,
                "until_step": until_step,
            },
        )

    def _new_segment_stats(self) -> JsonDict:
        return {
            "steps": 0,
            "cleanable_seen": False,
            "new_open_direction_seen": False,
            "failures": 0,
        }

    def emit(self, event: str, payload: Optional[JsonDict] = None) -> None:
        entry: JsonDict = {
            "event": event,
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if payload:
            entry.update(payload)
        if (not self.config.quiet) or event in {
            "patrol_stopped",
            "room_marked_complete",
            "mission_started",
            "dry_run_start_skipped",
        }:
            json_print(entry)
        self.append_daily_log(event, entry)

    def append_daily_log(self, event: str, payload: JsonDict) -> None:
        log_path = MEMORY_DIR / f"{datetime.now().date().isoformat()}-patrol-runner.md"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = f"- {datetime.now().strftime('%H:%M:%S')} `{event}` {compact_json(payload)}\n"
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)

    def run(self) -> int:
        if self.config.status:
            self.print_status()
            return 0

        if self.config.start:
            if self.config.dry_run:
                self.emit("dry_run_start_skipped", {"reason": "--dry-run does not write state"})
            else:
                max_steps = self.config.max_steps or DEFAULT_MAX_STEPS
                state = self.manager.start_mission(
                    room_name=self.config.room,
                    max_steps=max_steps,
                    reset_counters=not self.config.no_reset,
                )
                self.emit(
                    "mission_started",
                    {
                        "room": state.room.get("room_name"),
                        "max_steps": state.patrol.get("max_steps"),
                        "reset": not self.config.no_reset,
                        "task_mode": self.config.task_mode,
                        "interaction_grounding": self.config.interaction_grounding,
                    },
                )
                if not self.config.no_reset:
                    nav_status = self.navigation.reset(room_name=str(state.room.get("room_name") or self.config.room))
                    self.emit(
                        "navigation_memory_reset",
                        {
                            "visited_cell_count": nav_status.get("visited_cell_count"),
                            "coverage_estimate": nav_status.get("coverage_estimate"),
                        },
                    )
                    if str(self.config.task_mode or "clean").strip().lower() == "tidy":
                        self.reset_service_task_state(reason="mission_start")
        elif self.config.max_steps is not None and not self.config.dry_run:
            state = self.manager.load_state()
            state.patrol["max_steps"] = int(self.config.max_steps)
            state.mission["max_steps"] = int(self.config.max_steps)
            self.manager.save_state(state)
            self.emit("max_steps_updated", {"max_steps": self.config.max_steps})

        segment_count = 0
        while True:
            """ 核心 4：循环停止的所有条件（任务终止规则）
                分 两层停止条件，全部在代码里写死：
                    第一层：状态管理器判断（should_continue()，在状态管理程序中）,只要满足任意一条，主循环直接停止：
                            任务被手动关闭（mission disabled）
                            巡逻功能关闭（patrol disabled）
                            房间已清扫完成（room_complete=true）
                            步数达到上限 200 步（max_steps reached）
                            进入终止模式：ROOM_COMPLETE / MISSION_REPORT / DONE
                    第二层：单步执行判断（check_completion_after_step()）,满足任意一条，标记任务完成 / 失败，停止循环：
                            连续动作失败 ≥3 次 → 判定卡死，停止
                            覆盖率 ≥95% + 无未探索区域 → 清扫完成，停止
                            步数达到上限 → 强制停止
                            长时间无新目标 + 重复画面≥3 次 → 无垃圾，停止
                            旋转震荡 / 进退死循环 → 故障，停止"""
            #机器人每执行完一段任务（默认 3 步为 1 段），就会执行一次检查   
            if not self.config.dry_run:#dry_run = 调试模式，程序默认在真实运行模式
                continuation = self.manager.should_continue()
                if not continuation.get("continue", False):
                    self.emit("patrol_stopped", {"reasons": continuation.get("reasons", [])})
                    return 0
            # 达到最大运行片段数 → 强制停止
            ## 只有当【你手动设置了最大片段数】 并且 【运行次数达到了这个数】才会触发
            if self.config.max_segments is not None and segment_count >= self.config.max_segments:
                self.emit("patrol_stopped", {"reasons": ["max_segments reached"]})
                return 0

            segment_count += 1
            self.current_segment_stats = self._new_segment_stats()
            self.emit(
                "segment_started",
                {
                    "segment_index": segment_count,
                    "segment_steps": self.config.segment_steps,
                    "continuous": self.config.continuous,
                    "dry_run": self.config.dry_run,
                    "task_mode": self.config.task_mode,
                    "interaction_grounding": self.config.interaction_grounding,
                },
            )
            segment_stop = self.run_segment(segment_count)
            self.segment_history.append(dict(self.current_segment_stats))
            self.emit(
                "segment_finished",
                {
                    "segment_index": segment_count,
                    "stats": self.current_segment_stats,
                    "stop_requested": segment_stop,
                },
            )

            if segment_stop or not self.config.continuous:
                return 0

            if self.config.sleep_seconds > 0:
                time.sleep(self.config.sleep_seconds)

    def print_status(self) -> None:
        validation = self.manager.validate_state()
        continuation = self.manager.should_continue()
        json_print(
            {
                "status": "success",
                "validation": validation,
                "should_continue": {
                    "continue": continuation.get("continue"),
                    "reasons": continuation.get("reasons", []),
                },
                "service_task_state": self.service_state
                if str(self.config.task_mode or "clean").strip().lower() == "tidy"
                else None,
            }
        )

    def run_segment(self, segment_index: int) -> bool:
        for step_in_segment in range(1, self.config.segment_steps + 1):
            if not self.config.dry_run:
                continuation = self.manager.should_continue()
                if not continuation.get("continue", False):
                    self.emit("segment_stop_condition", {"reasons": continuation.get("reasons", [])})
                    return True

            self.emit(
                "step_started",
                {"segment_index": segment_index, "step_in_segment": step_in_segment},
            )
            outcome = self.run_one_step()
            self.emit(
                "step_finished",
                {
                    "segment_index": segment_index,
                    "step_in_segment": step_in_segment,
                    "action": outcome.action,
                    "success": outcome.success,
                    "reason": outcome.reason,
                    "step_recorded": outcome.step_recorded,
                },
            )
            if outcome.stop_requested:
                return True
        return False

    def run_one_step(self) -> StepOutcome:
        vision_result = self.call_get_vision()
        if not self.skill_success(vision_result, required_field="image_path"):
            return self.handle_perception_failure("vision_failed", vision_result.data)

        image_path = str(vision_result.data.get("image_path", ""))
        analysis_result = self.call_analyze(image_path)
        if not self.analysis_success(analysis_result):
            retry_result = self.call_analyze(image_path)
            if not self.analysis_success(retry_result):
                return self.handle_perception_failure("analysis_failed", retry_result.data)
            analysis_result = retry_result

        vision = vision_result.data
        analysis = analysis_result.data
        self.sync_inventory_state()
        repeated_view = self.update_repeated_view(vision, analysis)
        self.update_segment_stats(analysis)
        self.observe_navigation(vision, analysis)
        self.pending_navigation_recommendation = None

        decision = self.decide(analysis, vision)
        self.emit(
            "decision",
            {
                "action": decision.action,
                "kind": decision.kind,
                "mode": decision.mode,
                "reason": decision.reason,
                "candidate": self.short_candidate(decision.candidate),
            },
        )

        if decision.kind == "stop":
            if not self.config.dry_run:
                self.mark_room_complete(decision.reason)
            return StepOutcome(
                step_recorded=False,
                stop_requested=True,
                action=None,
                success=True,
                reason=decision.reason,
            )

        if self.config.dry_run:
            return StepOutcome(
                step_recorded=False,
                stop_requested=True,
                action=decision.action,
                success=True,
                reason="dry_run_decision_only",
            )

        if decision.kind == "clean":
            execution = self.call_clean()
            success, failure_reason, cleaned = self.verify_clean(execution.data, decision)
            if not success:
                self.register_clean_failure(decision, execution.data)
            action_label = "clean-garbage"
        elif decision.kind == "pick":
            execution = self.call_pick(decision.candidate)
            success, failure_reason = self.verify_service_action(execution.data, expected_result_type="pickup_executed")
            cleaned = []
            action_label = "pick-object"
            if success:
                self.holding_object = True
        elif decision.kind == "place":
            execution = self.call_place(decision.candidate)
            success, failure_reason = self.verify_service_action(execution.data, expected_result_type="place_executed")
            cleaned = []
            action_label = "place-object"
            if success:
                self.holding_object = False
        else:
            execution = self.call_move(decision.action)
            success, failure_reason = self.verify_move(execution.data, decision.action)
            cleaned = []
            action_label = decision.action

        self.update_service_state_after_action(
            decision=decision,
            execution=execution.data,
            success=success,
            failure_reason=failure_reason,
        )

        self.recent_actions.append(action_label)
        self.recent_actions = self.recent_actions[-8:]
        self.record_action_result(action_label, success)

        labels = detected_labels(analysis)
        target_seen = bool(analysis.get("floor_trash_detected", False)) or has_service_target(analysis)
        placed = list(self.pending_placed_objects)
        service_completed = list(self.pending_service_completions)
        self.pending_placed_objects = []
        self.pending_service_completions = []
        state = self.manager.record_step(
            action=action_label,
            mode=decision.mode,
            detected=labels,
            cleaned=cleaned,
            placed=placed,
            service_completed=service_completed,
            success=success,
            failure_reason=failure_reason,
            no_target=not target_seen,
            repeated_view=repeated_view,
        )

        self.emit(
            "state_updated",
            {
                "step_count": state.patrol.get("step_count"),
                "max_steps": state.patrol.get("max_steps"),
                "failed_attempts": state.patrol.get("failed_attempts"),
                "last_action": state.patrol.get("last_action"),
            },
        )

        if self.state_stopped_or_terminal(state):
            self.emit(
                "patrol_stop_observed",
                {
                    "patrol_enabled": state.patrol.get("enabled"),
                    "patrol_mode": state.patrol.get("mode"),
                    "mission_enabled": state.mission.get("enabled"),
                    "mission_mode": state.mission.get("mode"),
                    "step_count": state.patrol.get("step_count"),
                },
            )
            return StepOutcome(
                step_recorded=False,
                stop_requested=True,
                action=action_label,
                success=success,
                reason="mission_stopped",
            )

        self.record_navigation_step(
            action=action_label,
            success=success,
            vision=vision,
            analysis=analysis,
            execution=execution.data,
            failure_reason=failure_reason,
        )

        stop_requested = self.check_completion_after_step(state, analysis)
        return StepOutcome(
            step_recorded=True,
            stop_requested=stop_requested,
            action=action_label,
            success=success,
            reason=failure_reason if not success else "action_verified",
        )

    def state_stopped_or_terminal(self, state: StateSnapshot) -> bool:
        if not bool(state.patrol.get("enabled", False)) or not bool(state.mission.get("enabled", False)):
            return True
        if state.patrol.get("mode") in {"RECOVER", "ROOM_COMPLETE", "MISSION_REPORT", "DONE"}:
            return True
        if state.mission.get("mode") in {"RECOVER", "ROOM_COMPLETE", "MISSION_REPORT", "DONE"}:
            return True
        return False

    def call_get_vision(self) -> ScriptResult:
        result = run_script(GET_VISION_SCRIPT, [], self.config.timeout_seconds)
        self.emit_script_result("get_vision", result)
        return result

    def call_analyze(self, image_path: str) -> ScriptResult:
        backend = str(self.config.perception_backend or "yolo").strip().lower()
        script = PERCEPTION_BACKENDS.get(backend)
        if script is None:
            return ScriptResult(
                command=[],
                returncode=2,
                stdout="",
                stderr="",
                data={
                    "status": "error",
                    "result_type": "invalid_perception_backend",
                    "message": f"Unsupported perception backend: {backend}",
                },
            )
        if backend == "yolo":
            service_mode = str(os.getenv("ROBOT_YOLO_SERVICE_MODE", "auto")).strip().lower()
            if service_mode not in {"0", "false", "no", "off", "disabled"}:
                service_result = run_yolo_service(image_path, self.config.timeout_seconds)
                if service_result.ok and service_result.data.get("status") == "success":
                    self.emit_script_result("analyze_scene_yolo_service", service_result)
                    return service_result
                if service_mode in {"required", "require", "only"}:
                    self.emit_script_result("analyze_scene_yolo_service", service_result)
                    return service_result
        result = run_script(script, ["--image", image_path], self.config.timeout_seconds)
        self.emit_script_result(f"analyze_scene_{backend}", result)
        return result

    def call_move(self, action: str) -> ScriptResult:
        result = run_script(MOVE_SCRIPT, ["--action", action], self.config.timeout_seconds)
        self.emit_script_result("move_robot", result)
        return result

    def call_clean(self) -> ScriptResult:
        result = run_script(CLEAN_SCRIPT, [], self.config.timeout_seconds)
        self.emit_script_result("clean_garbage", result)
        return result

    def interaction_payload_for_candidate(self, candidate: Optional[JsonDict], *, role: str) -> JsonDict:
        if not isinstance(candidate, dict):
            return {}
        center = candidate.get("center") if isinstance(candidate.get("center"), dict) else {}
        bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
        payload = {
            "schema_version": 1,
            "role": role,
            "label": candidate.get("label"),
            "raw_label": candidate.get("raw_label") or candidate.get("label"),
            "task_semantic_class": candidate.get("task_semantic_class"),
            "bbox": bbox,
            "center": center,
            "confidence": candidate.get("confidence"),
            "position_hint": candidate.get("position_hint"),
            "surface_hint": candidate.get("surface_hint"),
        }
        for key in (
            "reachable",
            "pickup_now",
            "place_now",
            "needs_alignment",
            "needs_approach",
            "is_floor_level",
            "is_support_surface",
            "visual_box_ambiguous",
            "broad_front_receptacle",
            "front_edge_receptacle",
            "area_ratio",
            "center_y_ratio",
            "bottom_y_ratio",
        ):
            if key in candidate:
                payload[key] = candidate.get(key)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        if geometry:
            payload["geometry"] = dict(geometry)
        place_front_edge = bool(role == "place" and self.receptacle_front_edge_candidate(candidate))
        if place_front_edge:
            payload["front_edge_receptacle"] = True
            payload["broad_front_receptacle"] = True
            payload["visual_box_ambiguous"] = False
        if role == "place" and bbox:
            try:
                x = float(bbox.get("x", 0.0) or 0.0)
                y = float(bbox.get("y", 0.0) or 0.0)
                w = float(bbox.get("w", 0.0) or 0.0)
                h = float(bbox.get("h", 0.0) or 0.0)
                y_ratio = float(
                    os.getenv(
                        "ROBOT_PLACE_BROAD_POINT_BBOX_Y_RATIO"
                        if (place_front_edge or truthy(payload.get("broad_front_receptacle")))
                        else "ROBOT_PLACE_POINT_BBOX_Y_RATIO",
                        "0.62" if (place_front_edge or truthy(payload.get("broad_front_receptacle"))) else "0.88",
                    )
                )
                y_ratio = min(0.95, max(0.55, y_ratio))
                if w > 0 and h > 0:
                    payload["interaction_point"] = {
                        "x": round(x + w * 0.5, 3),
                        "y": round(y + h * y_ratio, 3),
                    }
            except (TypeError, ValueError):
                pass
        if "interaction_point" not in payload and center:
            payload["interaction_point"] = {
                "x": center.get("x"),
                "y": center.get("y"),
            }
        return payload

    def interaction_args_for_candidate(self, candidate: Optional[JsonDict], *, role: str) -> List[str]:
        if self.config.interaction_grounding != "metadata-hidden":
            return []
        payload = self.interaction_payload_for_candidate(candidate, role=role)
        if not payload:
            return ["--strict-visual-grounding"]
        return [
            "--strict-visual-grounding",
            "--candidate-json",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ]

    def call_pick(self, candidate: Optional[JsonDict] = None) -> ScriptResult:
        result = run_script(
            PICK_SCRIPT,
            self.interaction_args_for_candidate(candidate, role="pickup"),
            self.config.timeout_seconds,
        )
        self.emit_script_result("pick_object", result)
        return result

    def call_place(self, candidate: Optional[JsonDict] = None) -> ScriptResult:
        result = run_script(
            PLACE_SCRIPT,
            self.interaction_args_for_candidate(candidate, role="place"),
            self.config.timeout_seconds,
        )
        self.emit_script_result("place_object", result)
        return result

    def sync_inventory_state(self) -> None:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return
        url = f"{DEFAULT_BACKEND_BASE_URL.rstrip('/')}/inventory"
        try:
            with urlrequest.urlopen(url, timeout=min(max(self.config.timeout_seconds, 1), 5)) as response:
                payload = response.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
        except (OSError, ValueError, urlerror.URLError, TimeoutError):
            return
        if isinstance(data, dict) and data.get("status") == "success":
            previous = bool(self.holding_object)
            self.holding_object = bool(data.get("holding_object", False))
            self.service_state["holding_object"] = bool(self.holding_object)
            phase = self.service_phase()
            if self.holding_object and phase in SERVICE_PICKUP_PHASES:
                self.set_service_phase("SEARCH_RECEPTACLE", reason="inventory_holding_object")
            elif (not self.holding_object) and phase in SERVICE_PLACE_PHASES and phase != "VERIFY_TASK_DONE":
                self.set_service_phase(SERVICE_INITIAL_PHASE, reason="inventory_empty_before_place_done")
            elif previous != self.holding_object:
                self.append_service_history(
                    "inventory_changed",
                    holding_object=self.holding_object,
                    source="sync_inventory_state",
                )
                self.save_service_task_state()

    def emit_script_result(self, name: str, result: ScriptResult) -> None:
        payload: JsonDict = {
            "script": name,
            "returncode": result.returncode,
            "status": result.data.get("status"),
            "result_type": result.data.get("result_type"),
        }
        if self.config.verbose:
            payload["data"] = result.data
            if result.stderr.strip():
                payload["stderr"] = result.stderr.strip()
        self.emit("script_result", payload)

    def skill_success(self, result: ScriptResult, required_field: Optional[str] = None) -> bool:
        if not result.ok:
            return False
        if result.data.get("status") != "success":
            return False
        if required_field and not result.data.get(required_field):
            return False
        return True

    def analysis_success(self, result: ScriptResult) -> bool:
        if not self.skill_success(result):
            return False
        required = {
            "floor_trash_detected",
            "trash_candidates",
            "obstacle_ahead",
            "open_directions",
            "frontier_exists",
            "analysis_confidence",
        }
        if not required.issubset(result.data.keys()):
            return False
        try:
            confidence = float(result.data.get("analysis_confidence", 0.0))
        except (TypeError, ValueError):
            return False
        return confidence >= 0.0

    def handle_perception_failure(self, reason: str, data: JsonDict) -> StepOutcome:
        action = self.safe_turn_action()
        self.emit(
            "perception_failure",
            {
                "reason": reason,
                "details": {
                    "status": data.get("status"),
                    "result_type": data.get("result_type"),
                    "message": data.get("message"),
                },
                "fallback_action": action,
            },
        )

        if self.config.dry_run:
            return StepOutcome(
                step_recorded=False,
                stop_requested=True,
                action=action,
                success=False,
                reason=reason,
            )

        execution = self.call_move(action)
        action_success, action_reason = self.verify_move(execution.data, action)
        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-8:]
        self.record_action_result(action, False)

        state = self.manager.record_step(
            action=action,
            mode="EXPLORE",
            success=False,
            failure_reason=reason if action_success else f"{reason}; {action_reason}",
            no_target=True,
            repeated_view=False,
        )
        self.emit(
            "state_updated",
            {
                "step_count": state.patrol.get("step_count"),
                "failed_attempts": state.patrol.get("failed_attempts"),
                "last_action": state.patrol.get("last_action"),
            },
        )
        return StepOutcome(
            step_recorded=True,
            stop_requested=False,
            action=action,
            success=False,
            reason=reason,
        )

    def decide(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        try:
            confidence = float(analysis.get("analysis_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < 0.60:
            return Decision(
                kind="move",
                action=self.safe_turn_action(analysis),
                mode="EXPLORE",
                reason=f"low_analysis_confidence:{confidence:.2f}",
            )

        if str(self.config.task_mode or "clean").strip().lower() == "tidy":
            return self.decide_tidy(analysis, vision)

        clean_candidate = self.first_candidate(
            analysis,
            require_cleanable=True,
            require_alignment=False,
        )
        if clean_candidate is not None:
            validation = self.validate_clean_candidate(clean_candidate)
            if not bool(validation.get("clean_allowed", False)):
                return self.clean_validation_reject_decision(
                    analysis,
                    vision,
                    clean_candidate,
                    validation,
                )

            clean_candidate = dict(clean_candidate)
            clean_candidate["backend_validation"] = {
                "result_type": validation.get("result_type"),
                "best_backend_candidate": short_backend_candidate(
                    validation.get("best_backend_candidate")
                ),
            }
            return Decision(
                kind="clean",
                action="clean-garbage",
                mode="CLEAN",
                reason=f"direct_cleanable_floor_target;{self.config.clean_validation}_validated",
                candidate=clean_candidate,
            )
          #【需要对准】的垃圾  即（不在正前方，不能直接扫）
        align_candidate = self.first_candidate(
            analysis,
            require_cleanable=False,
            require_alignment=True,
        )

        if align_candidate is not None:
 # 1. 优先级：如果有【未探索的新区域】，优先去探索，不着急对准垃圾   机器人不会为了一个垃圾，放弃探索整个房间，保证覆盖率优先。
    # 导航探索 > 对准垃圾，保证房间先走遍，再回头扫垃圾
            if self.navigation_has_frontier():#navigation_has_frontier()：判断房间里还有没有「没扫过的新区域」
                return self.explore_decision(
                    analysis,
                    vision,
                    reason="alignment_deferred_for_navigation_frontier",
                )
# 2. 给这个垃圾生成一个唯一ID，方便统计对准次数（防反复对准同一个垃圾）
            key = candidate_key(align_candidate)
            self.alignment_attempts[key] = self.alignment_attempts.get(key, 0) + 1
             # 4. 防卡死：如果对准同一个垃圾超过2次都没成功 → 放弃！
            if self.alignment_attempts[key] > 2:
                # 暂时屏蔽这个垃圾4步，期间不再尝试对准它
                self.suppressed_until_step[key] = self.current_step_count() + 4
                  # 切换到探索模式，去别的地方
                return self.explore_decision(
                    analysis,
                    vision,
                    reason="alignment_attempt_limit_reached",
                )
            # #垃圾在左边/右边 → 旋转对准
            action = "RotateLeft" if align_candidate.get("position_hint") == "front-left" else "RotateRight"
            action, reason_suffix = self.break_rotation_oscillation(action, analysis, allow_forward_break=False)
             # 返回最终决策：执行转向，模式为探索，原因是对准地面垃圾
            return Decision(
                kind="move",
                action=action,
                mode="EXPLORE",
                reason=f"align_floor_target:{reason_suffix}",
                candidate=align_candidate,
            )

        return self.explore_decision(analysis, vision, reason="no_direct_cleanable_target")

    def decide_tidy(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        """ALFRED-style household service policy.

        The policy is phase-driven instead of frame-reactive: once a pickup
        target is locked, the runner keeps pursuing that subgoal until pickup
        succeeds, fails out, or the target is lost for several steps.
        """
        phase = self.service_phase()
        if phase == SERVICE_DONE_PHASE:
            self.set_service_phase(
                SERVICE_INITIAL_PHASE,
                reason="previous_service_subgoal_complete_continue_patrol",
            )
            phase = self.service_phase()
        if phase == "FAILED":
            return Decision(
                kind="stop",
                action="none",
                mode="RECOVER",
                reason="service_task_failed",
            )

        if self.holding_object:
            if phase not in SERVICE_PLACE_PHASES:
                self.set_service_phase("SEARCH_RECEPTACLE", reason="holding_object_enter_place_subgoal")
            return self.decide_tidy_place_phase(analysis, vision)

        if phase in SERVICE_PLACE_PHASES:
            self.set_service_phase(SERVICE_INITIAL_PHASE, reason="inventory_empty_return_to_pickup_search")

        return self.decide_tidy_pickup_phase(analysis, vision)

    def decide_tidy_pickup_phase(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        phase = self.service_phase()
        candidates = self.visible_service_candidates(analysis, task_class="pickup_target")
        locked_label = str(self.service_state.get("target_raw_label") or self.service_state.get("target_label") or "")

        if not candidates and truthy(analysis.get("pickup_target_detected")):
            if locked_label:
                self.clear_service_lock(role="pickup", reason="locked_pickup_target_not_actionable")
            clean_decision = self.tidy_clean_fallback_decision(analysis)
            if clean_decision is not None:
                return clean_decision
            return self.explore_decision(
                analysis,
                vision,
                reason="service_pickup_targets_visible_but_not_actionable",
            )

        candidate = next((item for item in candidates if self.service_lock_matches(item, role="pickup")), None)
        if locked_label and candidate is None:
            if candidates:
                self.clear_service_lock(role="pickup", reason="locked_pickup_target_lost_retarget_visible")
                candidate = candidates[0]
            else:
                lost_steps = self.locked_lost_steps(role="pickup")
                scan_count = self.increment_locked_scan_count(role="pickup")
                max_scan_steps = self.max_locked_scan_steps(role="pickup")
                if lost_steps > max_scan_steps or scan_count > max_scan_steps:
                    self.clear_service_lock(role="pickup", reason="locked_pickup_target_lost_or_not_actionable")
                    clean_decision = self.tidy_clean_fallback_decision(analysis)
                    if clean_decision is not None:
                        return clean_decision
                    return self.explore_decision(
                        analysis,
                        vision,
                        reason=(
                            f"locked_pickup_target_released:{locked_label};"
                            f"lost_steps={lost_steps};scan={scan_count}"
                        ),
                    )
                action, reason_suffix = self.break_rotation_oscillation(
                    "RotateLeft",
                    analysis,
                    allow_forward_break=False,
                )
                return Decision(
                    kind="move",
                    action=action,
                    mode="SERVICE",
                    reason=(
                        f"service_scan_locked_pickup_target:{locked_label};"
                        f"lost_steps={lost_steps};scan={scan_count};{reason_suffix}"
                    ),
                    candidate=None,
                )

        if candidate is None:
            candidate = candidates[0] if candidates else None

        if candidate is not None and not self.service_lock_matches(candidate, role="pickup"):
            self.lock_service_candidate(candidate, role="pickup", reason="pickup_target_locked")
        elif candidate is not None:
            self.service_state["target_last_seen_step"] = self.current_step_count()
            self.service_state["target_lost_scan_count"] = 0
            if phase == SERVICE_INITIAL_PHASE:
                self.set_service_phase("LOCK_PICKUP_TARGET", reason="locked_pickup_target_visible", candidate=candidate)

        if candidate is None:
            self.clear_service_lock(role="pickup", reason="pickup_target_not_visible")
            clean_decision = self.tidy_clean_fallback_decision(analysis)
            if clean_decision is not None:
                return clean_decision
            return self.explore_decision(analysis, vision, reason="service_phase_search_pickup_target")

        if self.service_pick_ready(candidate):
            self.set_service_phase("PICK_OBJECT", reason="pickup_candidate_action_ready", candidate=candidate)
            return Decision(
                kind="pick",
                action="pick-object",
                mode="SERVICE",
                reason="alfred_subgoal_pick_object",
                candidate=candidate,
            )

        self.set_service_phase("ALIGN_PICKUP_TARGET", reason="pickup_candidate_needs_positioning", candidate=candidate)
        return self.service_positioning_decision(
            analysis,
            vision,
            candidate,
            base_reason="alfred_align_pickup_target",
        )

    def decide_tidy_place_phase(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        phase = self.service_phase()
        candidates = self.visible_service_candidates(analysis, task_class="place_receptacle")
        locked_label = str(self.service_state.get("receptacle_raw_label") or self.service_state.get("receptacle_label") or "")
        candidate = next((item for item in candidates if self.service_lock_matches(item, role="place")), None)
        if locked_label and candidate is None:
            if candidates:
                self.clear_service_lock(role="place", reason="locked_receptacle_lost_retarget_visible")
                candidate = candidates[0]
            else:
                lost_steps = self.locked_lost_steps(role="place")
                scan_count = self.increment_locked_scan_count(role="place")
                max_scan_steps = self.max_locked_scan_steps(role="place")
                if lost_steps > max_scan_steps or scan_count > max_scan_steps:
                    self.clear_service_lock(role="place", reason="locked_receptacle_lost")
                    return self.explore_decision(
                        analysis,
                        vision,
                        reason=(
                            f"locked_receptacle_released:{locked_label};"
                            f"lost_steps={lost_steps};scan={scan_count}"
                        ),
                    )
                action, reason_suffix = self.break_rotation_oscillation(
                    "RotateLeft",
                    analysis,
                    allow_forward_break=False,
                )
                return Decision(
                    kind="move",
                    action=action,
                    mode="SERVICE",
                    reason=(
                        f"service_scan_locked_receptacle:{locked_label};"
                        f"lost_steps={lost_steps};scan={scan_count};{reason_suffix}"
                    ),
                    candidate=None,
                )

        if candidate is None:
            candidate = candidates[0] if candidates else None

        if candidate is not None and not self.service_lock_matches(candidate, role="place"):
            self.lock_service_candidate(candidate, role="place", reason="receptacle_locked")
        elif candidate is not None:
            self.service_state["receptacle_last_seen_step"] = self.current_step_count()
            self.service_state["receptacle_lost_scan_count"] = 0
            if phase == "SEARCH_RECEPTACLE":
                self.set_service_phase("LOCK_RECEPTACLE", reason="locked_receptacle_visible", candidate=candidate)

        if candidate is None:
            self.clear_service_lock(role="place", reason="receptacle_not_visible")
            return self.explore_decision(analysis, vision, reason="service_phase_search_receptacle")

        if self.service_place_ready(candidate):
            self.set_service_phase("PLACE_OBJECT", reason="receptacle_action_ready", candidate=candidate)
            return Decision(
                kind="place",
                action="place-object",
                mode="SERVICE",
                reason="alfred_subgoal_place_object",
                candidate=candidate,
            )

        try:
            phase_align_attempts = int(self.service_state.get("phase_attempts", 0) or 0)
        except (TypeError, ValueError):
            phase_align_attempts = 0
        try:
            streak_align_attempts = int(self.service_state.get("receptacle_alignment_streak", 0) or 0)
        except (TypeError, ValueError):
            streak_align_attempts = 0
        align_attempts = max(phase_align_attempts, streak_align_attempts)

        if (
            self.holding_object
            and self.service_place_probe_ready(candidate)
            and str(candidate.get("position_hint") or "") == "front-center"
            and not truthy(candidate.get("needs_alignment"))
            and self.service_failures.get(candidate_key(candidate), 0) == 0
        ):
            self.set_service_phase("PLACE_OBJECT", reason="holding_visible_receptacle_backend_probe", candidate=candidate)
            return Decision(
                kind="place",
                action="place-object",
                mode="SERVICE",
                reason="alfred_subgoal_place_object;holding_visible_backend_probe",
                candidate=candidate,
            )

        if self.service_phase() == "ALIGN_RECEPTACLE" and align_attempts >= self.max_receptacle_align_attempts():
            if self.service_place_probe_ready(candidate):
                self.set_service_phase("PLACE_OBJECT", reason="receptacle_align_timeout_backend_probe", candidate=candidate)
                return Decision(
                    kind="place",
                    action="place-object",
                    mode="SERVICE",
                    reason=f"alfred_subgoal_place_object;align_timeout_backend_probe:{align_attempts}",
                    candidate=candidate,
                )

            self.suppress_service_candidate(
                candidate,
                reason=f"receptacle_alignment_timeout:{align_attempts}",
            )
            self.clear_service_lock(role="place", reason="receptacle_alignment_timeout_retarget")
            return self.explore_decision(
                analysis,
                vision,
                reason=f"receptacle_alignment_timeout_retarget:{align_attempts}",
            )

        self.set_service_phase("ALIGN_RECEPTACLE", reason="receptacle_needs_positioning", candidate=candidate)
        return self.service_positioning_decision(
            analysis,
            vision,
            candidate,
            base_reason="alfred_align_receptacle",
        )

    def tidy_clean_fallback_decision(self, analysis: JsonDict) -> Optional[Decision]:
        clean_candidate = self.first_candidate(
            analysis,
            require_cleanable=True,
            require_alignment=False,
        )
        if clean_candidate is None:
            return None
        validation = self.validate_clean_candidate(clean_candidate)
        if not bool(validation.get("clean_allowed", False)):
            return None
        return Decision(
            kind="clean",
            action="clean-garbage",
            mode="SERVICE",
            reason=f"tidy_mode_cleanable_object;{self.config.clean_validation}_validated",
            candidate=clean_candidate,
        )

    def service_positioning_decision(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        candidate: JsonDict,
        *,
        base_reason: str,
    ) -> Decision:
        """Move the robot toward a service candidate without executing pick/place.

        For left/right candidates, rotate to align. For a front-center candidate
        that is visible but not actuator-ready, cautiously approach if forward
        is open. This is especially important after /place returns
        error_receptacle_not_centered.
        """
        position_hint = str(candidate.get("position_hint") or "")
        task_class = str(candidate.get("task_semantic_class") or "")

        if self.holding_object and task_class == "place_receptacle":
            last_failed_action = self.last_failed_action()
            if (
                self.service_place_probe_ready(candidate)
                and position_hint == "front-center"
                and not truthy(candidate.get("needs_alignment"))
                and self.service_failures.get(candidate_key(candidate), 0) == 0
            ):
                return Decision(
                    kind="place",
                    action="place-object",
                    mode="SERVICE",
                    reason=f"{base_reason};holding_visible_backend_probe",
                    candidate=candidate,
                )

            if (
                self.consecutive_action_failures > 0
                and last_failed_action in MOVE_ACTIONS
                and not self.recent_action_failed("MoveBack")
                and not self.recent_moveback_loop()
            ):
                return Decision(
                    kind="move",
                    action="MoveBack",
                    mode="SERVICE",
                    reason=f"{base_reason};held_object_clearance_after_failed_move",
                    candidate=candidate,
                )

        if position_hint == "front-left":
            action = "RotateLeft"
            if (
                self.holding_object
                and task_class == "place_receptacle"
                and self.recent_action_failed(action)
                and not self.recent_action_failed("MoveBack")
                and not self.recent_moveback_loop()
            ):
                return Decision(
                    kind="move",
                    action="MoveBack",
                    mode="SERVICE",
                    reason=f"{base_reason};held_object_clearance_before_left_align",
                    candidate=candidate,
                )
            action, reason_suffix = self.break_rotation_oscillation(action, analysis, allow_forward_break=False)
            return Decision(
                kind="move",
                action=action,
                mode="SERVICE",
                reason=f"{base_reason};align_left:{reason_suffix}",
                candidate=candidate,
            )

        if position_hint == "front-right":
            action = "RotateRight"
            if (
                self.holding_object
                and task_class == "place_receptacle"
                and self.recent_action_failed(action)
                and not self.recent_action_failed("MoveBack")
                and not self.recent_moveback_loop()
            ):
                return Decision(
                    kind="move",
                    action="MoveBack",
                    mode="SERVICE",
                    reason=f"{base_reason};held_object_clearance_before_right_align",
                    candidate=candidate,
                )
            action, reason_suffix = self.break_rotation_oscillation(action, analysis, allow_forward_break=False)
            return Decision(
                kind="move",
                action=action,
                mode="SERVICE",
                reason=f"{base_reason};align_right:{reason_suffix}",
                candidate=candidate,
            )

        if position_hint == "front-center":
            center_offset = self.candidate_center_offset(candidate)
            if abs(center_offset) > 0.10:
                action = "RotateLeft" if center_offset < 0 else "RotateRight"
                action, reason_suffix = self.break_rotation_oscillation(action, analysis, allow_forward_break=False)
                direction = "left" if center_offset < 0 else "right"
                return Decision(
                    kind="move",
                    action=action,
                    mode="SERVICE",
                    reason=f"{base_reason};fine_align_{direction}:{reason_suffix}",
                    candidate=candidate,
                )

        if position_hint == "front-center" and self.can_safely_move_ahead(analysis):
            return Decision(
                kind="move",
                action="MoveAhead",
                mode="SERVICE",
                reason=f"{base_reason};approach_front_center_{task_class}",
                candidate=candidate,
            )

        return self.explore_decision(
            analysis,
            vision,
            reason=f"{base_reason};candidate_not_action_ready",
        )

    def validate_clean_candidate(self, candidate: JsonDict) -> JsonDict:
        """Validate a visual clean candidate according to the selected policy.

        V2 default is ``visual``: trust the online RGB perception output and do
        not query AI2-THOR privileged state. ``metadata`` is retained only for
        V1 baseline/debug ablations and calls /eval/state internally.
        """
        policy = str(self.config.clean_validation or "visual").strip().lower()

        if policy in {"visual", "off"}:
            clean_allowed = bool(
                candidate.get("cleanable_now")
                and candidate.get("is_floor_level")
                and candidate.get("reachable")
            )
            validation = {
                "status": "success",
                "result_type": f"{policy}_clean_candidate_validation",
                "clean_allowed": clean_allowed,
                "message": "online RGB perception candidate accepted without metadata oracle"
                if clean_allowed
                else "online RGB perception candidate does not satisfy cleanable fields",
                "best_backend_candidate": None,
                "backend_candidates": [],
                "online_safe": True,
            }
            self.emit(
                "target_validation",
                {
                    "policy": policy,
                    "visual_candidate": self.short_candidate(candidate),
                    "status": validation.get("status"),
                    "result_type": validation.get("result_type"),
                    "clean_allowed": validation.get("clean_allowed"),
                    "online_safe": True,
                },
            )
            return validation

        if policy != "metadata":
            validation = {
                "status": "error",
                "result_type": "invalid_clean_validation_policy",
                "clean_allowed": False,
                "message": f"Unsupported clean validation policy: {policy}",
                "backend_candidates": [],
                "best_backend_candidate": None,
            }
            self.emit("target_validation", validation)
            return validation

        validation = validate_clean_target_with_backend(
            base_url=DEFAULT_BACKEND_BASE_URL,
            timeout=min(max(self.config.timeout_seconds, 1), 5),
        )
        self.emit(
            "target_validation",
            {
                "policy": "metadata",
                "visual_candidate": self.short_candidate(candidate),
                "status": validation.get("status"),
                "result_type": validation.get("result_type"),
                "clean_allowed": validation.get("clean_allowed"),
                "best_backend_candidate": short_backend_candidate(
                    validation.get("best_backend_candidate")
                ),
                "online_safe": False,
                "usage_scope": "v1_baseline_or_offline_debug_only",
            },
        )
        return validation

    def clean_validation_reject_decision(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        candidate: JsonDict,
        validation: JsonDict,
    ) -> Decision:
        result_type = str(validation.get("result_type") or "backend_validation_failed")
        best_backend = validation.get("best_backend_candidate")
        reject_reasons = []
        position_hint = None
        if isinstance(best_backend, dict):
            reject_reasons = list(best_backend.get("reject_reasons") or [])
            position_hint = str(best_backend.get("position_hint") or "")

        if result_type == "backend_target_too_far" or "target_too_far" in reject_reasons:
            if self.can_safely_move_ahead(analysis):
                return Decision(
                    kind="move",
                    action="MoveAhead",
                    mode="EXPLORE",
                    reason=f"clean_validation_rejected:{result_type};approach_backend_target",
                    candidate=candidate,
                )
            return self.explore_decision(
                analysis,
                vision,
                reason=f"clean_validation_rejected:{result_type};cannot_approach",
            )

        if result_type == "backend_target_not_centered" or "target_not_centered" in reject_reasons:
            if position_hint == "front-left":
                action = "RotateLeft"
            elif position_hint == "front-right":
                action = "RotateRight"
            else:
                action = self.safe_turn_action(analysis)
            action, reason_suffix = self.break_rotation_oscillation(action, analysis)
            return Decision(
                kind="move",
                action=action,
                mode="EXPLORE",
                reason=f"clean_validation_rejected:{result_type};align_backend_target:{reason_suffix}",
                candidate=candidate,
            )

        # Backend has no matching executable target, or the target is not a
        # floor-level trash proxy. Suppress this visual blob briefly so the
        # runner does not keep cleaning the same false positive.
        key = candidate_key(candidate)
        self.suppressed_until_step[key] = self.current_step_count() + 5
        self.emit(
            "candidate_suppressed",
            {
                "candidate": self.short_candidate(candidate),
                "result_type": result_type,
                "until_step": self.suppressed_until_step[key],
            },
        )
        return self.explore_decision(
            analysis,
            vision,
            reason=f"clean_validation_rejected:{result_type};visual_candidate_suppressed",
        )

    def can_safely_move_ahead(self, analysis: JsonDict) -> bool:
        open_directions = list(analysis.get("open_directions", []) or [])
        if bool(analysis.get("obstacle_ahead", False)):
            return False
        if "forward" not in open_directions:
            return False
        if self.recent_action_failed("MoveAhead"):
            return False
        if base_action(self.recent_actions[-1] if self.recent_actions else None) == "MoveBack":
            return False
        if self.recent_moveback_loop():
            return False
        return True

    def observe_navigation(self, vision: JsonDict, analysis: JsonDict) -> None:
        try:
            nav_status = self.navigation.observe(
                vision=vision,
                analysis=analysis,
                persist=not self.config.dry_run,
            )
            self.emit(
                "navigation_observed",
                {
                    "last_cell": nav_status.get("last_cell"),
                    "last_heading": nav_status.get("last_heading"),
                    "visited_cell_count": nav_status.get("visited_cell_count"),
                    "coverage_estimate": nav_status.get("coverage_estimate"),
                    "frontier_cells": nav_status.get("frontier_cells", [])[:4],
                },
            )
        except Exception as exc:
            self.emit("navigation_memory_error", {"phase": "observe", "message": str(exc)})

    def navigation_recommendation(self, vision: JsonDict, analysis: JsonDict) -> Optional[JsonDict]:
        recent_failed_action = None
        if self.recent_results:
            action, success = self.recent_results[-1]
            if not success:
                recent_failed_action = action
        try:
            # 调用【导航内存】计算推荐动作
            nav_status = self.navigation.recommend(
                vision=vision,
                analysis=analysis,
                recent_actions=self.recent_actions,
                recent_failed_action=recent_failed_action,
                persist=not self.config.dry_run,
            )
            recommendation = nav_status.get("recommendation")
            if isinstance(recommendation, dict):
                self.pending_navigation_recommendation = recommendation
                self.emit(
                    "navigation_recommendation",
                    {
                        "recommendation": recommendation,
                        "coverage_estimate": nav_status.get("coverage_estimate"),
                        "visited_cell_count": nav_status.get("visited_cell_count"),
                        "collision_count": nav_status.get("collision_count"),
                        "frontier_cells": nav_status.get("frontier_cells", [])[:4],
                    },
                )
                return recommendation
        except Exception as exc:
            self.emit("navigation_memory_error", {"phase": "recommend", "message": str(exc)})
        self.pending_navigation_recommendation = None
        return None

    def record_navigation_step(
        self,
        *,
        action: str,
        success: bool,
        vision: JsonDict,
        analysis: JsonDict,
        execution: JsonDict,
        failure_reason: Optional[str],
    ) -> None:
        try:
            nav_status = self.navigation.record_step(
                action=action,
                success=success,
                vision=vision,
                analysis=analysis,
                action_result=execution,
                failure_reason=failure_reason,
                recommendation=self.pending_navigation_recommendation,
            )
            self.emit(
                "navigation_updated",
                {
                    "action": action,
                    "success": success,
                    "last_cell": nav_status.get("last_cell"),
                    "last_heading": nav_status.get("last_heading"),
                    "visited_cell_count": nav_status.get("visited_cell_count"),
                    "coverage_estimate": nav_status.get("coverage_estimate"),
                    "collision_count": nav_status.get("collision_count"),
                    "oscillation_count": nav_status.get("oscillation_count"),
                    "blocked_edges": nav_status.get("blocked_edges", [])[-4:],
                },
            )
        except Exception as exc:
            self.emit("navigation_memory_error", {"phase": "record_step", "message": str(exc)})

    def first_candidate(
        self,
        analysis: JsonDict,
        *,
        require_cleanable: bool,
        require_alignment: bool,
    ) -> Optional[JsonDict]:
        for candidate in analysis.get("trash_candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            if self.is_suppressed(candidate):
                continue
            if not (
                truthy(candidate.get("is_floor_level"))
                and truthy(candidate.get("reachable"))
            ):
                continue
            if require_cleanable and truthy(candidate.get("cleanable_now")):
                return candidate
            if require_alignment and truthy(candidate.get("needs_alignment")):
                return candidate
        return None

    def first_service_candidate(
        self,
        analysis: JsonDict,
        *,
        task_class: str,
        require_now: bool = False,
        require_alignment: bool = False,
    ) -> Optional[JsonDict]:
        """Return the most decision-useful service candidate.

        New YOLO perception provides best_pickup_candidate and
        best_receptacle_candidate. Prefer those stable fields before falling
        back to the full candidate lists, because large scenes often contain
        many CounterTop/Cabinet boxes that should not all drive runner logic.
        """
        pools: List[Any] = []
        if task_class == "pickup_target":
            pools.append(analysis.get("best_pickup_candidate"))
        elif task_class == "place_receptacle":
            pools.append(analysis.get("best_receptacle_candidate"))

        pools.extend(analysis.get("service_candidates", []) or [])
        pools.extend(analysis.get("receptacle_candidates", []) or [])

        candidates: List[JsonDict] = []
        seen_keys = set()
        for candidate in pools:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("task_semantic_class") != task_class:
                continue
            if self.is_suppressed(candidate):
                continue
            if not truthy(candidate.get("reachable")):
                continue
            if task_class == "place_receptacle" and self.receptacle_visual_box_ambiguous(candidate):
                continue
            key = candidate_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)

        def score(candidate: JsonDict) -> float:
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
            geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
            cx_ratio = float(geometry.get("cx_ratio", 0.5) or 0.5)
            center_bonus = max(0.0, 0.5 - abs(cx_ratio - 0.5))
            value = confidence * 2.0 + min(area_ratio, 0.20) + center_bonus
            if task_class == "pickup_target" and truthy(candidate.get("pickup_now")):
                value += 5.0
            if task_class == "place_receptacle" and truthy(candidate.get("place_now")):
                value += 5.0
            if truthy(candidate.get("needs_alignment")):
                value += 0.4
            if truthy(candidate.get("needs_approach")):
                value += 0.2
            return value

        candidates.sort(key=score, reverse=True)

        for candidate in candidates:
            if require_now:
                if task_class == "pickup_target" and truthy(candidate.get("pickup_now")):
                    return candidate
                if task_class == "place_receptacle" and truthy(candidate.get("place_now")):
                    return candidate
                continue
            if require_alignment:
                if truthy(candidate.get("needs_alignment")):
                    return candidate
                continue
            return candidate
        return None

    def explore_decision(self, analysis: JsonDict, vision: JsonDict, *, reason: str) -> Decision:
       # 1. 读取场景分析结果：可走方向、前方是否有障碍、是否有未探索区域
        open_directions = list(analysis.get("open_directions", []) or [])
        obstacle_ahead = bool(analysis.get("obstacle_ahead", False))
        frontier_exists = bool(analysis.get("frontier_exists", False))
# ====================== 优先级1：用导航内存的智能推荐（最优） ======================
        nav_recommendation = self.navigation_recommendation(vision, analysis)

        if nav_recommendation:
            nav_action = str(nav_recommendation.get("action") or "")
            nav_reason = str(nav_recommendation.get("reason") or "navigation_memory")
            if (
                nav_action == "MoveAhead"
                and not obstacle_ahead
                and "forward" in open_directions
                and not self.recent_action_failed("MoveAhead")
                and base_action(self.recent_actions[-1] if self.recent_actions else None) != "MoveBack"
            ):
                return Decision(
                    kind="move",
                    action="MoveAhead",
                    mode="EXPLORE",
                    reason=f"{reason};nav:{nav_reason}",
                )
            if nav_action in ROTATE_ACTIONS:
                return Decision(
                    kind="move",
                    action=nav_action,
                    mode="EXPLORE",
                    reason=f"{reason};nav:{nav_reason}:memory_turn",
                )
            if (
                nav_action == "MoveBack"
                and obstacle_ahead
                and not self.recent_action_failed("MoveBack")
                and not self.last_action_is("MoveBack")
                and not self.last_action_is("MoveAhead")
                and not self.recent_moveback_loop()
            ):
                return Decision(
                    kind="move",
                    action="MoveBack",
                    mode="EXPLORE",
                    reason=f"{reason};nav:{nav_reason}",
                )
          # ====================== 优先级2：房间探索完成 → 停止机器人 ======================
        plateau_reason = self.navigation_plateau_completion_reason()
        if plateau_reason:
            return Decision(
                kind="stop",
                action="none",
                mode="ROOM_COMPLETE",
                reason=f"{reason};{plateau_reason}",
            )
        # ====================== 优先级3：前方有障碍 → 避障 ======================
        if obstacle_ahead:
            if "left" in open_directions:
                action = "RotateLeft"
            elif "right" in open_directions:
                action = "RotateRight"
            else:
                action = "MoveBack"
                if (
                    self.recent_action_failed("MoveBack")
                    or self.last_action_is("MoveBack")
                    or self.last_action_is("MoveAhead")
                    or self.recent_moveback_loop()
                ):
                    action = self.safe_turn_action(analysis)
            action, reason_suffix = self.break_rotation_oscillation(action, analysis)
            return Decision(
                kind="move",
                action=action,
                mode="EXPLORE",
                reason=f"{reason};obstacle_ahead:{reason_suffix}",
            )
        # ====================== 优先级4：有未探索区域 + 前方能走 → 直接前进 ======================
        if frontier_exists and "forward" in open_directions:
            if self.recent_action_failed("MoveAhead"):
                action = self.prefer_side_turn(open_directions)
                return Decision(
                    kind="move",
                    action=action,
                    mode="EXPLORE",
                    reason=f"{reason};avoid_failed_moveahead_repeat",
                )
            if base_action(self.recent_actions[-1] if self.recent_actions else None) == "MoveBack":
                action = self.prefer_side_turn(open_directions)
                return Decision(
                    kind="move",
                    action=action,
                    mode="EXPLORE",
                    reason=f"{reason};avoid_moveback_moveahead_loop",
                )
            return Decision(
                kind="move",
                action="MoveAhead",
                mode="EXPLORE",
                reason=f"{reason};forward_open",
            )
        # ====================== 优先级5：有未探索区域但在侧面 → 转向 ======================
        if frontier_exists:
            action = self.prefer_side_turn(open_directions)
            action, reason_suffix = self.break_rotation_oscillation(action, analysis)
            return Decision(
                kind="move",
                action=action,
                mode="EXPLORE",
                reason=f"{reason};side_frontier:{reason_suffix}",
            )
        # ====================== 优先级6：无新区域 + 无垃圾 → 判定房间完成 → 停止 ======================
        state = self.manager.load_state()
        no_target = int(state.patrol.get("consecutive_no_target", 0))
        repeated = int(state.patrol.get("consecutive_repeated_view", 0))
        if no_target >= 8 and repeated >= 3:
            return Decision(
                kind="stop",
                action="none",
                mode="ROOM_COMPLETE",
                reason=f"{reason};no_frontier_after_repeated_views",
            )
        # ====================== 兜底：无任何任务 → 原地转向扫描 ======================
        return Decision(
            kind="move",
            action=self.safe_turn_action(analysis),
            mode="EXPLORE",
            reason=f"{reason};no_frontier_conservative_scan",
        )

    def safe_turn_action(self, analysis: Optional[JsonDict] = None) -> str:
        open_directions = []
        if analysis:
            open_directions = list(analysis.get("open_directions", []) or [])
        if "left" in open_directions:
            return "RotateLeft"
        if "right" in open_directions:
            return "RotateRight"
        last = base_action(self.recent_actions[-1] if self.recent_actions else None)
        if last == "RotateRight":
            return "RotateRight"
        return "RotateLeft"

    def prefer_side_turn(self, open_directions: Iterable[str]) -> str:
        directions = list(open_directions)
        last = base_action(self.recent_actions[-1] if self.recent_actions else None)
        if "left" in directions and last != "RotateRight":
            return "RotateLeft"
        if "right" in directions:
            return "RotateRight"
        if "left" in directions:
            return "RotateLeft"
        return self.safe_turn_action()

    def break_rotation_oscillation(self, action: str, analysis: JsonDict, *, allow_forward_break: bool = True) -> Tuple[str, str]:
        if action not in ROTATE_ACTIONS:
            return action, "normal"

        last = base_action(self.recent_actions[-1] if self.recent_actions else None)
        previous = base_action(self.recent_actions[-2] if len(self.recent_actions) >= 2 else None)

        if last == OPPOSITE_ROTATION[action] and previous == action:
            if (
                allow_forward_break
                and
                not bool(analysis.get("obstacle_ahead", False))
                and "forward" in list(analysis.get("open_directions", []) or [])
                and not self.recent_action_failed("MoveAhead")
                and not self.recent_moveback_loop()
            ):
                return "MoveAhead", "break_rotate_oscillation_forward"
            return str(last), "break_rotate_oscillation_keep_direction"

        return action, "normal"

    def is_suppressed(self, candidate: JsonDict) -> bool:
        return any(
            self.suppressed_until_step.get(key, -1) > self.current_step_count()
            for key in (candidate_key(candidate), candidate_family_key(candidate))
        )

    def current_step_count(self) -> int:
        try:
            return int(self.manager.load_state().patrol.get("step_count", 0))
        except Exception:
            return 0

    def record_action_result(self, action: str, success: bool) -> None:
        self.recent_results.append((action, success))
        self.recent_results = self.recent_results[-8:]
        if success:
            self.consecutive_action_failures = 0
        else:
            self.consecutive_action_failures += 1

    def recent_action_failed(self, action: str) -> bool:
        if not self.recent_results:
            return False
        recent_action, recent_success = self.recent_results[-1]
        return recent_action == action and not recent_success

    def last_failed_action(self) -> Optional[str]:
        if not self.recent_results:
            return None
        recent_action, recent_success = self.recent_results[-1]
        if recent_success:
            return None
        return base_action(recent_action)

    def recent_moveback_loop(self) -> bool:
        bases = [base_action(action) for action in self.recent_actions[-6:]]
        text = ",".join(str(action) for action in bases)
        return (
            "MoveAhead,MoveBack" in text
            or "MoveBack,MoveAhead" in text
            or bases.count("MoveBack") >= 2
        )

    def navigation_has_frontier(self) -> bool:
        try:
            # 1. 从导航内存模块，获取当前房间的探索状态
            nav_status = self.navigation.status()
        except Exception:
            # 2. 导航模块报错 → 默认认为「没有新区域」
            return False
        # 3. 提取「未探索的前沿单元格」，如果列表不为空 → 有新区域(返回True)，否则False
        return bool(list(nav_status.get("frontier_cells", []) or []))#frontier_cells：前沿单元格 = 机器人没去过、没探索的新区域

    def turn_streak_from_actions(self, actions: Sequence[Any]) -> int:
        streak = 0
        for action in reversed(list(actions)):
            if base_action(action) in ROTATE_ACTIONS:
                streak += 1
                continue
            break
        return streak

    def navigation_plateau_completion_reason(self) -> Optional[str]:
        try:
            nav_status = self.navigation.status()
        except Exception:
            return None

        coverage = float(nav_status.get("coverage_estimate", 0.0) or 0.0)
        stagnation = int(nav_status.get("stagnation_count", 0) or 0)
        turn_streak = int(nav_status.get("turn_streak_count", 0) or 0)
        turn_streak = max(
            turn_streak,
            self.turn_streak_from_actions(nav_status.get("recent_navigation_actions", []) or []),
        )
        frontier_cells = list(nav_status.get("frontier_cells", []) or [])
        step_count = self.current_step_count()

        if (
            step_count >= 35
            and not frontier_cells
            and coverage >= 0.95
            and stagnation >= 6
            and turn_streak >= 4
        ):
            return (
                f"navigation_plateau;coverage={coverage:.2f};"
                f"stagnation={stagnation};turn_streak={turn_streak}"
            )
        return None

    def last_action_is(self, action: str) -> bool:
        if not self.recent_actions:
            return False
        return base_action(self.recent_actions[-1]) == action

    def register_clean_failure(self, decision: Decision, data: JsonDict) -> None:
        if decision.candidate is None:
            return

        key = candidate_key(decision.candidate)
        self.clean_failures[key] = self.clean_failures.get(key, 0) + 1
        result_type = str(data.get("result_type", ""))
        if result_type in NON_RETRYABLE_CLEAN_ERRORS or self.clean_failures[key] >= 2:
            self.suppressed_until_step[key] = self.current_step_count() + 5
            self.emit(
                "candidate_suppressed",
                {
                    "candidate": self.short_candidate(decision.candidate),
                    "result_type": result_type,
                    "until_step": self.suppressed_until_step[key],
                },
            )

    def verify_clean(self, data: JsonDict, decision: Decision) -> Tuple[bool, Optional[str], List[str]]:
        # 成功条件：清扫脚本执行完成(clean_executed)
        status = data.get("status")
        result_type = data.get("result_type")
        removed_from_view = data.get("removed_from_view")
        removed_from_scene = data.get("removed_from_scene")
        last_action_success = data.get("lastActionSuccess")

        # V2 RGB-only online contract:
        # /clean 默认会脱敏 object/metadata 细节，因此在线链路不能再依赖
        # removed_from_view / removed_from_scene 来判断清扫是否成功。
        #
        # 当前阶段先把 clean_executed 解释为“清扫执行器已成功执行”。
        # 后续接入 visual-action-verifier 后，再用前后帧视觉变化判断
        # target_disappeared / visual_clean_verified。
        if status == "success" and result_type == "clean_executed":
            label = "cleanable_floor_target"
            if decision.candidate:
                label = str(decision.candidate.get("label") or label)

            verification_notes = []

            if removed_from_view is True or removed_from_scene is True:
                verification_notes.append("metadata_or_backend_clean_verified")
            else:
                verification_notes.append("online_clean_actuator_executed")
                verification_notes.append("visual_recheck_pending")

            return True, None, [label]

        if last_action_success is True and result_type == "clean_executed":
            label = "cleanable_floor_target"
            if decision.candidate:
                label = str(decision.candidate.get("label") or label)
            return True, None, [label]

        reason = str(result_type or data.get("message") or "clean_failed")
        return False, reason, []

    def verify_service_action(self, data: JsonDict, *, expected_result_type: str) -> Tuple[bool, Optional[str]]:
        status = data.get("status")
        result_type = data.get("result_type")
        last_action_success = data.get("lastActionSuccess")
        success = (
            status == "success"
            and result_type == expected_result_type
            and last_action_success is not False
        )
        if expected_result_type == "place_executed" and data.get("placement_verified") is False:
            success = False
        if success:
            return True, None
        reason = str(
            result_type
            or data.get("error_message")
            or data.get("message")
            or f"{expected_result_type}_failed"
        )
        return False, reason

    def update_service_state_after_action(
        self,
        *,
        decision: Decision,
        execution: JsonDict,
        success: bool,
        failure_reason: Optional[str],
    ) -> None:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return

        if decision.kind == "pick":
            if success:
                self.holding_object = bool(execution.get("holding_object", True))
                self.service_state["holding_object"] = bool(self.holding_object)
                self.service_state["receptacle_alignment_streak"] = 0
                self.set_service_phase("VERIFY_HOLDING", reason="pickup_executed", candidate=decision.candidate)
                if self.holding_object:
                    self.set_service_phase(
                        "SEARCH_RECEPTACLE",
                        reason="pickup_verified_inventory_holding",
                        candidate=decision.candidate,
                        increment_subgoal=True,
                    )
                return
            self.register_service_failure(role="pickup", decision=decision, execution=execution, failure_reason=failure_reason)
            return

        if decision.kind == "place":
            if success:
                placed_label = str(
                    self.service_state.get("target_raw_label")
                    or self.service_state.get("target_label")
                    or "held_object"
                )
                receptacle_label = str(
                    (decision.candidate or {}).get("raw_label")
                    or (decision.candidate or {}).get("label")
                    or self.service_state.get("receptacle_raw_label")
                    or self.service_state.get("receptacle_label")
                    or "receptacle"
                )
                self.holding_object = bool(execution.get("holding_object", False))
                self.service_state["holding_object"] = bool(self.holding_object)
                self.service_state["receptacle_alignment_streak"] = 0
                self.set_service_phase("VERIFY_TASK_DONE", reason="place_executed", candidate=decision.candidate)
                if not self.holding_object:
                    completion = f"{placed_label}->{receptacle_label}"
                    self.pending_service_completions.append(completion)
                    self.pending_placed_objects.append(placed_label)
                    self.suppress_recently_placed_label(placed_label)
                    completed = self.service_state.setdefault("completed_subgoals", [])
                    if not isinstance(completed, list):
                        completed = []
                    completed.append(
                        {
                            "step": self.current_step_count(),
                            "object": placed_label,
                            "receptacle": receptacle_label,
                            "summary": completion,
                        }
                    )
                    self.service_state["completed_subgoals"] = completed[-80:]
                    self.service_state["target_label"] = None
                    self.service_state["target_raw_label"] = None
                    self.service_state["target_signature"] = None
                    self.service_state["target_last_seen_step"] = None
                    self.service_state["target_attempts"] = 0
                    self.service_state["receptacle_label"] = None
                    self.service_state["receptacle_raw_label"] = None
                    self.service_state["receptacle_signature"] = None
                    self.service_state["receptacle_last_seen_step"] = None
                    self.service_state["receptacle_attempts"] = 0
                    self.emit(
                        "service_subgoal_completed",
                        {
                            "object": placed_label,
                            "receptacle": receptacle_label,
                            "summary": completion,
                            "subgoal_index": self.service_state.get("subgoal_index"),
                        },
                    )
                    self.set_service_phase(
                        SERVICE_INITIAL_PHASE,
                        reason="place_verified_inventory_empty_continue_patrol",
                        candidate=decision.candidate,
                        increment_subgoal=True,
                    )
                return
            self.register_service_failure(role="place", decision=decision, execution=execution, failure_reason=failure_reason)
            return

        if decision.mode == "SERVICE" and decision.kind == "move":
            self.service_state["phase_attempts"] = int(self.service_state.get("phase_attempts", 0) or 0) + 1
            self.service_state["holding_object"] = bool(self.holding_object)
            if self.service_phase() == "ALIGN_RECEPTACLE" and self.holding_object:
                try:
                    receptacle_alignment_streak = int(
                        self.service_state.get("receptacle_alignment_streak", 0) or 0
                    )
                except (TypeError, ValueError):
                    receptacle_alignment_streak = 0
                self.service_state["receptacle_alignment_streak"] = (
                    receptacle_alignment_streak + 1
                )
            elif self.service_phase() not in SERVICE_PLACE_PHASES:
                self.service_state["receptacle_alignment_streak"] = 0
            self.append_service_history(
                "service_positioning_action",
                phase=self.service_phase(),
                action=decision.action,
                success=success,
                failure_reason=failure_reason,
                candidate=self.short_candidate(decision.candidate),
            )
            self.save_service_task_state()

    def register_service_failure(
        self,
        *,
        role: str,
        decision: Decision,
        execution: JsonDict,
        failure_reason: Optional[str],
    ) -> None:
        candidate = decision.candidate
        if "holding_object" in execution:
            self.holding_object = bool(execution.get("holding_object", False))
            self.service_state["holding_object"] = bool(self.holding_object)
        key = candidate_key(candidate) if isinstance(candidate, dict) else f"{role}:unknown"
        self.service_failures[key] = self.service_failures.get(key, 0) + 1
        result_type = str(execution.get("result_type") or failure_reason or "service_action_failed")
        self.append_service_history(
            "service_action_failed",
            role=role,
            result_type=result_type,
            failure_count=self.service_failures[key],
            candidate=self.short_candidate(candidate),
        )
        if role == "place" and isinstance(candidate, dict) and result_type in PLACE_RETARGET_ERRORS:
            self.suppress_service_candidate(candidate, reason=result_type)
            self.clear_service_lock(role=role, reason=f"place_retarget_after:{result_type}")
            return
        if role == "pickup" and isinstance(candidate, dict) and result_type in PICKUP_RETARGET_ERRORS:
            self.suppress_service_candidate(candidate, reason=result_type)
            self.clear_service_lock(role=role, reason=f"pickup_retarget_after:{result_type}")
            return

        if isinstance(candidate, dict) and self.service_failures[key] >= 3:
            self.suppressed_until_step[key] = self.current_step_count() + 6
            self.emit(
                "candidate_suppressed",
                {
                    "candidate": self.short_candidate(candidate),
                    "result_type": result_type,
                    "until_step": self.suppressed_until_step[key],
                },
            )
            self.clear_service_lock(role=role, reason=f"{role}_failure_limit:{result_type}")
            return

        if role == "pickup":
            self.set_service_phase("ALIGN_PICKUP_TARGET", reason=f"pickup_failed:{result_type}", candidate=candidate)
        else:
            self.set_service_phase("ALIGN_RECEPTACLE", reason=f"place_failed:{result_type}", candidate=candidate)

    def verify_move(self, data: JsonDict, action: str) -> Tuple[bool, Optional[str]]:
# 成功条件：
    # 1. 脚本返回状态=success
    # 2. 动作执行成功(lastActionSuccess≠False)
    # 3. 机器人状态发生变化(state_changed≠False)   三个条件都是指的同一个

    # 第一步：从移动脚本返回的结果里，提取4个关键判断字段
        status = data.get("status")
        result_type = data.get("result_type")
        last_action_success = data.get("lastActionSuccess")
        state_changed = data.get("state_changed")

        success = (
            status == "success"
            and last_action_success is not False
            and state_changed is not False
        )
        if success:
            return True, None

        reason = str(
            result_type
            or data.get("error_message")
            or data.get("message")
            or f"{action}_failed"
        )
        return False, reason
 #机器人每走一步、分析完画面后、做决策前 调用
    def update_repeated_view(self, vision: JsonDict, analysis: JsonDict) -> bool:
        signature = view_signature(vision, analysis)
        repeated = self.last_view_signature == signature
        self.last_view_signature = signature
        return repeated

    def update_segment_stats(self, analysis: JsonDict) -> None:
        self.current_segment_stats["steps"] = int(self.current_segment_stats["steps"]) + 1
        if bool(analysis.get("direct_cleanable_detected", False)) or bool(
            analysis.get("direct_pickup_detected", False)
        ):
            self.current_segment_stats["cleanable_seen"] = True
        if bool(analysis.get("open_directions", [])):
            self.current_segment_stats["new_open_direction_seen"] = True

    def check_completion_after_step(self, state: StateSnapshot, analysis: JsonDict) -> bool:
        if str(self.config.task_mode or "clean").strip().lower() == "tidy":
            phase = self.service_phase()
            if phase == SERVICE_DONE_PHASE:
                self.set_service_phase(
                    SERVICE_INITIAL_PHASE,
                    reason="service_subgoal_done_continue_room_patrol",
                )
            if phase == "FAILED":
                self.mark_recover_failed("service_task_failed")
                return True

        if self.consecutive_action_failures >= 3:
            self.mark_recover_failed("consecutive_action_failures")
            return True

        step_count = int(state.patrol.get("step_count", 0))
        max_steps = int(state.patrol.get("max_steps", DEFAULT_MAX_STEPS))

        no_target = int(state.patrol.get("consecutive_no_target", 0))
        repeated = int(state.patrol.get("consecutive_repeated_view", 0))
        frontier_exists = bool(analysis.get("frontier_exists", False))
        stagnation = 0
        oscillation_count = 0
        try:
            nav_status = self.navigation.status()
            coverage = float(nav_status.get("coverage_estimate", 0.0) or 0.0)
            stagnation = int(nav_status.get("stagnation_count", 0) or 0)
            oscillation_count = int(nav_status.get("oscillation_count", 0) or 0)
            turn_streak = int(nav_status.get("turn_streak_count", 0) or 0)
            turn_streak = max(
                turn_streak,
                self.turn_streak_from_actions(nav_status.get("recent_navigation_actions", []) or []),
            )
            nav_frontier_cells = list(nav_status.get("frontier_cells", []) or [])
        except Exception:
            coverage = 0.0
            turn_streak = 0
            nav_frontier_cells = []

        if (
            step_count >= 30
            and stagnation >= 12
            and turn_streak >= 12
            and self.consecutive_action_failures == 0
        ):
            self.mark_recover_failed(
                f"navigation_rotation_stalled:"
                f"coverage={coverage:.2f};frontiers={len(nav_frontier_cells)};"
                f"stagnation={stagnation};turn_streak={turn_streak}"
            )
            return True

        if (
            step_count >= 35
            and not nav_frontier_cells
            and coverage >= 0.95
            and stagnation >= 6
            and turn_streak >= 4
        ):
            reason = (
                f"navigation_plateau;coverage={coverage:.2f};"
                f"stagnation={stagnation};turn_streak={turn_streak}"
            )
            self.mark_room_complete(reason)
            return True

        if step_count >= max_steps:
            if coverage < 0.95 or nav_frontier_cells:
                self.mark_recover_failed(
                    f"max_steps_incomplete_or_frontier:"
                    f"coverage={coverage:.2f};frontiers={len(nav_frontier_cells)}"
                )
                return True
            self.mark_room_complete(f"max_steps_reached;coverage={coverage:.2f}")
            return True

        if (
            step_count >= 20
            and coverage < 0.95
            and not nav_frontier_cells
            and stagnation >= 18
            and (self.recent_moveback_loop() or oscillation_count >= 3 or turn_streak >= 12)
        ):
            self.mark_recover_failed(
                f"navigation_stalled_incomplete_coverage:"
                f"coverage={coverage:.2f};stagnation={stagnation};turn_streak={turn_streak}"
            )
            return True

        if (
            step_count >= min(30, max_steps)
            and coverage >= 0.95
            and not nav_frontier_cells
            and no_target >= 8
            and repeated >= 2
            and self.consecutive_action_failures == 0
        ):
            self.mark_room_complete(f"coverage_complete:{coverage:.2f}")
            return True

        if (
            no_target >= 8
            and repeated >= 3
            and not frontier_exists
            and not nav_frontier_cells
            and coverage >= 0.95
            and self.consecutive_action_failures == 0
        ):
            self.mark_room_complete("no_target_repeated_view_no_frontier")
            return True

        return False

    def mark_room_complete(self, reason: str) -> None:
        state = self.manager.load_state()
        cleaned = state.room.get("targets_cleaned", [])
        placed = state.room.get("objects_placed", [])
        service_completed = state.room.get("service_tasks_completed", [])
        found = state.room.get("targets_found", [])
        if service_completed:
            summary = f"room_complete:{reason}; service_completed={','.join(map(str, service_completed))}"
        elif placed:
            summary = f"room_complete:{reason}; placed={','.join(map(str, placed))}"
        elif cleaned:
            summary = f"room_complete:{reason}; cleaned={','.join(map(str, cleaned))}"
        elif found:
            summary = f"room_complete:{reason}; found={','.join(map(str, found))}"
        else:
            summary = f"room_complete:{reason}; no_target_found"
        state = self.manager.mark_room_complete(summary=summary)
        self.emit(
            "room_marked_complete",
            {
                "reason": reason,
                "summary": state.mission.get("final_summary"),
                "step_count": state.patrol.get("step_count"),
            },
        )

    def mark_recover_failed(self, reason: str) -> None:
        state = self.manager.mark_recover_failed(reason=reason)
        self.emit(
            "patrol_recover_failed",
            {
                "reason": reason,
                "summary": state.mission.get("final_summary"),
                "step_count": state.patrol.get("step_count"),
                "last_action": state.patrol.get("last_action"),
            },
        )

    def short_candidate(self, candidate: Optional[JsonDict]) -> Optional[JsonDict]:
        if not candidate:
            return None
        summary = {
            "label": candidate.get("label"),
            "raw_label": candidate.get("raw_label"),
            "task_semantic_class": candidate.get("task_semantic_class"),
            "position_hint": candidate.get("position_hint"),
            "surface_hint": candidate.get("surface_hint"),
            "cleanable_now": candidate.get("cleanable_now"),
            "pickup_now": candidate.get("pickup_now"),
            "place_now": candidate.get("place_now"),
            "needs_alignment": candidate.get("needs_alignment"),
            "needs_approach": candidate.get("needs_approach"),
            "reachable": candidate.get("reachable"),
            "confidence": candidate.get("confidence"),
            "bbox": candidate.get("bbox"),
        }
        if isinstance(candidate.get("backend_validation"), dict):
            summary["backend_validation"] = candidate.get("backend_validation")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the household service robot patrol loop without relying on chat context."
    )
    parser.add_argument("--start", action="store_true", help="Start/enable a patrol mission before running.")
    parser.add_argument("--no-reset", action="store_true", help="With --start, keep existing counters.")
    parser.add_argument("--room", default=DEFAULT_ROOM, help="Room name stored in memory state.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Maximum total patrol steps. Default on new mission: {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--segment-steps",
        type=int,
        default=3,
        help="Number of step loops to run in one segment.",
    )
    parser.add_argument("--continuous", action="store_true", help="Keep opening new segments until stop.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between segments.")
    parser.add_argument("--dry-run", action="store_true", help="Sense/analyze/decide only; do not move or write state.")
    parser.add_argument("--status", action="store_true", help="Print state validation and continuation status.")
    parser.add_argument("--max-segments", type=int, default=None, help="Optional safety cap for --continuous.")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout in seconds for each skill script.")
    parser.add_argument(
        "--clean-validation",
        choices=["visual", "metadata", "off"],
        default=os.getenv("ROBOT_CLEAN_VALIDATION", "visual"),
        help=(
            "Pre-clean validation policy. V2 default 'visual' does not query "
            "AI2-THOR metadata; 'metadata' is kept for V1 baseline/debug only."
        ),
    )
    parser.add_argument(
        "--perception-backend",
        choices=sorted(PERCEPTION_BACKENDS.keys()),
        default=os.getenv("ROBOT_PERCEPTION_BACKEND", "yolo"),
        help="Scene analysis backend. V2 default is 'yolo'; use 'opencv' for the V1 baseline.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["clean", "tidy"],
        default=os.getenv("ROBOT_TASK_MODE", "clean"),
        help=(
            "Task policy. 'clean' preserves the legacy cleaning patrol; "
            "'tidy' enables continuous household service patrol with pickup/place subgoals."
        ),
    )
    parser.add_argument(
        "--interaction-grounding",
        choices=["metadata-hidden", "legacy-metadata"],
        default=os.getenv("ROBOT_INTERACTION_GROUNDING", "metadata-hidden"),
        help=(
            "Pickup/place grounding policy. 'metadata-hidden' sends only the "
            "visual candidate to the executor; metadata remains hidden inside "
            "the simulator actuator. 'legacy-metadata' keeps the old empty "
            "pickup/place request for debugging."
        ),
    )
    parser.add_argument(
        "--pickup-surface-policy",
        choices=["floor-only", "any-surface"],
        default=os.getenv("ROBOT_PICKUP_SURFACE_POLICY", "floor-only"),
        help=(
            "Pickup target surface policy for tidy mode. The V2 safety default "
            "'floor-only' ignores tabletop/elevated pickup candidates; use "
            "'any-surface' only for explicit desktop-object tidy tests."
        ),
    )
    parser.add_argument(
        "--pickup-target-labels",
        default=os.getenv("ROBOT_PICKUP_TARGET_LABELS", ""),
        help=(
            "Optional comma-separated pickup label allowlist, for example "
            "'tomato,apple'. Empty means labels are not restricted."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Include full skill JSON results in events.")
    parser.add_argument("--quiet", action="store_true", help="Write detailed events to memory log, not stdout.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> RunnerConfig:
    args = build_parser().parse_args(argv)
    if args.segment_steps < 1:
        raise SystemExit("--segment-steps must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.max_segments is not None and args.max_segments < 1:
        raise SystemExit("--max-segments must be >= 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")
    pickup_labels = tuple(
        item.strip().lower()
        for item in str(args.pickup_target_labels or "").split(",")
        if item.strip()
    )
    return RunnerConfig(
        start=bool(args.start),
        no_reset=bool(args.no_reset),
        room=str(args.room),
        max_steps=args.max_steps,
        segment_steps=int(args.segment_steps),
        continuous=bool(args.continuous),
        sleep_seconds=float(args.sleep),
        dry_run=bool(args.dry_run),
        status=bool(args.status),
        max_segments=args.max_segments,
        timeout_seconds=int(args.timeout),
        clean_validation=str(args.clean_validation),
        perception_backend=str(args.perception_backend),
        task_mode=str(args.task_mode),
        interaction_grounding=str(args.interaction_grounding),
        pickup_surface_policy=str(args.pickup_surface_policy),
        pickup_target_labels=pickup_labels,
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    runner = PatrolRunner(config)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
