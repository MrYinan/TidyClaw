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
鏈哄櫒浜烘槸銆愪竴姝ヤ竴璁板綍銆戯紝姣忔墽琛屼竴涓姩浣滐紙涓€姝ワ級锛屽氨绔嬪埢鎶婄姸鎬佸啓锟?JSON 鏂囦欢
鍒嗘鐨勪綔鐢細
姣忔缁撴潫蹇呭仛 3 娆″畨鍏ㄦ鏌ワ紙浣犱唬鐮侀噷锟?run () 鍑芥暟锛夛細
    浠诲姟瀹屾垚 / 姝ユ暟婊′簡鍚楋紵
    杈惧埌鏈€澶у垎娈垫暟浜嗗悧锟?    瑕佷笉瑕佺户缁窇锟?    锟?闅忔椂鑳藉仠锛屼笉浼氭棤闄愬崱姝伙拷?3. 濂芥帶鍒讹細鏀寔 鈥滄柇缁繍琛岋拷?    鍒嗘鍙互瀹炵幇锟?    锟?1 娈靛氨鍋滐紙榛樿琛屼负锟?    杩炵画锟?N 娈碉紙锟?-continuous锟?    闄愬埗鏈€澶氳窇 10 娈碉紙锟?-max-segments 10锟?    濡傛灉涓嶅垎娈碉紝鍙兘涓€鐩磋窇鍋滀笉涓嬫潵锛屾牴鏈病娉曡皟璇曪拷?"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
from scripts.navigation_memory_core import NavigationMemory, local_costmap_motion_safety  # noqa: E402
from scripts.object_memory_core import ObjectMemory  # noqa: E402
from scripts.placement_viewpoint_planner import PlacementViewpointPlanner  # noqa: E402
from scripts.local_costmap import LocalCostmap  # noqa: E402
from scripts.perception_action_validator import (  # noqa: E402
    DEFAULT_BACKEND_BASE_URL,
    short_backend_candidate,
    validate_clean_target_with_backend,
)


TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
LOOK_ACTIONS = {"LookUp", "LookDown"}
# Backend-supported movement actions. Camera pitch remains available for manual
# debugging and future explicit active-perception skills. It is excluded from
# ordinary autonomous navigation unless ROBOT_AUTONOMOUS_CAMERA_PITCH_ENABLED=1.
MOVE_ACTIONS = TRANSLATION_ACTIONS | ROTATE_ACTIONS | LOOK_ACTIONS
AUTONOMOUS_BODY_ACTIONS = TRANSLATION_ACTIONS | ROTATE_ACTIONS
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
    "error_place_object_missing_after_action",
    "error_place_position_unreachable",
    "error_place_receptacle_mismatch",
    "error_place_target_deviation",
}
PLACE_APPROACH_ERRORS = {
    "error_receptacle_too_far",
    "error_place_pose_not_interactable",
}
PLACE_RETRY_ERRORS = {
    "error_place_no_reachable_point",
    "error_place_point_clearance",
    "error_place_exact_target_unavailable",
}
PLACE_COOLDOWN_ERRORS = {
    "error_place_no_reachable_point",
    "error_place_point_clearance",
    "error_place_exact_target_unavailable",
    "no_reachable_point",
    "receptacle_not_interactable",
    "controlled_point_unavailable",
}
PICKUP_RETARGET_ERRORS = {
    "error_no_pickup_target_in_front",
    "error_visual_pickup_target_not_grounded",
    "error_visual_pickup_instance_mismatch",
    "error_visual_pickup_ambiguous",
    "error_target_not_pickupable",
}
#涓嬮潰瀹氫箟浜嗕换鍔￠樁锟?浣犵幇鍦ㄧ殑 tidy 浠诲姟锟?ALFRED-style pick-place 鐘舵€佹満锟?#浣犲彲浠ユ妸瀹冪悊瑙ｆ垚鏈哄櫒浜哄仛鈥滄壘鐗╀綋 锟?鎹¤捣锟?锟?鎵惧湴锟?锟?鏀句笅 锟?楠岃瘉瀹屾垚鈥濈殑娴佺▼琛拷?#service_task_state_path
SERVICE_TASK_STATE_PATH = MEMORY_DIR / "service-task-state.json"
SURFACE_CANDIDATE_MEMORY_PATH = MEMORY_DIR / "surface-candidate-memory.json"
POINTCLOUD_SURFACE_SOURCE = "pointcloud_plane"
POINTCLOUD_COMPLETION_SURFACE_SOURCE = "pointcloud_plane_completion"
POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE = "pointcloud_plane_grid_completion"
DEPTH_REGION_SURFACE_SOURCE = "depth_region_geometry"
LEGACY_DEPTH_SURFACE_SOURCE = "depth_geometry"
SURFACE_REGION_SOURCES = {
    POINTCLOUD_SURFACE_SOURCE,
    POINTCLOUD_COMPLETION_SURFACE_SOURCE,
    POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE,
    DEPTH_REGION_SURFACE_SOURCE,
}
SURFACE_MEMORY_SOURCES = {
    POINTCLOUD_SURFACE_SOURCE,
    POINTCLOUD_COMPLETION_SURFACE_SOURCE,
    POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE,
    DEPTH_REGION_SURFACE_SOURCE,
    LEGACY_DEPTH_SURFACE_SOURCE,
}
#service_initial_phase锟?    phase:闃舵
SERVICE_INITIAL_PHASE = "SEARCH_PICKUP_TARGET"#浠诲姟涓€寮€濮嬪浜庝粈涔堥樁锟?#service_done_pase
SERVICE_DONE_PHASE = "TASK_DONE"#浠诲姟鏈€缁堝畬鎴愭椂鍙粈涔堥樁锟?"""鎷惧彇闃舵锟?鍏堟壘瑕佹崱鐨勪笢锟?閿佸畾锟?瀵归綈锟?鐩爣宸茬粡鎵惧埌浜嗭紝浣嗚繕娌″鍑嗭紝闇€瑕佽皟鏁存柟鍚戯拷?鎹¤捣锟?纭鎵嬮噷鏈変笢锟?""
SERVICE_PICKUP_PHASES = {
    "SEARCH_PICKUP_TARGET",
    "LOCK_PICKUP_TARGET",
    "ALIGN_PICKUP_TARGET",
    "PICK_OBJECT",
    "VERIFY_HOLDING",
}
#鏀剧疆闃舵锟?
"""
鍐嶆壘鏀剧疆锟?閿佸畾鏀剧疆锟?闈犺繎:鏀剧疆浣嶇疆宸茬粡鎵惧埌浜嗭紝浣嗚窛绂昏繕涓嶅锛岄渶瑕佸線鍓嶉潬杩戯拷?瀵归綈:鏀剧疆浣嶇疆鐪嬪埌浜嗭紝浣嗕笉澶熷眳涓紝闇€瑕佽皟鏁存柟鍚戯拷?鏀剧疆
楠岃瘉浠诲姟瀹屾垚
"""
SERVICE_PLACE_PHASES = {
    "SEARCH_RECEPTACLE",
    "LOCK_RECEPTACLE",
    "APPROACH_RECEPTACLE",
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
    performance_profile: str
    object_memory_update_interval: int
    semantic_map_update_interval: int
    log_detail: str
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


def run_yolo_service(
    image_path: str,
    timeout_seconds: int,
    *,
    depth_path: str = "",
    camera: Optional[JsonDict] = None,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
) -> ScriptResult:
    payload_data: JsonDict = {"image": image_path}
    if depth_path:
        payload_data["depth_path"] = depth_path
    if isinstance(camera, dict) and camera:
        payload_data["camera"] = camera
    labels = service_label_tokens(held_object_labels or [])
    if holding_object:
        payload_data["holding_object"] = True
    if labels:
        payload_data["held_object_labels"] = labels
        payload_data["held_object_label"] = labels[0]
        payload_data["held_object_family"] = held_object_family_for_labels(labels)
    payload = json.dumps(payload_data, ensure_ascii=False).encode("utf-8")
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


def normalize_service_label(label: Any) -> str:
    value = str(label or "").strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def service_label_tokens(*values: Any) -> List[str]:
    labels: List[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            parts = value
        else:
            parts = str(value or "").split(",")
        for part in parts:
            raw = str(part or "").strip()
            normalized = normalize_service_label(raw)
            for token in (normalized, raw.strip()):
                if token and token not in labels:
                    labels.append(token)
    return labels


HELD_FOOD_LABELS = {"apple", "banana", "lettuce", "orange", "potato", "tomato"}


def held_object_family_for_labels(labels: Iterable[Any]) -> str:
    tokens = {normalize_service_label(label) for label in labels if normalize_service_label(label)}
    if tokens & HELD_FOOD_LABELS:
        return "food"
    if tokens:
        return "pickup_target"
    return "unknown"


def pickup_approach_verify_label_allowed(candidate: JsonDict) -> bool:
    raw_allowed = str(os.getenv("ROBOT_PICKUP_APPROACH_VERIFY_LABELS", "") or "").strip()
    if raw_allowed in {"*", "all", "ALL"}:
        return True
    allowed = {
        normalize_service_label(item)
        for item in (raw_allowed.split(",") if raw_allowed else HELD_FOOD_LABELS)
        if normalize_service_label(item)
    }
    labels = {
        normalize_service_label(item)
        for item in service_label_tokens(candidate.get("label"), candidate.get("raw_label"))
        if normalize_service_label(item)
    }
    return bool(labels & allowed)


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


def env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
        "MoveLeft",
        "MoveRight",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
    ]:
        if action.startswith(known):
            return known
    return action


class PatrolRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.manager = StateManager(MEMORY_DIR)
        self.navigation = NavigationMemory(MEMORY_DIR)
        self.object_memory = ObjectMemory(MEMORY_DIR)
        self.placement_viewpoints = PlacementViewpointPlanner(MEMORY_DIR)
        self.local_costmap = LocalCostmap(MEMORY_DIR)
        # Keep camera pitch as a manually callable low-level capability, but
        # exclude it from normal patrol/A*/costmap recovery by default.
        self.allow_autonomous_camera_pitch = env_bool("ROBOT_AUTONOMOUS_CAMERA_PITCH_ENABLED", False)
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
        self.pending_object_memory_target: Optional[JsonDict] = None
        self.holding_object = False
        self.held_move_blocked_until: Dict[str, int] = {}
        self.service_failures: Dict[str, int] = {}
        self.service_state = self.load_service_task_state()
        self.holding_object = bool(self.service_state.get("holding_object", False))
        self.surface_candidate_memory = self.load_surface_candidate_memory()
        self.pending_service_completions: List[str] = []
        self.pending_placed_objects: List[str] = []
        self.current_analysis: JsonDict = {}
        self.current_local_costmap: JsonDict = {}

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

    def default_surface_candidate_memory(self) -> JsonDict:
        return {
            "schema_version": 1,
            "cooldown_default_steps": int(env_float("ROBOT_SURFACE_CANDIDATE_COOLDOWN_STEPS", 8.0)),
            "candidates": {},
            "failed_surfaces": {},
        }

    def load_surface_candidate_memory(self) -> JsonDict:
        if not SURFACE_CANDIDATE_MEMORY_PATH.exists():
            return self.default_surface_candidate_memory()
        try:
            data = json.loads(SURFACE_CANDIDATE_MEMORY_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            return self.default_surface_candidate_memory()
        if not isinstance(data, dict):
            return self.default_surface_candidate_memory()
        memory = self.default_surface_candidate_memory()
        memory.update(data)
        if not isinstance(memory.get("candidates"), dict):
            memory["candidates"] = {}
        if not isinstance(memory.get("failed_surfaces"), dict):
            memory["failed_surfaces"] = {}
        return memory

    def save_surface_candidate_memory(self) -> None:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return
        if self.config.dry_run:
            return
        self.surface_candidate_memory["last_update"] = datetime.now().astimezone().isoformat(timespec="seconds")
        SURFACE_CANDIDATE_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.surface_candidate_memory, ensure_ascii=False, indent=2) + "\n"
        SURFACE_CANDIDATE_MEMORY_PATH.write_text(payload, encoding="utf-8")

    def surface_candidate_identifier(self, candidate: Optional[JsonDict]) -> str:
        if not isinstance(candidate, dict):
            return "surface:unknown"
        for key in ("failed_candidate_id", "surface_candidate_id", "id"):
            value = candidate.get(key)
            if value:
                return str(value)
        return candidate_key(candidate)

    def surface_cooldown_remaining(self, candidate: Optional[JsonDict]) -> int:
        candidate_id = self.surface_candidate_identifier(candidate)
        entries = self.surface_candidate_memory.get("candidates")
        entry = entries.get(candidate_id) if isinstance(entries, dict) else None
        if not isinstance(entry, dict):
            failed_entries = self.surface_candidate_memory.get("failed_surfaces")
            entry = failed_entries.get(candidate_id) if isinstance(failed_entries, dict) else None
        if not isinstance(entry, dict):
            return 0
        try:
            until_step = int(entry.get("cooldown_until_step", 0) or 0)
        except (TypeError, ValueError):
            until_step = 0
        return max(0, until_step - self.current_step_count())

    def annotate_surface_memory(self, candidate: JsonDict) -> None:
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return
        if not (
            candidate.get("surface_candidate_id")
            or candidate.get("id")
            or str(candidate.get("surface_candidate_source") or "") in SURFACE_MEMORY_SOURCES
        ):
            return
        remaining = self.surface_cooldown_remaining(candidate)
        memory_checks = candidate.get("memory_checks") if isinstance(candidate.get("memory_checks"), dict) else {}
        memory_checks = dict(memory_checks)
        memory_checks["failed_recently"] = bool(remaining > 0)
        memory_checks["cooldown_remaining"] = int(remaining)
        candidate["memory_checks"] = memory_checks
        candidate["failed_recently"] = bool(remaining > 0)
        candidate["cooldown_remaining"] = int(remaining)
        if remaining > 0:
            candidate["history_penalty"] = round(min(1.0, remaining / max(1.0, env_float("ROBOT_SURFACE_CANDIDATE_COOLDOWN_STEPS", 8.0))), 4)
            candidate["visual_place_ready"] = False
            candidate["final_place_ready"] = False
            candidate["place_now"] = False
            reasons = list(candidate.get("rejection_reasons") or [])
            cooldown_reason = f"cooldown:{remaining}"
            if cooldown_reason not in reasons:
                reasons.append(cooldown_reason)
            candidate["rejection_reasons"] = reasons

    def mark_surface_candidate_failed(
        self,
        candidate: Optional[JsonDict],
        *,
        result_type: str,
        failed_candidate_id: Optional[str] = None,
    ) -> None:
        if not isinstance(candidate, dict) and not failed_candidate_id:
            return
        candidate_id = str(failed_candidate_id or self.surface_candidate_identifier(candidate))
        entries = self.surface_candidate_memory.setdefault("candidates", {})
        if not isinstance(entries, dict):
            entries = {}
            self.surface_candidate_memory["candidates"] = entries
        failed_entries = self.surface_candidate_memory.setdefault("failed_surfaces", {})
        if not isinstance(failed_entries, dict):
            failed_entries = {}
            self.surface_candidate_memory["failed_surfaces"] = failed_entries
        entry = entries.get(candidate_id) if isinstance(entries.get(candidate_id), dict) else {}
        failure_count = int(entry.get("failure_count", 0) or 0) + 1
        base_cooldown = max(1, int(env_float("ROBOT_SURFACE_CANDIDATE_COOLDOWN_STEPS", 8.0)))
        cooldown_steps = base_cooldown * min(4, failure_count)
        until_step = self.current_step_count() + cooldown_steps
        short = self.short_candidate(candidate) if isinstance(candidate, dict) else None
        center = candidate.get("center") if isinstance(candidate, dict) and isinstance(candidate.get("center"), dict) else {}
        center_3d = candidate.get("center_3d") if isinstance(candidate, dict) and isinstance(candidate.get("center_3d"), dict) else {}
        parent_object = str(candidate.get("parent_object") or candidate.get("label") or "") if isinstance(candidate, dict) else ""
        memory_entry = {
            "candidate_id": candidate_id,
            "parent_object": parent_object,
            "center_2d": dict(center),
            "center_3d": dict(center_3d),
            "failure_count": failure_count,
            "failed_count": failure_count,
            "last_result_type": result_type,
            "last_reason": result_type,
            "last_failed_step": self.current_step_count(),
            "cooldown_until_step": until_step,
            "cooldown_steps": cooldown_steps,
            "candidate": short,
            "last_update": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        entries[candidate_id] = dict(memory_entry)
        failed_entries[candidate_id] = dict(memory_entry)
        self.save_surface_candidate_memory()
        self.emit(
            "surface_candidate_cooldown_written",
            {
                "candidate_id": candidate_id,
                "result_type": result_type,
                "failure_count": failure_count,
                "cooldown_steps": cooldown_steps,
                "cooldown_until_step": until_step,
            },
        )

    def reset_service_task_state(self, *, reason: str) -> None:
        self.service_state = self.default_service_task_state()
        try:
            self.placement_viewpoints.reset(reason=reason)
        except Exception as exc:
            self.emit("placement_viewpoint_error", {"phase": "reset", "message": str(exc)})
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
    #鎶婂綋鍓嶄换鍔￠樁娈靛垏鎹㈡垚锛歱hase闃舵
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

    def held_object_label_tokens(self) -> List[str]:
        return service_label_tokens(
            self.service_state.get("held_object_labels"),
            self.service_state.get("held_object_label"),
            self.service_state.get("held_object_raw_label"),
        )

    def set_held_object_context(self, *, label: Any = None, raw_label: Any = None, labels: Any = None) -> None:
        tokens = service_label_tokens(labels, label, raw_label)
        if tokens:
            self.service_state["held_object_labels"] = tokens
            self.service_state["held_object_label"] = normalize_service_label(label or tokens[0])
            self.service_state["held_object_raw_label"] = str(raw_label or label or tokens[0]).strip()
            self.service_state["held_object_family"] = held_object_family_for_labels(tokens)
        self.service_state["holding_object"] = bool(self.holding_object)

    def set_held_object_context_from_candidate(self, candidate: Optional[JsonDict]) -> None:
        if not isinstance(candidate, dict):
            return
        self.set_held_object_context(
            label=candidate.get("label"),
            raw_label=candidate.get("raw_label") or candidate.get("label"),
        )
        if candidate.get("object_memory_track_id"):
            self.service_state["held_object_track_id"] = str(candidate.get("object_memory_track_id"))

    def clear_held_object_context(self) -> None:
        self.service_state["held_object_label"] = None
        self.service_state["held_object_raw_label"] = None
        self.service_state["held_object_family"] = "unknown"
        self.service_state["held_object_labels"] = []
        self.service_state["held_object_track_id"] = None
        self.service_state["holding_object"] = bool(self.holding_object)

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
            self.service_state["target_track_id"] = candidate.get("object_memory_track_id")
            self.service_state["target_last_seen_step"] = step
            self.service_state["target_lost_scan_count"] = 0
            self.service_state["target_attempts"] = int(self.service_state.get("target_attempts", 0) or 0) + 1
            phase = "LOCK_PICKUP_TARGET"
        else:
            self.service_state["receptacle_label"] = label
            self.service_state["receptacle_raw_label"] = raw_label
            self.service_state["receptacle_signature"] = signature
            self.service_state["receptacle_track_id"] = candidate.get("object_memory_track_id")
            self.service_state["receptacle_last_seen_step"] = step
            self.service_state["receptacle_lost_scan_count"] = 0
            self.service_state["receptacle_attempts"] = int(self.service_state.get("receptacle_attempts", 0) or 0) + 1
            self.service_state["receptacle_action_hint"] = None
            self.service_state["receptacle_action_hint_until_step"] = None
            self.service_state["receptacle_last_position_hint"] = str(candidate.get("position_hint") or "")
            phase = "LOCK_RECEPTACLE"
        self.set_service_phase(phase, reason=reason, candidate=candidate)

    def mark_released_pickup_track_stale(
        self,
        *,
        track_id: str,
        reason: str,
        had_visual_lock: bool,
    ) -> None:
        """Demote or reject a released pickup track after verification fails.

        A first visual loss is marked stale so a real target can be recovered.
        Once a memory-only target has failed to reacquire, or the same track has
        already been released before, reject it so it cannot keep stealing free
        exploration as an object-memory goal.
        """
        track_text = str(track_id or "").strip()
        reason_text = str(reason or "").strip()
        if not track_text:
            return
        if not (
            reason_text.startswith("locked_pickup_target_not_actionable")
            or reason_text.startswith("locked_pickup_target_lost")
            or reason_text.startswith("pickup_target_not_visible")
        ):
            return
        try:
            track_status: Optional[JsonDict] = None
            try:
                memory = self.object_memory.load_memory()
                tracks = memory.get("tracks") if isinstance(memory.get("tracks"), dict) else {}
                track_status = tracks.get(track_text) if isinstance(tracks.get(track_text), dict) else None
            except Exception:
                track_status = None
            interaction = track_status.get("interaction") if isinstance(track_status, dict) and isinstance(track_status.get("interaction"), dict) else {}
            previously_released = bool(interaction.get("last_stale_step") or interaction.get("last_rejected_step"))
            reject_now = bool(
                (not had_visual_lock)
                or previously_released
                or reason_text.startswith("locked_pickup_target_not_actionable")
                or reason_text.startswith("locked_pickup_target_lost_or_not_actionable")
            )
            if reject_now:
                status_change = self.object_memory.mark_rejected_false_positive(
                    track_id=track_text,
                    step=self.current_step_count(),
                    reason=f"pickup_target_rejected_after_release:{reason_text}",
                )
            else:
                status_change = self.object_memory.mark_stale(
                    track_id=track_text,
                    step=self.current_step_count(),
                    reason=f"pickup_lock_released:{reason_text}",
                )
            if isinstance(status_change, dict):
                self.emit("object_status_changed", status_change)
        except Exception as exc:
            self.emit(
                "object_memory_error",
                {
                    "phase": "mark_released_pickup_track_stale",
                    "track_id": track_text,
                    "reason": reason_text,
                    "message": str(exc),
                },
            )

    def clear_service_lock(self, *, role: str, reason: str) -> None:
        if role == "pickup":
            old_track_id = str(self.service_state.get("target_track_id") or "").strip()
            had_visual_lock = bool(
                self.service_state.get("target_label")
                or self.service_state.get("target_raw_label")
                or self.service_state.get("target_signature")
            )
            self.mark_released_pickup_track_stale(
                track_id=old_track_id,
                reason=reason,
                had_visual_lock=had_visual_lock,
            )
            self.service_state["target_label"] = None
            self.service_state["target_raw_label"] = None
            self.service_state["target_signature"] = None
            self.service_state["target_track_id"] = None
            self.service_state["target_last_seen_step"] = None
            self.service_state["target_lost_scan_count"] = 0
            self.service_state["target_attempts"] = 0
            next_phase = SERVICE_INITIAL_PHASE
        else:
            self.service_state["receptacle_label"] = None
            self.service_state["receptacle_raw_label"] = None
            self.service_state["receptacle_signature"] = None
            self.service_state["receptacle_track_id"] = None
            self.service_state["receptacle_last_seen_step"] = None
            self.service_state["receptacle_lost_scan_count"] = 0
            self.service_state["receptacle_attempts"] = 0
            self.service_state["receptacle_action_hint"] = None
            self.service_state["receptacle_action_hint_until_step"] = None
            self.service_state["receptacle_last_position_hint"] = None
            next_phase = "SEARCH_RECEPTACLE"
        self.set_service_phase(next_phase, reason=reason)

    def store_receptacle_action_hint(self, action: Any, *, steps: int = 4) -> None:
        hint = str(action or "").strip()
        if hint not in MOVE_ACTIONS:
            self.service_state["receptacle_action_hint"] = None
            self.service_state["receptacle_action_hint_until_step"] = None
            return
        self.service_state["receptacle_action_hint"] = hint
        self.service_state["receptacle_action_hint_until_step"] = self.current_step_count() + max(1, int(steps))

    def current_receptacle_action_hint(self) -> Optional[str]:
        hint = str(self.service_state.get("receptacle_action_hint") or "").strip()
        if hint not in MOVE_ACTIONS:
            return None
        try:
            until_step = int(self.service_state.get("receptacle_action_hint_until_step", -1) or -1)
        except (TypeError, ValueError):
            return None
        if until_step < self.current_step_count():
            self.service_state["receptacle_action_hint"] = None
            self.service_state["receptacle_action_hint_until_step"] = None
            return None
        return hint

    def remember_receptacle_position_hint(self, candidate: Optional[JsonDict]) -> None:
        if not isinstance(candidate, dict):
            return
        position_hint = str(candidate.get("position_hint") or "").strip()
        if position_hint:
            self.service_state["receptacle_last_position_hint"] = position_hint

    def block_held_move_action(self, action: Any, *, reason: str, steps: int = 4) -> None:
        action_text = str(action or "").strip()
        if action_text not in MOVE_ACTIONS:
            return
        until_step = self.current_step_count() + max(1, int(steps))
        self.held_move_blocked_until[action_text] = until_step
        self.emit(
            "held_move_action_blocked",
            {
                "action": action_text,
                "reason": reason,
                "until_step": until_step,
            },
        )

    def held_move_action_blocked(self, action: str) -> bool:
        if action not in MOVE_ACTIONS:
            return False
        until_step = int(self.held_move_blocked_until.get(action, -1) or -1)
        if until_step < self.current_step_count():
            self.held_move_blocked_until.pop(action, None)
            return False
        return True

    def held_move_action_recently_bad(self, action: str) -> bool:
        return bool(self.holding_object and (self.held_move_action_blocked(action) or self.recent_action_failed(action)))

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

    def active_recently_placed_track_ids(self) -> List[str]:
        tracks = self.service_state.get("recently_placed_tracks")
        if not isinstance(tracks, dict):
            return []
        active: List[str] = []
        changed = False
        for track_id, entry in list(tracks.items()):
            if isinstance(entry, dict):
                raw_until = entry.get("until_step")
            else:
                raw_until = entry
            try:
                until_step = int(raw_until)
            except (TypeError, ValueError):
                until_step = -1
            if until_step < self.current_step_count():
                tracks.pop(track_id, None)
                changed = True
                continue
            active.append(str(track_id))
        if changed:
            self.service_state["recently_placed_tracks"] = tracks
            self.save_service_task_state()
        return active

    def recently_placed_label_blocked(self, candidate: JsonDict) -> bool:
        track_id = str(candidate.get("object_memory_track_id") or "").strip()
        return bool(track_id and track_id in set(self.active_recently_placed_track_ids()))

    def suppress_recently_placed_track(
        self,
        *,
        track_id: Optional[str],
        label: str,
        steps: int = 12,
    ) -> None:
        normalized = str(label or "").strip().lower()
        track_text = str(track_id or "").strip()
        if not track_text:
            return
        tracks = self.service_state.setdefault("recently_placed_tracks", {})
        if not isinstance(tracks, dict):
            tracks = {}
            self.service_state["recently_placed_tracks"] = tracks
        tracks[track_text] = {
            "track_id": track_text,
            "label": normalized or None,
            "until_step": self.current_step_count() + max(1, int(steps)),
        }
        self.save_service_task_state()

    def pickup_candidate_memory_cooldown_active(self, candidate: JsonDict) -> bool:
        track_id = str(candidate.get("object_memory_track_id") or "").strip() if isinstance(candidate, dict) else ""
        if not track_id:
            return False
        try:
            remaining = self.object_memory.pickup_cooldown_remaining(
                track_id=track_id,
                candidate=candidate,
                step=self.current_step_count(),
            )
        except Exception as exc:
            self.emit(
                "object_memory_error",
                {
                    "phase": "pickup_candidate_memory_cooldown",
                    "track_id": track_id,
                    "message": str(exc),
                },
            )
            return False
        if remaining <= 0:
            return False

        memory_checks = candidate.get("memory_checks") if isinstance(candidate.get("memory_checks"), dict) else {}
        memory_checks = dict(memory_checks)
        memory_checks["pickup_cooldown_remaining"] = int(remaining)
        candidate["memory_checks"] = memory_checks
        candidate["object_memory_cooldown_remaining"] = int(remaining)

        # A stale verification cooldown only suppresses weak re-locks. If a
        # later observation becomes immediately actionable or gets strong RGB-D
        # floor-contact evidence, allow the true object to reactivate.
        if truthy(candidate.get("pickup_now")) or self.pickup_candidate_floor_contact_like(candidate):
            return False
        return True

    def pickup_candidate_floor_contact_like(self, candidate: JsonDict) -> bool:
        """Return True when RGB-D bottom-contact geometry supports a floor pickup.

        A large 2-D CounterTop/Cabinet box is only context.  A fitted floor
        plane plus a near-floor support strip is stronger evidence and should
        keep a visible object in the local pursuit controller.  Perception may
        still reject that contact when the support strip height is above the
        floor band; the runner must honor that rejection so floor-only tidy
        mode does not chase tabletop mugs/cups.
        """

        if not isinstance(candidate, dict):
            return False
        if str(candidate.get("floor_contact_rejected_reason") or "").strip():
            return False
        detail = candidate.get("floor_contact_geometry") if isinstance(candidate.get("floor_contact_geometry"), dict) else {}
        if not bool(detail.get("available") and detail.get("contact_floor_like")):
            return False
        try:
            support_height = float(detail.get("support_height_m"))
        except (TypeError, ValueError):
            support_height = None
        try:
            max_support_height = float(
                candidate.get("floor_contact_max_support_height_m")
                or os.getenv("ROBOT_DEPTH_FLOOR_CONTACT_MAX_SUPPORT_HEIGHT_M", "0.16")
            )
        except (TypeError, ValueError):
            max_support_height = 0.16
        if support_height is not None and math.isfinite(support_height) and support_height > max_support_height:
            return False
        return True

    def pickup_candidate_is_approach_verifiable(self, candidate: JsonDict) -> bool:
        """Return True for a visible pickup target worth approaching to re-check.

        This is deliberately weaker than pickup_task_filter_allowed(): in
        floor-only mode a small/far floor object can be mis-tagged as
        surface_or_elevated. We should not pick it immediately, but we should
        approach/align and re-observe instead of falling back to generic
        frontier exploration.
        """
        if self.holding_object:
            return False
        if not isinstance(candidate, dict):
            return False
        if str(candidate.get("task_semantic_class") or "") != "pickup_target":
            return False
        if self.is_suppressed(candidate):
            return False
        if self.recently_placed_label_blocked(candidate):
            return False
        if not self.pickup_label_allowed(candidate):
            return False
        if self.pickup_candidate_memory_cooldown_active(candidate):
            return False

        try:
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            area_ratio = 0.0

        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        try:
            bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        except (TypeError, ValueError):
            bottom_y_ratio = 0.0

        position_hint = str(candidate.get("position_hint") or "")
        surface_hint = str(candidate.get("surface_hint") or "")
        if position_hint not in {"front-left", "front-center", "front-right"}:
            return False

        min_conf = env_float("ROBOT_PICKUP_APPROACH_VERIFY_MIN_CONF", 0.60)
        min_area = env_float("ROBOT_PICKUP_APPROACH_VERIFY_MIN_AREA", 0.00035)
        if confidence < min_conf or area_ratio < min_area:
            return False

        ground_distance = self.candidate_ground_distance(candidate)
        if ground_distance is not None:
            min_distance = env_float("ROBOT_PICKUP_APPROACH_VERIFY_MIN_GROUND_DISTANCE", 0.25)
            max_distance = env_float("ROBOT_PICKUP_APPROACH_VERIFY_MAX_GROUND_DISTANCE", 2.20)
            if ground_distance < min_distance or ground_distance > max_distance:
                return False

        center_3d = candidate.get("center_3d") if isinstance(candidate.get("center_3d"), dict) else {}
        height_m: Optional[float] = None
        for value in (candidate.get("height_m"), candidate.get("height"), geometry.get("height_m"), center_3d.get("y")):
            try:
                height_m = float(value)
                if math.isfinite(height_m):
                    break
            except (TypeError, ValueError):
                height_m = None
        low_height_like = bool(
            height_m is not None
            and height_m <= env_float("ROBOT_PICKUP_APPROACH_VERIFY_MAX_HEIGHT_M", 0.15)
        )
        rgbd_floor_contact = self.pickup_candidate_floor_contact_like(candidate)
        floor_like = bool(
            rgbd_floor_contact
            or surface_hint == "floor"
            or truthy(candidate.get("is_floor_level"))
        )
        explicit_elevated = bool(
            surface_hint in {"surface_or_elevated", "support_surface", "table", "counter_top", "countertop"}
            and candidate.get("is_floor_level") is False
        )
        bottom_floor_like = bool(
            bottom_y_ratio >= env_float("ROBOT_PICKUP_FLOOR_MIN_BOTTOM_RATIO", 0.78)
        )
        approach_verify_bottom_like = bool(
            bottom_y_ratio >= env_float("ROBOT_PICKUP_APPROACH_VERIFY_MIN_BOTTOM_RATIO", 0.72)
        )
        approach_verify_floor_like = bool(
            pickup_approach_verify_label_allowed(candidate)
            and low_height_like
            and approach_verify_bottom_like
            and not truthy(candidate.get("support_context_blocked"))
        )

        if truthy(candidate.get("support_context_blocked")) and not rgbd_floor_contact:
            return False

        if explicit_elevated and not (rgbd_floor_contact or bottom_floor_like or approach_verify_floor_like):
            # A low center height alone is too weak for floor-only pursuit when
            # perception already says the object is elevated.  Real floor
            # objects still pass through bottom-band, RGB-D contact evidence,
            # or the food-only approach-to-verify path.
            return False

        if surface_hint == "surface_or_elevated" and not (rgbd_floor_contact or floor_like or low_height_like):
            # Keep floor-only semantics: high/tabletop objects should not be chased
            # by the floor pickup task. Low-height ambiguous objects can be checked.
            return False

        if str(self.config.pickup_surface_policy or "floor-only").strip().lower() == "any-surface":
            return bool(truthy(candidate.get("reachable")) or floor_like or low_height_like)

        # In floor-only tidy mode, reachability is not floor evidence.  A mug
        # on a counter can be reachable and still must not enter pickup pursuit.
        return bool(floor_like or low_height_like or bottom_floor_like)

    def pickup_candidate_is_actionable_or_promising(self, candidate: JsonDict) -> bool:
        """Keep tidy mode from chasing bad boxes, but allow approach-to-verify."""
        if not self.pickup_label_allowed(candidate):
            return False
        if self.pickup_candidate_memory_cooldown_active(candidate):
            return False
        if self.pickup_candidate_is_approach_verifiable(candidate):
            candidate["approach_verify_pickup"] = True
            candidate.setdefault("approach_verify_reason", "visible_pickup_target_not_action_ready")
            return True
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
        floor_like = self.pickup_candidate_floor_contact_like(candidate) or surface_hint == "floor" or (
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

    def visible_pickup_local_pursuit_candidates(self, analysis: JsonDict) -> List[JsonDict]:
        """Return raw visible pickup targets worth local pursuit before global A*.

        This deliberately runs before object-memory fallback.  A visible target
        may be slightly off-center or awaiting RGB-D re-check, but handing it to
        a coarse global viewpoint immediately can rotate the camera away from a
        perfectly usable local target.
        """

        pools: List[Any] = [analysis.get("best_pickup_candidate")]
        pools.extend(analysis.get("service_candidates", []) or [])
        candidates: List[JsonDict] = []
        seen_keys = set()
        for candidate in pools:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("task_semantic_class") or "") != "pickup_target":
                continue
            if not self.pickup_candidate_is_approach_verifiable(candidate):
                continue
            key = candidate_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidate["approach_verify_pickup"] = True
            candidate.setdefault("approach_verify_reason", "visible_pickup_local_pursuit_before_global_navigation")
            candidates.append(candidate)
        candidates.sort(key=lambda item: self.service_candidate_score(item, task_class="pickup_target"), reverse=True)
        return candidates

    def visible_service_candidates(self, analysis: JsonDict, *, task_class: str) -> List[JsonDict]:
        pools: List[Any] = []
        if task_class == "pickup_target":
            pools.append(analysis.get("best_pickup_candidate"))
        elif task_class == "place_receptacle":
            pools.append(analysis.get("best_surface_candidate"))
            pools.extend(analysis.get("visual_ready_surface_regions", []) or [])
        pools.extend(analysis.get("service_candidates", []) or [])
        pools.extend(analysis.get("receptacle_candidates", []) or [])
        pools.extend(analysis.get("surface_regions", []) or [])

        candidates: List[JsonDict] = []
        seen_keys = set()
        for candidate in pools:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("task_semantic_class") != task_class:
                continue
            if (
                task_class == "pickup_target"
                and str(candidate.get("object_memory_status") or "") in {"rejected", "rejected_false_positive"}
            ):
                self.emit(
                    "pickup_candidate_skipped",
                    {
                        "reason": "object_memory_track_rejected",
                        "track_id": candidate.get("object_memory_track_id"),
                        "status": candidate.get("object_memory_status"),
                        "candidate": self.short_candidate(candidate),
                    },
                )
                continue
            if task_class == "place_receptacle":
                self.annotate_surface_memory(candidate)
            if self.is_suppressed(candidate):
                continue
            if not truthy(candidate.get("reachable")):
                if not (task_class == "pickup_target" and self.pickup_candidate_is_approach_verifiable(candidate)):
                    continue
            if task_class == "pickup_target" and not self.pickup_candidate_is_actionable_or_promising(candidate):
                continue
            if task_class == "pickup_target" and self.recently_placed_label_blocked(candidate):
                continue
            if task_class == "place_receptacle" and self.receptacle_visual_box_ambiguous(candidate):
                continue
            if task_class == "place_receptacle" and truthy(candidate.get("failed_recently")):
                continue
            if (
                task_class == "place_receptacle"
                and str(candidate.get("surface_candidate_source") or "") not in SURFACE_REGION_SOURCES
            ):
                continue
            if task_class == "place_receptacle" and not truthy(candidate.get("visual_place_ready")):
                continue
            if task_class == "place_receptacle" and not self.receptacle_candidate_is_actionable_or_promising(candidate):
                continue
            key = candidate_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)
        candidates.sort(key=lambda item: self.service_candidate_score(item, task_class=task_class), reverse=True)
        return candidates

    def visible_receptacle_context_candidates(self, analysis: JsonDict) -> List[JsonDict]:
        """Visible receptacles that are not yet place-ready but should guide alignment.

        This is the ALFRED-style separation between interaction and navigation:
        a visible CounterTop may be too far or off-center for placement, but if
        the agent is already holding something it should rotate/approach toward
        that receptacle instead of handing control to frontier exploration.
        """
        pools: List[Any] = [
            analysis.get("best_surface_candidate"),
            analysis.get("best_receptacle_candidate"),
            *(analysis.get("surface_regions", []) or []),
            *(analysis.get("receptacle_candidates", []) or []),
            *(analysis.get("service_candidates", []) or []),
        ]
        candidates: List[JsonDict] = []
        seen_keys = set()
        for candidate in pools:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
                continue
            surface_source = str(candidate.get("surface_candidate_source") or "")
            if surface_source in SURFACE_REGION_SOURCES and not truthy(candidate.get("visual_place_ready")):
                continue
            self.annotate_surface_memory(candidate)
            if self.is_suppressed(candidate):
                continue
            if truthy(candidate.get("failed_recently")):
                continue
            if not truthy(candidate.get("reachable")):
                continue
            if not truthy(candidate.get("is_support_surface")):
                continue
            if self.receptacle_visual_box_ambiguous(candidate):
                continue
            position_hint = str(candidate.get("position_hint") or "")
            if position_hint not in {"front-left", "front-center", "front-right"}:
                continue
            try:
                confidence = float(candidate.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.50:
                continue
            key = candidate_key(candidate)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(candidate)

        def score(candidate: JsonDict) -> float:
            value = self.service_candidate_score(candidate, task_class="place_receptacle")
            if truthy(candidate.get("needs_alignment")):
                value += 0.8
            if truthy(candidate.get("needs_approach")):
                value += 0.4
            return value

        candidates.sort(key=score, reverse=True)
        return candidates

    def service_candidate_score(self, candidate: JsonDict, *, task_class: str) -> float:
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        cx_ratio = float(geometry.get("cx_ratio", 0.5) or 0.5)
        center_bonus = max(0.0, 0.5 - abs(cx_ratio - 0.5))
        value = confidence * 2.0 + min(area_ratio, 0.20) + center_bonus
        surface_source = str(candidate.get("surface_candidate_source") or "")
        if surface_source == LEGACY_DEPTH_SURFACE_SOURCE:
            value += float(candidate.get("score", 0.0) or 0.0) * 3.0
        if surface_source in SURFACE_REGION_SOURCES:
            value += float(candidate.get("score", 0.0) or 0.0) * 3.0
            if truthy(candidate.get("visual_place_ready")):
                value += 1.0
            if truthy(candidate.get("affordance_ready")):
                value += 1.0
        memory_checks = candidate.get("memory_checks") if isinstance(candidate.get("memory_checks"), dict) else {}
        try:
            history_penalty = float(candidate.get("history_penalty", 0.0) or 0.0)
        except (TypeError, ValueError):
            history_penalty = 0.0
        if history_penalty > 0:
            value -= min(2.5, history_penalty * 2.5)
        if truthy(candidate.get("failed_recently")) or truthy(memory_checks.get("failed_recently")):
            value -= 100.0
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

    def candidate_ground_distance(self, candidate: JsonDict) -> Optional[float]:
        for value in (candidate.get("ground_distance"), candidate.get("ground_distance_m")):
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
        for container_key in ("geometry", "depth", "center_3d"):
            container = candidate.get(container_key)
            if not isinstance(container, dict):
                continue
            for key in ("ground_distance_m", "ground_distance"):
                try:
                    result = float(container.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(result):
                    return result
        return None

    def depth_surface_place_distance_ready(self, candidate: JsonDict) -> bool:
        if str(candidate.get("surface_candidate_source") or "") != "depth_geometry":
            return True
        ground_distance = self.candidate_ground_distance(candidate)
        if ground_distance is None:
            return False
        min_distance = env_float("ROBOT_DEPTH_SURFACE_PLACE_NOW_MIN_GROUND_DISTANCE", 0.50)
        max_distance = env_float(
            "ROBOT_DEPTH_SURFACE_PLACE_NOW_MAX_GROUND_DISTANCE",
            env_float("ROBOT_PLACE_MAX_DISTANCE", 1.0),
        )
        return bool(min_distance <= ground_distance <= max_distance)

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
        if self.pickup_candidate_floor_contact_like(candidate):
            return True
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
        if str(candidate.get("surface_candidate_source") or "") == "depth_geometry":
            return False
        if self.service_surface_has_interaction_point(candidate):
            # A surface candidate is executed at its checked interaction point;
            # its bbox may only be an envelope around an irregular free region.
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

    def receptacle_candidate_is_actionable_or_promising(self, candidate: JsonDict) -> bool:
        """Keep only receptacles that are worth navigating toward.

        YOLO can see far table fragments at the image edge. Those are useful as
        context, but they are not good enough to lock a place subgoal. ALFRED
        separates navigation from interaction; this filter keeps the lock set
        focused on a receptacle with enough visual evidence to approach.
        """
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return False
        self.annotate_surface_memory(candidate)
        if truthy(candidate.get("failed_recently")):
            return False
        if not truthy(candidate.get("reachable")):
            return False
        if not truthy(candidate.get("is_support_surface")):
            return False
        if self.receptacle_visual_box_ambiguous(candidate):
            return False

        if str(candidate.get("surface_candidate_source") or "") == "depth_geometry":
            try:
                min_score = float(os.getenv("ROBOT_DEPTH_SURFACE_LOCK_MIN_SCORE", "0.40"))
            except (TypeError, ValueError):
                min_score = 0.40
            score = float(candidate.get("score", 0.0) or 0.0)
            return bool(
                not truthy(candidate.get("blocked"))
                and score >= min_score
                and (
                    truthy(candidate.get("place_now"))
                    or truthy(candidate.get("needs_alignment"))
                    or truthy(candidate.get("needs_approach"))
                    or truthy(candidate.get("reachable"))
                )
            )

        if str(candidate.get("surface_candidate_source") or "") in SURFACE_REGION_SOURCES:
            return bool(
                truthy(candidate.get("visual_place_ready"))
                and truthy(candidate.get("affordance_ready"))
            )

        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        position_hint = str(candidate.get("position_hint") or "")
        front_edge_like = bool(
            truthy(candidate.get("front_edge_receptacle"))
            or truthy(candidate.get("broad_front_receptacle"))
            or self.receptacle_front_edge_candidate(candidate)
        )

        try:
            min_conf = float(os.getenv("ROBOT_PLACE_LOCK_MIN_CONF", "0.72"))
        except (TypeError, ValueError):
            min_conf = 0.72
        try:
            min_area = float(os.getenv("ROBOT_PLACE_LOCK_MIN_AREA", "0.025"))
        except (TypeError, ValueError):
            min_area = 0.025
        try:
            edge_min_conf = float(os.getenv("ROBOT_PLACE_EDGE_LOCK_MIN_CONF", "0.70"))
        except (TypeError, ValueError):
            edge_min_conf = 0.70
        try:
            edge_min_area = float(os.getenv("ROBOT_PLACE_EDGE_LOCK_MIN_AREA", "0.010"))
        except (TypeError, ValueError):
            edge_min_area = 0.010

        if front_edge_like or truthy(candidate.get("place_now")):
            return bool(
                position_hint == "front-center"
                and confidence >= edge_min_conf
                and area_ratio >= edge_min_area
            )

        if position_hint not in {"front-left", "front-center", "front-right"}:
            return False
        if confidence < min_conf or area_ratio < min_area:
            return False
        # A tiny strip high in the image is usually a far tabletop edge. It can
        # guide exploration, but should not become the active receptacle lock.
        if position_hint != "front-center" and bottom_y_ratio < 0.55:
            return False
        return True

    def service_receptacle_visual_interaction_ready(self, candidate: JsonDict) -> bool:
        """Return True only when the visible receptacle looks close enough to try placement."""
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return False
        if str(candidate.get("surface_candidate_source") or "") in SURFACE_REGION_SOURCES:
            self.annotate_surface_memory(candidate)
            return bool(
                truthy(candidate.get("visual_place_ready"))
                and not truthy(candidate.get("failed_recently"))
                and not truthy(candidate.get("blocked"))
                and truthy(candidate.get("affordance_ready"))
            )
        if truthy(candidate.get("needs_alignment")) or truthy(candidate.get("needs_approach")):
            return False
        if str(candidate.get("surface_candidate_source") or "") == "depth_geometry":
            return bool(
                truthy(candidate.get("place_now"))
                and truthy(candidate.get("reachable"))
                and not truthy(candidate.get("blocked"))
                and self.depth_surface_place_distance_ready(candidate)
            )

        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        center_y_ratio = float(geometry.get("cy_ratio", candidate.get("center_y_ratio", 0.0)) or 0.0)
        front_edge_like = bool(
            truthy(candidate.get("front_edge_receptacle"))
            or self.receptacle_front_edge_candidate(candidate)
        )
        broad_front_like = bool(truthy(candidate.get("broad_front_receptacle")) or front_edge_like)

        try:
            place_now_min_bottom = float(os.getenv("ROBOT_PLACE_NOW_MIN_BOTTOM_RATIO", "0.90"))
        except (TypeError, ValueError):
            place_now_min_bottom = 0.90
        try:
            broad_front_min_bottom = float(os.getenv("ROBOT_PLACE_BROAD_FRONT_MIN_BOTTOM_RATIO", "0.90"))
        except (TypeError, ValueError):
            broad_front_min_bottom = 0.90
        try:
            min_center_y = float(os.getenv("ROBOT_PLACE_INTERACTION_MIN_CENTER_Y_RATIO", "0.54"))
        except (TypeError, ValueError):
            min_center_y = 0.54

        if front_edge_like:
            return True
        if broad_front_like:
            return bool(bottom_y_ratio >= broad_front_min_bottom and center_y_ratio >= min_center_y)
        if truthy(candidate.get("place_now")):
            return bool(bottom_y_ratio >= place_now_min_bottom and center_y_ratio >= min_center_y)
        return False

    def locked_or_best_service_candidate(self, analysis: JsonDict, *, task_class: str, role: str) -> Optional[JsonDict]:
        candidates = self.visible_service_candidates(analysis, task_class=task_class)
        for candidate in candidates:
            if self.service_lock_matches(candidate, role=role):
                return candidate
        return candidates[0] if candidates else None
#鍒ゆ柇褰撳墠杩欎釜鍊欓€夌墿浣擄紝鏄惁宸茬粡婊¤冻鈥滃彲浠ュ皾璇曟墽锟?pick-object鈥濈殑瑙嗚涓庣姸鎬佹潯浠讹拷?
#  娉ㄦ剰锛屾槸鍙互灏濊瘯鎹★紝涓嶆槸 100% 淇濊瘉涓€瀹氭崱鎴愬姛
    """
    绗竴灞傦細杩欎釜涓滆タ鏄笉鏄厑璁告崱锟?绗簩灞傦細杩欎釜涓滆タ鏄笉鏄瑙変笂鍙揪锟?绗笁灞傦細鏄笉鏄繕闇€瑕佸榻愶紵濡傛灉闇€瑕侊紝灏变笉鑳芥崱
绗洓灞傦細鏄笉鏄鍒ゅ畾鍦ㄦ锟?鏋跺瓙涓婏紵濡傛灉琚樆鏂紝灏变笉鑳芥崱
绗簲灞傦細缃俊搴﹀涓嶅锟?绗叚灞傦細浣嶇疆鏄笉鏄鍓嶆柟銆佸灞呬腑锟?绗竷灞傦細闈㈢Н銆佸簳閮ㄤ綅缃槸鍚﹁鏄庡畠宸茬粡瓒冲杩戯紵"""
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
        ground_distance = self.candidate_ground_distance(candidate)

        offcenter_max_offset = env_float("ROBOT_PICKUP_OFFCENTER_MAX_CENTER_OFFSET", 0.24)
        offcenter_max_ground_distance = env_float("ROBOT_PICKUP_OFFCENTER_MAX_GROUND_DISTANCE", 0.60)
        offcenter_alignment_ok = bool(
            position_hint in {"front-left", "front-center", "front-right"}
            and center_offset <= offcenter_max_offset
            and (
                truthy(candidate.get("pickup_now"))
                or (
                    ground_distance is not None
                    and ground_distance <= offcenter_max_ground_distance
                )
            )
        )

        if truthy(candidate.get("needs_alignment")) and not offcenter_alignment_ok:
            return False
        if truthy(candidate.get("support_context_blocked")) and not self.pickup_candidate_floor_contact_like(candidate):
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
                position_hint in {"front-left", "front-center", "front-right"}
                and center_offset <= pickup_now_max_offset
                and confidence >= pickup_min_conf
            )

        try:
            pick_ready_max_offset = float(os.getenv("ROBOT_PICK_READY_MAX_CENTER_OFFSET", "0.14"))
        except (TypeError, ValueError):
            pick_ready_max_offset = 0.14
        if center_offset > pick_ready_max_offset and not offcenter_alignment_ok:
            return False
        if position_hint != "front-center" and not offcenter_alignment_ok:
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
        self.annotate_surface_memory(candidate)
        if truthy(candidate.get("failed_recently")):
            return False
        if not truthy(candidate.get("reachable")):
            return False
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        bottom_y_ratio = float(geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0)
        center_offset = abs(self.candidate_center_offset(candidate))
        surface_source = str(candidate.get("surface_candidate_source") or "")
        if surface_source not in SURFACE_REGION_SOURCES:
            return False
        if not self.service_surface_has_interaction_point(candidate):
            return False
        if not self.service_receptacle_visual_interaction_ready(candidate):
            return False
        if surface_source in SURFACE_REGION_SOURCES:
            executor_checks = candidate.get("executor_checks") if isinstance(candidate.get("executor_checks"), dict) else {}
            return bool(
                truthy(candidate.get("visual_place_ready"))
                and truthy(candidate.get("affordance_ready"))
                and truthy(candidate.get("final_place_ready"))
                and truthy(candidate.get("place_now"))
                and truthy(executor_checks.get("precheck_ok"))
            )
        if str(candidate.get("surface_candidate_source") or "") == "depth_geometry":
            return bool(
                truthy(candidate.get("place_now"))
                and not truthy(candidate.get("blocked"))
                and self.depth_surface_place_distance_ready(candidate)
            )
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

    def service_surface_has_interaction_point(self, candidate: JsonDict) -> bool:
        if str(candidate.get("surface_candidate_source") or "") not in SURFACE_REGION_SOURCES:
            return False
        point = candidate.get("interaction_point")
        if not isinstance(point, dict):
            return False
        try:
            x = float(point.get("x"))
            y = float(point.get("y"))
        except (TypeError, ValueError):
            return False
        return math.isfinite(x) and math.isfinite(y)

    def service_place_precheck_ready(self, candidate: JsonDict) -> bool:
        if str(candidate.get("surface_candidate_source") or "") not in SURFACE_REGION_SOURCES:
            return False
        self.annotate_surface_memory(candidate)
        if truthy(candidate.get("failed_recently")):
            return False
        if not truthy(candidate.get("affordance_ready")):
            return False
        if not self.service_surface_has_interaction_point(candidate):
            return False
        if truthy(candidate.get("blocked")):
            return False
        return bool(truthy(candidate.get("visual_place_ready")) and truthy(candidate.get("reachable")))

    def max_receptacle_align_attempts(self) -> int:
        try:
            return max(1, int(os.getenv("ROBOT_MAX_RECEPTACLE_ALIGN_ATTEMPTS", "4")))
        except (TypeError, ValueError):
            return 4

    def service_place_probe_ready(self, candidate: JsonDict) -> bool:
        """Allow one backend-grounded place probe after visual alignment stalls."""
        if self.config.interaction_grounding == "metadata-hidden":
            return False
        if str(candidate.get("task_semantic_class") or "") != "place_receptacle":
            return False
        if str(candidate.get("surface_candidate_source") or "") in SURFACE_REGION_SOURCES:
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
            or self.receptacle_front_edge_candidate(candidate)
        )
        if not self.service_receptacle_visual_interaction_ready(candidate):
            return False
        if (
            str(candidate.get("surface_candidate_source") or "") == "depth_geometry"
            and not self.depth_surface_place_distance_ready(candidate)
        ):
            return False
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

    def performance_profile(self) -> str:
        value = str(self.config.performance_profile or "balanced").strip().lower()
        return value if value in {"balanced", "full"} else "balanced"

    def full_performance_profile(self) -> bool:
        return self.performance_profile() == "full"

    def interval_due(self, interval: int) -> bool:
        interval = max(1, int(interval or 1))
        return ((self.current_step_count() + 1) % interval) == 0

    def memory_update_priority_reason(self, analysis: JsonDict) -> Optional[str]:
        if self.full_performance_profile():
            return "full_profile"
        if self.config.dry_run:
            return "dry_run_debug"
        if self.consecutive_action_failures > 0:
            return "recent_action_failure"
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return None
        if self.holding_object:
            return "holding_object"
        phase = self.service_phase()
        if phase != SERVICE_INITIAL_PHASE:
            return f"active_service_phase:{phase}"
        if has_service_target(analysis):
            return "visible_service_target"
        if bool(analysis.get("floor_trash_detected", False)):
            return "visible_clean_target"
        return None

    def object_memory_update_reason(self, analysis: JsonDict) -> Optional[str]:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return None
        priority_reason = self.memory_update_priority_reason(analysis)
        if priority_reason:
            return priority_reason
        if self.interval_due(self.config.object_memory_update_interval):
            return f"interval:{max(1, int(self.config.object_memory_update_interval or 1))}"
        return None

    def semantic_mapping_update_reason(
        self,
        analysis: JsonDict,
        *,
        object_memory_reason: Optional[str],
    ) -> Optional[str]:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return None
        priority_reason = self.memory_update_priority_reason(analysis)
        if priority_reason:
            return priority_reason
        if object_memory_reason:
            return f"object_memory:{object_memory_reason}"
        if self.interval_due(self.config.semantic_map_update_interval):
            return f"interval:{max(1, int(self.config.semantic_map_update_interval or 1))}"
        return None

    def emit_update_skip(self, event: str, interval: int) -> None:
        if not self.config.verbose:
            return
        self.emit(
            event,
            {
                "reason": "balanced_profile_interval_not_due",
                "step": self.current_step_count() + 1,
                "interval": max(1, int(interval or 1)),
                "performance_profile": self.performance_profile(),
            },
        )

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
                        "performance_profile": self.performance_profile(),
                        "object_memory_update_interval": self.config.object_memory_update_interval,
                        "semantic_map_update_interval": self.config.semantic_map_update_interval,
                        "log_detail": self.config.log_detail,
                    },
                )
                if not self.config.no_reset:
                    nav_status = self.navigation.reset(room_name=str(state.room.get("room_name") or self.config.room))
                    self.emit(
                        "navigation_memory_reset",
                        {
                            "visited_cell_count": nav_status.get("visited_cell_count"),
                            "coverage_estimate": nav_status.get("coverage_estimate"),
                            "position_map_status": nav_status.get("position_map_status"),
                            "semantic_map_status": nav_status.get("semantic_map_status"),
                            "semantic_map_summary": nav_status.get("semantic_map_summary", {}),
                            "global_planner": (nav_status.get("last_global_plan") or {}).get("planner"),
                        },
                    )
                    try:
                        local_costmap_status = self.local_costmap.reset()
                        self.emit(
                            "local_costmap_reset",
                            {
                                "status": local_costmap_status.get("status"),
                                "result_type": local_costmap_status.get("result_type"),
                            },
                        )
                    except Exception as exc:
                        self.emit("local_costmap_error", {"phase": "reset", "message": str(exc)})
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
            """ 鏍稿績 4锛氬惊鐜仠姝㈢殑鎵€鏈夋潯浠讹紙浠诲姟缁堟瑙勫垯锟?                锟?涓ゅ眰鍋滄鏉′欢锛屽叏閮ㄥ湪浠ｇ爜閲屽啓姝伙細
                    绗竴灞傦細鐘舵€佺鐞嗗櫒鍒ゆ柇锛坰hould_continue()锛屽湪鐘舵€佺鐞嗙▼搴忎腑锟?鍙婊¤冻浠绘剰涓€鏉★紝涓诲惊鐜洿鎺ュ仠姝細
                            浠诲姟琚墜鍔ㄥ叧闂紙mission disabled锟?                            宸￠€诲姛鑳藉叧闂紙patrol disabled锟?                            鎴块棿宸叉竻鎵畬鎴愶紙room_complete=true锟?                            姝ユ暟杈惧埌涓婇檺 200 姝ワ紙max_steps reached锟?                            杩涘叆缁堟妯″紡锛歊OOM_COMPLETE / MISSION_REPORT / DONE
                    绗簩灞傦細鍗曟鎵ц鍒ゆ柇锛坈heck_completion_after_step()锟?婊¤冻浠绘剰涓€鏉★紝鏍囪浠诲姟瀹屾垚 / 澶辫触锛屽仠姝㈠惊鐜細
                            杩炵画鍔ㄤ綔澶辫触 锟? 锟?锟?鍒ゅ畾鍗℃锛屽仠锟?                            瑕嗙洊锟?锟?5% + 鏃犳湭鎺㈢储鍖哄煙 锟?娓呮壂瀹屾垚锛屽仠锟?                            姝ユ暟杈惧埌涓婇檺 锟?寮哄埗鍋滄
                            闀挎椂闂存棤鏂扮洰锟?+ 閲嶅鐢婚潰锟? 锟?锟?鏃犲瀮鍦撅紝鍋滄
                            鏃嬭浆闇囪崱 / 杩涢€€姝诲惊锟?锟?鏁呴殰锛屽仠锟?""
            #鏈哄櫒浜烘瘡鎵ц瀹屼竴娈典换鍔★紙榛樿 3 姝ヤ负 1 娈碉級锛屽氨浼氭墽琛屼竴娆℃锟?  """
            if not self.config.dry_run:#dry_run = 璋冭瘯妯″紡锛岀▼搴忛粯璁ゅ湪鐪熷疄杩愯妯″紡
                continuation = self.manager.should_continue()
                if not continuation.get("continue", False):
                    self.emit("patrol_stopped", {"reasons": continuation.get("reasons", [])})
                    return 0
            # 杈惧埌鏈€澶ц繍琛岀墖娈垫暟 锟?寮哄埗鍋滄
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
   
        # 1. 鎷嶄竴寮犵涓€瑙嗚鍥剧墖
        # 2. 锟?YOLO 鍒嗘瀽鍥剧墖
        # 3. 鍚屾褰撳墠鏄惁鎷跨潃鐗╀綋
        # 4. 鍒ゆ柇鏄笉鏄噸澶嶈锟?    5. 鏇存柊瀵艰埅璁板繂
        # 6. 鍐崇瓥涓嬩竴姝ュ姩锟?    7. 鎵ц鍔ㄤ綔
        # 8. 楠岃瘉鍔ㄤ綔鏄惁鎴愬姛
        # 9. 鏇存柊鏈嶅姟浠诲姟鐘讹拷?    10. 鎶婅繖涓€姝ュ啓鍏ョ姸鎬佹枃锟?    11. 鍒ゆ柇鏄惁璇ュ仠锟?    
    
    def run_one_step(self) -> StepOutcome:
        vision_result = self.call_get_vision()
        if not self.skill_success(vision_result, required_field="image_path"):
            return self.handle_perception_failure("vision_failed", vision_result.data)

        image_path = str(vision_result.data.get("image_path", ""))
        depth_path = str(vision_result.data.get("depth_path", "") or "")
        camera = vision_result.data.get("camera") if isinstance(vision_result.data.get("camera"), dict) else {}
        analysis_result = self.call_analyze(image_path, depth_path=depth_path, camera=camera)
        if not self.analysis_success(analysis_result):
            retry_result = self.call_analyze(image_path, depth_path=depth_path, camera=camera)
            if not self.analysis_success(retry_result):
                return self.handle_perception_failure("analysis_failed", retry_result.data)
            analysis_result = retry_result

        vision = vision_result.data
        analysis = analysis_result.data
        self.current_analysis = analysis
        self.sync_inventory_state()
        analysis["holding_object"] = bool(self.holding_object)
        analysis["held_object_labels"] = self.held_object_label_tokens()
        self.observe_local_costmap(vision, analysis)
        repeated_view = self.update_repeated_view(vision, analysis)
        self.update_segment_stats(analysis)
        self.observe_navigation(vision, analysis)#鎶婂綋鍓嶈瑙夊垎鏋愮粨鏋滀氦缁欏鑸蹇嗘ā锟?鏇存柊瀵艰埅璁板繂
        object_memory_reason = self.object_memory_update_reason(analysis)
        if object_memory_reason:
            analysis["object_memory_update_reason"] = object_memory_reason
            self.observe_object_memory(vision, analysis)
        else:
            self.emit_update_skip("object_memory_update_skipped", self.config.object_memory_update_interval)

        semantic_mapping_reason = self.semantic_mapping_update_reason(
            analysis,
            object_memory_reason=object_memory_reason,
        )
        if semantic_mapping_reason:
            analysis["semantic_mapping_update_reason"] = semantic_mapping_reason
            self.observe_semantic_mapping(analysis)
        else:
            self.emit_update_skip("semantic_map_update_skipped", self.config.semantic_map_update_interval)
        self.pending_navigation_recommendation = None
        self.pending_object_memory_target = None

        decision = self.decide(analysis, vision)#鍐崇瓥鍑芥暟
        # Final safety gate: every movement source (A*, service alignment,
        # placement viewpoint, or legacy fallback) must pass through the same
        # local RGB-D costmap before it reaches the backend.
        if decision.kind == "move" and decision.action in MOVE_ACTIONS:
            requested_action = decision.action
            safe_action, safety_suffix = self.service_safe_move_action(
                requested_action,
                analysis,
                allow_forward_break=True,
            )
            if safe_action != requested_action:
                self.emit(
                    "movement_action_adjusted",
                    {
                        "requested_action": requested_action,
                        "executed_action": safe_action,
                        "reason": safety_suffix,
                        "local_costmap_record": self.local_costmap_action_record(requested_action, analysis),
                    },
                )
                decision.action = safe_action
                decision.reason = f"{decision.reason};final_costmap_gate:{requested_action}->{safe_action}:{safety_suffix}"
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

    def yolo_service_depth_ready(self, analysis: JsonDict, *, depth_path: str) -> bool:
        if not depth_path:
            return True
        if "surface_free_space_status" not in analysis:
            return False
        if self.holding_object and "held_object_family" not in analysis:
            return False
        notes = [str(item) for item in (analysis.get("notes") or [])]
        if any(item.startswith("depth_geometry=enabled") for item in notes):
            return True
        if str(analysis.get("depth_path") or ""):
            return True
        if int(analysis.get("surface_candidate_count", 0) or 0) > 0:
            return True
        return False

    def call_analyze(self, image_path: str, *, depth_path: str = "", camera: Optional[JsonDict] = None) -> ScriptResult:
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
                service_result = run_yolo_service(
                    image_path,
                    self.config.timeout_seconds,
                    depth_path=depth_path,
                    camera=camera if isinstance(camera, dict) else {},
                    holding_object=bool(self.holding_object),
                    held_object_labels=self.held_object_label_tokens(),
                )
                if service_result.ok and service_result.data.get("status") == "success":
                    if self.yolo_service_depth_ready(service_result.data, depth_path=depth_path):
                        self.emit_script_result("analyze_scene_yolo_service", service_result)
                        return service_result
                    self.emit(
                        "yolo_service_depth_stale",
                        {
                            "reason": "depth_path_present_but_service_returned_rgb_only_analysis",
                            "depth_path": depth_path,
                            "service_notes": service_result.data.get("notes"),
                        },
                    )
                if service_mode in {"required", "require", "only"}:
                    self.emit_script_result("analyze_scene_yolo_service", service_result)
                    return service_result
        args = ["--image", image_path]
        if backend == "yolo" and depth_path:
            args.extend(["--depth", depth_path])
        if backend == "yolo" and isinstance(camera, dict) and camera:
            args.extend(["--camera-json", json.dumps(camera, ensure_ascii=False, separators=(",", ":"))])
        if backend == "yolo" and self.holding_object:
            args.append("--holding-object")
            held_labels = self.held_object_label_tokens()
            if held_labels:
                args.extend(["--held-object-labels", ",".join(held_labels)])
                args.extend(["--held-object-family", held_object_family_for_labels(held_labels)])
        result = run_script(script, args, self.config.timeout_seconds)
        self.emit_script_result(f"analyze_scene_{backend}", result)
        return result

    def call_move(self, action: str) -> ScriptResult:
        result = run_script(
            MOVE_SCRIPT,
            ["--action", action, "--memory-mode", "external"],
            self.config.timeout_seconds,
        )
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
            "id",
            "surface_candidate_id",
            "region_type",
            "region_bbox",
            "region_area_px",
            "region_area_ratio",
            "affordance",
            "parent_object",
            "parent_label",
            "source",
            "surface_candidate_source",
            "context_only",
            "context_reason",
            "blocked",
            "blocked_by",
            "score",
            "height",
            "distance",
            "ground_distance",
            "height_m",
            "distance_m",
            "bearing_deg",
            "reachable",
            "pickup_now",
            "place_now",
            "visual_place_ready",
            "affordance_ready",
            "final_place_ready",
            "failed_recently",
            "cooldown_remaining",
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
        if role == "place":
            occupants = candidate.get("visible_occupants")
            if isinstance(occupants, list):
                payload["visible_occupants"] = [item for item in occupants if isinstance(item, dict)][:8]
            avoidance = self.current_analysis.get("placement_avoidance_candidates")
            if isinstance(avoidance, list):
                payload["placement_avoidance_candidates"] = [
                    item for item in avoidance if isinstance(item, dict)
                ][:16]
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        if geometry:
            payload["geometry"] = dict(geometry)
        for key in (
            "interaction_point",
            "depth",
            "center_3d",
            "parent_bbox",
            "geometry_checks",
            "occupancy_checks",
            "memory_checks",
            "executor_checks",
            "free_space_completion",
            "placement_safety_contract",
        ):
            value = candidate.get(key)
            if isinstance(value, dict):
                payload[key] = dict(value)
        placement_points = candidate.get("placement_points")
        if role == "place" and isinstance(placement_points, list):
            payload["placement_points"] = [
                dict(item)
                for item in placement_points
                if isinstance(item, dict)
            ][:8]
            payload["placement_point_count"] = len(payload["placement_points"])
        for key in ("rejection_reasons", "blocked_by"):
            value = candidate.get(key)
            if isinstance(value, list):
                payload[key] = list(value)
        place_front_edge = bool(role == "place" and self.receptacle_front_edge_candidate(candidate))
        if place_front_edge:
            payload["front_edge_receptacle"] = True
            payload["broad_front_receptacle"] = True
            payload["visual_box_ambiguous"] = False
        if role == "place" and bbox and "interaction_point" not in payload:
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

    def call_place_precheck(self, candidate: Optional[JsonDict] = None) -> ScriptResult:
        args = ["--precheck-only", *self.interaction_args_for_candidate(candidate, role="place")]
        result = run_script(
            PLACE_SCRIPT,
            args,
            self.config.timeout_seconds,
        )
        self.emit_script_result("place_precheck", result)
        return result
    """鏈哄櫒浜烘墜閲屾湁娌℃湁鐗╀綋锟?    濡傛灉宸茬粡鎷跨潃涓滆タ锛屽氨搴旇杩涘叆鎵炬斁缃偣闃舵锟?    濡傛灉鎵嬮噷鏄┖鐨勶紝灏变笉鑳界户缁斁缃樁娈碉拷?    """
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
            if self.holding_object:
                inventory_objects = data.get("inventory_objects")
                if isinstance(inventory_objects, list) and inventory_objects:
                    first = inventory_objects[0] if isinstance(inventory_objects[0], dict) else {}
                    self.set_held_object_context(
                        label=first.get("label") or first.get("objectType"),
                        raw_label=first.get("objectType") or first.get("label"),
                    )
            else:
                self.clear_held_object_context()
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

    def script_result_summary(self, name: str, data: JsonDict) -> JsonDict:
        summary: JsonDict = {
            "status": data.get("status"),
            "result_type": data.get("result_type"),
        }
        for key in (
            "message",
            "image_path",
            "depth_path",
            "observation_contract",
            "analysis_confidence",
            "pickup_target_detected",
            "place_receptacle_detected",
            "floor_trash_detected",
            "surface_place_status",
            "surface_free_space_status",
            "recommended_action",
            "obstacle_ahead",
            "open_directions",
            "lastActionSuccess",
            "state_changed",
            "action",
            "holding_object",
            "pickup_executed",
            "place_executed",
            "precheck_ok",
            "precheck_reason",
            "suggested_recovery",
            "placement_point_source",
            "placement_execution_mode",
            "placement_target_resolution_error_m",
        ):
            if key in data:
                summary[key] = data.get(key)

        for key in (
            "trash_candidates",
            "service_candidates",
            "receptacle_candidates",
            "surface_candidates",
            "surface_regions",
            "visual_ready_surface_regions",
        ):
            value = data.get(key)
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)

        for key in (
            "best_pickup_candidate",
            "best_clean_candidate",
            "best_service_candidate",
            "best_receptacle_candidate",
            "best_surface_candidate",
            "best_rejected_surface_candidate",
            "best_obstacle_candidate",
        ):
            value = data.get(key)
            if isinstance(value, dict):
                summary[key] = self.short_candidate(value)

        return summary

    def emit_script_result(self, name: str, result: ScriptResult) -> None:
        payload: JsonDict = {
            "script": name,
            "returncode": result.returncode,
            "status": result.data.get("status"),
            "result_type": result.data.get("result_type"),
        }
        if self.config.verbose:
            if self.config.log_detail == "full":
                payload["data"] = result.data
            else:
                payload["summary"] = self.script_result_summary(name, result.data)
        if result.stderr.strip() and (self.config.verbose or result.returncode != 0):
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
        #缃俊搴﹂珮锛岃蛋鏁寸悊璺嚎
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
          #銆愰渶瑕佸鍑嗐€戠殑鍨冨溇  鍗筹紙涓嶅湪姝ｅ墠鏂癸紝涓嶈兘鐩存帴鎵級
        align_candidate = self.first_candidate(
            analysis,
            require_cleanable=False,
            require_alignment=True,
        )

        if align_candidate is not None:
            if self.navigation_has_frontier():
                return self.explore_decision(
                    analysis,
                    vision,
                    reason="alignment_deferred_for_navigation_frontier",
                )
            key = candidate_key(align_candidate)
            self.alignment_attempts[key] = self.alignment_attempts.get(key, 0) + 1
            if self.alignment_attempts[key] > 2:
                self.suppressed_until_step[key] = self.current_step_count() + 4
                return self.explore_decision(
                    analysis,
                    vision,
                    reason="alignment_attempt_limit_reached",
                )
            action = "RotateLeft" if align_candidate.get("position_hint") == "front-left" else "RotateRight"
            action, reason_suffix = self.break_rotation_oscillation(action, analysis, allow_forward_break=False)
            return Decision(
                kind="move",
                action=action,
                mode="EXPLORE",
                reason=f"align_floor_target:{reason_suffix}",
                candidate=align_candidate,
            )

        return self.explore_decision(analysis, vision, reason="no_direct_cleanable_target")

    def decide_tidy(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        """杩欐槸 ALFRED-style household service policy銆傚畠涓嶆槸 frame-reactive锛岃€屾槸
          phase-driven銆備篃灏辨槸璇达紝涓€鏃﹂攣锟?pickup target锛屽氨浼氭寔缁拷杩欎釜瀛愮洰鏍囷紝鐩村埌鎷惧彇鎴愬姛锟?          澶辫触閫€鍑猴紝鎴栬€呯洰鏍囦涪澶卞姝ワ拷?        """
        phase = self.service_phase()
        if phase == SERVICE_DONE_PHASE:#濡傛灉 phase == TASK_DONE 锟?閲嶇疆涓虹户缁贰瑙嗘壘鐗╀綋
            self.set_service_phase(
                SERVICE_INITIAL_PHASE,
                reason="previous_service_subgoal_complete_continue_patrol",
            )
            phase = self.service_phase()
        if phase == "FAILED":#濡傛灉 phase == FAILED 锟?鍋滄
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

        if phase in SERVICE_PLACE_PHASES:#濡傛灉娌℃湁鎷夸笢瑗夸絾 phase 杩樺湪鏀剧疆闃舵 锟?鍥炲埌 pickup 鎼滅储
            self.set_service_phase(SERVICE_INITIAL_PHASE, reason="inventory_empty_return_to_pickup_search")

        return self.decide_tidy_pickup_phase(analysis, vision)
    """杩欎釜鍑芥暟澶勭悊鎷惧彇闃舵锟?
    娴佺▼澶ф鏄細

    1. 锟?YOLO 鍒嗘瀽缁撴灉閲屾嬁 pickup_target 鍊欙拷?    2. 濡傛灉涔嬪墠閿佸畾杩囩洰鏍囷紝灏变紭鍏堟壘杩欎釜鐩爣
    3. 濡傛灉閿佸畾鐩爣涓簡锛屽氨鎵弿鍑犳
    4. 濡傛灉涓㈠け澶箙锛屽氨閲婃斁閿佸畾锛岄噸鏂版壘
    5. 濡傛灉鎵惧埌鏂扮洰鏍囷紝锟?lock_service_candidate()
    6. 濡傛灉 service_pick_ready(candidate) 鎴愮珛 锟?鎵ц pick-object
    7. 鍚﹀垯 锟?璋冪敤 service_positioning_decision() 鍘诲锟?闈犺繎"""
    def decide_tidy_pickup_phase(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        phase = self.service_phase()
        #1瀹冧笉鏄畝鍗曞湴鎷挎墍锟?YOLO 妫€娴嬪埌锟?
        candidates = self.visible_service_candidates(analysis, task_class="pickup_target")
        #璇诲彇褰撳墠閿佸畾鐨勭洰鏍囨爣锟?    濡傛灉瑙嗛噹閲岄潰鏈変箣鍓嶉攣瀹氳繃鐨勮嫻鏋滐紝灏辩户缁皢浠栭攣瀹氫负鐩爣
        locked_label = str(self.service_state.get("target_raw_label") or self.service_state.get("target_label") or "")
       #1.1.娌℃湁鍙搷浣滃€欙拷?浣嗘槸鏈夊彲锟?pickup 鐨勭被鍒紝灏辫В闄や箣鍓嶇殑閿佸畾
        if not candidates and truthy(analysis.get("pickup_target_detected")):
            # Visible RGB-D target beats coarse object-memory navigation.  Keep
            # the camera on the object and perform local align/approach first;
            # use A* only after the target is genuinely lost or locally
            # unverifiable.
            local_pursuit = self.visible_pickup_local_pursuit_candidates(analysis)
            if local_pursuit:
                candidate = local_pursuit[0]
                if not self.service_lock_matches(candidate, role="pickup"):
                    self.lock_service_candidate(candidate, role="pickup", reason="visible_pickup_local_pursuit_locked")
                self.set_service_phase("ALIGN_PICKUP_TARGET", reason="visible_pickup_local_pursuit", candidate=candidate)
                self.emit(
                    "visible_pickup_local_pursuit_selected",
                    {
                        "candidate": self.short_candidate(candidate),
                        "reason": candidate.get("approach_verify_reason"),
                        "floor_contact_like": self.pickup_candidate_floor_contact_like(candidate),
                    },
                )
                return self.service_positioning_decision(
                    analysis,
                    vision,
                    candidate,
                    base_reason="visible_pickup_local_pursuit",
                )
            if locked_label:
                self.clear_service_lock(role="pickup", reason="locked_pickup_target_not_actionable")
            clean_decision = self.tidy_clean_fallback_decision(analysis)
            if clean_decision is not None:
                return clean_decision
            memory_decision = self.select_pickup_object_memory_target(
                analysis,
                vision,
                reason="visible_pickup_not_actionable_use_object_memory",
            )
            if memory_decision is not None:
                return memory_decision
            return self.explore_decision(#杩斿洖鎺㈢储鍐崇瓥
                analysis,
                vision,
                reason="service_pickup_targets_visible_but_not_actionable",
            )
        candidate = next((item for item in candidates if self.service_lock_matches(item, role="pickup")), None)
        #1.3  涔嬪墠閿佸畾浜嗙洰鏍囷紝浣嗗綋鍓嶆病鎵惧埌锟? 鏈夊彲鎿嶄綔鍊欙拷?
        if locked_label and candidate is None:
            if candidates:#褰撳墠鏈夊叾锟?pickup candidates锛岄偅灏遍噸鏂伴攣瀹氭柊鐩爣
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
                    memory_decision = self.select_pickup_object_memory_target(
                        analysis,
                        vision,
                        reason=f"locked_pickup_released_use_object_memory:{locked_label}",
                    )
                    if memory_decision is not None:
                        return memory_decision
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
            memory_decision = self.select_pickup_object_memory_target(
                analysis,
                vision,
                reason="search_pickup_target_from_object_memory",
            )
            if memory_decision is not None:
                return memory_decision
            return self.explore_decision(analysis, vision, reason="service_phase_search_pickup_target")
#濡傛灉 candidate 宸茬粡鍙互鎹★紝杩涘叆 PICK_OBJECT
#鍒ゆ柇褰撳墠杩欎釜鍊欓€夌墿浣擄紝鏄惁宸茬粡婊¤冻鈥滃彲浠ュ皾璇曟墽锟?pick-object鈥濈殑瑙嗚涓庣姸鎬佹潯浠讹拷?
#  娉ㄦ剰锛屾槸鍙互灏濊瘯鎹★紝涓嶆槸 100% 淇濊瘉涓€瀹氭崱鎴愬姛
        if self.service_pick_ready(candidate):   
            self.set_service_phase("PICK_OBJECT", reason="pickup_candidate_action_ready", candidate=candidate)
            return Decision(
                kind="pick",
                action="pick-object",
                mode="SERVICE",
                reason="alfred_subgoal_pick_object",
                candidate=candidate,
            )
#candidate 杩樹笉鑳芥崱锛屽氨杩涘叆瀵归綈/闈犺繎閫昏緫锛堜笉鑳界洿鎺ユ崱锛屽氨鍏堢Щ锟?杞悜锛岃鐩爣鍙樺緱鍙崱銆傦級
        approach_verify = bool(candidate.get("approach_verify_pickup"))
        phase_reason = "pickup_visible_approach_to_verify" if approach_verify else "pickup_candidate_needs_positioning"
        base_reason = "approach_to_verify_pickup" if approach_verify else "alfred_align_pickup_target"
        self.set_service_phase("ALIGN_PICKUP_TARGET", reason=phase_reason, candidate=candidate)
        if approach_verify:
            self.emit(
                "pickup_approach_verify_selected",
                {
                    "candidate": self.short_candidate(candidate),
                    "surface_hint": candidate.get("surface_hint"),
                    "position_hint": candidate.get("position_hint"),
                    "ground_distance": self.candidate_ground_distance(candidate),
                    "reason": candidate.get("approach_verify_reason"),
                },
            )
        #涓嬮潰杩欎釜鍑芥暟浼氭牴锟?candidate 鐨勭姸鎬佸喅瀹氾細濡傛灉鐩爣鍋忓乏 锟?RotateLeft濡傛灉鐩爣鍋忓彸 锟?RotateRight濡傛灉鐩爣鍦ㄦ鍓嶆柟浣嗚繕锟?锟?MoveAhead濡傛灉鍓嶆柟鏈夐殰锟?锟?瀹夊叏閬胯
        return self.service_positioning_decision(
            analysis,
            vision,
            candidate,
            base_reason=base_reason,
        )
    """杩欎釜鍑芥暟澶勭悊鏀剧疆闃舵锛岄€昏緫鏇村鏉傦紝鍥犱负鏀剧疆鏇村鏄撳け璐ワ拷?
瀹冧細锟?
1. 锟?YOLO 鍒嗘瀽缁撴灉閲屾嬁 place_receptacle 鍊欙拷?2. 浼樺厛鎵句箣鍓嶉攣瀹氱殑 receptacle
3. 濡傛灉閿佸畾锟?receptacle 涓嶈浜嗭紝灏辨壂鎻忓嚑锟?4. 濡傛灉鐪嬭 context candidate锛屼篃鍙互鎷挎潵杈呭姪瀵归綈
5. 濡傛灉瀹屽叏鎵句笉鍒帮紝灏辨墽锟?holding_receptacle_search_decision()锛氳繘鍏ユ嬁鐫€涓滆タ锟?receptacle 鐨勬悳绱㈢瓥鐣ワ紝閫氬父浼氳繑鍥炲乏鍙宠浆銆佸墠锟?6. 濡傛灉 receptacle 宸茬粡 action ready 锟?place-object
            褰撳墠鏀剧疆鐩爣宸茬粡婊¤冻鏀剧疆鏉′欢锟?                鍦ㄦ鍓嶆柟
                鍙揪
                涓嶉渶瑕佸锟?                涓嶉渶瑕侀潬锟?                瑙嗚涓婅冻澶熷彲浜や簰
                缃俊锟?闈㈢Н/浣嶇疆锟?            浜庢槸杩涘叆 PLACE_OBJECT 闃舵锟?7. 濡傛灉鏈夊€欓€夋斁缃洰鏍囷紝浣嗚繕涓嶈兘 place 锟?锟?approach 锟?align 
8. 濡傛灉瀵归綈澶箙杩樹笉琛岋紝鍙兘 backend probe 鎴栨崲鐩爣"""
    def decide_tidy_place_phase(self, analysis: JsonDict, vision: JsonDict) -> Decision:
        phase = self.service_phase()
        candidates = self.visible_service_candidates(analysis, task_class="place_receptacle")
        context_candidates = self.visible_receptacle_context_candidates(analysis)
        locked_label = str(self.service_state.get("receptacle_raw_label") or self.service_state.get("receptacle_label") or "")
        candidate = next((item for item in candidates if self.service_lock_matches(item, role="place")), None)
        context_candidate = next(
            (item for item in context_candidates if self.service_lock_matches(item, role="place")),
            None,
        )
        ###濡傛灉娌℃湁鎵惧埌鍜岄攣瀹氱洰鏍囧尮閰嶇殑 context candidate(灏辨槸濡傛灉涔嬪墠閿佸畾鐨勭洰鏍囧鏋滀笉鍦ㄥ€欓€夌墿锟?锛屼絾鐢婚潰閲屾湁鍏朵粬鍙綔涓轰笂涓嬫枃鐨勬斁缃洰鏍囷紝閭ｅ氨鍏堟嬁鎺掑簭鏈€闈犲墠鐨勪竴锟?        #candidate锛氭瘮杈冮潬璋憋紝鍙兘鍙互閿佸畾銆侀潬杩戙€佹渶锟?place-object  
        # context_candidate锛氬彧鏄€滄垜濂藉儚鐪嬪埌妗屽瓙/鍙伴潰鍦ㄩ偅杈光€濓紝涓昏鐢ㄦ潵瀵艰埅/瀵归綈
        if context_candidate is None and context_candidates:
            context_candidate = context_candidates[0]
        surface_place_status = str(analysis.get("surface_place_status") or "")
        reported_best_surface = analysis.get("best_surface_candidate")
        reported_best_surface_ready = bool(
            isinstance(reported_best_surface, dict)
            and truthy(reported_best_surface.get("visual_place_ready"))
        )
        has_reported_surface_regions = bool(
            analysis.get("surface_region_count")
            or analysis.get("surface_regions")
            or analysis.get("surface_candidates")
            or analysis.get("best_rejected_surface_candidate")
        )
        no_visual_ready_surface = bool(
            surface_place_status == "no_visual_ready_surface"
            or (
                self.holding_object
                and not candidates
                and has_reported_surface_regions
                and not reported_best_surface_ready
            )
        )
        if no_visual_ready_surface:
            analysis["surface_place_status"] = "no_visual_ready_surface"
        if self.holding_object and no_visual_ready_surface and not candidates:
            self.clear_service_lock(role="place", reason="no_visual_ready_surface")
            self.set_service_phase(
                "SEARCH_RECEPTACLE",
                reason="surface_search_no_visual_ready_surface",
                candidate=None,
            )
            viewpoint_decision = self.select_receptacle_placement_viewpoint_target(
                analysis,
                vision,
                reason="no_visual_ready_surface_use_placement_viewpoint_planner",
                context_candidate=context_candidate,
            )
            if viewpoint_decision is not None:
                return viewpoint_decision
            rejected_candidate = analysis.get("best_rejected_surface_candidate")
            return self.no_ready_surface_search_decision(
                analysis,
                vision,
                reason="no_visual_ready_surface",
                candidate=rejected_candidate if isinstance(rejected_candidate, dict) else context_candidate,
            )
        if not no_visual_ready_surface:
            self.service_state["no_ready_surface_steps"] = 0
            self.service_state["surface_place_status"] = surface_place_status or None
            if reported_best_surface_ready:
                try:
                    planner_track_id = str(
                        self.service_state.get("receptacle_track_id")
                        or (self.pending_object_memory_target or {}).get("track_id")
                        or ""
                    ) or None
                    self.placement_viewpoints.mark_surface_ready(
                        track_id=planner_track_id,
                        step=self.current_step_count(),
                        surface_candidate_id=(reported_best_surface or {}).get("surface_candidate_id")
                        if isinstance(reported_best_surface, dict)
                        else None,
                    )
                except Exception as exc:
                    self.emit("placement_viewpoint_error", {"phase": "surface_ready", "message": str(exc)})
        if locked_label and candidate is None:
            if candidates:
                self.clear_service_lock(role="place", reason="locked_receptacle_lost_retarget_visible")
                candidate = candidates[0]
            else:
                if context_candidate is not None:
                    self.remember_receptacle_position_hint(context_candidate)
                    self.set_service_phase(
                        "ALIGN_RECEPTACLE",
                        reason="locked_receptacle_visible_as_context",
                        candidate=context_candidate,
                    )
                    return self.service_positioning_decision(#杩欎釜鍑芥暟璐熻矗鈥滆繕涓嶈兘 pick/place 鏃惰鎬庝箞璋冩暣浣嶇疆鈥濓拷?                        analysis,
                        vision,
                        context_candidate,
                        base_reason="alfred_align_receptacle_context",
                    )
                lost_steps = self.locked_lost_steps(role="place")
                scan_count = self.increment_locked_scan_count(role="place")
                max_scan_steps = self.max_locked_scan_steps(role="place")
                if lost_steps > max_scan_steps or scan_count > max_scan_steps:
                    self.clear_service_lock(role="place", reason="locked_receptacle_lost")
                    memory_decision = self.select_receptacle_object_memory_target(
                        analysis,
                        vision,
                        reason=f"locked_receptacle_released_use_object_memory:{locked_label}",
                    )
                    if memory_decision is not None:
                        return memory_decision
                    return self.holding_receptacle_search_decision(
                        analysis,
                        vision,
                        reason=(
                            f"locked_receptacle_released:{locked_label};"
                            f"lost_steps={lost_steps};scan={scan_count}"
                        ),
                    )
                preferred = "RotateLeft"
                last_position_hint = str(self.service_state.get("receptacle_last_position_hint") or "")
                if last_position_hint == "front-right":
                    preferred = "RotateRight"
                action, reason_suffix = self.service_safe_move_action(
                    preferred,
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
        if candidate is not None:
            self.remember_receptacle_position_hint(candidate)

        if candidate is None:
            if context_candidate is not None:
                self.remember_receptacle_position_hint(context_candidate)
                self.set_service_phase(
                    "ALIGN_RECEPTACLE",
                    reason="receptacle_context_visible",
                    candidate=context_candidate,
                )
                return self.service_positioning_decision(
                    analysis,
                    vision,
                    context_candidate,
                    base_reason="alfred_align_receptacle_context",
                )
            self.clear_service_lock(role="place", reason="receptacle_not_visible")
            memory_decision = self.select_receptacle_object_memory_target(
                analysis,
                vision,
                reason="search_receptacle_from_object_memory",
            )
            if memory_decision is not None:
                return memory_decision
            return self.holding_receptacle_search_decision(
                analysis,
                vision,
                reason="service_phase_search_receptacle",
            )

        if phase == "APPROACH_RECEPTACLE":
            hinted = self.receptacle_action_hint_decision(
                analysis,
                vision,
                candidate,
                base_reason="alfred_interactable_pose_guidance",
            )
            if hinted is not None:
                return hinted
            try:
                approach_attempts = int(self.service_state.get("phase_attempts", 0) or 0)
            except (TypeError, ValueError):
                approach_attempts = 0
            if approach_attempts <= 0 or not self.service_place_ready(candidate):
                return self.service_positioning_decision(
                    analysis,
                    vision,
                    candidate,
                    base_reason="alfred_approach_receptacle_without_hint",
                )

        if self.service_place_precheck_ready(candidate):
            precheck = self.call_place_precheck(candidate)
            precheck_data = precheck.data if isinstance(precheck.data, dict) else {}
            precheck_ok = bool(precheck.ok and precheck_data.get("precheck_ok"))
            executor_checks = candidate.get("executor_checks") if isinstance(candidate.get("executor_checks"), dict) else {}
            executor_checks = dict(executor_checks)
            executor_checks.update(
                {
                    "precheck_supported": True,
                    "precheck_ok": bool(precheck_ok),
                    "reason": precheck_data.get("precheck_reason") or precheck_data.get("result_type"),
                    "suggested_recovery": precheck_data.get("suggested_recovery"),
                    "failure_stage": precheck_data.get("precheck_failure_stage"),
                    "placement_point_source": precheck_data.get("placement_point_source"),
                    "placement_clearance_contract_applied": precheck_data.get("placement_clearance_contract_applied"),
                    "placement_execution_mode": precheck_data.get("placement_execution_mode"),
                    "placement_target_required": precheck_data.get("placement_target_required"),
                    "placement_target_resolution_error_m": precheck_data.get("placement_target_resolution_error_m"),
                    "placement_target_resolution_tolerance_m": precheck_data.get("placement_target_resolution_tolerance_m"),
                }
            )
            candidate["executor_checks"] = executor_checks
            candidate["final_place_ready"] = bool(precheck_ok)
            candidate["place_now"] = bool(precheck_ok)
            self.emit(
                "place_precheck_result",
                {
                    "candidate": self.short_candidate(candidate),
                    "precheck_ok": bool(precheck_ok),
                    "result_type": precheck_data.get("result_type"),
                    "suggested_recovery": precheck_data.get("suggested_recovery"),
                    "failed_candidate_id": precheck_data.get("failed_candidate_id"),
                    "precheck_failure_stage": precheck_data.get("precheck_failure_stage"),
                    "placement_point_source": precheck_data.get("placement_point_source"),
                    "placement_clearance_contract_applied": precheck_data.get("placement_clearance_contract_applied"),
                    "placement_execution_mode": precheck_data.get("placement_execution_mode"),
                    "placement_target_required": precheck_data.get("placement_target_required"),
                    "placement_target_resolution_error_m": precheck_data.get("placement_target_resolution_error_m"),
                    "placement_target_resolution_tolerance_m": precheck_data.get("placement_target_resolution_tolerance_m"),
                },
            )
            if precheck_ok and self.service_place_ready(candidate):
                self.set_service_phase("PLACE_OBJECT", reason="surface_executor_precheck_ok", candidate=candidate)
                return Decision(
                    kind="place",
                    action="place-object",
                    mode="SERVICE",
                    reason="alfred_subgoal_place_object;surface_precheck_ok",
                    candidate=candidate,
                )
            result_type = str(precheck_data.get("result_type") or "place_precheck_failed")
            suggested_action = str(
                precheck_data.get("executor_action_hint")
                or precheck_data.get("suggested_recovery")
                or ""
            ).strip()
            if suggested_action in MOVE_ACTIONS:
                self.store_receptacle_action_hint(suggested_action, steps=2)
                self.set_service_phase(
                    "APPROACH_RECEPTACLE",
                    reason=f"surface_precheck_requires_interactable_pose:{result_type}",
                    candidate=candidate,
                )
                hinted = self.receptacle_action_hint_decision(
                    analysis,
                    vision,
                    candidate,
                    base_reason="surface_precheck_interactable_pose_guidance",
                )
                if hinted is not None:
                    return hinted
            if result_type == "error_place_pose_not_interactable":
                self.clear_service_lock(role="place", reason=f"place_precheck_failed:{result_type}")
                return self.holding_receptacle_search_decision(
                    analysis,
                    vision,
                    reason=f"place_precheck_failed:{result_type}",
                    candidate=candidate,
                )
            self.mark_surface_candidate_failed(
                candidate,
                result_type=result_type,
                failed_candidate_id=str(precheck_data.get("failed_candidate_id") or "") or None,
            )
            self.clear_service_lock(role="place", reason=f"place_precheck_failed:{result_type}")
            return self.holding_receptacle_search_decision(
                analysis,
                vision,
                reason=f"place_precheck_failed:{result_type}",
                candidate=candidate,
            )

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
            return self.holding_receptacle_search_decision(
                analysis,
                vision,
                reason=f"receptacle_alignment_timeout_retarget:{align_attempts}",
                candidate=candidate,
            )

        if (
            str(candidate.get("position_hint") or "") == "front-center"
            and not truthy(candidate.get("needs_alignment"))
            and not self.service_receptacle_visual_interaction_ready(candidate)
            and self.can_safely_move_ahead(analysis)
        ):
            self.set_service_phase("APPROACH_RECEPTACLE", reason="receptacle_needs_approach", candidate=candidate)
        else:
            self.set_service_phase("ALIGN_RECEPTACLE", reason="receptacle_needs_positioning", candidate=candidate)
        return self.service_positioning_decision(
            analysis,
            vision,
            candidate,
            base_reason="alfred_align_receptacle",
        )

    def receptacle_action_hint_decision(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        candidate: JsonDict,
        *,
        base_reason: str,
    ) -> Optional[Decision]:
        action = self.current_receptacle_action_hint()
        if action is None:
            return None
        if action in MOVE_ACTIONS:
            safe_action, reason_suffix = self.service_safe_move_action(
                action,
                analysis,
                allow_forward_break=False,
            )
            if self.can_safely_move_action(safe_action, analysis):
                return Decision(
                    kind="move",
                    action=safe_action,
                    mode="SERVICE",
                    reason=f"{base_reason};receptacle_action_hint:{reason_suffix}",
                    candidate=candidate,
                )
            self.service_state["receptacle_action_hint"] = None
            self.service_state["receptacle_action_hint_until_step"] = None
            return self.holding_receptacle_search_decision(
                analysis,
                vision,
                reason=f"{base_reason};receptacle_action_hint_not_safe:{action}",
                candidate=candidate,
            )
        return None

    def autonomous_action_allowed(self, action: str) -> bool:
        """Return whether patrol logic may autonomously issue ``action``.

        LookUp / LookDown stay supported by the low-level move skill so they can
        still be called manually.  They are intentionally excluded from normal
        autonomous planning and recovery unless explicitly opted in through an
        environment flag for future active-perception experiments.
        """
        token = str(action or "")
        if token in AUTONOMOUS_BODY_ACTIONS:
            return True
        return bool(self.allow_autonomous_camera_pitch and token in LOOK_ACTIONS)

    def first_safe_autonomous_body_action(
        self,
        analysis: JsonDict,
        preferred: Optional[Sequence[str]] = None,
    ) -> Optional[str]:
        """Choose a safe body motion without silently introducing camera pitch."""
        order: List[str] = []
        for action in list(preferred or []) + [
            "MoveBack",
            "MoveLeft",
            "MoveRight",
            "RotateLeft",
            "RotateRight",
            "MoveAhead",
        ]:
            if action in order or action not in AUTONOMOUS_BODY_ACTIONS:
                continue
            order.append(action)
        for action in order:
            if self.can_safely_move_action(action, analysis):
                return action
        return None

    def service_safe_move_action(
        self,
        preferred: str,
        analysis: JsonDict,
        *,
        allow_forward_break: bool = False,
    ) -> Tuple[str, str]:
        """Apply the final movement safety gate before an autonomous action.

        Local RGB-D swept-volume checks are authoritative when they have
        evidence. Camera pitch remains available as a manual low-level command,
        but ordinary patrol, A* recovery, and local-costmap fallback do not pick
        LookUp / LookDown by default.
        """
        action = str(preferred or "")
        if action not in MOVE_ACTIONS:
            return action, "non_navigation_action"

        if action in LOOK_ACTIONS and not self.allow_autonomous_camera_pitch:
            fallback = self.first_safe_autonomous_body_action(
                analysis,
                ["MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight"],
            )
            return (
                fallback or "RotateLeft",
                "autonomous_camera_pitch_disabled_body_fallback",
            )

        reason_suffix = "normal"
        if action in ROTATE_ACTIONS:
            action, reason_suffix = self.break_rotation_oscillation(
                action,
                analysis,
                allow_forward_break=allow_forward_break,
            )

        if self.can_safely_move_action(action, analysis):
            return action, reason_suffix

        alternative = self.held_object_alternative_action(action, analysis)
        if alternative:
            if alternative in ROTATE_ACTIONS:
                alternative, alt_suffix = self.break_rotation_oscillation(
                    alternative,
                    analysis,
                    allow_forward_break=False,
                )
                if not self.can_safely_move_action(alternative, analysis):
                    second = self.held_object_alternative_action(alternative, analysis)
                    if second:
                        alternative = second
                        alt_suffix = "second_safe_alternative"
                return alternative, f"{reason_suffix};costmap_avoid_{action}:{alt_suffix}"
            return alternative, f"{reason_suffix};costmap_avoid_{action}"

        if self.allow_autonomous_camera_pitch:
            for look_action in ("LookDown", "LookUp"):
                if self.can_safely_move_action(look_action, analysis):
                    return look_action, f"{reason_suffix};explicit_active_perception_camera_scan"

        # No camera-pitch escape in ordinary autonomous runs. Prefer a safe
        # body-only fallback. If the local map vetoes every body action, rotate
        # in place as the least invasive deterministic scan instead of changing
        # camera horizon and contaminating downstream RGB-D state.
        fallback = self.first_safe_autonomous_body_action(analysis)
        return (
            fallback or "RotateLeft",
            f"{reason_suffix};no_safe_body_motion_rotation_scan",
        )

    def held_object_alternative_action(self, blocked_action: str, analysis: JsonDict) -> Optional[str]:
        """Choose a locally safe body-motion alternative for a blocked action.

        The historical name is preserved for compatibility. Camera-pitch
        actions are deliberately excluded from ordinary navigation fallback.
        """
        last_hint = str(self.service_state.get("receptacle_last_position_hint") or "")
        if blocked_action == "RotateLeft":
            candidates = ["MoveLeft", "RotateRight", "MoveBack", "MoveRight"]
        elif blocked_action == "RotateRight":
            candidates = ["MoveRight", "RotateLeft", "MoveBack", "MoveLeft"]
        elif blocked_action == "MoveAhead":
            lateral = ["MoveRight", "MoveLeft"] if last_hint == "front-right" else ["MoveLeft", "MoveRight"]
            turns = ["RotateRight", "RotateLeft"] if last_hint == "front-right" else ["RotateLeft", "RotateRight"]
            candidates = lateral + turns + ["MoveBack"]
        elif blocked_action == "MoveBack":
            candidates = ["MoveLeft", "MoveRight", "RotateRight", "RotateLeft"]
        elif blocked_action == "MoveLeft":
            candidates = ["RotateLeft", "MoveBack", "MoveRight", "RotateRight"]
        elif blocked_action == "MoveRight":
            candidates = ["RotateRight", "MoveBack", "MoveLeft", "RotateLeft"]
        elif blocked_action in LOOK_ACTIONS:
            candidates = ["MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight"]
        else:
            candidates = ["MoveLeft", "MoveRight", "RotateRight", "RotateLeft", "MoveBack"]

        for action in candidates:
            if action not in AUTONOMOUS_BODY_ACTIONS or action == blocked_action:
                continue
            if self.can_safely_move_action(action, analysis):
                return action
        return None

    def dominant_surface_rejection_reason(self, analysis: JsonDict) -> str:
        summary = analysis.get("surface_rejection_summary")
        if not isinstance(summary, dict):
            return "no_visual_ready_surface"
        items = [
            (str(key), int(value or 0))
            for key, value in summary.items()
            if int(value or 0) > 0
        ]
        if not items:
            return "no_visual_ready_surface"
        priority = {
            "too_close": 8,
            "too_far": 7,
            "blocked": 6,
            "depth_unstable": 5,
            "height_out_of_range": 4,
            "thin_region": 3,
            "single_row_region": 2,
            "too_small": 1,
        }
        items.sort(key=lambda item: (item[1], priority.get(item[0], 0)), reverse=True)
        return items[0][0]

    def no_ready_surface_search_decision(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        *,
        reason: str,
        candidate: Optional[JsonDict] = None,
    ) -> Decision:
        """Actively change the placement viewpoint using the RGB-D costmap.

        The old implementation mostly alternated rotations and MoveBack.  That
        loses useful tabletop views and can oscillate near counters.  This
        version prefers safe lateral translations when a surface is occluded
        or edge-dominated, then falls back to cautious backoff, rotation, and
        body-only scan rotations.
        """
        dominant_reason = self.dominant_surface_rejection_reason(analysis)
        if str(analysis.get("surface_free_space_status") or "") == "free_space_outside_current_reach":
            dominant_reason = "too_far"
        try:
            no_ready_steps = int(self.service_state.get("no_ready_surface_steps", 0) or 0) + 1
        except (TypeError, ValueError):
            no_ready_steps = 1
        try:
            attempts = int(self.service_state.get("surface_search_attempts", 0) or 0) + 1
        except (TypeError, ValueError):
            attempts = 1

        force_reposition = bool(no_ready_steps >= 4)
        last_surface_action = base_action(self.service_state.get("last_surface_search_action"))
        action = ""
        search_reason = dominant_reason

        def choose(preferred: Sequence[str], *, tag: str) -> Tuple[str, str]:
            for requested in preferred:
                if requested not in MOVE_ACTIONS:
                    continue
                resolved, suffix = self.service_safe_move_action(
                    requested,
                    analysis,
                    allow_forward_break=True,
                )
                if resolved in MOVE_ACTIONS and self.can_safely_move_action(resolved, analysis):
                    return resolved, f"{tag}:{requested}->{resolved};{suffix}"
            return "", f"{tag}:no_safe_action"

        if force_reposition:
            nav_recommendation = self.navigation_recommendation(vision, analysis)
            if isinstance(nav_recommendation, dict):
                nav_action = str(nav_recommendation.get("action") or "")
                nav_reason = str(nav_recommendation.get("reason") or "navigation_memory")
                action, detail = choose([nav_action], tag=f"nav:{nav_reason}")
                if action:
                    search_reason = f"reposition_after_{no_ready_steps}_no_ready_steps;{detail}"

        if not action and dominant_reason == "too_close":
            action, detail = choose(
                ["MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight"],
                tag="too_close_backoff_or_lateral",
            )
            search_reason = detail
        elif not action and dominant_reason == "too_far":
            action, detail = choose(
                ["MoveAhead", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight"],
                tag="too_far_approach",
            )
            search_reason = detail
        elif not action and dominant_reason in {"blocked", "depth_unstable", "height_out_of_range", "thin_region", "single_row_region", "touches_image_edge", "touches_parent_edge"}:
            preferred = ["MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "MoveBack"]
            if last_surface_action in {"MoveLeft", "RotateLeft"}:
                preferred = ["MoveRight", "RotateRight", "MoveBack", "MoveLeft", "RotateLeft"]
            action, detail = choose(preferred, tag=f"{dominant_reason}_view_change")
            search_reason = detail

        if not action:
            preferred = ["MoveLeft", "MoveRight", "MoveAhead", "MoveBack", "RotateLeft", "RotateRight"]
            if last_surface_action in {"MoveLeft", "RotateLeft"}:
                preferred = ["MoveRight", "RotateRight", "MoveBack", "MoveLeft", "RotateLeft"]
            elif last_surface_action in {"MoveRight", "RotateRight"}:
                preferred = ["MoveLeft", "RotateLeft", "MoveBack", "MoveRight", "RotateRight"]
            action, detail = choose(preferred, tag="surface_view_change")
            search_reason = f"{dominant_reason};{detail}"

        if not action:
            # This should be rare: keep perception alive without altering the
            # camera horizon. A deterministic body rotation is easier to reason
            # about than silently injecting LookDown / LookUp into the pipeline.
            action = self.safe_turn_action(analysis)
            search_reason = f"{search_reason};body_rotation_scan_only"

        self.service_state["surface_place_status"] = str(
            analysis.get("surface_place_status") or "no_visual_ready_surface"
        )
        self.service_state["surface_search_attempts"] = int(attempts)
        self.service_state["no_ready_surface_steps"] = int(no_ready_steps)
        self.service_state["last_surface_search_action"] = action
        self.service_state["last_surface_search_reason"] = search_reason
        self.save_service_task_state()
        analysis["surface_search_action"] = action
        analysis["surface_search_reason"] = search_reason
        analysis["no_ready_surface_steps"] = int(no_ready_steps)
        self.emit(
            "surface_search_action",
            {
                "action": action,
                "reason": search_reason,
                "dominant_rejection_reason": dominant_reason,
                "no_ready_surface_steps": no_ready_steps,
                "surface_place_status": self.service_state["surface_place_status"],
                "rejection_summary": analysis.get("surface_rejection_summary"),
                "local_costmap": {
                    "blocked_actions": (analysis.get("local_costmap") or {}).get("blocked_actions", []),
                    "front_clearance_m": (analysis.get("local_costmap") or {}).get("front_clearance_m"),
                    "left_clearance_m": (analysis.get("local_costmap") or {}).get("left_clearance_m"),
                    "right_clearance_m": (analysis.get("local_costmap") or {}).get("right_clearance_m"),
                },
            },
        )
        return Decision(
            kind="move",
            action=action,
            mode="SERVICE",
            reason=f"{reason};surface_search:{search_reason}",
            candidate=candidate,
        )

    def holding_receptacle_search_decision(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        *,
        reason: str,
        candidate: Optional[JsonDict] = None,
    ) -> Decision:
        """Search for a place target while holding an object without frontier wandering."""
        if (
            self.holding_object
            and str(analysis.get("surface_place_status") or "") == "no_visual_ready_surface"
            and not isinstance(analysis.get("best_surface_candidate"), dict)
        ):
            return self.no_ready_surface_search_decision(
                analysis,
                vision,
                reason=reason,
                candidate=candidate,
            )

        recommended = str(analysis.get("recommended_action") or "").strip()
        if recommended in MOVE_ACTIONS:
            action, reason_suffix = self.service_safe_move_action(
                recommended,
                analysis,
                allow_forward_break=False,
            )
            if action in MOVE_ACTIONS and self.can_safely_move_action(action, analysis):
                return Decision(
                    kind="move",
                    action=action,
                    mode="SERVICE",
                    reason=f"{reason};holding_yolo_guidance:{reason_suffix}",
                    candidate=candidate,
                )

        last_position_hint = str(self.service_state.get("receptacle_last_position_hint") or "")
        if last_position_hint == "front-right":
            preferred = "MoveRight" if self.can_safely_move_action("MoveRight", analysis) else "RotateRight"
        elif last_position_hint == "front-left":
            preferred = "MoveLeft" if self.can_safely_move_action("MoveLeft", analysis) else "RotateLeft"
        elif last_position_hint == "front-center" and self.can_safely_move_action("MoveAhead", analysis):
            preferred = "MoveAhead"
        else:
            preferred = self.safe_turn_action(analysis)

        action, reason_suffix = self.service_safe_move_action(
            preferred,
            analysis,
            allow_forward_break=False,
        )
        if not self.can_safely_move_action(action, analysis):
            action = self.held_object_alternative_action(action, analysis) or self.safe_turn_action(analysis)
            reason_suffix = f"{reason_suffix};holding_search_safe_fallback"

        return Decision(
            kind="move",
            action=action,
            mode="SERVICE",
            reason=f"{reason};holding_place_search:{reason_suffix}",
            candidate=candidate,
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

        if (
            task_class == "pickup_target"
            and position_hint in {"front-left", "front-right"}
            and not truthy(candidate.get("pickup_now"))
        ):
            ground_distance = self.candidate_ground_distance(candidate)
            center_offset = abs(self.candidate_center_offset(candidate))
            min_forward_distance = env_float("ROBOT_PICKUP_APPROACH_FORWARD_MIN_GROUND_DISTANCE", 0.55)
            max_forward_offset = env_float("ROBOT_PICKUP_APPROACH_FORWARD_MAX_CENTER_OFFSET", 0.24)
            if (
                ground_distance is not None
                and ground_distance >= min_forward_distance
                and center_offset <= max_forward_offset
                and self.can_safely_move_ahead(analysis)
            ):
                action, reason_suffix = self.service_safe_move_action(
                    "MoveAhead",
                    analysis,
                    allow_forward_break=False,
                )
                if action in MOVE_ACTIONS and self.can_safely_move_action(action, analysis):
                    return Decision(
                        kind="move",
                        action=action,
                        mode="SERVICE",
                        reason=(
                            f"{base_reason};approach_visible_pickup_before_alignment:"
                            f"ground_distance={ground_distance:.2f};offset={center_offset:.2f};{reason_suffix}"
                        ),
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
            action, reason_suffix = self.service_safe_move_action(action, analysis, allow_forward_break=False)
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
            action, reason_suffix = self.service_safe_move_action(action, analysis, allow_forward_break=False)
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
                action, reason_suffix = self.service_safe_move_action(action, analysis, allow_forward_break=False)
                direction = "left" if center_offset < 0 else "right"
                return Decision(
                    kind="move",
                    action=action,
                    mode="SERVICE",
                    reason=f"{base_reason};fine_align_{direction}:{reason_suffix}",
                    candidate=candidate,
                )

        if position_hint == "front-center" and self.can_safely_move_ahead(analysis):
            action, reason_suffix = self.service_safe_move_action("MoveAhead", analysis, allow_forward_break=False)
            if action != "MoveAhead" or self.can_safely_move_ahead(analysis):
                return Decision(
                    kind="move",
                    action=action,
                    mode="SERVICE",
                    reason=f"{base_reason};approach_front_center_{task_class}:{reason_suffix}",
                    candidate=candidate,
                )

        if self.holding_object and task_class == "place_receptacle":
            return self.holding_receptacle_search_decision(
                analysis,
                vision,
                reason=f"{base_reason};candidate_not_action_ready",
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

    def compact_costmap_blockers(self, blockers: Any) -> List[JsonDict]:
        if not isinstance(blockers, list):
            return []
        compact: List[JsonDict] = []
        for blocker in blockers[:3]:
            if not isinstance(blocker, dict):
                continue
            records: List[JsonDict] = []
            for record in (blocker.get("source_records") or [])[:3]:
                if not isinstance(record, dict):
                    continue
                overlaps: List[JsonDict] = []
                for overlap in (record.get("candidate_overlaps") or [])[:3]:
                    if not isinstance(overlap, dict):
                        continue
                    overlaps.append(
                        {
                            "label": overlap.get("label"),
                            "task_semantic_class": overlap.get("task_semantic_class"),
                            "source": overlap.get("source"),
                            "position_hint": overlap.get("position_hint"),
                            "confidence": overlap.get("confidence"),
                        }
                    )
                records.append(
                    {
                        "source_cell": record.get("source_cell"),
                        "pixel": record.get("pixel"),
                        "point_m": record.get("point_m"),
                        "inflation_distance_m": record.get("inflation_distance_m"),
                        "candidate_overlaps": overlaps,
                    }
                )
            compact.append(
                {
                    "blocked_cell": blocker.get("blocked_cell"),
                    "blocked_cell_m": blocker.get("blocked_cell_m"),
                    "source_records": records,
                }
            )
        return compact

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

    def observe_local_costmap(self, vision: JsonDict, analysis: JsonDict) -> None:
        """Build the RGB-D local obstacle layer before navigation decisions."""
        try:
            result = self.local_costmap.update(
                vision=vision,
                analysis=analysis,
                holding_object=bool(self.holding_object),
                held_object_labels=self.held_object_label_tokens(),
                step=self.current_step_count(),
                persist=not self.config.dry_run,
            )
            self.current_local_costmap = dict(result)
            analysis["local_costmap"] = dict(result)
            action_safety = result.get("action_safety") if isinstance(result.get("action_safety"), dict) else {}
            moveahead = action_safety.get("MoveAhead") if isinstance(action_safety.get("MoveAhead"), dict) else {}
            self.emit(
                "local_costmap_updated",
                {
                    "status": result.get("status"),
                    "result_type": result.get("result_type"),
                    "holding_object": result.get("holding_object"),
                    "inflated_robot_radius_m": result.get("inflated_robot_radius_m"),
                    "front_clearance_m": result.get("front_clearance_m"),
                    "left_clearance_m": result.get("left_clearance_m"),
                    "right_clearance_m": result.get("right_clearance_m"),
                    "blocked_actions": result.get("blocked_actions", []),
                    "occupied_cell_count": result.get("occupied_cell_count"),
                    "inflated_cell_count": result.get("inflated_cell_count"),
                    "moveahead_safe": moveahead.get("safe"),
                    "moveahead_reason": moveahead.get("reason"),
                    "moveahead_confidence": moveahead.get("confidence"),
                    "moveahead_observed_ratio": moveahead.get("observed_ratio"),
                    "moveahead_blocked_cell_count": moveahead.get("blocked_cell_count"),
                    "moveahead_blockers": self.compact_costmap_blockers(result.get("moveahead_blockers")),
                },
            )
        except Exception as exc:
            self.current_local_costmap = {}
            analysis["local_costmap"] = {}
            self.emit("local_costmap_error", {"phase": "observe", "message": str(exc)})

    def local_costmap_action_record(self, action: str, analysis: Optional[JsonDict] = None) -> Optional[JsonDict]:
        source = analysis if isinstance(analysis, dict) else {}
        costmap = source.get("local_costmap") if isinstance(source.get("local_costmap"), dict) else self.current_local_costmap
        if not isinstance(costmap, dict) or str(costmap.get("status") or "") != "success":
            return None
        rec = (costmap.get("action_safety") or {}).get(str(action))
        return dict(rec) if isinstance(rec, dict) else None

    def local_costmap_known_unsafe(self, action: str, analysis: Optional[JsonDict] = None) -> bool:
        safety = local_costmap_motion_safety(analysis if isinstance(analysis, dict) else {}, action)
        return bool(safety.known and safety.safe is False)

    def local_costmap_known_safe(self, action: str, analysis: Optional[JsonDict] = None) -> bool:
        safety = local_costmap_motion_safety(analysis if isinstance(analysis, dict) else {}, action)
        return bool(safety.known and safety.safe is True)

    def can_safely_move_action(self, action: str, analysis: JsonDict) -> bool:
        """Final online-safe action gate combining local costmap and legacy cues.

        The local RGB-D costmap owns hard obstacle vetoes. Legacy
        ``open_directions`` remains only as a backward-compatible cue when the
        depth costmap lacks a confident observation.
        """
        if action not in MOVE_ACTIONS:
            return False
        if action in LOOK_ACTIONS:
            return not self.recent_action_failed(action)
        local_safety = local_costmap_motion_safety(analysis if isinstance(analysis, dict) else {}, action)
        local_safe = bool(local_safety.known and local_safety.safe is True)
        local_unsafe = bool(local_safety.known and local_safety.safe is False)
        if local_unsafe:
            return False
        if self.recent_action_failed(action):
            return False
        if self.holding_object and self.held_move_action_blocked(action):
            return False

        open_directions = set(str(item) for item in (analysis.get("open_directions") or []))
        if action == "MoveAhead":
            if not local_safe and (bool(analysis.get("obstacle_ahead", False)) or "forward" not in open_directions):
                return False
            if base_action(self.recent_actions[-1] if self.recent_actions else None) == "MoveBack":
                return False
            if self.recent_moveback_loop():
                return False
        elif action == "MoveLeft":
            if not local_safe:
                return False
        elif action == "MoveRight":
            if not local_safe:
                return False
        elif action == "MoveBack":
            if not local_safe:
                return False
            if self.last_action_is("MoveBack") or self.recent_moveback_loop():
                return False
        return True

    def can_safely_move_ahead(self, analysis: JsonDict) -> bool:
        return self.can_safely_move_action("MoveAhead", analysis)

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
                    "position_map_status": nav_status.get("position_map_status"),
                    "pose_confidence": nav_status.get("pose_confidence"),
                    "position_uncertainty_cells": nav_status.get("position_uncertainty_cells"),
                    "heading_confidence": nav_status.get("heading_confidence"),
                    "pose_trust": nav_status.get("pose_trust", {}),
                    "occupancy_summary": nav_status.get("occupancy_summary", {}),
                },
            )
        except Exception as exc:
            self.emit("navigation_memory_error", {"phase": "observe", "message": str(exc)})

    def navigation_status_for_memory(self) -> JsonDict:
        try:
            return self.navigation.status()
        except Exception:
            return {
                "last_cell": "0,0",
                "last_heading": "north",
                "cell_size": 0.25,
                "visited_cells": [],
                "frontier_cells": [],
                "blocked_edges": [],
            }

    def observe_object_memory(self, vision: JsonDict, analysis: JsonDict) -> None:
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return
        nav_status = self.navigation_status_for_memory()
        current_cell = str(nav_status.get("last_cell") or "0,0")
        heading = str(nav_status.get("last_heading") or "north")
        try:
            result = self.object_memory.update_from_observation(
                analysis,
                current_cell=current_cell,
                heading=heading,
                step=self.current_step_count(),
                navigation_status=nav_status,
                persist=not self.config.dry_run,
            )
            self.emit(
                "object_memory_updated",
                {
                    "step": result.get("step"),
                    "reason": analysis.get("object_memory_update_reason"),
                    "observed_from": result.get("observed_from"),
                    "observation_count": result.get("observation_count"),
                    "created_count": result.get("created_count"),
                    "updated_count": result.get("updated_count"),
                    "track_count": result.get("track_count"),
                },
            )
            emit_track_details = bool(self.full_performance_profile() or self.config.log_detail == "full")
            for event in result.get("events") or []:
                if not isinstance(event, dict):
                    continue
                event_name = str(event.get("event") or "")
                if event_name in {"object_track_created", "object_status_changed"} or (
                    emit_track_details
                    and event_name in {"object_track_updated", "object_track_merged"}
                ):
                    self.emit(event_name, {key: value for key, value in event.items() if key != "event"})
        except Exception as exc:
            self.emit("object_memory_error", {"phase": "observe", "message": str(exc)})

    def observe_semantic_mapping(self, analysis: JsonDict) -> None:
        """Fuse current perception and object-memory tracks onto position cells."""
        if str(self.config.task_mode or "clean").strip().lower() != "tidy":
            return
        try:
            result = self.navigation.observe_semantics(
                analysis=analysis,
                step=self.current_step_count(),
                persist=not self.config.dry_run,
            )
            self.emit(
                "semantic_map_updated",
                {
                    "step": result.get("step"),
                    "reason": analysis.get("semantic_mapping_update_reason"),
                    "analysis_update_count": result.get("analysis_update_count"),
                    "track_update_count": result.get("track_update_count"),
                    "semantic_cell_count": (result.get("stats") or {}).get("semantic_cell_count"),
                    "label_count": (result.get("stats") or {}).get("label_count"),
                    "track_count": (result.get("stats") or {}).get("track_count"),
                    "frontier_score_count": len(result.get("frontier_scores", {}) or {}),
                },
            )
        except Exception as exc:
            self.emit("semantic_map_error", {"phase": "observe", "message": str(exc)})


    def object_memory_navigation_target(
        self,
        target: JsonDict,
        *,
        goal_type: str,
        reason: str,
    ) -> Decision:
        self.pending_object_memory_target = dict(target)
        nav_analysis = dict(self.current_analysis or {})
        nav_analysis["object_memory_target"] = {
            "track_id": target.get("track_id"),
            "goal_type": goal_type,
            "target_cell": target.get("recommended_view_cell") or target.get("goal_cell"),
        }
        recommendation = self.navigation_recommendation(
            {},
            nav_analysis,
            object_memory_target=target,
            goal_type=goal_type,
        )
        plan_status = str((recommendation or {}).get("plan_status") or "")
        nav_reason = str((recommendation or {}).get("reason") or "")
        if plan_status == "no_path" and goal_type == "pickup_target":
            try:
                status_change = self.object_memory.mark_unreachable(
                    track_id=str(target.get("track_id") or "") or None,
                    candidate=target,
                    step=self.current_step_count(),
                    reason=f"navigation_no_path:{nav_reason or 'astar_no_path'}",
                )
                if isinstance(status_change, dict):
                    self.emit("object_status_changed", status_change)
            except Exception as exc:
                self.emit("object_memory_error", {"phase": "mark_no_path_pickup_target", "message": str(exc)})
        action = str((recommendation or {}).get("action") or "")
        if action not in MOVE_ACTIONS:
            action = self.safe_turn_action(self.current_analysis or {})
        self.emit(
            "object_goal_selected",
            {
                "track_id": target.get("track_id"),
                "goal_type": goal_type,
                "goal_cell": target.get("recommended_view_cell") or target.get("goal_cell"),
                "recommended_heading": target.get("recommended_heading"),
                "action": action,
                "reason": reason,
                "score": target.get("score"),
            },
        )
        candidate = {
            "label": target.get("label"),
            "raw_label": target.get("raw_label") or target.get("label"),
            "task_semantic_class": target.get("goal_type"),
            "object_memory_track_id": target.get("track_id"),
            "object_memory_target": True,
            "recommended_view_cell": target.get("recommended_view_cell"),
            "recommended_heading": target.get("recommended_heading"),
            "estimated_object_cell": target.get("estimated_object_cell"),
            "position_hint": "memory",
            "goal_resolution": target.get("goal_resolution"),
            "pose_trust": target.get("pose_trust", {}),
            "placement_viewpoint_planner": bool(target.get("placement_viewpoint_planner")),
            "placement_viewpoint_id": target.get("placement_viewpoint_id"),
            "placement_viewpoint_status": target.get("placement_viewpoint_status"),
            "placement_viewpoint_target_cell": target.get("placement_viewpoint_target_cell"),
        }
        return Decision(
            kind="move",
            action=action,
            mode="SERVICE",
            reason=f"{reason};object_memory_target:{target.get('track_id')};nav:{(recommendation or {}).get('reason')}",
            candidate=candidate,
        )

    def select_pickup_object_memory_target(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        *,
        reason: str,
    ) -> Optional[Decision]:
        nav_status = self.navigation_status_for_memory()
        blocked_track_ids = self.active_recently_placed_track_ids()
        try:
            target = self.object_memory.select_pickup_target(
                current_cell=str(nav_status.get("last_cell") or "0,0"),
                heading=str(nav_status.get("last_heading") or "north"),
                step=self.current_step_count(),
                navigation_status=nav_status,
                analysis=analysis,
                pickup_surface_policy=self.config.pickup_surface_policy,
                blocked_track_ids=blocked_track_ids,
                persist=not self.config.dry_run,
            )
        except Exception as exc:
            self.emit("object_memory_error", {"phase": "select_pickup", "message": str(exc)})
            return None
        if not isinstance(target, dict):
            return None
        self.service_state["target_track_id"] = target.get("track_id")
        self.save_service_task_state()
        return self.object_memory_navigation_target(
            target,
            goal_type="pickup_target",
            reason=reason,
        )

    def placement_viewpoint_virtual_candidate(
        self,
        target: JsonDict,
        plan: JsonDict,
    ) -> JsonDict:
        return {
            "label": target.get("label"),
            "raw_label": target.get("raw_label") or target.get("label"),
            "task_semantic_class": "place_receptacle",
            "object_memory_track_id": target.get("track_id"),
            "object_memory_target": True,
            "position_hint": "memory",
            "goal_resolution": target.get("goal_resolution"),
            "pose_trust": plan.get("pose_trust", target.get("pose_trust", {})),
            "placement_viewpoint_planner": True,
            "placement_viewpoint_id": plan.get("viewpoint_id"),
            "placement_viewpoint_status": plan.get("status"),
            "placement_viewpoint_target_cell": plan.get("target_cell"),
            "placement_viewpoint_target_heading": plan.get("target_heading"),
        }

    def select_receptacle_placement_viewpoint_target(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        *,
        reason: str,
        context_candidate: Optional[JsonDict] = None,
    ) -> Optional[Decision]:
        """Choose a receptacle and actively search for a camera viewpoint.

        ObjectMemory chooses the remembered receptacle region. The persistent
        PlacementViewpointPlanner then chooses / exhausts concrete standoff
        viewpoints. This prevents endless RotateLeft / RotateRight scans after
        arriving at one stale recommended_view_cell.
        """
        nav_status = self.navigation_status_for_memory()
        try:
            target = self.object_memory.select_receptacle_target(
                holding_object=bool(self.holding_object),
                current_cell=str(nav_status.get("last_cell") or "0,0"),
                heading=str(nav_status.get("last_heading") or "north"),
                step=self.current_step_count(),
                navigation_status=nav_status,
                held_object_family=str(self.service_state.get("held_object_family") or "unknown"),
                analysis=analysis,
                persist=not self.config.dry_run,
            )
        except Exception as exc:
            self.emit("object_memory_error", {"phase": "select_receptacle_for_viewpoint", "message": str(exc)})
            return None
        if not isinstance(target, dict):
            return None

        self.service_state["receptacle_track_id"] = target.get("track_id")
        self.save_service_task_state()
        dominant_reason = self.dominant_surface_rejection_reason(analysis)
        if str(analysis.get("surface_free_space_status") or "") == "free_space_outside_current_reach":
            dominant_reason = "too_far"
        try:
            plan = self.placement_viewpoints.plan(
                target=target,
                current_cell=str(nav_status.get("last_cell") or "0,0"),
                current_heading=str(nav_status.get("last_heading") or "north"),
                step=self.current_step_count(),
                navigation_status=nav_status,
                analysis=analysis,
                context_candidate=context_candidate,
                dominant_rejection_reason=dominant_reason,
                persist=not self.config.dry_run,
            )
        except Exception as exc:
            self.emit("placement_viewpoint_error", {"phase": "plan", "message": str(exc)})
            return None

        self.emit(
            "placement_viewpoint_plan",
            {
                "track_id": target.get("track_id"),
                "status": plan.get("status"),
                "reason": plan.get("reason"),
                "viewpoint_id": plan.get("viewpoint_id"),
                "target_cell": plan.get("target_cell"),
                "target_heading": plan.get("target_heading"),
                "action": plan.get("action"),
                "pose_trust": plan.get("pose_trust"),
                "exhausted_viewpoint_id": plan.get("exhausted_viewpoint_id"),
            },
        )

        try:
            self.object_memory.update_active_goal_viewpoint(
                track_id=str(target.get("track_id") or ""),
                viewpoint=plan,
                step=self.current_step_count(),
                planner_status=str(plan.get("status") or "placement_viewpoint_planning"),
            )
        except Exception as exc:
            self.emit("object_memory_error", {"phase": "update_active_goal_viewpoint", "message": str(exc)})

        direct_action = str(plan.get("action") or "")
        virtual_candidate = self.placement_viewpoint_virtual_candidate(target, plan)
        if direct_action in MOVE_ACTIONS:
            if not self.can_safely_move_action(direct_action, analysis):
                direct_action = self.held_object_alternative_action(direct_action, analysis) or self.safe_turn_action(analysis)
            direct_action, suffix = self.service_safe_move_action(
                direct_action,
                analysis,
                allow_forward_break=True,
            )
            return Decision(
                kind="move",
                action=direct_action,
                mode="SERVICE",
                reason=f"{reason};placement_viewpoint:{plan.get('status')};{plan.get('reason')};{suffix}",
                candidate=virtual_candidate,
            )

        target_cell = str(plan.get("target_cell") or "")
        if target_cell:
            target_override = dict(target)
            target_override["recommended_view_cell"] = target_cell
            target_override["goal_cell"] = target_cell
            target_override["recommended_heading"] = plan.get("target_heading") or target.get("recommended_heading")
            target_override["placement_viewpoint_planner"] = True
            target_override["placement_viewpoint_id"] = plan.get("viewpoint_id")
            target_override["placement_viewpoint_status"] = plan.get("status")
            target_override["placement_viewpoint_target_cell"] = target_cell
            target_override["pose_trust"] = plan.get("pose_trust", target.get("pose_trust", {}))
            return self.object_memory_navigation_target(
                target_override,
                goal_type="place_receptacle",
                reason=f"{reason};placement_viewpoint:{plan.get('status')}",
            )
        return None

    def select_receptacle_object_memory_target(
        self,
        analysis: JsonDict,
        vision: JsonDict,
        *,
        reason: str,
    ) -> Optional[Decision]:
        nav_status = self.navigation_status_for_memory()
        try:
            target = self.object_memory.select_receptacle_target(
                holding_object=bool(self.holding_object),
                current_cell=str(nav_status.get("last_cell") or "0,0"),
                heading=str(nav_status.get("last_heading") or "north"),
                step=self.current_step_count(),
                navigation_status=nav_status,
                held_object_family=str(self.service_state.get("held_object_family") or "unknown"),
                analysis=analysis,
                persist=not self.config.dry_run,
            )
        except Exception as exc:
            self.emit("object_memory_error", {"phase": "select_receptacle", "message": str(exc)})
            return None
        if not isinstance(target, dict):
            return None
        self.service_state["receptacle_track_id"] = target.get("track_id")
        self.save_service_task_state()
        return self.object_memory_navigation_target(
            target,
            goal_type="place_receptacle",
            reason=reason,
        )

    def navigation_recommendation(
        self,
        vision: JsonDict,
        analysis: JsonDict,
        *,
        object_memory_target: Optional[JsonDict] = None,
        goal_type: Optional[str] = None,
    ) -> Optional[JsonDict]:
        recent_failed_action = None
        if self.recent_results:
            action, success = self.recent_results[-1]
            if not success:
                recent_failed_action = action
        try:
            target_cell = None
            target_heading = None
            target_track_id = None
            target_reason = None
            if isinstance(object_memory_target, dict):
                target_cell = object_memory_target.get("recommended_view_cell") or object_memory_target.get("goal_cell")
                target_heading = object_memory_target.get("recommended_heading")
                target_track_id = object_memory_target.get("track_id")
                target_reason = "object_memory_target"
            nav_status = self.navigation.recommend(
                vision=vision,
                analysis=analysis,
                recent_actions=self.recent_actions,
                recent_failed_action=recent_failed_action,
                target_cell=str(target_cell) if target_cell else None,
                target_reason=target_reason,
                target_track_id=str(target_track_id) if target_track_id else None,
                goal_type=goal_type,
                target_heading=str(target_heading) if target_heading else None,
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
                        "semantic_map_summary": nav_status.get("semantic_map_summary", {}),
                        "planner": recommendation.get("planner"),
                        "plan_status": recommendation.get("plan_status"),
                        "path": recommendation.get("path", []),
                        "next_cell": recommendation.get("next_cell"),
                        "route_cost": recommendation.get("route_cost"),
                        "requested_target_cell": recommendation.get("requested_target_cell"),
                        "selected_goal_cell": recommendation.get("selected_goal_cell"),
                        "selected_goal_kind": recommendation.get("selected_goal_kind"),
                        "semantic_score": recommendation.get("semantic_score"),
                        "replan_reason": recommendation.get("replan_reason"),
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
                    
                    "position_map_status": nav_status.get("position_map_status"),
                    "pose_confidence": nav_status.get("pose_confidence"),
                    "position_uncertainty_cells": nav_status.get("position_uncertainty_cells"),
                    "heading_confidence": nav_status.get("heading_confidence"),
                    "pose_trust": nav_status.get("pose_trust", {}),
                    "occupancy_summary": nav_status.get("occupancy_summary", {}),
                    "semantic_map_status": nav_status.get("semantic_map_status"),
                    "semantic_map_summary": nav_status.get("semantic_map_summary", {}),
                    "last_global_plan": nav_status.get("last_global_plan", {}),
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
        """Explore with global A* guidance and a local RGB-D safety gate."""
        open_directions = list(analysis.get("open_directions", []) or [])
        obstacle_ahead = bool(analysis.get("obstacle_ahead", False))
        frontier_exists = bool(analysis.get("frontier_exists", False))

        # 1. Prefer semantic/object-memory A* guidance.  The planner may now
        # emit lateral moves, backoff, or rotations.
        nav_recommendation = self.navigation_recommendation(vision, analysis)
        if nav_recommendation:
            nav_action = str(nav_recommendation.get("action") or "")
            nav_reason = str(nav_recommendation.get("reason") or "navigation_memory")
            if nav_action in MOVE_ACTIONS:
                action, suffix = self.service_safe_move_action(
                    nav_action,
                    analysis,
                    allow_forward_break=True,
                )
                if action in MOVE_ACTIONS and self.can_safely_move_action(action, analysis):
                    return Decision(
                        kind="move",
                        action=action,
                        mode="EXPLORE",
                        reason=f"{reason};nav:{nav_reason};{suffix}",
                    )

        # 2. Stop only after the map reports a durable exploration plateau.
        plateau_reason = self.navigation_plateau_completion_reason()
        if plateau_reason:
            return Decision(
                kind="stop",
                action="none",
                mode="ROOM_COMPLETE",
                reason=f"{reason};{plateau_reason}",
            )

        # 3. Depth-local obstacle recovery: prefer a scan turn before any
        # lateral translation because the robot's useful view is forward.
        if self.local_costmap_known_unsafe("MoveAhead", analysis) or (
            obstacle_ahead and not self.local_costmap_known_safe("MoveAhead", analysis)
        ):
            preferred = ["RotateLeft", "RotateRight", "MoveBack", "MoveLeft", "MoveRight"]
            for requested in preferred:
                action, suffix = self.service_safe_move_action(requested, analysis, allow_forward_break=False)
                if action in MOVE_ACTIONS and self.can_safely_move_action(action, analysis):
                    return Decision(
                        kind="move",
                        action=action,
                        mode="EXPLORE",
                        reason=f"{reason};local_costmap_obstacle_ahead:{requested}->{action};{suffix}",
                    )

        # 4. Continue frontier expansion with a safe translation when possible.
        if frontier_exists:
            if self.can_safely_move_action("MoveAhead", analysis):
                return Decision(
                    kind="move",
                    action="MoveAhead",
                    mode="EXPLORE",
                    reason=f"{reason};frontier_forward_open",
                )
            action = self.prefer_side_turn(open_directions, analysis=analysis)
            action, suffix = self.service_safe_move_action(action, analysis, allow_forward_break=True)
            if action in MOVE_ACTIONS and self.can_safely_move_action(action, analysis):
                return Decision(
                    kind="move",
                    action=action,
                    mode="EXPLORE",
                    reason=f"{reason};frontier_local_recovery:{suffix}",
                )

        # 5. Conservative completion fallback.
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

        action = self.safe_turn_action(analysis)
        action, suffix = self.service_safe_move_action(action, analysis, allow_forward_break=True)
        return Decision(
            kind="move",
            action=action,
            mode="EXPLORE",
            reason=f"{reason};no_frontier_conservative_scan:{suffix}",
        )

    def safe_turn_action(self, analysis: Optional[JsonDict] = None) -> str:
        source = analysis if isinstance(analysis, dict) else {}
        open_directions = list(source.get("open_directions", []) or [])
        candidates: List[str] = []
        if "left" in open_directions or self.local_costmap_known_safe("MoveLeft", source):
            candidates.extend(["RotateLeft", "MoveLeft"])
        if "right" in open_directions or self.local_costmap_known_safe("MoveRight", source):
            candidates.extend(["RotateRight", "MoveRight"])
        last = base_action(self.recent_actions[-1] if self.recent_actions else None)
        candidates.extend(["RotateRight" if last == "RotateRight" else "RotateLeft", "RotateRight", "RotateLeft"])
        for action in candidates:
            if action in AUTONOMOUS_BODY_ACTIONS and self.can_safely_move_action(action, source):
                return action
        return "RotateLeft"

    def prefer_side_turn(self, open_directions: Iterable[str], *, analysis: Optional[JsonDict] = None) -> str:
        """Prefer turning toward an open side; use lateral translation only with strong RGB-D evidence."""
        directions = list(open_directions)
        source = analysis if isinstance(analysis, dict) else {}
        last = base_action(self.recent_actions[-1] if self.recent_actions else None)
        left_open = "left" in directions or self.local_costmap_known_safe("MoveLeft", source)
        right_open = "right" in directions or self.local_costmap_known_safe("MoveRight", source)
        if left_open and self.can_safely_move_action("RotateLeft", source):
            return "RotateLeft"
        if right_open and self.can_safely_move_action("RotateRight", source):
            return "RotateRight"
        if left_open and last != "MoveRight" and self.can_safely_move_action("MoveLeft", source):
            return "MoveLeft"
        if right_open and self.can_safely_move_action("MoveRight", source):
            return "MoveRight"
        return self.safe_turn_action(source)

    def break_rotation_oscillation(self, action: str, analysis: JsonDict, *, allow_forward_break: bool = True) -> Tuple[str, str]:
        if action not in ROTATE_ACTIONS:
            return action, "normal"

        last = base_action(self.recent_actions[-1] if self.recent_actions else None)
        previous = base_action(self.recent_actions[-2] if len(self.recent_actions) >= 2 else None)

        if last == OPPOSITE_ROTATION[action] and previous == action:
            preferred = ["MoveAhead", str(last)] if allow_forward_break else [str(last)]
            for candidate in preferred:
                if candidate in AUTONOMOUS_BODY_ACTIONS and self.can_safely_move_action(candidate, analysis):
                    return candidate, f"break_rotate_oscillation_{candidate.lower()}"
            return self.safe_turn_action(analysis), "break_rotate_oscillation_body_scan"

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
            nav_status = self.navigation.status()
        except Exception:
            return False
        return bool(list(nav_status.get("frontier_cells", []) or []))

    def turn_streak_from_actions(self, actions: Sequence[Any]) -> int:
        streak = 0
        for action in reversed(list(actions)):
            if base_action(action) in ROTATE_ACTIONS:
                streak += 1
                continue
            break
        return streak

    def navigation_accessible_area_completion_reason(
        self,
        nav_status: JsonDict,
        *,
        step_count: int,
        max_steps: int,
        no_target: int,
        repeated: int,
        stagnation: int,
        turn_streak: int,
        nav_frontier_cells: Sequence[Any],
        at_max_steps: bool = False,
    ) -> Optional[str]:
        """Complete furnished rooms when all reachable frontier is exhausted."""
        if list(nav_frontier_cells or []):
            return None

        occupancy = nav_status.get("occupancy_summary") if isinstance(nav_status, dict) else {}
        if not isinstance(occupancy, dict):
            occupancy = {}

        def int_value(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def float_value(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        coverage = float_value(nav_status.get("coverage_estimate", 0.0) if isinstance(nav_status, dict) else 0.0)
        visited_count = int_value(nav_status.get("visited_cell_count", 0) if isinstance(nav_status, dict) else 0)
        free_count = int_value(occupancy.get("free_cell_count", 0))
        occupied_count = int_value(occupancy.get("occupied_cell_count", 0))
        inflated_count = int_value(occupancy.get("inflated_cell_count", 0))
        raw_unknown_frontiers = int_value(occupancy.get("unknown_frontier_count", 0))
        blocked_edges = len(nav_status.get("blocked_edges", []) or []) if isinstance(nav_status, dict) else 0

        min_steps = max(1, int(env_float("ROBOT_ACCESSIBLE_COMPLETION_MIN_STEPS", 35.0)))
        min_visited = max(1, int(env_float("ROBOT_ACCESSIBLE_COMPLETION_MIN_VISITED_CELLS", 12.0)))
        min_free = max(1, int(env_float("ROBOT_ACCESSIBLE_COMPLETION_MIN_FREE_CELLS", 20.0)))
        min_stagnation = max(1, int(env_float("ROBOT_ACCESSIBLE_COMPLETION_MIN_STAGNATION", 6.0)))
        min_no_target = max(1, int(env_float("ROBOT_ACCESSIBLE_COMPLETION_MIN_NO_TARGET", 6.0)))

        map_has_enough_evidence = bool(visited_count >= min_visited or free_count >= min_free)
        stable_no_new_work = bool(
            stagnation >= min_stagnation
            or no_target >= min_no_target
            or repeated >= 2
            or turn_streak >= 8
        )
        if not at_max_steps and step_count < min_steps:
            return None
        if not map_has_enough_evidence:
            return None
        if not at_max_steps and not stable_no_new_work:
            return None

        reason_prefix = "reachable_area_complete_at_max_steps" if at_max_steps else "reachable_area_complete"
        return (
            f"{reason_prefix}:coverage={coverage:.2f};reachable_frontiers=0;"
            f"raw_unknown_frontiers={raw_unknown_frontiers};visited={visited_count};"
            f"free={free_count};occupied={occupied_count};inflated={inflated_count};"
            f"blocked_edges={blocked_edges};stagnation={stagnation};"
            f"no_target={no_target};repeated={repeated}"
        )

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
        try:
            state = self.manager.load_state()
            no_target = int(state.patrol.get("consecutive_no_target", 0) or 0)
            repeated = int(state.patrol.get("consecutive_repeated_view", 0) or 0)
            max_steps = int(state.patrol.get("max_steps", DEFAULT_MAX_STEPS))
        except Exception:
            no_target = 0
            repeated = 0
            max_steps = DEFAULT_MAX_STEPS

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
        accessible_reason = self.navigation_accessible_area_completion_reason(
            nav_status,
            step_count=step_count,
            max_steps=max_steps,
            no_target=no_target,
            repeated=repeated,
            stagnation=stagnation,
            turn_streak=turn_streak,
            nav_frontier_cells=frontier_cells,
            at_max_steps=False,
        )
        if accessible_reason:
            return accessible_reason
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
        # 鎴愬姛鏉′欢锛氭竻鎵剼鏈墽琛屽畬锟?clean_executed)
        status = data.get("status")
        result_type = data.get("result_type")
        removed_from_view = data.get("removed_from_view")
        removed_from_scene = data.get("removed_from_scene")
        last_action_success = data.get("lastActionSuccess")

        # V2 RGB-only online contract:
        # /clean 榛樿浼氳劚锟?object/metadata 缁嗚妭锛屽洜姝ゅ湪绾块摼璺笉鑳藉啀渚濊禆
        # removed_from_view / removed_from_scene 鏉ュ垽鏂竻鎵槸鍚︽垚鍔燂拷?        #
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
    # Update tidy/service state after one executed physical action.
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
                if self.holding_object:
                    self.set_held_object_context_from_candidate(decision.candidate)
                    try:
                        picked = self.object_memory.mark_picked(
                            track_id=str(self.service_state.get("target_track_id") or "") or None,
                            candidate=decision.candidate,
                            step=self.current_step_count(),
                        )
                        if isinstance(picked, dict):
                            self.service_state["held_object_track_id"] = picked.get("track_id")
                            self.emit("object_status_changed", picked)
                    except Exception as exc:
                        self.emit("object_memory_error", {"phase": "mark_picked", "message": str(exc)})
                else:
                    self.clear_held_object_context()
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
                    held_track_id = str(self.service_state.get("held_object_track_id") or "") or None
                    receptacle_track_id = str(self.service_state.get("receptacle_track_id") or "") or None
                    try:
                        placed_result = self.object_memory.mark_placed(
                            held_track_id=held_track_id,
                            receptacle_track_id=receptacle_track_id,
                            receptacle_candidate=decision.candidate,
                            step=self.current_step_count(),
                        )
                        for event in placed_result.get("events", []) if isinstance(placed_result, dict) else []:
                            if isinstance(event, dict):
                                self.emit("object_status_changed", event)
                        self.emit(
                            "object_goal_cleared",
                            {
                                "reason": "place_executed",
                                "held_track_id": held_track_id,
                                "receptacle_track_id": receptacle_track_id,
                            },
                        )
                    except Exception as exc:
                        self.emit("object_memory_error", {"phase": "mark_placed", "message": str(exc)})
                    try:
                        self.placement_viewpoints.mark_place_success(
                            track_id=receptacle_track_id,
                            step=self.current_step_count(),
                        )
                    except Exception as exc:
                        self.emit("placement_viewpoint_error", {"phase": "mark_place_success", "message": str(exc)})
                    self.clear_held_object_context()
                    completion = f"{placed_label}->{receptacle_label}"
                    self.pending_service_completions.append(completion)
                    self.pending_placed_objects.append(placed_label)
                    self.suppress_recently_placed_track(track_id=held_track_id, label=placed_label)
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
                    self.service_state["target_track_id"] = None
                    self.service_state["target_last_seen_step"] = None
                    self.service_state["target_attempts"] = 0
                    self.service_state["receptacle_label"] = None
                    self.service_state["receptacle_raw_label"] = None
                    self.service_state["receptacle_signature"] = None
                    self.service_state["receptacle_track_id"] = None
                    self.service_state["receptacle_last_seen_step"] = None
                    self.service_state["receptacle_attempts"] = 0
                    self.service_state["receptacle_action_hint"] = None
                    self.service_state["receptacle_action_hint_until_step"] = None
                    self.service_state["receptacle_last_position_hint"] = None
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
            if isinstance(decision.candidate, dict) and decision.candidate.get("placement_viewpoint_planner"):
                try:
                    self.placement_viewpoints.record_move_result(
                        track_id=str(decision.candidate.get("object_memory_track_id") or "") or None,
                        viewpoint_id=str(decision.candidate.get("placement_viewpoint_id") or "") or None,
                        action=decision.action,
                        success=bool(success),
                        step=self.current_step_count(),
                        failure_reason=failure_reason,
                    )
                except Exception as exc:
                    self.emit("placement_viewpoint_error", {"phase": "record_move_result", "message": str(exc)})
            if (
                not success
                and isinstance(decision.candidate, dict)
                and decision.candidate.get("object_memory_target")
            ):
                try:
                    status_change = self.object_memory.mark_unreachable(
                        track_id=str(decision.candidate.get("object_memory_track_id") or "") or None,
                        candidate=decision.candidate,
                        step=self.current_step_count(),
                        reason=f"navigation_failed:{failure_reason or 'move_failed'}",
                    )
                    if isinstance(status_change, dict):
                        self.emit("object_status_changed", status_change)
                except Exception as exc:
                    self.emit("object_memory_error", {"phase": "mark_navigation_target_failed", "message": str(exc)})
            if self.holding_object and not success and decision.action in MOVE_ACTIONS:
                failure_text = " ".join(
                    str(value or "")
                    for value in (
                        failure_reason,
                        execution.get("result_type"),
                        execution.get("error_message"),
                        execution.get("message"),
                    )
                )
                block_steps = 6 if "held" in failure_text.lower() else 4
                self.block_held_move_action(
                    decision.action,
                    reason=failure_text.strip() or "service_move_failed_while_holding",
                    steps=block_steps,
                )
            self.service_state["phase_attempts"] = int(self.service_state.get("phase_attempts", 0) or 0) + 1
            self.service_state["holding_object"] = bool(self.holding_object)
            if self.service_phase() == "APPROACH_RECEPTACLE" and decision.action in MOVE_ACTIONS:
                self.service_state["receptacle_action_hint"] = None
                self.service_state["receptacle_action_hint_until_step"] = None
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
            if not self.holding_object:
                self.clear_held_object_context()
            elif role == "pickup":
                self.set_held_object_context_from_candidate(candidate)
        key = candidate_key(candidate) if isinstance(candidate, dict) else f"{role}:unknown"
        self.service_failures[key] = self.service_failures.get(key, 0) + 1
        result_type = str(execution.get("result_type") or failure_reason or "service_action_failed")
        self.append_service_history(
            "service_action_failed",
            role=role,
            result_type=result_type,
            failure_count=self.service_failures[key],
            executor_action_hint=execution.get("executor_action_hint"),
            interactable_current_pose=execution.get("interactable_current_pose"),
            interactable_pose_count=execution.get("interactable_pose_count"),
            interactable_distance_bucket=execution.get("interactable_distance_bucket"),
            interactable_angle_bucket=execution.get("interactable_angle_bucket"),
            candidate=self.short_candidate(candidate),
        )
        try:
            status_change = self.object_memory.mark_unreachable(
                track_id=str(
                    self.service_state.get("target_track_id" if role == "pickup" else "receptacle_track_id")
                    or ""
                ) or None,
                candidate=candidate,
                step=self.current_step_count(),
                reason=result_type,
            )
            if isinstance(status_change, dict):
                self.emit("object_status_changed", status_change)
        except Exception as exc:
            self.emit("object_memory_error", {"phase": "mark_unreachable", "message": str(exc)})
        if role == "place" and isinstance(candidate, dict) and result_type in PLACE_COOLDOWN_ERRORS:
            self.mark_surface_candidate_failed(
                candidate,
                result_type=result_type,
                failed_candidate_id=str(execution.get("failed_candidate_id") or "") or None,
            )
            self.clear_service_lock(role=role, reason=f"place_surface_cooldown_after:{result_type}")
            self.set_service_phase(
                "SEARCH_RECEPTACLE",
                reason=f"place_surface_cooldown_after:{result_type}",
                candidate=None,
            )
            return
        if role == "place" and isinstance(candidate, dict) and result_type in PLACE_APPROACH_ERRORS:
            self.store_receptacle_action_hint(execution.get("executor_action_hint"), steps=2)
            self.set_service_phase("APPROACH_RECEPTACLE", reason=f"place_needs_approach_after:{result_type}", candidate=candidate)
            return
        if role == "place" and isinstance(candidate, dict) and result_type in PLACE_RETRY_ERRORS:
            self.service_state["receptacle_alignment_streak"] = 0
            self.set_service_phase(
                "ALIGN_RECEPTACLE",
                reason=f"place_retry_same_receptacle_after:{result_type}",
                candidate=candidate,
            )
            return
        if role == "place" and isinstance(candidate, dict) and result_type in PLACE_RETARGET_ERRORS:
            self.suppress_service_candidate(candidate, reason=result_type, steps=6)
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
# 鎴愬姛鏉′欢锟?    # 1. 鑴氭湰杩斿洖鐘讹拷?success
    # 2. 鍔ㄤ綔鎵ц鎴愬姛(lastActionSuccess鈮燜alse)
    # 3. 鏈哄櫒浜虹姸鎬佸彂鐢熷彉锟?state_changed鈮燜alse)   涓変釜鏉′欢閮芥槸鎸囩殑鍚屼竴锟?
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
 #鏈哄櫒浜烘瘡璧颁竴姝ャ€佸垎鏋愬畬鐢婚潰鍚庛€佸仛鍐崇瓥锟?璋冪敤
    def update_repeated_view(self, vision: JsonDict, analysis: JsonDict) -> bool:
        signature = view_signature(vision, analysis)
        repeated = self.last_view_signature == signature
        self.last_view_signature = signature
        return repeated
    """
    鏇存柊褰撳墠娈电殑缁熻淇℃伅锛屾瘮濡傦細
    杩欎竴娈垫湁娌℃湁鐪嬪埌鍨冨溇锟?    鏈夋病鏈夌湅鍒版柊鏂瑰悜锟?    鏈夋病鏈夊け璐ワ紵
    """
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
        nav_status: JsonDict = {}
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
            and bool(nav_frontier_cells)
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

        accessible_reason = self.navigation_accessible_area_completion_reason(
            nav_status,
            step_count=step_count,
            max_steps=max_steps,
            no_target=no_target,
            repeated=repeated,
            stagnation=stagnation,
            turn_streak=turn_streak,
            nav_frontier_cells=nav_frontier_cells,
            at_max_steps=False,
        )
        if accessible_reason:
            self.mark_room_complete(accessible_reason)
            return True

        if step_count >= max_steps:
            if nav_frontier_cells:
                self.mark_recover_failed(
                    f"max_steps_incomplete_or_frontier:"
                    f"coverage={coverage:.2f};frontiers={len(nav_frontier_cells)}"
                )
                return True
            max_step_accessible_reason = self.navigation_accessible_area_completion_reason(
                nav_status,
                step_count=step_count,
                max_steps=max_steps,
                no_target=no_target,
                repeated=repeated,
                stagnation=stagnation,
                turn_streak=turn_streak,
                nav_frontier_cells=nav_frontier_cells,
                at_max_steps=True,
            )
            if max_step_accessible_reason:
                self.mark_room_complete(max_step_accessible_reason)
                return True
            if coverage < 0.95:
                self.mark_recover_failed(
                    f"max_steps_insufficient_map_evidence:"
                    f"coverage={coverage:.2f};frontiers=0"
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
            stalled_accessible_reason = self.navigation_accessible_area_completion_reason(
                nav_status,
                step_count=step_count,
                max_steps=max_steps,
                no_target=no_target,
                repeated=repeated,
                stagnation=stagnation,
                turn_streak=turn_streak,
                nav_frontier_cells=nav_frontier_cells,
                at_max_steps=True,
            )
            if stalled_accessible_reason:
                self.mark_room_complete(stalled_accessible_reason)
            else:
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
            "id": candidate.get("id"),
            "surface_candidate_id": candidate.get("surface_candidate_id"),
            "label": candidate.get("label"),
            "raw_label": candidate.get("raw_label"),
            "task_semantic_class": candidate.get("task_semantic_class"),
            "position_hint": candidate.get("position_hint"),
            "surface_hint": candidate.get("surface_hint"),
            "cleanable_now": candidate.get("cleanable_now"),
            "pickup_now": candidate.get("pickup_now"),
            "place_now": candidate.get("place_now"),
            "visual_place_ready": candidate.get("visual_place_ready"),
            "final_place_ready": candidate.get("final_place_ready"),
            "failed_recently": candidate.get("failed_recently"),
            "cooldown_remaining": candidate.get("cooldown_remaining"),
            "needs_alignment": candidate.get("needs_alignment"),
            "needs_approach": candidate.get("needs_approach"),
            "reachable": candidate.get("reachable"),
            "confidence": candidate.get("confidence"),
            "bbox": candidate.get("bbox"),
        }
        for key in (
            "front_edge_receptacle",
            "broad_front_receptacle",
            "visible_occupant_count",
            "distance",
            "ground_distance",
            "height_m",
            "distance_m",
            "bearing_deg",
            "score",
            "actionability_reject_reason",
            "rejection_reasons",
            "region_type",
            "region_area_px",
            "region_area_ratio",
            "floor_level_source",
            "projected_height_warning",
            "floor_plane_residual_m",
            "floor_contact_confidence",
        ):
            if key in candidate:
                summary[key] = candidate.get(key)
        floor_contact = candidate.get("floor_contact_geometry") if isinstance(candidate.get("floor_contact_geometry"), dict) else {}
        if floor_contact:
            summary["floor_contact_geometry"] = {
                "available": floor_contact.get("available"),
                "reason": floor_contact.get("reason"),
                "method": floor_contact.get("method"),
                "contact_floor_like": floor_contact.get("contact_floor_like"),
                "confidence": floor_contact.get("confidence"),
                "floor_plane_residual_m": floor_contact.get("floor_plane_residual_m"),
                "support_height_m": floor_contact.get("support_height_m"),
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
    parser.add_argument(
        "--performance-profile",
        choices=["balanced", "full"],
        default=os.getenv("ROBOT_PERFORMANCE_PROFILE", "balanced"),
        help=(
            "Runtime cost profile. 'balanced' keeps full action safety but "
            "decimates idle object/semantic memory updates; 'full' preserves "
            "the previous every-step memory fusion behavior."
        ),
    )
    parser.add_argument(
        "--object-memory-update-interval",
        type=int,
        default=env_int("ROBOT_OBJECT_MEMORY_UPDATE_INTERVAL", 3),
        help=(
            "In balanced profile, update object memory every N idle exploration "
            "steps. Service targets, held objects, and failures still update every step."
        ),
    )
    parser.add_argument(
        "--semantic-map-update-interval",
        type=int,
        default=env_int("ROBOT_SEMANTIC_MAP_UPDATE_INTERVAL", 3),
        help=(
            "In balanced profile, update semantic frontier scores every N idle "
            "exploration steps. Service targets, held objects, and failures still update every step."
        ),
    )
    parser.add_argument(
        "--log-detail",
        choices=["summary", "full"],
        default=os.getenv("ROBOT_PATROL_LOG_DETAIL", "summary"),
        help=(
            "Detail level for --verbose script_result events. 'summary' avoids "
            "writing full RGB-D/YOLO JSON each step; 'full' keeps the old debug payload."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Include script result summaries in events.")
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
    if args.object_memory_update_interval < 1:
        raise SystemExit("--object-memory-update-interval must be >= 1")
    if args.semantic_map_update_interval < 1:
        raise SystemExit("--semantic-map-update-interval must be >= 1")
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
        performance_profile=str(args.performance_profile),
        object_memory_update_interval=int(args.object_memory_update_interval),
        semantic_map_update_interval=int(args.semantic_map_update_interval),
        log_detail=str(args.log_detail),
        verbose=bool(args.verbose),
        quiet=bool(args.quiet),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    runner = PatrolRunner(config)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
