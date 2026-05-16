#!/usr/bin/env python3
"""Optional metadata validation helpers for perception-action consistency.

V2 online patrol should not call these helpers by default, because they read
AI2-THOR privileged state through GET /eval/state. Keep this module for V1
baseline comparisons, offline debugging and ablation experiments.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import requests


JsonDict = Dict[str, Any]

DEFAULT_BACKEND_BASE_URL = os.getenv("ROBOT_BACKEND_URL", "http://127.0.0.1:5000")
CLEAN_DISTANCE_THRESHOLD = 0.8
FLOOR_Y_THRESHOLD = 0.35


def normalize_delta(delta: float) -> float:
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def position_hint_from_delta(delta: float) -> str:
    if abs(delta) <= 15.0:
        return "front-center"
    if -60.0 <= delta < -15.0:
        return "front-left"
    if 15.0 < delta <= 60.0:
        return "front-right"
    if delta < -60.0:
        return "left-or-behind"
    return "right-or-behind"


def candidate_reject_reasons(
    *,
    is_floor_level: bool,
    is_near: bool,
    is_front_center: bool,
) -> List[str]:
    reasons: List[str] = []
    if not is_front_center:
        reasons.append("target_not_centered")
    if not is_near:
        reasons.append("target_too_far")
    if not is_floor_level:
        reasons.append("target_not_floor_level")
    return reasons


def backend_candidates_from_state(state: JsonDict) -> List[JsonDict]:
    robot = state.get("robot") if isinstance(state.get("robot"), dict) else {}
    robot_pos = robot.get("position") if isinstance(robot.get("position"), dict) else {}
    robot_rot = robot.get("rotation") if isinstance(robot.get("rotation"), dict) else {}
    rot_y = float(robot_rot.get("y", 0.0) or 0.0)

    candidates: List[JsonDict] = []
    for obj in state.get("visible_trash_candidates") or []:
        if not isinstance(obj, dict):
            continue

        obj_pos = obj.get("position") if isinstance(obj.get("position"), dict) else {}
        dx = float(obj_pos.get("x", 0.0) or 0.0) - float(robot_pos.get("x", 0.0) or 0.0)
        dz = float(obj_pos.get("z", 0.0) or 0.0) - float(robot_pos.get("z", 0.0) or 0.0)
        ground_distance = math.sqrt(dx * dx + dz * dz)
        target_angle = math.degrees(math.atan2(dx, dz))
        angle_delta = normalize_delta(target_angle - rot_y)
        position_hint = position_hint_from_delta(angle_delta)

        y = obj_pos.get("y")
        is_floor_level = y is not None and float(y) <= FLOOR_Y_THRESHOLD
        is_near = ground_distance <= CLEAN_DISTANCE_THRESHOLD
        is_front_center = position_hint == "front-center"
        clean_rule_passed = bool(is_floor_level and is_near and is_front_center)

        candidates.append(
            {
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "visible": bool(obj.get("visible", False)),
                "position": obj_pos,
                "metadata_distance": round(float(obj.get("distance", float("inf"))), 3),
                "distance": round(float(ground_distance), 3),
                "ground_distance": round(float(ground_distance), 3),
                "position_hint": position_hint,
                "angle_delta_deg": round(float(angle_delta), 2),
                "is_floor_level": bool(is_floor_level),
                "is_near": bool(is_near),
                "is_front_center": bool(is_front_center),
                "clean_rule_passed": clean_rule_passed,
                "reject_reasons": candidate_reject_reasons(
                    is_floor_level=is_floor_level,
                    is_near=is_near,
                    is_front_center=is_front_center,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            not bool(item.get("is_front_center")),
            not bool(item.get("is_floor_level")),
            float(item.get("ground_distance", float("inf"))),
        )
    )
    return candidates


def fetch_backend_state(
    *,
    base_url: str = DEFAULT_BACKEND_BASE_URL,
    timeout: int = 5,
) -> Tuple[Optional[JsonDict], Optional[str]]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/eval/state", timeout=timeout)
        try:
            data = response.json()
        except Exception:
            return None, f"invalid_eval_state_response:http_{response.status_code}"
        if response.status_code >= 500:
            return None, f"eval_state_http_{response.status_code}"
        if not isinstance(data, dict):
            return None, "eval_state_not_object"
        if data.get("status") != "success":
            return None, str(data.get("result_type") or data.get("message") or "eval_state_error")
        return data, None
    except requests.exceptions.Timeout:
        return None, "eval_state_timeout"
    except requests.exceptions.ConnectionError:
        return None, "eval_state_unavailable"
    except Exception as exc:
        return None, f"eval_state_error:{type(exc).__name__}"


def infer_validation_reject_type(candidates: List[JsonDict]) -> str:
    if not candidates:
        return "no_backend_visible_trash"

    cleanable = [candidate for candidate in candidates if candidate.get("clean_rule_passed")]
    if cleanable:
        return "backend_cleanable"

    # Prefer actionable geometry reasons over generic "not cleanable".
    for candidate in candidates:
        reasons = list(candidate.get("reject_reasons") or [])
        if (
            candidate.get("is_floor_level")
            and candidate.get("is_front_center")
            and "target_too_far" in reasons
        ):
            return "backend_target_too_far"

    for candidate in candidates:
        reasons = list(candidate.get("reject_reasons") or [])
        if candidate.get("is_floor_level") and "target_not_centered" in reasons:
            return "backend_target_not_centered"

    for candidate in candidates:
        if "target_not_floor_level" in list(candidate.get("reject_reasons") or []):
            return "backend_target_not_floor_level"

    return "backend_no_cleanable_target"


def validate_clean_target_with_backend(
    *,
    base_url: str = DEFAULT_BACKEND_BASE_URL,
    timeout: int = 5,
) -> JsonDict:
    state, error = fetch_backend_state(base_url=base_url, timeout=timeout)
    if error or state is None:
        return {
            "status": "error",
            "result_type": "backend_validation_unavailable",
            "clean_allowed": False,
            "message": error or "backend state unavailable",
            "backend_candidates": [],
            "best_backend_candidate": None,
        }

    candidates = backend_candidates_from_state(state)
    cleanable = [candidate for candidate in candidates if candidate.get("clean_rule_passed")]
    if cleanable:
        return {
            "status": "success",
            "result_type": "backend_cleanable_target",
            "clean_allowed": True,
            "message": "backend confirms a front-center near floor target",
            "backend_candidates": candidates,
            "best_backend_candidate": cleanable[0],
        }

    result_type = infer_validation_reject_type(candidates)
    best = candidates[0] if candidates else None
    return {
        "status": "success",
        "result_type": result_type,
        "clean_allowed": False,
        "message": "backend does not confirm a V1-cleanable target",
        "backend_candidates": candidates,
        "best_backend_candidate": best,
    }


def short_backend_candidate(candidate: Optional[JsonDict]) -> Optional[JsonDict]:
    if not candidate:
        return None
    return {
        "objectType": candidate.get("objectType"),
        "position_hint": candidate.get("position_hint"),
        "ground_distance": candidate.get("ground_distance"),
        "angle_delta_deg": candidate.get("angle_delta_deg"),
        "is_floor_level": candidate.get("is_floor_level"),
        "is_near": candidate.get("is_near"),
        "is_front_center": candidate.get("is_front_center"),
        "clean_rule_passed": candidate.get("clean_rule_passed"),
        "reject_reasons": candidate.get("reject_reasons"),
    }
