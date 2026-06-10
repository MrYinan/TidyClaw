#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///

"""Capture the current RGB/RGB-D observation from the backend.

The online chain receives local file paths rather than heavy base64 payloads:
`image_path` for RGB and, when available, `depth_path` for a float32 `.npy`
depth frame in meters. Object metadata and instance masks are still excluded.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests


JsonDict = Dict[str, Any]
DEFAULT_BACKEND_URL = os.getenv("ROBOT_BACKEND_URL", "http://127.0.0.1:5000")
DEFAULT_SAVE_DIR = os.getenv("ROBOT_VISION_SAVE_DIR", r"D:\photos")


def strip_data_url_prefix(base64_str: str) -> str:
    if "," in base64_str:
        return base64_str.split(",", 1)[1]
    return base64_str


def write_frame_bytes(
    frame_bytes: bytes,
    preferred_dir: str,
    *,
    stable_name: str,
    timestamp_prefix: str,
    suffix: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidates = []

    preferred = Path(preferred_dir)
    candidates.append(preferred / stable_name)
    candidates.append(preferred / f"{timestamp_prefix}_{timestamp}{suffix}")

    repo_root = Path(__file__).resolve().parents[3]
    fallback_dir = repo_root / "memory" / "vision-captures"
    candidates.append(fallback_dir / f"{timestamp_prefix}_{timestamp}{suffix}")

    last_error: Exception | None = None
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                f.write(frame_bytes)
            return str(path)
        except OSError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"无法保存观测帧: {last_error}")


def write_image_bytes(image_bytes: bytes, preferred_dir: str) -> str:
    return write_frame_bytes(
        image_bytes,
        preferred_dir,
        stable_name="openclaw_robot_vision.jpg",
        timestamp_prefix="openclaw_robot_vision",
        suffix=".jpg",
    )


def write_depth_bytes(depth_bytes: bytes, preferred_dir: str) -> str:
    return write_frame_bytes(
        depth_bytes,
        preferred_dir,
        stable_name="openclaw_robot_depth.npy",
        timestamp_prefix="openclaw_robot_depth",
        suffix=".npy",
    )


def online_safe_payload(data: JsonDict, image_path: str, depth_path: str | None = None) -> JsonDict:
    """Drop heavy frame data and return local paths for downstream perception."""
    feedback = data.get("last_action_feedback") if isinstance(data.get("last_action_feedback"), dict) else {}
    payload: JsonDict = {
        "status": data.get("status", "success"),
        "schema_version": data.get("schema_version", 3),
        "result_type": "vision_captured_rgbd" if depth_path else "vision_captured_rgb_only",
        "observation_contract": data.get(
            "observation_contract",
            "rgbd_action_feedback_v3" if depth_path else "rgb_only_action_feedback_v2",
        ),
        "online_safe": True,
        "image_path": image_path,
        "depth": data.get("depth") if isinstance(data.get("depth"), dict) else None,
        "camera": data.get("camera") if isinstance(data.get("camera"), dict) else None,
        "scene": data.get("scene"),
        "last_action_feedback": {
            "action": feedback.get("action", data.get("last_action")),
            "success": feedback.get("success", data.get("last_action_success")),
            "error_message": feedback.get("error_message", data.get("last_action_error", "")),
        },
        "last_action": data.get("last_action"),
        "last_action_success": data.get("last_action_success"),
        "last_action_error": data.get("last_action_error", ""),
    }
    if depth_path:
        payload["depth_path"] = depth_path
    return payload


def main() -> None:
    try:
        base_url = DEFAULT_BACKEND_URL.rstrip("/")
        response = requests.get(f"{base_url}/observation", timeout=10)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError("/observation response is not a JSON object")

        if data.get("status") != "success":
            print(json.dumps(data, ensure_ascii=False))
            sys.exit(1)

        base64_str = str(data.get("vision_base64") or "")
        if not base64_str:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "result_type": "error_empty_observation_image",
                        "message": "/observation did not return vision_base64.",
                    },
                    ensure_ascii=False,
                )
            )
            sys.exit(1)

        image_bytes = base64.b64decode(strip_data_url_prefix(base64_str))
        image_path = write_image_bytes(image_bytes, DEFAULT_SAVE_DIR)

        depth_path = None
        depth_base64 = str(data.get("depth_base64") or "")
        if depth_base64:
            depth_bytes = base64.b64decode(strip_data_url_prefix(depth_base64))
            depth_path = write_depth_bytes(depth_bytes, DEFAULT_SAVE_DIR)

        print(json.dumps(online_safe_payload(data, image_path, depth_path), ensure_ascii=False))

    except requests.exceptions.ConnectionError as exc:
        print(json.dumps({"status": "error", "result_type": "error_vision_service_unavailable", "message": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    except requests.exceptions.Timeout as exc:
        print(json.dumps({"status": "error", "result_type": "error_vision_timeout", "message": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"status": "error", "result_type": "error_vision_unknown", "message": f"视觉抓取失败: {str(exc)}"}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
