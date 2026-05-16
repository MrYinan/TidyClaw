#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///

"""Capture the current RGB-only observation from the backend.

V2 contract:
- call GET /observation, not /vision;
- save the RGB frame locally and pass only image_path/action feedback onward;
- do not expose AI2-THOR pose, object metadata, depth, or segmentation to the
  online Agent chain.
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


def write_image_bytes(image_bytes: bytes, preferred_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidates = []

    preferred = Path(preferred_dir)
    candidates.append(preferred / "openclaw_robot_vision.jpg")
    candidates.append(preferred / f"openclaw_robot_vision_{timestamp}.jpg")

    repo_root = Path(__file__).resolve().parents[3]
    fallback_dir = repo_root / "memory" / "vision-captures"
    candidates.append(fallback_dir / f"openclaw_robot_vision_{timestamp}.jpg")

    last_error: Exception | None = None
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                f.write(image_bytes)
            return str(path)
        except OSError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"无法保存视觉图像: {last_error}")


def online_safe_payload(data: JsonDict, image_path: str) -> JsonDict:
    """Drop heavy image data and enforce the RGB-only payload shape."""
    feedback = data.get("last_action_feedback") if isinstance(data.get("last_action_feedback"), dict) else {}
    return {
        "status": data.get("status", "success"),
        "schema_version": data.get("schema_version", 2),
        "result_type": "vision_captured_rgb_only",
        "observation_contract": data.get("observation_contract", "rgb_only_action_feedback_v2"),
        "online_safe": True,
        "image_path": image_path,
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
                        "message": "/observation 没有返回 vision_base64。",
                    },
                    ensure_ascii=False,
                )
            )
            sys.exit(1)

        image_bytes = base64.b64decode(strip_data_url_prefix(base64_str))
        image_path = write_image_bytes(image_bytes, DEFAULT_SAVE_DIR)
        print(json.dumps(online_safe_payload(data, image_path), ensure_ascii=False))

    except requests.exceptions.ConnectionError as e:
        print(json.dumps({"status": "error", "result_type": "error_vision_service_unavailable", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print(json.dumps({"status": "error", "result_type": "error_vision_timeout", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "result_type": "error_vision_unknown", "message": f"视觉抓取失败: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
