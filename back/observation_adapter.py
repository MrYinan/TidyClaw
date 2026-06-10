"""Online RGB-D observation adapter for the robot-cleaner backend.

This module is intentionally small and explicit: it is the boundary between
AI2-THOR's rich Event object and the online OpenClaw Agent.  AI2-THOR exposes
metadata and instance masks, but online decision-making only receives
first-person RGB, depth as a sensor frame, camera intrinsics, and action
feedback. Object metadata and object ids remain excluded.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict


JsonDict = Dict[str, Any]


class ObservationAdapter:
    """Create online-safe observations from :class:`RobotEnvironment`.

    Online-safe means the payload can be consumed by Agent/Skill logic during
    patrol execution. It must not contain object metadata, object ids, instance
    masks, exact agent position, or exact agent rotation. Depth is exposed as a
    first-person sensor frame for geometry-only affordance reasoning.
    """

    SCHEMA_VERSION = 3
    OBSERVATION_CONTRACT = "rgbd_action_feedback_v3"

    def __init__(self, env: Any) -> None:
        self.env = env

    def get_rgb_observation(self) -> JsonDict:
        """Return RGB-D image data and last-action feedback."""
        metadata = getattr(self.env, "last_event", None).metadata if getattr(self.env, "last_event", None) else {}
        metadata = metadata if isinstance(metadata, dict) else {}

        base64_img = self.env.get_first_person_view_base64()
        depth_base64 = self.env.get_depth_frame_base64_npy()
        payload = {
            "status": "success",
            "schema_version": self.SCHEMA_VERSION,
            "result_type": "rgbd_observation" if depth_base64 else "rgb_observation",
            "observation_contract": self.OBSERVATION_CONTRACT,
            "online_safe": True,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "image": {
                "encoding": "base64_jpeg",
                "width": getattr(self.env, "width", None),
                "height": getattr(self.env, "height", None),
            },
            "vision_base64": base64_img,
            "camera": self._camera_info(metadata),
            # Keep both nested and flat fields so existing skill code can migrate
            # without breaking. These fields are action feedback, not perception
            # oracle data.
            "last_action_feedback": self._last_action_feedback(metadata),
            "last_action": metadata.get("lastAction"),
            "last_action_success": metadata.get("lastActionSuccess"),
            "last_action_error": metadata.get("errorMessage", ""),
            "excluded_online_fields": [
                "metadata.objects",
                "objectId",
                "objectType",
                "agent.position",
                "agent.rotation",
                "instance_segmentation_frame",
                "instance_masks",
            ],
        }
        if depth_base64:
            payload["depth"] = {
                "encoding": "base64_npy_float32",
                "unit": "meter",
                "width": getattr(self.env, "width", None),
                "height": getattr(self.env, "height", None),
            }
            payload["depth_base64"] = depth_base64
        else:
            payload["depth"] = {
                "available": False,
                "reason": "ai2thor_depth_frame_unavailable",
            }
        return payload

    def _last_action_feedback(self, metadata: JsonDict) -> JsonDict:
        return {
            "action": metadata.get("lastAction"),
            "success": metadata.get("lastActionSuccess"),
            "error_message": metadata.get("errorMessage", ""),
        }

    def _camera_info(self, metadata: JsonDict) -> JsonDict:
        agent = metadata.get("agent") if isinstance(metadata.get("agent"), dict) else {}
        position = agent.get("position") if isinstance(agent.get("position"), dict) else {}
        try:
            horizon = float(agent.get("cameraHorizon", 0.0) or 0.0)
        except (TypeError, ValueError):
            horizon = 0.0
        try:
            camera_height = float(os.getenv("ROBOT_CAMERA_HEIGHT_M", position.get("y", 0.9)))
        except (TypeError, ValueError):
            camera_height = 0.9
        try:
            fov_deg = float(os.getenv("ROBOT_CAMERA_FOV_DEG", "90.0"))
        except (TypeError, ValueError):
            fov_deg = 90.0

        width = float(getattr(self.env, "width", 600) or 600)
        height = float(getattr(self.env, "height", 600) or 600)
        focal_px = width / (2.0 * math.tan(math.radians(max(1.0, min(179.0, fov_deg))) / 2.0))
        return {
            "width": int(width),
            "height": int(height),
            "fov_deg": round(fov_deg, 3),
            "fx": round(focal_px, 3),
            "fy": round(focal_px, 3),
            "cx": round(width / 2.0, 3),
            "cy": round(height / 2.0, 3),
            "camera_horizon_deg": round(horizon, 3),
            "camera_height_m": round(camera_height, 3),
            "coordinate_frame": "camera_relative_x_right_y_height_z_forward",
        }
