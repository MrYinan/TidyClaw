"""Online RGB-only observation adapter for the robot-cleaner backend.

This module is intentionally small and explicit: it is the boundary between
AI2-THOR's rich Event object and the online OpenClaw Agent.  AI2-THOR exposes
metadata, instance masks, depth and exact pose, but V2 online decision-making
must only receive first-person RGB plus action feedback.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


JsonDict = Dict[str, Any]


class ObservationAdapter:
    """Create online-safe observations from :class:`RobotEnvironment`.

    Online-safe means the payload can be consumed by Agent/Skill logic during
    patrol execution. It must not contain object metadata, object ids, instance
    masks, depth, exact agent position, exact rotation, or camera horizon.
    """

    SCHEMA_VERSION = 2
    OBSERVATION_CONTRACT = "rgb_only_action_feedback_v2"

    def __init__(self, env: Any) -> None:
        self.env = env

    def get_rgb_observation(self) -> JsonDict:
        """Return RGB image and last-action feedback only."""
        metadata = getattr(self.env, "last_event", None).metadata if getattr(self.env, "last_event", None) else {}
        metadata = metadata if isinstance(metadata, dict) else {}

        base64_img = self.env.get_first_person_view_base64()
        return {
            "status": "success",
            "schema_version": self.SCHEMA_VERSION,
            "result_type": "rgb_observation",
            "observation_contract": self.OBSERVATION_CONTRACT,
            "online_safe": True,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "image": {
                "encoding": "base64_jpeg",
                "width": getattr(self.env, "width", None),
                "height": getattr(self.env, "height", None),
            },
            "vision_base64": base64_img,
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
                "cameraHorizon",
                "depth_frame",
                "instance_segmentation_frame",
                "instance_masks",
            ],
        }

    def _last_action_feedback(self, metadata: JsonDict) -> JsonDict:
        return {
            "action": metadata.get("lastAction"),
            "success": metadata.get("lastActionSuccess"),
            "error_message": metadata.get("errorMessage", ""),
        }
