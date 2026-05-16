"""Response sanitizers for online-safe backend APIs."""

from __future__ import annotations

from typing import Any, Dict


JsonDict = Dict[str, Any]


def _copy_keys(data: JsonDict, keys: list[str]) -> JsonDict:
    return {key: data.get(key) for key in keys if key in data}


def sanitize_move_result(data: JsonDict) -> JsonDict:
    """Remove exact pose from /move's online response."""
    safe = _copy_keys(
        data,
        [
            "status",
            "result_type",
            "action",
            "lastActionSuccess",
            "message",
            "error_message",
            "state_changed",
        ],
    )
    safe.update(
        {
            "schema_version": 2,
            "online_safe": True,
            "observation_contract": "action_feedback_only_v2",
            "excluded_online_fields": [
                "position_before",
                "position_after",
                "rotation_before",
                "rotation_after",
            ],
        }
    )
    return safe


def sanitize_clean_result(data: JsonDict) -> JsonDict:
    """Remove object metadata from /clean's online response.

    The backend may use simulator metadata internally to implement the simulated
    actuator, but the Agent should not receive objectId/object position details.
    """
    safe = _copy_keys(
        data,
        [
            "status",
            "result_type",
            "message",
            "lastActionSuccess",
            "error_message",
        ],
    )
    safe.update(
        {
            "schema_version": 2,
            "online_safe": True,
            "observation_contract": "simulated_clean_actuator_feedback_v2",
            "eval_details_available": True,
            "excluded_online_fields": [
                "target",
                "target_after",
                "candidates",
                "objectId",
                "objectType",
                "position",
                "removed_from_view",
                "removed_from_scene",
            ],
        }
    )
    return safe


def sanitize_service_action_result(data: JsonDict, *, action_name: str) -> JsonDict:
    """Remove simulator object metadata from service action responses."""
    safe = _copy_keys(
        data,
        [
            "status",
            "result_type",
            "message",
            "lastActionSuccess",
            "error_message",
            "holding_object",
            "grounding_policy",
            "visual_grounding_required",
            "visual_candidate_received",
            "visual_candidate_label",
            "placement_verified",
            "object_on_target_receptacle",
            "object_reachable_from_agent",
            "placement_distance_bucket",
            "placement_angle_bucket",
            "visual_receptacle_grounding_passed",
            "visual_receptacle_grounding_result_type",
            "visual_receptacle_target_instance_ratio",
            "visual_box_ambiguous",
        ],
    )
    safe.update(
        {
            "schema_version": 2,
            "online_safe": True,
            "observation_contract": f"{action_name}_feedback_only_v2",
            "eval_details_available": True,
            "excluded_online_fields": [
                "target",
                "target_after",
                "visual_candidate",
                "candidates",
                "receptacle",
                "inventory_objects",
                "objectId",
                "objectType",
                "position",
                "metadata.objects",
            ],
        }
    )
    return safe


def sanitize_inventory_result(data: JsonDict) -> JsonDict:
    """Expose only whether the robot is currently holding something."""
    safe = _copy_keys(
        data,
        [
            "status",
            "result_type",
            "message",
            "holding_object",
            "held_object_count",
        ],
    )
    safe.update(
        {
            "schema_version": 2,
            "online_safe": True,
            "observation_contract": "inventory_state_v2",
            "eval_details_available": True,
            "excluded_online_fields": [
                "inventory_objects",
                "objectId",
                "objectType",
                "position",
                "metadata.objects",
            ],
        }
    )
    return safe
