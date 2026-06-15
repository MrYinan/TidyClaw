#!/usr/bin/env python3
"""Build a bounded public decision context for the household service agent.

The builder is intentionally read-only. It compresses current perception,
task state, navigation memory, and object memory into a small JSON packet that
an OpenClaw/LLM agent can use to choose one option id. The full maps and raw
memory stay in the backend files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scripts.exploration_context import build_exploration_context
except ImportError:  # pragma: no cover - direct script execution
    from exploration_context import build_exploration_context

try:
    from scripts.explore_planner import build_explore_plan
except ImportError:  # pragma: no cover - direct script execution
    from explore_planner import build_explore_plan

try:
    from scripts.map_backend import BackendUnavailableError, load_map_backend
except ImportError:  # pragma: no cover - direct script execution
    from map_backend import BackendUnavailableError, load_map_backend

try:
    from scripts.coverage_waypoint_state import normalize_coverage_waypoint_state
    from scripts.inspection_waypoints import build_inspection_waypoints
except ImportError:  # pragma: no cover - direct script execution
    from coverage_waypoint_state import normalize_coverage_waypoint_state
    from inspection_waypoints import build_inspection_waypoints

try:
    from scripts.navigation_core import build_navigation_core_state
except ImportError:  # pragma: no cover - direct script execution
    from navigation_core import build_navigation_core_state

try:
    from scripts.position_map_core import (
        CELL_FREE,
        CELL_UNKNOWN,
        four_neighbors,
        heading_between,
        left_heading,
        neighbor_cell,
        parse_cell,
        right_heading,
    )
except ImportError:  # pragma: no cover - direct script execution
    from position_map_core import (
        CELL_FREE,
        CELL_UNKNOWN,
        four_neighbors,
        heading_between,
        left_heading,
        neighbor_cell,
        parse_cell,
        right_heading,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_PERCEPTION_JSON = DEFAULT_MEMORY_DIR / "yolo-current-rgbd.json"
PLACE_PRECHECK_CACHE_NAME = "place-precheck-cache.json"

DECISION_CONTEXT_SCHEMA = "robot_cleaner_decision_context_v1"
WORKLIST_SCHEMA = "robot_cleaner_worklist_v1"
OPTION_SET_SCHEMA = "robot_cleaner_option_set_v1"
PLACE_PRECHECK_CACHE_SCHEMA = "robot_cleaner_place_precheck_cache_v1"

BODY_MOVE_ACTIONS = (
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
)
TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
CAMERA_ACTIONS = ("LookUp", "LookDown")
DEFAULT_FORWARD_MIN_OBSERVED_RATIO = 0.30
DEFAULT_LATERAL_MIN_OBSERVED_RATIO = 0.50
DEFAULT_REAR_MIN_OBSERVED_RATIO = 0.60
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
    "APPROACH_RECEPTACLE",
    "ALIGN_RECEPTACLE",
    "PLACE_OBJECT",
    "VERIFY_TASK_DONE",
}
NON_ACTIONABLE_TRACK_STATES = {"placed", "placed_closed", "skipped", "stale", "blocked"}
DURABLE_STATE_SOURCES = {
    "mission_state",
    "patrol_state",
    "room_state",
    "service_task_state",
    "object_memory",
    "position_map",
    "semantic_map",
    "global_plan",
}
VOLATILE_OBSERVATION_SOURCES = {"perception", "navigation_costmap"}
SURFACE_REGION_SOURCES = {
    "pointcloud_plane",
    "pointcloud_plane_completion",
    "pointcloud_plane_grid_completion",
    "depth_region_geometry",
}

GOAL_LEVEL_OPTION_KINDS = {
    "service_action",
    "pursue_pickup_target",
    "place_precheck",
    "clean_action",
    "orient_waypoint_floor_scan",
    "explore_inspection_waypoint",
    "continue_active_waypoint_goal",
    "explore_frontier_cluster",
    "explore_route_step",
    "explore_frontier",
    "explore_waypoint",
    "recovery_action",
}
SUPPORT_OPTION_KINDS = {
    "perception",
    "completion_probe",
}

FORBIDDEN_AGENT_VIEW_KEYS = {
    "objectid",
    "objectids",
    "metadata",
    "scene_objects",
    "sceneobject",
    "sceneobjects",
    "private_manifest",
    "private_evaluation",
    "acceptable_destination_sets",
    "generated_mess_set",
    "is_misplaced",
    "simulator_truth",
}


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class LoadedJson:
    path: Path
    data: JsonDict
    info: JsonDict


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_print(data: JsonDict, *, compact: bool = False) -> None:
    if compact:
        print(json.dumps(data, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def read_json(path: Path, *, root: Path = REPO_ROOT) -> LoadedJson:
    info: JsonDict = {
        "path": display_path(path, root=root),
        "exists": path.exists(),
        "loaded": False,
    }
    if not path.exists():
        info["error"] = "missing_file"
        return LoadedJson(path=path, data={}, info=info)
    try:
        stat = path.stat()
        info["bytes"] = stat.st_size
        info["last_modified"] = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
            timespec="seconds"
        )
    except OSError as exc:
        info["stat_error"] = str(exc)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8-sig")
            data = json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive path
            info["error"] = f"json_load_failed:{type(exc).__name__}:{exc}"
            return LoadedJson(path=path, data={}, info=info)
    except Exception as exc:
        info["error"] = f"json_load_failed:{type(exc).__name__}:{exc}"
        return LoadedJson(path=path, data={}, info=info)
    if not isinstance(data, dict):
        info["error"] = "json_root_not_object"
        return LoadedJson(path=path, data={}, info=info)
    info["loaded"] = True
    return LoadedJson(path=path, data=data, info=info)


def display_path(path: Path, *, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def number_or_none(value: Any, *, digits: int = 4) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def first_meaningful_text(*values: Any, unknown_values: Iterable[str] = ("unknown", "none", "null")) -> str:
    unknown = {str(item).lower() for item in unknown_values}
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in unknown:
            return text
    return ""


def compact_bbox(value: Any) -> JsonDict | None:
    box = as_dict(value)
    if not box:
        return None
    keys = ("x", "y", "w", "h")
    compact = {key: number_or_none(box.get(key), digits=2) for key in keys}
    return {key: value for key, value in compact.items() if value is not None} or None


def compact_geometry(source: JsonDict) -> JsonDict:
    geometry = as_dict(source.get("geometry"))
    depth = as_dict(source.get("depth"))
    center_3d = as_dict(source.get("center_3d")) or as_dict(depth.get("center_3d"))
    result: JsonDict = {}
    mapping = {
        "bearing_deg": first_present(geometry.get("bearing_deg"), source.get("bearing_deg")),
        "distance_m": first_present(
            source.get("distance"),
            geometry.get("distance_m"),
            depth.get("median_m"),
        ),
        "ground_distance_m": first_present(
            source.get("ground_distance"),
            geometry.get("ground_distance_m"),
            depth.get("ground_distance_m"),
            center_3d.get("ground_distance_m"),
        ),
        "ground_forward_m": first_present(depth.get("ground_forward_m"), center_3d.get("ground_forward_m")),
        "height_m": first_present(source.get("height"), geometry.get("height_m"), source.get("height_m")),
        "cx_ratio": geometry.get("cx_ratio"),
        "cy_ratio": geometry.get("cy_ratio"),
        "bottom_y_ratio": first_present(geometry.get("bottom_y_ratio"), source.get("bottom_y_ratio")),
        "area_ratio": geometry.get("area_ratio"),
    }
    for key, value in mapping.items():
        number = number_or_none(value)
        if number is not None:
            result[key] = number
    bbox = compact_bbox(source.get("bbox"))
    if bbox:
        result["bbox"] = bbox
    return result


def compact_executor_checks(value: Any) -> JsonDict:
    checks = as_dict(value)
    if not checks:
        return {}
    keep = (
        "precheck_supported",
        "precheck_ok",
        "reason",
        "placement_execution_mode",
        "executor_action_hint",
        "requires_exact_target_execution",
        "executor_precheck_required",
    )
    result = {key: checks.get(key) for key in keep if checks.get(key) not in (None, "", [])}
    return result


def compact_point2d(value: Any) -> JsonDict:
    point = as_dict(value)
    result = {
        "x": number_or_none(point.get("x"), digits=3),
        "y": number_or_none(point.get("y"), digits=3),
    }
    return {key: item for key, item in result.items() if item is not None}


def compact_point3d(value: Any) -> JsonDict:
    point = as_dict(value)
    result = {}
    for key in ("x", "y", "z", "ground_distance_m", "ground_forward_m"):
        number = number_or_none(point.get(key), digits=4)
        if number is not None:
            result[key] = number
    return result


def compact_number_sequence(value: Any, *, limit: int = 8, digits: int = 4) -> list[int | float]:
    result: list[int | float] = []
    for item in as_list(value)[:limit]:
        number = number_or_none(item, digits=digits)
        if number is not None:
            result.append(number)
    return result


def compact_scalar_mapping(value: Any, *, keys: Iterable[str]) -> JsonDict:
    source = as_dict(value)
    result: JsonDict = {}
    for key in keys:
        item = source.get(key)
        if isinstance(item, bool):
            result[key] = item
        elif isinstance(item, str) and item:
            result[key] = item
        else:
            number = number_or_none(item)
            if number is not None:
                result[key] = number
    return result


def compact_placement_point(value: Any) -> JsonDict:
    point = as_dict(value)
    result = compact_point2d(point)
    center_3d = compact_point3d(point.get("center_3d"))
    if center_3d:
        result["center_3d"] = center_3d
    for key in ("rank", "clearance_m"):
        number = number_or_none(point.get(key))
        if number is not None:
            result[key] = number
    grid_cell = compact_number_sequence(point.get("grid_cell"), limit=3, digits=0)
    if grid_cell:
        result["grid_cell"] = grid_cell
    return clean_empty(result)


def compact_placement_points(value: Any, *, limit: int = 8) -> list[JsonDict]:
    result = []
    for item in as_list(value)[:limit]:
        point = compact_placement_point(item)
        if point:
            result.append(point)
    return result


def compact_placement_contract(value: Any) -> JsonDict:
    return compact_scalar_mapping(
        value,
        keys=(
            "version",
            "clearance_owner",
            "surface_source",
            "target_coordinate_frame",
            "execution_target",
            "requires_exact_target_execution",
            "grid_occupancy_clear",
            "grid_edge_eroded",
            "camera_height_m",
            "held_footprint_radius_m",
            "blocker_dilate_m",
            "edge_margin_m",
            "placement_clearance_m",
            "placement_point_count",
        ),
    )


def compact_free_space_completion(value: Any) -> JsonDict:
    source = as_dict(value)
    result = compact_scalar_mapping(
        source,
        keys=(
            "mode",
            "method",
            "component_count",
            "component_id",
            "component_area_m2",
            "component_width_m",
            "component_depth_m",
            "component_point_count",
            "grid_resolution_m",
            "blocker_dilate_m",
            "edge_margin_m",
            "held_footprint_radius_m",
            "placement_clearance_m",
            "placement_point_count",
            "selected_center_clearance_m",
            "source_surface_id",
        ),
    )
    selected_center_3d = compact_point3d(source.get("selected_center_3d"))
    if selected_center_3d:
        result["selected_center_3d"] = selected_center_3d
    selected_center_grid = compact_number_sequence(source.get("selected_center_grid"), limit=3, digits=0)
    if selected_center_grid:
        result["selected_center_grid"] = selected_center_grid
    selected_center_uv = compact_number_sequence(source.get("selected_center_uv"), limit=3)
    if selected_center_uv:
        result["selected_center_uv"] = selected_center_uv
    counts = compact_scalar_mapping(
        source.get("occupancy_source_counts"),
        keys=("bbox_fallback", "depth_points"),
    )
    if counts:
        result["occupancy_source_counts"] = counts
    return clean_empty(result)


def compact_occupancy_checks(value: Any) -> JsonDict:
    source = as_dict(value)
    result = compact_scalar_mapping(
        source,
        keys=(
            "blocked",
            "free_space_grid_completion",
            "held_object_ignored_as_blocker",
        ),
    )
    blocked_by = [str(item) for item in as_list(source.get("blocked_by")) if str(item)]
    if blocked_by:
        result["blocked_by"] = blocked_by[:8]
    counts = compact_scalar_mapping(
        source.get("occupancy_source_counts"),
        keys=("bbox_fallback", "depth_points"),
    )
    if counts:
        result["occupancy_source_counts"] = counts
    return clean_empty(result)


def compact_public_object_candidate(value: Any) -> JsonDict:
    item = as_dict(value)
    if not item:
        return {}
    result: JsonDict = {
        "label": item.get("label") or "",
        "raw_label": item.get("raw_label") or "",
        "task_class": item.get("task_semantic_class") or item.get("task_class") or "",
        "confidence": number_or_none(item.get("confidence")),
    }
    bbox = compact_bbox(item.get("bbox"))
    if bbox:
        result["bbox"] = bbox
    center = compact_point2d(item.get("center"))
    if center:
        result["center"] = center
    geometry = compact_geometry(item)
    if geometry:
        result["geometry"] = geometry
    return clean_empty(result)


def compact_public_object_candidates(value: Any, *, limit: int) -> list[JsonDict]:
    result = []
    for item in as_list(value)[:limit]:
        candidate = compact_public_object_candidate(item)
        if candidate:
            result.append(candidate)
    return result


def compact_candidate(candidate: Any, *, source: str, index: int) -> JsonDict | None:
    item = as_dict(candidate)
    if not item:
        return None
    candidate_id = first_present(
        item.get("id"),
        item.get("candidate_id"),
        item.get("track_id"),
        item.get("surface_candidate_id"),
        item.get("candidate_signature"),
        f"{source}:{index}",
    )
    result: JsonDict = {
        "candidate_id": str(candidate_id),
        "source": source,
        "label": item.get("label") or "",
        "raw_label": item.get("raw_label") or "",
        "task_class": item.get("task_semantic_class") or item.get("task_class") or "",
        "confidence": number_or_none(item.get("confidence")),
        "position_hint": item.get("position_hint") or "",
        "surface_hint": item.get("surface_hint") or "",
    }
    for key in (
        "track_id",
        "candidate_signature",
        "surface_candidate_id",
        "surface_region_id",
        "surface_candidate_source",
        "region_type",
        "parent_object",
        "parent_label",
    ):
        if item.get(key):
            result[key] = item[key]
    actionability: JsonDict = {}
    for key in (
        "reachable",
        "blocked",
        "pickup_now",
        "place_now",
        "cleanable_now",
        "needs_alignment",
        "needs_approach",
        "obstacle_risk",
        "is_floor_level",
        "is_support_surface",
        "affordance_ready",
        "visual_place_ready",
        "final_place_ready",
        "failed_recently",
        "visual_box_ambiguous",
        "broad_front_receptacle",
        "front_edge_receptacle",
    ):
        value = bool_or_none(item.get(key))
        if value is not None:
            actionability[key] = value
    if actionability:
        result["actionability"] = actionability
    geometry = compact_geometry(item)
    if geometry:
        result["geometry"] = geometry
    executor_checks = compact_executor_checks(item.get("executor_checks"))
    if executor_checks:
        result["executor_checks"] = executor_checks
    for key, compacted in (
        ("interaction_point", compact_point2d(item.get("interaction_point"))),
        ("center_3d", compact_point3d(item.get("center_3d"))),
        ("parent_bbox", compact_bbox(item.get("parent_bbox"))),
        ("region_bbox", compact_bbox(item.get("region_bbox"))),
        (
            "geometry_checks",
            compact_scalar_mapping(
                item.get("geometry_checks"),
                keys=as_dict(item.get("geometry_checks")).keys(),
            ),
        ),
        ("occupancy_checks", compact_occupancy_checks(item.get("occupancy_checks"))),
        (
            "memory_checks",
            compact_scalar_mapping(
                item.get("memory_checks"),
                keys=("failed_recently", "cooldown_remaining", "failure_count"),
            ),
        ),
        ("free_space_completion", compact_free_space_completion(item.get("free_space_completion"))),
        ("placement_safety_contract", compact_placement_contract(item.get("placement_safety_contract"))),
    ):
        if compacted:
            result[key] = compacted
    placement_points = compact_placement_points(item.get("placement_points"), limit=8)
    if placement_points:
        result["placement_points"] = placement_points
        result["placement_point_count"] = len(placement_points)
    visible_occupants = compact_public_object_candidates(item.get("visible_occupants"), limit=8)
    if visible_occupants:
        result["visible_occupants"] = visible_occupants
    avoidance = compact_public_object_candidates(item.get("placement_avoidance_candidates"), limit=16)
    if avoidance:
        result["placement_avoidance_candidates"] = avoidance
    rejection_reasons = []
    for key in (
        "rejection_reasons",
        "goal_rejection_reasons",
        "action_rejection_reasons",
        "pickup_goal_rejection_reasons",
        "pickup_action_rejection_reasons",
    ):
        rejection_reasons.extend(str(reason) for reason in as_list(item.get(key)) if reason)
    if rejection_reasons:
        result["rejection_reasons"] = sorted(set(rejection_reasons))[:6]
    return clean_empty(result)


def unique_candidates(candidates: Iterable[JsonDict | None], *, limit: int) -> list[JsonDict]:
    seen: set[str] = set()
    result: list[JsonDict] = []
    for candidate in candidates:
        if not candidate:
            continue
        geometry = as_dict(candidate.get("geometry"))
        bbox = as_dict(geometry.get("bbox"))
        fallback_key = "|".join(
            str(part)
            for part in (
                candidate.get("label"),
                candidate.get("raw_label"),
                candidate.get("surface_hint"),
                bbox.get("x"),
                bbox.get("y"),
                bbox.get("w"),
                bbox.get("h"),
                geometry.get("distance_m"),
                geometry.get("bearing_deg"),
            )
        )
        key = str(
            first_present(
                candidate.get("track_id"),
                candidate.get("surface_candidate_id"),
                candidate.get("candidate_signature"),
                candidate.get("candidate_id"),
                fallback_key,
            )
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) >= limit:
            break
    return result


def track_values(object_memory: JsonDict) -> list[JsonDict]:
    tracks = object_memory.get("tracks")
    if isinstance(tracks, dict):
        return [dict(value) for value in tracks.values() if isinstance(value, dict)]
    if isinstance(tracks, list):
        return [dict(value) for value in tracks if isinstance(value, dict)]
    return []


def track_rank(track: JsonDict) -> tuple[float, float, float, float]:
    interaction = as_dict(track.get("interaction"))
    status = str(track.get("status") or "")
    status_bonus = {
        "pending": 5,
        "active": 5,
        "unpicked": 4,
        "tracked": 4,
        "held": 3,
        "stale": 1,
        "placed": -4,
        "placed_closed": -4,
        "skipped": -5,
        "blocked": -5,
    }.get(status, 2)
    ready_bonus = 0
    for key in ("pickup_action_ready", "pickup_goal_eligible", "place_ready", "pickup_ready"):
        if interaction.get(key) is True:
            ready_bonus += 1
    confidence = float(first_present(track.get("track_score"), track.get("confidence"), 0.0) or 0.0)
    staleness = float(track.get("staleness") or 0.0)
    last_seen = float(track.get("last_seen_step") or 0.0)
    return (status_bonus + ready_bonus, confidence, last_seen, -staleness)


def compact_track(track: JsonDict) -> JsonDict:
    interaction = as_dict(track.get("interaction"))
    last_observation = as_dict(track.get("last_observation"))
    estimated_location = as_dict(track.get("estimated_location"))
    viewpoint = as_dict(track.get("viewpoint"))
    result: JsonDict = {
        "track_id": track.get("track_id") or "",
        "label": track.get("label") or "",
        "raw_labels": as_list(track.get("raw_labels"))[:3],
        "label_family": track.get("label_family") or "",
        "task_class": track.get("task_class") or "",
        "status": track.get("status") or "",
        "confidence": number_or_none(first_present(track.get("track_score"), track.get("confidence"))),
        "staleness": number_or_none(track.get("staleness")),
        "last_seen_step": number_or_none(track.get("last_seen_step")),
        "seen_count": number_or_none(track.get("seen_count")),
    }
    observation = compact_geometry(last_observation)
    if last_observation.get("position_hint"):
        observation["position_hint"] = last_observation.get("position_hint")
    if last_observation.get("surface_hint"):
        observation["surface_hint"] = last_observation.get("surface_hint")
    if observation:
        result["last_observation"] = observation
    if estimated_location.get("estimated_object_cell"):
        result["estimated_cell"] = estimated_location.get("estimated_object_cell")
        result["location_uncertainty_cells"] = number_or_none(
            estimated_location.get("uncertainty_cells")
        )
    if viewpoint:
        result["recommended_viewpoint"] = clean_empty(
            {
                "cell": viewpoint.get("recommended_view_cell"),
                "heading": viewpoint.get("recommended_heading"),
                "standoff_distance_m": number_or_none(viewpoint.get("standoff_distance_m")),
                "reason": viewpoint.get("reason"),
            }
        )
    actionability = clean_empty(
        {
            "pickup_ready": bool_or_none(interaction.get("pickup_ready")),
            "place_ready": bool_or_none(interaction.get("place_ready")),
            "pickup_goal_eligible": bool_or_none(interaction.get("pickup_goal_eligible")),
            "pickup_action_ready": bool_or_none(interaction.get("pickup_action_ready")),
            "last_surface_id": interaction.get("last_surface_id"),
            "failure_count": number_or_none(interaction.get("failure_count")),
            "cooldown_until_step": number_or_none(interaction.get("cooldown_until_step")),
        }
    )
    if actionability:
        result["actionability"] = actionability
    reasons = []
    for key in ("pickup_goal_rejection_reasons", "pickup_action_rejection_reasons"):
        reasons.extend(str(reason) for reason in as_list(interaction.get(key)) if reason)
    if reasons:
        result["rejection_reasons"] = sorted(set(reasons))[:6]
    return clean_empty(result)


def build_memory_track_lists(object_memory: JsonDict, *, limit: int) -> JsonDict:
    tracks = track_values(object_memory)

    def is_pickup(track: JsonDict) -> bool:
        task_class = str(track.get("task_class") or "")
        label_family = str(track.get("label_family") or "")
        status = str(track.get("status") or "")
        return (
            task_class == "pickup_target"
            or label_family in {"pickup_target", "food", "dish", "book", "electronics"}
        ) and status not in {"placed", "placed_closed", "skipped"}

    def is_receptacle(track: JsonDict) -> bool:
        task_class = str(track.get("task_class") or "")
        label_family = str(track.get("label_family") or "")
        return task_class in {"place_receptacle", "surface_target"} or label_family == "receptacle"

    pickup_tracks = sorted((track for track in tracks if is_pickup(track)), key=track_rank, reverse=True)
    receptacle_tracks = sorted(
        (track for track in tracks if is_receptacle(track)), key=track_rank, reverse=True
    )
    return {
        "pickup_targets": [compact_track(track) for track in pickup_tracks[:limit]],
        "receptacles": [compact_track(track) for track in receptacle_tracks[:limit]],
        "stats": clean_empty(as_dict(object_memory.get("stats"))),
    }


def build_worklist(perception: JsonDict, object_memory: JsonDict, service_state: JsonDict, *, limit: int) -> JsonDict:
    pickup_sources: list[JsonDict | None] = []
    receptacle_sources: list[JsonDict | None] = []
    surface_sources: list[JsonDict | None] = []
    clean_sources: list[JsonDict | None] = []
    obstacle_sources: list[JsonDict | None] = []

    pickup_sources.append(compact_candidate(perception.get("best_pickup_candidate"), source="best_pickup_candidate", index=0))
    for index, item in enumerate(as_list(perception.get("service_candidates"))):
        candidate = compact_candidate(item, source="service_candidates", index=index)
        if not candidate:
            continue
        task_class = str(candidate.get("task_class") or "")
        if task_class == "pickup_target":
            pickup_sources.append(candidate)
        elif task_class == "place_receptacle":
            receptacle_sources.append(candidate)
        elif task_class == "cleanable_object":
            clean_sources.append(candidate)
        elif task_class == "obstacle":
            obstacle_sources.append(candidate)

    receptacle_sources.append(
        compact_candidate(perception.get("best_receptacle_candidate"), source="best_receptacle_candidate", index=0)
    )
    for index, item in enumerate(as_list(perception.get("receptacle_candidates"))):
        receptacle_sources.append(compact_candidate(item, source="receptacle_candidates", index=index))

    for source_name in (
        "best_surface_candidate",
        "best_place_affordance",
        "visual_ready_surface_regions",
        "place_affordance_candidates",
        "surface_candidates",
    ):
        for index, item in enumerate(as_list(perception.get(source_name))):
            surface_sources.append(compact_candidate(item, source=source_name, index=index))

    for index, item in enumerate(as_list(perception.get("trash_candidates"))):
        clean_sources.append(compact_candidate(item, source="trash_candidates", index=index))

    obstacle_sources.append(
        compact_candidate(perception.get("best_obstacle_candidate"), source="best_obstacle_candidate", index=0)
    )
    for index, item in enumerate(as_list(perception.get("top_obstacle_candidates"))):
        obstacle_sources.append(compact_candidate(item, source="top_obstacle_candidates", index=index))

    memory_tracks = build_memory_track_lists(object_memory, limit=limit)
    worklist = {
        "schema": WORKLIST_SCHEMA,
        "held_object": build_held_object_context(service_state, perception),
        "current_view": {
            "pickup_candidates": unique_candidates(pickup_sources, limit=limit),
            "receptacle_candidates": unique_candidates(receptacle_sources, limit=limit),
            "surface_candidates": unique_candidates(surface_sources, limit=limit),
            "cleanable_candidates": unique_candidates(clean_sources, limit=limit),
            "obstacles": unique_candidates(obstacle_sources, limit=limit),
        },
        "memory": memory_tracks,
        "recently_placed": clean_empty(
            {
                "labels": service_state.get("recently_placed_labels") or {},
                "tracks": service_state.get("recently_placed_tracks") or {},
            }
        ),
    }
    return clean_empty(worklist)


def build_held_object_context(service_state: JsonDict, perception: JsonDict) -> JsonDict:
    if "holding_object" in service_state:
        holding = bool(service_state.get("holding_object"))
        return clean_empty(
            {
                "holding_object": holding,
                "labels": service_state.get("held_object_labels") if holding else [],
                "family": first_meaningful_text(service_state.get("held_object_family")) if holding else "",
                "track_id": service_state.get("held_object_track_id") if holding else None,
                "source": "service_task_state",
            }
        )
    holding = bool(perception.get("holding_object_context"))
    return clean_empty(
        {
            "holding_object": holding,
            "labels": perception.get("held_object_labels") if holding else [],
            "family": first_meaningful_text(perception.get("held_object_family")) if holding else "",
            "source": "perception_fallback",
        }
    )


def build_task_summary(
    *,
    requested_mode: str,
    mission: JsonDict,
    patrol: JsonDict,
    room: JsonDict,
    service_state: JsonDict,
) -> JsonDict:
    if requested_mode == "auto":
        task_mode = "tidy" if service_state else str(mission.get("mission") or "unknown")
    else:
        task_mode = requested_mode
    return clean_empty(
        {
            "task_mode": task_mode,
            "runner_mode": patrol.get("mode") or mission.get("mode") or "IDLE",
            "mission_enabled": bool(mission.get("enabled")),
            "patrol_enabled": bool(patrol.get("enabled")),
            "room": mission.get("current_room") or room.get("room_name") or "current_room",
            "step_count": first_present(patrol.get("step_count"), mission.get("total_steps_completed")),
            "max_steps": first_present(patrol.get("max_steps"), mission.get("max_steps")),
            "phase": service_state.get("phase") or "",
            "phase_attempts": service_state.get("phase_attempts"),
            "last_reason": service_state.get("last_reason") or patrol.get("last_action"),
            "pickup_surface_policy": service_state.get("pickup_surface_policy") or "floor-only",
            "interaction_grounding": service_state.get("interaction_grounding") or "metadata-hidden",
            "locked_pickup_target": compact_locked_target(service_state, prefix="target"),
            "locked_receptacle": compact_locked_target(service_state, prefix="receptacle"),
            "completed_subgoal_count": len(as_list(service_state.get("completed_subgoals"))),
            "objects_placed_count": len(as_list(mission.get("objects_placed")))
            or len(as_list(room.get("objects_placed"))),
            "service_tasks_completed_count": len(as_list(mission.get("service_tasks_completed")))
            or len(as_list(room.get("service_tasks_completed"))),
            "room_complete": bool(room.get("room_complete")),
        }
    )


def compact_locked_target(service_state: JsonDict, *, prefix: str) -> JsonDict:
    return clean_empty(
        {
            "label": service_state.get(f"{prefix}_label"),
            "raw_label": service_state.get(f"{prefix}_raw_label"),
            "signature": service_state.get(f"{prefix}_signature"),
            "track_id": service_state.get(f"{prefix}_track_id"),
            "last_seen_step": service_state.get(f"{prefix}_last_seen_step"),
            "lost_scan_count": service_state.get(f"{prefix}_lost_scan_count"),
        }
    )


def build_perception_summary(perception: JsonDict, source_info: JsonDict) -> JsonDict:
    status = str(perception.get("status") or "missing")
    return clean_empty(
        {
            "status": status,
            "structured_perception_available": status == "success"
            and perception.get("result_type") in {"scene_analyzed_yolo", "scene_analyzed"},
            "result_type": perception.get("result_type"),
            "perception_mode": perception.get("perception_mode"),
            "backend": perception.get("perception_backend"),
            "online_safe": bool_or_none(perception.get("online_safe")),
            "image_path": perception.get("image_path"),
            "depth_path": perception.get("depth_path"),
            "analysis_confidence": number_or_none(perception.get("analysis_confidence")),
            "holding_object_context": bool_or_none(perception.get("holding_object_context")),
            "pickup_target_detected": bool_or_none(perception.get("pickup_target_detected")),
            "place_receptacle_detected": bool_or_none(perception.get("place_receptacle_detected")),
            "direct_pickup_detected": bool_or_none(perception.get("direct_pickup_detected")),
            "direct_place_detected": bool_or_none(perception.get("direct_place_detected")),
            "floor_trash_detected": bool_or_none(perception.get("floor_trash_detected")),
            "obstacle_ahead": bool_or_none(perception.get("obstacle_ahead")),
            "frontier_exists": bool_or_none(perception.get("frontier_exists")),
            "open_directions": as_list(perception.get("open_directions")),
            "recommended_action": perception.get("recommended_action"),
            "source_file": source_info,
        }
    )


def build_navigation_summary(
    *,
    perception: JsonDict,
    costmap: JsonDict,
    room: JsonDict,
    position_map: JsonDict,
    global_plan: JsonDict,
    map_backend_summary: JsonDict | None = None,
) -> JsonDict:
    action_safety = {}
    raw_safety = as_dict(costmap.get("action_safety"))
    for action in (*BODY_MOVE_ACTIONS, *CAMERA_ACTIONS):
        item = as_dict(raw_safety.get(action))
        if not item:
            continue
        action_safety[action] = clean_empty(
            {
                "safe": bool_or_none(item.get("safe")),
                "confidence": number_or_none(item.get("confidence")),
                "reason": item.get("reason"),
                "observed_ratio": number_or_none(item.get("observed_ratio")),
                "min_observed_ratio": number_or_none(
                    item.get("min_observed_ratio"),
                    digits=4,
                )
                if number_or_none(item.get("min_observed_ratio"), digits=4) is not None
                else default_min_observed_ratio_for_action(action),
                "blocked_cell_count": number_or_none(item.get("blocked_cell_count")),
            }
        )
    stats = as_dict(position_map.get("stats"))
    frontiers = as_list(position_map.get("frontiers"))
    return clean_empty(
        {
            "map_backend": clean_empty(as_dict(map_backend_summary)),
            "pose": clean_empty(as_dict(position_map.get("pose"))),
            "active_frontier_goal": clean_empty(
                as_dict(position_map.get("active_frontier_goal")) or as_dict(room.get("active_frontier_goal"))
            ),
            "active_route": clean_empty(
                as_dict(position_map.get("active_route")) or as_dict(room.get("active_route"))
            ),
            "coverage_patrol": clean_empty(
                as_dict(position_map.get("coverage_patrol")) or as_dict(room.get("coverage_patrol"))
            ),
            "coverage": clean_empty(
                {
                    "coverage_estimate": first_present(
                        number_or_none(position_map.get("coverage_estimate")),
                        number_or_none(stats.get("coverage_estimate")),
                        number_or_none(room.get("coverage_estimate")),
                    ),
                    "frontier_count": len(frontiers) or len(as_list(room.get("frontier_cells"))),
                    "frontier_sample": frontiers[:8],
                    "visited_cell_count": first_present(
                        stats.get("visited_cell_count"), len(as_list(room.get("visited_cells")))
                    ),
                    "collision_count": first_present(stats.get("collision_count"), room.get("collision_count")),
                    "stagnation_count": room.get("stagnation_count"),
                    "oscillation_count": room.get("oscillation_count"),
                }
            ),
            "local_costmap": clean_empty(
                {
                    "status": costmap.get("status"),
                    "step": costmap.get("step"),
                    "holding_object": bool_or_none(costmap.get("holding_object")),
                    "front_clearance_m": number_or_none(costmap.get("front_clearance_m")),
                    "left_clearance_m": number_or_none(costmap.get("left_clearance_m")),
                    "right_clearance_m": number_or_none(costmap.get("right_clearance_m")),
                    "blocked_actions": as_list(costmap.get("blocked_actions")),
                    "moveahead_blockers": as_list(costmap.get("moveahead_blockers"))[:5],
                    "action_safety": action_safety,
                }
            ),
            "current_perception_navigation": clean_empty(
                {
                    "open_directions": as_list(perception.get("open_directions")),
                    "obstacle_ahead": bool_or_none(perception.get("obstacle_ahead")),
                    "frontier_exists": bool_or_none(perception.get("frontier_exists")),
                    "recommended_action": perception.get("recommended_action"),
                    "occupancy": clean_empty(as_dict(perception.get("occupancy"))),
                }
            ),
            "global_plan": compact_global_plan(global_plan),
            "recent_actions": as_list(position_map.get("recent_actions"))[-8:],
        }
    )


def compact_global_plan(global_plan: JsonDict) -> JsonDict:
    return clean_empty(
        {
            "status": global_plan.get("status"),
            "planner": global_plan.get("planner"),
            "target_reason": global_plan.get("target_reason"),
            "selected_goal_cell": global_plan.get("selected_goal_cell"),
            "selected_goal_kind": global_plan.get("selected_goal_kind"),
            "next_cell": global_plan.get("next_cell"),
            "next_action": global_plan.get("next_action"),
            "target_track_id": global_plan.get("target_track_id"),
            "goal_type": global_plan.get("goal_type"),
            "route_cost": number_or_none(global_plan.get("route_cost")),
            "candidate_goal_count": number_or_none(global_plan.get("candidate_goal_count")),
            "replan_reason": global_plan.get("replan_reason"),
        }
    )


def option_id(prefix: str, value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"{prefix}:{safe or 'option'}"


def frontier_option_id(cell: Any) -> str:
    try:
        left, right = str(cell or "").split(",", 1)
        x = int(left)
        z = int(right)
        def encode(value: int) -> str:
            return f"m{abs(value)}" if value < 0 else str(value)

        return f"explore:frontier:x{encode(x)}_z{encode(z)}"
    except (TypeError, ValueError):
        return option_id("explore:frontier", str(cell or "frontier"))


def frontier_cluster_option_id(cell: Any) -> str:
    try:
        left, right = str(cell or "").split(",", 1)
        x = int(left)
        z = int(right)

        def encode(value: int) -> str:
            return f"m{abs(value)}" if value < 0 else str(value)

        return f"explore:frontier_cluster:x{encode(x)}_z{encode(z)}"
    except (TypeError, ValueError):
        return option_id("explore:frontier_cluster", str(cell or "frontier_cluster"))


def waypoint_option_id(cell: Any) -> str:
    try:
        left, right = str(cell or "").split(",", 1)
        x = int(left)
        z = int(right)

        def encode(value: int) -> str:
            return f"m{abs(value)}" if value < 0 else str(value)

        return f"explore:waypoint:x{encode(x)}_z{encode(z)}"
    except (TypeError, ValueError):
        return option_id("explore:waypoint", str(cell or "waypoint"))


def inspection_waypoint_option_id(waypoint_id: Any) -> str:
    safe = str(waypoint_id or "").strip()
    return f"explore:inspection_waypoint:{safe}" if safe else "explore:inspection_waypoint:unknown"


def parse_cell_safe(value: Any) -> tuple[int, int] | None:
    try:
        return parse_cell(value)
    except Exception:
        return None


def cell_text(value: Any) -> str:
    parsed = parse_cell_safe(value)
    if parsed is None:
        return ""
    return f"{parsed[0]},{parsed[1]}"


def map_cells(position_map: JsonDict | None) -> JsonDict:
    cells = as_dict(as_dict(position_map).get("cells"))
    return cells


def cell_record(position_map: JsonDict | None, cell: Any) -> JsonDict:
    key = cell_text(cell)
    if not key:
        return {}
    return as_dict(map_cells(position_map).get(key))


def cell_state(position_map: JsonDict | None, cell: Any) -> str:
    return str(cell_record(position_map, cell).get("state") or "")


def cell_visited(position_map: JsonDict | None, room: JsonDict | None, cell: Any) -> bool:
    key = cell_text(cell)
    if not key:
        return False
    rec = cell_record(position_map, key)
    if rec.get("visited") is True:
        return True
    return key in {str(item) for item in as_list(as_dict(room).get("visited_cells"))}


def frontier_cells(position_map: JsonDict | None, room: JsonDict | None) -> set[str]:
    values = []
    data = as_dict(position_map)
    state = as_dict(room)
    values.extend(as_list(data.get("frontiers")))
    values.extend(as_list(data.get("frontier_cells")))
    values.extend(as_list(state.get("frontier_cells")))
    values.extend(as_list(state.get("known_frontier_cells")))
    return {cell for item in values if (cell := cell_text(item))}


def recent_navigation_cells(room: JsonDict | None, *, limit: int = 10) -> set[str]:
    state = as_dict(room)
    cells: set[str] = set()
    for item in as_list(state.get("recent_navigation_cells"))[-limit:]:
        if cell := cell_text(item):
            cells.add(cell)
    for action in as_list(state.get("recent_navigation_actions"))[-limit:]:
        text = str(action or "")
        # The action list is often action-only, but some older entries contain
        # explicit cells; harvest them when present without relying on it.
        for token in text.replace("->", " ").replace(":", " ").split():
            if "," in token and (cell := cell_text(token)):
                cells.add(cell)
    if cell := cell_text(state.get("last_cell")):
        cells.add(cell)
    return cells


def unknown_neighbor_count(position_map: JsonDict | None, cell: Any) -> int:
    key = cell_text(cell)
    if not key:
        return 0
    cells = map_cells(position_map)
    count = 0
    for neighbor in four_neighbors(key):
        rec = as_dict(cells.get(neighbor))
        if not rec or str(rec.get("state") or CELL_UNKNOWN) == CELL_UNKNOWN:
            count += 1
    return count


def semantic_frontier_score(position_map: JsonDict | None, cell: Any) -> float:
    scores = as_dict(as_dict(position_map).get("frontier_scores"))
    item = as_dict(scores.get(cell_text(cell)))
    value = number_or_none(item.get("exploration_score"))
    return float(value or 0.0)


def waypoint_information_gain(
    waypoint: JsonDict,
    *,
    position_map: JsonDict | None,
    room: JsonDict | None,
) -> JsonDict:
    cell = cell_text(waypoint.get("cell"))
    frontiers = frontier_cells(position_map, room)
    recent = recent_navigation_cells(room)
    unknown = unknown_neighbor_count(position_map, cell)
    visited = cell_visited(position_map, room, cell) or waypoint.get("map_visited") is True
    frontier_bonus = 1.0 if cell in frontiers else 0.0
    semantic_bonus = semantic_frontier_score(position_map, cell)
    coverage = float(number_or_none(waypoint.get("coverage_estimate")) or 0.0)
    covered_cells = float(number_or_none(waypoint.get("covered_cell_count")) or 0.0)
    recent_penalty = 1.0 if cell in recent else 0.0
    visited_penalty = 0.6 if visited else 0.0
    score = (
        1.20 * frontier_bonus
        + 0.45 * unknown
        + 0.70 * semantic_bonus
        + 0.20 * coverage
        + min(0.8, 0.03 * covered_cells)
        - recent_penalty
        - visited_penalty
    )
    return clean_empty(
        {
            "cell": cell,
            "score": round(score, 6),
            "frontier_bonus": frontier_bonus,
            "unknown_neighbor_count": unknown,
            "semantic_frontier_score": round(semantic_bonus, 6),
            "coverage_estimate": waypoint.get("coverage_estimate"),
            "covered_cell_count": waypoint.get("covered_cell_count"),
            "visited_penalty": visited_penalty,
            "recent_path_penalty": recent_penalty,
            "policy": "semexp_frontier_information_gain",
        }
    )


def waypoint_sort_key(item: JsonDict) -> tuple[float, int, str]:
    gain = as_dict(item.get("information_gain"))
    score = float(number_or_none(gain.get("score")) or 0.0)
    distance = int(number_or_none(item.get("last_distance_cells"), digits=0) or 10**9)
    return (-score, distance, str(item.get("waypoint_id") or ""))


def route_step_option_id(route_id: Any, step_index: Any) -> str:
    return option_id("explore:route_step", f"{route_id or 'route'}:{step_index or 0}")


def waypoint_target_payload(item: JsonDict) -> JsonDict:
    return clean_empty(
        {
            "waypoint_id": item.get("waypoint_id"),
            "cell": item.get("cell"),
            "label": item.get("label"),
            "purpose": item.get("purpose"),
            "waypoint_source": item.get("waypoint_source"),
            "coverage_radius_cells": item.get("coverage_radius_cells"),
            "coverage_estimate": item.get("coverage_estimate"),
            "covered_cell_count": item.get("covered_cell_count"),
            "information_gain": item.get("information_gain"),
            "component_id": item.get("component_id"),
            "component_size": item.get("component_size"),
            "last_distance_cells": item.get("last_distance_cells"),
            "status": item.get("status"),
        }
    )


def candidate_ref(candidate: JsonDict) -> JsonDict:
    return clean_empty(
        {
            "candidate_id": candidate.get("candidate_id"),
            "track_id": candidate.get("track_id"),
            "candidate_signature": candidate.get("candidate_signature"),
            "surface_candidate_id": candidate.get("surface_candidate_id"),
            "label": candidate.get("label"),
            "source": candidate.get("source"),
        }
    )


def pickup_target_handle(candidate: JsonDict, *, index: int = 0) -> str:
    return str(
        first_present(
            candidate.get("track_id"),
            candidate.get("candidate_id"),
            candidate.get("candidate_signature"),
            candidate.get("label"),
            f"pickup_target_{index}",
        )
    )


def pursue_pickup_target_option_id(candidate: JsonDict, *, index: int = 0) -> str:
    return option_id("pursue:pickup_target", pickup_target_handle(candidate, index=index))


def pickup_candidate_pursuit_action(candidate: JsonDict, costmap: JsonDict) -> tuple[str, str]:
    actionability = as_dict(candidate.get("actionability"))
    geometry = as_dict(candidate.get("geometry"))
    position_hint = str(candidate.get("position_hint") or geometry.get("position_hint") or "")
    needs_alignment = actionability.get("needs_alignment") is True
    needs_approach = actionability.get("needs_approach") is True

    if position_hint == "front-left" or (needs_alignment and position_hint != "front-right"):
        action = "RotateLeft"
    elif position_hint == "front-right":
        action = "RotateRight"
    elif position_hint == "front-center" or needs_approach:
        action = "MoveAhead"
    else:
        action = "RotateLeft"

    if action == "MoveAhead":
        safe, reason = action_safe(costmap, action)
        if safe is True and not low_confidence_move_record(costmap, action):
            return action, reason or "approach_visible_pickup_target"
        return "RotateLeft", "approach_blocked_or_low_confidence; rotate_to_reobserve_visible_pickup_target"
    safe, reason = action_safe(costmap, action)
    if safe is False:
        alternate = "RotateRight" if action == "RotateLeft" else "RotateLeft"
        alternate_safe, alternate_reason = action_safe(costmap, alternate)
        if alternate_safe is not False:
            return alternate, alternate_reason or "alternate_turn_to_reobserve_visible_pickup_target"
    return action, reason or "align_visible_pickup_target"


def pickup_candidate_pursuit_ready(candidate: JsonDict) -> bool:
    actionability = as_dict(candidate.get("actionability"))
    if actionability.get("pickup_now") is True and actionability.get("reachable") is True:
        return False
    if str(candidate.get("task_class") or "") != "pickup_target":
        return False
    if actionability.get("blocked") is True or actionability.get("failed_recently") is True:
        return False
    if actionability.get("is_floor_level") is False:
        return False
    if actionability.get("visual_box_ambiguous") is True:
        return False
    try:
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.45:
        return False
    position_hint = str(candidate.get("position_hint") or as_dict(candidate.get("geometry")).get("position_hint") or "")
    return bool(
        position_hint in {"front-left", "front-center", "front-right"}
        and (
            actionability.get("needs_alignment") is True
            or actionability.get("needs_approach") is True
            or actionability.get("reachable") is True
        )
    )


def build_pursue_pickup_target_options(
    *,
    pickup_candidates: list[Any],
    costmap: JsonDict,
    max_targets: int = 2,
) -> list[JsonDict]:
    result: list[JsonDict] = []
    for index, raw in enumerate(pickup_candidates):
        item = as_dict(raw)
        if not pickup_candidate_pursuit_ready(item):
            continue
        action, safety_reason = pickup_candidate_pursuit_action(item, costmap)
        result.append(
            clean_empty(
                {
                    "option_id": pursue_pickup_target_option_id(item, index=index),
                    "kind": "pursue_pickup_target",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": action,
                    "executable_now": True,
                    "decision_level": "task_goal",
                    "llm_priority": "primary",
                    "fallback_only": False,
                    "one_step_only": True,
                    "candidate_ref": candidate_ref(item),
                    "observed_handle": {
                        "handle_id": pickup_target_handle(item, index=index),
                        "label": item.get("label"),
                        "task_class": item.get("task_class"),
                    },
                    "pursuit_policy": "visible_floor_pickup_target_interrupts_waypoint_patrol",
                    "required_next": "robot_cleaner_prepare_decision_turn",
                    "reason": (
                        "Visible floor pickup target is not pickup-ready yet; "
                        "approach or align one safe step, then re-observe before picking."
                    ),
                    "safety_source": "navigation-costmap",
                    "costmap_reason": safety_reason,
                }
            )
        )
        if len(result) >= max(1, int(max_targets)):
            break
    return result


def candidate_matches_ref(candidate: JsonDict, ref: JsonDict) -> bool:
    if not ref:
        return False
    for key in ("candidate_id", "track_id", "candidate_signature", "surface_candidate_id"):
        expected = ref.get(key)
        if expected and str(candidate.get(key) or "") == str(expected):
            return True
    return False


def has_interaction_point(candidate: JsonDict) -> bool:
    point = as_dict(candidate.get("interaction_point"))
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y)


def has_placement_contract(candidate: JsonDict) -> bool:
    if as_list(candidate.get("placement_points")):
        return True
    contract = as_dict(candidate.get("placement_safety_contract"))
    if contract.get("version") == "plane_local_grid_v1":
        return True
    completion = as_dict(candidate.get("free_space_completion"))
    return completion.get("mode") == "plane_local_2d_grid"


def place_candidate_precheck_ready(candidate: JsonDict) -> bool:
    actionability = as_dict(candidate.get("actionability"))
    executor_checks = as_dict(candidate.get("executor_checks"))
    if executor_checks.get("precheck_ok") is True:
        return False
    if str(candidate.get("surface_candidate_source") or "") not in SURFACE_REGION_SOURCES:
        return False
    if actionability.get("reachable") is False:
        return False
    if actionability.get("blocked") is True or actionability.get("failed_recently") is True:
        return False
    if actionability.get("visual_place_ready") is not True:
        return False
    if actionability.get("affordance_ready") is False:
        return False
    return bool(has_interaction_point(candidate) and has_placement_contract(candidate))


def precheck_cache_matches_current_perception(
    cache: JsonDict,
    perception_source_info: JsonDict,
) -> bool:
    cached_source_time = str(cache.get("perception_source_last_modified") or "")
    current_source_time = str(perception_source_info.get("last_modified") or "")
    if cached_source_time and current_source_time:
        return cached_source_time == current_source_time
    return True


def apply_place_precheck_cache(
    worklist: JsonDict,
    cache: JsonDict,
    *,
    perception_source_info: JsonDict,
) -> None:
    if cache.get("schema") != PLACE_PRECHECK_CACHE_SCHEMA:
        return
    candidate_ref_data = as_dict(cache.get("candidate_ref"))
    result = as_dict(cache.get("result"))
    if not candidate_ref_data or not result:
        return
    precheck_ok = result.get("precheck_ok") is True
    source_time_matches = precheck_cache_matches_current_perception(cache, perception_source_info)
    current_view = as_dict(worklist.get("current_view"))
    for key in ("surface_candidates", "receptacle_candidates"):
        for candidate in as_list(current_view.get(key)):
            if not isinstance(candidate, dict) or not candidate_matches_ref(candidate, candidate_ref_data):
                continue
            executor_checks = as_dict(candidate.get("executor_checks"))
            executor_checks.update(
                clean_empty(
                    {
                        "precheck_supported": True,
                        "precheck_ok": precheck_ok,
                        "reason": result.get("precheck_reason") or result.get("result_type"),
                        "suggested_recovery": result.get("suggested_recovery"),
                        "cached": True,
                        "cached_at": cache.get("cached_at"),
                        "cached_source_time_matches": source_time_matches,
                        "placement_point_source": result.get("placement_point_source"),
                        "placement_clearance_contract_applied": result.get(
                            "placement_clearance_contract_applied"
                        ),
                        "placement_execution_mode": result.get("placement_execution_mode"),
                        "placement_target_required": result.get("placement_target_required"),
                        "placement_target_resolution_error_m": result.get(
                            "placement_target_resolution_error_m"
                        ),
                        "placement_target_resolution_tolerance_m": result.get(
                            "placement_target_resolution_tolerance_m"
                        ),
                    }
                )
            )
            candidate["executor_checks"] = executor_checks
            actionability = as_dict(candidate.get("actionability"))
            if precheck_ok:
                actionability["final_place_ready"] = True
                actionability["place_now"] = True
            candidate["actionability"] = actionability


def action_safe(costmap: JsonDict, action: str) -> tuple[bool | None, str]:
    raw = as_dict(as_dict(costmap.get("action_safety")).get(action))
    if not raw:
        key = {
            "MoveAhead": "moveahead_safe",
            "MoveBack": "moveback_safe",
            "MoveLeft": "moveleft_safe",
            "MoveRight": "moveright_safe",
            "RotateLeft": "rotate_left_safe",
            "RotateRight": "rotate_right_safe",
            "LookUp": "lookup_safe",
            "LookDown": "lookdown_safe",
        }.get(action)
        if key and key in costmap:
            return bool(costmap.get(key)), "legacy_costmap_flag"
        return None, "missing_costmap_safety"
    return bool_or_none(raw.get("safe")), str(raw.get("reason") or "")


def costmap_action_record(costmap: JsonDict, action: str) -> JsonDict:
    return as_dict(as_dict(costmap.get("action_safety")).get(action))


def default_min_observed_ratio_for_action(action: str) -> float:
    if action == "MoveBack":
        return DEFAULT_REAR_MIN_OBSERVED_RATIO
    if action in {"MoveLeft", "MoveRight"}:
        return DEFAULT_LATERAL_MIN_OBSERVED_RATIO
    if action == "MoveAhead":
        return DEFAULT_FORWARD_MIN_OBSERVED_RATIO
    return 0.0


def required_min_observed_ratio_for_action(action: str, raw: JsonDict) -> float:
    observed_min = number_or_none(raw.get("min_observed_ratio"), digits=4)
    default_min = default_min_observed_ratio_for_action(action)
    if observed_min is None:
        return default_min
    return max(float(observed_min), float(default_min))


def low_confidence_move_record(costmap: JsonDict, action: str) -> JsonDict | None:
    if action not in TRANSLATION_ACTIONS:
        return None
    raw = costmap_action_record(costmap, action)
    if not raw:
        return None
    observed_ratio = number_or_none(raw.get("observed_ratio"), digits=4)
    if observed_ratio is None:
        return None
    min_observed_ratio = required_min_observed_ratio_for_action(action, raw)
    if float(observed_ratio) >= float(min_observed_ratio):
        return None
    return clean_empty(
        {
            "action": action,
            "observed_ratio": observed_ratio,
            "min_observed_ratio": min_observed_ratio,
            "confidence": number_or_none(raw.get("confidence"), digits=4),
            "reason": "low_observed_swept_volume",
            "costmap_reason": raw.get("reason"),
        }
    )


def perception_bootstrap_move_safe(perception: JsonDict, action: str) -> tuple[bool, str]:
    """Fallback only for cold-start exploration when local costmap has no verdict."""
    if action != "MoveAhead":
        return False, ""
    if perception.get("obstacle_ahead") is True:
        return False, ""
    open_directions = {str(item).lower() for item in as_list(perception.get("open_directions"))}
    if "forward" not in open_directions:
        return False, ""
    occupancy = as_dict(perception.get("occupancy"))
    forward_occupancy = number_or_none(occupancy.get("forward"), digits=4)
    if forward_occupancy is not None and float(forward_occupancy) >= 0.16:
        return False, ""
    return True, "current_rgbd_perception_marks_forward_open"


def place_candidate_executor_ready(candidate: JsonDict) -> bool:
    actionability = as_dict(candidate.get("actionability"))
    executor_checks = as_dict(candidate.get("executor_checks"))
    if executor_checks.get("precheck_ok") is True:
        return True
    if actionability.get("needs_alignment") is True or actionability.get("needs_approach") is True:
        return False
    if actionability.get("reachable") is False:
        return False
    return actionability.get("place_now") is True or actionability.get("final_place_ready") is True


def annotate_option_selection_contract(options: list[JsonDict]) -> JsonDict:
    """Annotate options with model-facing decision-level metadata.

    This is not a recommender. It keeps the LLM as the selector while making the
    interface explicit: pick/place/explore are goal-level choices; raw move:*
    actions are motor-level fallbacks whenever a goal-level option exists.
    """

    goal_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") in GOAL_LEVEL_OPTION_KINDS and option.get("executable_now") is True
    ]
    cluster_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "explore_frontier_cluster" and option.get("executable_now") is True
    ]
    inspection_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "explore_inspection_waypoint" and option.get("executable_now") is True
    ]
    continue_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "continue_active_waypoint_goal" and option.get("executable_now") is True
    ]
    pursue_pickup_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "pursue_pickup_target" and option.get("executable_now") is True
    ]
    floor_scan_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "orient_waypoint_floor_scan" and option.get("executable_now") is True
    ]
    waypoint_decision_option_ids = floor_scan_option_ids + continue_option_ids + inspection_option_ids
    explore_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind")
        in {
            "continue_active_waypoint_goal",
            "explore_inspection_waypoint",
            "explore_frontier_cluster",
            "explore_route_step",
            "explore_frontier",
            "explore_waypoint",
        }
        and option.get("executable_now") is True
    ]
    explore_route_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "explore_route_step" and option.get("executable_now") is True
    ]
    recovery_option_ids = [
        str(option.get("option_id"))
        for option in options
        if option.get("kind") == "recovery_action" and option.get("executable_now") is True
    ]
    has_goal_options = bool(goal_option_ids)

    primary_options: list[str] = []
    fallback_options: list[str] = []
    support_options: list[str] = []
    low_level_move_options: list[str] = []

    for option in options:
        kind = str(option.get("kind") or "")
        oid = str(option.get("option_id") or "")
        if not oid:
            continue
        if kind == "move_action":
            low_level_move_options.append(oid)
            option["decision_level"] = "motor"
            option["llm_priority"] = "fallback" if has_goal_options else "available_when_no_goal_option"
            option["fallback_only"] = bool(has_goal_options)
            option["allowed_when"] = (
                "no_goal_level_option_or_recovery_required"
                if has_goal_options
                else "no_goal_level_option_available"
            )
            if option["fallback_only"]:
                fallback_options.append(oid)
            else:
                primary_options.append(oid)
            continue

        option["fallback_only"] = False
        if kind == "orient_waypoint_floor_scan":
            option["decision_level"] = "waypoint_observation_setup"
            option["llm_priority"] = "primary"
            option["allowed_when"] = "active_inspection_waypoint_cell_reached_but_floor_scan_turn_not_prepared"
        elif kind == "continue_active_waypoint_goal":
            option["decision_level"] = "navigation_goal_continuation"
            option["llm_priority"] = "primary"
            option["allowed_when"] = "active_inspection_waypoint_goal_in_progress"
        elif kind == "explore_inspection_waypoint":
            option["decision_level"] = "goal"
            option["llm_priority"] = "primary"
            option["allowed_when"] = "choose_durable_inspection_waypoint_goal"
        elif kind == "explore_frontier_cluster":
            option["decision_level"] = "goal"
            if waypoint_decision_option_ids:
                option["llm_priority"] = "fallback"
                option["fallback_only"] = True
                option["allowed_when"] = "inspection_waypoint_options_unavailable_or_executor_requires_frontier_fallback"
                fallback_options.append(oid)
                continue
            option["llm_priority"] = "primary"
            option["allowed_when"] = "choose_navigation_goal_cluster_when_no_inspection_waypoint_api"
        elif kind == "explore_route_step":
            option["decision_level"] = "navigation_step"
            option["llm_priority"] = "fallback"
            option["fallback_only"] = True
            option["allowed_when"] = "legacy_internal_route_step_only_when_no_inspection_waypoint_option"
            fallback_options.append(oid)
            continue
        elif kind in {"explore_frontier", "explore_waypoint"}:
            option["decision_level"] = "goal"
            if waypoint_decision_option_ids or cluster_option_ids or explore_route_option_ids:
                option["llm_priority"] = "fallback"
                option["fallback_only"] = True
                option["allowed_when"] = "inspection_waypoint_or_cluster_unavailable_or_recovery_required"
                fallback_options.append(oid)
                continue
            option["llm_priority"] = "primary"
            option["allowed_when"] = "no_committed_route_or_as_exploration_target"
        elif kind == "recovery_action":
            option["decision_level"] = "recovery"
            option["llm_priority"] = "primary"
            option["allowed_when"] = "dead_end_or_blocked_region_recovery"
        elif kind == "pursue_pickup_target":
            option["decision_level"] = "task_goal"
            option["llm_priority"] = "primary"
            option["allowed_when"] = "visible_floor_pickup_target_requires_approach_alignment_or_reobservation"
        elif kind in {"service_action", "place_precheck", "clean_action"}:
            option["decision_level"] = "task"
            option["llm_priority"] = "primary"
            option["allowed_when"] = "task_subgoal_available"
        elif kind in SUPPORT_OPTION_KINDS:
            option["decision_level"] = "support"
            option["llm_priority"] = "support"
            option["allowed_when"] = "refresh_status_or_completion_check"
            support_options.append(oid)
            if has_goal_options:
                option["fallback_only"] = True
                option["allowed_when"] = "support_only_after_goal_options_unavailable_or_required"
                fallback_options.append(oid)
                continue
        else:
            option["decision_level"] = "support"
            option["llm_priority"] = "available"
            option["allowed_when"] = "option_specific_contract"
        primary_options.append(oid)

    return clean_empty(
        {
            "primary_options": (
                [
                    oid
                    for oid in recovery_option_ids
                    if oid in primary_options
                ]
                + [oid for oid in primary_options if oid not in recovery_option_ids]
                if recovery_option_ids
                else primary_options
            ),
            "fallback_options": fallback_options,
            "goal_options": goal_option_ids,
            "pursue_pickup_target_options": pursue_pickup_option_ids,
            "waypoint_floor_scan_options": floor_scan_option_ids,
            "explore_goal_options": explore_option_ids,
            "explore_inspection_waypoint_options": inspection_option_ids,
            "continue_active_waypoint_options": continue_option_ids,
            "explore_frontier_cluster_options": cluster_option_ids,
            "explore_route_options": explore_route_option_ids,
            "explore_frontier_options": [
                str(option.get("option_id"))
                for option in options
                if option.get("kind") == "explore_frontier" and option.get("executable_now") is True
            ],
            "explore_waypoint_options": [
                str(option.get("option_id"))
                for option in options
                if option.get("kind") == "explore_waypoint" and option.get("executable_now") is True
            ],
            "recovery_options": recovery_option_ids,
            "support_options": support_options,
            "low_level_move_options": low_level_move_options,
            "raw_move_policy_active": bool(has_goal_options),
        }
    )


def build_recovery_options(
    *,
    exploration: JsonDict,
    explore_plan: JsonDict | None = None,
    costmap: JsonDict,
    move_options: list[JsonDict],
    max_recovery_options: int = 3,
) -> list[JsonDict]:
    dead_end = as_dict(exploration.get("dead_end"))
    camera_posture = as_dict(exploration.get("camera_posture"))
    plan = as_dict(explore_plan)
    camera_needs_normalization = camera_posture.get("needs_normalization") is True
    planner_recovery_actions = as_list(plan.get("recovery_actions"))
    if (
        dead_end.get("active") is not True
        and not camera_needs_normalization
        and not planner_recovery_actions
    ):
        return []

    move_by_action = {
        str(option.get("action") or ""): option
        for option in move_options
        if option.get("kind") == "move_action" and option.get("executable_now") is True
    }

    def build_recovery_option(action: str, reason: str) -> JsonDict | None:
        safe, safety_reason = action_safe(costmap, action)
        step_option = as_dict(move_by_action.get(action))
        if action in BODY_MOVE_ACTIONS and not step_option:
            return None
        if safe is False:
            return None
        oid = option_id("recover", action)
        return clean_empty(
            {
                "option_id": oid,
                "kind": "recovery_action",
                "physical_action": True,
                "tool": "move-robot",
                "action": action,
                "executable_now": True,
                "decision_level": "recovery",
                "llm_priority": "primary",
                "fallback_only": False,
                "one_step_only": True,
                "reason": reason,
                "safety_source": "navigation-costmap" if safe is True else "camera-recovery-fallback",
                "costmap_reason": safety_reason,
                "recovery_policy": (
                    "body_motion_first"
                    if action in BODY_MOVE_ACTIONS
                    else "camera_pitch_only_after_body_recovery_unavailable"
                ),
                "dead_end_context": {
                    "current_cell": dead_end.get("current_cell"),
                    "current_heading": dead_end.get("current_heading"),
                    "blocked_actions": dead_end.get("blocked_actions"),
                    "suggested_backtrack_cell": dead_end.get("suggested_backtrack_cell"),
                },
                "camera_posture": {
                    "needs_normalization": camera_posture.get("needs_normalization"),
                    "normalize_action": camera_posture.get("normalize_action"),
                    "pitch_offset_steps": camera_posture.get("pitch_offset_steps"),
                    "last_camera_action": camera_posture.get("last_camera_action"),
                },
            }
        )

    def collect_options(candidates: list[tuple[str, str]]) -> list[JsonDict]:
        result: list[JsonDict] = []
        seen: set[str] = set()
        for action, reason in candidates:
            if action in seen:
                continue
            seen.add(action)
            option = build_recovery_option(action, reason)
            if option is None:
                continue
            result.append(option)
            if len(result) >= max(1, int(max_recovery_options)):
                break
        return result

    body_candidates: list[tuple[str, str]] = []
    camera_candidates: list[tuple[str, str]] = []
    normalize_action = str(camera_posture.get("normalize_action") or "")
    for raw in planner_recovery_actions:
        item = as_dict(raw)
        action = str(item.get("action") or "").strip()
        reason = str(item.get("reason") or "planner_recovery_action")
        if action in BODY_MOVE_ACTIONS:
            body_candidates.append((action, reason))
        elif action in CAMERA_ACTIONS:
            camera_candidates.append((action, reason))
    suggested = str(dead_end.get("suggested_backtrack_action") or "")
    if suggested:
        body_candidates.append((suggested, "backtrack_toward_previous_pose"))
    body_candidates.extend(
        [
            ("RotateLeft", "scan_left_for_exit"),
            ("RotateRight", "scan_right_for_exit"),
            ("MoveBack", "step_back_if_costmap_allows"),
        ]
    )
    if camera_needs_normalization and normalize_action:
        camera_candidates.append((normalize_action, "restore_default_camera_pitch_before_navigation"))
    camera_candidates.extend(
        [
            ("LookDown", "refresh_depth_with_lower_camera_pitch_after_body_recovery_unavailable"),
            ("LookUp", "refresh_depth_with_higher_camera_pitch_after_body_recovery_unavailable"),
        ]
    )

    body_recovery_required = bool(dead_end.get("active") is True or planner_recovery_actions)
    if body_recovery_required:
        body_result = collect_options(body_candidates)
        if body_result:
            return body_result

    if camera_needs_normalization and not body_recovery_required:
        any_body_action_available = any(
            str(option.get("action") or "") in BODY_MOVE_ACTIONS
            for option in move_options
            if option.get("kind") == "move_action" and option.get("executable_now") is True
        )
        if any_body_action_available:
            return []

    return collect_options(camera_candidates)


def build_inspection_waypoint_options(
    *,
    coverage_waypoints: JsonDict | None,
    costmap: JsonDict,
    position_map: JsonDict | None = None,
    room: JsonDict | None = None,
    holding: bool,
    has_task_options: bool,
    max_waypoints: int = 4,
) -> list[JsonDict]:
    """Expose durable inspection waypoint choices as the primary exploration API."""

    coverage = as_dict(coverage_waypoints)
    required_count = int(number_or_none(coverage.get("required_waypoint_count")) or 0)
    pending_count = int(number_or_none(coverage.get("pending_waypoint_count")) or 0)
    if holding or has_task_options or required_count < 2 or pending_count <= 0:
        return []

    active_goal = as_dict(coverage.get("active_waypoint_goal"))
    active_waypoint_id = str(active_goal.get("waypoint_id") or "").strip()
    active_status = str(active_goal.get("status") or "").strip()
    if active_waypoint_id and active_status not in {"blocked", "failed"}:
        active_route = as_dict(coverage.get("active_waypoint_route"))
        route_step = as_dict(active_route.get("route_step"))
        if active_waypoint_reached_and_unprepared(coverage)[1]:
            scan_option = build_waypoint_floor_scan_option(
                coverage_waypoints=coverage,
                costmap=costmap,
                position_map=position_map,
                room=room,
            )
            if scan_option:
                return [scan_option]
        return [
            clean_empty(
                {
                    "option_id": "continue:active_waypoint_goal",
                    "kind": "continue_active_waypoint_goal",
                    "physical_action": True,
                    "tool": "waypoint-planner + move-robot",
                    "action": "continue-active-waypoint-goal",
                    "executable_now": True,
                    "decision_level": "navigation_goal_continuation",
                    "llm_priority": "primary",
                    "fallback_only": False,
                    "one_step_only": True,
                    "active_waypoint_goal": {
                        "waypoint_id": active_waypoint_id,
                        "cell": active_goal.get("cell"),
                        "status": active_status or "active",
                        "route_id": active_goal.get("route_id"),
                        "next_action": active_goal.get("next_action"),
                        "next_cell": active_goal.get("next_cell"),
                        "purpose": active_goal.get("purpose"),
                    },
                    "active_waypoint_route": {
                        "status": active_route.get("status"),
                        "route_id": active_route.get("route_id"),
                        "waypoint_id": active_route.get("waypoint_id"),
                        "goal_cell": active_route.get("goal_cell"),
                        "current_cell": active_route.get("current_cell"),
                        "current_heading": active_route.get("current_heading"),
                        "next_action": active_route.get("next_action"),
                        "next_cell": active_route.get("next_cell"),
                        "path": active_route.get("path"),
                        "path_length": active_route.get("path_length"),
                        "action_safety": active_route.get("action_safety"),
                    },
                    "route_step": {
                        "route_id": route_step.get("route_id"),
                        "step_index": route_step.get("step_index"),
                        "status": route_step.get("status"),
                        "action": route_step.get("action"),
                        "current_cell": route_step.get("current_cell"),
                        "current_heading": route_step.get("current_heading"),
                        "target_cell": route_step.get("target_cell"),
                        "next_cell": route_step.get("next_cell"),
                        "goal_cell": route_step.get("goal_cell"),
                        "desired_heading": route_step.get("desired_heading"),
                        "heading_after_action": route_step.get("heading_after_action"),
                        "progress_effect": route_step.get("progress_effect"),
                    },
                    "coverage_progress": {
                        "sweep_coverage_rate": coverage.get("sweep_coverage_rate"),
                        "observed_waypoint_count": coverage.get("observed_waypoint_count"),
                        "blocked_waypoint_count": coverage.get("blocked_waypoint_count"),
                        "required_waypoint_count": coverage.get("required_waypoint_count"),
                        "pending_waypoint_count": coverage.get("pending_waypoint_count"),
                    },
                    "reason": (
                        "Continue the currently active inspection waypoint goal; "
                        "the waypoint planner will resolve exactly one safe route step."
                    ),
                }
            )
        ]

    result: list[JsonDict] = []
    pending_waypoints = []
    for raw in as_list(coverage.get("next_unobserved_waypoints")):
        item = dict(as_dict(raw))
        item["information_gain"] = waypoint_information_gain(
            item,
            position_map=position_map,
            room=room,
        )
        pending_waypoints.append(item)
    pending_waypoints = sorted(pending_waypoints, key=waypoint_sort_key)
    for item in pending_waypoints:
        waypoint_id = str(item.get("waypoint_id") or "").strip()
        if not waypoint_id:
            continue
        result.append(
            clean_empty(
                {
                    "option_id": inspection_waypoint_option_id(waypoint_id),
                    "kind": "explore_inspection_waypoint",
                    "physical_action": True,
                    "tool": "waypoint-planner + move-robot",
                    "action": "plan-to-inspection-waypoint",
                    "executable_now": True,
                    "decision_level": "goal",
                    "llm_priority": "primary",
                    "fallback_only": False,
                    "one_step_only": True,
                    "waypoint_target": waypoint_target_payload(item),
                    "coverage_progress": {
                        "sweep_coverage_rate": coverage.get("sweep_coverage_rate"),
                        "observed_waypoint_count": coverage.get("observed_waypoint_count"),
                        "blocked_waypoint_count": coverage.get("blocked_waypoint_count"),
                        "required_waypoint_count": coverage.get("required_waypoint_count"),
                        "pending_waypoint_count": coverage.get("pending_waypoint_count"),
                    },
                    "reason": (
                        f"Select inspection waypoint {waypoint_id} as the durable patrol goal; "
                        "the waypoint planner will resolve one safe route step this turn."
                    ),
                }
            )
        )
        if len(result) >= max(1, int(max_waypoints)):
            break
    return result


def active_waypoint_reached_and_unprepared(coverage_waypoints: JsonDict | None) -> tuple[JsonDict, str]:
    coverage = as_dict(coverage_waypoints)
    active_goal = as_dict(coverage.get("active_waypoint_goal"))
    waypoint_id = str(active_goal.get("waypoint_id") or "").strip()
    if not waypoint_id:
        return {}, ""
    status = str(active_goal.get("status") or "").strip().lower()
    prepared = active_goal.get("floor_scan_prepared") is True
    if prepared:
        return {}, ""
    if status not in {"reached", "arrived", "active"}:
        return {}, ""
    current_cell = str(active_goal.get("current_cell") or "").strip()
    target_cell = str(active_goal.get("cell") or "").strip()
    if current_cell and target_cell and current_cell != target_cell:
        return {}, ""
    waypoint_status = as_dict(as_dict(coverage.get("waypoint_status")).get(waypoint_id))
    distance = number_or_none(waypoint_status.get("last_distance_cells"), digits=0)
    if status not in {"reached", "arrived"} and not (current_cell and target_cell == current_cell) and distance != 0:
        return {}, ""
    return active_goal, waypoint_id


def scan_action_information_gain(
    *,
    action: str,
    active_goal: JsonDict,
    position_map: JsonDict | None,
    room: JsonDict | None,
) -> JsonDict:
    current_cell = cell_text(active_goal.get("cell"))
    if not current_cell:
        return {}
    pose = as_dict(as_dict(position_map).get("pose"))
    heading = str(pose.get("heading") or as_dict(position_map).get("last_heading") or as_dict(room).get("last_heading") or "north")
    if action == "RotateLeft":
        scan_heading = left_heading(heading)
    elif action == "RotateRight":
        scan_heading = right_heading(heading)
    else:
        scan_heading = heading
    try:
        facing_cell = neighbor_cell(current_cell, scan_heading)
    except Exception:
        facing_cell = ""
    frontiers = frontier_cells(position_map, room)
    recent = recent_navigation_cells(room)
    state = cell_state(position_map, facing_cell)
    visited = cell_visited(position_map, room, facing_cell)
    unknown = unknown_neighbor_count(position_map, facing_cell)
    semantic_bonus = semantic_frontier_score(position_map, facing_cell)
    score = (
        (1.4 if facing_cell in frontiers else 0.0)
        + (0.9 if not visited else 0.0)
        + (0.45 * unknown)
        + (0.70 * semantic_bonus)
        + (0.5 if state in {"", CELL_UNKNOWN} else 0.0)
        - (0.8 if facing_cell in recent else 0.0)
    )
    return clean_empty(
        {
            "action": action,
            "scan_heading": scan_heading,
            "facing_cell": facing_cell,
            "facing_cell_state": state or "unknown",
            "facing_frontier": facing_cell in frontiers,
            "facing_cell_visited": visited,
            "unknown_neighbor_count": unknown,
            "semantic_frontier_score": round(semantic_bonus, 6),
            "recent_path_penalty": 0.8 if facing_cell in recent else 0.0,
            "score": round(score, 6),
            "policy": "semexp_frontier_information_gain",
        }
    )


def choose_waypoint_floor_scan_action(
    costmap: JsonDict,
    *,
    active_goal: JsonDict | None = None,
    position_map: JsonDict | None = None,
    room: JsonDict | None = None,
) -> tuple[str, str, JsonDict, JsonDict]:
    left = costmap_action_record(costmap, "RotateLeft")
    right = costmap_action_record(costmap, "RotateRight")
    gains = {
        "RotateLeft": scan_action_information_gain(
            action="RotateLeft",
            active_goal=as_dict(active_goal),
            position_map=position_map,
            room=room,
        ),
        "RotateRight": scan_action_information_gain(
            action="RotateRight",
            active_goal=as_dict(active_goal),
            position_map=position_map,
            room=room,
        ),
    }

    def score(action: str, record: JsonDict) -> tuple[int, float, float]:
        if record.get("safe") is False:
            return (-1000, -1.0, -1.0)
        safety = 10 if record.get("safe") is True else 1
        clearance = number_or_none(
            first_present(
                record.get("clearance_m"),
                record.get("min_clearance_m"),
                record.get("side_clearance_m"),
                record.get("observed_ratio"),
            ),
            digits=4,
        )
        gain = number_or_none(as_dict(gains.get(action)).get("score"))
        return (safety, float(gain or 0.0), float(clearance) if clearance is not None else 0.0)

    left_score = score("RotateLeft", left)
    right_score = score("RotateRight", right)
    if right_score > left_score:
        return "RotateRight", str(right.get("reason") or "right_turn_selected_for_waypoint_floor_scan"), right, gains["RotateRight"]
    return "RotateLeft", str(left.get("reason") or "left_turn_selected_for_waypoint_floor_scan"), left, gains["RotateLeft"]


def build_waypoint_floor_scan_option(
    *,
    coverage_waypoints: JsonDict | None,
    costmap: JsonDict,
    position_map: JsonDict | None = None,
    room: JsonDict | None = None,
) -> JsonDict | None:
    active_goal, waypoint_id = active_waypoint_reached_and_unprepared(coverage_waypoints)
    if not waypoint_id:
        return None
    action, safety_reason, action_record, scan_gain = choose_waypoint_floor_scan_action(
        costmap,
        active_goal=active_goal,
        position_map=position_map,
        room=room,
    )
    safe, costmap_reason = action_safe(costmap, action)
    if safe is False:
        fallback = "RotateRight" if action == "RotateLeft" else "RotateLeft"
        fallback_safe, fallback_reason = action_safe(costmap, fallback)
        if fallback_safe is False:
            return None
        action = fallback
        safety_reason = fallback_reason or "fallback_safe_turn_for_waypoint_floor_scan"
        action_record = costmap_action_record(costmap, fallback)
        scan_gain = scan_action_information_gain(
            action=fallback,
            active_goal=active_goal,
            position_map=position_map,
            room=room,
        )
    return clean_empty(
        {
            "option_id": "orient:waypoint_floor_scan",
            "kind": "orient_waypoint_floor_scan",
            "physical_action": True,
            "tool": "move-robot",
            "action": action,
            "executable_now": True,
            "decision_level": "waypoint_observation_setup",
            "llm_priority": "primary",
            "fallback_only": False,
            "one_step_only": True,
            "active_waypoint_goal": {
                "waypoint_id": waypoint_id,
                "cell": active_goal.get("cell"),
                "status": active_goal.get("status"),
                "floor_scan_prepared": False,
            },
            "floor_scan_policy": "safe_information_gain_turn_before_full_waypoint_observe",
            "information_gain": scan_gain,
            "reason": (
                "Active inspection waypoint cell is reached; rotate in place once before full "
                "YOLO/depth observation, preferring a safe view toward frontier or unvisited space."
            ),
            "safety_source": "navigation-costmap",
            "costmap_reason": safety_reason or costmap_reason,
            "action_safety": {
                "safe": action_record.get("safe"),
                "reason": action_record.get("reason"),
                "observed_ratio": action_record.get("observed_ratio"),
            },
            "required_next": "robot_cleaner_prepare_decision_turn",
        }
    )


def build_options(
    *,
    task: JsonDict,
    perception: JsonDict,
    costmap: JsonDict,
    exploration: JsonDict | None = None,
    explore_plan: JsonDict | None = None,
    navigation_core: JsonDict | None = None,
    coverage_waypoints: JsonDict | None = None,
    position_map: JsonDict | None = None,
    room: JsonDict | None = None,
    worklist: JsonDict,
    done_readiness: JsonDict,
    max_options: int,
) -> JsonDict:
    options: list[JsonDict] = []
    phase = str(task.get("phase") or "")
    holding = bool(as_dict(worklist.get("held_object")).get("holding_object"))
    structured_ok = bool(perception.get("structured_perception_available"))

    options.append(
        {
            "option_id": "observe:refresh",
            "kind": "perception",
            "physical_action": False,
            "tool": "get-vision + perceive-scene-yolo",
            "executable_now": True,
            "reason": "Refresh RGB-D observation and structured perception before any physical action.",
        }
    )

    current_view = as_dict(worklist.get("current_view"))
    pickup_candidates = as_list(current_view.get("pickup_candidates"))
    receptacle_candidates = as_list(current_view.get("receptacle_candidates"))
    surface_candidates = as_list(current_view.get("surface_candidates"))
    cleanable_candidates = as_list(current_view.get("cleanable_candidates"))
    action_effects = as_dict(as_dict(exploration).get("action_effects"))
    nav_core = as_dict(navigation_core)

    if structured_ok and not holding and phase in SERVICE_PICKUP_PHASES | {""}:
        for candidate in pickup_candidates:
            item = as_dict(candidate)
            actionability = as_dict(item.get("actionability"))
            ready = actionability.get("pickup_now") is True and actionability.get("reachable") is True
            if ready:
                options.append(
                    {
                        "option_id": option_id("pick", str(item.get("candidate_id") or item.get("label"))),
                        "kind": "service_action",
                        "physical_action": True,
                        "tool": "pick-object",
                        "action": "pick-object",
                        "candidate_ref": candidate_ref(item),
                        "executable_now": True,
                        "reason": "Visible pickup target is reachable and pickup_now=true.",
                    }
                )
        options.extend(
            build_pursue_pickup_target_options(
                pickup_candidates=pickup_candidates,
                costmap=costmap,
            )
        )

    if structured_ok and holding and phase in SERVICE_PLACE_PHASES | {""}:
        for candidate in [*surface_candidates, *receptacle_candidates]:
            item = as_dict(candidate)
            if place_candidate_executor_ready(item):
                options.append(
                    {
                        "option_id": option_id("place", str(item.get("candidate_id") or item.get("label"))),
                        "kind": "service_action",
                        "physical_action": True,
                        "tool": "place-object",
                        "action": "place-object",
                        "candidate_ref": candidate_ref(item),
                        "executable_now": True,
                        "reason": "Holding object and current receptacle/surface candidate is executor-ready.",
                    }
                )
            elif place_candidate_precheck_ready(item):
                options.append(
                    {
                        "option_id": option_id(
                            "place_precheck",
                            str(item.get("candidate_id") or item.get("label")),
                        ),
                        "kind": "place_precheck",
                        "physical_action": False,
                        "tool": "place-object",
                        "action": "place-precheck",
                        "candidate_ref": candidate_ref(item),
                        "executable_now": True,
                        "reason": (
                            "Validate the visual-ready pointcloud surface with the backend "
                            "before exposing a physical place action."
                        ),
                    }
                )

    if structured_ok and not holding:
        for candidate in cleanable_candidates:
            item = as_dict(candidate)
            actionability = as_dict(item.get("actionability"))
            if actionability.get("cleanable_now") is True:
                options.append(
                    {
                        "option_id": option_id("clean", str(item.get("candidate_id") or item.get("label"))),
                        "kind": "clean_action",
                        "physical_action": True,
                        "tool": "clean-garbage",
                        "action": "clean-garbage",
                        "candidate_ref": candidate_ref(item),
                        "executable_now": True,
                        "reason": "Visible floor cleanable target is executable now.",
                    }
                )

    movement_allowed = structured_ok
    if not movement_allowed:
        conservative_turns = ("RotateLeft", "RotateRight")
        move_reason = "Structured perception is unavailable; only conservative turns are proposed."
    else:
        conservative_turns = BODY_MOVE_ACTIONS
        move_reason = "Movement option is gated by latest local costmap safety."
    move_options: list[JsonDict] = []
    low_confidence_moves: list[JsonDict] = []
    for action in conservative_turns:
        safe, reason = action_safe(costmap, action)
        if safe is False:
            continue
        if action == "MoveAhead" and perception.get("obstacle_ahead") is True:
            continue
        bootstrap_safe, bootstrap_reason = (False, "")
        if safe is None:
            bootstrap_safe, bootstrap_reason = perception_bootstrap_move_safe(perception, action)
        if safe is True or action in {"RotateLeft", "RotateRight"} or bootstrap_safe:
            low_confidence = None if bootstrap_safe else low_confidence_move_record(costmap, action)
            if low_confidence:
                low_confidence_moves.append(low_confidence)
                continue
            move_option = {
                "option_id": option_id("move", action),
                "kind": "move_action",
                "physical_action": True,
                "tool": "move-robot",
                "action": action,
                "executable_now": True,
                "reason": bootstrap_reason if bootstrap_safe else (reason or move_reason),
                "safety_source": (
                    "navigation-costmap"
                    if safe is True or action in {"RotateLeft", "RotateRight"}
                    else "perception-navigation-bootstrap"
                ),
            }
            effect = as_dict(action_effects.get(action))
            if effect:
                move_option["exploration_effect"] = effect
            move_options.append(move_option)

    recovery_options = build_recovery_options(
        exploration=as_dict(exploration),
        explore_plan=as_dict(explore_plan),
        costmap=costmap,
        move_options=move_options,
    )
    options.extend(recovery_options)
    has_task_options = any(
        option.get("kind") in {"service_action", "place_precheck", "clean_action"}
        and option.get("executable_now") is True
        for option in options
    )
    inspection_waypoint_options = build_inspection_waypoint_options(
        coverage_waypoints=as_dict(coverage_waypoints),
        costmap=costmap,
        position_map=position_map,
        room=room,
        holding=holding,
        has_task_options=has_task_options,
    )
    options.extend(inspection_waypoint_options)
    cluster_options = build_explore_frontier_cluster_options(
        exploration=as_dict(exploration),
        move_options=move_options,
        explore_plan=as_dict(explore_plan),
    )
    options.extend(cluster_options)
    route_options = build_explore_route_step_options(
        explore_plan=as_dict(explore_plan),
        move_options=move_options,
    )
    options.extend(route_options)
    route_locked = bool(route_options) or (
        nav_core.get("route_locked") is True and bool(recovery_options)
    )
    if not route_locked:
        options.extend(
            build_explore_waypoint_options(
                explore_plan=as_dict(explore_plan),
                move_options=move_options,
            )
        )
        options.extend(
            build_explore_frontier_options(
                exploration=as_dict(exploration),
                move_options=move_options,
                explore_plan=as_dict(explore_plan),
            )
        )
    options.extend(move_options)

    options.append(
        {
            "option_id": "done:probe",
            "kind": "completion_probe",
            "physical_action": False,
            "tool": "state-manager/report",
            "executable_now": True,
            "reason": "Ask backend/state gates whether room service patrol can finish.",
            "advisory_can_finish": bool(done_readiness.get("can_finish")),
        }
    )

    options = options[: max(1, max_options)]
    selection_metadata = annotate_option_selection_contract(options)
    rule_baseline = choose_rule_baseline_option(options, task=task, perception=perception, worklist=worklist)
    return {
        "schema": OPTION_SET_SCHEMA,
        "selection_contract": {
            "llm_should_return": {
                "selected_option_id": "<one option_id from options>",
                "brief_reason": "<one short sentence>",
            },
            "model_must_choose_from_options": True,
            "rule_baseline_is_not_instruction": True,
            "rule_baseline_usage": "diagnostic_only_for_rule_fallback_and_regression_tests",
            "executor_must_validate_before_action": True,
            "one_physical_action_per_turn": True,
            "llm_should_choose_from": "primary_options_first",
            "raw_move_policy": "fallback_only_when_goal_level_options_exist",
            "explore_policy": (
                "when no pick/place/pursue-pickup option is appropriate and explore goal options exist, "
                "choose continue:active_waypoint_goal when an inspection waypoint goal is active; "
                "otherwise choose explore:inspection_waypoint:<id> as the model decision target; "
                "legacy explore:frontier_cluster, explore:route_step, explore:waypoint, explore:frontier, "
                "and move:* are fallback/internal-step options"
            ),
            "raw_move_allowed_when": "no goal-level option exists or recovery after failure requires it",
            "recovery_policy": (
                "if recovery_options are present, choose one before repeated observe:refresh, "
                "done:probe, or normal exploration; recover:lookdown restores camera pitch after recover:lookup"
            ),
            "pickup_pursuit_policy": (
                "visible floor pickup targets interrupt waypoint patrol. If a target is seen but not "
                "pickup-ready, choose pursue:pickup_target:<handle> before continue/new inspection waypoint; "
                "the executor performs exactly one safe align/approach step and the next turn must re-observe."
            ),
        },
        "rule_baseline_option_id": rule_baseline,
        **selection_metadata,
        "low_confidence_moves": low_confidence_moves,
        "options": options,
}


def build_explore_frontier_cluster_options(
    *,
    exploration: JsonDict,
    move_options: list[JsonDict],
    explore_plan: JsonDict | None = None,
    max_clusters: int = 4,
) -> list[JsonDict]:
    """Build LLM-facing navigation goal options.

    A cluster option is the target-level decision.  It still carries a resolved
    one-step action so execute_option can validate and execute exactly one
    physical action for the current turn.
    """

    move_by_action = {
        str(option.get("action") or ""): option
        for option in move_options
        if option.get("kind") == "move_action" and option.get("executable_now") is True
    }
    action_effects = as_dict(exploration.get("action_effects"))
    plan = as_dict(explore_plan)
    active_goal = as_dict(plan.get("active_frontier_goal"))
    active_route = as_dict(plan.get("active_route"))
    active_route_step = as_dict(active_route.get("route_step"))
    current_pose = as_dict(exploration.get("current_pose")) or as_dict(plan.get("current_pose"))
    current_cell = str(current_pose.get("cell") or "").strip()
    current_heading = str(current_pose.get("heading") or "").strip()
    result: list[JsonDict] = []
    seen_cells: set[str] = set()

    def direct_step_option_for_cell(cell: str) -> JsonDict:
        action_order = ("MoveAhead", "RotateLeft", "RotateRight", "MoveLeft", "MoveRight", "MoveBack")
        ranked: list[tuple[int, JsonDict]] = []
        for option in move_options:
            effect = as_dict(option.get("exploration_effect"))
            if str(effect.get("target_cell") or "").strip() != cell:
                continue
            if effect.get("enters_frontier") is not True and effect.get("toward_frontier") is not True:
                continue
            action = str(option.get("action") or "")
            try:
                rank = action_order.index(action)
            except ValueError:
                rank = len(action_order)
            ranked.append((rank, option))
        if not ranked:
            return {}
        ranked.sort(key=lambda item: item[0])
        return as_dict(ranked[0][1])

    def route_step_matches_current_pose(route_step: JsonDict) -> bool:
        if not route_step:
            return False
        step_cell = str(route_step.get("current_cell") or "").strip()
        step_heading = str(route_step.get("current_heading") or "").strip()
        if step_cell and current_cell and step_cell != current_cell:
            return False
        if step_heading and current_heading and step_heading != current_heading:
            return False
        return True

    def append_cluster(
        *,
        cell: str,
        action: str,
        target: JsonDict,
        step_effect: JsonDict | None = None,
        step_option: JsonDict | None = None,
        route_step: JsonDict | None = None,
        active_route_payload: JsonDict | None = None,
    ) -> None:
        if len(result) >= max(1, int(max_clusters)):
            return
        cell = str(cell or "").strip()
        action = str(action or "").strip()
        if not cell or not action or cell in seen_cells:
            return
        step_option = as_dict(step_option)
        route_step = as_dict(route_step)
        active_route_payload = as_dict(active_route_payload)
        seen_cells.add(cell)
        route_id = active_route_payload.get("route_id") or route_step.get("route_id")
        result.append(
            clean_empty(
                {
                    "option_id": frontier_cluster_option_id(cell),
                    "kind": "explore_frontier_cluster",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": action,
                    "executable_now": True,
                    "decision_level": "goal",
                    "llm_priority": "primary",
                    "fallback_only": False,
                    "allowed_when": "select_navigation_goal_cluster; executor_resolves_one_safe_step",
                    "one_step_only": True,
                    "resolved_step_option_id": step_option.get("option_id"),
                    "frontier_cluster_target": clean_empty(target),
                    "active_route": clean_empty(
                        {
                            "route_id": route_id,
                            "goal_cell": active_route_payload.get("goal_cell") or route_step.get("goal_cell") or cell,
                            "goal_type": active_route_payload.get("goal_type"),
                            "path_length": active_route_payload.get("path_length"),
                            "distance_to_goal": active_route_payload.get("distance_to_goal"),
                            "status": active_route_payload.get("status"),
                        }
                    ),
                    "route_step": clean_empty(
                        {
                            "route_id": route_id,
                            "step_index": route_step.get("step_index"),
                            "action": route_step.get("action") or action,
                            "current_cell": route_step.get("current_cell"),
                            "current_heading": route_step.get("current_heading"),
                            "target_cell": route_step.get("target_cell"),
                            "next_cell": route_step.get("next_cell"),
                            "goal_cell": route_step.get("goal_cell") or cell,
                            "desired_heading": route_step.get("desired_heading"),
                            "heading_after_action": route_step.get("heading_after_action"),
                            "turns_remaining": route_step.get("turns_remaining"),
                            "progress_effect": route_step.get("progress_effect"),
                            "path_remaining": route_step.get("path_remaining"),
                            "source": "active_route" if route_step else "frontier_cluster_target",
                        }
                    ),
                    "exploration_effect": as_dict(step_effect)
                    or as_dict(step_option.get("exploration_effect")),
                    "reason": (
                        f"Select frontier cluster {cell} as the navigation goal; "
                        f"executor will perform one validated {action} step this turn."
                    ),
                    "safety_source": step_option.get("safety_source")
                    or ("active-route-costmap" if route_step else "navigation-costmap"),
                }
            )
        )

    active_cell = str(active_goal.get("cell") or active_route.get("goal_cell") or "").strip()
    active_action = str(active_route_step.get("action") or active_route.get("next_action") or active_goal.get("next_action") or "").strip()
    if active_cell and active_action:
        direct_step_option = direct_step_option_for_cell(active_cell)
        active_route_step_usable = route_step_matches_current_pose(active_route_step)
        resolved_action = str(direct_step_option.get("action") or "").strip()
        if not resolved_action and active_route_step_usable:
            resolved_action = active_action
        if not resolved_action:
            resolved_action = active_action if active_action in move_by_action else ""
        append_cluster(
            cell=active_cell,
            action=resolved_action,
            target={
                "cell": active_cell,
                "cluster_cells": active_goal.get("cluster_cells") or active_route.get("cluster_cells"),
                "cluster_size": active_goal.get("cluster_size"),
                "distance_steps": active_goal.get("last_distance") or active_route.get("distance_to_goal"),
                "planner": active_goal.get("planner"),
                "plan_status": active_goal.get("plan_status"),
                "route_cost": active_goal.get("route_cost"),
                "score": active_goal.get("objective_score"),
                "reason": active_goal.get("reason") or "active_frontier_goal",
                "source": "active_frontier_goal",
                "route_locked": active_route.get("status") == "active",
                "route_step_reused": active_route_step_usable,
            },
            step_option=direct_step_option or move_by_action.get(resolved_action),
            step_effect=action_effects.get(resolved_action),
            route_step=active_route_step if active_route_step_usable else {},
            active_route_payload=active_route,
        )

    for raw_candidate in as_list(exploration.get("frontier_candidates")):
        candidate = as_dict(raw_candidate)
        cell = str(candidate.get("cell") or "").strip()
        direct_step_option = direct_step_option_for_cell(cell)
        action = str(direct_step_option.get("action") or candidate.get("first_action_hint") or candidate.get("first_step_action") or "").strip()
        if not cell or action not in move_by_action:
            continue
        step_option = as_dict(direct_step_option or move_by_action.get(action))
        append_cluster(
            cell=cell,
            action=action,
            step_option=step_option,
            step_effect=action_effects.get(action) or step_option.get("exploration_effect"),
            target={
                "cell": cell,
                "cluster_size": candidate.get("cluster_size"),
                "distance_steps": candidate.get("distance_steps"),
                "direction_from_current": candidate.get("direction_from_current"),
                "score": candidate.get("score"),
                "unknown_neighbor_count": candidate.get("unknown_neighbor_count"),
                "planner_selected": candidate.get("planner_selected"),
                "reasons": candidate.get("reasons"),
                "first_action_policy": candidate.get("first_action_policy"),
                "first_step_action": candidate.get("first_step_action"),
                "first_step_target_cell": candidate.get("first_step_target_cell"),
                "route_risk": candidate.get("route_risk"),
                "source": "frontier_candidate",
            },
        )
    return result


def build_explore_route_step_options(
    *,
    explore_plan: JsonDict,
    move_options: list[JsonDict],
) -> list[JsonDict]:
    active_route = as_dict(explore_plan.get("active_route"))
    route_step = as_dict(active_route.get("route_step"))
    if active_route.get("status") != "active" or route_step.get("status") != "active":
        return []
    action = str(route_step.get("action") or active_route.get("next_action") or "").strip()
    if not action:
        return []
    move_by_action = {
        str(option.get("action") or ""): option
        for option in move_options
        if option.get("kind") == "move_action" and option.get("executable_now") is True
    }
    step_option = as_dict(move_by_action.get(action))
    route_id = str(active_route.get("route_id") or route_step.get("route_id") or "route")
    step_index = route_step.get("step_index", active_route.get("step_index", 0))
    safety_source = step_option.get("safety_source") if step_option else "active-route-costmap"
    return [
        clean_empty(
            {
                "option_id": route_step_option_id(route_id, step_index),
                "kind": "explore_route_step",
                "physical_action": True,
                "tool": "move-robot",
                "action": action,
                "executable_now": True,
                "decision_level": "goal",
                "llm_priority": "primary",
                "fallback_only": False,
                "allowed_when": "continue_committed_active_frontier_route",
                "one_step_only": True,
                "resolved_step_option_id": step_option.get("option_id"),
                "navigation_core_source": "active_route",
                "route_step": {
                    "route_id": route_id,
                    "step_index": step_index,
                    "action": action,
                    "current_cell": route_step.get("current_cell"),
                    "current_heading": route_step.get("current_heading"),
                    "target_cell": route_step.get("target_cell"),
                    "next_cell": route_step.get("next_cell"),
                    "goal_cell": route_step.get("goal_cell") or active_route.get("goal_cell"),
                    "desired_heading": route_step.get("desired_heading"),
                    "heading_after_action": route_step.get("heading_after_action"),
                    "turns_remaining": route_step.get("turns_remaining"),
                    "progress_effect": route_step.get("progress_effect"),
                    "path_remaining": route_step.get("path_remaining"),
                    "source": "active_route",
                },
                "active_route": {
                    "route_id": route_id,
                    "goal_cell": active_route.get("goal_cell"),
                    "goal_type": active_route.get("goal_type"),
                    "path_length": active_route.get("path_length"),
                    "distance_to_goal": active_route.get("distance_to_goal"),
                },
                "exploration_effect": as_dict(step_option.get("exploration_effect"))
                or clean_empty(
                    {
                        "action": action,
                        "effect": route_step.get("progress_effect"),
                        "target_cell": route_step.get("target_cell") or route_step.get("next_cell"),
                        "next_cell": route_step.get("next_cell"),
                        "goal_cell": route_step.get("goal_cell") or active_route.get("goal_cell"),
                    }
                ),
                "reason": (
                    f"Continue committed route {route_id} toward frontier "
                    f"{active_route.get('goal_cell')}; execute one validated {action} step."
                ),
                "safety_source": safety_source,
            }
        )
    ]


def build_explore_waypoint_options(
    *,
    explore_plan: JsonDict,
    move_options: list[JsonDict],
    max_waypoints: int = 3,
) -> list[JsonDict]:
    move_by_action = {
        str(option.get("action") or ""): option
        for option in move_options
        if option.get("kind") == "move_action" and option.get("executable_now") is True
    }
    result: list[JsonDict] = []
    seen_cells: set[str] = set()
    for raw_candidate in as_list(explore_plan.get("waypoint_candidates")):
        candidate = as_dict(raw_candidate)
        cell = str(candidate.get("cell") or "").strip()
        action = str(candidate.get("action") or "").strip()
        if not cell or not action:
            continue
        step_option = as_dict(move_by_action.get(action))
        if not step_option:
            continue
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        result.append(
            clean_empty(
                {
                    "option_id": waypoint_option_id(cell),
                    "kind": "explore_waypoint",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": action,
                    "executable_now": True,
                    "decision_level": "goal",
                    "llm_priority": "primary",
                    "fallback_only": False,
                    "allowed_when": (
                        "safe_translation_waypoint_for_exploration_or_rotation_loop_escape"
                    ),
                    "one_step_only": True,
                    "resolved_step_option_id": step_option.get("option_id"),
                    "waypoint_target": {
                        "cell": cell,
                        "source": "explore_planner",
                        "purpose": candidate.get("purpose"),
                        "score": candidate.get("score"),
                        "target_state": candidate.get("target_state"),
                        "target_visited": candidate.get("target_visited"),
                        "target_recent": candidate.get("target_recent"),
                        "nearest_frontier_distance_delta": candidate.get(
                            "nearest_frontier_distance_delta"
                        ),
                        "active_frontier_goal_cell": candidate.get("active_frontier_goal_cell"),
                        "active_frontier_distance_delta": candidate.get("active_frontier_distance_delta"),
                        "coverage_patrol_target_cell": candidate.get("coverage_patrol_target_cell"),
                        "coverage_patrol_distance_delta": candidate.get("coverage_patrol_distance_delta"),
                        "reasons": candidate.get("reasons"),
                    },
                    "exploration_effect": as_dict(step_option.get("exploration_effect")),
                    "reason": (
                        f"Move one validated {action} step toward waypoint {cell}; "
                        "used to make exploration progress instead of repeating rotation-only goals."
                    ),
                    "safety_source": step_option.get("safety_source"),
                }
            )
        )
        if len(result) >= max(1, int(max_waypoints)):
            break
    return result


def build_explore_frontier_options(
    *,
    exploration: JsonDict,
    move_options: list[JsonDict],
    explore_plan: JsonDict | None = None,
    max_frontiers: int = 4,
) -> list[JsonDict]:
    move_by_action = {
        str(option.get("action") or ""): option
        for option in move_options
        if option.get("kind") == "move_action" and option.get("executable_now") is True
    }
    action_effects = as_dict(exploration.get("action_effects"))
    plan = as_dict(explore_plan)
    suppressed = {
        str(as_dict(item).get("cell") or "").strip(): as_dict(item)
        for item in as_list(plan.get("suppressed_frontiers"))
        if str(as_dict(item).get("cell") or "").strip()
    }
    suppress_when_escape_exists = bool(as_list(plan.get("waypoint_candidates"))) and bool(suppressed)
    result: list[JsonDict] = []
    seen_ids: set[str] = set()
    seen_cells: set[str] = set()

    def append_frontier_option(
        *,
        cell: str,
        action: str,
        step_option: JsonDict,
        target: JsonDict,
        step_effect: JsonDict,
    ) -> None:
        if len(result) >= max(1, int(max_frontiers)):
            return
        oid = frontier_option_id(cell)
        if oid in seen_ids or cell in seen_cells:
            return
        suppression = as_dict(suppressed.get(cell))
        if suppression and suppress_when_escape_exists and str(suppression.get("severity") or "") == "block":
            return
        seen_ids.add(oid)
        seen_cells.add(cell)
        result.append(
            clean_empty(
                {
                    "option_id": oid,
                    "kind": "explore_frontier",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": action,
                    "executable_now": True,
                    "decision_level": "goal",
                    "llm_priority": "primary",
                    "fallback_only": False,
                    "allowed_when": "no_immediate_pick_place_goal_or_as_exploration_target",
                    "one_step_only": True,
                    "resolved_step_option_id": step_option.get("option_id"),
                    "frontier_target": target,
                    "exploration_effect": step_effect,
                    "reason": (
                        f"Explore toward frontier {cell}; executor will perform one validated "
                        f"{action} step this turn."
                    ),
                    "safety_source": step_option.get("safety_source"),
                }
            )
        )

    for raw_candidate in as_list(exploration.get("frontier_candidates")):
        candidate = as_dict(raw_candidate)
        cell = str(candidate.get("cell") or "").strip()
        action = str(candidate.get("first_action_hint") or "").strip()
        if not cell or action not in move_by_action:
            continue
        step_option = move_by_action[action]
        step_effect = as_dict(action_effects.get(action)) or as_dict(step_option.get("exploration_effect"))
        append_frontier_option(
            cell=cell,
            action=action,
            step_option=step_option,
            target={
                "cell": cell,
                "distance_steps": candidate.get("distance_steps"),
                "direction_from_current": candidate.get("direction_from_current"),
                "score": candidate.get("score"),
                "unknown_neighbor_count": candidate.get("unknown_neighbor_count"),
                "cluster_size": candidate.get("cluster_size"),
                "planner_selected": candidate.get("planner_selected"),
                "reasons": candidate.get("reasons"),
                "first_action_policy": candidate.get("first_action_policy"),
                "first_step_action": candidate.get("first_step_action"),
                "first_step_target_cell": candidate.get("first_step_target_cell"),
                "first_step_enters_recent_cell": candidate.get("first_step_enters_recent_cell"),
                "first_step_enters_visited_cell": candidate.get("first_step_enters_visited_cell"),
                "route_risk": candidate.get("route_risk"),
                "source": "frontier_candidate",
            },
            step_effect=step_effect,
        )
    for step_option in move_options:
        if len(result) >= max(1, int(max_frontiers)):
            break
        action = str(step_option.get("action") or "").strip()
        step_effect = as_dict(step_option.get("exploration_effect"))
        if step_effect.get("enters_frontier") is not True:
            continue
        cell = str(step_effect.get("target_cell") or "").strip()
        if not cell:
            continue
        append_frontier_option(
            cell=cell,
            action=action,
            step_option=step_option,
            target={
                "cell": cell,
                "source": "move_exploration_effect",
                "target_state": step_effect.get("target_state"),
                "best_frontier_cell": step_effect.get("best_frontier_cell"),
                "best_frontier_distance_delta": step_effect.get("best_frontier_distance_delta"),
                "nearest_frontier_distance_delta": step_effect.get("nearest_frontier_distance_delta"),
                "reasons": ["move_enters_frontier"],
            },
            step_effect=step_effect,
        )
    return result


def choose_rule_baseline_option(
    options: list[JsonDict],
    *,
    task: JsonDict,
    perception: JsonDict,
    worklist: JsonDict,
) -> str:
    holding = bool(as_dict(worklist.get("held_object")).get("holding_object"))
    if holding:
        for option in options:
            if str(option.get("option_id", "")).startswith("place:"):
                return str(option["option_id"])
        for option in options:
            if str(option.get("option_id", "")).startswith("place_precheck:"):
                return str(option["option_id"])
    else:
        for option in options:
            if str(option.get("option_id", "")).startswith("pick:"):
                return str(option["option_id"])
    for option in options:
        if option.get("kind") == "recovery_action":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "orient_waypoint_floor_scan":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "continue_active_waypoint_goal":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "explore_inspection_waypoint":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "explore_frontier_cluster":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "explore_route_step":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "explore_waypoint":
            return str(option["option_id"])
    for option in options:
        if option.get("kind") == "explore_frontier":
            return str(option["option_id"])
    recommended_action = str(perception.get("recommended_action") or "")
    for option in options:
        if option.get("kind") == "move_action" and option.get("action") == recommended_action:
            return str(option["option_id"])
    for action in ("MoveAhead", "RotateLeft", "RotateRight", "MoveLeft", "MoveRight"):
        for option in options:
            if option.get("kind") == "move_action" and option.get("action") == action:
                return str(option["option_id"])
    return "observe:refresh"


def build_done_readiness(
    *,
    task: JsonDict,
    perception: JsonDict,
    worklist: JsonDict,
    navigation: JsonDict,
) -> JsonDict:
    blockers: list[JsonDict] = []
    held = as_dict(worklist.get("held_object"))
    current_view = as_dict(worklist.get("current_view"))
    coverage = as_dict(navigation.get("coverage"))
    structured_ok = bool(perception.get("structured_perception_available"))
    if not structured_ok:
        blockers.append(
            {
                "type": "perception_not_ready",
                "required_next": "observe:refresh",
                "reason": "No successful structured perception is available.",
            }
        )
    if held.get("holding_object") is True:
        blockers.append(
            {
                "type": "holding_object",
                "required_next": "place-object or search_receptacle",
                "reason": "A pickup/place subtask is incomplete while the robot is holding an object.",
            }
        )
    actionable_pickups = [
        item
        for item in as_list(current_view.get("pickup_candidates"))
        if as_dict(item).get("actionability", {}).get("pickup_now") is True
    ]
    if actionable_pickups and not held.get("holding_object"):
        blockers.append(
            {
                "type": "visible_pickup_candidates",
                "required_next": "pick-object or align_pickup_target",
                "candidate_count": len(actionable_pickups),
            }
        )
    if perception.get("frontier_exists") is True or int(coverage.get("frontier_count") or 0) > 0:
        blockers.append(
            {
                "type": "frontier_remaining",
                "required_next": "explore",
                "frontier_count": coverage.get("frontier_count"),
            }
        )
    room_complete = bool(task.get("room_complete"))
    can_finish = room_complete and not blockers
    return {
        "schema": "robot_cleaner_done_readiness_v1",
        "advisory_only": True,
        "can_finish": can_finish,
        "blockers": blockers,
    }


def parse_source_time(info: JsonDict) -> datetime | None:
    value = info.get("last_modified")
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def build_consistency_warnings(
    *,
    service_state: JsonDict,
    perception: JsonDict,
    sources: dict[str, LoadedJson],
) -> list[JsonDict]:
    warnings: list[JsonDict] = []
    service_holding = bool_or_none(service_state.get("holding_object"))
    perception_holding = bool_or_none(perception.get("holding_object_context"))
    if (
        service_holding is not None
        and perception_holding is not None
        and service_holding != perception_holding
    ):
        warning = {
            "type": "holding_state_mismatch",
            "service_task_state_holding_object": service_holding,
            "perception_holding_object_context": perception_holding,
            "recovery_option_id": "observe:refresh",
        }
        if service_holding is False and perception_holding is True:
            warning["blocking"] = False
            warning["severity"] = "advisory"
            warning["reason"] = "service_task_state_is_authoritative_after_place"
        warnings.append(warning)
    source_times = [
        (name, parse_source_time(loaded.info))
        for name, loaded in sources.items()
        if loaded.info.get("loaded") and name not in {"place_precheck_cache"}
    ]
    dated = [(name, value) for name, value in source_times if value is not None]
    if len(dated) >= 2:
        oldest_name, oldest_time = min(dated, key=lambda item: item[1])
        newest_name, newest_time = max(dated, key=lambda item: item[1])
        skew_seconds = (newest_time - oldest_time).total_seconds()
        if skew_seconds > 300:
            warning = classify_source_time_skew(
                oldest_name=oldest_name,
                newest_name=newest_name,
                sources=sources,
            )
            warning["skew_seconds"] = round(skew_seconds, 3)
            warnings.append(warning)
    return warnings


def classify_source_time_skew(
    *,
    oldest_name: str,
    newest_name: str,
    sources: dict[str, LoadedJson],
) -> JsonDict:
    warning: JsonDict = {
        "type": "source_time_skew",
        "oldest_source": oldest_name,
        "newest_source": newest_name,
        "recovery_option_id": "observe:refresh",
    }

    global_plan_source = sources.get("global_plan")
    global_plan_data = global_plan_source.data if global_plan_source is not None else {}
    if (
        oldest_name == "global_plan"
        and str(as_dict(global_plan_data).get("status") or "") == "reset"
    ):
        warning["blocking"] = False
        warning["severity"] = "advisory"
        warning["reason"] = "stale_reset_global_plan_not_required_for_local_option"
        return warning

    if newest_name in VOLATILE_OBSERVATION_SOURCES and oldest_name in DURABLE_STATE_SOURCES:
        warning["blocking"] = False
        warning["severity"] = "advisory"
        warning["reason"] = "fresh_observation_with_unchanged_durable_state"
        return warning

    if oldest_name in VOLATILE_OBSERVATION_SOURCES:
        warning["blocking"] = True
        warning["severity"] = "blocking"
        warning["reason"] = "stale_observation_source"
        return warning

    warning["blocking"] = False
    warning["severity"] = "advisory"
    warning["reason"] = "durable_state_sources_update_only_on_state_changes"
    return warning


def clean_empty(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cleaned = clean_empty(item)
            if cleaned in (None, "", [], {}):
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        return [clean_empty(item) for item in value if clean_empty(item) not in (None, "", [], {})]
    return value


def find_forbidden_keys(value: Any, *, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = str(key).lower()
            if key_norm in FORBIDDEN_AGENT_VIEW_KEYS:
                hits.append(f"{prefix}.{key}" if prefix else str(key))
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            hits.extend(find_forbidden_keys(item, prefix=child_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(find_forbidden_keys(item, prefix=f"{prefix}[{index}]"))
    return hits


def build_context(args: argparse.Namespace) -> JsonDict:
    memory_dir = Path(args.memory_dir)
    perception_path = Path(args.perception_json)
    if not perception_path.is_absolute():
        perception_path = REPO_ROOT / perception_path

    try:
        map_snapshot = load_map_backend(memory_dir).load_snapshot()
    except BackendUnavailableError as exc:
        return {
            "status": "error",
            "result_type": "error_map_backend_unavailable",
            "schema": DECISION_CONTEXT_SCHEMA,
            "schema_version": 1,
            "generated_at": now_iso(),
            "error_message": str(exc),
        }

    sources = {
        "mission_state": read_json(memory_dir / "mission-state.json"),
        "room_state": read_json(memory_dir / "room-state.json"),
        "patrol_state": read_json(memory_dir / "patrol-state.json"),
        "service_task_state": read_json(memory_dir / "service-task-state.json"),
        "navigation_costmap": read_json(memory_dir / "navigation-costmap.json"),
        "position_map": LoadedJson(
            path=memory_dir / "position-map.json",
            data=map_snapshot.to_position_status(),
            info=map_snapshot.source_loaded_info(),
        ),
        "object_memory": read_json(memory_dir / "object-memory.json"),
        "global_plan": read_json(memory_dir / "global-plan.json"),
        "place_precheck_cache": read_json(memory_dir / PLACE_PRECHECK_CACHE_NAME),
        "perception": read_json(perception_path),
    }

    mission = sources["mission_state"].data
    room = sources["room_state"].data
    patrol = sources["patrol_state"].data
    service_state = sources["service_task_state"].data
    costmap = sources["navigation_costmap"].data
    position_map = sources["position_map"].data
    object_memory = sources["object_memory"].data
    global_plan = sources["global_plan"].data
    place_precheck_cache = sources["place_precheck_cache"].data
    perception = sources["perception"].data

    task = build_task_summary(
        requested_mode=args.task_mode,
        mission=mission,
        patrol=patrol,
        room=room,
        service_state=service_state,
    )
    perception_summary = build_perception_summary(perception, sources["perception"].info)
    worklist = build_worklist(
        perception,
        object_memory,
        service_state,
        limit=max(1, args.max_candidates),
    )
    apply_place_precheck_cache(
        worklist,
        place_precheck_cache,
        perception_source_info=sources["perception"].info,
    )
    navigation = build_navigation_summary(
        perception=perception,
        costmap=costmap,
        room=room,
        position_map=position_map,
        global_plan=global_plan,
        map_backend_summary=map_snapshot.public_summary(),
    )
    current_pose = as_dict(position_map.get("pose"))
    current_cell = str(current_pose.get("cell") or position_map.get("last_cell") or room.get("last_cell") or "")
    inspection_waypoints = build_inspection_waypoints(map_snapshot)
    coverage_waypoints = normalize_coverage_waypoint_state(
        inspection_waypoints,
        as_dict(room.get("coverage_waypoints")),
        current_cell=current_cell,
    )
    exploration = build_exploration_context(
        position_map=position_map,
        navigation_costmap=costmap,
        global_plan=global_plan,
        max_frontier_candidates=max(3, args.max_candidates),
    )
    explore_plan = build_explore_plan(
        position_map=position_map,
        navigation_costmap=costmap,
        exploration=exploration,
    )
    navigation_core = build_navigation_core_state(
        navigation=navigation,
        exploration=exploration,
        explore_plan=explore_plan,
    )
    done_readiness = build_done_readiness(
        task=task,
        perception=perception_summary,
        worklist=worklist,
        navigation=navigation,
    )
    option_set = build_options(
        task=task,
        perception=perception_summary,
        costmap=costmap,
        exploration=exploration,
        explore_plan=explore_plan,
        navigation_core=navigation_core,
        coverage_waypoints=coverage_waypoints,
        position_map=position_map,
        room=room,
        worklist=worklist,
        done_readiness=done_readiness,
        max_options=max(1, args.max_options),
    )
    consistency_warnings = build_consistency_warnings(
        service_state=service_state,
        perception=perception,
        sources=sources,
    )
    context: JsonDict = {
        "status": "success",
        "result_type": "decision_context_built",
        "schema": DECISION_CONTEXT_SCHEMA,
        "schema_version": 1,
        "generated_at": now_iso(),
        "public_agent_view": True,
        "private_fields_excluded": True,
        "context_policy": {
            "full_maps_not_included": True,
            "full_memory_history_not_included": True,
            "model_should_choose_option_id_only": True,
            "executor_remains_authoritative": True,
        },
        "context_lifecycle": {
            "valid_for_execution": True,
            "stale_after_option_execution": False,
            "required_next_after_execution": "robot_cleaner_prepare_decision_turn",
        },
        "task": task,
        "perception": perception_summary,
        "navigation": navigation,
        "navigation_core": navigation_core,
        "inspection_waypoints": inspection_waypoints,
        "coverage_waypoints": coverage_waypoints,
        "exploration": exploration,
        "explore_plan": explore_plan,
        "worklist": worklist,
        "done_readiness": done_readiness,
        "option_set": option_set,
        "consistency_warnings": consistency_warnings,
        "guardrails": [
            "Run get-vision and structured perception before physical actions.",
            "Execute at most one physical action per decision turn.",
            "Do not move ahead when latest costmap or perception marks the path blocked.",
            "Do not pick or place unless the selected candidate remains visible and executor-ready.",
            "A successful place-object completes one subtask, not the whole room patrol.",
        ],
        "sources": {name: loaded.info for name, loaded in sources.items()},
    }
    forbidden = find_forbidden_keys(context)
    context["forbidden_private_fields_absent"] = not forbidden
    if forbidden:
        context["status"] = "error"
        context["result_type"] = "error_forbidden_agent_view_field"
        context["forbidden_field_paths"] = forbidden
    return context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact public decision context for OpenClaw/LLM control."
    )
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR), help="Directory with memory/*.json files.")
    parser.add_argument(
        "--perception-json",
        default=str(DEFAULT_PERCEPTION_JSON),
        help="Latest structured perception JSON. Defaults to memory/yolo-current-rgbd.json.",
    )
    parser.add_argument("--output", default="", help="Optional path to write the context JSON.")
    parser.add_argument("--max-candidates", type=int, default=6, help="Maximum candidates per worklist section.")
    parser.add_argument("--max-options", type=int, default=12, help="Maximum options exposed to the model.")
    parser.add_argument(
        "--task-mode",
        choices=("auto", "tidy", "clean"),
        default="auto",
        help="Task mode label to expose in the context.",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "compact"),
        default="pretty",
        help="Stdout JSON formatting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    context = build_context(args)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        context["output_path"] = display_path(output_path)
    json_print(context, compact=args.format == "compact")
    return 0 if context.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
