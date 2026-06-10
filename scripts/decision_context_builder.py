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
CAMERA_ACTIONS = ("LookUp", "LookDown")
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
                "blocked_cell_count": number_or_none(item.get("blocked_cell_count")),
            }
        )
    stats = as_dict(position_map.get("stats"))
    frontiers = as_list(position_map.get("frontiers"))
    return clean_empty(
        {
            "pose": clean_empty(as_dict(position_map.get("pose"))),
            "coverage": clean_empty(
                {
                    "coverage_estimate": number_or_none(room.get("coverage_estimate")),
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


def build_options(
    *,
    task: JsonDict,
    perception: JsonDict,
    costmap: JsonDict,
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
            options.append(
                {
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
            )

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
        },
        "rule_baseline_option_id": rule_baseline,
        "options": options,
    }


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

    sources = {
        "mission_state": read_json(memory_dir / "mission-state.json"),
        "room_state": read_json(memory_dir / "room-state.json"),
        "patrol_state": read_json(memory_dir / "patrol-state.json"),
        "service_task_state": read_json(memory_dir / "service-task-state.json"),
        "navigation_costmap": read_json(memory_dir / "navigation-costmap.json"),
        "position_map": read_json(memory_dir / "position-map.json"),
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
        "task": task,
        "perception": perception_summary,
        "navigation": navigation,
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
