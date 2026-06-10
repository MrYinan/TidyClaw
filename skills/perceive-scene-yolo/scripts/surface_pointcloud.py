#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current-frame RGB-D point-cloud support-plane extraction.

This module is intentionally independent from the YOLO adapter. YOLO provides
semantic parent boxes and blockers; this file uses the current depth frame to
extract local horizontal support planes with Open3D and turns plane clusters
into the same surface_region contract consumed by patrol_runner.
"""

from __future__ import annotations

import math
import os
import re
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]
#可作为父对象的台面  parent
POINTCLOUD_PARENT_LABELS = {
    "counter_top",
    "countertop",
    "dining_table",
    "diningtable",
    "coffee_table",
    "coffeetable",
    "side_table",
    "sidetable",
    "table",
}

POINTCLOUD_LOW_PRIORITY_PARENTS = {"sink", "stove"}
#food
POINTCLOUD_FOOD_LABELS = {
    "apple",
    "banana",
    "lettuce",
    "orange",
    "potato",
    "tomato",
}

POINTCLOUD_PICKUP_LABELS = POINTCLOUD_FOOD_LABELS | {
    "book",
    "bottle",
    "bowl",
    "butter_knife",
    "cup",
    "kettle",
    "mug",
    "pan",
    "plate",
    "pot",
    "remote",
    "remote_control",
    "soap_bottle",
    "vase",
}
"""
strong blocker：
苹果、杯子、碗、锅、盘子等真正占台面的物体

structural core blocker：
sink / stove，结构性强阻挡物

weak context blocker：
cabinet / drawer / dishwasher，通常是背景/柜体，不应该轻易挡掉台面
"""
POINTCLOUD_STRONG_BLOCKER_LABELS = POINTCLOUD_PICKUP_LABELS
POINTCLOUD_STRUCTURAL_CORE_BLOCKERS = {"sink", "stove"}
POINTCLOUD_WEAK_CONTEXT_BLOCKERS = {"cabinet", "drawer", "dishwasher"}
POINTCLOUD_BLOCKING_LABELS = (
    POINTCLOUD_STRONG_BLOCKER_LABELS
    | POINTCLOUD_STRUCTURAL_CORE_BLOCKERS
    | POINTCLOUD_WEAK_CONTEXT_BLOCKERS
)
POINTCLOUD_COMPLETION_SOURCE = "pointcloud_plane_completion"
POINTCLOUD_GRID_COMPLETION_SOURCE = "pointcloud_plane_grid_completion"
POINTCLOUD_COUNTERTOP_LABELS = {"counter_top", "countertop"}
POINTCLOUD_TABLELIKE_LABELS = POINTCLOUD_PARENT_LABELS
POINTCLOUD_GRID_RECOVERABLE_REJECTION_REASONS = {
    "touches_image_edge",
    "touches_parent_edge",
    "too_close",
    "too_far",
}
POINTCLOUD_GRID_INTRINSIC_REJECTION_REASONS = {
    "too_small",
    "too_narrow",
    "too_shallow",
    "too_few_points",
    "not_horizontal",
    "height_out_of_range",
    "thin_region",
    "edge_only_region",
    "raised_object_top",
}


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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def pointcloud_ready_distance_range_m() -> Tuple[float, float]:
    """Distance band for a pointcloud surface that can be tried immediately."""
    min_distance = max(0.0, env_float("ROBOT_PC_REGION_MIN_DISTANCE_M", 0.45))
    max_distance = max(min_distance, env_float("ROBOT_PC_REGION_MAX_DISTANCE_M", 0.95))
    return min_distance, max_distance


def support_surface_height_range(parent_label: Any) -> Tuple[float, float]:
    parent = normalize_label(parent_label)
    global_min_raw = os.getenv("ROBOT_PC_SURFACE_MIN_HEIGHT_M")
    global_min = env_float("ROBOT_PC_SURFACE_MIN_HEIGHT_M", 0.55) if global_min_raw is not None else None
    if parent in POINTCLOUD_COUNTERTOP_LABELS:
        default_min = float(global_min) if global_min is not None else 0.20
        min_h = env_float("ROBOT_PC_COUNTERTOP_SURFACE_MIN_HEIGHT_M", default_min)
    elif parent in POINTCLOUD_TABLELIKE_LABELS:
        default_min = float(global_min) if global_min is not None else 0.40
        min_h = env_float("ROBOT_PC_TABLE_SURFACE_MIN_HEIGHT_M", default_min)
    else:
        min_h = env_float("ROBOT_PC_SURFACE_MIN_HEIGHT_M", 0.55)
    max_h = env_float("ROBOT_PC_SURFACE_MAX_HEIGHT_M", 1.15)
    return float(min_h), float(max_h)


def allow_countertop_parent_top_edge(parent_label: Any) -> bool:
    return bool(
        normalize_label(parent_label) in POINTCLOUD_COUNTERTOP_LABELS
        and env_bool("ROBOT_PC_COUNTERTOP_ALLOW_PARENT_TOP_EDGE", True)
    )


def support_surface_ideal_height(parent_label: Any) -> float:
    parent = normalize_label(parent_label)
    if parent in POINTCLOUD_COUNTERTOP_LABELS:
        return env_float("ROBOT_PC_COUNTERTOP_IDEAL_HEIGHT_M", 0.45)
    if parent in POINTCLOUD_TABLELIKE_LABELS:
        return env_float("ROBOT_PC_TABLE_IDEAL_HEIGHT_M", 0.62)
    return env_float("ROBOT_PC_SURFACE_IDEAL_HEIGHT_M", 0.85)


def normalize_label(label: Any) -> str:
    value = str(label or "").strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_label_tokens(labels: Optional[Iterable[Any]]) -> List[str]:
    tokens: List[str] = []
    for label in labels or []:
        for part in str(label or "").split(","):
            token = normalize_label(part)
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def depth_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def bbox_xyxy(candidate_or_bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    bbox = candidate_or_bbox.get("bbox") if isinstance(candidate_or_bbox, dict) and "bbox" in candidate_or_bbox else candidate_or_bbox
    if not isinstance(bbox, dict):
        return None
    try:
        x = float(bbox.get("x"))
        y = float(bbox.get("y"))
        w = float(bbox.get("w"))
        h = float(bbox.get("h"))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def candidate_label_matches(candidate: JsonDict, labels: Iterable[str]) -> bool:
    held = {normalize_label(label) for label in labels if normalize_label(label)}
    held_compact = {label.replace("_", "") for label in held}
    if not held:
        return False
    candidates = {
        normalize_label(candidate.get("label")),
        normalize_label(candidate.get("raw_label")),
    }
    candidates_compact = {label.replace("_", "") for label in candidates if label}
    return bool((candidates & held) or (candidates_compact & held_compact))


def held_object_family_for_labels(labels: Iterable[str]) -> str:
    tokens = {normalize_label(label) for label in labels if normalize_label(label)}
    if tokens & POINTCLOUD_FOOD_LABELS:
        return "food"
    if tokens & POINTCLOUD_PICKUP_LABELS:
        return "pickup_target"
    return "unknown"


def held_object_footprint_radius_m(
    *,
    holding_object: bool,
    held_object_labels: Optional[Iterable[str]],
    held_object_family: Optional[str],
) -> float:
    """Return a conservative planar radius for the object being placed.

    Grid obstacle dilation is a Minkowski-style clearance operation: free
    points must leave space for the held object's footprint, not just its
    nominal center. Explicit overrides support future per-task calibration.
    """
    if not holding_object:
        return 0.0
    override = os.getenv("ROBOT_PC_FREE_GRID_HELD_FOOTPRINT_RADIUS_M")
    if override is not None:
        return max(0.0, env_float("ROBOT_PC_FREE_GRID_HELD_FOOTPRINT_RADIUS_M", 0.04))
    labels = normalize_label_tokens(held_object_labels)
    family = normalize_label(held_object_family or "") or held_object_family_for_labels(labels)
    if family == "food":
        return max(0.0, env_float("ROBOT_PC_FREE_GRID_HELD_FOOD_RADIUS_M", 0.035))
    if family == "pickup_target":
        return max(0.0, env_float("ROBOT_PC_FREE_GRID_HELD_PICKUP_RADIUS_M", 0.05))
    return max(0.0, env_float("ROBOT_PC_FREE_GRID_HELD_UNKNOWN_RADIUS_M", 0.04))


def candidate_family(candidate: JsonDict) -> str:
    label = normalize_label(candidate.get("label") or candidate.get("raw_label"))
    if label in POINTCLOUD_FOOD_LABELS:
        return "food"
    if str(candidate.get("task_semantic_class") or "") == "pickup_target" or label in POINTCLOUD_PICKUP_LABELS:
        return "pickup_target"
    return "unknown"


def candidate_ground_distance(candidate: JsonDict) -> Optional[float]:
    for key in ("ground_distance", "distance"):
        value = depth_number(candidate.get(key))
        if value is not None:
            return value
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    for key in ("ground_distance_m", "distance_m"):
        value = depth_number(geometry.get(key))
        if value is not None:
            return value
    center_3d = candidate.get("center_3d") if isinstance(candidate.get("center_3d"), dict) else {}
    for key in ("ground_distance_m", "z"):
        value = depth_number(center_3d.get(key))
        if value is not None:
            return value
    depth = candidate.get("depth") if isinstance(candidate.get("depth"), dict) else {}
    for key in ("ground_distance_m", "distance_m", "median_m"):
        value = depth_number(depth.get(key))
        if value is not None:
            return value
    return None


def candidate_foreground_overlay_score(candidate: JsonDict) -> Tuple[bool, JsonDict]:
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    cx_ratio = depth_number(geometry.get("cx_ratio"), 0.5) or 0.5
    cy_ratio = depth_number(geometry.get("cy_ratio"), candidate.get("center_y_ratio")) or 0.5
    bottom_y_ratio = depth_number(geometry.get("bottom_y_ratio"), candidate.get("bottom_y_ratio")) or 0.0
    area_ratio = depth_number(candidate.get("area_ratio"), geometry.get("area_ratio")) or 0.0
    distance_m = candidate_ground_distance(candidate)
    center_tolerance = env_float(
        "ROBOT_HELD_OBJECT_FOREGROUND_CENTER_TOLERANCE",
        env_float("ROBOT_HELD_OBJECT_IGNORE_CENTER_TOLERANCE", 0.28),
    )
    center_ok = abs(cx_ratio - 0.5) <= center_tolerance
    lower_center_ok = bool(
        center_ok
        and cy_ratio >= env_float("ROBOT_HELD_OBJECT_FOREGROUND_MIN_CENTER_Y_RATIO", 0.55)
        and bottom_y_ratio >= env_float("ROBOT_HELD_OBJECT_FOREGROUND_MIN_BOTTOM_RATIO", 0.65)
    )
    close_ok = bool(
        distance_m is not None
        and center_ok
        and distance_m <= env_float("ROBOT_HELD_OBJECT_FOREGROUND_MAX_GROUND_DISTANCE", 0.55)
    )
    large_ok = area_ratio >= env_float("ROBOT_HELD_OBJECT_FOREGROUND_MIN_AREA_RATIO", 0.003)
    overlay = bool(center_ok and large_ok and (close_ok or lower_center_ok))
    return overlay, {
        "cx_ratio": round(float(cx_ratio), 4),
        "cy_ratio": round(float(cy_ratio), 4),
        "bottom_y_ratio": round(float(bottom_y_ratio), 4),
        "area_ratio": round(float(area_ratio), 6),
        "distance_m": round(float(distance_m), 4) if distance_m is not None else None,
        "center_tolerance": round(float(center_tolerance), 4),
        "center_ok": bool(center_ok),
        "lower_center_ok": bool(lower_center_ok),
        "close_ok": bool(close_ok),
        "large_ok": bool(large_ok),
    }

#判断是不是手里拿着的物体 判断它是不是“画面前景里的手持物体”。主要看：

# 1. 是否靠近图像中心
# 2. 是否在画面下半部分
# 3. 是否距离很近
# 4. bbox 面积是否够大
def candidate_likely_held_object(
    candidate: JsonDict,
    held_object_labels: Iterable[str],
    held_object_family: Optional[str] = None,
) -> bool:
    if str(candidate.get("task_semantic_class") or "") != "pickup_target":
        return False

    overlay, detail = candidate_foreground_overlay_score(candidate)
    if not overlay:
        return False

    held_labels = normalize_label_tokens(held_object_labels)
    held_family = normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels)
    label_match = candidate_label_matches(candidate, held_labels)
    family = candidate_family(candidate)
    family_match = bool(
        held_family == "pickup_target"
        or (held_family == "food" and family == "food")
        or (held_family == family and held_family != "unknown")
    )
    if label_match or family_match or env_bool("ROBOT_HELD_OBJECT_IGNORE_ANY_FOREGROUND_PICKUP", False):
        candidate["likely_held_object_overlay"] = True
        candidate["held_object_overlay_checks"] = detail
        candidate["held_object_family_match"] = bool(family_match)
        return True
    return False


def bbox_overlap_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))

#把 depth 图变成点云
def depth_to_pointcloud(depth: Any, camera: JsonDict, image: Any = None, stride: int = 2) -> JsonDict:
    """Project a depth frame into camera-relative x_right, y_height, z_forward points."""
    import numpy as np  # type: ignore

    if depth is None:
        return {"xyz": np.zeros((0, 3), dtype="float32"), "uv": np.zeros((0, 2), dtype="float32")}
    depth_arr = np.asarray(depth, dtype="float32")
    if depth_arr.ndim != 2:
        return {"xyz": np.zeros((0, 3), dtype="float32"), "uv": np.zeros((0, 2), dtype="float32")}

    h, w = depth_arr.shape[:2]
    stride = max(1, int(stride))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    values = depth_arr[yy, xx]
    valid = np.isfinite(values) & (values > 0.05) & (values < 20.0)
    if not bool(valid.any()):
        return {
            "xyz": np.zeros((0, 3), dtype="float32"),
            "uv": np.zeros((0, 2), dtype="float32"),
            "depth_m": np.zeros((0,), dtype="float32"),
            "image_size": {"w": int(w), "h": int(h)},
            "stride": int(stride),
        }

    u = xx[valid].astype("float32")
    v = yy[valid].astype("float32")
    z_cam = values[valid].astype("float32")

    fx = max(1e-6, float(camera.get("fx", 1.0) or 1.0))
    fy = max(1e-6, float(camera.get("fy", fx) or fx))
    cx = float(camera.get("cx", float(w) / 2.0) or float(w) / 2.0)
    cy = float(camera.get("cy", float(h) / 2.0) or float(h) / 2.0)
    x_cam = (u - cx) * z_cam / fx
    y_cam_up = -(v - cy) * z_cam / fy

    pitch = math.radians(float(camera.get("camera_horizon_deg", 0.0) or 0.0))
    camera_height = float(camera.get("camera_height_m", 0.9) or 0.9)
    height_from_floor = camera_height + y_cam_up * math.cos(pitch) - z_cam * math.sin(pitch)
    ground_forward = y_cam_up * math.sin(pitch) + z_cam * math.cos(pitch)

    xyz = np.stack([x_cam, height_from_floor, ground_forward], axis=1).astype("float32")
    uv = np.stack([u, v], axis=1).astype("float32")
    result: JsonDict = {
        "xyz": xyz,
        "uv": uv,
        "depth_m": z_cam.astype("float32"),
        "source_indices": np.arange(int(xyz.shape[0]), dtype="int64"),
        "image_size": {"w": int(w), "h": int(h)},
        "stride": int(stride),
    }
    if image is not None:
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[0] == h and img.shape[1] == w:
            colors = img[uv[:, 1].astype("int32"), uv[:, 0].astype("int32"), :3].astype("float32")
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            result["rgb"] = colors
    return result


def build_open3d_pointcloud(points: JsonDict) -> Any:
    import numpy as np  # type: ignore
    import open3d as o3d  # type: ignore

    pcd = o3d.geometry.PointCloud()
    xyz = np.asarray(points.get("xyz", []), dtype="float64")
    if xyz.size == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = points.get("rgb")
    if rgb is not None:
        colors = np.asarray(rgb, dtype="float64")
        if colors.shape[0] == xyz.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


def crop_points_by_bbox(points: JsonDict, parent_bbox: JsonDict) -> JsonDict:
    import numpy as np  # type: ignore

    box = bbox_xyxy(parent_bbox)
    xyz = np.asarray(points.get("xyz", []))
    uv = np.asarray(points.get("uv", []))
    if box is None or xyz.size == 0 or uv.size == 0:
        return dict(points, xyz=xyz[:0], uv=uv[:0])
    x1, y1, x2, y2 = box
    mask = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    cropped: JsonDict = {}
    for key, value in points.items():
        if key in {"image_size", "stride"}:
            cropped[key] = value
            continue
        arr = np.asarray(value)
        if arr.shape[:1] == mask.shape[:1]:
            cropped[key] = arr[mask]
        else:
            cropped[key] = value
    return cropped


def shrink_box(box: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    scale = max(0.05, min(1.0, float(scale)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) * scale / 2.0
    half_h = (y2 - y1) * scale / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def expand_box_pixels(
    box: Tuple[float, float, float, float],
    *,
    margin: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, x1 - margin),
        max(0.0, y1 - margin),
        min(float(image_w), x2 + margin),
        min(float(image_h), y2 + margin),
    )


def blocker_category(candidate: JsonDict) -> str:
    label = normalize_label(candidate.get("label") or candidate.get("raw_label"))
    if label in POINTCLOUD_STRUCTURAL_CORE_BLOCKERS:
        return "structural_core"
    if label in POINTCLOUD_WEAK_CONTEXT_BLOCKERS:
        return "weak_context"
    if label in POINTCLOUD_STRONG_BLOCKER_LABELS or str(candidate.get("task_semantic_class") or "") == "pickup_target":
        return "strong"
    return "none"


def occupancy_test_box(
    box: Tuple[float, float, float, float],
    category: str,
) -> Tuple[float, float, float, float]:
    """Return the visual footprint used by both region and grid occupancy.

    Structural blockers such as a stove or sink define hard occupied space on
    a counter. A shared full-footprint default prevents an initial plane check
    from accepting space that grid completion or the executor must later
    reject. Lower scales are retained only as an explicit calibration option.
    """
    if category != "structural_core":
        return box
    scale = max(
        0.05,
        min(1.0, env_float("ROBOT_PC_STRUCTURAL_CORE_BBOX_SCALE", 1.0)),
    )
    return shrink_box(box, scale)


def points_in_bbox(points: Optional[JsonDict], box: Tuple[float, float, float, float]) -> Tuple[Any, Any]:
    import numpy as np  # type: ignore

    if not isinstance(points, dict):
        return np.zeros((0, 3), dtype="float64"), np.zeros((0, 2), dtype="float64")
    xyz = np.asarray(points.get("xyz", []), dtype="float64")
    uv = np.asarray(points.get("uv", []), dtype="float64")
    if xyz.ndim != 2 or uv.ndim != 2 or xyz.shape[0] != uv.shape[0] or xyz.shape[0] == 0:
        return xyz[:0], uv[:0]
    x1, y1, x2, y2 = box
    mask = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
    return xyz[mask], uv[mask]


def held_overlay_record(candidate: JsonDict, *, held_family: str) -> JsonDict:
    label = normalize_label(candidate.get("label") or candidate.get("raw_label")) or "held_object"
    record: JsonDict = {
        "label": label,
        "raw_label": candidate.get("raw_label") or candidate.get("label"),
        "task_semantic_class": candidate.get("task_semantic_class"),
        "reason": "likely_held_object_overlay",
        "held_object_family": held_family or "unknown",
        "overlay_checks": candidate.get("held_object_overlay_checks"),
    }
    box = bbox_xyxy(candidate)
    if box is not None:
        record["bbox"] = {
            "x": int(round(box[0])),
            "y": int(round(box[1])),
            "w": int(round(box[2] - box[0])),
            "h": int(round(box[3] - box[1])),
        }
    return record


def collect_held_overlay_candidates(
    candidates: Iterable[JsonDict],
    *,
    holding_object: bool,
    held_object_labels: Optional[Iterable[str]],
    held_object_family: Optional[str],
) -> Tuple[List[JsonDict], List[JsonDict]]:
    if not holding_object or not env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True):
        return [], []
    held_labels = normalize_label_tokens(held_object_labels)
    held_family = normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels)
    overlays: List[JsonDict] = []
    records: List[JsonDict] = []
    seen: set = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate_likely_held_object(candidate, held_labels, held_family):
            candidate["held_object_candidate"] = True
            candidate["ignored_as_held_object_blocker"] = True
            candidate["likely_held_object_overlay"] = True
            key = (
                normalize_label(candidate.get("label") or candidate.get("raw_label")),
                str(candidate.get("bbox")),
            )
            if key in seen:
                continue
            seen.add(key)
            overlays.append(candidate)
            records.append(held_overlay_record(candidate, held_family=held_family))
    return overlays, records

"""如果系统判断某个 apple / tomato 是手里拿着的，它会把这个 bbox 对应的 depth 点从点云里删掉：

原始点云
    ↓
去掉 held object bbox 内的点
    ↓
再拿剩下点云找台面平面"""
def remove_points_inside_held_overlays(
    points: JsonDict,
    overlays: Iterable[JsonDict],
    *,
    image_w: int,
    image_h: int,
) -> Tuple[JsonDict, List[JsonDict], int]:
    """Remove points classified as held-object foreground before plane extraction.

    This is the local equivalent of a robot self-filter: keep raw detections, but
    do not let carried-object depth points enter support-plane clustering.
    """
    import numpy as np  # type: ignore

    uv = np.asarray(points.get("uv", []), dtype="float64")
    if uv.ndim != 2 or uv.shape[0] == 0:
        return points, [], 0

    keep = np.ones((uv.shape[0],), dtype=bool)
    records: List[JsonDict] = []
    total_removed = 0
    margin = env_float("ROBOT_PC_HELD_OBJECT_MASK_MARGIN_PIXELS", 10.0)
    for candidate in overlays:
        box = bbox_xyxy(candidate)
        if box is None:
            continue
        expanded = expand_box_pixels(box, margin=margin, image_w=image_w, image_h=image_h)
        x1, y1, x2, y2 = expanded
        inside = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        removed = int(np.count_nonzero(inside & keep))
        if removed <= 0:
            continue
        keep &= ~inside
        total_removed += removed
        label = normalize_label(candidate.get("label") or candidate.get("raw_label")) or "held_object"
        record = {
            "label": label,
            "raw_label": candidate.get("raw_label") or candidate.get("label"),
            "reason": "held_object_depth_mask",
            "removed_points": removed,
            "bbox": {
                "x": int(round(box[0])),
                "y": int(round(box[1])),
                "w": int(round(box[2] - box[0])),
                "h": int(round(box[3] - box[1])),
            },
            "expanded_bbox": {
                "x": int(round(expanded[0])),
                "y": int(round(expanded[1])),
                "w": int(round(expanded[2] - expanded[0])),
                "h": int(round(expanded[3] - expanded[1])),
            },
        }
        records.append(record)
        candidate["held_object_point_mask"] = record

    if total_removed <= 0:
        return points, records, 0

    filtered: JsonDict = {}
    for key, value in points.items():
        if key in {"image_size", "stride"}:
            filtered[key] = value
            continue
        arr = np.asarray(value)
        if arr.shape[:1] == keep.shape[:1]:
            filtered[key] = arr[keep]
        else:
            filtered[key] = value
    return filtered, records, total_removed

# #取 blocker bbox 里的点云
#     ↓
# 看这些点是否落在 surface region 的 x-z 范围内
#     ↓
# 看这些点高度是否在支撑平面上方 0~0.15m
# #     ↓
# 满足一定点数和比例，才认为真的占用
#这个 blocker 的点云里，有一部分点落在候选放置区域(靠点云选出来的)的 XZ 范围内，并
#且高度刚好在台面上方 0~15cm，因此认为它真的占用了这个放置区域。
def blocker_3d_occupancy(
    *,
    blocker: JsonDict,
    scene_points: Optional[JsonDict],
    region_xz: Tuple[float, float, float, float],
    plane_height_m: float,
    bbox: Tuple[float, float, float, float],
) -> JsonDict:
    import numpy as np  # type: ignore

    xyz, _ = points_in_bbox(scene_points, bbox)
    if xyz.size == 0:
        return {"occupied": False, "sample_count": 0, "surface_point_count": 0, "reason": "no_depth_samples"}

    min_x, max_x, min_z, max_z = region_xz
    margin_xz = env_float("ROBOT_PC_OCCUPANCY_XZ_MARGIN_M", 0.035)
    min_above = env_float("ROBOT_PC_OCCUPANCY_MIN_ABOVE_PLANE_M", 0.0)
    max_above = env_float("ROBOT_PC_OCCUPANCY_MAX_ABOVE_PLANE_M", 0.15)
    above = xyz[:, 1] - float(plane_height_m)
    mask = (
        (xyz[:, 0] >= min_x - margin_xz)
        & (xyz[:, 0] <= max_x + margin_xz)
        & (xyz[:, 2] >= min_z - margin_xz)
        & (xyz[:, 2] <= max_z + margin_xz)
        & (above >= min_above)
        & (above <= max_above)
    )
    count = int(mask.sum())
    min_points = max(1, env_int("ROBOT_PC_OCCUPANCY_MIN_SURFACE_POINTS", 3))
    min_ratio = env_float("ROBOT_PC_OCCUPANCY_MIN_SURFACE_RATIO", 0.04)
    occupied = bool(count >= min_points and count / max(1, int(xyz.shape[0])) >= min_ratio)
    return {
        "occupied": occupied,
        "sample_count": int(xyz.shape[0]),
        "surface_point_count": count,
        "surface_point_ratio": round(float(count / max(1, int(xyz.shape[0]))), 4),
        "height_band_m": [round(float(min_above), 4), round(float(max_above), 4)],
        "reason": "3d_occupancy" if occupied else "no_same_plane_occupancy",
    }

#综合判断哪些东西真的挡住了 region
def blocker_records_for_region(
    region_box: Tuple[float, float, float, float],
    blockers: Iterable[JsonDict],
    *,
    margin: float,
    image_w: int,
    image_h: int,
    parent: JsonDict,
    scene_points: Optional[JsonDict],
    region_xz: Tuple[float, float, float, float],
    plane_height_m: float,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
    skipped_blockers: Optional[List[JsonDict]] = None,
) -> Tuple[List[JsonDict], List[JsonDict]]:
    rx1, ry1, rx2, ry2 = region_box
    expanded = (
        max(0.0, rx1 - margin),
        max(0.0, ry1 - margin),
        min(float(image_w), rx2 + margin),
        min(float(image_h), ry2 + margin),
    )
    region_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    blocked: List[JsonDict] = []
    skipped: List[JsonDict] = []
    parent_box = bbox_xyxy(parent)
    held_labels = normalize_label_tokens(held_object_labels)
    for blocker in blockers:
        if blocker is parent:
            continue
        label = normalize_label(blocker.get("label") or blocker.get("raw_label"))
        task_class = str(blocker.get("task_semantic_class") or "")
        if (
            holding_object
            and env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True)
            and candidate_likely_held_object(blocker, held_labels, held_object_family)
        ):
            token = label or str(blocker.get("raw_label") or "held_object")
            record = {
                "label": token,
                "raw_label": blocker.get("raw_label") or blocker.get("label"),
                "task_semantic_class": task_class,
                "reason": "likely_held_object_overlay",
                "held_object_family": normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels),
                "overlay_checks": blocker.get("held_object_overlay_checks"),
            }
            held_box = bbox_xyxy(blocker)
            if held_box is not None:
                record["bbox"] = {
                    "x": int(round(held_box[0])),
                    "y": int(round(held_box[1])),
                    "w": int(round(held_box[2] - held_box[0])),
                    "h": int(round(held_box[3] - held_box[1])),
                }
            skipped.append(record)
            if skipped_blockers is not None:
                skipped_blockers.append(record)
            blocker["held_object_candidate"] = True
            blocker["ignored_as_held_object_blocker"] = True
            blocker["likely_held_object_overlay"] = True
            continue
        category = blocker_category(blocker)
        if category == "none":
            continue
        if task_class not in {"pickup_target", "ignored_object", "obstacle", "place_receptacle"} and label not in POINTCLOUD_BLOCKING_LABELS:
            continue
        box = bbox_xyxy(blocker)
        if box is None:
            continue
        if parent_box is not None and box == parent_box:
            continue
        test_box = occupancy_test_box(box, category)
        overlap = bbox_overlap_area(expanded, test_box)
        overlap_ratio = overlap / region_area
        center = blocker.get("center") if isinstance(blocker.get("center"), dict) else {}
        center_inside = False
        try:
            cx = float(center.get("x"))
            cy = float(center.get("y"))
            center_inside = expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]
        except (TypeError, ValueError):
            center_inside = False

        occupancy = blocker_3d_occupancy(
            blocker=blocker,
            scene_points=scene_points,
            region_xz=region_xz,
            plane_height_m=plane_height_m,
            bbox=test_box,
        )
        block_reason = ""
        min_overlap_ratio = env_float("ROBOT_PC_OCCUPANCY_BLOCK_OVERLAP_RATIO", 0.15)
        if category == "strong":
            if bool(occupancy.get("occupied")):
                block_reason = "3d_occupancy"
            elif center_inside:
                block_reason = "center_inside"
            elif overlap_ratio > min_overlap_ratio:
                block_reason = "overlap"
        elif category == "structural_core":
            if bool(occupancy.get("occupied")):
                block_reason = "3d_occupancy"
            elif overlap_ratio > env_float("ROBOT_PC_CORE_BLOCKER_OVERLAP_RATIO", 0.12):
                block_reason = "core_overlap"
        elif category == "weak_context" and bool(occupancy.get("occupied")):
            weak_surface_ratio = float(occupancy.get("surface_point_ratio", 0.0) or 0.0)
            if center_inside and weak_surface_ratio >= env_float("ROBOT_PC_WEAK_CONTEXT_MIN_SURFACE_RATIO", 0.08):
                block_reason = "3d_occupancy"

        token = label or str(blocker.get("raw_label") or "object")
        if not block_reason:
            if overlap > 0 or center_inside:
                skipped_record = {
                    "label": token,
                    "raw_label": blocker.get("raw_label") or blocker.get("label"),
                    "task_semantic_class": task_class,
                    "category": category,
                    "reason": str(occupancy.get("reason") or "2d_overlap_not_sufficient"),
                    "overlap_ratio": round(float(overlap_ratio), 4),
                    "center_inside": bool(center_inside),
                    "occupancy": occupancy,
                    "bbox": {
                        "x": int(round(test_box[0])),
                        "y": int(round(test_box[1])),
                        "w": int(round(test_box[2] - test_box[0])),
                        "h": int(round(test_box[3] - test_box[1])),
                    },
                }
                skipped.append(skipped_record)
            continue
        blocked.append(
            {
                "label": token,
                "raw_label": blocker.get("raw_label") or blocker.get("label"),
                "task_semantic_class": task_class,
                "category": category,
                "reason": block_reason,
                "blocked_token": f"{token}:{block_reason}",
                "overlap_ratio": round(float(overlap_ratio), 4),
                "center_inside": bool(center_inside),
                "occupancy": occupancy,
                "bbox": {
                    "x": int(round(test_box[0])),
                    "y": int(round(test_box[1])),
                    "w": int(round(test_box[2] - test_box[0])),
                    "h": int(round(test_box[3] - test_box[1])),
                },
            }
        )
    return blocked, skipped

"""找水平支撑平面
这个函数是点云平面提取核心。

它做几件事：
1. 过滤高度 0.45~1.25m 的点
2. voxel downsample 降采样
3. remove_statistical_outlier 去除离群点
4. estimate_normals 估计法向量
5. segment_plane 用 RANSAC 找平面
6. 检查 normal_up_score
7. 检查 plane_height
8. 映射回原始点云 inliers

新版还加了多次 RANSAC attempts：
ROBOT_PC_RANSAC_ATTEMPTS 默认 6
ROBOT_PC_RANSAC_SEED 控制随机种子
每次尝试都打分，选最好的那个平面"""

def extract_support_planes_open3d(pcd: JsonDict, parent_candidate: JsonDict, blockers: Iterable[JsonDict]) -> Tuple[List[JsonDict], JsonDict]:
    """Extract horizontal support plane inliers from a cropped point set."""
    import numpy as np  # type: ignore
    import open3d as o3d  # type: ignore

    xyz = np.asarray(pcd.get("xyz", []), dtype="float64")
    uv = np.asarray(pcd.get("uv", []), dtype="float64")
    stats: JsonDict = {
        "parent_label": parent_candidate.get("label") or parent_candidate.get("raw_label"),
        "input_points": int(xyz.shape[0]) if xyz.ndim == 2 else 0,
        "candidate_planes": 0,
        "accepted_planes": 0,
        "rejected_planes": 0,
    }
    if xyz.ndim != 2 or xyz.shape[0] < env_int("ROBOT_PC_PLANE_MIN_LOCAL_POINTS", 80):
        stats["reason"] = "not_enough_local_points"
        return [], stats

    parent_object = normalize_label(parent_candidate.get("label") or parent_candidate.get("raw_label"))
    min_support_h, max_support_h = support_surface_height_range(parent_object)
    min_candidate_h = env_float(
        "ROBOT_PC_PLANE_CANDIDATE_MIN_HEIGHT_M",
        max(0.0, min_support_h - env_float("ROBOT_PC_PLANE_CANDIDATE_HEIGHT_MARGIN_M", 0.035)),
    )
    max_candidate_h = env_float(
        "ROBOT_PC_PLANE_CANDIDATE_MAX_HEIGHT_M",
        max_support_h + env_float("ROBOT_PC_PLANE_CANDIDATE_MAX_HEIGHT_MARGIN_M", 0.10),
    )
    height_mask = (xyz[:, 1] >= min_candidate_h) & (xyz[:, 1] <= max_candidate_h)
    xyz_h = xyz[height_mask]
    uv_h = uv[height_mask]
    if xyz_h.shape[0] < env_int("ROBOT_PC_PLANE_MIN_LOCAL_POINTS", 80):
        stats["height_filtered_points"] = int(xyz_h.shape[0])
        stats["reason"] = "not_enough_height_filtered_points"
        return [], stats
    stats["height_filtered_points"] = int(xyz_h.shape[0])

    source_points: JsonDict = {"xyz": xyz_h, "uv": uv_h, "image_size": pcd.get("image_size"), "stride": pcd.get("stride")}
    o3d_pcd = build_open3d_pointcloud(source_points)
    voxel_size = env_float("ROBOT_PC_VOXEL_SIZE_M", 0.012)
    if voxel_size > 0:
        o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(o3d_pcd.points) >= env_int("ROBOT_PC_OUTLIER_MIN_POINTS", 80):
        try:
            o3d_pcd, _ = o3d_pcd.remove_statistical_outlier(
                nb_neighbors=max(4, env_int("ROBOT_PC_OUTLIER_NB_NEIGHBORS", 16)),
                std_ratio=env_float("ROBOT_PC_OUTLIER_STD_RATIO", 2.0),
            )
        except Exception as exc:
            stats["outlier_filter_warning"] = str(exc)
    try:
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=env_float("ROBOT_PC_NORMAL_RADIUS_M", 0.05),
                max_nn=env_int("ROBOT_PC_NORMAL_MAX_NN", 24),
            )
        )
    except Exception as exc:
        stats["normal_warning"] = str(exc)

    remaining = o3d_pcd
    min_inliers = env_int("ROBOT_PC_PLANE_MIN_INLIERS", 45)
    max_planes = env_int("ROBOT_PC_MAX_PLANES_PER_PARENT", 4)
    threshold = env_float("ROBOT_PC_PLANE_DISTANCE_THRESHOLD_M", 0.025)
    map_threshold = env_float("ROBOT_PC_PLANE_ORIGINAL_MAP_THRESHOLD_M", threshold * 1.5)
    min_up_score = env_float("ROBOT_PC_MIN_NORMAL_UP_SCORE", 0.85)
    planes: List[JsonDict] = []
    stats["support_height_range_m"] = [round(float(min_support_h), 4), round(float(max_support_h), 4)]
    stats["candidate_height_range_m"] = [round(float(min_candidate_h), 4), round(float(max_candidate_h), 4)]

    for plane_id in range(max_planes):
        if len(remaining.points) < min_inliers:
            break
        attempts = max(1, env_int("ROBOT_PC_RANSAC_ATTEMPTS", 8))
        min_attempts_before_stop = min(
            attempts - 1,
            max(0, env_int("ROBOT_PC_RANSAC_MIN_ATTEMPTS", 4)),
        )
        early_stop_min_original = env_int("ROBOT_PC_RANSAC_EARLY_STOP_MIN_ORIGINAL_INLIERS", 1200)
        seed_base = env_int("ROBOT_PC_RANSAC_SEED", 17)
        best_attempt: Optional[JsonDict] = None
        short_attempts = 0
        for attempt_index in range(attempts):
            try:
                try:
                    o3d.utility.random.seed(int(seed_base + plane_id * 997 + attempt_index))
                except Exception:
                    pass
                model, inliers = remaining.segment_plane(
                    distance_threshold=threshold,
                    ransac_n=3,
                    num_iterations=env_int("ROBOT_PC_RANSAC_ITERATIONS", 220),
                )
            except Exception as exc:
                stats["segment_plane_error"] = str(exc)
                break
            if len(inliers) < min_inliers:
                short_attempts += 1
                continue
            a, b, c, d = [float(v) for v in model]
            normal_norm = max(1e-9, math.sqrt(a * a + b * b + c * c))
            normal = [a / normal_norm, b / normal_norm, c / normal_norm]
            normal_up_score = abs(normal[1])
            down_points = np.asarray(remaining.points)[inliers]
            plane_height = float(np.median(down_points[:, 1])) if down_points.size else 0.0
            distances = np.abs((xyz_h @ np.asarray(normal)) + (d / normal_norm))
            original_mask = distances <= map_threshold
            original_mask &= np.abs(xyz_h[:, 1] - plane_height) <= env_float("ROBOT_PC_PLANE_HEIGHT_MAP_TOLERANCE_M", 0.045)
            original_indices = np.where(original_mask)[0]
            valid = bool(
                normal_up_score >= min_up_score
                and min_support_h <= plane_height <= max_support_h
                and original_indices.shape[0] >= min_inliers
            )
            support_center = max(min_support_h, min(max_support_h, support_surface_ideal_height(parent_object)))
            height_penalty = abs(plane_height - support_center) * 1000.0
            score = (
                (100000.0 if valid else 0.0)
                + float(original_indices.shape[0]) * 4.0
                + float(len(inliers))
                + float(normal_up_score) * 1000.0
                - height_penalty
            )
            attempt_record: JsonDict = {
                "attempt_index": int(attempt_index),
                "model": model,
                "inliers": list(inliers),
                "normal": normal,
                "normal_up_score": float(normal_up_score),
                "plane_height": float(plane_height),
                "original_indices": original_indices,
                "downsample_inlier_count": int(len(inliers)),
                "valid": bool(valid),
                "score": float(score),
            }
            if best_attempt is None or float(attempt_record["score"]) > float(best_attempt["score"]):
                best_attempt = attempt_record
            if valid and original_indices.shape[0] >= early_stop_min_original and attempt_index >= min_attempts_before_stop:
                break
        if best_attempt is None:
            if short_attempts:
                stats.setdefault("short_ransac_attempts", 0)
                stats["short_ransac_attempts"] += int(short_attempts)
            break
        stats["candidate_planes"] += 1
        stats.setdefault("ransac_attempts", 0)
        stats["ransac_attempts"] += int(attempts - short_attempts)
        model = best_attempt["model"]
        inliers = best_attempt["inliers"]
        normal = best_attempt["normal"]
        normal_up_score = float(best_attempt["normal_up_score"])
        plane_height = float(best_attempt["plane_height"])
        original_indices = best_attempt["original_indices"]
        if (
            normal_up_score >= min_up_score
            and min_support_h <= plane_height <= max_support_h
            and original_indices.shape[0] >= min_inliers
        ):
            planes.append(
                {
                    "plane_id": int(plane_id),
                    "plane_model": [round(float(v), 6) for v in model],
                    "plane_normal": [round(float(v), 6) for v in normal],
                    "normal_up_score": round(float(normal_up_score), 4),
                    "height_m": round(float(plane_height), 4),
                    "xyz": xyz_h[original_indices],
                    "uv": uv_h[original_indices],
                    "inlier_count": int(original_indices.shape[0]),
                    "downsample_inlier_count": int(len(inliers)),
                    "parent_candidate": parent_candidate,
                }
            )
            stats["accepted_planes"] += 1
        else:
            reasons: List[str] = []
            if normal_up_score < min_up_score:
                reasons.append("normal_not_up")
            if plane_height < min_support_h:
                reasons.append("height_below_support_range")
            elif plane_height > max_support_h:
                reasons.append("height_above_support_range")
            if original_indices.shape[0] < min_inliers:
                reasons.append("too_few_original_inliers")
            stats.setdefault("rejected_plane_records", []).append(
                {
                    "plane_id": int(plane_id),
                    "reasons": reasons or ["unknown"],
                    "normal_up_score": round(float(normal_up_score), 4),
                    "height_m": round(float(plane_height), 4),
                    "original_inlier_count": int(original_indices.shape[0]),
                    "downsample_inlier_count": int(len(inliers)),
                }
            )
            stats["rejected_planes"] += 1
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes, stats

#把平面点分成多个 region
def cluster_plane_regions_open3d(plane_inliers: JsonDict) -> List[JsonDict]:
    import numpy as np  # type: ignore

    xyz = np.asarray(plane_inliers.get("xyz", []), dtype="float64")
    uv = np.asarray(plane_inliers.get("uv", []), dtype="float64")
    if xyz.ndim != 2 or xyz.shape[0] == 0:
        return []
    points = {"xyz": xyz, "uv": uv}
    pcd = build_open3d_pointcloud(points)
    labels = pcd.cluster_dbscan(
        eps=env_float("ROBOT_PC_DBSCAN_EPS_M", 0.055),
        min_points=env_int("ROBOT_PC_DBSCAN_MIN_POINTS", 12),
        print_progress=False,
    )
    labels_np = np.asarray(labels)
    clusters: List[JsonDict] = []
    for cluster_id in sorted(int(v) for v in set(labels_np.tolist()) if int(v) >= 0):
        mask = labels_np == cluster_id
        if int(mask.sum()) <= 0:
            continue
        cluster = dict(plane_inliers)
        cluster["xyz"] = xyz[mask]
        cluster["uv"] = uv[mask]
        cluster["dbscan_cluster_id"] = int(cluster_id)
        cluster["cluster_size"] = int(mask.sum())
        clusters.append(cluster)
    return clusters


def pointcloud_surface_id(parent_label: str, bbox: JsonDict, height_m: float, distance_m: float, plane_id: int, cluster_id: int) -> str:
    token = normalize_label(parent_label) or "surface"
    x = int(round(float(bbox.get("x", 0) or 0)))
    y = int(round(float(bbox.get("y", 0) or 0)))
    w = int(round(float(bbox.get("w", 0) or 0)))
    h = int(round(float(bbox.get("h", 0) or 0)))
    return f"pc_surface:{token}:{plane_id}:{cluster_id}:{x}:{y}:{w}:{h}:{int(round(height_m * 100))}:{int(round(distance_m * 100))}"


def score_pointcloud_surface_region(region: JsonDict) -> float:
    score = 0.0
    score += min(1.0, float(region.get("region_area_m2", 0.0) or 0.0) / max(0.001, env_float("ROBOT_PC_REGION_MIN_AREA_M2", 0.018) * 3.0)) * 0.22
    score += min(1.0, float(region.get("point_count", 0) or 0) / max(1.0, env_int("ROBOT_PC_REGION_MIN_POINTS", 60) * 3.0)) * 0.16
    score += float(region.get("normal_up_score", 0.0) or 0.0) * 0.18
    height_m = float(region.get("height_m", 0.0) or 0.0)
    distance_m = float(region.get("distance_m", 0.0) or 0.0)
    parent_object = normalize_label(region.get("parent_object") or region.get("parent_label") or "")
    ideal_height = support_surface_ideal_height(parent_object)
    height_spread = env_float("ROBOT_PC_SURFACE_HEIGHT_SCORE_SPREAD_M", 0.35)
    score += max(0.0, 1.0 - min(1.0, abs(height_m - ideal_height) / max(0.05, height_spread))) * 0.16
    score += max(0.0, 1.0 - min(1.0, abs(distance_m - 0.95) / 0.65)) * 0.14
    score += min(1.0, float(region.get("density", 0.0) or 0.0) / max(1.0, env_float("ROBOT_PC_REGION_IDEAL_DENSITY", 2200.0))) * 0.08
    if parent_object in {"counter_top", "countertop", "dining_table", "diningtable", "coffee_table", "coffeetable", "side_table", "sidetable", "table"}:
        score += 0.06
    if region.get("rejection_reasons"):
        score *= 0.35
    if region.get("blocked"):
        score *= 0.25
    return round(float(max(0.0, min(1.0, score))), 4)


def select_best_pointcloud_surface(regions: Iterable[JsonDict]) -> Optional[JsonDict]:
    ready = [region for region in regions if isinstance(region, dict) and region.get("visual_place_ready")]
    if not ready:
        return None
    ready.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return ready[0]


def add_region_rejection(region: JsonDict, reason: str) -> None:
    for key in ("rejection_reasons", "affordance_rejection_reasons"):
        values = region.get(key)
        if not isinstance(values, list):
            values = []
            region[key] = values
        if reason not in values:
            values.append(reason)


def mark_region_not_placeable(region: JsonDict, reason: str, detail: Optional[JsonDict] = None) -> None:
    add_region_rejection(region, reason)
    checks = region.get("geometry_checks") if isinstance(region.get("geometry_checks"), dict) else {}
    checks[reason] = True
    region["geometry_checks"] = checks
    region["visual_place_ready"] = False
    region["affordance_ready"] = False
    region["reachable"] = False
    region["region_type"] = "rejected_surface_region"
    region["affordance"] = []
    region["place_affordance"] = {
        "action": "place",
        "affordance_ready": False,
    }
    if detail:
        region[reason] = detail
    region["score"] = score_pointcloud_surface_region(region)
    region["affordance_score"] = region["score"]


def mark_raised_object_top_regions(parent_regions: List[JsonDict]) -> int:
    if not env_bool("ROBOT_PC_RAISED_OBJECT_TOP_FILTER_ENABLED", True):
        return 0
    plane_regions = [
        region
        for region in parent_regions
        if isinstance(region, dict)
        and str(region.get("source") or "") == "pointcloud_plane"
    ]
    candidates = [region for region in plane_regions if not bool(region.get("blocked"))]
    reference_regions = [
        region
        for region in plane_regions
        if all(
            bool((region.get("geometry_checks") or {}).get(key))
            for key in ("normal_ok", "height_ok", "distance_ok", "point_count_ok")
        )
    ]
    if not candidates or len(reference_regions) < 2:
        return 0

    min_delta = env_float("ROBOT_PC_RAISED_OBJECT_TOP_HEIGHT_DELTA_M", 0.055)
    max_area = env_float("ROBOT_PC_RAISED_OBJECT_TOP_MAX_AREA_M2", 0.060)
    max_width = env_float("ROBOT_PC_RAISED_OBJECT_TOP_MAX_WIDTH_M", 0.38)
    max_depth = env_float("ROBOT_PC_RAISED_OBJECT_TOP_MAX_DEPTH_M", 0.26)
    reference_scale = env_float("ROBOT_PC_RAISED_OBJECT_TOP_REFERENCE_SCALE", 1.15)
    filtered = 0

    for region in candidates:
        try:
            height = float(region.get("height_m", 0.0) or 0.0)
            area = float(region.get("region_area_m2", 0.0) or 0.0)
            width = float(region.get("region_width_m", 0.0) or 0.0)
            depth = float(region.get("region_depth_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if area > max_area and width > max_width and depth > max_depth:
            continue

        lower_refs: List[JsonDict] = []
        for other in reference_regions:
            if other is region:
                continue
            try:
                other_height = float(other.get("height_m", 0.0) or 0.0)
                other_area = float(other.get("region_area_m2", 0.0) or 0.0)
                other_width = float(other.get("region_width_m", 0.0) or 0.0)
                other_depth = float(other.get("region_depth_m", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if other_height > height - min_delta:
                continue
            broader = bool(
                other_area >= area * reference_scale
                or other_width >= width * reference_scale
                or other_depth >= depth * reference_scale
            )
            if broader:
                lower_refs.append(other)
        if not lower_refs:
            continue
        reference = max(lower_refs, key=lambda item: float(item.get("region_area_m2", 0.0) or 0.0))
        detail = {
            "reason": "higher_than_broader_support_plane",
            "height_delta_m": round(float(height - float(reference.get("height_m", 0.0) or 0.0)), 4),
            "reference_surface_id": reference.get("id"),
            "reference_height_m": reference.get("height_m"),
            "reference_area_m2": reference.get("region_area_m2"),
        }
        mark_region_not_placeable(region, "raised_object_top", detail)
        filtered += 1
    return filtered


def _normalize_vector(values: Any) -> Any:
    import numpy as np  # type: ignore

    vector = np.asarray(values, dtype="float64")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return vector / norm


def _plane_local_basis(surface_xyz: Any, plane_normal: Any) -> Optional[JsonDict]:
    """Build a deterministic local (u, v) frame on a support plane."""
    import numpy as np  # type: ignore

    xyz = np.asarray(surface_xyz, dtype="float64")
    if xyz.ndim != 2 or xyz.shape[0] < 3 or xyz.shape[1] < 3:
        return None
    normal = _normalize_vector(plane_normal)
    if normal is None:
        normal = np.asarray([0.0, 1.0, 0.0], dtype="float64")
    if normal[1] < 0:
        normal = -normal
    origin = np.median(xyz[:, :3], axis=0)
    centered = xyz[:, :3] - origin
    flattened = centered - np.outer(centered @ normal, normal)
    u_axis = None
    try:
        covariance = np.cov(flattened, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        u_axis = _normalize_vector(vectors[:, int(np.argmax(values))])
        if u_axis is not None:
            u_axis = _normalize_vector(u_axis - normal * float(np.dot(u_axis, normal)))
    except Exception:
        u_axis = None
    if u_axis is None:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype="float64")
        u_axis = _normalize_vector(fallback - normal * float(np.dot(fallback, normal)))
    if u_axis is None:
        fallback = np.asarray([0.0, 0.0, 1.0], dtype="float64")
        u_axis = _normalize_vector(fallback - normal * float(np.dot(fallback, normal)))
    if u_axis is None:
        return None
    if float(np.dot(u_axis, np.asarray([1.0, 0.0, 0.0], dtype="float64"))) < 0:
        u_axis = -u_axis
    v_axis = _normalize_vector(np.cross(normal, u_axis))
    if v_axis is None:
        return None
    if float(np.dot(v_axis, np.asarray([0.0, 0.0, 1.0], dtype="float64"))) < 0:
        v_axis = -v_axis
    return {"origin": origin, "normal": normal, "u_axis": u_axis, "v_axis": v_axis}


def _project_plane_local(xyz: Any, frame: JsonDict) -> Any:
    import numpy as np  # type: ignore

    values = np.asarray(xyz, dtype="float64")
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((0, 2), dtype="float64")
    offset = values[:, :3] - frame["origin"]
    return np.stack([offset @ frame["u_axis"], offset @ frame["v_axis"]], axis=1)


def _grid_indices(local_uv: Any, *, u_min: float, v_min: float, resolution: float, shape: Tuple[int, int]) -> Tuple[Any, Any]:
    import numpy as np  # type: ignore

    values = np.asarray(local_uv, dtype="float64")
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((0,), dtype="int32"), np.zeros((0,), dtype="int32")
    cols = np.floor((values[:, 0] - u_min) / resolution).astype("int32")
    rows = np.floor((values[:, 1] - v_min) / resolution).astype("int32")
    rows = np.clip(rows, 0, shape[0] - 1)
    cols = np.clip(cols, 0, shape[1] - 1)
    return rows, cols


def _binary_dilate_or_erode(mask: Any, radius: int, operation: str) -> Any:
    import numpy as np  # type: ignore

    binary = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return binary.copy()
    try:
        import cv2  # type: ignore

        kernel_size = int(radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        source = binary.astype("uint8")
        if operation == "dilate":
            result = cv2.dilate(source, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            # Pixels outside the observed support grid are unsafe. Explicitly
            # treating them as empty prevents a placement cell on the grid edge
            # from surviving the physical edge-margin erosion.
            result = cv2.erode(source, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return result > 0
    except Exception:
        padded = np.pad(binary, radius, constant_values=False)
        windows = [
            padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]]
            for dy in range(radius * 2 + 1)
            for dx in range(radius * 2 + 1)
            if (dy - radius) * (dy - radius) + (dx - radius) * (dx - radius) <= radius * radius
        ]
        result = windows[0].copy()
        for window in windows[1:]:
            result = (result | window) if operation == "dilate" else (result & window)
        return result


def _binary_close(mask: Any, radius: int) -> Any:
    return _binary_dilate_or_erode(_binary_dilate_or_erode(mask, radius, "dilate"), radius, "erode")


def _connected_components(mask: Any) -> Tuple[int, Any, Any]:
    import numpy as np  # type: ignore

    binary = np.asarray(mask, dtype=bool)
    try:
        import cv2  # type: ignore

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype("uint8"), connectivity=8)
        return int(count), labels.astype("int32"), stats
    except Exception:
        labels = np.zeros(binary.shape, dtype="int32")
        records: List[List[int]] = [[0, 0, 0, 0, 0]]
        component = 0
        for start_row, start_col in zip(*np.where(binary & (labels == 0))):
            if labels[start_row, start_col] != 0:
                continue
            component += 1
            stack = [(int(start_row), int(start_col))]
            labels[start_row, start_col] = component
            cells: List[Tuple[int, int]] = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nr, nc = row + dy, col + dx
                        if 0 <= nr < binary.shape[0] and 0 <= nc < binary.shape[1] and binary[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = component
                            stack.append((nr, nc))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            records.append([min(cols), min(rows), max(cols) - min(cols) + 1, max(rows) - min(rows) + 1, len(cells)])
        return component + 1, labels, np.asarray(records, dtype="int32")


def _safe_component_cell(component_mask: Any) -> Tuple[int, int]:
    import numpy as np  # type: ignore

    binary = np.asarray(component_mask, dtype=bool)
    try:
        import cv2  # type: ignore

        distances = cv2.distanceTransform(binary.astype("uint8"), cv2.DIST_L2, 5)
        row, col = np.unravel_index(int(np.argmax(distances)), distances.shape)
        return int(row), int(col)
    except Exception:
        cells = np.argwhere(binary)
        if cells.shape[0] == 0:
            return 0, 0
        center = np.median(cells, axis=0)
        nearest = cells[int(np.argmin(np.sum((cells - center) ** 2, axis=1)))]
        return int(nearest[0]), int(nearest[1])


def _component_safe_cells(
    component_mask: Any,
    *,
    resolution: float,
    max_points: int,
    min_separation_m: float,
) -> List[Tuple[int, int, float]]:
    """Choose separated high-clearance cells from one connected free region."""
    import numpy as np  # type: ignore

    binary = np.asarray(component_mask, dtype=bool)
    if not bool(np.any(binary)):
        return []
    try:
        import cv2  # type: ignore

        distances = cv2.distanceTransform(binary.astype("uint8"), cv2.DIST_L2, 5)
    except Exception:
        distances = np.zeros(binary.shape, dtype="float32")
        layer = binary.copy()
        distance = 0
        while bool(np.any(layer)):
            distance += 1
            distances[layer] = float(distance)
            layer = _binary_dilate_or_erode(layer, 1, "erode")
    working = distances.copy()
    separation_cells = max(1, int(math.ceil(max(0.0, min_separation_m) / max(0.005, resolution))))
    ranked: List[Tuple[int, int, float]] = []
    for _ in range(max(1, int(max_points))):
        flat_index = int(np.argmax(working))
        clearance_cells = float(working.flat[flat_index])
        if clearance_cells <= 0.0:
            break
        row, col = np.unravel_index(flat_index, working.shape)
        ranked.append((int(row), int(col), round(clearance_cells * resolution, 4)))
        rr, cc = np.ogrid[:working.shape[0], :working.shape[1]]
        suppress = (rr - int(row)) ** 2 + (cc - int(col)) ** 2 <= separation_cells ** 2
        working[suppress] = 0.0
    if ranked:
        return ranked
    row, col = _safe_component_cell(binary)
    return [(int(row), int(col), 0.0)]


def _plane_local_ground_distance_grid(
    *,
    frame: JsonDict,
    u_min: float,
    v_min: float,
    resolution: float,
    shape: Tuple[int, int],
) -> Any:
    """Return robot-relative horizontal distance for every plane-local grid cell."""
    import numpy as np  # type: ignore

    rows, cols = shape
    u_values = u_min + (np.arange(cols, dtype="float64") + 0.5) * resolution
    v_values = v_min + (np.arange(rows, dtype="float64") + 0.5) * resolution
    u_grid, v_grid = np.meshgrid(u_values, v_values)
    xyz = (
        np.asarray(frame["origin"], dtype="float64")[None, None, :]
        + u_grid[:, :, None] * np.asarray(frame["u_axis"], dtype="float64")[None, None, :]
        + v_grid[:, :, None] * np.asarray(frame["v_axis"], dtype="float64")[None, None, :]
    )
    return np.hypot(xyz[:, :, 0], xyz[:, :, 2])


def _strip_internal_surface_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_internal_surface_fields(item)
            for key, item in value.items()
            if key not in {"_surface_xyz", "_surface_uv", "_grid_debug_mask"} and not str(key).startswith("_grid_")
        }
    if isinstance(value, list):
        return [_strip_internal_surface_fields(item) for item in value]
    return value


def is_grid_completion_candidate(region: JsonDict) -> bool:
    """Return whether a trustworthy plane should be searched cell-by-cell.

    Region-level image edges and center distance are not terminal failures for
    a large support plane: erosion and per-cell reachability can still expose
    a valid interaction point. Intrinsic plane failures remain terminal.
    """
    if str(region.get("source") or "") != "pointcloud_plane":
        return False
    parent_object = normalize_label(region.get("parent_object") or region.get("parent_label") or "")
    if parent_object not in POINTCLOUD_TABLELIKE_LABELS or parent_object in POINTCLOUD_LOW_PRIORITY_PARENTS:
        return False
    checks = region.get("geometry_checks") if isinstance(region.get("geometry_checks"), dict) else {}
    required_checks = ("area_ok", "width_ok", "depth_ok", "normal_ok", "height_ok", "point_count_ok")
    if not all(bool(checks.get(key)) for key in required_checks):
        return False
    if bool(checks.get("thin_region")) or bool(checks.get("edge_only_region")):
        return False
    reasons = {
        str(reason)
        for reason in (region.get("rejection_reasons") or [])
        if str(reason)
    }
    non_blocking_reasons = {
        reason for reason in reasons if not reason.startswith("blocked:")
    }
    if non_blocking_reasons & POINTCLOUD_GRID_INTRINSIC_REJECTION_REASONS:
        return False
    if not non_blocking_reasons.issubset(POINTCLOUD_GRID_RECOVERABLE_REJECTION_REASONS):
        return False
    if bool(region.get("blocked")):
        return True
    return bool(non_blocking_reasons)


def build_free_space_completions_from_blocked_region_grid(
    region: JsonDict,
    *,
    scene_points: Optional[JsonDict],
    blockers: Iterable[JsonDict],
    image_w: int,
    image_h: int,
    camera: Optional[JsonDict] = None,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
    debug_stats: Optional[JsonDict] = None,
) -> List[JsonDict]:
    """Generate free placement candidates on a support plane using a local 2D grid.

    The metric rasterization follows the heightmap mapping idea in CLIPort
    (`cliport/utils/utils.py`, Apache-2.0): project 3D surface samples into
    fixed-resolution 2D cells and map selected cells back to 3D. This version
    uses the detected support plane's PCA frame instead of a global workspace
    heightmap and does not depend on CLIPort's training stack. Despite the
    legacy function name, inputs may be blocked planes or otherwise reliable
    planes whose edge/center-distance rejection is recoverable in the grid.
    """
    import numpy as np  # type: ignore

    summary = debug_stats if isinstance(debug_stats, dict) else {}
    resolution = max(0.005, env_float("ROBOT_PC_FREE_GRID_RES_M", 0.02))
    configured_edge_margin_m = max(0.0, env_float("ROBOT_PC_FREE_GRID_EDGE_MARGIN_M", 0.05))
    configured_blocker_dilate_m = max(0.0, env_float("ROBOT_PC_FREE_GRID_BLOCKER_DILATE_M", 0.04))
    footprint_radius_m = held_object_footprint_radius_m(
        holding_object=bool(holding_object),
        held_object_labels=held_object_labels,
        held_object_family=held_object_family,
    )
    placement_clearance_m = max(0.0, env_float("ROBOT_PC_FREE_GRID_PLACEMENT_CLEARANCE_M", 0.01))
    footprint_margin_m = footprint_radius_m + placement_clearance_m if holding_object else 0.0
    edge_margin_m = max(configured_edge_margin_m, footprint_margin_m)
    blocker_dilate_m = max(configured_blocker_dilate_m, footprint_margin_m)
    min_distance, max_distance = pointcloud_ready_distance_range_m()
    camera_data = camera if isinstance(camera, dict) else {}
    target_coordinate_frame = str(
        camera_data.get("coordinate_frame")
        or "camera_relative_x_right_y_height_z_forward"
    )
    camera_height_m = depth_number(camera_data.get("camera_height_m"))
    summary.update(
        {
            "mode": "plane_local_2d_grid",
            "grid_resolution_m": round(float(resolution), 4),
            "edge_margin_m": round(float(edge_margin_m), 4),
            "blocker_dilate_m": round(float(blocker_dilate_m), 4),
            "configured_edge_margin_m": round(float(configured_edge_margin_m), 4),
            "configured_blocker_dilate_m": round(float(configured_blocker_dilate_m), 4),
            "held_footprint_radius_m": round(float(footprint_radius_m), 4),
            "placement_clearance_m": round(float(placement_clearance_m), 4),
            "ready_distance_range_m": [round(float(min_distance), 4), round(float(max_distance), 4)],
            "component_count": 0,
            "raw_component_count": 0,
            "component_kept": 0,
            "raw_free_cell_count": 0,
            "reachable_free_cell_count": 0,
            "reachability_filtered_cell_count": 0,
            "near_free_cell_count": 0,
            "far_free_cell_count": 0,
            "has_near_free_space": False,
            "has_far_free_space": False,
            "approachable_component_count": 0,
            "has_approachable_free_space": False,
            "raw_component_rejection_counts": {},
            "raw_component_rejections": [],
            "far_placeable_component_count": 0,
            "has_far_placeable_free_space": False,
            "far_component_rejection_counts": {},
            "far_component_rejections": [],
            "component_rejection_counts": {},
            "component_rejections": [],
            "occupancy_mask_blocker_count": 0,
            "occupancy_source_counts": {"depth_points": 0, "bbox_fallback": 0},
            "hard_image_exclusion_box_count": 0,
            "mapped_point_hard_rejection_count": 0,
            "attempted": False,
            "candidate_triggers": [],
            "result": "not_attempted",
            "suggested_recovery": None,
        }
    )
    if not env_bool("ROBOT_PC_FREE_GRID_COMPLETION_ENABLED", True):
        summary["reason"] = "disabled"
        summary["result"] = "disabled"
        return []
    if not is_grid_completion_candidate(region):
        summary["reason"] = "not_grid_completion_candidate"
        summary["result"] = "not_grid_completion_candidate"
        return []
    summary["attempted"] = True
    input_reasons = {
        str(reason)
        for reason in (region.get("rejection_reasons") or [])
        if str(reason) and not str(reason).startswith("blocked:")
    }
    summary["candidate_triggers"] = (
        (["blocked"] if bool(region.get("blocked")) else [])
        + sorted(input_reasons & POINTCLOUD_GRID_RECOVERABLE_REJECTION_REASONS)
    )
    parent_object = normalize_label(region.get("parent_object") or region.get("parent_label") or "")
    surface_xyz = np.asarray(region.get("_surface_xyz", []), dtype="float64")
    surface_uv = np.asarray(region.get("_surface_uv", []), dtype="float64")
    if surface_xyz.ndim != 2 or surface_uv.ndim != 2 or surface_xyz.shape[0] < 3 or surface_xyz.shape[0] != surface_uv.shape[0]:
        summary["reason"] = "surface_samples_unavailable"
        summary["result"] = "surface_samples_unavailable"
        summary["suggested_recovery"] = "change_viewpoint"
        return []

    frame = _plane_local_basis(surface_xyz, region.get("plane_normal") or [0.0, 1.0, 0.0])
    if not frame:
        summary["reason"] = "plane_frame_failed"
        summary["result"] = "plane_frame_failed"
        summary["suggested_recovery"] = "change_viewpoint"
        return []
    surface_local = _project_plane_local(surface_xyz, frame)
    u_min = math.floor(float(np.min(surface_local[:, 0])) / resolution) * resolution
    v_min = math.floor(float(np.min(surface_local[:, 1])) / resolution) * resolution
    u_max = math.ceil(float(np.max(surface_local[:, 0])) / resolution) * resolution
    v_max = math.ceil(float(np.max(surface_local[:, 1])) / resolution) * resolution
    cols = max(1, int(round((u_max - u_min) / resolution)) + 1)
    rows = max(1, int(round((v_max - v_min) / resolution)) + 1)
    if rows * cols > env_int("ROBOT_PC_FREE_GRID_MAX_CELLS", 300000):
        summary["reason"] = "grid_too_large"
        summary["result"] = "grid_too_large"
        return []
    shape = (rows, cols)
    surface_rows, surface_cols = _grid_indices(surface_local, u_min=u_min, v_min=v_min, resolution=resolution, shape=shape)
    surface_mask = np.zeros(shape, dtype=bool)
    surface_mask[surface_rows, surface_cols] = True
    close_radius = max(0, int(round(env_float("ROBOT_PC_FREE_GRID_SURFACE_CLOSE_M", resolution * 1.5) / resolution)))
    dilate_radius = max(0, int(round(env_float("ROBOT_PC_FREE_GRID_SURFACE_DILATE_M", resolution) / resolution)))
    surface_mask = _binary_close(surface_mask, close_radius)
    surface_mask = _binary_dilate_or_erode(surface_mask, dilate_radius, "dilate")

    occupancy_mask = np.zeros(shape, dtype=bool)
    occupancy_records: List[JsonDict] = []
    occupancy_debug_uv: List[Any] = []
    hard_image_exclusion_boxes: List[Tuple[float, float, float, float]] = []
    region_box = bbox_xyxy(region.get("bbox"))
    parent_box = bbox_xyxy(region.get("parent_bbox"))
    plane_height = float(region.get("height_m", 0.0) or 0.0)
    min_above = env_float("ROBOT_PC_OCCUPANCY_MIN_ABOVE_PLANE_M", 0.0)
    max_above = env_float("ROBOT_PC_OCCUPANCY_MAX_ABOVE_PLANE_M", 0.15)
    held_labels = normalize_label_tokens(held_object_labels)
    recorded_labels = {normalize_label(item.get("label")) for item in (region.get("blocker_records") or []) if isinstance(item, dict)}
    for blocker in [item for item in blockers if isinstance(item, dict)]:
        box = bbox_xyxy(blocker)
        if box is None or (parent_box is not None and box == parent_box):
            continue
        if region_box is not None and bbox_overlap_area(region_box, box) <= 0:
            continue
        if (
            holding_object
            and env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True)
            and candidate_likely_held_object(blocker, held_labels, held_object_family)
        ):
            continue
        category = blocker_category(blocker)
        label = normalize_label(blocker.get("label") or blocker.get("raw_label"))
        if category == "none" or (category == "weak_context" and label not in recorded_labels):
            continue
        if category in {"strong", "structural_core"}:
            hard_image_exclusion_boxes.append(box)
        test_box = occupancy_test_box(box, category)
        blocker_xyz, blocker_uv = points_in_bbox(scene_points, test_box)
        occupancy_source = ""
        projected_local = np.zeros((0, 2), dtype="float64")
        projected_image_uv = np.zeros((0, 2), dtype="float64")
        if blocker_xyz.ndim == 2 and blocker_xyz.shape[0] > 0:
            above = blocker_xyz[:, 1] - plane_height
            valid_above = (above >= min_above) & (above <= max_above)
            if int(np.count_nonzero(valid_above)) >= env_int("ROBOT_PC_FREE_GRID_MIN_BLOCKER_POINTS", 2):
                projected_local = _project_plane_local(blocker_xyz[valid_above], frame)
                projected_image_uv = blocker_uv[valid_above]
                occupancy_source = "depth_points"
        if projected_local.shape[0] == 0:
            inside_surface = (
                (surface_uv[:, 0] >= test_box[0])
                & (surface_uv[:, 0] <= test_box[2])
                & (surface_uv[:, 1] >= test_box[1])
                & (surface_uv[:, 1] <= test_box[3])
            )
            fallback_count = int(np.count_nonzero(inside_surface))
            overlap_ratio = (
                bbox_overlap_area(region_box, test_box)
                / max(1.0, (region_box[2] - region_box[0]) * (region_box[3] - region_box[1]))
                if region_box is not None
                else 0.0
            )
            if (
                fallback_count >= env_int("ROBOT_PC_FREE_GRID_BBOX_FALLBACK_MIN_POINTS", 3)
                and overlap_ratio >= env_float("ROBOT_PC_FREE_GRID_BBOX_FALLBACK_MIN_OVERLAP_RATIO", 0.01)
            ):
                projected_local = surface_local[inside_surface]
                projected_image_uv = surface_uv[inside_surface]
                occupancy_source = "bbox_fallback"
        if projected_local.shape[0] == 0:
            continue
        block_rows, block_cols = _grid_indices(projected_local, u_min=u_min, v_min=v_min, resolution=resolution, shape=shape)
        occupancy_mask[block_rows, block_cols] = True
        occupancy_debug_uv.append(projected_image_uv)
        summary["occupancy_source_counts"][occupancy_source] += 1
        occupancy_records.append(
            {
                "label": label or str(blocker.get("raw_label") or "object"),
                "raw_label": blocker.get("raw_label") or blocker.get("label"),
                "source": occupancy_source,
                "projected_point_count": int(projected_local.shape[0]),
                "bbox": {
                    "x": int(round(test_box[0])),
                    "y": int(round(test_box[1])),
                    "w": int(round(test_box[2] - test_box[0])),
                    "h": int(round(test_box[3] - test_box[1])),
                },
            }
        )
    blocker_radius = max(0, int(math.ceil(blocker_dilate_m / resolution)))
    occupancy_mask = _binary_dilate_or_erode(occupancy_mask, blocker_radius, "dilate")
    edge_radius = max(0, int(math.ceil(edge_margin_m / resolution)))
    safe_surface_mask = _binary_dilate_or_erode(surface_mask, edge_radius, "erode")
    raw_free_mask = safe_surface_mask & ~occupancy_mask
    ground_distance_grid = _plane_local_ground_distance_grid(
        frame=frame,
        u_min=u_min,
        v_min=v_min,
        resolution=resolution,
        shape=shape,
    )
    reachable_mask = (ground_distance_grid >= min_distance) & (ground_distance_grid <= max_distance)
    near_free_mask = raw_free_mask & (ground_distance_grid < min_distance)
    far_free_mask = raw_free_mask & (ground_distance_grid > max_distance)
    free_mask = raw_free_mask & reachable_mask
    raw_count, raw_labels, _ = _connected_components(raw_free_mask)
    count, labels, component_stats = _connected_components(free_mask)
    summary["occupancy_mask_blocker_count"] = len(occupancy_records)
    summary["hard_image_exclusion_box_count"] = len(hard_image_exclusion_boxes)
    summary["raw_component_count"] = max(0, raw_count - 1)
    summary["component_count"] = max(0, count - 1)
    summary["raw_free_cell_count"] = int(np.count_nonzero(raw_free_mask))
    summary["reachable_free_cell_count"] = int(np.count_nonzero(free_mask))
    summary["reachability_filtered_cell_count"] = int(np.count_nonzero(raw_free_mask & ~reachable_mask))
    summary["near_free_cell_count"] = int(np.count_nonzero(near_free_mask))
    summary["far_free_cell_count"] = int(np.count_nonzero(far_free_mask))
    summary["has_near_free_space"] = bool(summary["near_free_cell_count"])
    summary["has_far_free_space"] = bool(summary["far_free_cell_count"])
    if summary["raw_free_cell_count"]:
        raw_distances = ground_distance_grid[raw_free_mask]
        summary["raw_free_distance_range_m"] = [
            round(float(np.min(raw_distances)), 4),
            round(float(np.max(raw_distances)), 4),
        ]
    if summary["reachable_free_cell_count"]:
        reachable_distances = ground_distance_grid[free_mask]
        summary["reachable_free_distance_range_m"] = [
            round(float(np.min(reachable_distances)), 4),
            round(float(np.max(reachable_distances)), 4),
        ]
    elif summary["raw_free_cell_count"]:
        summary["reason"] = "free_space_outside_current_reach"
    free_surface_points = free_mask[surface_rows, surface_cols]
    out_of_reach_surface_points = (raw_free_mask & ~reachable_mask)[surface_rows, surface_cols]
    occupied_surface_points = occupancy_mask[surface_rows, surface_cols]
    unsafe_surface_points = (~safe_surface_mask[surface_rows, surface_cols]) & ~occupied_surface_points
    projected_occupancy_uv = [surface_uv[occupied_surface_points], *occupancy_debug_uv]
    summary["_grid_debug_source_surface_id"] = region.get("id")
    summary["_grid_debug_plane_id"] = region.get("plane_id")
    summary["_grid_debug_cluster_id"] = region.get("dbscan_cluster_id")
    summary["_grid_debug_free_uv"] = surface_uv[free_surface_points]
    summary["_grid_debug_out_of_reach_uv"] = surface_uv[out_of_reach_surface_points]
    summary["_grid_debug_occupied_uv"] = (
        np.concatenate(projected_occupancy_uv, axis=0)
        if any(item.ndim == 2 and item.shape[0] > 0 for item in projected_occupancy_uv)
        else np.zeros((0, 2), dtype="float64")
    )
    summary["_grid_debug_unsafe_uv"] = surface_uv[unsafe_surface_points]

    min_area = env_float("ROBOT_PC_FREE_GRID_MIN_COMPONENT_AREA_M2", 0.018)
    min_width = env_float("ROBOT_PC_FREE_GRID_MIN_COMPONENT_WIDTH_M", env_float("ROBOT_PC_REGION_MIN_WIDTH_M", 0.12))
    min_depth = env_float("ROBOT_PC_FREE_GRID_MIN_COMPONENT_DEPTH_M", env_float("ROBOT_PC_REGION_MIN_DEPTH_M", 0.10))
    max_points_per_component = max(1, env_int("ROBOT_PC_FREE_GRID_MAX_POINTS_PER_COMPONENT", 4))
    point_separation_m = max(0.0, env_float("ROBOT_PC_FREE_GRID_POINT_SEPARATION_M", 0.10))

    def point_inside_visible_blocker(pixel_x: float, pixel_y: float) -> bool:
        return any(
            box[0] <= pixel_x <= box[2] and box[1] <= pixel_y <= box[3]
            for box in hard_image_exclusion_boxes
        )

    for raw_component_id in range(1, raw_count):
        raw_component_mask = raw_labels == raw_component_id
        raw_component_cells = np.argwhere(raw_component_mask)
        if raw_component_cells.shape[0] <= 0:
            continue
        raw_area = float(raw_component_cells.shape[0] * resolution * resolution)
        raw_depth = float((int(raw_component_cells[:, 0].max()) - int(raw_component_cells[:, 0].min()) + 1) * resolution)
        raw_width = float((int(raw_component_cells[:, 1].max()) - int(raw_component_cells[:, 1].min()) + 1) * resolution)
        raw_rejection_reasons: List[str] = []
        if raw_area < min_area:
            raw_rejection_reasons.append("too_small")
        if raw_width < min_width:
            raw_rejection_reasons.append("too_narrow")
        if raw_depth < min_depth:
            raw_rejection_reasons.append("too_shallow")
        contains_far_cells = bool(np.any(raw_component_mask & far_free_mask))
        if not raw_rejection_reasons and contains_far_cells:
            summary["approachable_component_count"] += 1
        if raw_rejection_reasons:
            for reason in raw_rejection_reasons:
                existing = int(summary["raw_component_rejection_counts"].get(reason, 0) or 0)
                summary["raw_component_rejection_counts"][reason] = existing + 1
            if len(summary["raw_component_rejections"]) < 12:
                summary["raw_component_rejections"].append(
                    {
                        "component_id": int(raw_component_id),
                        "reasons": raw_rejection_reasons,
                        "cell_count": int(raw_component_cells.shape[0]),
                        "component_area_m2": round(float(raw_area), 5),
                        "component_width_m": round(float(raw_width), 4),
                        "component_depth_m": round(float(raw_depth), 4),
                        "contains_far_cells": contains_far_cells,
                    }
                )
    summary["has_approachable_free_space"] = bool(summary["approachable_component_count"])
    far_count, far_labels, _ = _connected_components(far_free_mask)
    for far_component_id in range(1, far_count):
        far_component_cells = np.argwhere(far_labels == far_component_id)
        if far_component_cells.shape[0] <= 0:
            continue
        far_area = float(far_component_cells.shape[0] * resolution * resolution)
        far_depth = float((int(far_component_cells[:, 0].max()) - int(far_component_cells[:, 0].min()) + 1) * resolution)
        far_width = float((int(far_component_cells[:, 1].max()) - int(far_component_cells[:, 1].min()) + 1) * resolution)
        far_rejection_reasons: List[str] = []
        if far_area < min_area:
            far_rejection_reasons.append("too_small")
        if far_width < min_width:
            far_rejection_reasons.append("too_narrow")
        if far_depth < min_depth:
            far_rejection_reasons.append("too_shallow")
        if not far_rejection_reasons:
            summary["far_placeable_component_count"] += 1
        else:
            for reason in far_rejection_reasons:
                existing = int(summary["far_component_rejection_counts"].get(reason, 0) or 0)
                summary["far_component_rejection_counts"][reason] = existing + 1
            if len(summary["far_component_rejections"]) < 12:
                summary["far_component_rejections"].append(
                    {
                        "component_id": int(far_component_id),
                        "reasons": far_rejection_reasons,
                        "cell_count": int(far_component_cells.shape[0]),
                        "component_area_m2": round(float(far_area), 5),
                        "component_width_m": round(float(far_width), 4),
                        "component_depth_m": round(float(far_depth), 4),
                    }
                )
    summary["has_far_placeable_free_space"] = bool(summary["far_placeable_component_count"])
    candidates: List[JsonDict] = []

    def record_component_rejection(component_id: int, reasons: List[str], detail: JsonDict) -> None:
        counts = summary["component_rejection_counts"]
        for reason in reasons:
            counts[reason] = int(counts.get(reason, 0) or 0) + 1
        if len(summary["component_rejections"]) < 12:
            summary["component_rejections"].append(
                {
                    "component_id": int(component_id),
                    "reasons": list(reasons),
                    **detail,
                }
            )

    for component_id in range(1, count):
        component_mask = labels == component_id
        cell_count = int(np.count_nonzero(component_mask))
        if cell_count <= 0:
            continue
        component_area = float(cell_count * resolution * resolution)
        component_cells = np.argwhere(component_mask)
        component_depth = float((int(component_cells[:, 0].max()) - int(component_cells[:, 0].min()) + 1) * resolution)
        component_width = float((int(component_cells[:, 1].max()) - int(component_cells[:, 1].min()) + 1) * resolution)
        rejection_reasons: List[str] = []
        if component_area < min_area:
            rejection_reasons.append("too_small")
        if component_width < min_width:
            rejection_reasons.append("too_narrow")
        if component_depth < min_depth:
            rejection_reasons.append("too_shallow")
        if rejection_reasons:
            record_component_rejection(
                component_id,
                rejection_reasons,
                {
                    "cell_count": int(cell_count),
                    "component_area_m2": round(float(component_area), 5),
                    "component_width_m": round(float(component_width), 4),
                    "component_depth_m": round(float(component_depth), 4),
                },
            )
            continue
        safe_cells = _component_safe_cells(
            component_mask,
            resolution=resolution,
            max_points=max_points_per_component * 4,
            min_separation_m=point_separation_m,
        )
        surface_component = labels[surface_rows, surface_cols] == component_id
        if not bool(np.any(surface_component)):
            first_row, first_col, _ = safe_cells[0]
            first_local = np.asarray(
                [u_min + (first_col + 0.5) * resolution, v_min + (first_row + 0.5) * resolution],
                dtype="float64",
            )
            nearest_index = int(np.argmin(np.sum((surface_local - first_local) ** 2, axis=1)))
            component_uv = surface_uv[[nearest_index]]
            component_surface_points = surface_xyz[[nearest_index]]
        else:
            component_uv = surface_uv[surface_component]
            component_surface_points = surface_xyz[surface_component]
        placement_points: List[JsonDict] = []
        seen_pixels: set[Tuple[int, int]] = set()
        component_local_points = _project_plane_local(component_surface_points, frame)
        for point_rank, (point_row, point_col, clearance_m) in enumerate(safe_cells, start=1):
            point_local = np.asarray(
                [u_min + (point_col + 0.5) * resolution, v_min + (point_row + 0.5) * resolution],
                dtype="float64",
            )
            point_xyz = frame["origin"] + point_local[0] * frame["u_axis"] + point_local[1] * frame["v_axis"]
            point_distance_m = float(math.hypot(float(point_xyz[0]), float(point_xyz[2])))
            if point_distance_m < min_distance or point_distance_m > max_distance:
                continue
            point_surface_index = int(np.argmin(np.sum((component_local_points - point_local) ** 2, axis=1)))
            point_uv = component_uv[point_surface_index]
            pixel_key = (int(round(float(point_uv[0]))), int(round(float(point_uv[1]))))
            if point_inside_visible_blocker(float(pixel_key[0]), float(pixel_key[1])):
                summary["mapped_point_hard_rejection_count"] += 1
                continue
            if pixel_key in seen_pixels:
                continue
            seen_pixels.add(pixel_key)
            placement_points.append(
                {
                    "rank": int(point_rank),
                    "x": float(pixel_key[0]),
                    "y": float(pixel_key[1]),
                    "grid_cell": [int(point_row), int(point_col)],
                    "clearance_m": round(float(clearance_m), 4),
                    "center_3d": {
                        "x": round(float(point_xyz[0]), 4),
                        "y": round(float(point_xyz[1]), 4),
                        "z": round(float(point_xyz[2]), 4),
                        "ground_forward_m": round(float(point_xyz[2]), 4),
                        "ground_distance_m": round(float(point_distance_m), 4),
                    },
                }
            )
            if len(placement_points) >= max_points_per_component:
                break
        if not placement_points:
            record_component_rejection(
                component_id,
                ["no_reachable_safe_center"],
                {"cell_count": int(cell_count)},
            )
            continue
        primary_point = placement_points[0]
        safe_row, safe_col = [int(item) for item in primary_point["grid_cell"]]
        primary_clearance_m = float(primary_point.get("clearance_m", 0.0) or 0.0)
        safe_local = np.asarray(
            [u_min + (safe_col + 0.5) * resolution, v_min + (safe_row + 0.5) * resolution],
            dtype="float64",
        )
        safe_xyz = frame["origin"] + safe_local[0] * frame["u_axis"] + safe_local[1] * frame["v_axis"]
        distance_m = float(math.hypot(float(safe_xyz[0]), float(safe_xyz[2])))
        center_2d = {"x": int(round(float(primary_point["x"]))), "y": int(round(float(primary_point["y"])))}
        selected_uv = np.asarray([float(primary_point["x"]), float(primary_point["y"])], dtype="float64")
        x1 = max(0.0, float(np.min(component_uv[:, 0])))
        y1 = max(0.0, float(np.min(component_uv[:, 1])))
        x2 = min(float(image_w), float(np.max(component_uv[:, 0])))
        y2 = min(float(image_h), float(np.max(component_uv[:, 1])))
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        bbox = {"x": int(round(x1)), "y": int(round(y1)), "w": int(round(width_px)), "h": int(round(height_px))}
        bearing_deg = math.degrees(math.atan2(float(safe_xyz[0]), max(0.001, float(safe_xyz[2]))))
        completion = copy.deepcopy(region)
        completion_id = (
            f"pc_grid:{parent_object}:{int(region.get('plane_id', 0) or 0)}:"
            f"{int(region.get('dbscan_cluster_id', 0) or 0)}:{component_id}:"
            f"{bbox['x']}:{bbox['y']}:{bbox['w']}:{bbox['h']}"
        )
        completion.update(
            {
                "id": completion_id,
                "surface_candidate_id": completion_id,
                "source": POINTCLOUD_GRID_COMPLETION_SOURCE,
                "surface_candidate_source": POINTCLOUD_GRID_COMPLETION_SOURCE,
                "actionability_source": POINTCLOUD_GRID_COMPLETION_SOURCE,
                "bbox": bbox,
                "bbox_2d": dict(bbox),
                "region_bbox": dict(bbox),
                "center": center_2d,
                "center_2d": dict(center_2d),
                "interaction_point": {"x": float(center_2d["x"]), "y": float(center_2d["y"])},
                "placement_points": placement_points,
                "placement_point_count": int(len(placement_points)),
                "center_3d": {
                    "x": round(float(safe_xyz[0]), 4),
                    "y": round(float(safe_xyz[1]), 4),
                    "z": round(float(safe_xyz[2]), 4),
                    "ground_forward_m": round(float(safe_xyz[2]), 4),
                    "ground_distance_m": round(float(distance_m), 4),
                },
                "region_area_m2": round(float(component_area), 5),
                "region_width_m": round(float(component_width), 4),
                "region_depth_m": round(float(component_depth), 4),
                "distance_m": round(float(distance_m), 4),
                "height_m": round(float(safe_xyz[1]), 4),
                "point_count": int(component_uv.shape[0]),
                "cluster_size": int(cell_count),
                "density": round(float(component_uv.shape[0] / max(1e-6, component_area)), 4),
                "edge_margin_m": round(float(edge_margin_m), 4),
                "position_hint": "front-left" if bearing_deg < -8.0 else "front-right" if bearing_deg > 8.0 else "front-center",
                "surface_hint": "support_surface",
                "is_floor_level": False,
                "is_support_surface": True,
                "reachable": True,
                "pickup_now": False,
                "place_now": False,
                "visual_place_ready": True,
                "affordance_ready": True,
                "final_place_ready": False,
                "executor_precheck_required": True,
                "cleanable_now": False,
                "needs_alignment": bool(abs(bearing_deg) > env_float("ROBOT_DEPTH_CENTER_TOLERANCE_DEG", 8.0)),
                "needs_approach": False,
                "obstacle_risk": False,
                "visual_box_ambiguous": False,
                "region_type": "placeable_surface_region",
                "affordance": ["place"],
                "place_affordance": {
                    "action": "place",
                    "affordance_ready": True,
                    "source": POINTCLOUD_GRID_COMPLETION_SOURCE,
                    "requires_executor_precheck": True,
                },
                "rejection_reasons": [],
                "affordance_rejection_reasons": [],
                "blocked": False,
                "blocked_by": [],
                "blocker_records": [],
                "area": round(float(width_px * height_px), 2),
                "area_ratio": round(float((width_px * height_px) / max(1.0, float(image_w * image_h))), 6),
                "region_area_px": int(round(width_px * height_px)),
                "region_area_ratio": round(float((width_px * height_px) / max(1.0, float(image_w * image_h))), 6),
                "bearing_deg": round(float(bearing_deg), 3),
                "distance": round(float(distance_m), 4),
                "ground_distance": round(float(distance_m), 4),
                "height": round(float(safe_xyz[1]), 4),
            }
        )
        completion["geometry_checks"] = dict(region.get("geometry_checks") or {})
        completion["geometry_checks"].update(
            {
                "grid_component_area_ok": True,
                "grid_component_width_ok": True,
                "grid_component_depth_ok": True,
                "grid_edge_eroded": True,
                "grid_occupancy_clear": True,
                "distance_ok": True,
                "raised_object_top": False,
                "completion_requires_executor_precheck": True,
            }
        )
        completion["occupancy_checks"] = {
            "blocked": False,
            "blocked_by": [],
            "blocker_records": [],
            "source_blocker_records": occupancy_records,
            "skipped_blockers": region.get("skipped_blockers") or [],
            "held_object_ignored_as_blocker": bool(region.get("held_object_ignored_as_blocker")),
            "free_space_grid_completion": True,
            "occupancy_source_counts": dict(summary["occupancy_source_counts"]),
        }
        completion["geometry"] = dict(region.get("geometry") or {})
        completion["geometry"].update(
            {
                "cx_ratio": round(float(center_2d["x"] / max(1.0, image_w)), 4),
                "cy_ratio": round(float(center_2d["y"] / max(1.0, image_h)), 4),
                "area_ratio": completion["area_ratio"],
                "bearing_deg": round(float(bearing_deg), 3),
                "distance_m": round(float(distance_m), 4),
                "ground_distance_m": round(float(distance_m), 4),
                "height_m": round(float(safe_xyz[1]), 4),
            }
        )
        completion["executor_checks"] = {
            "precheck_supported": True,
            "precheck_ok": False,
            "reason": "precheck_not_run",
            "suggested_recovery": None,
        }
        completion["affordance_reasons"] = list(region.get("affordance_reasons") or []) + [
            "plane_local_grid_completion",
            "executor_precheck_required",
        ]
        completion["free_space_completion"] = {
            "mode": "plane_local_2d_grid",
            "source_surface_id": region.get("id"),
            "grid_resolution_m": round(float(resolution), 4),
            "component_id": int(component_id),
            "component_count": int(max(0, count - 1)),
            "component_area_m2": round(float(component_area), 5),
            "component_width_m": round(float(component_width), 4),
            "component_depth_m": round(float(component_depth), 4),
            "component_point_count": int(component_uv.shape[0]),
            "selected_center_grid": [int(safe_row), int(safe_col)],
            "selected_center_uv": [round(float(safe_local[0]), 4), round(float(safe_local[1]), 4)],
            "selected_center_3d": dict(completion["center_3d"]),
            "selected_center_clearance_m": round(float(primary_clearance_m), 4),
            "placement_point_count": int(len(placement_points)),
            "occupancy_source_counts": dict(summary["occupancy_source_counts"]),
            "edge_margin_m": round(float(edge_margin_m), 4),
            "blocker_dilate_m": round(float(blocker_dilate_m), 4),
            "held_footprint_radius_m": round(float(footprint_radius_m), 4),
            "placement_clearance_m": round(float(placement_clearance_m), 4),
            "method": "surface_mask_minus_occupancy_mask_connected_components",
        }
        completion["placement_safety_contract"] = {
            "version": "plane_local_grid_v1",
            "clearance_owner": "pointcloud_plane_local_grid",
            "surface_source": POINTCLOUD_GRID_COMPLETION_SOURCE,
            "grid_occupancy_clear": True,
            "grid_edge_eroded": True,
            "execution_target": "placement_points.center_3d",
            "requires_exact_target_execution": True,
            "target_coordinate_frame": target_coordinate_frame,
            "held_footprint_radius_m": round(float(footprint_radius_m), 4),
            "placement_clearance_m": round(float(placement_clearance_m), 4),
            "edge_margin_m": round(float(edge_margin_m), 4),
            "blocker_dilate_m": round(float(blocker_dilate_m), 4),
            "placement_point_count": int(len(placement_points)),
        }
        if camera_height_m is not None:
            completion["placement_safety_contract"]["camera_height_m"] = round(float(camera_height_m), 4)
        completion["score"] = round(
            max(score_pointcloud_surface_region(completion), env_float("ROBOT_PC_FREE_GRID_MIN_SCORE", 0.54)),
            4,
        )
        completion["affordance_score"] = completion["score"]
        candidates.append(completion)
    candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    max_components = max(1, env_int("ROBOT_PC_FREE_GRID_MAX_COMPONENTS", 8))
    kept = candidates[:max_components]
    summary["component_kept"] = len(kept)
    summary["candidate_ids"] = [item.get("id") for item in kept]
    summary["_grid_debug_selected_points"] = [
        dict(item.get("interaction_point") or {})
        for item in kept
        if isinstance(item.get("interaction_point"), dict)
    ]
    if kept:
        summary["reason"] = "grid_components_ready"
        summary["result"] = "grid_components_ready"
    elif int(summary.get("reachable_free_cell_count", 0) or 0) > 0:
        summary["reason"] = "no_grid_component_passed_safety_checks"
        summary["result"] = "no_grid_component_passed_safety_checks"
        summary["suggested_recovery"] = "change_viewpoint_or_select_other_surface"
    elif int(summary.get("raw_free_cell_count", 0) or 0) > 0:
        summary["reason"] = "free_space_outside_current_reach"
        summary["result"] = "free_space_outside_current_reach"
        summary["suggested_recovery"] = "reposition_for_reachable_surface"
    else:
        summary["reason"] = "no_grid_free_mask_after_clearance"
        summary["result"] = "no_grid_free_mask_after_clearance"
        summary["suggested_recovery"] = "change_viewpoint_or_select_other_surface"
    return kept


#把 cluster 变成 surface region
#几何过滤
"""它会计算：
bbox_2d
center_2d
interaction_point
center_3d
region_area_m2
region_width_m
region_depth_m
distance_m
height_m
normal_up_score
point_count
density
bearing_deg

然后做几何检查：
area_ok
width_ok
depth_ok
normal_ok
height_ok
distance_ok
point_count_ok
image_edge_ok
parent_edge_ok
thin_region
edge_only_region

只要这些几何条件都过，才可能是 geometry_ready。"""
def build_region_from_cluster(
    cluster: JsonDict,
    *,
    parent: JsonDict,
    blockers: Iterable[JsonDict],
    scene_points: Optional[JsonDict],
    image_w: int,
    image_h: int,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
    skipped_blockers: Optional[List[JsonDict]] = None,
) -> JsonDict:
    import numpy as np  # type: ignore
    import open3d as o3d  # type: ignore

    xyz = np.asarray(cluster.get("xyz", []), dtype="float64")
    uv = np.asarray(cluster.get("uv", []), dtype="float64")
    parent_label = str(parent.get("label") or parent.get("raw_label") or "surface")
    parent_object = normalize_label(parent_label)
    if xyz.ndim != 2 or xyz.shape[0] == 0 or uv.ndim != 2:
        return {}

    x1 = max(0.0, float(np.min(uv[:, 0])))
    y1 = max(0.0, float(np.min(uv[:, 1])))
    x2 = min(float(image_w), float(np.max(uv[:, 0])))
    y2 = min(float(image_h), float(np.max(uv[:, 1])))
    width_px = max(1.0, x2 - x1)
    height_px = max(1.0, y2 - y1)
    bbox = {"x": int(round(x1)), "y": int(round(y1)), "w": int(round(width_px)), "h": int(round(height_px))}
    center_2d = {"x": int(round(float(np.median(uv[:, 0])))), "y": int(round(float(np.median(uv[:, 1]))))}
    center_xyz = np.median(xyz, axis=0)
    point_count = int(xyz.shape[0])

    pcd = build_open3d_pointcloud({"xyz": xyz})
    try:
        obb = pcd.get_oriented_bounding_box()
        extent = np.asarray(obb.extent, dtype="float64")
    except Exception:
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(xyz))
        extent = np.asarray(aabb.get_extent(), dtype="float64")
    x_extent = float(np.max(xyz[:, 0]) - np.min(xyz[:, 0]))
    z_extent = float(np.max(xyz[:, 2]) - np.min(xyz[:, 2]))
    horizontal_extents = sorted([max(x_extent, float(extent[0])), max(z_extent, float(extent[2]))], reverse=True)
    region_width_m = float(horizontal_extents[0])
    region_depth_m = float(horizontal_extents[1])
    region_area_m2 = float(max(0.0, region_width_m * region_depth_m))
    density = float(point_count / max(1e-6, region_area_m2))
    normal = cluster.get("plane_normal") if isinstance(cluster.get("plane_normal"), list) else [0.0, 1.0, 0.0]
    normal_up_score = float(cluster.get("normal_up_score", 0.0) or 0.0)
    distance_m = float(math.hypot(float(center_xyz[0]), float(center_xyz[2])))
    height_m = float(center_xyz[1])
    bearing_deg = math.degrees(math.atan2(float(center_xyz[0]), max(0.001, float(center_xyz[2]))))

    parent_box = bbox_xyxy(parent)
    if parent_box is None:
        parent_box = (0.0, 0.0, float(image_w), float(image_h))
    image_margin = env_float("ROBOT_PC_IMAGE_EDGE_MARGIN_PIXELS", 8.0)
    parent_margin = env_float("ROBOT_PC_PARENT_EDGE_MARGIN_PIXELS", 6.0)
    image_edge_ok = bool(
        x1 >= image_margin
        and y1 >= image_margin
        and x2 <= float(image_w) - image_margin
        and y2 <= float(image_h) - image_margin
    )
    parent_left_ok = bool(x1 >= parent_box[0] + parent_margin)
    parent_top_ok = bool(y1 >= parent_box[1] + parent_margin)
    parent_right_ok = bool(x2 <= parent_box[2] - parent_margin)
    parent_bottom_ok = bool(y2 <= parent_box[3] - parent_margin)
    parent_edge_strict_ok = bool(parent_left_ok and parent_top_ok and parent_right_ok and parent_bottom_ok)
    min_area = env_float("ROBOT_PC_REGION_MIN_AREA_M2", 0.018)
    min_width = env_float("ROBOT_PC_REGION_MIN_WIDTH_M", 0.12)
    min_depth = env_float("ROBOT_PC_REGION_MIN_DEPTH_M", 0.10)
    min_height, max_height = support_surface_height_range(parent_object)
    min_distance, max_distance = pointcloud_ready_distance_range_m()
    min_points = env_int("ROBOT_PC_REGION_MIN_POINTS", 60)
    min_up = env_float("ROBOT_PC_MIN_NORMAL_UP_SCORE", 0.85)
    aspect_ratio = region_width_m / max(0.001, region_depth_m)
    max_aspect = env_float("ROBOT_PC_REGION_MAX_ASPECT_RATIO", 5.0)
    aspect_depth_floor = env_float("ROBOT_PC_REGION_ASPECT_DEPTH_FLOOR_M", 0.16)
    thin_region = bool(
        region_depth_m < min_depth
        or (aspect_ratio > max_aspect and region_depth_m < aspect_depth_floor)
    )
    edge_only_region = bool(height_px < env_float("ROBOT_PC_REGION_MIN_BBOX_H_PIXELS", 28.0) or region_depth_m < min_depth)
    parent_top_edge_relaxed = bool(
        not parent_edge_strict_ok
        and allow_countertop_parent_top_edge(parent_object)
        and not parent_top_ok
        and parent_left_ok
        and parent_right_ok
        and parent_bottom_ok
        and y1 <= parent_box[1] + parent_margin
        and y1 >= parent_box[1] - env_float("ROBOT_PC_COUNTERTOP_PARENT_TOP_EDGE_OUTSIDE_TOL_PIXELS", 2.0)
        and width_px >= env_float("ROBOT_PC_COUNTERTOP_PARENT_TOP_EDGE_MIN_WIDTH_PIXELS", 24.0)
        and height_px >= env_float("ROBOT_PC_COUNTERTOP_PARENT_TOP_EDGE_MIN_HEIGHT_PIXELS", 12.0)
        and region_area_m2 >= min_area
        and region_width_m >= min_width
        and region_depth_m >= min_depth
        and normal_up_score >= min_up
        and min_height <= height_m <= max_height
    )
    parent_edge_ok = bool(parent_edge_strict_ok or parent_top_edge_relaxed)
    region_xz = (
        float(np.min(xyz[:, 0])),
        float(np.max(xyz[:, 0])),
        float(np.min(xyz[:, 2])),
        float(np.max(xyz[:, 2])),
    )
    #还要检查 blocker，占用才会影响能不能放
#     比如：

# blocked:apple
# blocked:cup
# blocked:plate
    blocker_records, skipped_records = blocker_records_for_region(
        (x1, y1, x2, y2),
        blockers,
        margin=env_float("ROBOT_PC_OCCUPANCY_MARGIN_PIXELS", 12.0),
        image_w=image_w,
        image_h=image_h,
        parent=parent,
        scene_points=scene_points,
        region_xz=region_xz,
        plane_height_m=height_m,
        holding_object=holding_object,
        held_object_labels=held_object_labels,
        held_object_family=held_object_family,
        skipped_blockers=skipped_blockers,
    )
    blocked_by = [str(item.get("blocked_token") or item.get("label") or "object") for item in blocker_records]
    blocked = bool(blocked_by)
    held_overlay_records = [
        item for item in skipped_records
        if str(item.get("reason") or "") == "likely_held_object_overlay"
    ]
    held_overlay_ignored = bool(held_overlay_records)
    #代码里真正决定 geometry_ready 的条件是这一组： 第四层：几何过滤条件 checks
    """条件	   默认阈值	        不满足时原因
    面积够大	>= 0.018 m²	too_small
    宽度够大	>= 0.12 m	too_narrow
    深度够大	>= 0.10 m	too_shallow
    法向量朝上	normal_up_score >= 0.85	not_horizontal
    高度合理	0.55m ~ 1.15m	height_out_of_range
    距离合理	0.55m ~ 1.50m	too_close / too_far
    点数够多	>= 60	too_few_points
    不贴图像边缘	边缘留 8 px	touches_image_edge
    不贴父物体边缘	边缘留 6 px	touches_parent_edge
    不能太薄	深度不能太小，长宽比不能太夸张	thin_region
    不能只是边缘线	图像高度不能太小，深度不能太小	edge_only_region"""
    checks = {
        "area_ok": bool(region_area_m2 >= min_area),
        "width_ok": bool(region_width_m >= min_width),
        "depth_ok": bool(region_depth_m >= min_depth),
        "normal_ok": bool(normal_up_score >= min_up),
        "height_ok": bool(min_height <= height_m <= max_height),
        "distance_ok": bool(min_distance <= distance_m <= max_distance),
        "point_count_ok": bool(point_count >= min_points),
        "image_edge_ok": bool(image_edge_ok),
        "parent_edge_ok": bool(parent_edge_ok),
        "parent_edge_strict_ok": bool(parent_edge_strict_ok),
        "parent_top_edge_relaxed": bool(parent_top_edge_relaxed),
        "thin_region": bool(thin_region),
        "edge_only_region": bool(edge_only_region),
        "support_height_min_m": round(float(min_height), 4),
        "support_height_max_m": round(float(max_height), 4),
        "raised_object_top": False,
    }

    rejection_reasons: List[str] = []
    if not checks["area_ok"]:
        rejection_reasons.append("too_small")
    if not checks["width_ok"]:
        rejection_reasons.append("too_narrow")
    if not checks["depth_ok"]:
        rejection_reasons.append("too_shallow")
    if not checks["normal_ok"]:
        rejection_reasons.append("not_horizontal")
    if not checks["height_ok"]:
        rejection_reasons.append("height_out_of_range")
    if distance_m < min_distance:
        rejection_reasons.append("too_close")
    elif distance_m > max_distance:
        rejection_reasons.append("too_far")
    if not checks["point_count_ok"]:
        rejection_reasons.append("too_few_points")
    if not image_edge_ok:
        rejection_reasons.append("touches_image_edge")
    if not parent_edge_ok:
        rejection_reasons.append("touches_parent_edge")
    if thin_region:
        rejection_reasons.append("thin_region")
    if edge_only_region:
        rejection_reasons.append("edge_only_region")
    for blocker in blocked_by:
        rejection_reasons.append(f"blocked:{blocker}")

    geometry_ready = bool(
        checks["area_ok"]
        and checks["width_ok"]
        and checks["depth_ok"]
        and checks["normal_ok"]
        and checks["height_ok"]
        and checks["distance_ok"]
        and checks["point_count_ok"]
        and image_edge_ok
        and parent_edge_ok
        and not thin_region
        and not edge_only_region
    )
    """geometry_ready = 几何形状合格
blocked = 没有被其他东西占用

两个都满足，才是：

visual_place_ready = true"""
    visual_place_ready = bool(geometry_ready and not blocked)
    affordance_reasons = ["semantic_parent_table_like", "horizontal_support_plane"] if parent_object not in POINTCLOUD_LOW_PRIORITY_PARENTS else []
    affordance_rejections = list(rejection_reasons)
    """即使它已经是 visual_place_ready，还要看它的父物体语义是不是合适。

比如父物体是：
table
countertop
desk
shelf
那比较像可放置支撑面。

但如果父物体属于低优先级父类：
POINTCLOUD_LOW_PRIORITY_PARENTS

那它可能不会给 place affordance。"""
    affordance_ready = bool(visual_place_ready and parent_object not in POINTCLOUD_LOW_PRIORITY_PARENTS)
    if parent_object in POINTCLOUD_LOW_PRIORITY_PARENTS:
        affordance_rejections.append("low_priority_parent")

    candidate_id = pointcloud_surface_id(
        parent_label,
        bbox,
        height_m,
        distance_m,
        int(cluster.get("plane_id", 0) or 0),
        int(cluster.get("dbscan_cluster_id", 0) or 0),
    )
    region: JsonDict = {
        "id": candidate_id,
        "surface_candidate_id": candidate_id,
        "label": "pc_surface",
        "raw_label": parent.get("raw_label") or parent.get("label"),
        "task_semantic_class": "place_receptacle",
        "confidence": round(float(parent.get("confidence", 0.0) or 0.0), 4),
        "bbox": bbox,
        "bbox_2d": dict(bbox),
        "region_bbox": dict(bbox),
        "image_size": {"w": int(image_w), "h": int(image_h)},
        "center": center_2d,
        "center_2d": dict(center_2d),
        "interaction_point": {"x": float(center_2d["x"]), "y": float(center_2d["y"])},
        "center_3d": {
            "x": round(float(center_xyz[0]), 4),
            "y": round(float(center_xyz[1]), 4),
            "z": round(float(center_xyz[2]), 4),
            "ground_forward_m": round(float(center_xyz[2]), 4),
            "ground_distance_m": round(float(distance_m), 4),
        },
        "plane_id": int(cluster.get("plane_id", 0) or 0),
        "dbscan_cluster_id": int(cluster.get("dbscan_cluster_id", 0) or 0),
        "plane_normal": normal,
        "normal_up_score": round(float(normal_up_score), 4),
        "region_area_m2": round(float(region_area_m2), 5),
        "region_width_m": round(float(region_width_m), 4),
        "region_depth_m": round(float(region_depth_m), 4),
        "distance_m": round(float(distance_m), 4),
        "height_m": round(float(height_m), 4),
        "point_count": int(point_count),
        "cluster_size": int(cluster.get("cluster_size", point_count) or point_count),
        "density": round(float(density), 4),
        "edge_margin_m": round(float(min(region_width_m, region_depth_m) / 2.0), 4),
        "position_hint": "front-left" if bearing_deg < -8.0 else "front-right" if bearing_deg > 8.0 else "front-center",
        "surface_hint": "support_surface",
        "is_floor_level": False,
        "is_support_surface": True,
        "reachable": bool(visual_place_ready),
        "pickup_now": False,
        "place_now": False,
        "visual_place_ready": bool(visual_place_ready),
        "affordance_ready": bool(affordance_ready),
        "final_place_ready": False,
        "cleanable_now": False,
        "needs_alignment": bool(visual_place_ready and abs(bearing_deg) > env_float("ROBOT_DEPTH_CENTER_TOLERANCE_DEG", 8.0)),
        "needs_approach": False,
        "obstacle_risk": False,
        "visual_box_ambiguous": False,
        "region_type": "placeable_surface_region" if visual_place_ready else "blocked_surface_region" if blocked else "rejected_surface_region",
        "affordance": ["place"] if affordance_ready else [],
        "place_affordance": {
            "action": "place",
            "affordance_ready": bool(affordance_ready),
        },
        "affordance_reasons": affordance_reasons,
        "affordance_rejection_reasons": affordance_rejections,
        "parent_object": parent_object,
        "parent_label": parent_label,
        "parent_bbox": parent.get("bbox"),
        "source": "pointcloud_plane",
        "surface_candidate_source": "pointcloud_plane",
        "actionability_source": "pointcloud_plane",
        "geometry_checks": checks,
        "occupancy_checks": {
            "blocked": bool(blocked),
            "blocked_by": blocked_by,
            "blocker_records": blocker_records,
            "skipped_blockers": skipped_records,
            "held_object_ignored_as_blocker": bool(held_overlay_ignored),
        },
        "memory_checks": {"failed_recently": False, "cooldown_remaining": 0},
        "executor_checks": {
            "precheck_supported": True,
            "precheck_ok": False,
            "reason": "precheck_not_run",
            "suggested_recovery": None,
        },
        "blocked": bool(blocked),
        "blocked_by": blocked_by,
        "blocker_records": blocker_records,
        "skipped_blockers": skipped_records,
        "held_object_ignored_as_blocker": bool(held_overlay_ignored),
        "rejection_reasons": rejection_reasons,
        "bearing_deg": round(float(bearing_deg), 3),
        "area": round(float(width_px * height_px), 2),
        "area_ratio": round(float((width_px * height_px) / max(1.0, float(image_w * image_h))), 6),
        "region_area_px": int(round(width_px * height_px)),
        "region_area_ratio": round(float((width_px * height_px) / max(1.0, float(image_w * image_h))), 6),
        "geometry": {
            "cx_ratio": round(float(center_2d["x"] / max(1.0, image_w)), 4),
            "cy_ratio": round(float(center_2d["y"] / max(1.0, image_h)), 4),
            "area_ratio": round(float((width_px * height_px) / max(1.0, float(image_w * image_h))), 6),
            "bearing_deg": round(float(bearing_deg), 3),
            "distance_m": round(float(distance_m), 4),
            "ground_distance_m": round(float(distance_m), 4),
            "height_m": round(float(height_m), 4),
            "normal_up_score": round(float(normal_up_score), 4),
        },
        "distance": round(float(distance_m), 4),
        "ground_distance": round(float(distance_m), 4),
        "height": round(float(height_m), 4),
    }
    # Private surface samples feed plane-local completion only. They are stripped
    # before the perception result leaves this module.
    region["_surface_xyz"] = xyz
    region["_surface_uv"] = uv
    region["score"] = score_pointcloud_surface_region(region)
    region["affordance_score"] = region["score"]
    return region

"""这个函数的含义是：如果某个区域本来因为“手持物体前景遮挡”导致可见面积变小，但系统确认这个 blocker 是手里拿着的物体，并且几何核心条件还不错，就生成一个 completion region。

它会把原 region 复制一份，然后改成：
"source": "pointcloud_plane_completion"
"visual_place_ready": true
"affordance_ready": true
"region_type": "placeable_surface_region"
"affordance": ["place"]

同时加上：
"held_object_plane_completion"
"executor_precheck_required"

这意思是：
虽然这块区域被手持物体遮住了一部分，
但系统认为它背后可能是连续台面，
所以允许作为候选，
不过必须经过 executor_precheck。

这个设计很适合你现在“拿着苹果找台面”的场景。否则手里的苹果永远挡住视野，系统可能永远找不到 ready surface。

但它也有风险：这属于“补全猜测”，所以必须依赖 executor precheck，不能直接执行 place。"""
def build_completion_region_from_held_overlay(region: JsonDict) -> Optional[JsonDict]:
    if not env_bool("ROBOT_PC_HELD_OVERLAY_COMPLETION_ENABLED", True):
        return None
    if not bool(region.get("held_object_ignored_as_blocker")):
        return None
    if bool(region.get("blocked")):
        return None
    rejection_reasons = {
        str(reason)
        for reason in (region.get("rejection_reasons") or [])
        if str(reason)
    }
    disallowed_reasons = {
        "too_small",
        "too_narrow",
        "too_shallow",
        "too_few_points",
        "touches_image_edge",
        "touches_parent_edge",
        "thin_region",
        "edge_only_region",
        "raised_object_top",
        "not_horizontal",
        "height_out_of_range",
    }
    if rejection_reasons & disallowed_reasons:
        return None
    parent_object = normalize_label(region.get("parent_object") or region.get("parent_label") or "")
    if parent_object in POINTCLOUD_LOW_PRIORITY_PARENTS:
        return None
    checks = region.get("geometry_checks") if isinstance(region.get("geometry_checks"), dict) else {}
    required_checks = ("normal_ok", "height_ok", "distance_ok", "point_count_ok", "image_edge_ok", "parent_edge_ok")
    if not all(bool(checks.get(key)) for key in required_checks):
        return None
    try:
        visible_area = float(region.get("region_area_m2", 0.0) or 0.0)
        visible_width = float(region.get("region_width_m", 0.0) or 0.0)
        visible_depth = float(region.get("region_depth_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if visible_area < env_float("ROBOT_PC_COMPLETION_MIN_VISIBLE_AREA_M2", 0.006):
        return None
    if visible_width < env_float("ROBOT_PC_COMPLETION_MIN_VISIBLE_WIDTH_M", 0.07):
        return None
    if visible_depth < env_float("ROBOT_PC_COMPLETION_MIN_VISIBLE_DEPTH_M", 0.045):
        return None

    completion = copy.deepcopy(region)
    completion_id = str(region.get("id") or "pc_surface").replace("pc_surface:", "pc_completion:", 1)
    if completion_id == str(region.get("id") or ""):
        completion_id = f"pc_completion:{completion_id}"
    completion["id"] = completion_id
    completion["surface_candidate_id"] = completion_id
    completion["source"] = POINTCLOUD_COMPLETION_SOURCE
    completion["surface_candidate_source"] = POINTCLOUD_COMPLETION_SOURCE
    completion["actionability_source"] = POINTCLOUD_COMPLETION_SOURCE
    completion["region_type"] = "placeable_surface_region"
    completion["visual_place_ready"] = True
    completion["affordance_ready"] = True
    completion["reachable"] = True
    completion["place_now"] = False
    completion["final_place_ready"] = False
    completion["rejection_reasons"] = []
    completion["affordance"] = ["place"]
    completion["affordance_reasons"] = list(completion.get("affordance_reasons") or []) + [
        "held_object_plane_completion",
        "executor_precheck_required",
    ]
    completion["affordance_rejection_reasons"] = []
    completion["place_affordance"] = {
        "action": "place",
        "affordance_ready": True,
        "source": POINTCLOUD_COMPLETION_SOURCE,
        "requires_executor_precheck": True,
    }
    completion["geometry_checks"] = dict(checks)
    completion["geometry_checks"]["completion_from_held_overlay"] = True
    completion["geometry_checks"]["completion_requires_executor_precheck"] = True
    completion["occupancy_checks"] = dict(completion.get("occupancy_checks") or {})
    completion["occupancy_checks"]["held_overlay_completion"] = True
    completion["blocked"] = False
    completion["blocked_by"] = []
    completion["score"] = round(max(float(region.get("score", 0.0) or 0.0), env_float("ROBOT_PC_COMPLETION_MIN_SCORE", 0.52)), 4)
    completion["affordance_score"] = completion["score"]
    return completion


def save_initial_plane_visualization(
    *,
    image_path: str,
    save_path: str,
    planes: Iterable[JsonDict],
) -> str:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image_unreadable:{image_path}")
    height, width = image.shape[:2]
    debug = image.copy()
    overlay = image.copy()
    palette = [
        (46, 204, 113),
        (255, 178, 0),
        (220, 78, 255),
        (255, 108, 64),
        (60, 220, 220),
    ]
    accepted = [plane for plane in planes if isinstance(plane, dict)]
    for plane_index, plane in enumerate(accepted):
        uv = np.asarray(plane.get("uv", []), dtype="float64")
        if uv.ndim != 2 or uv.shape[0] <= 0 or uv.shape[1] < 2:
            continue
        px = np.rint(uv[:, 0]).astype("int32")
        py = np.rint(uv[:, 1]).astype("int32")
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px = px[valid]
        py = py[valid]
        if px.size <= 0:
            continue
        color = palette[plane_index % len(palette)]
        mask = np.zeros((height, width), dtype="uint8")
        mask[py, px] = 255
        mask = cv2.dilate(mask, np.ones((3, 3), dtype="uint8"), iterations=1)
        overlay[mask > 0] = color

    debug = cv2.addWeighted(overlay, 0.42, debug, 0.58, 0.0)
    for plane_index, plane in enumerate(accepted):
        uv = np.asarray(plane.get("uv", []), dtype="float64")
        if uv.ndim != 2 or uv.shape[0] <= 0 or uv.shape[1] < 2:
            continue
        px = np.clip(np.rint(uv[:, 0]).astype("int32"), 0, width - 1)
        py = np.clip(np.rint(uv[:, 1]).astype("int32"), 0, height - 1)
        x1, y1, x2, y2 = int(px.min()), int(py.min()), int(px.max()), int(py.max())
        color = palette[plane_index % len(palette)]
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)

    if not accepted:
        cv2.putText(debug, "No accepted initial support plane", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        legend_width = min(width - 16, 350)
        legend_height = 30 + 19 * len(accepted)
        legend_overlay = debug.copy()
        cv2.rectangle(legend_overlay, (8, 8), (8 + legend_width, 8 + legend_height), (0, 0, 0), -1)
        debug = cv2.addWeighted(legend_overlay, 0.70, debug, 0.30, 0.0)
        cv2.putText(debug, "Initial RANSAC support planes", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        for plane_index, plane in enumerate(accepted):
            color = palette[plane_index % len(palette)]
            row_y = 47 + plane_index * 19
            cv2.rectangle(debug, (16, row_y - 10), (27, row_y + 1), color, -1)
            label = (
                f"plane {plane_index}: h={float(plane.get('height_m', 0.0) or 0.0):.3f} "
                f"up={float(plane.get('normal_up_score', 0.0) or 0.0):.2f} "
                f"pts={int(plane.get('inlier_count', 0) or 0)}"
            )
            cv2.putText(debug, label, (34, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), debug):
        raise ValueError(f"write_failed:{save_path}")
    return str(output)


def derive_stage_visualization_paths(initial_plane_path: str) -> JsonDict:
    path = Path(initial_plane_path)
    suffix = path.suffix or ".jpg"
    stem = path.stem
    initial_suffix = "-pointcloud-planes"
    base_stem = stem[:-len(initial_suffix)] if stem.endswith(initial_suffix) else stem
    return {
        "split_regions": str(path.with_name(f"{base_stem}-pointcloud-stage-02-split-regions{suffix}")),
        "geometry_filter": str(path.with_name(f"{base_stem}-pointcloud-stage-03-geometry-filter{suffix}")),
        "occupancy_filter": str(path.with_name(f"{base_stem}-pointcloud-stage-04-occupancy-filter{suffix}")),
        "raised_top_filter": str(path.with_name(f"{base_stem}-pointcloud-stage-05-raised-top-filter{suffix}")),
        "completion_ready": str(path.with_name(f"{base_stem}-pointcloud-stage-06-completion-ready{suffix}")),
        "grid_free_mask": str(path.with_name(f"{base_stem}-pointcloud-stage-07-grid-free-mask{suffix}")),
        "grid_free_mask_all_attempts": str(path.with_name(f"{base_stem}-pointcloud-stage-07-grid-free-mask-all-attempts{suffix}")),
    }


def _write_stage_image(save_path: str, image: Any) -> str:
    import cv2  # type: ignore

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"write_failed:{save_path}")
    return str(output)


def _stage_canvas(image: Any, title: str, row_count: int) -> Tuple[Any, int]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    height, width = image.shape[:2]
    panel_width = 560
    canvas_height = max(height, 54 + max(1, row_count) * 20)
    canvas = np.full((canvas_height, width + panel_width, 3), (30, 30, 30), dtype="uint8")
    canvas[:height, :width] = image
    cv2.rectangle(canvas, (width, 0), (width + panel_width - 1, canvas_height - 1), (52, 52, 52), 1)
    cv2.putText(canvas, title, (width + 14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, width


def save_split_region_visualization(
    *,
    image_path: str,
    save_path: str,
    clusters: Iterable[JsonDict],
) -> str:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image_unreadable:{image_path}")
    height, width = image.shape[:2]
    entries = [cluster for cluster in clusters if isinstance(cluster, dict)]
    overlay = image.copy()
    annotated = image.copy()
    palette = [
        (46, 204, 113),
        (255, 178, 0),
        (220, 78, 255),
        (255, 108, 64),
        (60, 220, 220),
        (200, 120, 240),
    ]
    rows: List[Tuple[Tuple[int, int, int], str]] = []
    for index, cluster in enumerate(entries):
        uv = np.asarray(cluster.get("uv", []), dtype="float64")
        if uv.ndim != 2 or uv.shape[0] <= 0 or uv.shape[1] < 2:
            continue
        px = np.clip(np.rint(uv[:, 0]).astype("int32"), 0, width - 1)
        py = np.clip(np.rint(uv[:, 1]).astype("int32"), 0, height - 1)
        color = palette[index % len(palette)]
        mask = np.zeros((height, width), dtype="uint8")
        mask[py, px] = 255
        mask = cv2.dilate(mask, np.ones((3, 3), dtype="uint8"), iterations=1)
        overlay[mask > 0] = color
        x1, y1, x2, y2 = int(px.min()), int(py.min()), int(px.max()), int(py.max())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, str(index), (x1 + 2, max(13, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        rows.append(
            (
                color,
                f"{index:02d} P{int(cluster.get('plane_id', 0) or 0)} C{int(cluster.get('dbscan_cluster_id', 0) or 0)} "
                f"points={int(cluster.get('cluster_size', 0) or 0)} box=({x1},{y1},{x2 - x1},{y2 - y1})",
            )
        )
    blended = cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0.0)
    canvas, panel_x = _stage_canvas(blended, "02 DBSCAN connected regions", len(rows))
    if not rows:
        cv2.putText(canvas, "No connected plane region", (panel_x + 14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)
    for row_index, (color, text) in enumerate(rows):
        y = 52 + row_index * 20
        cv2.rectangle(canvas, (panel_x + 14, y - 10), (panel_x + 25, y + 1), color, -1)
        cv2.putText(canvas, text, (panel_x + 32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (230, 230, 230), 1, cv2.LINE_AA)
    return _write_stage_image(save_path, canvas)


def _geometry_ready_for_debug(region: JsonDict) -> bool:
    checks = region.get("geometry_checks") if isinstance(region.get("geometry_checks"), dict) else {}
    required = (
        "area_ok",
        "width_ok",
        "depth_ok",
        "normal_ok",
        "height_ok",
        "distance_ok",
        "point_count_ok",
        "image_edge_ok",
        "parent_edge_ok",
    )
    return bool(
        all(bool(checks.get(key)) for key in required)
        and not bool(checks.get("thin_region"))
        and not bool(checks.get("edge_only_region"))
    )


def _short_debug_reasons(region: JsonDict, *, include_blocked: bool = True) -> str:
    reasons = [
        str(reason)
        for reason in (region.get("rejection_reasons") or [])
        if include_blocked or not str(reason).startswith("blocked:")
    ]
    if not reasons:
        return "pass"
    text = ",".join(reasons[:3])
    return text if len(text) <= 48 else text[:45] + "..."


def _region_stage_style(region: JsonDict, stage: str) -> Tuple[Tuple[int, int, int], str]:
    geometry_ready = _geometry_ready_for_debug(region)
    raised = "raised_object_top" in set(str(item) for item in (region.get("rejection_reasons") or []))
    blocked = bool(region.get("blocked"))
    if stage == "geometry":
        if geometry_ready:
            return (46, 204, 113), "GEOMETRY PASS"
        return (40, 50, 230), f"REJECT {_short_debug_reasons(region, include_blocked=False)}"
    if stage == "occupancy":
        if blocked:
            blocked_by = ",".join(str(item) for item in (region.get("blocked_by") or [])[:2])
            detail = f"BLOCKED {blocked_by or 'object'}"
            if not geometry_ready:
                detail += " + geometry reject"
            return (0, 165, 255), detail
        if not geometry_ready:
            return (125, 125, 125), "PRIOR GEOMETRY REJECT"
        return (46, 204, 113), "UNOCCUPIED"
    if stage == "raised_top":
        if raised:
            return (40, 50, 230), "REJECT raised_object_top"
        if blocked:
            return (0, 165, 255), "BLOCKED (completion input)"
        if not geometry_ready:
            return (125, 125, 125), "PRIOR GEOMETRY REJECT"
        return (46, 204, 113), "SURVIVES RAISED-TOP"
    if region.get("visual_place_ready"):
        source = str(region.get("source") or "")
        return (46, 204, 113), "READY grid" if source == POINTCLOUD_GRID_COMPLETION_SOURCE else "READY completion" if source == POINTCLOUD_COMPLETION_SOURCE else "READY plane"
    if raised:
        return (40, 50, 230), "REJECT raised_object_top"
    if blocked:
        return (0, 165, 255), "BLOCKED no free completion"
    return (125, 125, 125), f"REJECT {_short_debug_reasons(region)}"


def save_region_stage_visualization(
    *,
    image_path: str,
    save_path: str,
    regions: Iterable[JsonDict],
    stage: str,
    title: str,
) -> str:
    import cv2  # type: ignore

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image_unreadable:{image_path}")
    entries = [region for region in regions if isinstance(region, dict)]
    canvas, panel_x = _stage_canvas(image.copy(), title, len(entries))
    if not entries:
        cv2.putText(canvas, "No region at this stage", (panel_x + 14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)
    for index, region in enumerate(entries):
        box = region.get("bbox") if isinstance(region.get("bbox"), dict) else {}
        x = int(round(float(box.get("x", 0) or 0)))
        y = int(round(float(box.get("y", 0) or 0)))
        w = int(round(float(box.get("w", 0) or 0)))
        h = int(round(float(box.get("h", 0) or 0)))
        color, status = _region_stage_style(region, stage)
        grid_ready_surface = bool(
            str(region.get("source") or "") == POINTCLOUD_GRID_COMPLETION_SOURCE
            and region.get("visual_place_ready")
        )
        if not grid_ready_surface:
            cv2.rectangle(canvas, (x, y), (x + max(1, w), y + max(1, h)), color, 2)
            cv2.putText(canvas, str(index), (x + 2, max(13, y + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        if grid_ready_surface:
            point = region.get("interaction_point") if isinstance(region.get("interaction_point"), dict) else {}
            try:
                point_x = int(round(float(point.get("x"))))
                point_y = int(round(float(point.get("y"))))
                cv2.drawMarker(canvas, (point_x, point_y), color, cv2.MARKER_CROSS, 18, 2)
                cv2.circle(canvas, (point_x, point_y), 6, color, 2)
                cv2.putText(canvas, str(index), (point_x + 8, max(13, point_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
            except (TypeError, ValueError):
                pass
        row_y = 52 + index * 20
        cv2.rectangle(canvas, (panel_x + 14, row_y - 10), (panel_x + 25, row_y + 1), color, -1)
        region_source = str(region.get("source") or "")
        source = "grid" if region_source == POINTCLOUD_GRID_COMPLETION_SOURCE else "free" if region_source == POINTCLOUD_COMPLETION_SOURCE else "plane"
        line = (
            f"{index:02d} {source} P{int(region.get('plane_id', 0) or 0)} "
            f"C{int(region.get('dbscan_cluster_id', 0) or 0)} {status} box=({x},{y},{w},{h})"
        )
        cv2.putText(canvas, line, (panel_x + 32, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
    return _write_stage_image(save_path, canvas)


def save_grid_free_mask_visualization(
    *,
    image_path: str,
    save_path: str,
    grid_records: Iterable[JsonDict],
    selected_only: bool = False,
) -> str:
    """Render the projected free/occupied cells without replacing them by a bbox."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image_unreadable:{image_path}")
    height, width = image.shape[:2]
    all_records = [record for record in grid_records if isinstance(record, dict)]
    selected_records = [
        record
        for record in all_records
        if int(record.get("component_kept", 0) or 0) > 0
    ]
    records = selected_records if selected_only and selected_records else all_records
    free_pixels = np.zeros((height, width), dtype=bool)
    out_of_reach_pixels = np.zeros((height, width), dtype=bool)
    occupied_pixels = np.zeros((height, width), dtype=bool)
    unsafe_pixels = np.zeros((height, width), dtype=bool)
    selected_points: List[Tuple[int, int, int]] = []

    def mark_uv(target: Any, uv_values: Any) -> int:
        values = np.asarray(uv_values, dtype="float64")
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] < 2:
            return 0
        px = np.clip(np.rint(values[:, 0]).astype("int32"), 0, width - 1)
        py = np.clip(np.rint(values[:, 1]).astype("int32"), 0, height - 1)
        target[py, px] = True
        return int(values.shape[0])

    debug_rows: List[str] = []
    for index, record in enumerate(records):
        free_count = mark_uv(free_pixels, record.get("_grid_debug_free_uv"))
        out_of_reach_count = mark_uv(out_of_reach_pixels, record.get("_grid_debug_out_of_reach_uv"))
        occupied_count = mark_uv(occupied_pixels, record.get("_grid_debug_occupied_uv"))
        unsafe_count = mark_uv(unsafe_pixels, record.get("_grid_debug_unsafe_uv"))
        for point in record.get("_grid_debug_selected_points") or []:
            if not isinstance(point, dict):
                continue
            try:
                selected_points.append(
                    (index, int(round(float(point.get("x")))), int(round(float(point.get("y")))))
                )
            except (TypeError, ValueError):
                continue
        debug_rows.append(
            f"{index:02d} P{int(record.get('_grid_debug_plane_id', 0) or 0)} "
            f"C{int(record.get('_grid_debug_cluster_id', 0) or 0)} "
            f"ready_free={free_count} out_reach={out_of_reach_count} occupied={occupied_count} edge={unsafe_count} "
            f"cc={int(record.get('component_count', 0) or 0)} kept={int(record.get('component_kept', 0) or 0)}"
        )

    # Occupied pixels override free pixels when overlapping plane attempts disagree.
    visible_free = free_pixels & ~occupied_pixels
    visible_out_of_reach = out_of_reach_pixels & ~free_pixels & ~occupied_pixels
    visible_unsafe = unsafe_pixels & ~free_pixels & ~out_of_reach_pixels & ~occupied_pixels
    overlay = image.copy()
    overlay[visible_unsafe] = (210, 120, 30)
    overlay[visible_out_of_reach] = (190, 65, 210)
    overlay[visible_free] = (46, 204, 113)
    overlay[occupied_pixels] = (35, 35, 230)
    blended = cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0)
    if selected_only and selected_records:
        title = "07 Selected plane-local free_mask"
    elif selected_only:
        title = "07 No kept component - all free_mask attempts"
    else:
        title = "07 All plane-local free_mask attempts"
    canvas, panel_x = _stage_canvas(blended, title, len(debug_rows) + 6)
    for index, point_x, point_y in selected_points:
        cv2.drawMarker(canvas, (point_x, point_y), (46, 204, 113), cv2.MARKER_CROSS, 18, 2)
        cv2.circle(canvas, (point_x, point_y), 6, (46, 204, 113), 2)
        cv2.putText(canvas, f"G{index}", (point_x + 8, max(13, point_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 2, cv2.LINE_AA)
    legend = [
        ((46, 204, 113), "reachable free_mask samples"),
        ((190, 65, 210), "free but outside ready distance"),
        ((35, 35, 230), "occupancy_mask / dilation projection"),
        ((210, 120, 30), "edge-eroded unsafe surface"),
    ]
    row_y = 52
    for color, text in legend:
        cv2.rectangle(canvas, (panel_x + 14, row_y - 10), (panel_x + 25, row_y + 1), color, -1)
        cv2.putText(canvas, text, (panel_x + 32, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        row_y += 20
    if selected_only and selected_records:
        hidden_count = max(0, len(all_records) - len(records))
        cv2.putText(canvas, f"showing kept source masks; hidden attempts={hidden_count}", (panel_x + 14, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
        row_y += 20
    if not debug_rows:
        cv2.putText(canvas, "No grid free_mask generated", (panel_x + 14, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 80, 255), 1, cv2.LINE_AA)
    for text in debug_rows:
        cv2.putText(canvas, text, (panel_x + 14, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (230, 230, 230), 1, cv2.LINE_AA)
        row_y += 20
    return _write_stage_image(save_path, canvas)


def build_pointcloud_surface_output(
    *,
    points: JsonDict,
    parents: Iterable[JsonDict],
    blockers: Iterable[JsonDict],
    image_w: int,
    image_h: int,
    camera: Optional[JsonDict] = None,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
    debug_image_path: str = "",
    debug_plane_vis_path: str = "",
) -> JsonDict:
    parent_list = [item for item in parents if isinstance(item, dict)]
    blocker_list = [item for item in blockers if isinstance(item, dict)]
    regions: List[JsonDict] = []
    stats: List[JsonDict] = []
    initial_planes: List[JsonDict] = []
    split_clusters: List[JsonDict] = []
    geometry_stage_regions: List[JsonDict] = []
    occupancy_stage_regions: List[JsonDict] = []
    raised_top_stage_regions: List[JsonDict] = []
    completion_stage_regions: List[JsonDict] = []
    grid_mask_debug_records: List[JsonDict] = []
    skipped_blockers: List[JsonDict] = []
    grid_summary: JsonDict = {
        "mode": "plane_local_2d_grid",
        "grid_resolution_m": round(float(max(0.005, env_float("ROBOT_PC_FREE_GRID_RES_M", 0.02))), 4),
        "edge_margin_m": round(float(max(0.0, env_float("ROBOT_PC_FREE_GRID_EDGE_MARGIN_M", 0.05))), 4),
        "blocker_dilate_m": round(float(max(0.0, env_float("ROBOT_PC_FREE_GRID_BLOCKER_DILATE_M", 0.04))), 4),
        "ready_distance_range_m": [round(float(value), 4) for value in pointcloud_ready_distance_range_m()],
        "component_count": 0,
        "raw_component_count": 0,
        "component_kept": 0,
        "raw_free_cell_count": 0,
        "reachable_free_cell_count": 0,
        "reachability_filtered_cell_count": 0,
        "near_free_cell_count": 0,
        "far_free_cell_count": 0,
        "has_near_free_space": False,
        "has_far_free_space": False,
        "approachable_component_count": 0,
        "has_approachable_free_space": False,
        "raw_component_rejection_counts": {},
        "raw_component_rejections": [],
        "far_placeable_component_count": 0,
        "has_far_placeable_free_space": False,
        "far_component_rejection_counts": {},
        "far_component_rejections": [],
        "raw_free_distance_range_m": None,
        "reachable_free_distance_range_m": None,
        "component_rejection_counts": {},
        "component_rejections": [],
        "occupancy_mask_blocker_count": 0,
        "occupancy_source_counts": {"depth_points": 0, "bbox_fallback": 0},
        "hard_image_exclusion_box_count": 0,
        "mapped_point_hard_rejection_count": 0,
        "attempted_region_count": 0,
        "successful_region_count": 0,
        "candidate_trigger_counts": {},
        "attempt_result_counts": {},
        "status": "not_attempted",
        "suggested_recovery": None,
    }
    held_labels = normalize_label_tokens(held_object_labels)
    held_family = normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels)
    held_overlays, held_overlay_records = collect_held_overlay_candidates(
        blocker_list,
        holding_object=bool(holding_object),
        held_object_labels=held_labels,
        held_object_family=held_family,
    )
    filtered_points, held_point_filter_records, held_points_removed = remove_points_inside_held_overlays(
        points,
        held_overlays,
        image_w=image_w,
        image_h=image_h,
    )
    max_per_parent = max(1, env_int("ROBOT_PC_MAX_REGIONS_PER_PARENT", 5))
    for parent in parent_list:
        label = normalize_label(parent.get("label") or parent.get("raw_label"))
        if label in POINTCLOUD_LOW_PRIORITY_PARENTS:
            continue
        if label not in POINTCLOUD_PARENT_LABELS:
            continue
        cropped = crop_points_by_bbox(
            filtered_points,
            parent.get("bbox") if isinstance(parent.get("bbox"), dict) else {},
        )
        planes, plane_stats = extract_support_planes_open3d(cropped, parent, blocker_list)
        initial_planes.extend(planes)
        if held_points_removed:
            plane_stats["held_object_points_removed"] = int(held_points_removed)
        parent_regions: List[JsonDict] = []
        cluster_stats: List[JsonDict] = []
        for plane in planes:
            clusters = cluster_plane_regions_open3d(plane)
            split_clusters.extend(clusters)
            cluster_stats.append(
                {
                    "plane_id": plane.get("plane_id"),
                    "plane_inlier_count": plane.get("inlier_count"),
                    "dbscan_cluster_count": len(clusters),
                }
            )
            for cluster in clusters:
                region = build_region_from_cluster(
                    cluster,
                    parent=parent,
                    blockers=blocker_list,
                    scene_points=filtered_points,
                    image_w=image_w,
                    image_h=image_h,
                    holding_object=holding_object,
                    held_object_labels=held_labels,
                    held_object_family=held_family,
                    skipped_blockers=skipped_blockers,
                )
                if region:
                    parent_regions.append(region)
        geometry_stage_regions.extend(copy.deepcopy(parent_regions))
        occupancy_stage_regions.extend(copy.deepcopy(parent_regions))
        raised_top_count = mark_raised_object_top_regions(parent_regions)
        if raised_top_count:
            plane_stats["raised_object_top_filtered"] = int(raised_top_count)
        raised_top_stage_regions.extend(copy.deepcopy(parent_regions))
        completion_regions: List[JsonDict] = []
        for region in parent_regions:
            completion = build_completion_region_from_held_overlay(region)
            if completion:
                completion_regions.append(completion)
            region_grid_stats: JsonDict = {}
            grid_completions = build_free_space_completions_from_blocked_region_grid(
                region,
                scene_points=filtered_points,
                blockers=blocker_list,
                image_w=image_w,
                image_h=image_h,
                camera=camera,
                holding_object=holding_object,
                held_object_labels=held_labels,
                held_object_family=held_family,
                debug_stats=region_grid_stats,
            )
            if "_grid_debug_free_uv" in region_grid_stats:
                grid_mask_debug_records.append(region_grid_stats)
            if bool(region_grid_stats.get("attempted")):
                grid_summary["attempted_region_count"] += 1
                attempt_result = str(region_grid_stats.get("result") or region_grid_stats.get("reason") or "unknown")
                grid_summary["attempt_result_counts"][attempt_result] = (
                    int(grid_summary["attempt_result_counts"].get(attempt_result, 0) or 0) + 1
                )
                for trigger in region_grid_stats.get("candidate_triggers") or []:
                    trigger_name = str(trigger)
                    grid_summary["candidate_trigger_counts"][trigger_name] = (
                        int(grid_summary["candidate_trigger_counts"].get(trigger_name, 0) or 0) + 1
                    )
                if int(region_grid_stats.get("component_kept", 0) or 0) > 0:
                    grid_summary["successful_region_count"] += 1
                grid_summary["raw_component_count"] += int(region_grid_stats.get("raw_component_count", 0) or 0)
                grid_summary["component_count"] += int(region_grid_stats.get("component_count", 0) or 0)
                grid_summary["component_kept"] += int(region_grid_stats.get("component_kept", 0) or 0)
                for key in (
                    "raw_free_cell_count",
                    "reachable_free_cell_count",
                    "reachability_filtered_cell_count",
                    "near_free_cell_count",
                    "far_free_cell_count",
                ):
                    grid_summary[key] += int(region_grid_stats.get(key, 0) or 0)
                grid_summary["has_near_free_space"] = bool(
                    grid_summary["has_near_free_space"] or region_grid_stats.get("has_near_free_space")
                )
                grid_summary["has_far_free_space"] = bool(
                    grid_summary["has_far_free_space"] or region_grid_stats.get("has_far_free_space")
                )
                grid_summary["approachable_component_count"] += int(
                    region_grid_stats.get("approachable_component_count", 0) or 0
                )
                grid_summary["has_approachable_free_space"] = bool(
                    grid_summary["has_approachable_free_space"]
                    or region_grid_stats.get("has_approachable_free_space")
                )
                for reason, count in (region_grid_stats.get("raw_component_rejection_counts") or {}).items():
                    existing_count = int(grid_summary["raw_component_rejection_counts"].get(reason, 0) or 0)
                    grid_summary["raw_component_rejection_counts"][str(reason)] = existing_count + int(count or 0)
                remaining_raw_records = max(0, 20 - len(grid_summary["raw_component_rejections"]))
                if remaining_raw_records:
                    grid_summary["raw_component_rejections"].extend(
                        (region_grid_stats.get("raw_component_rejections") or [])[:remaining_raw_records]
                    )
                grid_summary["far_placeable_component_count"] += int(
                    region_grid_stats.get("far_placeable_component_count", 0) or 0
                )
                grid_summary["has_far_placeable_free_space"] = bool(
                    grid_summary["has_far_placeable_free_space"]
                    or region_grid_stats.get("has_far_placeable_free_space")
                )
                for reason, count in (region_grid_stats.get("far_component_rejection_counts") or {}).items():
                    existing_count = int(grid_summary["far_component_rejection_counts"].get(reason, 0) or 0)
                    grid_summary["far_component_rejection_counts"][str(reason)] = existing_count + int(count or 0)
                remaining_far_records = max(0, 20 - len(grid_summary["far_component_rejections"]))
                if remaining_far_records:
                    grid_summary["far_component_rejections"].extend(
                        (region_grid_stats.get("far_component_rejections") or [])[:remaining_far_records]
                    )
                for range_key in ("raw_free_distance_range_m", "reachable_free_distance_range_m"):
                    source_range = region_grid_stats.get(range_key)
                    if not isinstance(source_range, list) or len(source_range) != 2:
                        continue
                    current_range = grid_summary.get(range_key)
                    if isinstance(current_range, list) and len(current_range) == 2:
                        grid_summary[range_key] = [
                            round(min(float(current_range[0]), float(source_range[0])), 4),
                            round(max(float(current_range[1]), float(source_range[1])), 4),
                        ]
                    else:
                        grid_summary[range_key] = [
                            round(float(source_range[0]), 4),
                            round(float(source_range[1]), 4),
                        ]
                for reason, count in (region_grid_stats.get("component_rejection_counts") or {}).items():
                    existing_count = int(grid_summary["component_rejection_counts"].get(reason, 0) or 0)
                    grid_summary["component_rejection_counts"][str(reason)] = existing_count + int(count or 0)
                remaining_records = max(0, 20 - len(grid_summary["component_rejections"]))
                if remaining_records:
                    grid_summary["component_rejections"].extend(
                        (region_grid_stats.get("component_rejections") or [])[:remaining_records]
                    )
                grid_summary["occupancy_mask_blocker_count"] += int(region_grid_stats.get("occupancy_mask_blocker_count", 0) or 0)
                source_counts = region_grid_stats.get("occupancy_source_counts") if isinstance(region_grid_stats.get("occupancy_source_counts"), dict) else {}
                for source_name in ("depth_points", "bbox_fallback"):
                    grid_summary["occupancy_source_counts"][source_name] += int(source_counts.get(source_name, 0) or 0)
                for count_key in ("hard_image_exclusion_box_count", "mapped_point_hard_rejection_count"):
                    grid_summary[count_key] += int(region_grid_stats.get(count_key, 0) or 0)
                for contract_key in (
                    "edge_margin_m",
                    "blocker_dilate_m",
                    "configured_edge_margin_m",
                    "configured_blocker_dilate_m",
                    "held_footprint_radius_m",
                    "placement_clearance_m",
                ):
                    if contract_key in region_grid_stats:
                        grid_summary[contract_key] = region_grid_stats.get(contract_key)
            if grid_completions:
                completion_regions.extend(grid_completions)
        parent_regions.extend(completion_regions)
        completion_stage_regions.extend(copy.deepcopy(parent_regions))
        parent_regions.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        grid_ready_regions = [
            item
            for item in parent_regions
            if str(item.get("source") or "") == POINTCLOUD_GRID_COMPLETION_SOURCE
            and bool(item.get("visual_place_ready"))
        ]
        diagnostic_regions = [
            item
            for item in parent_regions
            if not (
                str(item.get("source") or "") == POINTCLOUD_GRID_COMPLETION_SOURCE
                and bool(item.get("visual_place_ready"))
            )
        ]
        retained_grid_limit = max(1, env_int("ROBOT_PC_FREE_GRID_MAX_COMPONENTS", 8))
        regions.extend(grid_ready_regions[:retained_grid_limit])
        regions.extend(diagnostic_regions[:max_per_parent])
        plane_stats["dbscan_clusters"] = cluster_stats
        plane_stats["surface_regions"] = len(parent_regions)
        stats.append(plane_stats)

    if int(grid_summary.get("component_kept", 0) or 0) > 0:
        grid_summary["status"] = "ready_components"
    elif int(grid_summary.get("attempted_region_count", 0) or 0) <= 0:
        grid_summary["status"] = "not_attempted"
        grid_summary["suggested_recovery"] = "change_viewpoint"
    elif bool(grid_summary.get("has_approachable_free_space")):
        grid_summary["status"] = "free_space_outside_current_reach"
        grid_summary["suggested_recovery"] = "reposition_for_reachable_surface"
    elif int(grid_summary.get("reachable_free_cell_count", 0) or 0) > 0:
        grid_summary["status"] = "no_grid_component_passed_safety_checks"
        grid_summary["suggested_recovery"] = "change_viewpoint_or_select_other_surface"
    else:
        grid_summary["status"] = "no_grid_free_mask"
        grid_summary["suggested_recovery"] = "change_viewpoint_or_select_other_surface"

    held_ignored_records = list(held_overlay_records)
    for item in skipped_blockers:
        if str(item.get("reason") or "") != "likely_held_object_overlay":
            continue
        key = (item.get("label"), item.get("raw_label"), item.get("reason"))
        if not any(
            (existing.get("label"), existing.get("raw_label"), existing.get("reason")) == key
            for existing in held_ignored_records
        ):
            held_ignored_records.append(item)
    initial_plane_vis_path = ""
    initial_plane_vis_error = ""
    stage_vis_paths: JsonDict = {}
    stage_vis_errors: JsonDict = {}
    if debug_plane_vis_path:
        try:
            initial_plane_vis_path = save_initial_plane_visualization(
                image_path=debug_image_path,
                save_path=debug_plane_vis_path,
                planes=initial_planes,
            )
        except Exception as exc:
            initial_plane_vis_error = str(exc)
        requested_stage_paths = derive_stage_visualization_paths(debug_plane_vis_path)
        stage_savers = {
            "split_regions": lambda path: save_split_region_visualization(
                image_path=debug_image_path,
                save_path=path,
                clusters=split_clusters,
            ),
            "geometry_filter": lambda path: save_region_stage_visualization(
                image_path=debug_image_path,
                save_path=path,
                regions=geometry_stage_regions,
                stage="geometry",
                title="03 Geometry filter",
            ),
            "occupancy_filter": lambda path: save_region_stage_visualization(
                image_path=debug_image_path,
                save_path=path,
                regions=occupancy_stage_regions,
                stage="occupancy",
                title="04 Occupancy filter",
            ),
            "raised_top_filter": lambda path: save_region_stage_visualization(
                image_path=debug_image_path,
                save_path=path,
                regions=raised_top_stage_regions,
                stage="raised_top",
                title="05 Raised-object-top filter",
            ),
            "completion_ready": lambda path: save_region_stage_visualization(
                image_path=debug_image_path,
                save_path=path,
                regions=completion_stage_regions,
                stage="completion_ready",
                title="06 Completion and ready output",
            ),
            "grid_free_mask": lambda path: save_grid_free_mask_visualization(
                image_path=debug_image_path,
                save_path=path,
                grid_records=grid_mask_debug_records,
                selected_only=True,
            ),
            "grid_free_mask_all_attempts": lambda path: save_grid_free_mask_visualization(
                image_path=debug_image_path,
                save_path=path,
                grid_records=grid_mask_debug_records,
            ),
        }
        for stage_name, stage_path in requested_stage_paths.items():
            try:
                stage_vis_paths[stage_name] = stage_savers[stage_name](stage_path)
            except Exception as exc:
                stage_vis_errors[stage_name] = str(exc)
    public_regions = [_strip_internal_surface_fields(region) for region in regions]
    ready_regions = [region for region in public_regions if region.get("visual_place_ready")]
    rejected_regions = [region for region in public_regions if not region.get("visual_place_ready")]
    return {
        "regions": public_regions,
        "ready_regions": ready_regions,
        "rejected_regions": rejected_regions,
        "best_region": select_best_pointcloud_surface(public_regions),
        "stats": stats,
        "region_count": len(regions),
        "ready_count": len(ready_regions),
        "holding_object": bool(holding_object),
        "held_object_labels": held_labels,
        "held_object_family": held_family,
        "held_object_point_filter": held_point_filter_records,
        "held_object_points_removed": int(held_points_removed),
        "skipped_blockers": skipped_blockers,
        "held_object_ignored_as_blocker": held_ignored_records,
        "ignored_held_blockers": [item.get("label") for item in held_ignored_records],
        "ignored_held_blocker_count": len(held_ignored_records),
        "initial_plane_count": len(initial_planes),
        "initial_plane_vis_path": initial_plane_vis_path,
        "initial_plane_vis_error": initial_plane_vis_error,
        "stage_vis_paths": stage_vis_paths,
        "stage_vis_errors": stage_vis_errors,
        "free_space_grid_summary": grid_summary,
    }
