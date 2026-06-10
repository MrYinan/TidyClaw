#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB-D floor-contact reasoning for small pickup targets.

The perception stack already has object-level RGB boxes and metric depth.  This
module adds a deliberately separate *contact* layer for floor-only pickup tasks:

1. fit a robust local floor plane from lower-image RGB-D samples;
2. inspect a narrow support strip immediately below a pickup box;
3. decide whether the target is resting on the floor rather than merely being
   visually enclosed by a large CounterTop / Cabinet box.

The implementation is metadata-hidden: it consumes only RGB-D camera geometry
and 2-D detections.  It intentionally does not query AI2-THOR object metadata.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


JsonDict = Dict[str, Any]
Projector = Callable[[float, float, float, JsonDict], JsonDict]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def depth_valid_mask(depth: Any) -> Any:
    return np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)


def depth_convention(camera: JsonDict) -> str:
    """Return the explicit depth convention used by the projector.

    Existing project deployments historically treated AI2-THOR depth as
    optical-axis Z.  Keep that as the compatibility default while making the
    convention explicit and allowing a calibrated ``ray_range`` mode.
    """

    value = str(
        camera.get("depth_convention")
        or os.getenv("ROBOT_DEPTH_CONVENTION", "optical_axis_z")
        or "optical_axis_z"
    ).strip().lower()
    aliases = {
        "z": "optical_axis_z",
        "axis_z": "optical_axis_z",
        "optical_z": "optical_axis_z",
        "optical_axis_z": "optical_axis_z",
        "range": "ray_range",
        "ray": "ray_range",
        "ray_range": "ray_range",
    }
    return aliases.get(value, "optical_axis_z")


def project_pixel_to_robot_3d(u: float, v: float, depth_m: float, camera: JsonDict) -> JsonDict:
    """Project one RGB-D sample into robot-relative metric coordinates.

    Returned axes match the rest of the project:
      x: right, y: height above floor, z: camera-forward depth,
      ground_forward_m: floor-parallel forward distance.
    """

    fx = max(1e-6, float(camera.get("fx", 1.0) or 1.0))
    fy = max(1e-6, float(camera.get("fy", fx) or fx))
    cx = float(camera.get("cx", 0.0) or 0.0)
    cy = float(camera.get("cy", 0.0) or 0.0)
    depth = float(depth_m)

    ray_x = (float(u) - cx) / fx
    ray_y_up = -(float(v) - cy) / fy
    ray_z = 1.0
    convention = depth_convention(camera)
    if convention == "ray_range":
        norm = max(1e-9, math.sqrt(ray_x * ray_x + ray_y_up * ray_y_up + ray_z * ray_z))
        scale = depth / norm
        x_cam = ray_x * scale
        y_cam_up = ray_y_up * scale
        z_cam = ray_z * scale
    else:
        z_cam = depth
        x_cam = ray_x * z_cam
        y_cam_up = ray_y_up * z_cam

    pitch = math.radians(float(camera.get("camera_horizon_deg", 0.0) or 0.0))
    camera_height = float(camera.get("camera_height_m", 0.9) or 0.9)
    height_from_floor = camera_height + y_cam_up * math.cos(pitch) - z_cam * math.sin(pitch)
    ground_forward_m = y_cam_up * math.sin(pitch) + z_cam * math.cos(pitch)
    ground_distance_m = math.hypot(x_cam, ground_forward_m)

    return {
        "x": round(float(x_cam), 4),
        "y": round(float(height_from_floor), 4),
        "z": round(float(z_cam), 4),
        "ground_forward_m": round(float(ground_forward_m), 4),
        "ground_distance_m": round(float(ground_distance_m), 4),
        "depth_convention": convention,
    }


def _project_samples(
    depth_frame: Any,
    camera: JsonDict,
    coords: Iterable[Tuple[int, int]],
    projector: Projector,
) -> List[JsonDict]:
    h, w = depth_frame.shape[:2]
    samples: List[JsonDict] = []
    for x, y in coords:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        value = _finite_number(depth_frame[y, x])
        if value is None or value <= 0.05 or value >= 20.0:
            continue
        point = projector(float(x), float(y), float(value), camera)
        px = _finite_number(point.get("x"))
        py = _finite_number(point.get("y"))
        pz = _finite_number(point.get("ground_forward_m"))
        if px is None or py is None or pz is None or pz <= 0.02:
            continue
        samples.append({"x": px, "y": py, "z": pz, "u": int(x), "v": int(y), "depth_m": value})
    return samples


def _fit_plane_y_from_xz(samples: Sequence[JsonDict]) -> Optional[JsonDict]:
    """Robustly fit y = ax + bz + c and return floor-plane diagnostics."""

    if len(samples) < 3:
        return None
    xyz = np.asarray([[float(p["x"]), float(p["z"]), float(p["y"])] for p in samples], dtype=np.float64)
    xz1 = np.column_stack([xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0], dtype=np.float64)])
    y = xyz[:, 2]
    mask = np.ones(y.shape[0], dtype=bool)
    coeff = None
    for _ in range(4):
        if int(mask.sum()) < 3:
            return None
        coeff, *_ = np.linalg.lstsq(xz1[mask], y[mask], rcond=None)
        residual = y - xz1.dot(coeff)
        active = np.abs(residual[mask])
        median = float(np.median(active)) if active.size else 0.0
        mad = float(np.median(np.abs(active - median))) if active.size else 0.0
        clip = max(_env_float("ROBOT_PICKUP_FLOOR_PLANE_MIN_CLIP_M", 0.025), median + 3.5 * max(mad, 1e-6))
        new_mask = np.abs(residual) <= clip
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    if coeff is None or int(mask.sum()) < 3:
        return None
    residual = y - xz1.dot(coeff)
    active_residual = np.abs(residual[mask])
    a, b, c = [float(v) for v in coeff]
    normal_up = 1.0 / math.sqrt(1.0 + a * a + b * b)
    return {
        "model": "y=ax+bz+c",
        "a": round(a, 7),
        "b": round(b, 7),
        "c": round(c, 7),
        "normal_up_score": round(normal_up, 6),
        "inlier_count": int(mask.sum()),
        "sample_count": int(len(samples)),
        "residual_p50_m": round(float(np.percentile(active_residual, 50)), 6),
        "residual_p90_m": round(float(np.percentile(active_residual, 90)), 6),
        "residual_max_m": round(float(np.max(active_residual)), 6),
    }


def plane_height_at(plane: JsonDict, x: float, z: float) -> float:
    return float(plane.get("a", 0.0) or 0.0) * float(x) + float(plane.get("b", 0.0) or 0.0) * float(z) + float(plane.get("c", 0.0) or 0.0)


def point_plane_residual(point: JsonDict, plane: JsonDict) -> Optional[float]:
    x = _finite_number(point.get("x"))
    y = _finite_number(point.get("y"))
    z = _finite_number(point.get("z"))
    if x is None or y is None or z is None:
        return None
    return float(y - plane_height_at(plane, x, z))


def estimate_local_floor_plane(
    depth_frame: Any,
    camera: JsonDict,
    *,
    projector: Projector = project_pixel_to_robot_3d,
) -> JsonDict:
    """Fit a local floor plane from lower-image RGB-D samples.

    Samples are selected geometrically rather than semantically.  A broad
    lower-image crop is projected to 3-D, then only points close to expected
    floor height are admitted to the robust fit.  This avoids depending on raw
    AI2-THOR metadata or any hard-coded object instance.
    """

    if depth_frame is None or getattr(depth_frame, "ndim", 0) != 2:
        return {"available": False, "reason": "missing_or_invalid_depth"}
    h, w = depth_frame.shape[:2]
    stride = max(2, _env_int("ROBOT_PICKUP_FLOOR_PLANE_SAMPLE_STRIDE", 8))
    y_start = max(0, min(h - 1, int(round(h * _env_float("ROBOT_PICKUP_FLOOR_PLANE_MIN_Y_RATIO", 0.54)))))
    y_end = max(y_start + 1, min(h, int(round(h * _env_float("ROBOT_PICKUP_FLOOR_PLANE_MAX_Y_RATIO", 0.98)))))
    x_margin = max(0, min(w // 3, int(round(w * _env_float("ROBOT_PICKUP_FLOOR_PLANE_X_MARGIN_RATIO", 0.04)))))
    coords = ((x, y) for y in range(y_start, y_end, stride) for x in range(x_margin, w - x_margin, stride))
    raw = _project_samples(depth_frame, camera, coords, projector)

    min_height = _env_float("ROBOT_PICKUP_FLOOR_PLANE_MIN_HEIGHT_M", -0.28)
    max_height = _env_float("ROBOT_PICKUP_FLOOR_PLANE_MAX_HEIGHT_M", 0.24)
    min_forward = _env_float("ROBOT_PICKUP_FLOOR_PLANE_MIN_FORWARD_M", 0.12)
    max_forward = _env_float("ROBOT_PICKUP_FLOOR_PLANE_MAX_FORWARD_M", 4.0)
    plausible = [
        point
        for point in raw
        if min_height <= float(point["y"]) <= max_height
        and min_forward <= float(point["z"]) <= max_forward
    ]
    min_points = max(8, _env_int("ROBOT_PICKUP_FLOOR_PLANE_MIN_POINTS", 24))
    if len(plausible) < min_points:
        return {
            "available": False,
            "reason": "insufficient_floor_samples",
            "raw_sample_count": len(raw),
            "plausible_sample_count": len(plausible),
            "depth_convention": depth_convention(camera),
        }
    plane = _fit_plane_y_from_xz(plausible)
    if not plane:
        return {
            "available": False,
            "reason": "floor_plane_fit_failed",
            "raw_sample_count": len(raw),
            "plausible_sample_count": len(plausible),
            "depth_convention": depth_convention(camera),
        }
    normal_score = _finite_number(plane.get("normal_up_score"))
    residual_p90 = _finite_number(plane.get("residual_p90_m"))
    normal_ok = bool(normal_score is not None and normal_score >= _env_float("ROBOT_PICKUP_FLOOR_PLANE_MIN_NORMAL_UP", 0.94))
    stable = bool(residual_p90 is not None and residual_p90 <= _env_float("ROBOT_PICKUP_FLOOR_PLANE_MAX_RESIDUAL_P90_M", 0.08))
    plane.update(
        {
            "available": bool(normal_ok and stable),
            "reason": "floor_plane_ready" if normal_ok and stable else "floor_plane_unstable",
            "raw_sample_count": len(raw),
            "plausible_sample_count": len(plausible),
            "normal_ok": bool(normal_ok),
            "stable": bool(stable),
            "depth_convention": depth_convention(camera),
        }
    )
    return plane


def _rect_coords(x1: int, y1: int, x2: int, y2: int, stride: int) -> Iterable[Tuple[int, int]]:
    for y in range(y1, y2, stride):
        for x in range(x1, x2, stride):
            yield x, y


def _residual_stats(points: Sequence[JsonDict], plane: JsonDict) -> JsonDict:
    residuals: List[float] = []
    signed: List[float] = []
    for point in points:
        value = point_plane_residual(point, plane)
        if value is None:
            continue
        signed.append(float(value))
        residuals.append(abs(float(value)))
    if not residuals:
        return {"sample_count": 0}
    arr = np.asarray(residuals, dtype=np.float64)
    signed_arr = np.asarray(signed, dtype=np.float64)
    tolerance = _env_float("ROBOT_PICKUP_FLOOR_CONTACT_MAX_RESIDUAL_M", 0.10)
    near = arr <= tolerance
    return {
        "sample_count": int(arr.size),
        "median_abs_residual_m": round(float(np.median(arr)), 6),
        "p90_abs_residual_m": round(float(np.percentile(arr, 90)), 6),
        "median_signed_residual_m": round(float(np.median(signed_arr)), 6),
        "near_floor_ratio": round(float(np.mean(near)), 6),
        "min_abs_residual_m": round(float(np.min(arr)), 6),
    }


def _representative_point(points: Sequence[JsonDict], plane: Optional[JsonDict]) -> Optional[JsonDict]:
    if not points:
        return None
    if plane:
        ranked = sorted(
            points,
            key=lambda p: abs(float(point_plane_residual(p, plane) or 999.0)),
        )
        point = ranked[0]
    else:
        point = points[len(points) // 2]
    return {key: point[key] for key in ("x", "y", "z", "u", "v", "depth_m") if key in point}


def analyze_pickup_floor_contact(
    depth_frame: Any,
    camera: JsonDict,
    bbox: Tuple[int, int, int, int],
    *,
    floor_plane: Optional[JsonDict] = None,
    projector: Projector = project_pixel_to_robot_3d,
) -> JsonDict:
    """Estimate whether a pickup box rests on the local floor plane.

    The strongest evidence comes from a narrow strip immediately *below* the
    detected object.  For a floor object this strip should be floor.  For a mug
    or tomato on a countertop this strip should instead be near countertop
    height, preventing a giant 2-D support bbox from corrupting floor pickup.
    """

    if depth_frame is None or getattr(depth_frame, "ndim", 0) != 2:
        return {"available": False, "reason": "missing_or_invalid_depth"}
    h, w = depth_frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    plane = floor_plane if isinstance(floor_plane, dict) and floor_plane.get("available") else estimate_local_floor_plane(depth_frame, camera, projector=projector)
    if not isinstance(plane, dict) or not plane.get("available"):
        return {
            "available": False,
            "reason": "local_floor_plane_unavailable",
            "floor_plane": plane if isinstance(plane, dict) else {},
        }

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    center_x = (x1 + x2) // 2
    half_width = max(2, int(round(box_w * _env_float("ROBOT_PICKUP_FLOOR_CONTACT_CENTER_WIDTH_RATIO", 0.72) * 0.5)))
    support_pad = max(4, int(round(box_h * _env_float("ROBOT_PICKUP_FLOOR_CONTACT_SUPPORT_PAD_RATIO", 0.45))))
    inside_height = max(3, int(round(box_h * _env_float("ROBOT_PICKUP_FLOOR_CONTACT_INSIDE_HEIGHT_RATIO", 0.30))))
    stride = max(1, _env_int("ROBOT_PICKUP_FLOOR_CONTACT_SAMPLE_STRIDE", 2))

    sx1 = max(0, center_x - half_width)
    sx2 = min(w, center_x + half_width + 1)
    support_y1 = min(h, y2)
    support_y2 = min(h, y2 + support_pad)
    inside_y1 = max(0, y2 - inside_height)
    inside_y2 = min(h, y2)

    support_points = _project_samples(depth_frame, camera, _rect_coords(sx1, support_y1, sx2, support_y2, stride), projector)
    inside_points = _project_samples(depth_frame, camera, _rect_coords(sx1, inside_y1, sx2, inside_y2, stride), projector)
    support_stats = _residual_stats(support_points, plane)
    inside_stats = _residual_stats(inside_points, plane)

    min_support_samples = max(2, _env_int("ROBOT_PICKUP_FLOOR_CONTACT_MIN_SUPPORT_SAMPLES", 4))
    min_floor_ratio = _env_float("ROBOT_PICKUP_FLOOR_CONTACT_MIN_NEAR_FLOOR_RATIO", 0.55)
    max_median = _env_float("ROBOT_PICKUP_FLOOR_CONTACT_MAX_MEDIAN_RESIDUAL_M", 0.08)
    support_count = int(support_stats.get("sample_count", 0) or 0)
    support_ratio_value = _finite_number(support_stats.get("near_floor_ratio"))
    support_median_value = _finite_number(support_stats.get("median_abs_residual_m"))
    support_ratio = float(support_ratio_value if support_ratio_value is not None else 0.0)
    support_median = float(support_median_value if support_median_value is not None else 999.0)
    contact_floor_like = bool(
        support_count >= min_support_samples
        and support_ratio >= min_floor_ratio
        and support_median <= max_median
    )

    # If the support strip is clipped by the image boundary, allow a stricter
    # in-box bottom-strip fallback rather than silently dropping depth evidence.
    if not contact_floor_like and support_count < min_support_samples:
        inside_count = int(inside_stats.get("sample_count", 0) or 0)
        inside_ratio_value = _finite_number(inside_stats.get("near_floor_ratio"))
        inside_median_value = _finite_number(inside_stats.get("median_abs_residual_m"))
        inside_ratio = float(inside_ratio_value if inside_ratio_value is not None else 0.0)
        inside_median = float(inside_median_value if inside_median_value is not None else 999.0)
        contact_floor_like = bool(
            inside_count >= min_support_samples
            and inside_ratio >= max(min_floor_ratio, 0.70)
            and inside_median <= min(max_median, 0.06)
        )

    confidence = 0.0
    if support_count > 0:
        confidence = max(0.0, min(1.0, support_ratio * max(0.0, 1.0 - support_median / max(max_median, 1e-6))))

    support_height = None
    support_rep = _representative_point(support_points, plane)
    if support_rep:
        heights = [_finite_number(point.get("y")) for point in support_points]
        finite_heights = [value for value in heights if value is not None]
        if finite_heights:
            support_height = float(np.median(np.asarray(finite_heights, dtype=np.float64)))

    return {
        "available": True,
        "reason": "rgbd_floor_contact" if contact_floor_like else "support_strip_not_on_floor",
        "method": "local_floor_plane_plus_bottom_support_strip_v1",
        "contact_floor_like": bool(contact_floor_like),
        "confidence": round(float(confidence), 6),
        "floor_plane": dict(plane),
        "support_strip_bbox": {"x": sx1, "y": support_y1, "w": max(0, sx2 - sx1), "h": max(0, support_y2 - support_y1)},
        "inside_bottom_bbox": {"x": sx1, "y": inside_y1, "w": max(0, sx2 - sx1), "h": max(0, inside_y2 - inside_y1)},
        "support_strip": support_stats,
        "inside_bottom_strip": inside_stats,
        "support_contact_3d": support_rep,
        "bottom_contact_3d": _representative_point(inside_points, plane),
        "support_height_m": round(float(support_height), 6) if support_height is not None else None,
        "floor_plane_residual_m": support_stats.get("median_abs_residual_m"),
        "depth_convention": depth_convention(camera),
    }
