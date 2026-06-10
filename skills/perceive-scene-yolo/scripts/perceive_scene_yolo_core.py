#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO perception with optional depth geometry for V2 household service tasks.

This script is the online perception adapter used by patrol_runner.py.
It keeps the online contract strict: the runner receives RGB-derived YOLO
candidates, optional depth-derived surface candidates, and sanitized action
hints only. It does not read AI2-THOR object metadata.

Main V2.2 changes compared with the previous version:
- Distinguishes pickup targets, receptacles and obstacles during geometry parsing.
- Stops calling every large CounterTop/Sink box "floor".
- Adds best_pickup_candidate / best_receptacle_candidate for stable runner decisions.
- Disables raw YOLO receptacle place_now; depth surface candidates own placement readiness.
- Keeps legacy clean fields so clean mode remains compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pickup_floor_geometry import (
    analyze_pickup_floor_contact,
    depth_convention as pickup_depth_convention,
    estimate_local_floor_plane,
    project_pixel_to_robot_3d,
)


JsonDict = Dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO_ROOT / "skills" / "perceive-scene-yolo" / "weights" / "best.pt"
DEFAULT_ONTOLOGY = REPO_ROOT / "configs" / "service_task_ontology_v2.json"
DEFAULT_FALLBACK_WEIGHTS = os.getenv("ROBOT_YOLO_FALLBACK_WEIGHTS", "yolo11n.pt")

SERVICE_CLASSES = {"pickup_target", "place_receptacle", "cleanable_object"}

#定义“什么东西属于什么任务类别”，把 label 映射成任务语义类别
DEFAULT_TASK_CLASS_MAP = {
    "book": "pickup_target",
    "apple": "pickup_target",
    "tomato": "pickup_target",
    "potato": "pickup_target",
    "mug": "pickup_target",
    "cup": "pickup_target",
    "bowl": "pickup_target",
    "plate": "pickup_target",
    "remote": "pickup_target",
    "remotecontrol": "pickup_target",
    "remote_control": "pickup_target",
    "banana": "pickup_target",
    "orange": "pickup_target",

    "diningtable": "place_receptacle",
    "dining_table": "place_receptacle",
    "dining table": "place_receptacle",
    "coffeetable": "place_receptacle",
    "coffee_table": "place_receptacle",
    "sidetable": "place_receptacle",
    "side_table": "place_receptacle",
    "countertop": "place_receptacle",
    "counter_top": "place_receptacle",
    "sink": "place_receptacle",

    "chair": "obstacle",
    "armchair": "obstacle",
    "arm_chair": "obstacle",
    "sofa": "obstacle",
    "couch": "obstacle",
    "cabinet": "obstacle",
    "drawer": "obstacle",
    "dishwasher": "obstacle",
    "microwave": "obstacle",
    "stove": "obstacle",
    "shelf": "obstacle",
    "shelvingunit": "obstacle",
    "shelving_unit": "obstacle",
    "bed": "obstacle",

    "bottle": "ignored_object",
    "soapbottle": "ignored_object",
    "soap_bottle": "ignored_object",
    "vase": "ignored_object",
    "houseplant": "ignored_object",
    "house_plant": "ignored_object",
    "kettle": "ignored_object",
    "lettuce": "ignored_object",
    "pan": "ignored_object",
    "pot": "ignored_object",
    "butterknife": "ignored_object",
    "butter_knife": "ignored_object",
    "spatula": "ignored_object",
    "peppershaker": "ignored_object",
    "pepper_shaker": "ignored_object",
    "saltshaker": "ignored_object",
    "salt_shaker": "ignored_object",
    "papertowelroll": "ignored_object",
    "paper_towel_roll": "ignored_object",

    "cleanable_floor_trash": "cleanable_object",
    "floor_trash": "cleanable_object",
    "trash": "cleanable_object",
    "garbage": "cleanable_object",
    "red_small_object": "cleanable_object",
    "yellow_small_object": "cleanable_object",
}


# Some raw model labels are normalized for cleaner JSON outputs.
# 显示标签别名映射表  aliases:别名
DISPLAY_LABEL_ALIASES = {
    "countertop": "counter_top",
    "diningtable": "dining_table",
    "coffeetable": "coffee_table",
    "sidetable": "side_table",
    "remotecontrol": "remote_control",
    "shelvingunit": "shelving_unit",
    "soapbottle": "soap_bottle",
    "houseplant": "house_plant",
    "butterknife": "butter_knife",
    "peppershaker": "pepper_shaker",
    "saltshaker": "salt_shaker",
    "papertowelroll": "paper_towel_roll",
}

#support context   可以被当作 放置目标的物体
SUPPORT_CONTEXT_LABELS = {
    "counter_top",
    "countertop",
    "dining_table",#餐厅卓
    "diningtable",
    "coffee_table",
    "coffeetable",
    "side_table",
    "sidetable",
    "cabinet",
    "drawer",
    "dishwasher",
    "microwave",
    "stove",
    "shelf",
    "shelving_unit",
    "shelvingunit",
}

SURFACE_PARENT_LABELS = {
    "counter_top",
    "countertop",
    "dining_table",
    "diningtable",
    "coffee_table",
    "coffeetable",
    "side_table",
    "sidetable",
}

SURFACE_BLOCKING_LABELS = {
    "sink",
    "stove",
    "microwave",
    "dishwasher",
    "cabinet",
    "drawer",
}

POINTCLOUD_SURFACE_SOURCE = "pointcloud_plane"
POINTCLOUD_COMPLETION_SURFACE_SOURCE = "pointcloud_plane_completion"
POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE = "pointcloud_plane_grid_completion"
DEPTH_REGION_SURFACE_SOURCE = "depth_region_geometry"
LEGACY_DEPTH_SURFACE_SOURCE = "depth_geometry"
SURFACE_REGION_SOURCES = {
    POINTCLOUD_SURFACE_SOURCE,
    POINTCLOUD_COMPLETION_SURFACE_SOURCE,
    POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE,
    DEPTH_REGION_SURFACE_SOURCE,
}
SURFACE_CANDIDATE_SOURCES = {
    POINTCLOUD_SURFACE_SOURCE,
    POINTCLOUD_COMPLETION_SURFACE_SOURCE,
    POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE,
    DEPTH_REGION_SURFACE_SOURCE,
    LEGACY_DEPTH_SURFACE_SOURCE,
}

HELD_FOOD_LABELS = {"apple", "banana", "lettuce", "orange", "potato", "tomato"}
HELD_PICKUP_LABELS = HELD_FOOD_LABELS | {
    "book",
    "bottle",
    "bowl",
    "cup",
    "kettle",
    "mug",
    "pan",
    "plate",
    "pot",
    "remote",
    "remote_control",
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


def json_print(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO
    except Exception as exc:
        json_print(
            {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_yolo_dependency_missing",
                "message": "Cannot import ultralytics. Install with: pip install ultralytics",
                "detail": str(exc),
                "online_safe": True,
            }
        )
        sys.exit(1)


def normalize_label(label: Any) -> str:
    value = str(label).strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def display_label(raw_label: str) -> str:
    normalized = normalize_label(raw_label)
    return DISPLAY_LABEL_ALIASES.get(normalized.replace("_", ""), normalized)


def load_task_class_map(path: Path) -> Dict[str, str]:
    mapping = dict(DEFAULT_TASK_CLASS_MAP)
    if not path.exists():
        return mapping
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return mapping
    raw_map = data.get("task_class_map") if isinstance(data, dict) else None
    if not isinstance(raw_map, dict):
        return mapping
    for key, value in raw_map.items():
        mapping[normalize_label(key)] = str(value)
    return mapping


def resolve_weights(requested: str, fallback: str) -> Tuple[str, List[str]]:
    notes: List[str] = []
    requested_path = Path(requested)
    if requested and requested_path.exists():
        return str(requested_path), notes
    if requested and str(requested_path) != str(DEFAULT_WEIGHTS):
        notes.append(f"requested_weights_not_found:{requested}")
    notes.append("using_fallback_weights_before_custom_best_pt")
    return fallback, notes


def infer_position_hint(cx_ratio: float) -> str:
    if cx_ratio < 0.33:
        return "front-left"
    if cx_ratio > 0.66:
        return "front-right"
    return "front-center"


def infer_position_hint_from_bearing(bearing_deg: float, *, center_tolerance_deg: Optional[float] = None) -> str:
    tolerance = float(center_tolerance_deg if center_tolerance_deg is not None else env_float("ROBOT_DEPTH_CENTER_TOLERANCE_DEG", 8.0))
    if bearing_deg < -tolerance:
        return "front-left"
    if bearing_deg > tolerance:
        return "front-right"
    return "front-center"

#support_surface_hint：支撑面提示
def support_surface_hint(task_semantic_class: str, is_floor_level: bool) -> str:
    if task_semantic_class == "place_receptacle":
        return "support_surface"
    if task_semantic_class == "obstacle":
        return "structural_obstacle"
    if task_semantic_class == "pickup_target":
        return "floor" if is_floor_level else "surface_or_elevated"
    if task_semantic_class == "cleanable_object":
        return "floor" if is_floor_level else "non_floor"
    return "unknown"

#负责把 YOLO 的一个检测框变成一个结构化候选对象
#raw_label：原始标签
"""
YOLO 原始输出大概是：

{
  "label": "apple",
  "confidence": 0.83,
  "bbox": [x1, y1, x2, y2]
}
但你的机器人不能只知道这些。它还要知道：
这个苹果在左边、中间还是右边？
离不离我近？
是不是在地面？
需不需要转向？
需不需要靠近？
现在能不能直接捡？

所以 build_candidate() 先根据检测框计算宽、高、面积、中心点、中心点比例、底部 y 比例、面积比例、
宽度比例等几何信息。然后根据 cx_ratio 判断位置是 front-left、front-center 还是 front-right；
根据 bottom_y_ratio 判断是否接近地面；根据面积和底部位置判断是否“视觉上比较近”。"""
def build_candidate(
    *,
    raw_label: str,
    task_semantic_class: str,
    confidence: float,
    xyxy: Iterable[float],
    image_w: int,
    image_h: int,
    floor_bottom_ratio: float,
    near_area_ratio: float,
    place_max_area_ratio: float,
    place_max_width_ratio: float,
    pickup_min_conf: float,
    small_floor_pickup_min_conf: float,
) -> JsonDict:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(x1, float(image_w)))
    x2 = max(0.0, min(x2, float(image_w)))
    y1 = max(0.0, min(y1, float(image_h)))
    y2 = max(0.0, min(y2, float(image_h)))

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0

    cx_ratio = cx / max(image_w, 1)#检测框中心点在图像宽度上的比例，判断目标是局左/中/右
    center_y_ratio = cy / max(image_h, 1)
    bottom_y_ratio = y2 / max(image_h, 1)#检测框底部在图像高度上的比例
    area_ratio = area / max(image_w * image_h, 1)#检测框面积占整张图面积的比例
    width_ratio = width / max(image_w, 1)
    position_hint = infer_position_hint(cx_ratio)#根据cx_ratio比例，判断目标是局左/中/右
    center_offset = cx_ratio - 0.5

    is_floor_level = bottom_y_ratio >= floor_bottom_ratio#视觉上像不像在地面
    surface_hint = support_surface_hint(task_semantic_class, is_floor_level)
    visually_near = bool(area_ratio >= near_area_ratio or bottom_y_ratio >= 0.58)#接近的条件：1. 目标框面积够大2. 目标框底部足够靠下
    service_reachable = bool(visually_near and position_hint in {"front-left", "front-center", "front-right"})
    try:
        small_floor_min_area = float(os.getenv("ROBOT_PICKUP_SMALL_FLOOR_MIN_AREA", "0.0007"))
    except (TypeError, ValueError):
        small_floor_min_area = 0.0007
    try:
        small_floor_bottom_ratio = float(os.getenv("ROBOT_PICKUP_SMALL_FLOOR_BOTTOM_RATIO", "0.88"))
    except (TypeError, ValueError):
        small_floor_bottom_ratio = 0.88

    # Pickup objects can be on the floor or on a support surface. For the
    # online policy, "pickup_now" only means "visually centered and near"; the
    # backend still validates the real PickupObject feasibility.
    small_floor_pickup_ready = bool(
        task_semantic_class == "pickup_target"
        and confidence >= small_floor_pickup_min_conf
        and surface_hint == "floor"
        and service_reachable
        and position_hint == "front-center"
        and abs(center_offset) <= 0.12
        and bottom_y_ratio >= small_floor_bottom_ratio
        and area_ratio >= small_floor_min_area
    )
    #拾取条件
# 1. 这个目标必须是 pickup_target
# 2. 检测置信度要够
# 3. 视觉上可达
# 4. 必须在画面正中央
# 5. 要么面积够大，要么满足小型地面物体拾取条件
    pickup_now = bool(
        task_semantic_class == "pickup_target"
        and confidence >= pickup_min_conf
        and service_reachable
        and position_hint == "front-center"
        and (area_ratio >= near_area_ratio or small_floor_pickup_ready)
    )

    # Raw YOLO receptacle boxes are semantic context only. A whole CounterTop
    # bbox may cross floor gaps, sinks, or stove regions, so placement readiness
    # is produced only by depth-derived placeable_surface_region candidates.
    place_centered = bool(abs(center_offset) <= 0.10)
    front_edge_receptacle = False
    broad_front_receptacle = False
    receptacle_box_ambiguous = bool(
        task_semantic_class == "place_receptacle"
        and (area_ratio > place_max_area_ratio or width_ratio > place_max_width_ratio)
    )
    place_now = False

    cleanable_now = bool(
        task_semantic_class == "cleanable_object"
        and is_floor_level
        and service_reachable
        and position_hint == "front-center"
    )
#是否需要先转向对齐
    needs_alignment = False
    needs_approach = False#是否需要先往前靠近
    if task_semantic_class in {"pickup_target", "cleanable_object"}:
        needs_alignment = bool(service_reachable and position_hint in {"front-left", "front-right"})
        needs_approach = bool(service_reachable and position_hint == "front-center" and not (pickup_now or cleanable_now))
    elif task_semantic_class == "place_receptacle":
        needs_alignment = bool(
            service_reachable
            and not receptacle_box_ambiguous
            and (
                position_hint in {"front-left", "front-right"}
                or (position_hint == "front-center" and not place_centered)
            )
        )
        needs_approach = bool(
            service_reachable
            and position_hint == "front-center"
            and not receptacle_box_ambiguous
            and not place_now
            and not needs_alignment
        )

    obstacle_risk = bool(
        task_semantic_class == "obstacle"
        and bottom_y_ratio >= 0.55
        and area_ratio >= 0.012
    )
#返回的是真正会看的结构化候选物。
# label                     展示用标签
# raw_label                 原始检测标签
# task_semantic_class       任务类别
# confidence                置信度
# bbox                      检测框
# center                    中心点
# position_hint             前左 / 前中 / 前右
# surface_hint              地面 / 支撑面
# is_floor_level            是否地面目标
# reachable                 是否视觉可达
# pickup_now                是否现在可拾取
# place_now                 是否现在可放置
# cleanable_now             是否现在可清理
# needs_alignment           是否需要先对齐
# needs_approach            是否需要靠近
# obstacle_risk             是否有障碍物风险
# visual_box_ambiguous      检测框是否模糊
# area_ratio                面积比例
# bottom_y_ratio            底部位置比例
#geometry                  更详细的几何信息
    return {
        "label": display_label(raw_label),
        "raw_label": raw_label,
        "task_semantic_class": task_semantic_class,
        "confidence": round(float(confidence), 4),
        "bbox": {"x": int(round(x1)), "y": int(round(y1)), "w": int(round(width)), "h": int(round(height))},
        "image_size": {"w": int(image_w), "h": int(image_h)},
        "center": {"x": int(round(cx)), "y": int(round(cy))},
        "position_hint": position_hint,
        "surface_hint": surface_hint,
        "is_floor_level": bool(is_floor_level),
        "is_support_surface": bool(task_semantic_class == "place_receptacle"),
        "reachable": bool(service_reachable),
        "pickup_now": bool(pickup_now),
        "place_now": bool(place_now),
        "cleanable_now": bool(cleanable_now),
        "needs_alignment": bool(needs_alignment),
        "needs_approach": bool(needs_approach),
        "obstacle_risk": bool(obstacle_risk),
        "visual_box_ambiguous": bool(receptacle_box_ambiguous),
        "broad_front_receptacle": bool(broad_front_receptacle),
        "front_edge_receptacle": bool(front_edge_receptacle),
        "area": round(float(area), 2),
        "area_ratio": round(float(area_ratio), 6),
        "center_y_ratio": round(float(center_y_ratio), 3),
        "bottom_y_ratio": round(float(bottom_y_ratio), 3),
        "geometry": {
            "cx_ratio": round(cx_ratio, 4),
            "cy_ratio": round(center_y_ratio, 4),
            "bottom_y_ratio": round(bottom_y_ratio, 4),
            "area_ratio": round(area_ratio, 6),
            "width_ratio": round(width_ratio, 6),
        },
    }

#解决视野里多个目标时选谁
"""candidate_sort_score() 就是给候选目标打分。它会综合：

置信度
面积
是否居中
是否已经 pickup_now / place_now
是否在地面
是否需要靠近
是否需要对齐
语义类别是否符合当前意图"""
def candidate_sort_score(candidate: JsonDict, *, intent: str = "service") -> float:
    conf = float(candidate.get("confidence", 0.0) or 0.0)
    area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
    cx_ratio = float((candidate.get("geometry") or {}).get("cx_ratio", 0.5) or 0.5)
    center_bonus = max(0.0, 0.5 - abs(cx_ratio - 0.5))
    score = conf * 2.0 + min(area_ratio, 0.20) + center_bonus
    if candidate.get("context_only"):
        score -= 8.0
    surface_source = str(candidate.get("surface_candidate_source") or "")
    if surface_source in SURFACE_CANDIDATE_SOURCES:
        score += float(candidate.get("score", 0.0) or 0.0) * 2.0
    if surface_source in SURFACE_REGION_SOURCES:
        if candidate.get("visual_place_ready"):
            score += 2.0
        if surface_source in {POINTCLOUD_SURFACE_SOURCE, POINTCLOUD_COMPLETION_SURFACE_SOURCE, POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE} and candidate.get("affordance_ready"):
            score += 1.0
        if candidate.get("rejection_reasons"):
            score -= 2.0
        if candidate.get("blocked"):
            score -= 4.0

    if intent == "pickup":
        if candidate.get("pickup_now"):
            score += 4.0
        if candidate.get("is_floor_level"):
            score += 1.2
        if candidate.get("position_hint") == "front-center":
            score += 0.8
        if candidate.get("needs_approach"):
            score += 0.5
        if candidate.get("needs_alignment"):
            score += 0.25
        if candidate.get("task_semantic_class") != "pickup_target":
            score -= 10.0
    elif intent == "receptacle":
        if candidate.get("place_now"):
            score += 4.0
        if not candidate.get("visual_place_ready"):
            if candidate.get("needs_alignment"):
                score += 0.8
            if candidate.get("needs_approach"):
                score += 0.4
        if candidate.get("task_semantic_class") != "place_receptacle":
            score -= 10.0
    elif intent == "obstacle":
        if candidate.get("obstacle_risk"):
            score += 3.0
        if candidate.get("task_semantic_class") != "obstacle":
            score -= 10.0
    else:
        if candidate.get("pickup_now") or candidate.get("place_now") or candidate.get("cleanable_now"):
            score += 2.0
        if candidate.get("needs_alignment"):
            score += 0.4
    return score


def sorted_candidates(candidates: Iterable[JsonDict], *, intent: str, max_items: Optional[int] = None) -> List[JsonDict]:
    items = sorted(
        [c for c in candidates if isinstance(c, dict)],
        key=lambda c: candidate_sort_score(c, intent=intent),
        reverse=True,
    )
    if max_items is None:
        return items
    return items[:max_items]


def best_candidate(candidates: Iterable[JsonDict], *, intent: str) -> Optional[JsonDict]:
    items = sorted_candidates(candidates, intent=intent, max_items=1)
    return items[0] if items else None


def bbox_ratios(candidate: JsonDict) -> Tuple[float, float, float, float]:
    bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
    image_size = candidate.get("image_size") if isinstance(candidate.get("image_size"), dict) else {}
    image_w = max(1.0, float(image_size.get("w", 600.0) or 600.0))
    image_h = max(1.0, float(image_size.get("h", 600.0) or 600.0))
    x = float(bbox.get("x", 0.0) or 0.0) / image_w
    y = float(bbox.get("y", 0.0) or 0.0) / image_h
    w = float(bbox.get("w", 0.0) or 0.0) / image_w
    h = float(bbox.get("h", 0.0) or 0.0) / image_h
    return x, y, x + w, y + h


def bbox_pixel_tuple(candidate: JsonDict) -> Optional[Tuple[float, float, float, float]]:
    bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
    try:
        x = float(bbox.get("x"))
        y = float(bbox.get("y"))
        w = float(bbox.get("w"))
        h = float(bbox.get("h"))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def load_depth_frame(depth_path: str, *, image_w: int, image_h: int, notes: List[str]) -> Optional[Any]:
    if not depth_path:
        notes.append("depth_geometry=disabled:no_depth_path")
        return None
    path = Path(depth_path)
    if not path.exists():
        notes.append(f"depth_geometry=disabled:depth_not_found:{depth_path}")
        return None
    try:
        import numpy as np  # type: ignore

        suffix = path.suffix.lower()
        if suffix == ".npy":
            depth = np.load(str(path)).astype("float32", copy=False)
        elif suffix == ".npz":
            data = np.load(str(path))
            key = "depth" if "depth" in data.files else data.files[0]
            depth = data[key].astype("float32", copy=False)
        else:
            from PIL import Image  # type: ignore

            raw = np.asarray(Image.open(path))
            depth = raw.astype("float32")
            if raw.dtype.kind in {"u", "i"}:
                depth = depth / 1000.0

        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.ndim != 2 or depth.size == 0:
            notes.append(f"depth_geometry=disabled:invalid_depth_shape:{getattr(depth, 'shape', None)}")
            return None

        if depth.shape != (int(image_h), int(image_w)):
            try:
                import cv2  # type: ignore

                depth = cv2.resize(depth, (int(image_w), int(image_h)), interpolation=cv2.INTER_NEAREST)
                notes.append(f"depth_geometry=resized:{depth.shape[1]}x{depth.shape[0]}")
            except Exception as exc:
                notes.append(f"depth_geometry=disabled:shape_mismatch:{depth.shape}:{exc}")
                return None
        notes.append(f"depth_geometry=enabled:{path}")
        return depth
    except Exception as exc:
        notes.append(f"depth_geometry=disabled:load_failed:{exc}")
        return None


def parse_camera_info(camera_info: Any, *, image_w: int, image_h: int) -> JsonDict:
    parse_error = ""
    if isinstance(camera_info, str) and camera_info.strip():
        try:
            loaded = json.loads(camera_info)
            camera_info = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            parse_error = "json_decode_error"
            camera_info = {}
    if not isinstance(camera_info, dict):
        parse_error = parse_error or "not_dict"
        camera_info = {}

    try:
        camera_fov_default = float(camera_info.get("fov_deg", 90.0) or 90.0)
    except (TypeError, ValueError):
        camera_fov_default = 90.0
    fov_deg = env_float("ROBOT_CAMERA_FOV_DEG", camera_fov_default)
    fov_rad = math.radians(max(1.0, min(179.0, fov_deg)))
    fallback_focal = float(image_w) / (2.0 * math.tan(fov_rad / 2.0))

    def number(key: str, default: float) -> float:
        try:
            return float(camera_info.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    parsed = {
        "width": int(number("width", image_w)),
        "height": int(number("height", image_h)),
        "fx": number("fx", fallback_focal),
        "fy": number("fy", fallback_focal),
        "cx": number("cx", float(image_w) / 2.0),
        "cy": number("cy", float(image_h) / 2.0),
        "fov_deg": fov_deg,
        "camera_horizon_deg": number("camera_horizon_deg", number("cameraHorizon", 0.0)),
        "camera_height_m": number("camera_height_m", env_float("ROBOT_CAMERA_HEIGHT_M", 0.9)),
        # Keep the historical projection mode as the compatibility default, but
        # make the convention explicit so it can be calibrated without silently
        # changing every point-cloud threshold.
        "depth_convention": str(
            camera_info.get("depth_convention")
            or os.getenv("ROBOT_DEPTH_CONVENTION", "optical_axis_z")
            or "optical_axis_z"
        ).strip().lower(),
    }
    if parse_error:
        parsed["_parse_error"] = parse_error
    return parsed


def depth_valid_mask(depth: Any) -> Any:
    import numpy as np  # type: ignore

    return np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)


def project_pixel_to_3d(u: float, v: float, depth_m: float, camera: JsonDict) -> JsonDict:
    """Project one RGB-D sample using an explicit, calibratable convention.

    The implementation lives in ``pickup_floor_geometry`` so floor-contact
    classification, point-cloud surfaces, and legacy depth candidates share the
    same metric transform.
    """

    return project_pixel_to_robot_3d(u, v, depth_m, camera)


def candidate_depth_summary(
    candidate: JsonDict,
    depth_frame: Any,
    camera: JsonDict,
    *,
    floor_plane: Optional[JsonDict] = None,
) -> Optional[JsonDict]:
    import numpy as np  # type: ignore

    box = bbox_pixel_tuple(candidate)
    if box is None:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = depth_frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = depth_frame[y1:y2, x1:x2]
    valid = depth_valid_mask(crop)
    valid_count = int(valid.sum())
    total = int(crop.size)
    if total <= 0 or valid_count <= 0:
        return {"available": False, "valid_ratio": 0.0}

    values = crop[valid]
    center = candidate.get("center") if isinstance(candidate.get("center"), dict) else {}
    try:
        cx = int(round(float(center.get("x"))))
        cy = int(round(float(center.get("y"))))
        center_depth = float(depth_frame[max(0, min(cy, h - 1)), max(0, min(cx, w - 1))])
    except (TypeError, ValueError):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_depth = float("nan")

    median_depth = float(np.median(values))
    point_3d = project_pixel_to_3d(cx, cy, median_depth, camera)
    result: JsonDict = {
        "available": True,
        "unit": "meter",
        "valid_ratio": round(valid_count / max(1, total), 4),
        "median_m": round(median_depth, 4),
        "min_m": round(float(np.min(values)), 4),
        "max_m": round(float(np.max(values)), 4),
        "p10_m": round(float(np.percentile(values, 10)), 4),
        "p90_m": round(float(np.percentile(values, 90)), 4),
        "center_m": round(center_depth, 4) if math.isfinite(center_depth) else None,
        "ground_distance_m": point_3d.get("ground_distance_m"),
        "ground_forward_m": point_3d.get("ground_forward_m"),
        "center_3d": point_3d,
        "depth_convention": pickup_depth_convention(camera),
    }
    if str(candidate.get("task_semantic_class") or "") in {"pickup_target", "cleanable_object"}:
        result["floor_contact"] = analyze_pickup_floor_contact(
            depth_frame,
            camera,
            (x1, y1, x2, y2),
            floor_plane=floor_plane,
            projector=project_pixel_to_3d,
        )
    return result


def depth_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def candidate_depth_distance(candidate: JsonDict) -> Optional[float]:
    direct = depth_number(candidate.get("distance"))
    if direct is not None:
        return direct
    depth = candidate.get("depth") if isinstance(candidate.get("depth"), dict) else {}
    for key in ("distance_m", "median_m", "center_m"):
        value = depth_number(depth.get(key))
        if value is not None:
            return value
    return None


def candidate_ground_distance(candidate: JsonDict) -> Optional[float]:
    direct = depth_number(candidate.get("ground_distance"))
    if direct is not None:
        return direct
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    value = depth_number(geometry.get("ground_distance_m"))
    if value is not None:
        return value
    center_3d = candidate_center_3d(candidate)
    if isinstance(center_3d, dict):
        value = depth_number(center_3d.get("ground_distance_m"))
        if value is not None:
            return value
    depth = candidate.get("depth") if isinstance(candidate.get("depth"), dict) else {}
    value = depth_number(depth.get("ground_distance_m"))
    if value is not None:
        return value
    return None


def candidate_center_3d(candidate: JsonDict) -> Optional[JsonDict]:
    direct = candidate.get("center_3d") if isinstance(candidate.get("center_3d"), dict) else None
    if direct:
        return direct
    depth = candidate.get("depth") if isinstance(candidate.get("depth"), dict) else {}
    nested = depth.get("center_3d") if isinstance(depth.get("center_3d"), dict) else None
    return nested


def candidate_bearing_deg(candidate: JsonDict) -> Optional[float]:
    existing = depth_number((candidate.get("geometry") or {}).get("bearing_deg") if isinstance(candidate.get("geometry"), dict) else None)
    if existing is not None:
        return existing
    point = candidate_center_3d(candidate)
    if not isinstance(point, dict):
        return None
    x = depth_number(point.get("x"))
    z = depth_number(point.get("ground_forward_m"))
    if z is None:
        z = depth_number(point.get("z"))
    if x is None or z is None or z <= 0:
        return None
    return math.degrees(math.atan2(x, z))


def normalize_label_tokens(labels: Optional[Iterable[Any]]) -> List[str]:
    tokens: List[str] = []
    for label in labels or []:
        for part in str(label or "").split(","):
            token = normalize_label(part)
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def candidate_label_matches(candidate: JsonDict, labels: Iterable[str]) -> bool:
    held = {normalize_label(label) for label in labels if normalize_label(label)}
    held_compact = {label.replace("_", "") for label in held}
    if not held:
        return False
    candidate_tokens = {
        normalize_label(candidate.get("label")),
        normalize_label(candidate.get("raw_label")),
    }
    candidate_compact = {label.replace("_", "") for label in candidate_tokens if label}
    return bool((candidate_tokens & held) or (candidate_compact & held_compact))


def held_object_family_for_labels(labels: Iterable[str]) -> str:
    tokens = {normalize_label(label) for label in labels if normalize_label(label)}
    if tokens & HELD_FOOD_LABELS:
        return "food"
    if tokens & HELD_PICKUP_LABELS:
        return "pickup_target"
    return "unknown"


def candidate_family(candidate: JsonDict) -> str:
    label = normalize_label(candidate.get("label") or candidate.get("raw_label"))
    if label in HELD_FOOD_LABELS:
        return "food"
    if str(candidate.get("task_semantic_class") or "") == "pickup_target" or label in HELD_PICKUP_LABELS:
        return "pickup_target"
    return "unknown"


def candidate_foreground_overlay_score(candidate: JsonDict) -> Tuple[bool, JsonDict]:
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    cx_ratio = depth_number(geometry.get("cx_ratio"), 0.5) or 0.5
    cy_ratio = depth_number(geometry.get("cy_ratio"), candidate.get("center_y_ratio")) or 0.5
    bottom_y_ratio = depth_number(geometry.get("bottom_y_ratio"), candidate.get("bottom_y_ratio")) or 0.0
    area_ratio = depth_number(candidate.get("area_ratio"), geometry.get("area_ratio")) or 0.0
    distance_m = candidate_ground_distance(candidate)
    if distance_m is None:
        distance_m = candidate_depth_distance(candidate)
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


def filter_held_object_blockers(
    candidates: Iterable[JsonDict],
    *,
    holding_object: bool,
    held_object_labels: Iterable[str],
    held_object_family: Optional[str] = None,
) -> Tuple[List[JsonDict], int]:
    held_labels = normalize_label_tokens(held_object_labels)
    filtered: List[JsonDict] = []
    ignored = 0
    for candidate in candidates:
        if (
            holding_object
            and env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True)
            and isinstance(candidate, dict)
            and candidate_likely_held_object(candidate, held_labels, held_object_family)
        ):
            candidate["held_object_candidate"] = True
            candidate["ignored_as_held_object_blocker"] = True
            ignored += 1
            continue
        filtered.append(candidate)
    return filtered, ignored


def mark_held_object_overlays(
    candidates: Iterable[JsonDict],
    *,
    holding_object: bool,
    held_object_labels: Iterable[str],
    held_object_family: Optional[str] = None,
) -> List[JsonDict]:
    if not holding_object or not env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True):
        return []
    held_labels = normalize_label_tokens(held_object_labels)
    overlays: List[JsonDict] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate_likely_held_object(candidate, held_labels, held_object_family):
            candidate["held_object_candidate"] = True
            candidate["ignored_as_held_object_blocker"] = True
            candidate["likely_held_object_overlay"] = True
            key = (
                normalize_label(candidate.get("label") or candidate.get("raw_label")),
                str(candidate.get("bbox")),
            )
            if key not in seen:
                seen.add(key)
                overlays.append(candidate)
    return overlays


def infer_held_object_labels_from_candidates(candidates: Iterable[JsonDict]) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("task_semantic_class") or "") != "pickup_target":
            continue
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        cx_ratio = depth_number(geometry.get("cx_ratio"), 0.5) or 0.5
        bottom_y_ratio = depth_number(geometry.get("bottom_y_ratio"), candidate.get("bottom_y_ratio")) or 0.0
        area_ratio = depth_number(candidate.get("area_ratio"), geometry.get("area_ratio")) or 0.0
        if abs(cx_ratio - 0.5) > env_float("ROBOT_HELD_OBJECT_INFER_CENTER_TOLERANCE", 0.28):
            continue
        if bottom_y_ratio < env_float("ROBOT_HELD_OBJECT_INFER_MIN_BOTTOM_RATIO", 0.65):
            continue
        if area_ratio < env_float("ROBOT_HELD_OBJECT_INFER_MIN_AREA_RATIO", 0.003):
            continue
        label = normalize_label(candidate.get("label") or candidate.get("raw_label") or "")
        if not label:
            continue
        score = (
            area_ratio * 10.0
            + bottom_y_ratio
            + max(0.0, 0.5 - abs(cx_ratio - 0.5)) * 2.0
            + float(candidate.get("confidence", 0.0) or 0.0)
        )
        scored.append((score, label))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [scored[0][1]]


def apply_depth_actionability(candidate: JsonDict) -> None:
    """Use metric depth for action hints that used to be estimated from bbox size."""
    task_class = str(candidate.get("task_semantic_class") or "")
    if task_class not in {"pickup_target", "place_receptacle", "cleanable_object", "obstacle"}:
        return

    distance_m = candidate_depth_distance(candidate)
    center_3d = candidate_center_3d(candidate)
    bearing_deg = candidate_bearing_deg(candidate)
    if distance_m is None or center_3d is None or bearing_deg is None:
        return

    height_m = depth_number(center_3d.get("y"))
    ground_distance_m = candidate_ground_distance(candidate)
    action_distance_m = ground_distance_m if ground_distance_m is not None else distance_m
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    geometry["bearing_deg"] = round(float(bearing_deg), 3)
    geometry["distance_m"] = round(float(distance_m), 4)
    if ground_distance_m is not None:
        geometry["ground_distance_m"] = round(float(ground_distance_m), 4)
    if height_m is not None:
        geometry["height_m"] = round(float(height_m), 4)
    candidate["geometry"] = geometry
    candidate["center_3d"] = dict(center_3d)
    candidate["distance"] = round(float(distance_m), 4)
    if ground_distance_m is not None:
        candidate["ground_distance"] = round(float(ground_distance_m), 4)
    if height_m is not None:
        candidate["height"] = round(float(height_m), 4)

    center_tol = env_float("ROBOT_DEPTH_CENTER_TOLERANCE_DEG", 8.0)
    align_tol = env_float("ROBOT_DEPTH_ALIGNMENT_TOLERANCE_DEG", center_tol)
    max_side_bearing = env_float("ROBOT_DEPTH_MAX_SERVICE_BEARING_DEG", 42.0)
    centered = abs(bearing_deg) <= center_tol
    service_visible = abs(bearing_deg) <= max_side_bearing
    candidate["position_hint"] = infer_position_hint_from_bearing(bearing_deg, center_tolerance_deg=center_tol)
    candidate["actionability_source"] = "depth_geometry"

    floor_min_h = env_float("ROBOT_DEPTH_FLOOR_MIN_HEIGHT_M", -0.12)
    floor_max_h = env_float("ROBOT_DEPTH_FLOOR_MAX_HEIGHT_M", 0.22)
    support_min_h = env_float("ROBOT_DEPTH_SUPPORT_MIN_HEIGHT_M", 0.35)
    support_max_h = env_float("ROBOT_DEPTH_SUPPORT_MAX_HEIGHT_M", 1.25)
    floor_like_by_height = bool(height_m is not None and floor_min_h <= height_m <= floor_max_h)
    floor_like_by_bottom_band = False
    depth_payload = candidate.get("depth") if isinstance(candidate.get("depth"), dict) else {}
    floor_contact = depth_payload.get("floor_contact") if isinstance(depth_payload.get("floor_contact"), dict) else {}
    contact_support_height = depth_number(floor_contact.get("support_height_m"))
    contact_max_support_height = env_float("ROBOT_DEPTH_FLOOR_CONTACT_MAX_SUPPORT_HEIGHT_M", 0.16)
    raw_floor_like_by_contact = bool(floor_contact.get("contact_floor_like"))
    floor_like_by_contact = bool(
        raw_floor_like_by_contact
        and (contact_support_height is None or contact_support_height <= contact_max_support_height)
    )
    floor_level_source = "rgbd_floor_plane_contact" if floor_like_by_contact else None
    if floor_contact:
        candidate["floor_contact_geometry"] = dict(floor_contact)
        candidate["bottom_contact_3d"] = floor_contact.get("bottom_contact_3d")
        candidate["floor_plane_residual_m"] = floor_contact.get("floor_plane_residual_m")
        candidate["floor_contact_confidence"] = floor_contact.get("confidence")
        if raw_floor_like_by_contact and not floor_like_by_contact:
            candidate["floor_contact_rejected_reason"] = "support_height_above_floor_band"
            candidate["floor_contact_max_support_height_m"] = round(float(contact_max_support_height), 4)

    geometry_bottom = depth_number(geometry.get("bottom_y_ratio"))
    candidate_bottom = depth_number(candidate.get("bottom_y_ratio"))
    bottom_y_ratio = geometry_bottom if geometry_bottom is not None else candidate_bottom
    if bottom_y_ratio is None:
        bottom_y_ratio = 0.0

    # A single projected center is weak evidence: it may hit the object side,
    # table surface, or background. For floor-only pickup targets, require
    # bottom contact or a conservative bottom-band ground-distance fallback.
    if (
        task_class == "pickup_target"
        and env_bool("ROBOT_DEPTH_PICKUP_FLOOR_FALLBACK_ENABLED", True)
    ):
        fallback_min_bottom = env_float(
            "ROBOT_DEPTH_PICKUP_FLOOR_FALLBACK_MIN_BOTTOM_RATIO",
            env_float("ROBOT_PICKUP_SMALL_FLOOR_BOTTOM_RATIO", 0.88),
        )
        fallback_min_ground = env_float("ROBOT_DEPTH_PICKUP_FLOOR_FALLBACK_MIN_GROUND_DISTANCE", 0.15)
        fallback_max_ground = env_float(
            "ROBOT_DEPTH_PICKUP_FLOOR_FALLBACK_MAX_GROUND_DISTANCE",
            env_float("ROBOT_DEPTH_PICKUP_NOW_MAX_DISTANCE", 1.35),
        )
        height_not_elevated = bool(height_m is None or height_m <= floor_max_h)
        floor_like_by_bottom_band = bool(
            height_not_elevated
            and bottom_y_ratio >= fallback_min_bottom
            and fallback_min_ground <= action_distance_m <= fallback_max_ground
        )
        if floor_like_by_bottom_band:
            floor_level_source = "bottom_band_ground_distance_fallback"

    height_band_decisive = bool(task_class != "pickup_target" and floor_like_by_height)
    if height_band_decisive and floor_level_source is None:
        floor_level_source = "height_band"

    floor_like = bool(floor_like_by_contact or floor_like_by_bottom_band or height_band_decisive)
    support_like = bool(height_m is not None and support_min_h <= height_m <= support_max_h)

    if task_class in {"pickup_target", "cleanable_object"}:
        candidate["is_floor_level"] = bool(floor_like)
        candidate["surface_hint"] = "floor" if floor_like else "surface_or_elevated"
        candidate["height_band_floor_like"] = bool(floor_like_by_height)
        candidate["height_band_floor_like_used"] = bool(height_band_decisive)
        candidate["floor_level_source"] = floor_level_source or (
            "height_band_weak" if floor_like_by_height else "height_out_of_floor_band"
        )
        if floor_like_by_contact and not floor_like_by_height:
            candidate["projected_height_warning"] = "center_height_out_of_floor_band_but_rgbd_bottom_contact_is_floor_like"
        elif floor_like_by_bottom_band:
            candidate["projected_height_warning"] = "height_out_of_floor_band_but_bottom_band_ground_distance_floor_like"

    if task_class == "pickup_target":
        max_context_distance = env_float("ROBOT_DEPTH_PICKUP_CONTEXT_MAX_DISTANCE", 2.2)
        max_now_distance = env_float("ROBOT_DEPTH_PICKUP_NOW_MAX_DISTANCE", 1.35)
        min_conf = env_float("ROBOT_PICKUP_MIN_CONF", 0.55)
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        candidate["reachable"] = bool(service_visible and action_distance_m <= max_context_distance)
        candidate["pickup_now"] = bool(
            floor_like
            and centered
            and action_distance_m <= max_now_distance
            and confidence >= min_conf
        )
        candidate["needs_alignment"] = bool(candidate["reachable"] and not centered and abs(bearing_deg) > align_tol)
        candidate["needs_approach"] = bool(
            floor_like
            and centered
            and max_now_distance < action_distance_m <= max_context_distance
            and confidence >= min_conf
        )
    elif task_class == "cleanable_object":
        max_context_distance = env_float("ROBOT_DEPTH_CLEAN_CONTEXT_MAX_DISTANCE", 2.0)
        max_now_distance = env_float("ROBOT_DEPTH_CLEAN_NOW_MAX_DISTANCE", 1.25)
        candidate["reachable"] = bool(service_visible and action_distance_m <= max_context_distance)
        candidate["cleanable_now"] = bool(floor_like and centered and action_distance_m <= max_now_distance)
        candidate["needs_alignment"] = bool(candidate["reachable"] and not centered and abs(bearing_deg) > align_tol)
        candidate["needs_approach"] = bool(floor_like and centered and max_now_distance < action_distance_m <= max_context_distance)
    elif task_class == "place_receptacle":
        max_context_distance = env_float("ROBOT_DEPTH_RECEPTACLE_CONTEXT_MAX_DISTANCE", 2.6)
        candidate["reachable"] = bool(service_visible and action_distance_m <= max_context_distance)
        candidate["is_floor_level"] = False
        candidate["is_support_surface"] = True
        candidate["surface_hint"] = "support_surface" if support_like or height_m is None else "support_surface"
        candidate["place_now"] = False
        candidate["needs_alignment"] = bool(candidate["reachable"] and not centered and abs(bearing_deg) > align_tol)
        candidate["needs_approach"] = bool(candidate["reachable"] and centered)
    elif task_class == "obstacle":
        max_risk_distance = env_float("ROBOT_DEPTH_OBSTACLE_RISK_MAX_DISTANCE", 1.35)
        risk_bearing = env_float("ROBOT_DEPTH_OBSTACLE_RISK_BEARING_DEG", 38.0)
        candidate["reachable"] = bool(action_distance_m <= max_risk_distance and abs(bearing_deg) <= risk_bearing)
        candidate["obstacle_risk"] = bool(action_distance_m <= max_risk_distance and abs(bearing_deg) <= risk_bearing)


def pixel_in_candidate_bbox(px: float, py: float, candidate: JsonDict, *, margin: float = 0.0) -> bool:
    box = bbox_pixel_tuple(candidate)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin)


def surface_blockers_for(
    parent: JsonDict,
    candidates: List[JsonDict],
    *,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
) -> List[JsonDict]:
    blockers: List[JsonDict] = []
    parent_box = bbox_pixel_tuple(parent)
    if parent_box is None:
        return blockers
    px1, py1, px2, py2 = parent_box
    held_labels = normalize_label_tokens(held_object_labels)
    for candidate in candidates:
        if candidate is parent:
            continue
        if (
            holding_object
            and env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True)
            and candidate_likely_held_object(candidate, held_labels, held_object_family)
        ):
            candidate["held_object_candidate"] = True
            candidate["ignored_as_held_object_blocker"] = True
            continue
        label = normalize_label(candidate.get("label") or candidate.get("raw_label") or "")
        task_class = str(candidate.get("task_semantic_class") or "")
        if task_class not in {"pickup_target", "ignored_object", "obstacle", "place_receptacle"}:
            continue
        if task_class == "place_receptacle" and label not in SURFACE_BLOCKING_LABELS:
            continue
        box = bbox_pixel_tuple(candidate)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        overlap = max(0.0, min(px2, x2) - max(px1, x1)) * max(0.0, min(py2, y2) - max(py1, y1))
        if overlap > 0:
            blockers.append(candidate)
    return blockers


def bbox_overlap_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def expand_bbox(
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


def surface_region_id(parent_label: str, bbox: JsonDict, height_m: Optional[float], distance_m: float) -> str:
    token = normalize_label(parent_label or "surface") or "surface"
    try:
        x = int(round(float(bbox.get("x", 0) or 0)))
        y = int(round(float(bbox.get("y", 0) or 0)))
        w = int(round(float(bbox.get("w", 0) or 0)))
        h = int(round(float(bbox.get("h", 0) or 0)))
    except (TypeError, ValueError):
        x, y, w, h = 0, 0, 0, 0
    height_bucket = int(round(float(height_m if height_m is not None else 0.0) * 100.0))
    distance_bucket = int(round(float(distance_m) * 100.0))
    return f"surface:{token}:{x}:{y}:{w}:{h}:{height_bucket}:{distance_bucket}"


def surface_region_blockers(
    region_box: Tuple[float, float, float, float],
    blockers: List[JsonDict],
    *,
    image_w: int,
    image_h: int,
) -> List[str]:
    expanded = expand_bbox(
        region_box,
        margin=env_float("ROBOT_DEPTH_SURFACE_OCCUPANCY_MARGIN_PIXELS", 12.0),
        image_w=image_w,
        image_h=image_h,
    )
    blocked_by: List[str] = []
    for blocker in blockers:
        box = bbox_pixel_tuple(blocker)
        if box is None:
            continue
        label = str(blocker.get("label") or blocker.get("raw_label") or "object")
        if bbox_overlap_area(expanded, box) <= 0:
            continue
        if label not in blocked_by:
            blocked_by.append(label)
    return blocked_by


def summarize_surface_rejections(surface_regions: Iterable[JsonDict]) -> JsonDict:
    summary: Dict[str, int] = {
        "too_small": 0,
        "too_close": 0,
        "too_far": 0,
        "height_out_of_range": 0,
        "depth_unstable": 0,
        "thin_region": 0,
        "single_row_region": 0,
        "touches_image_edge": 0,
        "touches_parent_edge": 0,
        "blocked": 0,
        "failed_recently": 0,
    }
    for region in surface_regions:
        if not isinstance(region, dict):
            continue
        reasons = list(region.get("rejection_reasons") or [])
        memory_checks = region.get("memory_checks") if isinstance(region.get("memory_checks"), dict) else {}
        if region.get("failed_recently") or memory_checks.get("failed_recently"):
            reasons.append("failed_recently")
        for reason in reasons:
            token = str(reason or "")
            if token.startswith("blocked"):
                key = "blocked"
            elif token.startswith("cooldown"):
                key = "failed_recently"
            else:
                key = token
            if key not in summary:
                summary[key] = 0
            summary[key] += 1
    return summary


def make_surface_region_candidate(
    *,
    parent: JsonDict,
    patches: List[JsonDict],
    blockers: List[JsonDict],
    image_w: int,
    image_h: int,
    camera: JsonDict,
) -> JsonDict:
    import numpy as np  # type: ignore

    if not patches:
        return {}

    x1 = min(float(patch["bbox"][0]) for patch in patches)
    y1 = min(float(patch["bbox"][1]) for patch in patches)
    x2 = max(float(patch["bbox"][2]) for patch in patches)
    y2 = max(float(patch["bbox"][3]) for patch in patches)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area_px = width * height
    area_ratio = area_px / max(1.0, float(image_w * image_h))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    patch_count = len(patches)
    grid_rows = {
        int(patch.get("grid", (0, 0))[0])
        for patch in patches
        if isinstance(patch.get("grid"), tuple)
    }
    grid_cols = {
        int(patch.get("grid", (0, 0))[1])
        for patch in patches
        if isinstance(patch.get("grid"), tuple)
    }
    patch_rows = len(grid_rows) if grid_rows else 1
    patch_cols = len(grid_cols) if grid_cols else 1

    depths = [float(patch["median_depth"]) for patch in patches]
    heights = [
        float(patch["height_m"])
        for patch in patches
        if patch.get("height_m") is not None and math.isfinite(float(patch["height_m"]))
    ]
    ground_distances = [
        float(patch["ground_distance_m"])
        for patch in patches
        if patch.get("ground_distance_m") is not None and math.isfinite(float(patch["ground_distance_m"]))
    ]
    depth_q10 = min(float(patch["q10"]) for patch in patches)
    depth_q90 = max(float(patch["q90"]) for patch in patches)
    depth_span = depth_q90 - depth_q10
    median_depth = float(np.median(depths))
    height_m = float(np.median(heights)) if heights else None
    center_height_std = float(np.std(heights)) if len(heights) > 1 else 0.0
    local_height_stds = [
        float(patch.get("height_std", 0.0) or 0.0)
        for patch in patches
        if math.isfinite(float(patch.get("height_std", 0.0) or 0.0))
    ]
    local_height_std = max(local_height_stds, default=0.0)
    height_std = max(center_height_std, local_height_std)
    dominant_support_heights = [
        float(patch.get("dominant_support_height_m"))
        for patch in patches
        if patch.get("dominant_support_height_m") is not None
    ]
    dominant_support_height_m = (
        float(np.median(dominant_support_heights))
        if dominant_support_heights
        else height_m
    )
    center_3d = project_pixel_to_3d(center_x, center_y, median_depth, camera)
    center_ground_distance = depth_number(center_3d.get("ground_distance_m"))
    if ground_distances:
        distance_m = float(np.median(ground_distances))
    elif center_ground_distance is not None:
        distance_m = center_ground_distance
    else:
        distance_m = median_depth
    if height_m is None:
        height_m = depth_number(center_3d.get("y"))
    x_3d = depth_number(center_3d.get("x"), 0.0) or 0.0
    ground_forward = depth_number(center_3d.get("ground_forward_m"), median_depth) or median_depth
    bearing_deg = math.degrees(math.atan2(x_3d, max(0.001, ground_forward)))
    center_tol = env_float("ROBOT_DEPTH_CENTER_TOLERANCE_DEG", 8.0)
    position_hint = infer_position_hint_from_bearing(bearing_deg, center_tolerance_deg=center_tol)

    parent_box = bbox_pixel_tuple(parent)
    if parent_box is None:
        parent_box = (0.0, 0.0, float(image_w), float(image_h))
    px1, py1, px2, py2 = parent_box
    image_margin = env_float("ROBOT_DEPTH_SURFACE_IMAGE_EDGE_MARGIN_PIXELS", 8.0)
    parent_margin = env_float("ROBOT_DEPTH_SURFACE_PARENT_EDGE_MARGIN_PIXELS", 6.0)
    min_w = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_W_PIXELS", 45.0)
    min_h = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_H_PIXELS", 35.0)
    min_area_ratio = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_AREA_RATIO", 0.004)
    min_patch_count = max(1, env_int("ROBOT_DEPTH_SURFACE_REGION_MIN_PATCH_COUNT", 4))
    min_patch_rows = max(1, env_int("ROBOT_DEPTH_SURFACE_REGION_MIN_PATCH_ROWS", 2))
    min_patch_cols = max(1, env_int("ROBOT_DEPTH_SURFACE_REGION_MIN_PATCH_COLS", 2))
    max_aspect_ratio = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_ASPECT_RATIO", 5.0)
    min_distance = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_DISTANCE_M", 0.55)
    max_distance = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_DISTANCE_M", 1.50)
    min_height = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_HEIGHT_M", 0.55)
    max_height = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_HEIGHT_M", 1.15)
    max_depth_span = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_DEPTH_SPAN_M", 0.12)
    max_height_std = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_HEIGHT_STD_M", 0.05)

    area_ok = bool(width >= min_w and height >= min_h and area_ratio >= min_area_ratio)
    single_row_region = bool(patch_rows < min_patch_rows)
    single_col_region = bool(patch_cols < min_patch_cols)
    patch_count_ok = bool(patch_count >= min_patch_count)
    patch_grid_ok = bool(patch_count_ok and not single_row_region and not single_col_region)
    aspect_ratio = max(width, height) / max(1.0, min(width, height))
    thin_region = bool(width < min_w or height < min_h or aspect_ratio > max_aspect_ratio)
    distance_ok = bool(min_distance <= distance_m <= max_distance)
    height_ok = bool(height_m is not None and min_height <= float(height_m) <= max_height)
    image_edge_ok = bool(
        x1 >= image_margin
        and y1 >= image_margin
        and x2 <= float(image_w) - image_margin
        and y2 <= float(image_h) - image_margin
    )
    parent_edge_ok = bool(
        x1 >= px1 + parent_margin
        and y1 >= py1 + parent_margin
        and x2 <= px2 - parent_margin
        and y2 <= py2 - parent_margin
    )
    edge_ok = bool(image_edge_ok and parent_edge_ok)
    depth_stable = bool(depth_span <= max_depth_span)
    normal_like_horizontal = bool(height_std <= max_height_std)
    blocked_by = surface_region_blockers((x1, y1, x2, y2), blockers, image_w=image_w, image_h=image_h)
    blocked = bool(blocked_by)

    rejection_reasons: List[str] = []
    if not area_ok:
        rejection_reasons.append("too_small")
    if thin_region:
        rejection_reasons.append("thin_region")
    if single_row_region:
        rejection_reasons.append("single_row_region")
    if single_col_region:
        rejection_reasons.append("single_col_region")
    if not patch_count_ok:
        rejection_reasons.append("too_few_patches")
    if distance_m < min_distance:
        rejection_reasons.append("too_close")
    elif distance_m > max_distance:
        rejection_reasons.append("too_far")
    if not height_ok:
        rejection_reasons.append("height_out_of_range")
    if not image_edge_ok:
        rejection_reasons.append("touches_image_edge")
    if not parent_edge_ok:
        rejection_reasons.append("touches_parent_edge")
    if not depth_stable:
        rejection_reasons.append("depth_unstable")
    if not normal_like_horizontal:
        rejection_reasons.append("not_horizontal")
    for blocker in blocked_by:
        rejection_reasons.append(f"blocked:{normalize_label(blocker) or blocker}")

    visual_place_ready = bool(
        area_ok
        and patch_grid_ok
        and not thin_region
        and distance_ok
        and height_ok
        and edge_ok
        and depth_stable
        and normal_like_horizontal
        and not blocked
    )
    depth_score = max(0.0, 1.0 - min(1.0, depth_span / max(0.001, max_depth_span)))
    height_score = 0.0 if height_m is None else max(0.0, 1.0 - min(1.0, abs(float(height_m) - 0.85) / 0.45))
    distance_score = max(0.0, 1.0 - min(1.0, abs(distance_m - 0.95) / 0.65))
    area_score = min(1.0, area_ratio / max(0.001, min_area_ratio * 4.0))
    shape_score = min(1.0, patch_count / max(1.0, float(min_patch_count) * 2.0))
    center_score = max(0.0, 1.0 - min(1.0, abs((center_x / max(1.0, image_w)) - 0.5) / 0.5))
    score = (
        0.22 * area_score
        + 0.20 * depth_score
        + 0.18 * height_score
        + 0.16 * distance_score
        + 0.12 * shape_score
        + 0.12 * center_score
    )
    if not visual_place_ready:
        score *= 0.45
    if blocked:
        score *= 0.35

    bbox = {"x": int(round(x1)), "y": int(round(y1)), "w": int(round(width)), "h": int(round(height))}
    parent_label = str(parent.get("label") or parent.get("raw_label") or "receptacle")
    candidate_id = surface_region_id(parent_label, bbox, height_m, distance_m)
    checks = {
        "area_ok": bool(area_ok),
        "distance_ok": bool(distance_ok),
        "height_ok": bool(height_ok),
        "edge_ok": bool(edge_ok),
        "depth_stable": bool(depth_stable),
        "normal_like_horizontal": bool(normal_like_horizontal),
        "patch_count_ok": bool(patch_count_ok),
        "patch_grid_ok": bool(patch_grid_ok),
        "thin_region": bool(thin_region),
        "single_row_region": bool(single_row_region),
        "single_col_region": bool(single_col_region),
        "patch_count": int(patch_count),
        "patch_rows": int(patch_rows),
        "patch_cols": int(patch_cols),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "region_w": round(float(width), 3),
        "region_h": round(float(height), 3),
        "depth_span_m": round(float(depth_span), 4),
        "height_std_m": round(float(height_std), 4),
        "center_height_std_m": round(float(center_height_std), 4),
        "local_height_std_m": round(float(local_height_std), 4),
        "dominant_support_height_m": round(float(dominant_support_height_m), 4)
        if dominant_support_height_m is not None
        else None,
        "image_edge_ok": bool(image_edge_ok),
        "parent_edge_ok": bool(parent_edge_ok),
    }

    region_type = "placeable_surface_region" if visual_place_ready else "rejected_surface_region"
    if blocked:
        region_type = "blocked_surface_region"

    return {
        "id": candidate_id,
        "surface_candidate_id": candidate_id,
        "label": parent.get("label"),
        "raw_label": parent.get("raw_label") or parent.get("label"),
        "task_semantic_class": "place_receptacle",
        "confidence": round(float(parent.get("confidence", 0.0) or 0.0) * max(0.35, min(1.0, score)), 4),
        "bbox": bbox,
        "region_bbox": dict(bbox),
        "image_size": {"w": int(image_w), "h": int(image_h)},
        "center": {"x": int(round(center_x)), "y": int(round(center_y))},
        "interaction_point": {"x": round(float(center_x), 3), "y": round(float(center_y), 3)},
        "center_3d": center_3d,
        "position_hint": position_hint,
        "surface_hint": "support_surface",
        "is_floor_level": False,
        "is_support_surface": True,
        "reachable": bool(visual_place_ready or (not blocked and distance_ok and height_ok)),
        "pickup_now": False,
        "place_now": False,
        "visual_place_ready": bool(visual_place_ready),
        "final_place_ready": False,
        "cleanable_now": False,
        "needs_alignment": bool(visual_place_ready and abs(bearing_deg) > center_tol),
        "needs_approach": bool(visual_place_ready and abs(bearing_deg) <= center_tol and distance_m > max_distance),
        "obstacle_risk": False,
        "visual_box_ambiguous": False,
        "broad_front_receptacle": False,
        "front_edge_receptacle": False,
        "area": round(float(area_px), 2),
        "area_ratio": round(float(area_ratio), 6),
        "region_area_px": int(round(area_px)),
        "region_area_ratio": round(float(area_ratio), 6),
        "patch_count": int(patch_count),
        "patch_rows": int(patch_rows),
        "patch_cols": int(patch_cols),
        "center_y_ratio": round(float(center_y / max(1.0, image_h)), 3),
        "bottom_y_ratio": round(float(y2 / max(1.0, image_h)), 3),
        "height_m": round(float(height_m), 4) if height_m is not None else None,
        "dominant_support_height_m": round(float(dominant_support_height_m), 4)
        if dominant_support_height_m is not None
        else None,
        "distance_m": round(float(distance_m), 4),
        "depth_m": round(float(median_depth), 4),
        "bearing_deg": round(float(bearing_deg), 3),
        "geometry": {
            "cx_ratio": round(float(center_x / max(1.0, image_w)), 4),
            "cy_ratio": round(float(center_y / max(1.0, image_h)), 4),
            "bottom_y_ratio": round(float(y2 / max(1.0, image_h)), 4),
            "area_ratio": round(float(area_ratio), 6),
            "width_ratio": round(float(width / max(1.0, image_w)), 6),
            "height_ratio": round(float(height / max(1.0, image_h)), 6),
            "bearing_deg": round(float(bearing_deg), 3),
            "distance_m": round(float(distance_m), 4),
            "depth_m": round(float(median_depth), 4),
            "ground_forward_m": center_3d.get("ground_forward_m"),
            "ground_distance_m": round(float(distance_m), 4),
            "height_m": round(float(height_m), 4) if height_m is not None else None,
            "dominant_support_height_m": round(float(dominant_support_height_m), 4)
            if dominant_support_height_m is not None
            else None,
            "height_std_m": round(float(height_std), 4),
            "depth_span_m": round(float(depth_span), 4),
        },
        "geometry_checks": checks,
        "occupancy_checks": {
            "blocked": bool(blocked),
            "blocked_by": blocked_by,
        },
        "memory_checks": {
            "failed_recently": False,
            "cooldown_remaining": 0,
        },
        "executor_checks": {
            "precheck_supported": True,
            "precheck_ok": False,
            "reason": "precheck_not_run",
            "suggested_recovery": None,
        },
        "region_type": region_type,
        "affordance": ["place"] if visual_place_ready else [],
        "parent_object": normalize_label(parent_label),
        "parent_label": parent_label,
        "parent_bbox": parent.get("bbox"),
        "source": "depth_region_geometry",
        "surface_candidate_source": "depth_region_geometry",
        "actionability_source": "depth_region_geometry",
        "score": round(float(score), 4),
        "blocked": bool(blocked),
        "blocked_by": blocked_by,
        "rejection_reasons": rejection_reasons,
        "depth": {
            "unit": "meter",
            "median_m": round(float(median_depth), 4),
            "camera_forward_m": round(float(median_depth), 4),
            "distance_m": round(float(distance_m), 4),
            "ground_distance_m": round(float(distance_m), 4),
            "p10_m": round(float(depth_q10), 4),
            "p90_m": round(float(depth_q90), 4),
        },
        "height": round(float(height_m), 4) if height_m is not None else None,
        "distance": round(float(distance_m), 4),
        "ground_distance": round(float(distance_m), 4),
    }


def make_surface_candidate(
    *,
    parent: JsonDict,
    center_x: float,
    center_y: float,
    box_w: float,
    box_h: float,
    image_w: int,
    image_h: int,
    depth_m: float,
    height_m: Optional[float],
    score: float,
    blocked: bool,
    blocked_by: List[str],
    camera: JsonDict,
) -> JsonDict:
    x1 = max(0.0, center_x - box_w / 2.0)
    y1 = max(0.0, center_y - box_h / 2.0)
    x2 = min(float(image_w), center_x + box_w / 2.0)
    y2 = min(float(image_h), center_y + box_h / 2.0)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx_ratio = center_x / max(1.0, float(image_w))
    cy_ratio = center_y / max(1.0, float(image_h))
    bottom_y_ratio = y2 / max(1.0, float(image_h))
    area_ratio = (width * height) / max(1.0, float(image_w * image_h))
    center_3d = project_pixel_to_3d(center_x, center_y, depth_m, camera)
    x_3d = depth_number(center_3d.get("x"), 0.0) or 0.0
    ground_forward_m = depth_number(center_3d.get("ground_forward_m"), depth_m) or depth_m
    ground_distance_m = depth_number(center_3d.get("ground_distance_m"), math.hypot(x_3d, ground_forward_m))
    if ground_distance_m is None:
        ground_distance_m = math.hypot(x_3d, ground_forward_m)
    bearing_deg = math.degrees(math.atan2(x_3d, max(0.001, ground_forward_m)))
    center_tolerance_deg = env_float("ROBOT_DEPTH_CENTER_TOLERANCE_DEG", 8.0)
    position_hint = infer_position_hint_from_bearing(bearing_deg, center_tolerance_deg=center_tolerance_deg)

    max_place_now_depth = env_float("ROBOT_DEPTH_SURFACE_PLACE_NOW_MAX_DISTANCE", 1.8)
    max_context_depth = env_float("ROBOT_DEPTH_SURFACE_CONTEXT_MAX_DISTANCE", 2.6)
    min_context_ground = env_float("ROBOT_DEPTH_SURFACE_CONTEXT_MIN_GROUND_DISTANCE", 0.35)
    max_context_ground = env_float("ROBOT_DEPTH_SURFACE_CONTEXT_MAX_GROUND_DISTANCE", max_context_depth)
    min_place_now_ground = env_float("ROBOT_DEPTH_SURFACE_PLACE_NOW_MIN_GROUND_DISTANCE", 0.50)
    max_place_now_ground = env_float(
        "ROBOT_DEPTH_SURFACE_PLACE_NOW_MAX_GROUND_DISTANCE",
        env_float("ROBOT_PLACE_MAX_DISTANCE", 1.0),
    )
    hard_min_w = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_W_PIXELS", 45.0)
    hard_min_h = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_H_PIXELS", 35.0)
    hard_min_area_ratio = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_AREA_RATIO", 0.004)
    hard_min_ground = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_DISTANCE_M", 0.55)
    hard_max_ground = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_DISTANCE_M", 1.50)
    hard_min_height = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_HEIGHT_M", 0.55)
    hard_max_height = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_HEIGHT_M", 1.15)
    image_margin = env_float("ROBOT_DEPTH_SURFACE_IMAGE_EDGE_MARGIN_PIXELS", 8.0)
    parent_margin = env_float("ROBOT_DEPTH_SURFACE_PARENT_EDGE_MARGIN_PIXELS", 6.0)
    parent_box = bbox_pixel_tuple(parent)
    if parent_box is None:
        parent_box = (0.0, 0.0, float(image_w), float(image_h))
    px1, py1, px2, py2 = parent_box
    area_ok = bool(width >= hard_min_w and height >= hard_min_h and area_ratio >= hard_min_area_ratio)
    hard_distance_ok = bool(hard_min_ground <= ground_distance_m <= hard_max_ground)
    hard_height_ok = bool(height_m is not None and hard_min_height <= float(height_m) <= hard_max_height)
    image_edge_ok = bool(
        x1 >= image_margin
        and y1 >= image_margin
        and x2 <= float(image_w) - image_margin
        and y2 <= float(image_h) - image_margin
    )
    parent_edge_ok = bool(
        x1 >= px1 + parent_margin
        and y1 >= py1 + parent_margin
        and x2 <= px2 - parent_margin
        and y2 <= py2 - parent_margin
    )
    edge_ok = bool(image_edge_ok and parent_edge_ok)
    hard_geometry_ready = bool(area_ok and hard_distance_ok and hard_height_ok and edge_ok)
    min_place_score = env_float("ROBOT_DEPTH_SURFACE_PLACE_NOW_MIN_SCORE", 0.48)
    distance_in_context = bool(
        depth_m <= max_context_depth
        and min_context_ground <= ground_distance_m <= max_context_ground
    )
    distance_place_ready = bool(
        depth_m <= max_place_now_depth
        and min_place_now_ground <= ground_distance_m <= max_place_now_ground
    )
    usable_surface = bool((not blocked) and score >= min_place_score and distance_in_context and hard_geometry_ready)
    place_now = bool(usable_surface and distance_place_ready and abs(bearing_deg) <= center_tolerance_deg)
    rejection_reasons: List[str] = []
    if not area_ok:
        rejection_reasons.append("too_small")
    if ground_distance_m < hard_min_ground:
        rejection_reasons.append("too_close")
    elif ground_distance_m > hard_max_ground:
        rejection_reasons.append("too_far")
    if not hard_height_ok:
        rejection_reasons.append("height_out_of_range")
    if not image_edge_ok:
        rejection_reasons.append("touches_image_edge")
    if not parent_edge_ok:
        rejection_reasons.append("touches_parent_edge")
    if blocked:
        rejection_reasons.append("blocked_surface")
    if score < min_place_score:
        rejection_reasons.append("surface_score_too_low")
    if ground_distance_m < min_context_ground and "too_close" not in rejection_reasons:
        rejection_reasons.append("too_close_for_surface_context")
    elif ground_distance_m > max_context_ground and "too_far" not in rejection_reasons:
        rejection_reasons.append("too_far_for_surface_context")
    if depth_m > max_context_depth:
        rejection_reasons.append("depth_too_far_for_surface_context")
    if not distance_place_ready and "too_close" not in rejection_reasons and "too_far" not in rejection_reasons:
        rejection_reasons.append("outside_place_distance_range")
    if abs(bearing_deg) > center_tolerance_deg:
        rejection_reasons.append("needs_alignment")
    reject_reason = rejection_reasons[0] if rejection_reasons else None

    parent_label = str(parent.get("label") or parent.get("raw_label") or "receptacle")
    bbox_dict = {"x": int(round(x1)), "y": int(round(y1)), "w": int(round(width)), "h": int(round(height))}
    candidate_id = surface_region_id(parent_label, bbox_dict, height_m, ground_distance_m)
    geometry_checks = {
        "area_ok": bool(area_ok),
        "distance_ok": bool(hard_distance_ok),
        "height_ok": bool(hard_height_ok),
        "edge_ok": bool(edge_ok),
        "depth_stable": True,
        "normal_like_horizontal": True,
        "region_w": round(float(width), 3),
        "region_h": round(float(height), 3),
        "image_edge_ok": bool(image_edge_ok),
        "parent_edge_ok": bool(parent_edge_ok),
    }
    candidate: JsonDict = {
        "id": candidate_id,
        "surface_candidate_id": candidate_id,
        "label": parent.get("label"),
        "raw_label": parent.get("raw_label") or parent.get("label"),
        "task_semantic_class": "place_receptacle",
        "confidence": round(float(parent.get("confidence", 0.0) or 0.0) * max(0.35, min(1.0, score)), 4),
        "bbox": bbox_dict,
        "region_bbox": dict(bbox_dict),
        "image_size": {"w": int(image_w), "h": int(image_h)},
        "center": {"x": int(round(center_x)), "y": int(round(center_y))},
        "interaction_point": {"x": round(float(center_x), 3), "y": round(float(center_y), 3)},
        "position_hint": position_hint,
        "surface_hint": "support_surface",
        "is_floor_level": False,
        "is_support_surface": True,
        "reachable": bool(usable_surface),
        "pickup_now": False,
        "place_now": bool(place_now),
        "visual_place_ready": bool(place_now),
        "final_place_ready": bool(place_now),
        "cleanable_now": False,
        "needs_alignment": bool(usable_surface and not place_now and abs(bearing_deg) > center_tolerance_deg),
        "needs_approach": bool(
            usable_surface
            and not place_now
            and abs(bearing_deg) <= center_tolerance_deg
            and ground_distance_m > max_place_now_ground
        ),
        "obstacle_risk": False,
        "visual_box_ambiguous": False,
        "broad_front_receptacle": True,
        "front_edge_receptacle": True,
        "area": round(float(width * height), 2),
        "area_ratio": round(float(area_ratio), 6),
        "region_area_px": int(round(width * height)),
        "region_area_ratio": round(float(area_ratio), 6),
        "center_y_ratio": round(float(cy_ratio), 3),
        "bottom_y_ratio": round(float(bottom_y_ratio), 3),
        "height_m": round(float(height_m), 4) if height_m is not None else center_3d.get("y"),
        "distance_m": round(float(ground_distance_m), 4),
        "bearing_deg": round(float(bearing_deg), 3),
        "geometry": {
            "cx_ratio": round(cx_ratio, 4),
            "cy_ratio": round(cy_ratio, 4),
            "bottom_y_ratio": round(bottom_y_ratio, 4),
            "area_ratio": round(area_ratio, 6),
            "width_ratio": round(width / max(1.0, float(image_w)), 6),
            "bearing_deg": round(float(bearing_deg), 3),
            "distance_m": round(float(depth_m), 4),
            "ground_forward_m": round(float(ground_forward_m), 4),
            "ground_distance_m": round(float(ground_distance_m), 4),
            "height_m": round(float(height_m), 4) if height_m is not None else center_3d.get("y"),
        },
        "region_type": "blocked_surface" if blocked else "placeable_surface_region",
        "affordance": [] if blocked else ["place"],
        "parent_object": normalize_label(parent_label),
        "parent_label": parent_label,
        "parent_bbox": parent.get("bbox"),
        "source": "depth_geometry",
        "surface_candidate_source": "depth_geometry",
        "actionability_source": "depth_geometry",
        "score": round(float(score), 4),
        "geometry_checks": geometry_checks,
        "occupancy_checks": {
            "blocked": bool(blocked),
            "blocked_by": blocked_by,
        },
        "memory_checks": {
            "failed_recently": False,
            "cooldown_remaining": 0,
        },
        "executor_checks": {
            "precheck_supported": False,
            "precheck_ok": bool(place_now),
            "reason": "legacy_depth_patch_no_precheck",
            "suggested_recovery": None,
        },
        "blocked": bool(blocked),
        "blocked_by": blocked_by,
        "rejection_reasons": rejection_reasons,
        "depth": {
            "unit": "meter",
            "median_m": round(float(depth_m), 4),
            "distance_m": round(float(depth_m), 4),
            "ground_forward_m": round(float(ground_forward_m), 4),
            "ground_distance_m": round(float(ground_distance_m), 4),
        },
        "center_3d": center_3d,
        "height": round(float(height_m), 4) if height_m is not None else center_3d.get("y"),
        "distance": round(float(depth_m), 4),
        "ground_distance": round(float(ground_distance_m), 4),
    }
    if reject_reason:
        candidate["actionability_reject_reason"] = reject_reason
    return candidate


def generate_surface_candidates_for_receptacle(
    *,
    parent: JsonDict,
    depth_frame: Any,
    camera: JsonDict,
    all_candidates: List[JsonDict],
    image_w: int,
    image_h: int,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
) -> List[JsonDict]:
    import numpy as np  # type: ignore

    parent_box = bbox_pixel_tuple(parent)
    if parent_box is None:
        return []
    label = normalize_label(parent.get("label") or parent.get("raw_label") or "")
    if label not in SURFACE_PARENT_LABELS:
        return []
    if label in {"sink", "stove"}:
        return []

    x1, y1, x2, y2 = parent_box
    x1_i = max(0, min(int(round(x1)), image_w - 1))
    x2_i = max(0, min(int(round(x2)), image_w))
    y1_i = max(0, min(int(round(y1)), image_h - 1))
    y2_i = max(0, min(int(round(y2)), image_h))
    if x2_i <= x1_i or y2_i <= y1_i:
        return []

    crop = depth_frame[y1_i:y2_i, x1_i:x2_i]
    valid = depth_valid_mask(crop)
    if int(valid.sum()) < env_int("ROBOT_DEPTH_SURFACE_MIN_VALID_PIXELS", 80):
        return []

    cell = max(8, env_int("ROBOT_DEPTH_SURFACE_CELL_PIXELS", 28))
    min_valid_ratio = env_float("ROBOT_DEPTH_SURFACE_CELL_MIN_VALID_RATIO", 0.55)
    max_iqr = env_float("ROBOT_DEPTH_SURFACE_MAX_IQR_M", 0.18)
    min_height = env_float("ROBOT_DEPTH_SURFACE_MIN_HEIGHT_M", 0.35)
    max_height = env_float("ROBOT_DEPTH_SURFACE_MAX_HEIGHT_M", 1.25)
    support_min_height = env_float("ROBOT_DEPTH_SURFACE_REGION_MIN_HEIGHT_M", 0.55)
    support_max_height = env_float("ROBOT_DEPTH_SURFACE_REGION_MAX_HEIGHT_M", 1.15)
    support_band_width = env_float("ROBOT_DEPTH_SURFACE_HEIGHT_BAND_WIDTH_M", 0.08)
    support_band_tolerance = env_float("ROBOT_DEPTH_SURFACE_HEIGHT_BAND_TOLERANCE_M", 0.07)
    ideal_height = env_float("ROBOT_DEPTH_SURFACE_IDEAL_HEIGHT_M", 0.85)
    max_distance = env_float("ROBOT_DEPTH_SURFACE_MAX_DISTANCE_M", 2.4)
    min_ground_distance = env_float("ROBOT_DEPTH_SURFACE_MIN_GROUND_DISTANCE_M", 0.35)
    max_ground_distance = env_float(
        "ROBOT_DEPTH_SURFACE_MAX_GROUND_DISTANCE_M",
        env_float("ROBOT_DEPTH_SURFACE_CONTEXT_MAX_GROUND_DISTANCE", 2.6),
    )
    ideal_ground_distance = env_float("ROBOT_DEPTH_SURFACE_IDEAL_GROUND_DISTANCE_M", 0.75)
    ground_distance_score_span = env_float("ROBOT_DEPTH_SURFACE_GROUND_DISTANCE_SCORE_SPAN_M", 0.65)
    blockers = surface_blockers_for(
        parent,
        all_candidates,
        holding_object=holding_object,
        held_object_labels=held_object_labels,
        held_object_family=held_object_family,
    )
    patches: List[JsonDict] = []

    for yy in range(y1_i, y2_i, cell):
        for xx in range(x1_i, x2_i, cell):
            yy2 = min(yy + cell, y2_i)
            xx2 = min(xx + cell, x2_i)
            cell_depth = depth_frame[yy:yy2, xx:xx2]
            cell_valid = depth_valid_mask(cell_depth)
            total = int(cell_depth.size)
            valid_count = int(cell_valid.sum())
            if total <= 0 or valid_count / max(1, total) < min_valid_ratio:
                continue
            values = cell_depth[cell_valid]
            median_depth = float(np.median(values))
            if median_depth > max_distance:
                continue
            q10 = float(np.percentile(values, 10))
            q90 = float(np.percentile(values, 90))
            iqr = q90 - q10
            if iqr > max_iqr:
                continue

            cx = xx + min(cell, x2_i - xx) / 2.0
            cy = yy + min(cell, y2_i - yy) / 2.0
            center_3d = project_pixel_to_3d(cx, cy, median_depth, camera)
            ground_distance_m = depth_number(center_3d.get("ground_distance_m"))
            if ground_distance_m is not None and not (min_ground_distance <= ground_distance_m <= max_ground_distance):
                continue
            try:
                height_m = float(center_3d.get("y"))
            except (TypeError, ValueError):
                height_m = None
            if height_m is not None and not (min_height <= height_m <= max_height):
                continue
            height_samples: List[float] = []
            sample_step = max(1, int(max(1, min(yy2 - yy, xx2 - xx)) / 3))
            for sample_y in range(yy, yy2, sample_step):
                for sample_x in range(xx, xx2, sample_step):
                    sample_depth = float(depth_frame[sample_y, sample_x])
                    if not math.isfinite(sample_depth) or sample_depth <= 0.05 or sample_depth >= 20.0:
                        continue
                    sample_3d = project_pixel_to_3d(float(sample_x), float(sample_y), sample_depth, camera)
                    sample_height = depth_number(sample_3d.get("y"))
                    if sample_height is not None:
                        height_samples.append(float(sample_height))
            height_std = float(np.std(height_samples)) if len(height_samples) > 1 else 0.0

            patches.append(
                {
                    "bbox": (float(xx), float(yy), float(xx2), float(yy2)),
                    "grid": (int((yy - y1_i) // cell), int((xx - x1_i) // cell)),
                    "center": (float(cx), float(cy)),
                    "center_3d": center_3d,
                    "valid_ratio": valid_count / max(1, total),
                    "median_depth": float(median_depth),
                    "q10": float(q10),
                    "q90": float(q90),
                    "depth_range": float(iqr),
                    "iqr": float(iqr),
                    "height_m": height_m,
                    "height_std": height_std,
                    "ground_distance_m": ground_distance_m,
                }
            )

    if not patches:
        return []

    support_height_patches = [
        patch for patch in patches
        if patch.get("height_m") is not None
        and support_min_height <= float(patch.get("height_m")) <= support_max_height
    ]
    if not support_height_patches:
        return []

    band_width = max(0.01, float(support_band_width))
    bins: Dict[int, List[JsonDict]] = {}
    for patch in support_height_patches:
        height_value = float(patch.get("height_m"))
        band_index = int(math.floor((height_value - support_min_height) / band_width))
        bins.setdefault(band_index, []).append(patch)
    dominant_band = max(
        bins.values(),
        key=lambda items: (len(items), -float(np.std([float(item.get("height_m")) for item in items]))),
    )
    dominant_height = float(np.median([float(patch.get("height_m")) for patch in dominant_band]))
    band_tolerance = max(0.01, float(support_band_tolerance))
    for patch in patches:
        height_value = depth_number(patch.get("height_m"))
        band_delta = abs(float(height_value) - dominant_height) if height_value is not None else float("inf")
        patch["dominant_support_height_m"] = dominant_height
        patch["support_height_band_delta_m"] = band_delta
        patch["support_height_band_ok"] = bool(
            height_value is not None
            and support_min_height <= float(height_value) <= support_max_height
            and band_delta <= band_tolerance
        )
    patches = [patch for patch in patches if patch.get("support_height_band_ok")]
    if not patches:
        return []

    by_grid = {tuple(patch["grid"]): index for index, patch in enumerate(patches)}
    visited: set[int] = set()
    clusters: List[List[JsonDict]] = []
    max_cluster_height_delta = env_float("ROBOT_DEPTH_SURFACE_CLUSTER_MAX_HEIGHT_DELTA_M", 0.06)
    max_cluster_depth_delta = env_float("ROBOT_DEPTH_SURFACE_CLUSTER_MAX_DEPTH_DELTA_M", 0.14)

    def patch_continuous(a: JsonDict, b: JsonDict) -> bool:
        a_h = a.get("height_m")
        b_h = b.get("height_m")
        if a_h is not None and b_h is not None and abs(float(a_h) - float(b_h)) > max_cluster_height_delta:
            return False
        return abs(float(a.get("median_depth", 0.0)) - float(b.get("median_depth", 0.0))) <= max_cluster_depth_delta

    for index, patch in enumerate(patches):
        if index in visited:
            continue
        cluster: List[JsonDict] = []
        queue = [index]
        visited.add(index)
        while queue:
            current_index = queue.pop(0)
            current = patches[current_index]
            cluster.append(current)
            gy, gx = current["grid"]
            for neighbor_key in ((gy - 1, gx), (gy + 1, gx), (gy, gx - 1), (gy, gx + 1)):
                neighbor_index = by_grid.get(neighbor_key)
                if neighbor_index is None or neighbor_index in visited:
                    continue
                neighbor = patches[neighbor_index]
                if not patch_continuous(current, neighbor):
                    continue
                visited.add(neighbor_index)
                queue.append(neighbor_index)
        clusters.append(cluster)

    candidates = [
        region
        for region in (
            make_surface_region_candidate(
                parent=parent,
                patches=cluster,
                blockers=blockers,
                image_w=image_w,
                image_h=image_h,
                camera=camera,
            )
            for cluster in clusters
        )
        if region
    ]
    candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    kept: List[JsonDict] = []
    min_sep = env_float("ROBOT_DEPTH_SURFACE_MIN_SEPARATION_PIXELS", 42.0)
    max_per_parent = max(1, env_int("ROBOT_DEPTH_SURFACE_MAX_PER_PARENT", 4))
    for candidate in candidates:
        center = candidate.get("center") if isinstance(candidate.get("center"), dict) else {}
        try:
            cx = float(center.get("x"))
            cy = float(center.get("y"))
        except (TypeError, ValueError):
            continue
        duplicate = False
        for existing in kept:
            existing_center = existing.get("center") if isinstance(existing.get("center"), dict) else {}
            try:
                ex = float(existing_center.get("x"))
                ey = float(existing_center.get("y"))
            except (TypeError, ValueError):
                continue
            if math.hypot(cx - ex, cy - ey) < min_sep:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= max_per_parent:
            break
    return kept


def apply_depth_geometry(
    candidates: List[JsonDict],
    *,
    depth_frame: Any,
    camera_info: Any,
    image_path: str = "",
    save_plane_vis: str = "",
    image_w: int,
    image_h: int,
    notes: List[str],
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
    diagnostics: Optional[JsonDict] = None,
) -> List[JsonDict]:
    if depth_frame is None:
        return []

    camera = parse_camera_info(camera_info, image_w=image_w, image_h=image_h)
    if camera.get("_parse_error"):
        notes.append(f"camera_json_parse_failed={camera.get('_parse_error')}")
    notes.append(
        "camera_used="
        + json.dumps(
            {
                "camera_height_m": camera.get("camera_height_m"),
                "camera_horizon_deg": camera.get("camera_horizon_deg"),
                "fx": camera.get("fx"),
                "fy": camera.get("fy"),
                "cx": camera.get("cx"),
                "cy": camera.get("cy"),
                "width": camera.get("width"),
                "height": camera.get("height"),
                "depth_convention": pickup_depth_convention(camera),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    held_labels = normalize_label_tokens(held_object_labels)
    held_family = normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels)
    generated: List[JsonDict] = []
    pointcloud_generated: List[JsonDict] = []
    patch_generated: List[JsonDict] = []
    original_candidates = list(candidates)
    floor_plane = estimate_local_floor_plane(depth_frame, camera, projector=project_pixel_to_3d)
    notes.append(
        "pickup_floor_plane="
        + json.dumps(
            {
                "available": floor_plane.get("available"),
                "reason": floor_plane.get("reason"),
                "inlier_count": floor_plane.get("inlier_count"),
                "residual_p90_m": floor_plane.get("residual_p90_m"),
                "normal_up_score": floor_plane.get("normal_up_score"),
                "depth_convention": floor_plane.get("depth_convention"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    for candidate in original_candidates:
        summary = candidate_depth_summary(candidate, depth_frame, camera, floor_plane=floor_plane)
        if summary is not None:
            candidate["depth"] = summary
            apply_depth_actionability(candidate)
    blocker_candidates, held_blockers_ignored = filter_held_object_blockers(
        original_candidates,
        holding_object=bool(holding_object),
        held_object_labels=held_labels,
        held_object_family=held_family,
    )

    def attach_surface_regions(surface_regions: List[JsonDict], *, context_reason: str) -> None:
        for candidate in original_candidates:
            if candidate.get("task_semantic_class") != "place_receptacle":
                continue
            matching = [
                region
                for region in surface_regions
                if isinstance(region, dict)
                and region.get("parent_bbox") == candidate.get("bbox")
            ]
            if not matching:
                continue
            candidate["context_only"] = True
            candidate["context_reason"] = context_reason
            candidate["place_now"] = False
            candidate["needs_alignment"] = False
            candidate["needs_approach"] = False
            existing_regions = candidate.get("surface_regions") if isinstance(candidate.get("surface_regions"), list) else []
            existing_candidates = (
                candidate.get("surface_candidates")
                if isinstance(candidate.get("surface_candidates"), list)
                else []
            )
            candidate["surface_regions"] = list(existing_regions) + matching
            candidate["surface_candidates"] = list(existing_candidates) + matching

    if env_bool("ROBOT_POINTCLOUD_SURFACE_ENABLED", True):
        try:
            script_dir = str(Path(__file__).resolve().parent)
            if script_dir not in sys.path:
                sys.path.append(script_dir)
            from surface_pointcloud import build_pointcloud_surface_output, depth_to_pointcloud  # type: ignore

            stride = max(1, env_int("ROBOT_POINTCLOUD_STRIDE", 2))
            pointcloud = depth_to_pointcloud(depth_frame, camera, stride=stride)
            pointcloud_output = build_pointcloud_surface_output(
                points=pointcloud,
                parents=original_candidates,
                blockers=original_candidates,
                image_w=int(image_w),
                image_h=int(image_h),
                camera=camera,
                holding_object=bool(holding_object),
                held_object_labels=held_labels,
                held_object_family=held_family,
                debug_image_path=str(image_path or ""),
                debug_plane_vis_path=str(save_plane_vis or ""),
            )
            raw_regions = pointcloud_output.get("regions") if isinstance(pointcloud_output, dict) else []
            pointcloud_generated = [region for region in raw_regions if isinstance(region, dict)]
            if pointcloud_generated:
                attach_surface_regions(
                    pointcloud_generated,
                    context_reason="pointcloud_surface_regions_generated",
                )
                generated.extend(pointcloud_generated)
            notes.append(f"pointcloud_surface_regions={len(pointcloud_generated)}")
            notes.append(
                f"pointcloud_surface_ready={sum(1 for region in pointcloud_generated if region.get('visual_place_ready'))}"
            )
            try:
                held_blockers_ignored = max(
                    int(held_blockers_ignored),
                    int(pointcloud_output.get("ignored_held_blocker_count", 0) or 0),
                )
            except (TypeError, ValueError):
                pass
            skipped = pointcloud_output.get("skipped_blockers") if isinstance(pointcloud_output, dict) else []
            if isinstance(skipped, list) and skipped:
                notes.append(f"skipped_blockers={json.dumps(skipped[:12], ensure_ascii=False, sort_keys=True)}")
            ignored_as_blocker = (
                pointcloud_output.get("held_object_ignored_as_blocker")
                if isinstance(pointcloud_output, dict)
                else []
            )
            if isinstance(ignored_as_blocker, list) and ignored_as_blocker:
                notes.append(
                    "held_object_ignored_as_blocker="
                    + json.dumps(ignored_as_blocker[:8], ensure_ascii=False, sort_keys=True)
                )
            point_filter = pointcloud_output.get("held_object_point_filter") if isinstance(pointcloud_output, dict) else []
            if isinstance(point_filter, list) and point_filter:
                notes.append(
                    "held_object_point_filter="
                    + json.dumps(point_filter[:8], ensure_ascii=False, sort_keys=True)
                )
            try:
                removed_points = int(pointcloud_output.get("held_object_points_removed", 0) or 0)
            except (TypeError, ValueError):
                removed_points = 0
            if removed_points:
                notes.append(f"held_object_points_removed={removed_points}")
            stats = pointcloud_output.get("stats") if isinstance(pointcloud_output, dict) else []
            notes.append(f"open3d_plane_segmentation_stats={json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
            notes.append(f"pointcloud_initial_plane_count={int(pointcloud_output.get('initial_plane_count', 0) or 0)}")
            if pointcloud_output.get("initial_plane_vis_path"):
                notes.append(f"pointcloud_initial_plane_vis={pointcloud_output.get('initial_plane_vis_path')}")
            if pointcloud_output.get("initial_plane_vis_error"):
                notes.append(f"pointcloud_initial_plane_vis_failed={pointcloud_output.get('initial_plane_vis_error')}")
            stage_vis_paths = pointcloud_output.get("stage_vis_paths") if isinstance(pointcloud_output, dict) else {}
            if isinstance(stage_vis_paths, dict):
                for stage_name, stage_path in stage_vis_paths.items():
                    notes.append(f"pointcloud_stage_vis_{stage_name}={stage_path}")
            stage_vis_errors = pointcloud_output.get("stage_vis_errors") if isinstance(pointcloud_output, dict) else {}
            if isinstance(stage_vis_errors, dict):
                for stage_name, stage_error in stage_vis_errors.items():
                    notes.append(f"pointcloud_stage_vis_{stage_name}_failed={stage_error}")
            grid_summary = pointcloud_output.get("free_space_grid_summary") if isinstance(pointcloud_output, dict) else {}
            if isinstance(grid_summary, dict):
                if isinstance(diagnostics, dict):
                    diagnostics["free_space_grid_summary"] = dict(grid_summary)
                notes.append(f"free_space_completion_mode={grid_summary.get('mode', 'plane_local_2d_grid')}")
                notes.append(f"free_space_grid_resolution_m={grid_summary.get('grid_resolution_m', 0.0)}")
                notes.append(f"free_space_grid_status={grid_summary.get('status', 'not_attempted')}")
                notes.append(
                    f"free_space_grid_attempted_region_count={int(grid_summary.get('attempted_region_count', 0) or 0)}"
                )
                notes.append(
                    f"free_space_grid_successful_region_count={int(grid_summary.get('successful_region_count', 0) or 0)}"
                )
                notes.append(
                    "free_space_grid_candidate_trigger_counts="
                    + json.dumps(grid_summary.get("candidate_trigger_counts", {}), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    "free_space_grid_attempt_result_counts="
                    + json.dumps(grid_summary.get("attempt_result_counts", {}), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    f"free_space_grid_suggested_recovery={grid_summary.get('suggested_recovery') or 'none'}"
                )
                notes.append(
                    "free_space_ready_distance_range_m="
                    + json.dumps(grid_summary.get("ready_distance_range_m", []), ensure_ascii=False)
                )
                notes.append(f"free_space_raw_component_count={int(grid_summary.get('raw_component_count', 0) or 0)}")
                notes.append(f"free_space_component_count={int(grid_summary.get('component_count', 0) or 0)}")
                notes.append(f"free_space_component_kept={int(grid_summary.get('component_kept', 0) or 0)}")
                notes.append(f"free_space_raw_free_cell_count={int(grid_summary.get('raw_free_cell_count', 0) or 0)}")
                notes.append(f"free_space_reachable_free_cell_count={int(grid_summary.get('reachable_free_cell_count', 0) or 0)}")
                notes.append(
                    f"free_space_reachability_filtered_cell_count={int(grid_summary.get('reachability_filtered_cell_count', 0) or 0)}"
                )
                notes.append(f"free_space_near_free_cell_count={int(grid_summary.get('near_free_cell_count', 0) or 0)}")
                notes.append(f"free_space_far_free_cell_count={int(grid_summary.get('far_free_cell_count', 0) or 0)}")
                notes.append(
                    f"free_space_approachable_component_count={int(grid_summary.get('approachable_component_count', 0) or 0)}"
                )
                notes.append(
                    f"free_space_has_approachable_free_space={str(bool(grid_summary.get('has_approachable_free_space'))).lower()}"
                )
                notes.append(
                    "free_space_raw_component_rejection_counts="
                    + json.dumps(grid_summary.get("raw_component_rejection_counts", {}), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    "free_space_raw_component_rejections="
                    + json.dumps(grid_summary.get("raw_component_rejections", []), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    f"free_space_far_placeable_component_count={int(grid_summary.get('far_placeable_component_count', 0) or 0)}"
                )
                notes.append(
                    f"free_space_has_far_placeable_free_space={str(bool(grid_summary.get('has_far_placeable_free_space'))).lower()}"
                )
                notes.append(
                    "free_space_far_component_rejection_counts="
                    + json.dumps(grid_summary.get("far_component_rejection_counts", {}), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    "free_space_far_component_rejections="
                    + json.dumps(grid_summary.get("far_component_rejections", []), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    "free_space_raw_free_distance_range_m="
                    + json.dumps(grid_summary.get("raw_free_distance_range_m"), ensure_ascii=False)
                )
                notes.append(
                    "free_space_reachable_free_distance_range_m="
                    + json.dumps(grid_summary.get("reachable_free_distance_range_m"), ensure_ascii=False)
                )
                notes.append(
                    "free_space_component_rejection_counts="
                    + json.dumps(grid_summary.get("component_rejection_counts", {}), ensure_ascii=False, sort_keys=True)
                )
                notes.append(
                    "free_space_component_rejections="
                    + json.dumps(grid_summary.get("component_rejections", []), ensure_ascii=False, sort_keys=True)
                )
                notes.append(f"occupancy_mask_blocker_count={int(grid_summary.get('occupancy_mask_blocker_count', 0) or 0)}")
                notes.append(
                    "free_space_occupancy_source_counts="
                    + json.dumps(grid_summary.get("occupancy_source_counts", {}), ensure_ascii=False, sort_keys=True)
                )
                notes.append(f"edge_margin_m={grid_summary.get('edge_margin_m', 0.0)}")
                notes.append(f"blocker_dilate_m={grid_summary.get('blocker_dilate_m', 0.0)}")
            notes.append(f"held_footprint_radius_m={grid_summary.get('held_footprint_radius_m', 0.0)}")
            notes.append(f"placement_clearance_m={grid_summary.get('placement_clearance_m', 0.0)}")
            notes.append(
                f"free_space_hard_image_exclusion_box_count={grid_summary.get('hard_image_exclusion_box_count', 0)}"
            )
            notes.append(
                f"free_space_mapped_point_hard_rejection_count={grid_summary.get('mapped_point_hard_rejection_count', 0)}"
            )
            dbscan_stats = []
            if isinstance(stats, list):
                for item in stats:
                    if isinstance(item, dict):
                        dbscan_stats.extend(item.get("dbscan_clusters") or [])
            notes.append(f"dbscan_cluster_stats={json.dumps(dbscan_stats, ensure_ascii=False, sort_keys=True)}")
        except ModuleNotFoundError as exc:
            notes.append(f"pointcloud_surface=disabled:dependency_missing:{exc}")
        except Exception as exc:
            notes.append(f"pointcloud_surface=disabled:failed:{exc}")
    else:
        notes.append("pointcloud_surface=disabled:env")

    patch_debug = env_bool("ROBOT_PATCH_SURFACE_DEBUG_INCLUDE", False)
    patch_fallback_enabled = env_bool("ROBOT_PATCH_SURFACE_FALLBACK_ENABLED", True)
    run_patch_surface = bool(patch_debug or (patch_fallback_enabled and not pointcloud_generated))
    if run_patch_surface:
        notes.append("patch_surface_mode=debug" if patch_debug else "patch_surface_mode=fallback")
        for candidate in original_candidates:
            if candidate.get("task_semantic_class") != "place_receptacle":
                continue
            surface_regions = generate_surface_candidates_for_receptacle(
                parent=candidate,
                depth_frame=depth_frame,
                camera=camera,
                all_candidates=blocker_candidates,
                image_w=image_w,
                image_h=image_h,
                holding_object=bool(holding_object),
                held_object_labels=held_labels,
                held_object_family=held_family,
            )
            if not surface_regions:
                continue
            patch_generated.extend(surface_regions)
        if patch_generated:
            attach_surface_regions(
                patch_generated,
                context_reason="patch_depth_surface_regions_generated",
            )
            generated.extend(patch_generated)
    else:
        notes.append("patch_surface_mode=skipped_pointcloud_available")

    if generated:
        candidates.extend(generated)
    notes.append(f"depth_surface_regions={len(patch_generated)}")
    notes.append(f"surface_candidates_total={len(generated)}")
    notes.append(f"held_object_blockers_ignored={held_blockers_ignored}")
    return generated


def horizontal_overlap_ratio(a: JsonDict, b: JsonDict) -> float:
    abox = bbox_pixel_tuple(a)
    bbox = bbox_pixel_tuple(b)
    if abox is None or bbox is None:
        return 0.0
    ax1, _, ax2, _ = abox
    bx1, _, bx2, _ = bbox
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1.0, min(ax2 - ax1, bx2 - bx1))
    return overlap / denom


def bbox_min_overlap_ratio(a: JsonDict, b: JsonDict) -> float:
    abox = bbox_pixel_tuple(a)
    bbox = bbox_pixel_tuple(b)
    if abox is None or bbox is None:
        return 0.0
    ax1, ay1, ax2, ay2 = abox
    bx1, by1, bx2, by2 = bbox
    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = overlap_w * overlap_h
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return overlap / max(1.0, min(area_a, area_b))


def placement_obstacle_candidate(candidate: JsonDict) -> bool:
    task_class = str(candidate.get("task_semantic_class") or "")
    label = normalize_label(candidate.get("label") or candidate.get("raw_label") or "")
    placement_blocking_labels = {
        "apple",
        "banana",
        "book",
        "bowl",
        "cup",
        "kettle",
        "lettuce",
        "mug",
        "pan",
        "plate",
        "pot",
        "remote",
        "remote_control",
        "sink",
        "stove",
        "tomato",
        "vase",
    }
    if task_class == "pickup_target":
        return True
    if task_class == "ignored_object":
        return True
    if label in placement_blocking_labels:
        return True
    return bool(task_class == "place_receptacle" and label in SURFACE_BLOCKING_LABELS)


def short_avoidance_candidate(candidate: JsonDict) -> JsonDict:
    item: JsonDict = {
        "label": candidate.get("label"),
        "raw_label": candidate.get("raw_label"),
        "task_semantic_class": candidate.get("task_semantic_class"),
        "confidence": candidate.get("confidence"),
    }
    for key in ("bbox", "center", "geometry", "image_size"):
        value = candidate.get(key)
        if isinstance(value, dict):
            item[key] = dict(value)
    if candidate.get("likely_held_object_overlay"):
        item["likely_held_object_overlay"] = True
    if isinstance(candidate.get("held_object_overlay_checks"), dict):
        item["held_object_overlay_checks"] = dict(candidate.get("held_object_overlay_checks") or {})
    return item


def build_placement_avoidance_candidates(
    candidates: List[JsonDict],
    *,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
) -> List[JsonDict]:
    avoidance: List[JsonDict] = []
    held_labels = normalize_label_tokens(held_object_labels)
    for candidate in candidates:
        if (
            holding_object
            and env_bool("ROBOT_HELD_OBJECT_BLOCKER_IGNORE_ENABLED", True)
            and candidate_likely_held_object(candidate, held_labels, held_object_family)
        ):
            candidate["held_object_candidate"] = True
            candidate["ignored_as_held_object_blocker"] = True
            continue
        if not placement_obstacle_candidate(candidate):
            continue
        if bbox_pixel_tuple(candidate) is None:
            continue
        try:
            area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            area_ratio = 0.0
        if area_ratio > 0.08:
            continue
        avoidance.append(short_avoidance_candidate(candidate))
    avoidance.sort(key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
    return avoidance[:16]


def save_candidate_visualization(
    *,
    image_path: Path,
    save_path: Path,
    candidates: List[JsonDict],
    notes: List[str],
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        notes.append(f"save_vis_failed:cv2_unavailable:{exc}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        notes.append(f"save_vis_failed:image_unreadable:{image_path}")
        return

    colors = {
        "pickup_target": (0, 220, 0),
        "place_receptacle": (255, 220, 0),
        "cleanable_object": (0, 220, 255),
        "obstacle": (0, 140, 255),
        "ignored_object": (180, 180, 180),
    }

    held_overlay_boxes = [
        box
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("likely_held_object_overlay")
        for box in [bbox_pixel_tuple(candidate)]
        if box is not None
    ]
    ready_surface_boxes = [
        box
        for candidate in candidates
        if isinstance(candidate, dict)
        and str(candidate.get("surface_candidate_source") or "") in SURFACE_REGION_SOURCES
        and candidate.get("visual_place_ready")
        for box in [bbox_pixel_tuple(candidate)]
        if box is not None
    ]
    hidden_held = 0
    hidden_surface_overlap = 0
    hidden_ready_duplicate = 0

    def draw_order(candidate: JsonDict) -> int:
        source = str(candidate.get("surface_candidate_source") or "")
        is_surface = source in SURFACE_REGION_SOURCES
        if is_surface and not candidate.get("visual_place_ready"):
            return 0
        if not is_surface:
            return 1
        return 2

    for candidate in sorted(candidates, key=draw_order):
        box = bbox_pixel_tuple(candidate)
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        task_class = str(candidate.get("task_semantic_class") or "ignored_object")
        surface_source = str(candidate.get("surface_candidate_source") or "")
        is_surface_region = surface_source in SURFACE_REGION_SOURCES
        if candidate.get("likely_held_object_overlay") and not env_bool("ROBOT_SAVE_VIS_DRAW_HELD_OVERLAY", False):
            hidden_held += 1
            continue
        if (
            is_surface_region
            and not candidate.get("visual_place_ready")
            and ready_surface_boxes
            and any(
                bbox_overlap_area(box, ready_box) / max(1.0, (x2 - x1) * (y2 - y1)) > 0.80
                for ready_box in ready_surface_boxes
            )
            and env_bool("ROBOT_SAVE_VIS_HIDE_REJECTED_READY_DUPLICATES", True)
        ):
            hidden_ready_duplicate += 1
            continue
        if (
            is_surface_region
            and held_overlay_boxes
            and not candidate.get("visual_place_ready")
            and env_bool("ROBOT_SAVE_VIS_HIDE_REJECTED_SURFACE_HELD_OVERLAP", True)
            and any(bbox_overlap_area(box, held_box) > 0 for held_box in held_overlay_boxes)
        ):
            hidden_surface_overlap += 1
            continue
        if is_surface_region:
            executor_checks = candidate.get("executor_checks") if isinstance(candidate.get("executor_checks"), dict) else {}
            precheck_failed = bool(
                executor_checks.get("precheck_supported")
                and executor_checks.get("reason") not in {None, "", "precheck_not_run"}
                and not executor_checks.get("precheck_ok")
            )
            memory_checks = candidate.get("memory_checks") if isinstance(candidate.get("memory_checks"), dict) else {}
            failed_recently = bool(candidate.get("failed_recently") or memory_checks.get("failed_recently"))
            rejected = bool(candidate.get("rejection_reasons") or candidate.get("blocked") or failed_recently)
            if precheck_failed:
                color = (180, 0, 180)
            elif candidate.get("final_place_ready") or candidate.get("place_now"):
                color = (0, 220, 0)
            elif rejected:
                color = (0, 0, 255)
            elif candidate.get("visual_place_ready") and candidate.get("affordance_ready", True):
                color = (0, 220, 0)
            elif candidate.get("visual_place_ready"):
                color = (255, 120, 0)
            else:
                color = (255, 120, 0)
        else:
            color = colors.get(task_class, (220, 220, 220))
        thickness = 3 if is_surface_region and candidate.get("visual_place_ready") else 2
        grid_ready_surface = bool(
            surface_source == POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE
            and candidate.get("visual_place_ready")
        )
        if not grid_ready_surface:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if grid_ready_surface:
            interaction = candidate.get("interaction_point") if isinstance(candidate.get("interaction_point"), dict) else {}
            try:
                point_x = int(round(float(interaction.get("x"))))
                point_y = int(round(float(interaction.get("y"))))
                cv2.drawMarker(image, (point_x, point_y), color, cv2.MARKER_CROSS, 18, 2)
                cv2.circle(image, (point_x, point_y), 6, color, 2)
            except (TypeError, ValueError):
                pass

        label = str(candidate.get("raw_label") or candidate.get("label") or task_class)
        if surface_source in {POINTCLOUD_SURFACE_SOURCE, POINTCLOUD_COMPLETION_SURFACE_SOURCE, POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE}:
            label = "PLACE" if candidate.get("visual_place_ready") else "pc_surface"
            if surface_source == POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE and candidate.get("visual_place_ready"):
                completion = candidate.get("free_space_completion") if isinstance(candidate.get("free_space_completion"), dict) else {}
                label = f"PLACE G{int(completion.get('component_id', 0) or 0)}"
        try:
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if is_surface_region:
            reasons = [str(item) for item in candidate.get("rejection_reasons", []) if str(item)]
            memory_checks = candidate.get("memory_checks") if isinstance(candidate.get("memory_checks"), dict) else {}
            cooldown_remaining = int(candidate.get("cooldown_remaining") or memory_checks.get("cooldown_remaining") or 0)
            if candidate.get("failed_recently") or memory_checks.get("failed_recently"):
                reasons.insert(0, f"cooldown:{cooldown_remaining}")
            if candidate.get("blocked") and not any(item.startswith("blocked") for item in reasons):
                blocked_by = candidate.get("blocked_by") if isinstance(candidate.get("blocked_by"), list) else []
                if blocked_by:
                    reasons.insert(0, f"blocked:{blocked_by[0]}")
                else:
                    reasons.insert(0, "blocked")
            executor_checks = candidate.get("executor_checks") if isinstance(candidate.get("executor_checks"), dict) else {}
            precheck_failed = bool(
                executor_checks.get("precheck_supported")
                and executor_checks.get("reason") not in {None, "", "precheck_not_run"}
                and not executor_checks.get("precheck_ok")
            )
            if precheck_failed:
                status = "precheck_failed"
                if executor_checks.get("reason"):
                    reasons.insert(0, str(executor_checks.get("reason")))
            elif candidate.get("final_place_ready") or candidate.get("place_now"):
                status = "ready"
            elif candidate.get("visual_place_ready") and candidate.get("affordance_ready", True):
                status = "ready"
            elif candidate.get("visual_place_ready"):
                status = "visual"
            else:
                status = "reject"
            reason_text = ",".join(reasons[:2]) if reasons else status
            try:
                surface_score = float(candidate.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                surface_score = 0.0
            text = f"{label} {status} s={surface_score:.2f} {reason_text}"
            if surface_source in {POINTCLOUD_SURFACE_SOURCE, POINTCLOUD_COMPLETION_SURFACE_SOURCE, POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE}:
                plane_id = candidate.get("plane_id")
                cluster_id = candidate.get("dbscan_cluster_id")
                try:
                    up_score = float(candidate.get("normal_up_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    up_score = 0.0
                if candidate.get("visual_place_ready"):
                    text = f"{label} {status} s={surface_score:.2f}"
                else:
                    text = f"{label} p={plane_id} c={cluster_id} up={up_score:.2f} s={surface_score:.2f} {reason_text}"
        else:
            text = f"{label} {confidence:.2f}"
        text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_y = max(14, y1 - 4)
        cv2.rectangle(image, (x1, text_y - text_h - 4), (x1 + text_w + 4, text_y + 3), color, -1)
        cv2.putText(
            image,
            text,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), image):
            notes.append(f"save_vis_failed:write_failed:{save_path}")
            return
        notes.append(f"save_vis_mode=reported_candidates:{len(candidates)}")
        if hidden_held:
            notes.append(f"save_vis_held_overlay_hidden={hidden_held}")
        if hidden_surface_overlap:
            notes.append(f"save_vis_held_overlap_surfaces_hidden={hidden_surface_overlap}")
        if hidden_ready_duplicate:
            notes.append(f"save_vis_ready_duplicate_rejected_hidden={hidden_ready_duplicate}")
    except Exception as exc:
        notes.append(f"save_vis_failed:{exc}")


def decorate_receptacle_occupancy_context(
    candidates: List[JsonDict],
    *,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[str]] = None,
    held_object_family: Optional[str] = None,
) -> None:
    avoidance = build_placement_avoidance_candidates(
        candidates,
        holding_object=holding_object,
        held_object_labels=held_object_labels,
        held_object_family=held_object_family,
    )
    for receptacle in [c for c in candidates if c.get("task_semantic_class") == "place_receptacle"]:
        rx1, ry1, rx2, ry2 = bbox_ratios(receptacle)
        support_label = normalize_label(receptacle.get("label") or receptacle.get("raw_label") or "")
        shelf_like = support_label in {"shelf", "shelving_unit", "shelvingunit"}
        occupants: List[JsonDict] = []
        for obstacle in avoidance:
            center = obstacle.get("center") if isinstance(obstacle.get("center"), dict) else {}
            image_size = obstacle.get("image_size") if isinstance(obstacle.get("image_size"), dict) else {}
            geometry = obstacle.get("geometry") if isinstance(obstacle.get("geometry"), dict) else {}
            try:
                iw = float(image_size.get("w", 600) or 600)
                ih = float(image_size.get("h", 600) or 600)
                cx = float(center.get("x")) / max(1.0, iw)
                cy = float(center.get("y")) / max(1.0, ih)
                bottom = float(geometry.get("bottom_y_ratio", cy) or cy)
            except (TypeError, ValueError):
                continue
            inside_x = rx1 - 0.035 <= cx <= rx2 + 0.035
            inside_y = ry1 - 0.04 <= cy <= ry2 + 0.065
            if not (inside_x and inside_y):
                continue
            if not shelf_like:
                relative_bottom = support_relative_y(bottom, ry1, ry2)
                # Broad countertop/table boxes can include the floor in the lower image.
                # Only objects whose lower edge sits in the support-plane band should
                # be treated as occupying that receptacle.
                if bottom > ry2 + 0.055 or relative_bottom > 0.82:
                    continue
            occupants.append(obstacle)
        if occupants:
            receptacle["visible_occupants"] = occupants[:8]
            receptacle["visible_occupant_count"] = len(occupants)


def active_receptacle_candidates(candidates: List[JsonDict]) -> List[JsonDict]:
    suppressed = set()
    candidates = [candidate for candidate in candidates if not candidate.get("context_only")]

    def keep_score(candidate: JsonDict) -> Tuple[int, int, int, float, float]:
        try:
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            area = float(candidate.get("area_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            area = 0.0
        return (
            1 if candidate.get("place_now") else 0,
            1 if candidate.get("front_edge_receptacle") or candidate.get("broad_front_receptacle") else 0,
            0 if candidate.get("visual_box_ambiguous") else 1,
            confidence,
            area,
        )

    tabletop_labels = {
        "counter_top",
        "countertop",
        "dining_table",
        "diningtable",
        "coffee_table",
        "coffeetable",
        "side_table",
        "sidetable",
    }
    for index, candidate in enumerate(candidates):
        if index in suppressed:
            continue
        label = normalize_label(candidate.get("label") or candidate.get("raw_label") or "")
        if label not in tabletop_labels:
            continue
        for other_index, other in enumerate(candidates):
            if index == other_index or other_index in suppressed:
                continue
            other_label = normalize_label(other.get("label") or other.get("raw_label") or "")
            if other_label != label:
                continue
            if bbox_min_overlap_ratio(candidate, other) < 0.88:
                continue
            loser_index, winner = (
                (index, other)
                if keep_score(other) > keep_score(candidate)
                else (other_index, candidate)
            )
            if loser_index == index:
                candidate["suppressed_by_receptacle_merge"] = True
                candidate["merged_with_receptacle"] = {
                    "label": winner.get("label"),
                    "raw_label": winner.get("raw_label"),
                    "reason": "same_support_duplicate_candidate",
                }
                suppressed.add(index)
                break
            other["suppressed_by_receptacle_merge"] = True
            other["merged_with_receptacle"] = {
                "label": candidate.get("label"),
                "raw_label": candidate.get("raw_label"),
                "reason": "same_support_duplicate_candidate",
            }
            suppressed.add(other_index)
        if index in suppressed:
            continue
        if candidate.get("place_now") or candidate.get("front_edge_receptacle"):
            continue
        try:
            candidate_bottom = float(
                (candidate.get("geometry") or {}).get("bottom_y_ratio", candidate.get("bottom_y_ratio", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            candidate_bottom = 0.0
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_label = normalize_label(other.get("label") or other.get("raw_label") or "")
            if other_label != label:
                continue
            if not (other.get("place_now") or other.get("front_edge_receptacle")):
                continue
            try:
                other_bottom = float(
                    (other.get("geometry") or {}).get("bottom_y_ratio", other.get("bottom_y_ratio", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                other_bottom = 0.0
            if other_bottom < candidate_bottom + 0.12:
                continue
            if horizontal_overlap_ratio(candidate, other) < 0.55:
                continue
            candidate["suppressed_by_receptacle_merge"] = True
            candidate["merged_with_receptacle"] = {
                "label": other.get("label"),
                "raw_label": other.get("raw_label"),
                "reason": "same_support_front_edge_candidate",
            }
            suppressed.add(index)
            break
    return [candidate for index, candidate in enumerate(candidates) if index not in suppressed]


def support_relative_y(value: float, sy1: float, sy2: float) -> float:
    height = max(0.001, sy2 - sy1)
    return (value - sy1) / height

#支撑面规则：避免把桌上的东西误判成地面可捡物
def pickup_floor_contact_like(candidate: JsonDict) -> bool:
    floor_contact = candidate.get("floor_contact_geometry") if isinstance(candidate.get("floor_contact_geometry"), dict) else {}
    if not bool(floor_contact.get("available") and floor_contact.get("contact_floor_like")):
        return False
    support_height = depth_number(floor_contact.get("support_height_m"))
    if support_height is not None and support_height > env_float("ROBOT_DEPTH_FLOOR_CONTACT_MAX_SUPPORT_HEIGHT_M", 0.16):
        return False
    return True


def pickup_support_height_m(candidate: JsonDict) -> Optional[float]:
    floor_contact = candidate.get("floor_contact_geometry") if isinstance(candidate.get("floor_contact_geometry"), dict) else {}
    return depth_number(floor_contact.get("support_height_m"))


def support_candidate_height_m(candidate: JsonDict) -> Optional[float]:
    point = candidate_center_3d(candidate)
    if isinstance(point, dict):
        value = depth_number(point.get("y"))
        if value is not None:
            return value
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    return depth_number(geometry.get("height_m"))


def apply_support_context_rules(candidates: List[JsonDict]) -> None:
    """代码会检查 pickup candidate 是否被可见支撑物包住。如
    果它在桌子、架子、柜子之类的支撑结构里，就把它标记为：
        surface_or_elevated
        pickup_now = False
        support_context_blocked = True
    YOLO 可能看到架子上的杯子/水果，但因为 2D 图像 y 坐标低，看起来像地面物体；
    所以 floor-only tidy 策略不能把它当作直接地面拾取目标
    """
    supports = []
    for candidate in candidates:
        label_token = normalize_label(candidate.get("label") or candidate.get("raw_label") or "")
        task_class = str(candidate.get("task_semantic_class") or "")
        if label_token in SUPPORT_CONTEXT_LABELS or task_class == "place_receptacle":
            supports.append(candidate)

    for candidate in candidates:
        if candidate.get("task_semantic_class") != "pickup_target":
            continue
        if candidate.get("actionability_source") == "depth_geometry" and candidate.get("is_floor_level"):
            floor_source = str(candidate.get("floor_level_source") or "")
            if floor_source in {"rgbd_floor_plane_contact", "bottom_band_ground_distance_fallback"}:
                continue
        # Strong RGB-D contact evidence owns the decision.  A coarse 2-D
        # CounterTop/Cabinet box may overlap floor pixels, but it must not turn
        # a floor object into an elevated object when the support strip below
        # that object lies on the fitted floor plane.
        if pickup_floor_contact_like(candidate):
            candidate.setdefault("support_context_skipped", []).append(
                {
                    "reason": "rgbd_floor_contact_overrides_2d_support_bbox",
                    "floor_plane_residual_m": candidate.get("floor_plane_residual_m"),
                    "floor_contact_confidence": candidate.get("floor_contact_confidence"),
                }
            )
            continue
        cx_ratio = float((candidate.get("geometry") or {}).get("cx_ratio", 0.5) or 0.5)
        cy_ratio = float((candidate.get("geometry") or {}).get("cy_ratio", 0.5) or 0.5)
        bottom_ratio = float((candidate.get("geometry") or {}).get("bottom_y_ratio", 0.0) or 0.0)

        for support in supports:
            sx1, sy1, sx2, sy2 = bbox_ratios(support)
            support_label = normalize_label(support.get("label") or support.get("raw_label") or "")
            inside_x = sx1 - 0.025 <= cx_ratio <= sx2 + 0.025
            inside_y = sy1 - 0.025 <= cy_ratio <= sy2 + 0.055
            shelf_like = support_label in {"shelf", "shelving_unit", "shelvingunit"}
            relative_bottom = support_relative_y(bottom_ratio, sy1, sy2)
            on_support_plane = bool(
                inside_x
                and inside_y
                and bottom_ratio <= sy2 + 0.055
                and relative_bottom <= 0.72
            )

            # A 2-D overlap is not enough to claim tabletop support.  When both
            # surfaces have RGB-D height evidence, require the pickup support
            # strip to agree with the candidate support height.
            pickup_support_h = pickup_support_height_m(candidate)
            support_h = support_candidate_height_m(support)
            support_height_tolerance = env_float("ROBOT_PICKUP_SUPPORT_CONTEXT_HEIGHT_TOLERANCE_M", 0.18)
            if (
                (shelf_like and inside_x and inside_y) or on_support_plane
            ) and pickup_support_h is not None and support_h is not None:
                height_delta = abs(float(pickup_support_h) - float(support_h))
                if height_delta > support_height_tolerance:
                    candidate.setdefault("support_context_skipped", []).append(
                        {
                            "label": support.get("label"),
                            "raw_label": support.get("raw_label"),
                            "reason": "rgbd_support_height_mismatch",
                            "pickup_support_height_m": round(float(pickup_support_h), 4),
                            "support_height_m": round(float(support_h), 4),
                            "height_delta_m": round(float(height_delta), 4),
                            "height_tolerance_m": round(float(support_height_tolerance), 4),
                        }
                    )
                    continue

            if not (shelf_like and inside_x and inside_y) and not on_support_plane:
                if inside_x and inside_y and bottom_ratio >= 0.90 and relative_bottom > 0.72:
                    candidate.setdefault("support_context_skipped", []).append(
                        {
                            "label": support.get("label"),
                            "raw_label": support.get("raw_label"),
                            "reason": "candidate_in_lower_support_box_floor_band",
                            "support_relative_bottom": round(float(relative_bottom), 3),
                        }
                    )
                continue

            candidate["surface_hint"] = "surface_or_elevated"
            candidate["is_floor_level"] = False
            candidate["pickup_now"] = False
            candidate["cleanable_now"] = False
            candidate["needs_approach"] = False
            candidate["support_context_blocked"] = True
            candidate["support_context"] = {
                "label": support.get("label"),
                "raw_label": support.get("raw_label"),
                "reason": "pickup_candidate_inside_visible_support",
            }
            break


def estimate_occupancy(candidates: Iterable[JsonDict]) -> Dict[str, float]:
    occupancy = {"left": 0.0, "forward": 0.0, "right": 0.0}
    risk_distance = env_float("ROBOT_DEPTH_OBSTACLE_RISK_MAX_DISTANCE", 1.35)
    for candidate in candidates:
        if candidate.get("task_semantic_class") != "obstacle":
            continue
        if not candidate.get("obstacle_risk"):
            continue
        hint = str(candidate.get("position_hint") or "")
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
        distance_m = candidate_depth_distance(candidate)
        if distance_m is not None:
            depth_weight = max(0.0, 1.0 - min(1.0, distance_m / max(0.001, risk_distance))) * 0.24
            area_ratio = max(area_ratio, depth_weight)
        if hint == "front-left":
            occupancy["left"] += area_ratio
        elif hint == "front-right":
            occupancy["right"] += area_ratio
        elif hint == "front-center":
            occupancy["forward"] += area_ratio
    return {key: round(min(0.99, value), 3) for key, value in occupancy.items()}


def infer_open_directions(candidates: List[JsonDict], occupancy: Dict[str, float], threshold: float) -> Tuple[bool, List[str]]:
    obstacle_ahead = any(
        c.get("task_semantic_class") == "obstacle"
        and c.get("position_hint") == "front-center"
        and c.get("obstacle_risk")
        and float(c.get("area_ratio", 0.0) or 0.0) >= threshold
        for c in candidates
    )
    obstacle_ahead = bool(obstacle_ahead or occupancy.get("forward", 0.0) >= threshold)
    open_directions = [name for name in ["left", "forward", "right"] if occupancy.get(name, 0.0) < threshold]
    if obstacle_ahead and "forward" in open_directions:
        open_directions.remove("forward")
    return obstacle_ahead, open_directions or ["left", "right"]

#视觉层给一个“建议动作”
def recommended_action(
    *,
    best_pickup_candidate: Optional[JsonDict],
    best_receptacle_candidate: Optional[JsonDict],
    direct_cleanable_detected: bool,
    service_candidates: List[JsonDict],
    obstacle_ahead: bool,
    open_directions: List[str],
) -> str:
    def alignment_action(candidate: JsonDict) -> str:
        bearing = candidate_bearing_deg(candidate)
        if bearing is not None:
            return "RotateLeft" if bearing < 0 else "RotateRight"
        position_hint = str(candidate.get("position_hint") or "")
        geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
        try:
            cx_ratio = float(geometry.get("cx_ratio", 0.5) or 0.5)
        except (TypeError, ValueError):
            cx_ratio = 0.5
        if position_hint == "front-left" or (position_hint == "front-center" and cx_ratio < 0.5):
            return "RotateLeft"
        return "RotateRight"

    if best_pickup_candidate and best_pickup_candidate.get("pickup_now"):
        return "pick-object"
    # This is a visual recommendation only. The runner still checks inventory
    # before it can execute place-object.
    if best_receptacle_candidate and best_receptacle_candidate.get("place_now"):
        return "place-object"
    if direct_cleanable_detected:
        return "clean-garbage"
    if best_pickup_candidate:
        if best_pickup_candidate.get("needs_alignment"):
            return alignment_action(best_pickup_candidate)
        if best_pickup_candidate.get("needs_approach") and "forward" in open_directions:
            return "MoveAhead"
    for candidate in service_candidates:
        if candidate.get("needs_alignment"):
            return alignment_action(candidate)
        if candidate.get("needs_approach") and "forward" in open_directions:
            return "MoveAhead"
    if obstacle_ahead:
        if "left" in open_directions:
            return "RotateLeft"
        if "right" in open_directions:
            return "RotateRight"
        return "MoveBack"
    if "forward" in open_directions:
        return "MoveAhead"
    if "left" in open_directions:
        return "RotateLeft"
    if "right" in open_directions:
        return "RotateRight"
    return "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + optional depth scene perception for V2 service tasks.")
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--depth", default="", help="Optional AI2-THOR depth frame path (.npy float32 meters).")
    parser.add_argument("--camera-json", default="", help="Optional camera intrinsics/extrinsics JSON from get-vision.")
    parser.add_argument("--weights", default=os.getenv("ROBOT_YOLO_WEIGHTS", str(DEFAULT_WEIGHTS)))
    parser.add_argument("--fallback-weights", default=DEFAULT_FALLBACK_WEIGHTS)
    parser.add_argument("--ontology", default=os.getenv("ROBOT_SERVICE_ONTOLOGY", str(DEFAULT_ONTOLOGY)))
    parser.add_argument("--conf", type=float, default=float(os.getenv("ROBOT_YOLO_CONF", "0.35")))
    parser.add_argument("--iou", type=float, default=float(os.getenv("ROBOT_YOLO_IOU", "0.70")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("ROBOT_YOLO_IMGSZ", "640")))
    parser.add_argument("--floor-bottom-ratio", type=float, default=0.70)
    parser.add_argument("--near-area-ratio", type=float, default=0.0015)
    parser.add_argument("--pickup-min-conf", type=float, default=float(os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55")))
    parser.add_argument(
        "--small-floor-pickup-min-conf",
        type=float,
        default=float(os.getenv("ROBOT_PICKUP_SMALL_FLOOR_MIN_CONF", os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55"))),
    )
    parser.add_argument(
        "--place-min-conf",
        type=float,
        default=float(os.getenv("ROBOT_PLACE_MIN_CONF", "0.72")),
        help="Legacy RGB place threshold kept for CLI compatibility; depth surface candidates own place_now.",
    )
    parser.add_argument(
        "--place-min-area-ratio",
        type=float,
        default=float(os.getenv("ROBOT_PLACE_MIN_AREA", "0.025")),
        help="Legacy RGB place threshold kept for CLI compatibility; depth surface candidates own place_now.",
    )
    parser.add_argument("--place-max-area-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MAX_AREA", "0.45")))
    parser.add_argument("--place-max-width-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MAX_WIDTH", "0.995")))
    parser.add_argument("--obstacle-area-threshold", type=float, default=0.035)
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("ROBOT_YOLO_MAX_CANDIDATES", "8")))
    parser.add_argument("--save-vis", default="", help="Optional path to save YOLO annotated image.")
    parser.add_argument("--save-plane-vis", default="", help="Optional path to save initial RANSAC plane visualization and derived filtering-stage images.")
    parser.add_argument(
        "--holding-object",
        action="store_true",
        default=env_bool("ROBOT_HOLDING_OBJECT", False),
        help="Tell perception that the robot is carrying an object, so matching foreground detections can be ignored as self-held blockers.",
    )
    parser.add_argument(
        "--held-object-labels",
        default=os.getenv("ROBOT_HELD_OBJECT_LABELS", os.getenv("ROBOT_HELD_OBJECT_LABEL", "")),
        help="Comma-separated labels for the currently held object, used only for self-held blocker filtering.",
    )
    parser.add_argument(
        "--held-object-family",
        default=os.getenv("ROBOT_HELD_OBJECT_FAMILY", ""),
        help="Optional held object family for self-held overlay filtering: food, pickup_target, or unknown.",
    )
    return parser


def analyze_image_with_model(
    *,
    model: Any,
    image_path: Path,
    weights: str,
    ontology: str,
    task_class_map: Dict[str, str],
    notes: Optional[List[str]] = None,
    conf: float = 0.35,
    iou: float = 0.70,
    imgsz: int = 640,
    floor_bottom_ratio: float = 0.70,
    near_area_ratio: float = 0.0015,
    pickup_min_conf: float = 0.55,
    small_floor_pickup_min_conf: float = 0.55,
    place_min_conf: float = 0.72,
    place_min_area_ratio: float = 0.025,
    place_max_area_ratio: float = 0.45,
    place_max_width_ratio: float = 0.995,
    obstacle_area_threshold: float = 0.035,
    max_candidates: int = 8,
    save_vis: str = "",
    save_plane_vis: str = "",
    depth_path: str = "",
    camera_info: Any = None,
    holding_object: bool = False,
    held_object_labels: Optional[Iterable[Any]] = None,
    held_object_family: Optional[str] = None,
) -> JsonDict:
    if not image_path.exists():
        return {
            "status": "error",
            "schema_version": 2,
            "result_type": "error_image_not_found",
            "message": f"Image not found: {image_path}",
            "online_safe": True,
        }

    output_notes = list(notes or [])
    held_labels = normalize_label_tokens(held_object_labels)
    held_family = normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels)
    output_notes.append(f"holding_object_context={str(bool(holding_object)).lower()}")
    try:
        results = model(str(image_path), conf=float(conf), iou=float(iou), imgsz=int(imgsz), verbose=False)
    except Exception as exc:
        return {
            "status": "error",
            "schema_version": 2,
            "result_type": "error_yolo_inference_failed",
            "message": str(exc),
            "weights": weights,
            "online_safe": True,
        }

    all_candidates: List[JsonDict] = []
    image_h = 0
    image_w = 0

    for result in results:
        image_h, image_w = result.orig_shape
        names = getattr(model, "names", {}) or {}
        for box in result.boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_label = str(names.get(cls_id, cls_id))
            normalized = normalize_label(raw_label)
            task_class = task_class_map.get(normalized, "ignored_object")
            candidate = build_candidate(
                raw_label=raw_label,
                task_semantic_class=task_class,
                confidence=conf,
                xyxy=xyxy,
                image_w=int(image_w),
                image_h=int(image_h),
                floor_bottom_ratio=float(floor_bottom_ratio),
                near_area_ratio=float(near_area_ratio),
                place_max_area_ratio=float(place_max_area_ratio),
                place_max_width_ratio=float(place_max_width_ratio),
                pickup_min_conf=float(pickup_min_conf),
                small_floor_pickup_min_conf=float(small_floor_pickup_min_conf),
            )
            all_candidates.append(candidate)

    if bool(holding_object) and not held_labels:
        held_labels = infer_held_object_labels_from_candidates(all_candidates)
        if held_labels:
            output_notes.append(f"held_object_label_inferred_from_foreground={','.join(held_labels)}")
        held_family = normalize_label(held_object_family or "") or held_object_family_for_labels(held_labels)
    output_notes.append(f"held_object_labels={','.join(held_labels) if held_labels else 'none'}")
    output_notes.append(f"held_object_family={held_family or 'unknown'}")

    depth_frame = load_depth_frame(str(depth_path or ""), image_w=int(image_w), image_h=int(image_h), notes=output_notes)
    resolved_plane_vis = str(save_plane_vis or "")
    if not resolved_plane_vis and save_vis:
        vis_path = Path(save_vis)
        resolved_plane_vis = str(vis_path.with_name(f"{vis_path.stem}-pointcloud-planes{vis_path.suffix or '.jpg'}"))
    surface_diagnostics: JsonDict = {}
    surface_candidates = apply_depth_geometry(
        all_candidates,
        depth_frame=depth_frame,
        camera_info=camera_info,
        image_path=str(image_path),
        save_plane_vis=resolved_plane_vis,
        image_w=int(image_w),
        image_h=int(image_h),
        notes=output_notes,
        holding_object=bool(holding_object),
        held_object_labels=held_labels,
        held_object_family=held_family,
        diagnostics=surface_diagnostics,
    )

    apply_support_context_rules(all_candidates)#支撑面规则：避免把桌上的东西误判成地面可捡物
    decorate_receptacle_occupancy_context(
        all_candidates,
        holding_object=bool(holding_object),
        held_object_labels=held_labels,
        held_object_family=held_family,
    )
    held_overlay_candidates = mark_held_object_overlays(
        all_candidates,
        holding_object=bool(holding_object),
        held_object_labels=held_labels,
        held_object_family=held_family,
    )

    pickup_candidates = [
        c for c in all_candidates
        if c.get("task_semantic_class") == "pickup_target"
        and not c.get("likely_held_object_overlay")
    ]
    receptacle_all = [c for c in all_candidates if c.get("task_semantic_class") == "place_receptacle"]
    raw_receptacle_all = [
        c for c in receptacle_all
        if str(c.get("surface_candidate_source") or "") not in SURFACE_REGION_SOURCES
    ]
    active_receptacles = active_receptacle_candidates(raw_receptacle_all)
    pointcloud_surface_regions = [
        c for c in all_candidates
        if c.get("task_semantic_class") == "place_receptacle"
        and c.get("surface_candidate_source") in {POINTCLOUD_SURFACE_SOURCE, POINTCLOUD_COMPLETION_SURFACE_SOURCE, POINTCLOUD_GRID_COMPLETION_SURFACE_SOURCE}
    ]
    fallback_surface_regions = [
        c for c in all_candidates
        if c.get("task_semantic_class") == "place_receptacle"
        and c.get("surface_candidate_source") == DEPTH_REGION_SURFACE_SOURCE
    ]
    surface_regions = pointcloud_surface_regions + fallback_surface_regions
    active_surface_regions = pointcloud_surface_regions if pointcloud_surface_regions else fallback_surface_regions
    visual_ready_surface_regions = [c for c in active_surface_regions if c.get("visual_place_ready")]
    rejected_surface_regions = [c for c in active_surface_regions if not c.get("visual_place_ready")]
    surface_rejection_summary = summarize_surface_rejections(rejected_surface_regions)
    place_affordance_candidates = [
        c for c in visual_ready_surface_regions
        if c.get("affordance_ready", True)
    ]
    unblocked_surface_regions = [c for c in active_surface_regions if not c.get("blocked")]
    grid_summary = (
        surface_diagnostics.get("free_space_grid_summary")
        if isinstance(surface_diagnostics.get("free_space_grid_summary"), dict)
        else {}
    )
    has_approachable_free_space = bool(grid_summary.get("has_approachable_free_space"))
    grid_no_ready_status = str(grid_summary.get("status") or "")
    surface_free_space_status = (
        "free_unblocked_affordance"
        if place_affordance_candidates
        else "free_space_outside_current_reach"
        if has_approachable_free_space
        else grid_no_ready_status
        if grid_no_ready_status in {"no_grid_free_mask", "no_grid_component_passed_safety_checks"}
        else "unblocked_but_not_affordance_ready"
        if unblocked_surface_regions
        else "no_free_unblocked_surface"
        if active_surface_regions
        else "no_surface_regions"
    )
    placement_avoidance_candidates = build_placement_avoidance_candidates(
        all_candidates,
        holding_object=bool(holding_object),
        held_object_labels=held_labels,
        held_object_family=held_family,
    )
    trash_all = [c for c in all_candidates if c.get("task_semantic_class") == "cleanable_object"]
    ignored_all = [c for c in all_candidates if c.get("task_semantic_class") in {"ignored_object", "obstacle"}]
    obstacle_all = [c for c in all_candidates if c.get("task_semantic_class") == "obstacle"]

    best_pickup = best_candidate(pickup_candidates, intent="pickup")
    best_receptacle = best_candidate(active_receptacles, intent="receptacle")
    best_surface = best_candidate(place_affordance_candidates, intent="receptacle")
    best_rejected_surface = best_candidate(rejected_surface_regions, intent="receptacle")
    best_place_affordance = best_candidate(place_affordance_candidates, intent="receptacle")
    best_obstacle = best_candidate(obstacle_all, intent="obstacle")
    surface_place_status = "visual_ready_surface" if best_surface else "no_visual_ready_surface"

    max_candidates = max(1, int(max_candidates))
    service_candidates = sorted_candidates(
        pickup_candidates + active_receptacles + trash_all,
        intent="service",
        max_items=max_candidates,
    )
    receptacle_candidates = sorted_candidates(active_receptacles, intent="receptacle", max_items=max_candidates)
    surface_region_candidates = sorted_candidates(surface_regions, intent="receptacle", max_items=max_candidates)
    pointcloud_surface_region_candidates = sorted_candidates(
        pointcloud_surface_regions,
        intent="receptacle",
        max_items=max_candidates,
    )
    place_affordance_candidates = sorted_candidates(
        place_affordance_candidates,
        intent="receptacle",
        max_items=max_candidates,
    )
    trash_candidates = sorted_candidates(trash_all, intent="service", max_items=max_candidates)
    ignored_candidates = sorted_candidates(ignored_all, intent="obstacle", max_items=max_candidates)
    top_obstacle_candidates = sorted_candidates(obstacle_all, intent="obstacle", max_items=max_candidates)

    occupancy = estimate_occupancy(all_candidates)
    obstacle_ahead, open_directions = infer_open_directions(
        all_candidates,
        occupancy,
        threshold=float(obstacle_area_threshold),
    )

    pickup_target_detected = bool(pickup_candidates)
    place_receptacle_detected = bool(receptacle_all)
    direct_pickup_detected = bool(best_pickup and best_pickup.get("pickup_now"))
    direct_place_detected = bool(best_surface and best_surface.get("place_now"))
    direct_cleanable_detected = any(c.get("cleanable_now") for c in trash_candidates)
    alignment_needed = any(c.get("needs_alignment") for c in service_candidates + trash_candidates)
    approach_needed = bool(
        any(c.get("needs_approach") for c in service_candidates + trash_candidates)
        or (not place_affordance_candidates and has_approachable_free_space)
    )
    frontier_exists = bool(open_directions)
    max_conf = max([float(c.get("confidence", 0.0) or 0.0) for c in all_candidates], default=0.0)

    if service_candidates or trash_candidates:
        analysis_confidence = round(max(0.65, min(0.95, max_conf)), 2)
    elif frontier_exists:
        analysis_confidence = 0.72 if not obstacle_ahead else 0.68
    else:
        analysis_confidence = 0.65
    held_object_blockers_ignored = sum(
        1
        for candidate in all_candidates
        if isinstance(candidate, dict) and candidate.get("ignored_as_held_object_blocker")
    )

    output_notes.extend(
        [
            "service_task_perception_active",
            "candidate_postprocess=v2.2_stable_best_candidates",
            f"raw_candidates={len(all_candidates)}",
            f"service_candidates={len(service_candidates)}",
            f"receptacle_candidates={len(receptacle_candidates)}",
            f"receptacle_candidates_merged={len(raw_receptacle_all) - len(active_receptacles)}",
            f"ignored_or_obstacle_candidates={len(ignored_candidates)}",
            f"surface_candidates={len(surface_candidates)}",
            f"surface_region_count={len(surface_regions)}",
            f"pointcloud_surface_region_count={len(pointcloud_surface_regions)}",
            f"surface_region_visual_ready={len(visual_ready_surface_regions)}",
            f"visual_ready_surface_region_count={len(visual_ready_surface_regions)}",
            f"place_affordance_candidate_count={len(place_affordance_candidates)}",
            f"surface_place_status={surface_place_status}",
            f"surface_free_space_status={surface_free_space_status}",
            f"held_object_blockers_ignored={held_object_blockers_ignored}",
            f"held_object_overlay_candidates={len(held_overlay_candidates)}",
            f"surface_rejection_summary={json.dumps(surface_rejection_summary, ensure_ascii=False, sort_keys=True)}",
            f"yolo_iou={float(iou):.2f}",
            "raw_receptacle_place_now=disabled_depth_surface_required",
            "surface_place_distance=ground_distance_m",
            f"surface_place_ground_range={env_float('ROBOT_DEPTH_SURFACE_PLACE_NOW_MIN_GROUND_DISTANCE', 0.50):.2f}-{env_float('ROBOT_DEPTH_SURFACE_PLACE_NOW_MAX_GROUND_DISTANCE', env_float('ROBOT_PLACE_MAX_DISTANCE', 1.0)):.2f}m",
            f"place_max_area_ratio={float(place_max_area_ratio):.4f}",
            f"place_max_width_ratio={float(place_max_width_ratio):.4f}",
            f"pickup_min_conf={float(pickup_min_conf):.2f}",
            f"small_floor_pickup_min_conf={float(small_floor_pickup_min_conf):.2f}",
        ]
    )

    output: JsonDict = {
        "status": "success",
        "schema_version": 2,
        "result_type": "scene_analyzed_yolo",
        "perception_backend": "yolo",
        "online_safe": True,
        "image_path": str(image_path),
        "depth_path": str(depth_path or "") if depth_path else None,
        "weights": weights,
        "ontology": str(ontology),
        "notes": output_notes,
        "holding_object_context": bool(holding_object),
        "held_object_labels": held_labels,
        "held_object_family": held_family,
        "held_object_blockers_ignored": int(held_object_blockers_ignored),
        "held_object_overlay_count": len(held_overlay_candidates),
        "held_object_overlay_candidates": [
            short_avoidance_candidate(candidate)
            for candidate in held_overlay_candidates[:8]
        ],

        "pickup_target_detected": bool(pickup_target_detected),
        "place_receptacle_detected": bool(place_receptacle_detected),
        "direct_pickup_detected": bool(direct_pickup_detected),
        "direct_place_detected": bool(direct_place_detected),
        "best_pickup_candidate": best_pickup,
        "best_receptacle_candidate": best_receptacle,
        "best_surface_candidate": best_surface,
        "best_rejected_surface_candidate": best_rejected_surface,
        "best_place_affordance": best_place_affordance,
        "best_obstacle_candidate": best_obstacle,
        "service_candidates": service_candidates,
        "receptacle_candidates": receptacle_candidates,
        "visual_ready_surface_regions": sorted_candidates(
            visual_ready_surface_regions,
            intent="receptacle",
            max_items=max_candidates,
        ),
        "pointcloud_surface_regions": pointcloud_surface_region_candidates,
        "place_affordance_candidates": place_affordance_candidates,
        "surface_candidates": sorted_candidates(surface_candidates, intent="receptacle", max_items=max_candidates),
        "surface_regions": surface_region_candidates,
        "surface_rejection_summary": surface_rejection_summary,
        "surface_place_status": surface_place_status,
        "surface_free_space_status": surface_free_space_status,
        "surface_search_action": None,
        "surface_search_reason": None,
        "no_ready_surface_steps": 0,
        "placement_avoidance_candidates": placement_avoidance_candidates,
        "top_obstacle_candidates": top_obstacle_candidates,

        "floor_trash_detected": bool(trash_candidates),
        "direct_cleanable_detected": bool(direct_cleanable_detected),
        "alignment_needed": bool(alignment_needed),
        "approach_needed": bool(approach_needed),
        "trash_candidates": trash_candidates,
        "ignored_candidates": ignored_candidates,
        "obstacle_ahead": bool(obstacle_ahead),
        "open_directions": open_directions,
        "frontier_exists": bool(frontier_exists),
        "floor_clean": not bool(trash_candidates),
        "analysis_confidence": analysis_confidence,
        "occupancy": occupancy,
        "recommended_action": recommended_action(
            best_pickup_candidate=best_pickup,
            best_receptacle_candidate=best_surface,
            direct_cleanable_detected=bool(direct_cleanable_detected),
            service_candidates=service_candidates,
            obstacle_ahead=bool(obstacle_ahead),
            open_directions=open_directions,
        ),
        "candidate_count": len(all_candidates),
        "reported_candidate_count": len(service_candidates) + len(ignored_candidates),
        "surface_candidate_count": len(surface_candidates),
        "surface_region_count": len(surface_regions),
        "pointcloud_surface_region_count": len(pointcloud_surface_regions),
        "visual_ready_surface_region_count": len(visual_ready_surface_regions),
    }
    if save_vis:
        save_candidate_visualization(
            image_path=image_path,
            save_path=Path(save_vis),
            candidates=surface_region_candidates + service_candidates + ignored_candidates + held_overlay_candidates,
            notes=output_notes,
        )
    return output


def main() -> None:
    args = build_parser().parse_args()
    task_class_map = load_task_class_map(Path(args.ontology))
    weights, notes = resolve_weights(args.weights, args.fallback_weights)
    YOLO = import_yolo()
    try:
        model = YOLO(weights)
    except Exception as exc:
        json_print(
            {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_yolo_model_load_failed",
                "message": str(exc),
                "weights": weights,
                "online_safe": True,
            }
        )
        sys.exit(1)
    output = analyze_image_with_model(
        model=model,
        image_path=Path(args.image),
        weights=weights,
        ontology=str(args.ontology),
        task_class_map=task_class_map,
        notes=notes,
        conf=float(args.conf),
        iou=float(args.iou),
        imgsz=int(args.imgsz),
        floor_bottom_ratio=float(args.floor_bottom_ratio),
        near_area_ratio=float(args.near_area_ratio),
        pickup_min_conf=float(args.pickup_min_conf),
        small_floor_pickup_min_conf=float(args.small_floor_pickup_min_conf),
        place_min_conf=float(args.place_min_conf),
        place_min_area_ratio=float(args.place_min_area_ratio),
        place_max_area_ratio=float(args.place_max_area_ratio),
        place_max_width_ratio=float(args.place_max_width_ratio),
        obstacle_area_threshold=float(args.obstacle_area_threshold),
        max_candidates=int(args.max_candidates),
        save_vis=str(args.save_vis or ""),
        save_plane_vis=str(args.save_plane_vis or ""),
        depth_path=str(args.depth or ""),
        camera_info=str(args.camera_json or ""),
        holding_object=bool(args.holding_object),
        held_object_labels=str(args.held_object_labels or "").split(","),
        held_object_family=str(args.held_object_family or ""),
    )
    json_print(output)
    if output.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
