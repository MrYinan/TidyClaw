#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB-only YOLO perception for V2.2 household service tasks.

This script is the online perception adapter used by patrol_runner.py.
It keeps the online contract strict: the runner receives RGB-derived YOLO
candidates and sanitized action hints only. It does not read AI2-THOR metadata.

Main V2.2 changes compared with the previous version:
- Distinguishes pickup targets, receptacles and obstacles during geometry parsing.
- Stops calling every large CounterTop/Sink box "floor".
- Adds best_pickup_candidate / best_receptacle_candidate for stable runner decisions.
- Makes place_now deliberately conservative to avoid repeated /place failures.
- Keeps legacy clean fields so clean mode remains compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = REPO_ROOT / "skills" / "perceive-scene-yolo" / "weights" / "best.pt"
DEFAULT_ONTOLOGY = REPO_ROOT / "configs" / "service_task_ontology_v2.json"
DEFAULT_FALLBACK_WEIGHTS = os.getenv("ROBOT_YOLO_FALLBACK_WEIGHTS", "yolo11n.pt")

SERVICE_CLASSES = {"pickup_target", "place_receptacle", "cleanable_object"}

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


SUPPORT_CONTEXT_LABELS = {
    "counter_top",
    "countertop",
    "dining_table",
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
    place_min_conf: float,
    place_min_area_ratio: float,
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

    cx_ratio = cx / max(image_w, 1)
    center_y_ratio = cy / max(image_h, 1)
    bottom_y_ratio = y2 / max(image_h, 1)
    area_ratio = area / max(image_w * image_h, 1)
    width_ratio = width / max(image_w, 1)
    position_hint = infer_position_hint(cx_ratio)
    center_offset = cx_ratio - 0.5

    is_floor_level = bottom_y_ratio >= floor_bottom_ratio
    surface_hint = support_surface_hint(task_semantic_class, is_floor_level)
    visually_near = bool(area_ratio >= near_area_ratio or bottom_y_ratio >= 0.58)
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
    pickup_now = bool(
        task_semantic_class == "pickup_target"
        and confidence >= pickup_min_conf
        and service_reachable
        and position_hint == "front-center"
        and (area_ratio >= near_area_ratio or small_floor_pickup_ready)
    )

    # Place is intentionally stricter than pickup. The previous version marked
    # large/slanted CounterTop boxes as place_now too easily, which could cause
    # repeated /place failures. The backend remains the final actuator check.
    place_centered = bool(abs(center_offset) <= 0.10)
    try:
        front_edge_min_bottom_ratio = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MIN_BOTTOM_RATIO", "0.88"))
    except (TypeError, ValueError):
        front_edge_min_bottom_ratio = 0.88
    try:
        front_edge_min_center_y_ratio = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MIN_CENTER_Y_RATIO", "0.58"))
    except (TypeError, ValueError):
        front_edge_min_center_y_ratio = 0.58
    try:
        front_edge_max_area_ratio = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MAX_AREA_RATIO", "0.72"))
    except (TypeError, ValueError):
        front_edge_max_area_ratio = 0.72
    try:
        front_edge_max_width_ratio = float(os.getenv("ROBOT_PLACE_FRONT_EDGE_MAX_WIDTH_RATIO", "1.01"))
    except (TypeError, ValueError):
        front_edge_max_width_ratio = 1.01
    front_edge_receptacle = bool(
        task_semantic_class == "place_receptacle"
        and position_hint == "front-center"
        and place_centered
        and bottom_y_ratio >= front_edge_min_bottom_ratio
        and center_y_ratio >= front_edge_min_center_y_ratio
        and area_ratio <= front_edge_max_area_ratio
        and width_ratio <= front_edge_max_width_ratio
    )
    broad_front_receptacle = bool(
        task_semantic_class == "place_receptacle"
        and position_hint == "front-center"
        and place_centered
        and (
            (bottom_y_ratio >= 0.66 and area_ratio <= place_max_area_ratio)
            or front_edge_receptacle
        )
    )
    receptacle_box_ambiguous = bool(
        task_semantic_class == "place_receptacle"
        and not broad_front_receptacle
        and (
            area_ratio > place_max_area_ratio
            or width_ratio > place_max_width_ratio
        )
    )
    receptacle_visually_ready = bool(
        task_semantic_class == "place_receptacle"
        and service_reachable
        and position_hint == "front-center"
        and place_centered
        and not receptacle_box_ambiguous
        and confidence >= place_min_conf
        and area_ratio >= place_min_area_ratio
        and bottom_y_ratio >= 0.68
    )
    place_now = bool(receptacle_visually_ready)

    cleanable_now = bool(
        task_semantic_class == "cleanable_object"
        and is_floor_level
        and service_reachable
        and position_hint == "front-center"
    )

    needs_alignment = False
    needs_approach = False
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


def candidate_sort_score(candidate: JsonDict, *, intent: str = "service") -> float:
    conf = float(candidate.get("confidence", 0.0) or 0.0)
    area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
    cx_ratio = float((candidate.get("geometry") or {}).get("cx_ratio", 0.5) or 0.5)
    center_bonus = max(0.0, 0.5 - abs(cx_ratio - 0.5))
    score = conf * 2.0 + min(area_ratio, 0.20) + center_bonus

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


def support_relative_y(value: float, sy1: float, sy2: float) -> float:
    height = max(0.001, sy2 - sy1)
    return (value - sy1) / height


def apply_support_context_rules(candidates: List[JsonDict]) -> None:
    """Mark pickup candidates on visible supports as elevated/non-actionable.

    YOLO can see a mug/fruit on a rack shelf with a low image y-coordinate that
    looks "floor-like" in pure 2D geometry. If a trained support structure
    encloses that object, the floor-only tidy policy must not treat it as a
    direct floor pickup.

    CounterTop/Table boxes can be broad perspective boxes that include the
    floor in the lower part of the image. For those supports we only block a
    pickup candidate when its bottom is in the upper/middle part of the support
    box, which is the best RGB-only proxy for "resting on the support". Shelf
    boxes remain stronger blockers because their vertical structure is itself
    the support context.
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
    for candidate in candidates:
        if candidate.get("task_semantic_class") != "obstacle":
            continue
        if not candidate.get("obstacle_risk"):
            continue
        hint = str(candidate.get("position_hint") or "")
        area_ratio = float(candidate.get("area_ratio", 0.0) or 0.0)
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
    parser = argparse.ArgumentParser(description="RGB-only YOLO scene perception for V2 service tasks.")
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--weights", default=os.getenv("ROBOT_YOLO_WEIGHTS", str(DEFAULT_WEIGHTS)))
    parser.add_argument("--fallback-weights", default=DEFAULT_FALLBACK_WEIGHTS)
    parser.add_argument("--ontology", default=os.getenv("ROBOT_SERVICE_ONTOLOGY", str(DEFAULT_ONTOLOGY)))
    parser.add_argument("--conf", type=float, default=float(os.getenv("ROBOT_YOLO_CONF", "0.35")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("ROBOT_YOLO_IMGSZ", "640")))
    parser.add_argument("--floor-bottom-ratio", type=float, default=0.70)
    parser.add_argument("--near-area-ratio", type=float, default=0.0015)
    parser.add_argument("--pickup-min-conf", type=float, default=float(os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55")))
    parser.add_argument(
        "--small-floor-pickup-min-conf",
        type=float,
        default=float(os.getenv("ROBOT_PICKUP_SMALL_FLOOR_MIN_CONF", os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55"))),
    )
    parser.add_argument("--place-min-conf", type=float, default=float(os.getenv("ROBOT_PLACE_MIN_CONF", "0.72")))
    parser.add_argument("--place-min-area-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MIN_AREA", "0.025")))
    parser.add_argument("--place-max-area-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MAX_AREA", "0.45")))
    parser.add_argument("--place-max-width-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MAX_WIDTH", "0.995")))
    parser.add_argument("--obstacle-area-threshold", type=float, default=0.035)
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("ROBOT_YOLO_MAX_CANDIDATES", "8")))
    parser.add_argument("--save-vis", default="", help="Optional path to save YOLO annotated image.")
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
    try:
        results = model(str(image_path), conf=float(conf), imgsz=int(imgsz), verbose=False)
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
                place_min_conf=float(place_min_conf),
                place_min_area_ratio=float(place_min_area_ratio),
                place_max_area_ratio=float(place_max_area_ratio),
                place_max_width_ratio=float(place_max_width_ratio),
                pickup_min_conf=float(pickup_min_conf),
                small_floor_pickup_min_conf=float(small_floor_pickup_min_conf),
            )
            all_candidates.append(candidate)

        if save_vis:
            try:
                import cv2  # type: ignore
                vis = result.plot()
                save_path = Path(save_vis)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), vis)
            except Exception as exc:
                output_notes.append(f"save_vis_failed:{exc}")

    apply_support_context_rules(all_candidates)

    pickup_candidates = [c for c in all_candidates if c.get("task_semantic_class") == "pickup_target"]
    receptacle_all = [c for c in all_candidates if c.get("task_semantic_class") == "place_receptacle"]
    trash_all = [c for c in all_candidates if c.get("task_semantic_class") == "cleanable_object"]
    ignored_all = [c for c in all_candidates if c.get("task_semantic_class") in {"ignored_object", "obstacle"}]
    obstacle_all = [c for c in all_candidates if c.get("task_semantic_class") == "obstacle"]

    best_pickup = best_candidate(pickup_candidates, intent="pickup")
    best_receptacle = best_candidate(receptacle_all, intent="receptacle")
    best_obstacle = best_candidate(obstacle_all, intent="obstacle")

    max_candidates = max(1, int(max_candidates))
    service_candidates = sorted_candidates(
        pickup_candidates + receptacle_all + trash_all,
        intent="service",
        max_items=max_candidates,
    )
    receptacle_candidates = sorted_candidates(receptacle_all, intent="receptacle", max_items=max_candidates)
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
    direct_place_detected = bool(best_receptacle and best_receptacle.get("place_now"))
    direct_cleanable_detected = any(c.get("cleanable_now") for c in trash_candidates)
    alignment_needed = any(c.get("needs_alignment") for c in service_candidates + trash_candidates)
    approach_needed = any(c.get("needs_approach") for c in service_candidates + trash_candidates)
    frontier_exists = bool(open_directions)
    max_conf = max([float(c.get("confidence", 0.0) or 0.0) for c in all_candidates], default=0.0)

    if service_candidates or trash_candidates:
        analysis_confidence = round(max(0.65, min(0.95, max_conf)), 2)
    elif frontier_exists:
        analysis_confidence = 0.72 if not obstacle_ahead else 0.68
    else:
        analysis_confidence = 0.65

    output_notes.extend(
        [
            "service_task_perception_active",
            "candidate_postprocess=v2.2_stable_best_candidates",
            f"raw_candidates={len(all_candidates)}",
            f"service_candidates={len(service_candidates)}",
            f"receptacle_candidates={len(receptacle_candidates)}",
            f"ignored_or_obstacle_candidates={len(ignored_candidates)}",
            f"place_min_conf={float(place_min_conf):.2f}",
            f"place_min_area_ratio={float(place_min_area_ratio):.4f}",
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
        "weights": weights,
        "ontology": str(ontology),
        "notes": output_notes,

        "pickup_target_detected": bool(pickup_target_detected),
        "place_receptacle_detected": bool(place_receptacle_detected),
        "direct_pickup_detected": bool(direct_pickup_detected),
        "direct_place_detected": bool(direct_place_detected),
        "best_pickup_candidate": best_pickup,
        "best_receptacle_candidate": best_receptacle,
        "best_obstacle_candidate": best_obstacle,
        "service_candidates": service_candidates,
        "receptacle_candidates": receptacle_candidates,
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
            best_receptacle_candidate=best_receptacle,
            direct_cleanable_detected=bool(direct_cleanable_detected),
            service_candidates=service_candidates,
            obstacle_ahead=bool(obstacle_ahead),
            open_directions=open_directions,
        ),
        "candidate_count": len(all_candidates),
        "reported_candidate_count": len(service_candidates) + len(ignored_candidates),
    }
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
    )
    json_print(output)
    if output.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
