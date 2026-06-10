#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTTP client adapter for persistent YOLO perception service.

This file keeps the old CLI contract:

    python perceive_scene_yolo.py --image xxx.jpg

But it no longer loads Ultralytics or best.pt by itself.
It sends the image path to the persistent YOLO service:

    POST http://127.0.0.1:5055/analyze

The real YOLO model is loaded once by yolo_service.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests


JsonDict = Dict[str, Any]


def json_print(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def parse_camera_json(raw: str) -> JsonDict:
    if not str(raw or "").strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_label_list(*values: Any) -> List[str]:
    labels: List[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            parts = value
        else:
            parts = str(value or "").split(",")
        for part in parts:
            label = str(part or "").strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def load_service_state_held_context() -> JsonDict:
    if not env_bool("ROBOT_YOLO_USE_SERVICE_STATE_HELD_CONTEXT", True):
        return {"holding_object": False, "held_object_labels": []}
    state_path = Path(__file__).resolve().parents[3] / "memory" / "service-task-state.json"
    if not state_path.exists():
        return {"holding_object": False, "held_object_labels": []}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"holding_object": False, "held_object_labels": []}
    if not isinstance(data, dict):
        return {"holding_object": False, "held_object_labels": []}
    return {
        "holding_object": bool(data.get("holding_object", False)),
        "held_object_labels": parse_label_list(
            data.get("held_object_labels"),
            data.get("held_object_label"),
            data.get("held_object_raw_label"),
        ),
        "held_object_family": str(data.get("held_object_family") or "").strip(),
    }


def held_context_from_args(args: argparse.Namespace) -> JsonDict:
    labels = parse_label_list(args.held_object_label, args.held_object_labels)
    holding_object = bool(args.holding_object or labels)
    family = str(args.held_object_family or "").strip()

    state_context: JsonDict = {}
    should_load_state = (not holding_object and not labels) or (
        holding_object and (not labels or not family)
    )
    if should_load_state:
        state_context = load_service_state_held_context()
    if not holding_object and not labels:
        holding_object = bool(state_context.get("holding_object", False))
    if holding_object and not labels:
        labels = parse_label_list(state_context.get("held_object_labels"))
    if holding_object and not family:
        family = str(state_context.get("held_object_family") or family).strip()
    return {"holding_object": holding_object, "held_object_labels": labels, "held_object_family": family}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO scene perception HTTP client for persistent YOLO service."
    )

    # Keep the old perceive_scene_yolo.py interface.
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--depth", default="", help="Optional depth frame path from get-vision (.npy float32 meters).")
    parser.add_argument("--camera-json", default="", help="Optional camera JSON from get-vision.")

    # HTTP service settings.
    parser.add_argument(
        "--service-url",
        default=os.getenv("ROBOT_YOLO_SERVICE_URL", "http://127.0.0.1:5055/analyze"),
        help="Persistent YOLO service analyze endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("ROBOT_YOLO_SERVICE_TIMEOUT", "60")),
        help="HTTP request timeout in seconds.",
    )

    # Keep old arguments for patrol_runner / skill compatibility.
    # These are forwarded to the service if the service supports them.
    parser.add_argument("--weights", default=os.getenv("ROBOT_YOLO_WEIGHTS", ""))
    parser.add_argument("--fallback-weights", default=os.getenv("ROBOT_YOLO_FALLBACK_WEIGHTS", ""))
    parser.add_argument("--ontology", default=os.getenv("ROBOT_SERVICE_ONTOLOGY", ""))
    parser.add_argument("--conf", type=float, default=float(os.getenv("ROBOT_YOLO_CONF", "0.35")))
    parser.add_argument("--iou", type=float, default=float(os.getenv("ROBOT_YOLO_IOU", "0.70")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("ROBOT_YOLO_IMGSZ", "640")))
    parser.add_argument("--floor-bottom-ratio", type=float, default=0.70)
    parser.add_argument("--near-area-ratio", type=float, default=0.0015)
    parser.add_argument("--pickup-min-conf", type=float, default=float(os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55")))
    parser.add_argument(
        "--small-floor-pickup-min-conf",
        type=float,
        default=float(
            os.getenv(
                "ROBOT_PICKUP_SMALL_FLOOR_MIN_CONF",
                os.getenv("ROBOT_PICKUP_MIN_CONF", "0.55"),
            )
        ),
    )
    parser.add_argument("--place-min-conf", type=float, default=float(os.getenv("ROBOT_PLACE_MIN_CONF", "0.72")))
    parser.add_argument("--place-min-area-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MIN_AREA", "0.025")))
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
        help="Tell perception the robot is carrying an object so the self-held foreground object is not treated as a placement blocker.",
    )
    parser.add_argument("--held-object-label", default=os.getenv("ROBOT_HELD_OBJECT_LABEL", ""))
    parser.add_argument("--held-object-labels", default=os.getenv("ROBOT_HELD_OBJECT_LABELS", ""))
    parser.add_argument("--held-object-family", default=os.getenv("ROBOT_HELD_OBJECT_FAMILY", ""))
    parser.add_argument(
        "--local-core",
        action="store_true",
        default=str(os.getenv("ROBOT_YOLO_FORCE_LOCAL_CORE", "")).strip().lower() in {"1", "true", "yes", "on"},
        help="Run local perceive_scene_yolo_core.py instead of the persistent HTTP service.",
    )
    parser.add_argument(
        "--disable-stale-service-fallback",
        action="store_true",
        default=str(os.getenv("ROBOT_YOLO_DISABLE_STALE_SERVICE_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"},
        help="Do not fall back to local core when the persistent service returns stale surface semantics.",
    )

    return parser


def build_payload(args: argparse.Namespace) -> JsonDict:
    image_path = Path(args.image)
    held_context = held_context_from_args(args)
    held_object_labels = parse_label_list(held_context.get("held_object_labels"))

    # 这里不再本地读取 best.pt，也不 import ultralytics。
    # 只把图片路径和参数发给常驻 YOLO 服务。
    return {
        "image_path": str(image_path),
        "depth_path": str(args.depth or ""),
        "camera": parse_camera_json(str(args.camera_json or "")),

        "conf": float(args.conf),
        "iou": float(args.iou),
        "imgsz": int(args.imgsz),
        "floor_bottom_ratio": float(args.floor_bottom_ratio),
        "near_area_ratio": float(args.near_area_ratio),
        "pickup_min_conf": float(args.pickup_min_conf),
        "small_floor_pickup_min_conf": float(args.small_floor_pickup_min_conf),
        "place_min_conf": float(args.place_min_conf),
        "place_min_area_ratio": float(args.place_min_area_ratio),
        "place_max_area_ratio": float(args.place_max_area_ratio),
        "place_max_width_ratio": float(args.place_max_width_ratio),
        "obstacle_area_threshold": float(args.obstacle_area_threshold),
        "max_candidates": int(args.max_candidates),
        "save_vis": str(args.save_vis or ""),
        "save_plane_vis": str(args.save_plane_vis or ""),
        "holding_object": bool(held_context.get("holding_object")),
        "held_object_labels": held_object_labels,
        "held_object_label": held_object_labels[0] if held_object_labels else "",
        "held_object_family": str(held_context.get("held_object_family") or ""),
    }


def stale_surface_semantics(data: JsonDict) -> bool:
    """Detect an old persistent service that predates ready/rejected surface split."""
    if not isinstance(data, dict) or data.get("status") != "success":
        return False
    if data.get("depth_path") and "pointcloud_surface_region_count" not in data:
        return True
    if data.get("depth_path") and "surface_free_space_status" not in data:
        return True
    if data.get("holding_object_context") and "held_object_family" not in data:
        return True
    best_surface = data.get("best_surface_candidate")
    if isinstance(best_surface, dict) and not bool(best_surface.get("visual_place_ready")):
        return True
    try:
        surface_region_count = int(data.get("surface_region_count", 0) or 0)
        visual_ready_count = int(data.get("visual_ready_surface_region_count", 0) or 0)
    except (TypeError, ValueError):
        surface_region_count = 0
        visual_ready_count = 0
    return bool(
        surface_region_count > 0
        and visual_ready_count <= 0
        and str(data.get("surface_place_status") or "") == "visual_ready_surface"
    )


def run_local_core(args: argparse.Namespace, *, reason: str) -> JsonDict:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from perceive_scene_yolo_core import (  # type: ignore
        DEFAULT_FALLBACK_WEIGHTS,
        DEFAULT_ONTOLOGY,
        DEFAULT_WEIGHTS,
        analyze_image_with_model,
        import_yolo,
        load_task_class_map,
        resolve_weights,
    )

    requested_weights = str(args.weights or DEFAULT_WEIGHTS)
    fallback_weights = str(args.fallback_weights or DEFAULT_FALLBACK_WEIGHTS)
    ontology = Path(str(args.ontology or DEFAULT_ONTOLOGY))
    task_class_map = load_task_class_map(ontology)
    weights, notes = resolve_weights(requested_weights, fallback_weights)
    notes.append(f"local_core_fallback:{reason}")
    held_context = held_context_from_args(args)
    YOLO = import_yolo()
    model = YOLO(weights)
    return analyze_image_with_model(
        model=model,
        image_path=Path(args.image),
        weights=weights,
        ontology=str(ontology),
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
        holding_object=bool(held_context.get("holding_object")),
        held_object_labels=parse_label_list(held_context.get("held_object_labels")),
        held_object_family=str(held_context.get("held_object_family") or ""),
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.local_core:
        data = run_local_core(args, reason="forced_local_core")
        json_print(data)
        if data.get("status") != "success":
            sys.exit(1)
        return

    payload = build_payload(args)

    try:
        response = requests.post(
            args.service_url,
            json=payload,
            timeout=float(args.timeout),
        )

        try:
            data = response.json()
        except Exception:
            data = {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_yolo_service_non_json_response",
                "message": response.text[:1000],
                "service_url": args.service_url,
                "http_status": response.status_code,
                "online_safe": True,
            }

        if stale_surface_semantics(data) and not bool(args.disable_stale_service_fallback):
            data = run_local_core(args, reason="stale_service_surface_semantics")

        json_print(data)

        if response.status_code >= 400 or data.get("status") != "success":
            sys.exit(1)

    except requests.exceptions.ConnectionError as exc:
        json_print(
            {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_yolo_service_unreachable",
                "message": (
                    "Cannot connect to persistent YOLO service. "
                    "Please start yolo_service.py first."
                ),
                "detail": str(exc),
                "service_url": args.service_url,
                "online_safe": True,
            }
        )
        sys.exit(1)

    except requests.exceptions.Timeout as exc:
        json_print(
            {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_yolo_service_timeout",
                "message": "Persistent YOLO service request timed out.",
                "detail": str(exc),
                "service_url": args.service_url,
                "online_safe": True,
            }
        )
        sys.exit(1)

    except Exception as exc:
        json_print(
            {
                "status": "error",
                "schema_version": 2,
                "result_type": "error_yolo_service_request_failed",
                "message": str(exc),
                "service_url": args.service_url,
                "online_safe": True,
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
