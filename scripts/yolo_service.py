#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent YOLO perception service.

The normal perceive_scene_yolo.py skill is process-per-call. This runtime
service keeps the Ultralytics model in memory so patrol_runner can avoid
loading PyTorch and best.pt on every perception step.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from flask import Flask, jsonify, request


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_SKILL_SCRIPT_DIR = REPO_ROOT / "skills" / "perceive-scene-yolo" / "scripts"
sys.path.insert(0, str(YOLO_SKILL_SCRIPT_DIR))

from perceive_scene_yolo_core import (  # noqa: E402
    DEFAULT_FALLBACK_WEIGHTS,
    DEFAULT_ONTOLOGY,
    DEFAULT_WEIGHTS,
    analyze_image_with_model,
    import_yolo,
    load_task_class_map,
    resolve_weights,
)


JsonDict = Dict[str, Any]


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent YOLO perception HTTP service.")
    parser.add_argument("--host", default=os.getenv("ROBOT_YOLO_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ROBOT_YOLO_SERVICE_PORT", "5055")))
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
    parser.add_argument("--place-min-conf", type=float, default=float(os.getenv("ROBOT_PLACE_MIN_CONF", "0.72")))
    parser.add_argument("--place-min-area-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MIN_AREA", "0.025")))
    parser.add_argument("--place-max-area-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MAX_AREA", "0.45")))
    parser.add_argument("--place-max-width-ratio", type=float, default=float(os.getenv("ROBOT_PLACE_MAX_WIDTH", "0.995")))
    parser.add_argument("--obstacle-area-threshold", type=float, default=0.035)
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("ROBOT_YOLO_MAX_CANDIDATES", "8")))
    return parser


def create_app(args: argparse.Namespace) -> Flask:
    weights, startup_notes = resolve_weights(args.weights, args.fallback_weights)
    task_class_map = load_task_class_map(Path(args.ontology))
    YOLO = import_yolo()
    model = YOLO(weights)
    model_lock = Lock()

    app = Flask(__name__)
    app.config["YOLO_MODEL"] = model
    app.config["YOLO_MODEL_LOCK"] = model_lock
    app.config["YOLO_WEIGHTS"] = weights
    app.config["YOLO_STARTUP_NOTES"] = startup_notes
    app.config["YOLO_TASK_CLASS_MAP"] = task_class_map
    app.config["YOLO_ARGS"] = args

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "success",
                "result_type": "yolo_service_ready",
                "weights": app.config["YOLO_WEIGHTS"],
                "ontology": str(args.ontology),
            }
        )

    @app.post("/analyze")
    def analyze():
        payload = request.get_json(silent=True) or {}
        image_value = payload.get("image") or payload.get("image_path")
        raw_depth = payload.get("depth")
        depth_value = payload.get("depth_path") or (raw_depth if isinstance(raw_depth, str) else "")
        if not image_value:
            return jsonify(
                {
                    "status": "error",
                    "schema_version": 2,
                    "result_type": "error_image_path_required",
                    "message": "Request JSON must include image or image_path.",
                    "online_safe": True,
                }
            ), 400

        def number(name: str, default: float) -> float:
            try:
                return float(payload.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        def integer(name: str, default: int) -> int:
            try:
                return int(payload.get(name, default))
            except (TypeError, ValueError):
                return int(default)

        notes = list(app.config["YOLO_STARTUP_NOTES"])
        notes.append("yolo_service_persistent_model")
        with app.config["YOLO_MODEL_LOCK"]:
            result = analyze_image_with_model(
                model=app.config["YOLO_MODEL"],
                image_path=Path(str(image_value)),
                weights=app.config["YOLO_WEIGHTS"],
                ontology=str(args.ontology),
                task_class_map=app.config["YOLO_TASK_CLASS_MAP"],
                notes=notes,
                conf=number("conf", args.conf),
                iou=number("iou", args.iou),
                imgsz=integer("imgsz", args.imgsz),
                floor_bottom_ratio=number("floor_bottom_ratio", args.floor_bottom_ratio),
                near_area_ratio=number("near_area_ratio", args.near_area_ratio),
                pickup_min_conf=number("pickup_min_conf", args.pickup_min_conf),
                small_floor_pickup_min_conf=number("small_floor_pickup_min_conf", args.small_floor_pickup_min_conf),
                place_min_conf=number("place_min_conf", args.place_min_conf),
                place_min_area_ratio=number("place_min_area_ratio", args.place_min_area_ratio),
                place_max_area_ratio=number("place_max_area_ratio", args.place_max_area_ratio),
                place_max_width_ratio=number("place_max_width_ratio", args.place_max_width_ratio),
                obstacle_area_threshold=number("obstacle_area_threshold", args.obstacle_area_threshold),
                max_candidates=integer("max_candidates", args.max_candidates),
                save_vis=str(payload.get("save_vis") or ""),
                save_plane_vis=str(payload.get("save_plane_vis") or ""),
                depth_path=str(depth_value or ""),
                camera_info=payload.get("camera") if isinstance(payload.get("camera"), dict) else payload.get("camera_json"),
                holding_object=truthy(payload.get("holding_object")),
                held_object_labels=parse_label_list(payload.get("held_object_label"), payload.get("held_object_labels")),
                held_object_family=str(payload.get("held_object_family") or ""),
            )
        status_code = 200 if result.get("status") == "success" else 500
        if result.get("result_type") == "error_image_not_found":
            status_code = 404
        return jsonify(result), status_code

    return app


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(args)
    print(
        f"YOLO service ready: http://{args.host}:{args.port} "
        f"weights={app.config['YOLO_WEIGHTS']}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
