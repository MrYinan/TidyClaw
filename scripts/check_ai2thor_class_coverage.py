#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import requests


CLASS_ALIASES = {
    "SinkBasin": "Sink",
    "Table": "DiningTable",
    "StandardCounterHeightWidth": "CounterTop",
    "StandardIslandHeight": "CounterTop",
    "UpperCabinets": "Cabinet",
    "StoveBurner": "Stove",
    "StoveKnob": "Stove",
}


def load_yaml_names(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8-sig")

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        names = data.get("names", {})
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
        if isinstance(names, list):
            return [str(x) for x in names]
    except Exception:
        pass

    # Fallback parser for simple YOLO yaml:
    names: List[str] = []
    in_names = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if not line.startswith("  "):
                break
            if ":" in line:
                _, value = line.split(":", 1)
                names.append(value.strip().strip("'\""))
    return names


def get_json(url: str, *, timeout: float = 10.0) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Non-object JSON from {url}")
    return data


def post_json(url: str, payload: Dict[str, Any], *, timeout: float = 10.0) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Non-object JSON from {url}")
    return data


def alias_name(name: str) -> str:
    return CLASS_ALIASES.get(name, name)


def summarize_state(state: Dict[str, Any], classes: Sequence[str]) -> Dict[str, Any]:
    yaml_classes = set(classes)

    annotation_objects = state.get("annotation_objects", []) or []
    object_type_by_id: Dict[str, str] = {}

    present_counts: Counter[str] = Counter()
    visible_counts: Counter[str] = Counter()

    for obj in annotation_objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("objectId") or "")
        raw_type = str(obj.get("objectType") or "")
        if not raw_type:
            continue
        cls_name = alias_name(raw_type)
        object_type_by_id[object_id] = cls_name
        if cls_name in yaml_classes:
            present_counts[cls_name] += 1
            if obj.get("visible") is True:
                visible_counts[cls_name] += 1

    detections = state.get("instance_detections2D", {})
    if not isinstance(detections, dict):
        detections = {}

    box_counts: Counter[str] = Counter()
    for object_id in detections.keys():
        object_id_text = str(object_id)
        raw_type = object_type_by_id.get(object_id_text) or object_id_text.split("|", 1)[0]
        cls_name = alias_name(raw_type)
        if cls_name in yaml_classes:
            box_counts[cls_name] += 1

    missing_present = [c for c in classes if present_counts[c] == 0]
    missing_visible = [c for c in classes if visible_counts[c] == 0]
    missing_boxes = [c for c in classes if box_counts[c] == 0]

    return {
        "scene": state.get("scene"),
        "scenario": (state.get("scenario") or {}).get("name") if isinstance(state.get("scenario"), dict) else None,
        "segmentation_debug": state.get("segmentation_debug", {}),
        "present_counts": dict(present_counts),
        "visible_counts": dict(visible_counts),
        "box_counts": dict(box_counts),
        "missing_present": missing_present,
        "missing_visible": missing_visible,
        "missing_boxes": missing_boxes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        default="datasets/ai2thor_service_yolo_v2/ai2thor_service.yaml",
        help="YOLO dataset yaml path.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--scenarios", default="", help="Comma-separated scenario names, or 'all'.")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    classes = load_yaml_names(yaml_path)
    if not classes:
        raise SystemExit(f"No classes found in {yaml_path}")

    base = args.base_url.rstrip("/")
    results: List[Dict[str, Any]] = []

    if args.scenarios.strip():
        if args.scenarios.strip().lower() == "all":
            scenario_list = get_json(f"{base}/scenario/list")
            scenario_names = [
                item.get("name")
                for item in scenario_list.get("scenarios", [])
                if isinstance(item, dict) and item.get("name")
            ]
        else:
            scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

        for name in scenario_names:
            load_result = post_json(f"{base}/scenario/load", {"name": name})
            time.sleep(args.sleep)
            state = get_json(f"{base}/eval/state")
            one = summarize_state(state, classes)
            one["scenario_load_status"] = load_result.get("status")
            one["scenario_name_requested"] = name
            results.append(one)
    else:
        state = get_json(f"{base}/eval/state")
        results.append(summarize_state(state, classes))

    total_present: Counter[str] = Counter()
    total_visible: Counter[str] = Counter()
    total_boxes: Counter[str] = Counter()

    for item in results:
        total_present.update(item.get("present_counts", {}))
        total_visible.update(item.get("visible_counts", {}))
        total_boxes.update(item.get("box_counts", {}))

    summary = {
        "status": "success",
        "result_type": "ai2thor_yaml_class_coverage",
        "yaml": str(yaml_path),
        "class_count": len(classes),
        "classes": classes,
        "scenario_count": len(results),
        "total_present_counts": dict(total_present),
        "total_visible_counts": dict(total_visible),
        "total_box_counts": dict(total_boxes),
        "classes_missing_everywhere": [c for c in classes if total_present[c] == 0],
        "classes_present_but_no_visible": [c for c in classes if total_present[c] > 0 and total_visible[c] == 0],
        "classes_visible_but_no_boxes": [c for c in classes if total_visible[c] > 0 and total_boxes[c] == 0],
        "classes_with_boxes": [c for c in classes if total_boxes[c] > 0],
        "per_scenario": results,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
