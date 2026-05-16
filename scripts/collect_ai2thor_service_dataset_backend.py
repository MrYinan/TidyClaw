#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect YOLO service-task data from the running robot backend.

This is the recommended collector when AI2-THOR is already running through
`back/robot_server.py`. It uses:

- `/observation` for the RGB frame
- `/eval/state` for offline-only 2D instance boxes
- `/scenario/load` and `/move` for benchmark scene variation

The generated labels are for training only. Online Agent decisions must not use
`/eval/state`.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except Exception as exc:  # pragma: no cover
    print(json.dumps({"status": "error", "result_type": "error_requests_missing", "message": str(exc)}, ensure_ascii=False))
    sys.exit(1)


JsonDict = Dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ai2thor_service_yolo_v2"
DEFAULT_ONTOLOGY = REPO_ROOT / "configs" / "service_task_ontology_v2.json"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_SCENARIOS = "book_front_pick,apple_front_pick,book_on_floor_to_table,mixed_book_and_cabinet"
VALID_ACTIONS = ["RotateLeft", "RotateRight", "MoveAhead", "MoveBack"]
CLASS_ALIASES = {
    "SinkBasin": "Sink",
    "Table": "DiningTable",
    "StandardCounterHeightWidth": "CounterTop",
    "StandardIslandHeight": "CounterTop",
    "UpperCabinets": "Cabinet",
    "StoveBurner": "Stove",
    "StoveKnob": "Stove",
}


@dataclass
class CollectorConfig:
    dataset_root: str
    ontology: str
    base_url: str
    count: int
    split: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    scenarios: str
    scenario_interval: int
    action_policy: str
    actions: str
    move_after_capture: bool
    min_box_area: int
    random_seed: int
    prefix: str
    start_index: int
    timeout: float
    interval: float
    allow_empty_detections: bool
    init_only: bool
    summary: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_print(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def append_jsonl(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_classes(path: Path) -> List[str]:
    fallback = [
        "Book",
        "Apple",
        "Tomato",
        "Potato",
        "Mug",
        "Cup",
        "Bowl",
        "Plate",
        "RemoteControl",
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "CounterTop",
        "Sink",
        "Chair",
        "ArmChair",
        "Sofa",
        "Cabinet",
        "Bed",
    ]
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return fallback
    classes = data.get("object_classes") if isinstance(data, dict) else None
    if not isinstance(classes, list):
        return fallback
    cleaned = [str(item).strip() for item in classes if str(item).strip()]
    return cleaned or fallback


def ensure_dataset_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def write_yaml(root: Path, classes: Sequence[str]) -> Path:
    yaml_path = root / "ai2thor_service.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    yaml_path.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def summarize(root: Path, classes: Sequence[str]) -> JsonDict:
    class_counts = {name: 0 for name in classes}
    splits: JsonDict = {}
    for split in ["train", "val", "test"]:
        images = sorted((root / "images" / split).glob("*.jpg"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        boxes = 0
        empty = 0
        for label_file in labels:
            text = label_file.read_text(encoding="utf-8").strip()
            if not text:
                empty += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(classes):
                    class_counts[classes[class_id]] += 1
                    boxes += 1
        splits[split] = {"images": len(images), "labels": len(labels), "boxes": boxes, "empty_labels": empty}
    return {
        "status": "success",
        "result_type": "ai2thor_service_backend_dataset_summary",
        "dataset_root": str(root),
        "yaml": str(root / "ai2thor_service.yaml"),
        "splits": splits,
        "class_counts": class_counts,
    }


def request_json(method: str, url: str, *, timeout: float, json_body: Optional[JsonDict] = None) -> JsonDict:
    if method == "GET":
        response = requests.get(url, timeout=timeout)
    elif method == "POST":
        response = requests.post(url, json=json_body or {}, timeout=timeout)
    else:
        raise ValueError(f"unsupported method: {method}")
    try:
        data = response.json()
    except Exception:
        data = {"status": "error", "result_type": "error_invalid_json_response", "message": response.text}
    if isinstance(data, dict):
        data.setdefault("http_status", response.status_code)
        return data
    return {"status": "error", "result_type": "error_non_object_json", "value": data, "http_status": response.status_code}


def strip_data_url_prefix(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value


def save_observation_image(observation: JsonDict, image_path: Path) -> int:
    encoded = str(observation.get("vision_base64") or "")
    if not encoded:
        raise RuntimeError("/observation response has no vision_base64")
    payload = base64.b64decode(strip_data_url_prefix(encoded))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(payload)
    return len(payload)


def choose_split(index: int, rng: random.Random, split: str, train_ratio: float, val_ratio: float) -> str:
    if split in {"train", "val", "test"}:
        return split
    value = rng.random()
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def make_sample_id(prefix: str, index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}_{index:06d}"


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def labels_from_eval_state(
    eval_state: JsonDict,
    *,
    classes: Sequence[str],
    width: int,
    height: int,
    min_box_area: int,
) -> Tuple[List[str], List[JsonDict]]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    objects = eval_state.get("annotation_objects", []) or []
    object_type_by_id = {
        str(item.get("objectId")): str(item.get("objectType"))
        for item in objects
        if isinstance(item, dict) and item.get("objectId") and item.get("objectType")
    }
    detections = eval_state.get("instance_detections2D", {})
    if not isinstance(detections, dict):
        raise RuntimeError("/eval/state missing instance_detections2D; restart backend after latest code update")

    lines: List[str] = []
    meta: List[JsonDict] = []
    for object_id, raw_box in detections.items():
        object_id_text = str(object_id)
        object_type = object_type_by_id.get(object_id_text) or object_id_text.split("|", 1)[0]
        if not object_type:
            continue
        class_name = CLASS_ALIASES.get(object_type, object_type)
        if class_name not in class_to_id:
            continue
        clipped = clip_box(raw_box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        area = int((x2 - x1) * (y2 - y1))
        if area < min_box_area:
            continue
        x_center = ((x1 + x2) / 2.0) / float(width)
        y_center = ((y1 + y2) / 2.0) / float(height)
        box_w = (x2 - x1) / float(width)
        box_h = (y2 - y1) / float(height)
        class_id = class_to_id[class_name]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        meta.append(
            {
                "object_id": str(object_id),
                "object_type": object_type,
                "class_name": class_name,
                "class_id": class_id,
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area": area,
            }
        )
    return lines, meta


def sample_action(policy: str, actions: Sequence[str], rng: random.Random) -> Optional[str]:
    if policy == "none":
        return None
    if policy == "rotate":
        return rng.choice(["RotateLeft", "RotateRight"])
    if policy == "scripted":
        pool = [action for action in actions if action in VALID_ACTIONS]
        return rng.choice(pool) if pool else None
    if policy == "random":
        pool = [action for action in actions if action in VALID_ACTIONS] or ["RotateLeft", "RotateRight", "MoveAhead"]
        return rng.choice(pool)
    raise ValueError(f"unknown action policy: {policy}")


def maybe_load_scenario(config: CollectorConfig, scenarios: Sequence[str], index: int) -> Optional[JsonDict]:
    if not scenarios:
        return None
    if index % max(1, config.scenario_interval) != 0:
        return None
    scenario_name = scenarios[(index // max(1, config.scenario_interval)) % len(scenarios)]
    return request_json(
        "POST",
        f"{config.base_url.rstrip('/')}/scenario/load",
        timeout=config.timeout,
        json_body={"name": scenario_name},
    )


def collect(config: CollectorConfig) -> JsonDict:
    root = Path(config.dataset_root)
    classes = load_classes(Path(config.ontology))
    ensure_dataset_dirs(root)
    yaml_path = write_yaml(root, classes)

    if config.init_only:
        return {"status": "success", "result_type": "dataset_initialized", "dataset_root": str(root), "yaml": str(yaml_path), "classes": classes}
    if config.summary:
        return summarize(root, classes)

    base_url = config.base_url.rstrip("/")
    health = request_json("GET", f"{base_url}/health", timeout=config.timeout)
    if health.get("status") != "success":
        return {"status": "error", "result_type": "error_backend_unavailable", "backend_health": health}

    rng = random.Random(config.random_seed)
    scenarios = [item.strip() for item in config.scenarios.split(",") if item.strip()]
    if len(scenarios) == 1 and scenarios[0].lower() == "all":
        scenario_list = request_json("GET", f"{base_url}/scenario/list", timeout=config.timeout)
        scenarios = [
            str(item.get("name"))
            for item in scenario_list.get("scenarios", [])
            if isinstance(item, dict) and item.get("name")
        ]
        if not scenarios:
            return {"status": "error", "result_type": "error_no_scenarios_listed", "scenario_list": scenario_list}
    actions = [item.strip() for item in config.actions.split(",") if item.strip()]
    run_id = datetime.now().strftime("backend_service_%Y%m%d_%H%M%S")
    append_jsonl(
        root / "meta" / "collection-runs.jsonl",
        {
            "schema_version": 2,
            "run_id": run_id,
            "started_at": now_iso(),
            "config": asdict(config),
            "classes": classes,
            "yaml": str(yaml_path),
            "backend_health": health,
            "offline_label_source": "/eval/state instance_detections2D_or_instance_masks",
        },
    )

    saved = 0
    errors = 0
    for local_index in range(config.count):
        sample_index = config.start_index + local_index
        scenario_result = maybe_load_scenario(config, scenarios, local_index)
        if scenario_result is not None:
            json_print(
                {
                    "event": "scenario_loaded",
                    "index": sample_index,
                    "status": scenario_result.get("status"),
                    "result_type": scenario_result.get("result_type"),
                    "scenario": (scenario_result.get("scenario") or {}).get("name")
                    if isinstance(scenario_result.get("scenario"), dict)
                    else None,
                }
            )

        split = choose_split(sample_index, rng, config.split, config.train_ratio, config.val_ratio)
        sample_id = make_sample_id(config.prefix, sample_index)
        image_path = root / "images" / split / f"{sample_id}.jpg"
        label_path = root / "labels" / split / f"{sample_id}.txt"

        try:
            observation = request_json("GET", f"{base_url}/observation", timeout=config.timeout)
            if observation.get("status") != "success":
                raise RuntimeError(f"observation failed: {observation}")
            image = observation.get("image", {}) if isinstance(observation.get("image"), dict) else {}
            width = int(image.get("width") or 600)
            height = int(image.get("height") or 600)

            eval_state = request_json("GET", f"{base_url}/eval/state", timeout=config.timeout)
            if eval_state.get("status") != "success":
                raise RuntimeError(f"eval state failed: {eval_state}")
            detections = eval_state.get("instance_detections2D", {})
            if not config.allow_empty_detections and isinstance(detections, dict) and not detections:
                raise RuntimeError(
                    "eval state has zero 2D boxes; restart backend and inspect /eval/state segmentation_debug"
                )
            lines, boxes = labels_from_eval_state(
                eval_state,
                classes=classes,
                width=width,
                height=height,
                min_box_area=config.min_box_area,
            )
            image_bytes = save_observation_image(observation, image_path)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            append_jsonl(
                root / "meta" / "frames.jsonl",
                {
                    "schema_version": 2,
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "created_at": now_iso(),
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_bytes": image_bytes,
                    "box_count": len(lines),
                    "boxes": boxes,
                    "scenario": eval_state.get("scenario"),
                    "robot": eval_state.get("robot"),
                    "offline_label_source": "/eval/state instance_detections2D_or_instance_masks",
                    "online_policy": "Do not use eval metadata for online Agent decisions.",
                },
            )
            saved += 1
            json_print(
                {
                    "event": "sample_saved",
                    "index": sample_index,
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "box_count": len(lines),
                }
            )
        except Exception as exc:
            errors += 1
            for partial_path in (image_path, label_path):
                try:
                    if partial_path.exists():
                        partial_path.unlink()
                except Exception:
                    pass
            append_jsonl(root / "meta" / "collection-errors.jsonl", {"run_id": run_id, "index": sample_index, "time": now_iso(), "message": str(exc)})
            json_print({"event": "sample_error", "index": sample_index, "message": str(exc)})

        action = sample_action(config.action_policy, actions, rng)
        if action and config.move_after_capture:
            move = request_json("POST", f"{base_url}/move", timeout=config.timeout, json_body={"action": action})
            json_print({"event": "move_after_capture", "index": sample_index, "action": action, "status": move.get("status"), "result_type": move.get("result_type")})

        if config.interval > 0:
            time.sleep(config.interval)

    append_jsonl(root / "meta" / "collection-runs.jsonl", {"run_id": run_id, "finished_at": now_iso(), "saved": saved, "errors": errors})
    summary = summarize(root, classes)
    summary.update({"run_id": run_id, "saved_this_run": saved, "errors_this_run": errors})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect AI2-THOR service YOLO labels from the running backend.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--split", choices=["auto", "train", "val", "test"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--scenario-interval", type=int, default=10)
    parser.add_argument("--action-policy", choices=["none", "rotate", "random", "scripted"], default="random")
    parser.add_argument("--actions", default="RotateLeft,RotateRight,MoveAhead")
    parser.add_argument("--move-after-capture", action="store_true", default=True)
    parser.add_argument("--no-move-after-capture", action="store_false", dest="move_after_capture")
    parser.add_argument("--min-box-area", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--prefix", default="backend_service")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument(
        "--allow-empty-detections",
        action="store_true",
        help="Allow saving frames when /eval/state reports zero 2D boxes. Usually only useful for explicit negative samples.",
    )
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> CollectorConfig:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    total = float(args.train_ratio + args.val_ratio + args.test_ratio)
    if not 0.99 <= total <= 1.01:
        raise SystemExit("train/val/test ratios should sum to 1.0")
    if args.scenario_interval < 1:
        raise SystemExit("--scenario-interval must be >= 1")
    return CollectorConfig(
        dataset_root=str(args.dataset_root),
        ontology=str(args.ontology),
        base_url=str(args.base_url),
        count=int(args.count),
        split=str(args.split),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        scenarios=str(args.scenarios),
        scenario_interval=int(args.scenario_interval),
        action_policy=str(args.action_policy),
        actions=str(args.actions),
        move_after_capture=bool(args.move_after_capture),
        min_box_area=int(args.min_box_area),
        random_seed=int(args.random_seed),
        prefix=str(args.prefix),
        start_index=int(args.start_index),
        timeout=float(args.timeout),
        interval=float(args.interval),
        allow_empty_detections=bool(args.allow_empty_detections),
        init_only=bool(args.init_only),
        summary=bool(args.summary),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_config(argv)
    result = collect(config)
    json_print(result)
    return 0 if result.get("status") == "success" and int(result.get("errors_this_run", 0) or 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
