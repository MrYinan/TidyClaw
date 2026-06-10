#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect an AI2-THOR household-service YOLO dataset.

This script is offline tooling. It may use AI2-THOR metadata and
instance_detections2D to auto-label RGB frames, but the online Agent must still
use only RGB perception plus sanitized action feedback.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ai2thor_service_yolo_v2"
DEFAULT_ONTOLOGY = REPO_ROOT / "configs" / "service_task_ontology_v2.json"
DEFAULT_SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
DEFAULT_CLASSES = [
    "Book",
    "Apple",
    "Tomato",
    "Potato",
    "Mug",
    "Bowl",
    "Plate",
    "RemoteControl",
    "DiningTable",
    "CoffeeTable",
    "CounterTop",
    "Sink",
    "Chair",
    "Sofa",
    "Cabinet",
    "Bed",
]
MOVE_ACTIONS = ["RotateLeft", "RotateRight", "MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "LookDown", "LookUp"]


@dataclass
class CollectorConfig:
    dataset_root: str
    ontology: str
    scenes: str
    count: int
    width: int
    height: int
    grid_size: float
    visibility_distance: float
    split: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    min_box_area: int
    action_policy: str
    prefix: str
    start_index: int
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


def load_classes(ontology_path: Path) -> List[str]:
    if not ontology_path.exists():
        return list(DEFAULT_CLASSES)
    try:
        data = json.loads(ontology_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return list(DEFAULT_CLASSES)
    classes = data.get("object_classes") if isinstance(data, dict) else None
    if not isinstance(classes, list):
        return list(DEFAULT_CLASSES)
    cleaned = [str(item).strip() for item in classes if str(item).strip()]
    return cleaned or list(DEFAULT_CLASSES)


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


def save_frame(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(frame).save(path)
        return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio  # type: ignore

        imageio.imwrite(path, frame)
        return
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        cv2.imwrite(str(path), frame[:, :, ::-1])
        return
    except Exception as exc:
        raise RuntimeError("Install pillow, imageio, or opencv-python to save frames.") from exc


def object_type_by_id(event: Any) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for obj in event.metadata.get("objects", []) or []:
        object_id = str(obj.get("objectId") or "")
        object_type = str(obj.get("objectType") or "")
        if object_id and object_type:
            mapping[object_id] = object_type
    return mapping


def get_detections_2d(event: Any) -> Dict[str, Sequence[float]]:
    detections = getattr(event, "instance_detections2D", None)
    if isinstance(detections, dict):
        return detections
    metadata_detections = event.metadata.get("instanceDetections2D")
    if isinstance(metadata_detections, dict):
        return metadata_detections
    return {}


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


def labels_from_event(
    event: Any,
    *,
    classes: Sequence[str],
    width: int,
    height: int,
    min_box_area: int,
) -> Tuple[List[str], List[JsonDict]]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    object_types = object_type_by_id(event)
    detections = get_detections_2d(event)
    lines: List[str] = []
    boxes_meta: List[JsonDict] = []

    for object_id, raw_box in detections.items():
        object_type = object_types.get(str(object_id))
        if object_type not in class_to_id:
            continue
        clipped = clip_box(raw_box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        box_area = int((x2 - x1) * (y2 - y1))
        if box_area < min_box_area:
            continue
        x_center = ((x1 + x2) / 2.0) / float(width)
        y_center = ((y1 + y2) / 2.0) / float(height)
        box_w = (x2 - x1) / float(width)
        box_h = (y2 - y1) / float(height)
        class_id = class_to_id[object_type]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        boxes_meta.append(
            {
                "object_type": object_type,
                "object_id": str(object_id),
                "class_id": class_id,
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area": box_area,
            }
        )
    return lines, boxes_meta


def sample_action(policy: str, rng: random.Random) -> Optional[str]:
    if policy == "none":
        return None
    if policy == "rotate":
        return rng.choice(["RotateLeft", "RotateRight"])
    if policy == "random":
        return rng.choice(MOVE_ACTIONS)
    raise ValueError(f"Unknown action policy: {policy}")


def maybe_move(controller: Any, action: Optional[str]) -> None:
    if not action:
        return
    kwargs: JsonDict = {"action": action}
    if action == "MoveAhead":
        kwargs["moveMagnitude"] = 0.25
    controller.step(**kwargs)


def summarize(root: Path, classes: Sequence[str]) -> JsonDict:
    class_counts = {name: 0 for name in classes}
    split_summary: JsonDict = {}
    for split in ["train", "val", "test"]:
        images = sorted((root / "images" / split).glob("*.jpg"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        box_count = 0
        for label_file in labels:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(classes):
                    class_counts[classes[class_id]] += 1
                    box_count += 1
        split_summary[split] = {"images": len(images), "labels": len(labels), "boxes": box_count}
    return {
        "status": "success",
        "result_type": "ai2thor_service_dataset_summary",
        "dataset_root": str(root),
        "yaml": str(root / "ai2thor_service.yaml"),
        "splits": split_summary,
        "class_counts": class_counts,
    }


def collect(config: CollectorConfig) -> JsonDict:
    root = Path(config.dataset_root)
    classes = load_classes(Path(config.ontology))
    ensure_dataset_dirs(root)
    yaml_path = write_yaml(root, classes)

    if config.init_only:
        return {
            "status": "success",
            "result_type": "dataset_initialized",
            "dataset_root": str(root),
            "yaml": str(yaml_path),
            "classes": classes,
        }
    if config.summary:
        return summarize(root, classes)

    try:
        from ai2thor.controller import Controller  # type: ignore
    except Exception as exc:
        return {
            "status": "error",
            "result_type": "error_ai2thor_missing",
            "message": "Install ai2thor in the environment that will run dataset collection.",
            "detail": str(exc),
        }

    rng = random.Random(config.random_seed)
    scenes = [scene.strip() for scene in config.scenes.split(",") if scene.strip()]
    if not scenes:
        scenes = list(DEFAULT_SCENES)

    run_id = datetime.now().strftime("ai2thor_service_%Y%m%d_%H%M%S")
    append_jsonl(
        root / "meta" / "collection-runs.jsonl",
        {
            "schema_version": 2,
            "run_id": run_id,
            "started_at": now_iso(),
            "config": asdict(config),
            "classes": classes,
            "yaml": str(yaml_path),
            "offline_label_source": "AI2-THOR instance_detections2D",
            "online_policy": "Do not use metadata in online Agent decisions.",
        },
    )

    controller = Controller(
        scene=scenes[0],
        width=config.width,
        height=config.height,
        gridSize=config.grid_size,
        visibilityDistance=config.visibility_distance,
        renderInstanceSegmentation=True,
    )

    saved = 0
    errors = 0
    try:
        event = controller.last_event
        for local_index in range(config.count):
            sample_index = config.start_index + local_index
            scene = scenes[local_index % len(scenes)]
            if local_index == 0 or controller.last_event.metadata.get("sceneName") != scene:
                event = controller.reset(scene=scene)
                controller.step(action="Pass")
                event = controller.last_event

            split = choose_split(sample_index, rng, config.split, config.train_ratio, config.val_ratio)
            sample_id = make_sample_id(config.prefix, sample_index)
            image_path = root / "images" / split / f"{sample_id}.jpg"
            label_path = root / "labels" / split / f"{sample_id}.txt"

            try:
                save_frame(event.frame, image_path)
                lines, boxes_meta = labels_from_event(
                    event,
                    classes=classes,
                    width=config.width,
                    height=config.height,
                    min_box_area=config.min_box_area,
                )
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                append_jsonl(
                    root / "meta" / "frames.jsonl",
                    {
                        "schema_version": 2,
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "created_at": now_iso(),
                        "scene": scene,
                        "split": split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "box_count": len(lines),
                        "boxes": boxes_meta,
                        "offline_label_source": "AI2-THOR instance_detections2D",
                    },
                )
                saved += 1
                json_print(
                    {
                        "event": "sample_saved",
                        "index": sample_index,
                        "scene": scene,
                        "split": split,
                        "image_path": str(image_path),
                        "box_count": len(lines),
                    }
                )
            except Exception as exc:
                errors += 1
                append_jsonl(root / "meta" / "collection-errors.jsonl", {"time": now_iso(), "index": sample_index, "message": str(exc)})
                json_print({"event": "sample_error", "index": sample_index, "message": str(exc)})

            action = sample_action(config.action_policy, rng)
            maybe_move(controller, action)
            event = controller.last_event
    finally:
        try:
            controller.stop()
        except Exception:
            pass

    append_jsonl(root / "meta" / "collection-runs.jsonl", {"run_id": run_id, "finished_at": now_iso(), "saved": saved, "errors": errors})
    summary = summarize(root, classes)
    summary.update({"run_id": run_id, "saved_this_run": saved, "errors_this_run": errors})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect AI2-THOR household-service YOLO labels offline.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--split", choices=["auto", "train", "val", "test"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-box-area", type=int, default=64)
    parser.add_argument("--action-policy", choices=["none", "rotate", "random"], default="random")
    parser.add_argument("--prefix", default="ai2thor_service")
    parser.add_argument("--start-index", type=int, default=0)
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
    return CollectorConfig(
        dataset_root=str(args.dataset_root),
        ontology=str(args.ontology),
        scenes=str(args.scenes),
        count=int(args.count),
        width=int(args.width),
        height=int(args.height),
        grid_size=float(args.grid_size),
        visibility_distance=float(args.visibility_distance),
        split=str(args.split),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        random_seed=int(args.random_seed),
        min_box_area=int(args.min_box_area),
        action_policy=str(args.action_policy),
        prefix=str(args.prefix),
        start_index=int(args.start_index),
        init_only=bool(args.init_only),
        summary=bool(args.summary),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_config(argv)
    result = collect(config)
    json_print(result)
    return 0 if result.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
