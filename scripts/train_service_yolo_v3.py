#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect and train the long-horizon service YOLO model.

This script keeps the retraining route reproducible:

1. initialize a YOLO dataset from configs/service_task_ontology_v3.json
2. optionally collect AI2-THOR RGB frames with offline instance labels
3. train YOLO from yolo11n.pt by default

It deliberately writes into a new v3 dataset folder so existing v2 weights and
labels remain usable until the new model is validated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONTOLOGY = REPO_ROOT / "configs" / "service_task_ontology_v3.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ai2thor_service_yolo_v3"
DEFAULT_MODEL = REPO_ROOT / "yolo11n.pt"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "detect"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_SCENARIOS = "all"
DATASET_CACHE_VERSION = "1.0.3"


def run_command(args: Sequence[str], *, dry_run: bool = False) -> int:
    printable = " ".join(str(item) for item in args)
    print(printable, flush=True)
    if dry_run:
        return 0
    return subprocess.run(list(args), cwd=str(REPO_ROOT)).returncode


def json_print(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)


def image_files_for_split(dataset_root: Path, split: str) -> List[Path]:
    image_dir = dataset_root / "images" / split
    if not image_dir.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)


def label_file_for_image(dataset_root: Path, split: str, image_path: Path) -> Path:
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def dataset_hash(paths: Sequence[Path]) -> str:
    import hashlib

    size = sum(path.stat().st_size for path in paths if path.exists())
    h = hashlib.sha256(str(size).encode())
    h.update("".join(str(path) for path in paths).encode())
    return h.hexdigest()


def read_yolo_label(label_path: Path, class_count: int):
    import numpy as np

    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), []

    rows = []
    warnings = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            warnings.append(f"{label_path}:{line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            warnings.append(f"{label_path}:{line_number}: non-numeric label row")
            continue
        cls = int(values[0])
        if cls < 0 or cls >= class_count:
            warnings.append(f"{label_path}:{line_number}: class id {cls} out of range 0..{class_count - 1}")
            continue
        if any(value < 0.0 or value > 1.0 for value in values[1:]):
            warnings.append(f"{label_path}:{line_number}: bbox values are not normalized")
            continue
        rows.append(values)

    if not rows:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), warnings
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:, 0:1], arr[:, 1:5], warnings


def prebuild_ultralytics_label_caches(dataset_root: Path, ontology_path: Path) -> None:
    """Build YOLO label caches sequentially to avoid Windows multiprocessing pipe failures."""
    import numpy as np
    from PIL import Image

    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    class_count = len(ontology.get("classes") or ontology.get("object_classes") or [])
    if class_count <= 0:
        raise RuntimeError(f"No classes found in ontology: {ontology_path}")

    for split in ("train", "val", "test"):
        image_paths = image_files_for_split(dataset_root, split)
        if not image_paths:
            continue

        labels = []
        msgs = []
        missing = 0
        empty = 0
        corrupt = 0
        label_paths = []

        for image_path in image_paths:
            label_path = label_file_for_image(dataset_root, split, image_path)
            label_paths.append(label_path)
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:
                corrupt += 1
                msgs.append(f"{image_path}: failed to read image: {exc}")
                continue

            if not label_path.exists():
                missing += 1
            cls, bboxes, label_warnings = read_yolo_label(label_path, class_count)
            if len(cls) == 0:
                empty += 1
            msgs.extend(label_warnings)
            labels.append(
                {
                    "im_file": str(image_path),
                    "shape": (height, width),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        cache_path = dataset_root / "labels" / f"{split}.cache"
        cache = {
            "labels": labels,
            "hash": dataset_hash(label_paths + image_paths),
            "results": (len(labels), missing, empty, corrupt, len(image_paths)),
            "msgs": msgs,
            "version": DATASET_CACHE_VERSION,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            np.save(handle, cache)
        print(
            json.dumps(
                {
                    "event": "ultralytics_label_cache_prebuilt",
                    "split": split,
                    "cache_path": str(cache_path),
                    "images": len(image_paths),
                    "empty": empty,
                    "missing": missing,
                    "corrupt": corrupt,
                    "warnings": len(msgs),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and train service YOLO v3.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--collect-count", type=int, default=1200)
    parser.add_argument("--scenario-interval", type=int, default=8)
    parser.add_argument("--action-policy", choices=["none", "rotate", "random", "scripted"], default="random")
    parser.add_argument("--actions", default="RotateLeft,RotateRight,MoveAhead,MoveBack")
    parser.add_argument("--min-box-area", type=int, default=24)
    parser.add_argument("--prefix", default="service_v3")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.03)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--name", default="")
    parser.add_argument("--skip-prebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root)
    yaml_path = dataset_root / "ai2thor_service.yaml"
    run_name = args.name or f"service_v3_general_{datetime.now().strftime('%Y%m%d_%H%M')}"

    init_cmd: List[str] = [
        sys.executable,
        "scripts\\collect_ai2thor_service_dataset_backend.py",
        "--dataset-root",
        str(dataset_root),
        "--ontology",
        str(args.ontology),
        "--init-only",
    ]
    code = run_command(init_cmd, dry_run=args.dry_run)
    if code != 0:
        return code

    if not args.skip_collect and args.collect_count > 0:
        collect_cmd: List[str] = [
            sys.executable,
            "scripts\\collect_ai2thor_service_dataset_backend.py",
            "--dataset-root",
            str(dataset_root),
            "--ontology",
            str(args.ontology),
            "--base-url",
            str(args.base_url),
            "--count",
            str(args.collect_count),
            "--scenarios",
            str(args.scenarios),
            "--scenario-interval",
            str(args.scenario_interval),
            "--action-policy",
            str(args.action_policy),
            "--actions",
            str(args.actions),
            "--min-box-area",
            str(args.min_box_area),
            "--prefix",
            str(args.prefix),
            "--start-index",
            str(args.start_index),
            "--timeout",
            str(args.timeout),
            "--interval",
            str(args.interval),
        ]
        code = run_command(collect_cmd, dry_run=args.dry_run)
        if code != 0:
            return code

    if not args.skip_train:
        if not args.skip_prebuild_cache:
            prebuild_ultralytics_label_caches(dataset_root, Path(args.ontology))

        train_cmd: List[str] = [
            "yolo",
            "detect",
            "train",
            f"model={args.model}",
            f"data={yaml_path}",
            f"imgsz={args.imgsz}",
            f"epochs={args.epochs}",
            f"batch={args.batch}",
            f"device={args.device}",
            f"workers={args.workers}",
            f"patience={args.patience}",
            "pretrained=True",
            "cache=False",
            "close_mosaic=10",
            f"project={DEFAULT_PROJECT}",
            f"name={run_name}",
            "exist_ok=True",
        ]
        code = run_command(train_cmd, dry_run=args.dry_run)
        if code != 0:
            return code

    json_print(
        {
            "status": "success",
            "result_type": "service_yolo_v3_pipeline_finished",
            "dataset_root": str(dataset_root),
            "yaml": str(yaml_path),
            "run_name": run_name,
            "weights": str(DEFAULT_PROJECT / run_name / "weights" / "best.pt"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
