#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render quick visual previews for YOLO labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ai2thor_service_yolo_v2"


def parse_names(yaml_path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if not yaml_path.exists():
        return names
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            in_names = True
            continue
        if in_names and ":" in line:
            key, value = line.split(":", 1)
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                continue
    return names


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (220, 64, 64),
        (57, 130, 255),
        (55, 175, 90),
        (240, 160, 40),
        (165, 95, 220),
        (40, 190, 190),
        (230, 90, 150),
        (135, 150, 35),
    ]
    return palette[class_id % len(palette)]


def read_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        boxes.append((class_id, cx, cy, width, height))
    return boxes


def draw_label(
    draw: ImageDraw.ImageDraw,
    xyxy: Tuple[float, float, float, float],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text_box = draw.textbbox((x1, y1), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    label_y = max(0, y1 - text_h - 5)
    draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 5], fill=color)
    draw.text((x1 + 4, label_y + 2), text, fill=(255, 255, 255), font=font)


def render_preview(image_path: Path, label_path: Path, output_path: Path, names: Dict[int, str]) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_w, image_h = image.size
    boxes = read_label_file(label_path)

    for class_id, cx, cy, box_w, box_h in boxes:
        x1 = (cx - box_w / 2.0) * image_w
        y1 = (cy - box_h / 2.0) * image_h
        x2 = (cx + box_w / 2.0) * image_w
        y2 = (cy + box_h / 2.0) * image_h
        class_name = names.get(class_id, f"class_{class_id}")
        draw_label(draw, (x1, y1, x2, y2), f"{class_name} {class_id}", color_for_class(class_id), font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return len(boxes)


def iter_images(root: Path, splits: Sequence[str]) -> Iterable[Tuple[str, Path]]:
    for split in splits:
        for image_path in sorted((root / "images" / split).glob("*.jpg")):
            yield split, image_path


def choose_images(root: Path, splits: Sequence[str], count: int, seed: int) -> List[Tuple[str, Path]]:
    images = list(iter_images(root, splits))
    rng = random.Random(seed)
    rng.shuffle(images)
    return images[: max(0, count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw YOLO boxes on sampled dataset images.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--yaml", default="")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    yaml_path = Path(args.yaml) if args.yaml else root / "ai2thor_service.yaml"
    output_root = Path(args.output_dir) if args.output_dir else root / "preview"
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    names = parse_names(yaml_path)

    rendered = []
    for split, image_path in choose_images(root, splits, args.count, args.seed):
        label_path = root / "labels" / split / f"{image_path.stem}.txt"
        output_path = output_root / split / f"{image_path.stem}_preview.jpg"
        box_count = render_preview(image_path, label_path, output_path, names)
        rendered.append({"split": split, "image": str(image_path), "preview": str(output_path), "boxes": box_count})

    for item in rendered:
        print(item)
    print({"status": "success", "rendered": len(rendered), "output_dir": str(output_root)})


if __name__ == "__main__":
    main()
