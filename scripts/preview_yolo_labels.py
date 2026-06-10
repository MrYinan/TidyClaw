#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render annotated preview images from a YOLO-format dataset."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "ai2thor_service_yolo_v3"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    values: Tuple[float, ...]
    line_no: int

    @property
    def is_box(self) -> bool:
        return len(self.values) == 4

    @property
    def is_polygon(self) -> bool:
        return len(self.values) >= 6 and len(self.values) % 2 == 0


def find_default_yaml(dataset_root: Path) -> Path:
    for name in ("data.yaml", "dataset.yaml", "ai2thor_service.yaml"):
        yaml_path = dataset_root / name
        if yaml_path.exists():
            return yaml_path

    yaml_files = sorted([*dataset_root.glob("*.yaml"), *dataset_root.glob("*.yml")])
    if yaml_files:
        return yaml_files[0]
    return dataset_root / "data.yaml"


def parse_names(yaml_path: Path) -> Dict[int, str]:
    if not yaml_path.exists():
        return {}

    text = yaml_path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        raw_names = data.get("names", {})
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, list):
            return {idx: str(value) for idx, value in enumerate(raw_names)}
    except Exception:
        pass

    names: Dict[int, str] = {}
    in_names = False
    for raw_line in text.splitlines():
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
        (255, 115, 70),
        (80, 120, 35),
        (30, 165, 210),
        (180, 70, 150),
    ]
    return palette[class_id % len(palette)]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def read_label_file(label_path: Path) -> List[YoloLabel]:
    labels: List[YoloLabel] = []
    if not label_path.exists():
        return labels

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = tuple(float(value) for value in parts[1:])
        except ValueError:
            continue
        if len(values) == 4 or (len(values) >= 6 and len(values) % 2 == 0):
            labels.append(YoloLabel(class_id=class_id, values=values, line_no=line_no))
    return labels


def label_path_for_image(dataset_root: Path, split: str, image_path: Path) -> Path:
    image_split_root = dataset_root / "images" / split
    try:
        relative_path = image_path.relative_to(image_split_root)
    except ValueError:
        relative_path = Path(image_path.name)
    return dataset_root / "labels" / split / relative_path.with_suffix(".txt")


def iter_images(dataset_root: Path, splits: Sequence[str]) -> Iterable[Tuple[str, Path, Path]]:
    for split in splits:
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, image_path, label_path_for_image(dataset_root, split, image_path)


def choose_images(
    dataset_root: Path,
    splits: Sequence[str],
    count: int,
    seed: int,
    include_empty: bool,
) -> List[Tuple[str, Path, Path]]:
    images = list(iter_images(dataset_root, splits))
    if not include_empty:
        images = [item for item in images if item[2].exists() and item[2].stat().st_size > 0]

    rng = random.Random(seed)
    rng.shuffle(images)
    if count <= 0:
        return images
    return images[:count]


def draw_text_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    image_w: int,
) -> None:
    x, y = xy
    text_box = draw.textbbox((x, y), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x = clamp(x, 0, max(0, image_w - text_w - 8))
    y = max(0, y - text_h - 5)
    draw.rectangle([x, y, x + text_w + 8, y + text_h + 5], fill=color)
    draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)


def draw_box(
    draw: ImageDraw.ImageDraw,
    label: YoloLabel,
    image_w: int,
    image_h: int,
    names: Dict[int, str],
    font: ImageFont.ImageFont,
    line_width: int,
) -> None:
    cx, cy, box_w, box_h = label.values
    x1 = clamp((cx - box_w / 2.0) * image_w, 0, image_w - 1)
    y1 = clamp((cy - box_h / 2.0) * image_h, 0, image_h - 1)
    x2 = clamp((cx + box_w / 2.0) * image_w, 0, image_w - 1)
    y2 = clamp((cy + box_h / 2.0) * image_h, 0, image_h - 1)
    color = color_for_class(label.class_id)
    class_name = names.get(label.class_id, f"class_{label.class_id}")

    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    draw_text_label(draw, (x1, y1), f"{class_name} {label.class_id}", color, font, image_w)


def draw_polygon(
    draw: ImageDraw.ImageDraw,
    label: YoloLabel,
    image_w: int,
    image_h: int,
    names: Dict[int, str],
    font: ImageFont.ImageFont,
    line_width: int,
) -> None:
    points = []
    values = label.values
    for idx in range(0, len(values), 2):
        x = clamp(values[idx] * image_w, 0, image_w - 1)
        y = clamp(values[idx + 1] * image_h, 0, image_h - 1)
        points.append((x, y))

    if len(points) < 3:
        return

    color = color_for_class(label.class_id)
    class_name = names.get(label.class_id, f"class_{label.class_id}")
    draw.line([*points, points[0]], fill=color, width=line_width, joint="curve")
    label_x = min(point[0] for point in points)
    label_y = min(point[1] for point in points)
    draw_text_label(draw, (label_x, label_y), f"{class_name} {label.class_id}", color, font, image_w)


def render_preview(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    names: Dict[int, str],
    line_width: int,
) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_w, image_h = image.size
    labels = read_label_file(label_path)

    for label in labels:
        if label.is_box:
            draw_box(draw, label, image_w, image_h, names, font, line_width)
        elif label.is_polygon:
            draw_polygon(draw, label, image_w, image_h, names, font, line_width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return len(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO labels on sampled dataset images.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="YOLO dataset root.")
    parser.add_argument("--yaml", default="", help="Dataset yaml path. Defaults to data.yaml/dataset.yaml/ai2thor_service.yaml.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--count", type=int, default=12, help="Number of previews to render. Use 0 to render all.")
    parser.add_argument("--seed", type=int, default=7, help="Random sampling seed.")
    parser.add_argument("--output-dir", default="", help="Preview output directory.")
    parser.add_argument("--include-empty", action="store_true", help="Also render images with missing or empty labels.")
    parser.add_argument("--line-width", type=int, default=3, help="Box/polygon line width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    yaml_path = Path(args.yaml) if args.yaml else find_default_yaml(dataset_root)
    output_root = Path(args.output_dir) if args.output_dir else dataset_root / "preview_labels"
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    names = parse_names(yaml_path)

    rendered = []
    for split, image_path, label_path in choose_images(
        dataset_root=dataset_root,
        splits=splits,
        count=args.count,
        seed=args.seed,
        include_empty=args.include_empty,
    ):
        try:
            relative_image = image_path.relative_to(dataset_root / "images" / split)
        except ValueError:
            relative_image = Path(image_path.name)
        output_path = output_root / split / relative_image.with_name(f"{image_path.stem}_labels.jpg")
        label_count = render_preview(image_path, label_path, output_path, names, max(1, args.line_width))
        rendered.append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "preview": str(output_path),
                "labels": label_count,
            }
        )

    for item in rendered:
        print(item)
    print({"status": "success", "rendered": len(rendered), "output_dir": str(output_root), "yaml": str(yaml_path)})


if __name__ == "__main__":
    main()
