#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect RGB-only AI2-THOR observations for the V2 YOLO dataset.

The collector creates a YOLO-compatible dataset scaffold and records every
sample in a JSONL meta file.  By default it only calls /observation and /move,
so it is online-safe.  An optional --include-eval-state flag can store privileged
/eval/state snapshots for offline annotation/evaluation, but these snapshots
must never be fed into the online Agent.

Typical use:

    python scripts/collect_rgb_dataset.py --count 200 --action-policy random --reset-before

Output:

    datasets/trash_ai2thor_v2/
      images/{train,val,test}/xxx.jpg
      labels/{train,val,test}/xxx.txt
      meta/frames.jsonl
      meta/collection-runs.jsonl
      trash_ai2thor.yaml
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except Exception as exc:  # pragma: no cover
    print(json.dumps({"status": "error", "result_type": "error_requests_missing", "message": str(exc)}, ensure_ascii=False))
    sys.exit(1)

JsonDict = Dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "trash_ai2thor_v2"
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_CLASSES = ["cleanable_floor_trash", "non_floor_object", "obstacle"]
VALID_ACTIONS = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown"]


@dataclass
class CollectorConfig:
    dataset_root: str
    base_url: str
    count: int
    split: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    action_policy: str
    actions: str
    move_after_capture: bool
    reset_before: bool
    seed_trash_every: int
    include_eval_state: bool
    label_stub: str
    interval: float
    timeout: float
    random_seed: int
    prefix: str
    start_index: int
    dry_run: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_print(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def strip_data_url_prefix(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value


def ensure_dataset_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(root: Path, classes: Sequence[str] = DEFAULT_CLASSES) -> Path:
    yaml_path = root / "trash_ai2thor.yaml"
    names_text = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    yaml_path.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names_text}\n",
        encoding="utf-8",
    )
    return yaml_path


def append_jsonl(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


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
        data = {
            "status": "error",
            "result_type": "error_invalid_json_response",
            "http_status": response.status_code,
            "message": response.text,
        }
    data.setdefault("http_status", response.status_code)
    return data if isinstance(data, dict) else {"value": data, "http_status": response.status_code}


def choose_split(index: int, rng: random.Random, split: str, train_ratio: float, val_ratio: float) -> str:
    if split in {"train", "val", "test"}:
        return split
    value = rng.random()
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def sample_action(policy: str, actions: Sequence[str], index: int, rng: random.Random) -> Optional[str]:
    if policy == "none":
        return None
    if policy == "pass":
        return None
    if policy == "rotate":
        return "RotateLeft" if index % 2 == 0 else "RotateRight"
    if policy == "scripted":
        if not actions:
            return None
        action = actions[index % len(actions)]
        return action if action in VALID_ACTIONS else None
    if policy == "random":
        pool = list(actions) if actions else ["RotateLeft", "RotateRight", "MoveAhead", "MoveLeft", "MoveRight", "LookDown", "LookUp"]
        pool = [a for a in pool if a in VALID_ACTIONS]
        return rng.choice(pool) if pool else None
    raise ValueError(f"unknown action policy: {policy}")


def save_observation_image(observation: JsonDict, image_path: Path) -> int:
    base64_str = str(observation.get("vision_base64") or "")
    if not base64_str:
        raise RuntimeError("/observation response has no vision_base64")
    image_bytes = base64.b64decode(strip_data_url_prefix(base64_str))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    return len(image_bytes)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sample_id(prefix: str, index: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}_{index:06d}"


def build_meta_record(
    *,
    run_id: str,
    sample_id: str,
    split: str,
    index: int,
    image_path: Path,
    label_path: Optional[Path],
    image_bytes: int,
    observation: JsonDict,
    action_after_capture: Optional[str],
    action_result: Optional[JsonDict],
    eval_state: Optional[JsonDict],
) -> JsonDict:
    feedback = observation.get("last_action_feedback") if isinstance(observation.get("last_action_feedback"), dict) else {}
    record: JsonDict = {
        "schema_version": 2,
        "run_id": run_id,
        "sample_id": sample_id,
        "index": index,
        "split": split,
        "created_at": now_iso(),
        "image_path": str(image_path),
        "label_path": str(label_path) if label_path else None,
        "image_sha1": sha1_file(image_path),
        "image_bytes": image_bytes,
        "observation_contract": observation.get("observation_contract"),
        "online_safe": True,
        "last_action_feedback": {
            "action": feedback.get("action", observation.get("last_action")),
            "success": feedback.get("success", observation.get("last_action_success")),
            "error_message": feedback.get("error_message", observation.get("last_action_error", "")),
        },
        "action_after_capture": action_after_capture,
        "action_result": {
            "status": action_result.get("status") if isinstance(action_result, dict) else None,
            "result_type": action_result.get("result_type") if isinstance(action_result, dict) else None,
            "lastActionSuccess": action_result.get("lastActionSuccess") if isinstance(action_result, dict) else None,
            "error_message": action_result.get("error_message") if isinstance(action_result, dict) else None,
        }
        if action_result is not None
        else None,
        "annotation_status": "unlabeled_empty_stub" if label_path else "unlabeled_no_stub",
        "class_schema": DEFAULT_CLASSES,
    }
    if eval_state is not None:
        record["eval_state"] = eval_state
        record["eval_state_warning"] = "offline annotation/evaluation only; never use in online Agent decisions"
    return record


def write_empty_label(label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")


def summarize_dataset(root: Path) -> JsonDict:
    summary: JsonDict = {"status": "success", "result_type": "dataset_summary", "dataset_root": str(root), "splits": {}}
    class_counts = {name: 0 for name in DEFAULT_CLASSES}
    for split in ["train", "val", "test"]:
        images = sorted((root / "images" / split).glob("*.jpg"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        labeled_boxes = 0
        empty_labels = 0
        for label_file in labels:
            text = label_file.read_text(encoding="utf-8").strip()
            if not text:
                empty_labels += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    labeled_boxes += 1
                    try:
                        cls_id = int(float(parts[0]))
                        if 0 <= cls_id < len(DEFAULT_CLASSES):
                            class_counts[DEFAULT_CLASSES[cls_id]] += 1
                    except ValueError:
                        pass
        summary["splits"][split] = {
            "images": len(images),
            "label_files": len(labels),
            "labeled_boxes": labeled_boxes,
            "empty_label_files": empty_labels,
        }
    summary["class_counts"] = class_counts
    summary["yaml"] = str(root / "trash_ai2thor.yaml")
    return summary


def collect(config: CollectorConfig) -> JsonDict:
    root = Path(config.dataset_root)
    base_url = config.base_url.rstrip("/")
    ensure_dataset_dirs(root)
    yaml_path = write_dataset_yaml(root)

    rng = random.Random(config.random_seed)
    action_list = [a.strip() for a in config.actions.split(",") if a.strip()]
    run_id = datetime.now().strftime("collect_%Y%m%d_%H%M%S")
    run_record = {"schema_version": 2, "run_id": run_id, "started_at": now_iso(), "config": asdict(config), "yaml": str(yaml_path)}
    append_jsonl(root / "meta" / "collection-runs.jsonl", run_record)

    if config.reset_before:
        reset = request_json("POST", f"{base_url}/debug/reset", timeout=config.timeout)
        json_print({"event": "debug_reset", "status": reset.get("status"), "result_type": reset.get("result_type")})

    saved = 0
    errors = 0
    for local_index in range(config.count):
        sample_index = config.start_index + local_index
        split = choose_split(sample_index, rng, config.split, config.train_ratio, config.val_ratio)
        sample_id = make_sample_id(config.prefix, sample_index)
        image_path = root / "images" / split / f"{sample_id}.jpg"
        label_path = root / "labels" / split / f"{sample_id}.txt" if config.label_stub == "empty" else None

        if config.seed_trash_every > 0 and local_index % config.seed_trash_every == 0:
            seed_result = request_json("POST", f"{base_url}/debug/seed_trash", timeout=config.timeout, json_body={"distance": 0.65})
            json_print({"event": "debug_seed_trash", "index": sample_index, "status": seed_result.get("status"), "result_type": seed_result.get("result_type")})

        try:
            observation = request_json("GET", f"{base_url}/observation", timeout=config.timeout)
            if observation.get("status") != "success":
                raise RuntimeError(f"observation failed: {observation}")
            if config.dry_run:
                image_bytes = 0
            else:
                image_bytes = save_observation_image(observation, image_path)
                if label_path:
                    write_empty_label(label_path)

            eval_state = None
            if config.include_eval_state:
                eval_state = request_json("GET", f"{base_url}/eval/state", timeout=config.timeout)

            action = sample_action(config.action_policy, action_list, sample_index, rng)
            action_result = None
            if action and config.move_after_capture:
                action_result = request_json("POST", f"{base_url}/move", timeout=config.timeout, json_body={"action": action})

            if not config.dry_run:
                meta = build_meta_record(
                    run_id=run_id,
                    sample_id=sample_id,
                    split=split,
                    index=sample_index,
                    image_path=image_path,
                    label_path=label_path,
                    image_bytes=image_bytes,
                    observation=observation,
                    action_after_capture=action,
                    action_result=action_result,
                    eval_state=eval_state,
                )
                append_jsonl(root / "meta" / "frames.jsonl", meta)
            saved += 1
            json_print(
                {
                    "event": "sample_saved" if not config.dry_run else "sample_dry_run",
                    "index": sample_index,
                    "sample_id": sample_id,
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path) if label_path else None,
                    "action_after_capture": action,
                    "action_status": action_result.get("status") if isinstance(action_result, dict) else None,
                    "action_result_type": action_result.get("result_type") if isinstance(action_result, dict) else None,
                }
            )
        except Exception as exc:
            errors += 1
            append_jsonl(
                root / "meta" / "collection-errors.jsonl",
                {"run_id": run_id, "index": sample_index, "time": now_iso(), "message": str(exc)},
            )
            json_print({"event": "sample_error", "index": sample_index, "message": str(exc)})

        if config.interval > 0:
            time.sleep(config.interval)

    finish = {"schema_version": 2, "run_id": run_id, "finished_at": now_iso(), "saved": saved, "errors": errors}
    append_jsonl(root / "meta" / "collection-runs.jsonl", finish)
    summary = summarize_dataset(root)
    summary.update({"run_id": run_id, "saved_this_run": saved, "errors_this_run": errors})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect RGB-only AI2-THOR frames for YOLO V2 training.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=["auto", "train", "val", "test"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--action-policy", choices=["none", "pass", "rotate", "random", "scripted"], default="random")
    parser.add_argument("--actions", default="RotateLeft,RotateRight,MoveAhead,MoveLeft,MoveRight,LookDown,LookUp")
    parser.add_argument("--move-after-capture", action="store_true", default=True)
    parser.add_argument("--no-move-after-capture", action="store_false", dest="move_after_capture")
    parser.add_argument("--reset-before", action="store_true")
    parser.add_argument("--seed-trash-every", type=int, default=0, help="Debug only: call /debug/seed_trash every N samples. 0 disables.")
    parser.add_argument("--include-eval-state", action="store_true", help="Store /eval/state snapshots in meta for offline annotation only.")
    parser.add_argument("--label-stub", choices=["empty", "none"], default="empty")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--prefix", default="ai2thor")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--init-only", action="store_true", help="Only create dataset folders and yaml, then exit.")
    parser.add_argument("--summary", action="store_true", help="Print dataset summary and exit.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.dataset_root)
    ensure_dataset_dirs(root)
    yaml_path = write_dataset_yaml(root)

    if args.init_only:
        json_print({"status": "success", "result_type": "dataset_initialized", "dataset_root": str(root), "yaml": str(yaml_path)})
        return 0

    if args.summary:
        json_print(summarize_dataset(root))
        return 0

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    total = float(args.train_ratio + args.val_ratio + args.test_ratio)
    if not 0.99 <= total <= 1.01:
        raise SystemExit("train/val/test ratios should sum to 1.0")

    config = CollectorConfig(
        dataset_root=str(root),
        base_url=str(args.base_url),
        count=int(args.count),
        split=str(args.split),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        action_policy=str(args.action_policy),
        actions=str(args.actions),
        move_after_capture=bool(args.move_after_capture),
        reset_before=bool(args.reset_before),
        seed_trash_every=int(args.seed_trash_every),
        include_eval_state=bool(args.include_eval_state),
        label_stub=str(args.label_stub),
        interval=float(args.interval),
        timeout=float(args.timeout),
        random_seed=int(args.random_seed),
        prefix=str(args.prefix),
        start_index=int(args.start_index),
        dry_run=bool(args.dry_run),
    )
    summary = collect(config)
    json_print(summary)
    return 0 if summary.get("errors_this_run", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
