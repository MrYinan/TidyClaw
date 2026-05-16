#!/usr/bin/env python3
"""
Collect visual/backend alignment samples for the robot-cleaner project.

This script is intentionally non-destructive and eval-only: it does not call
/clean. Instead, it compares analyze-scene-opencv visual candidates with the
backend metadata exposed by GET /eval/state, using the same V1 clean eligibility
rule as back/robot_env.py:

- visible trash proxy object
- floor-level object
- ground distance <= 0.8
- relative position is front-center

The output is meant for debugging perception-action consistency, not for
training a final vision model.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


JsonDict = Dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE_SCRIPT = REPO_ROOT / "skills" / "analyze-scene-opencv" / "scripts" / "analyze_scene_opencv.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "memory" / "alignment-samples"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name_time() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def print_json(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False))


def request_json(
    method: str,
    url: str,
    *,
    payload: Optional[JsonDict] = None,
    timeout: int = 10,
) -> Tuple[int, JsonDict]:
    if method == "GET":
        response = requests.get(url, timeout=timeout)
    elif method == "POST":
        response = requests.post(url, json=payload or {}, timeout=timeout)
    else:
        raise ValueError(f"Unsupported method: {method}")

    try:
        data = response.json()
    except Exception:
        data = {
            "status": "error",
            "result_type": "invalid_json_response",
            "message": response.text,
        }
    return response.status_code, data


def capture_vision(base_url: str, image_path: Path, timeout: int) -> JsonDict:
    status_code, data = request_json("GET", f"{base_url}/observation", timeout=timeout)
    data["http_status"] = status_code

    base64_str = data.pop("vision_base64", "")
    if data.get("status") == "success" and base64_str:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(base64.b64decode(base64_str))
        data["image_path"] = str(image_path)

    return data


def run_analyze(image_path: Path, timeout: int) -> JsonDict:
    completed = subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT), "--image", str(image_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    stdout = completed.stdout.strip()
    try:
        data = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        data = {
            "status": "error",
            "result_type": "invalid_analyze_json",
            "stdout": stdout,
        }

    data.setdefault("status", "error" if completed.returncode else "success")
    data["returncode"] = completed.returncode
    if completed.stderr.strip():
        data["stderr"] = completed.stderr.strip()
    return data


def get_state(base_url: str, timeout: int) -> JsonDict:
    status_code, data = request_json("GET", f"{base_url}/eval/state", timeout=timeout)
    data["http_status"] = status_code
    return data


def normalize_delta(delta: float) -> float:
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def position_hint_from_delta(delta: float) -> str:
    if abs(delta) <= 15.0:
        return "front-center"
    if -60.0 <= delta < -15.0:
        return "front-left"
    if 15.0 < delta <= 60.0:
        return "front-right"
    if delta < -60.0:
        return "left-or-behind"
    return "right-or-behind"


def backend_candidates_from_state(state: JsonDict) -> List[JsonDict]:
    robot = state.get("robot") or {}
    robot_pos = robot.get("position") or {}
    robot_rot = robot.get("rotation") or {}
    rot_y = float(robot_rot.get("y", 0.0))

    candidates: List[JsonDict] = []
    for obj in state.get("visible_trash_candidates") or []:
        obj_pos = obj.get("position") or {}
        dx = float(obj_pos.get("x", 0.0)) - float(robot_pos.get("x", 0.0))
        dz = float(obj_pos.get("z", 0.0)) - float(robot_pos.get("z", 0.0))
        ground_distance = math.sqrt(dx * dx + dz * dz)
        target_angle = math.degrees(math.atan2(dx, dz))
        angle_delta = normalize_delta(target_angle - rot_y)
        position_hint = position_hint_from_delta(angle_delta)

        y = obj_pos.get("y")
        is_floor_level = y is not None and float(y) <= 0.35
        is_near = ground_distance <= 0.8
        is_front_center = position_hint == "front-center"
        clean_rule_passed = bool(is_floor_level and is_near and is_front_center)

        reject_reasons: List[str] = []
        if not is_front_center:
            reject_reasons.append("target_not_centered")
        if not is_near:
            reject_reasons.append("target_too_far")
        if not is_floor_level:
            reject_reasons.append("target_not_floor_level")

        candidate = {
            "objectId": obj.get("objectId"),
            "objectType": obj.get("objectType"),
            "visible": bool(obj.get("visible", False)),
            "position": obj_pos,
            "metadata_distance": round(float(obj.get("distance", float("inf"))), 3),
            "ground_distance": round(float(ground_distance), 3),
            "distance": round(float(ground_distance), 3),
            "position_hint": position_hint,
            "angle_delta_deg": round(float(angle_delta), 2),
            "is_floor_level": bool(is_floor_level),
            "is_near": bool(is_near),
            "is_front_center": bool(is_front_center),
            "clean_rule_passed": clean_rule_passed,
            "reject_reasons": reject_reasons,
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: item.get("ground_distance", float("inf")))
    return candidates


def visual_summary(analysis: JsonDict) -> JsonDict:
    candidates = analysis.get("trash_candidates") or []
    direct_candidates = [
        c
        for c in candidates
        if c.get("is_floor_level") and c.get("reachable") and c.get("cleanable_now")
    ]
    reachable_candidates = [
        c
        for c in candidates
        if c.get("is_floor_level") and c.get("reachable")
    ]
    return {
        "status": analysis.get("status"),
        "floor_trash_detected": bool(analysis.get("floor_trash_detected")),
        "direct_cleanable_detected": bool(direct_candidates),
        "reachable_candidate_count": len(reachable_candidates),
        "direct_candidate_count": len(direct_candidates),
        "ignored_candidate_count": len(analysis.get("ignored_candidates") or []),
        "recommended_action": analysis.get("recommended_action"),
        "open_directions": analysis.get("open_directions") or [],
        "obstacle_ahead": bool(analysis.get("obstacle_ahead")),
        "analysis_confidence": analysis.get("analysis_confidence"),
        "direct_candidates": direct_candidates[:3],
        "reachable_candidates": reachable_candidates[:3],
    }


def classify_alignment(visual: JsonDict, backend_candidates: List[JsonDict]) -> str:
    visual_cleanable = bool(visual.get("direct_cleanable_detected"))
    backend_cleanable = any(c.get("clean_rule_passed") for c in backend_candidates)

    if visual_cleanable and backend_cleanable:
        return "TP"
    if visual_cleanable and not backend_cleanable:
        return "FP"
    if not visual_cleanable and backend_cleanable:
        return "FN"
    return "TN"


def choose_next_action(
    index: int,
    sample_count: int,
    visual: JsonDict,
    *,
    last_move_failed: bool,
) -> Optional[str]:
    if index >= sample_count:
        return None

    open_directions = set(visual.get("open_directions") or [])
    obstacle_ahead = bool(visual.get("obstacle_ahead"))

    if last_move_failed:
        return "RotateRight" if index % 2 == 0 else "RotateLeft"

    if not obstacle_ahead and "forward" in open_directions and index % 3 == 0:
        return "MoveAhead"

    if obstacle_ahead:
        if "left" in open_directions:
            return "RotateLeft"
        if "right" in open_directions:
            return "RotateRight"
        return "RotateRight"

    return "RotateLeft" if index % 2 == 0 else "RotateRight"


def execute_move(base_url: str, action: str, timeout: int) -> JsonDict:
    status_code, data = request_json(
        "POST",
        f"{base_url}/move",
        payload={"action": action},
        timeout=timeout,
    )
    data["http_status"] = status_code
    return data


def update_counts(counts: JsonDict, sample: JsonDict) -> None:
    label = sample.get("alignment_label", "UNKNOWN")
    counts[label] = int(counts.get(label, 0)) + 1
    if sample.get("visual", {}).get("direct_cleanable_detected"):
        counts["visual_cleanable_frames"] = int(counts.get("visual_cleanable_frames", 0)) + 1
    if sample.get("backend", {}).get("cleanable_count", 0) > 0:
        counts["backend_cleanable_frames"] = int(counts.get("backend_cleanable_frames", 0)) + 1


def compute_metrics(counts: JsonDict) -> JsonDict:
    tp = int(counts.get("TP", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    tn = int(counts.get("TN", 0))

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else None

    return {
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


def write_report(output_dir: Path, samples: List[JsonDict], summary: JsonDict) -> None:
    lines = [
        "# Visual-Backend Alignment Samples",
        "",
        f"- created_at: {summary['created_at']}",
        f"- sample_count: {summary['sample_count']}",
        f"- method: {summary['method']}",
        "",
        "## Counts",
        "",
        f"- TP: {summary['counts'].get('TP', 0)}",
        f"- FP: {summary['counts'].get('FP', 0)}",
        f"- FN: {summary['counts'].get('FN', 0)}",
        f"- TN: {summary['counts'].get('TN', 0)}",
        f"- precision: {summary['metrics'].get('precision')}",
        f"- recall: {summary['metrics'].get('recall')}",
        f"- accuracy: {summary['metrics'].get('accuracy')}",
        "",
        "## Notable Samples",
        "",
    ]

    for sample in samples:
        label = sample.get("alignment_label")
        if label not in {"FP", "FN"}:
            continue
        visual = sample.get("visual", {})
        backend = sample.get("backend", {})
        lines.append(
            "- sample_{sid:03d}: {label}, visual_direct={vd}, "
            "backend_cleanable={bc}, image={image}".format(
                sid=sample.get("sample_id", 0),
                label=label,
                vd=visual.get("direct_cleanable_detected"),
                bc=backend.get("cleanable_count", 0),
                image=sample.get("image_path"),
            )
        )

    if lines[-1] == "## Notable Samples":
        lines.append("- none")

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect visual/backend alignment samples.")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to collect.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--no-move", action="store_true", help="Do not move between samples.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.samples > 80:
        raise SystemExit("--samples is capped at 80 for safety")

    output_dir = args.output_root / safe_name_time()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    samples: List[JsonDict] = []
    counts: JsonDict = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    jsonl_path = output_dir / "samples.jsonl"
    last_move_failed = False

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl:
        for sample_id in range(1, args.samples + 1):
            image_path = image_dir / f"sample_{sample_id:03d}.jpg"
            sample: JsonDict = {
                "sample_id": sample_id,
                "time": now_iso(),
                "image_path": str(image_path),
            }

            try:
                vision = capture_vision(args.base_url, image_path, args.timeout)
                sample["vision"] = vision

                if vision.get("status") != "success":
                    sample["status"] = "error"
                    sample["error"] = "vision_failed"
                else:
                    analysis = run_analyze(image_path, args.timeout)
                    state = get_state(args.base_url, args.timeout)
                    backend_candidates = backend_candidates_from_state(state)
                    visual = visual_summary(analysis)
                    alignment_label = classify_alignment(visual, backend_candidates)

                    sample.update(
                        {
                            "status": "success",
                            "analysis": analysis,
                            "visual": visual,
                            "backend_state": {
                                "status": state.get("status"),
                                "scene": state.get("scene"),
                                "mode": state.get("mode"),
                                "robot": state.get("robot"),
                            },
                            "backend": {
                                "visible_trash_count": len(backend_candidates),
                                "cleanable_count": sum(
                                    1 for c in backend_candidates if c.get("clean_rule_passed")
                                ),
                                "candidates": backend_candidates,
                            },
                            "alignment_label": alignment_label,
                        }
                    )
                    update_counts(counts, sample)

                    if not args.no_move:
                        action = choose_next_action(
                            sample_id,
                            args.samples,
                            visual,
                            last_move_failed=last_move_failed,
                        )
                        if action:
                            move_result = execute_move(args.base_url, action, args.timeout)
                            sample["transition_after_sample"] = {
                                "action": action,
                                "result": move_result,
                            }
                            last_move_failed = not bool(move_result.get("lastActionSuccess"))
                            time.sleep(args.sleep)

            except Exception as exc:
                sample["status"] = "error"
                sample["error"] = type(exc).__name__
                sample["message"] = str(exc)

            samples.append(sample)
            jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")
            jsonl.flush()

            if not args.quiet:
                print_json(
                    {
                        "event": "sample_collected",
                        "sample_id": sample_id,
                        "status": sample.get("status"),
                        "alignment_label": sample.get("alignment_label"),
                        "visual_direct": sample.get("visual", {}).get("direct_cleanable_detected"),
                        "backend_cleanable": sample.get("backend", {}).get("cleanable_count"),
                    }
                )

    summary = {
        "status": "success",
        "result_type": "alignment_sampling_finished",
        "created_at": now_iso(),
        "sample_count": len(samples),
        "output_dir": str(output_dir),
        "samples_jsonl": str(jsonl_path),
        "summary_json": str(output_dir / "summary.json"),
        "report_md": str(output_dir / "report.md"),
        "method": "vision image + analyze-scene-opencv compared against GET /eval/state metadata with backend V1 clean rule; /clean was not called",
        "counts": counts,
        "metrics": compute_metrics(counts),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, samples, summary)
    print_json(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
