#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run fixed service-robot regression checks.

The suite turns past bugs into repeatable checks. It loads benchmark scenarios,
captures an RGB observation, runs the YOLO perception adapter, and verifies the
online-safe decision fields. By default it does not execute pick/place actions;
it checks whether the perception/decision gate would allow the dangerous action.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


JsonDict = Dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_WEIGHTS = REPO_ROOT / "skills" / "perceive-scene-yolo" / "weights" / "best.pt"
DEFAULT_ONTOLOGY = REPO_ROOT / "configs" / "service_task_ontology_v3.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "memory" / "service-regression"
ANALYZER = REPO_ROOT / "skills" / "perceive-scene-yolo" / "scripts" / "perceive_scene_yolo.py"


@dataclass(frozen=True)
class RegressionCase:
    name: str
    bug: str
    scenario: str
    expects: Sequence[str]
    setup_actions: Sequence[str] = field(default_factory=tuple)
    description: str = ""


CASES: Dict[str, RegressionCase] = {
    "floor_apple_should_pick": RegressionCase(
        name="floor_apple_should_pick",
        bug="regression guard: clear floor pickup targets must still be actionable",
        scenario="apple_front_pick",
        expects=("floor_pickup_ready",),
        description="A floor Apple/Tomato/Potato candidate in front-center should pass the pickup gate.",
    ),
    "floor_potato_far_should_approach_or_pick": RegressionCase(
        name="floor_potato_far_should_approach_or_pick",
        bug="regression guard: small far floor Potato must not be ignored just because the bbox is tiny",
        scenario="potato_far_front_pick",
        expects=("best_pickup_approach_or_pick",),
        description="A small floor Potato should be visible and either pickup-ready or approach-needed.",
    ),
    "tabletop_decoy_should_not_pick": RegressionCase(
        name="tabletop_decoy_should_not_pick",
        bug="bug: tabletop/elevated objects were treated as floor pickup targets",
        scenario="regression_tabletop_decoy_no_pick",
        expects=("no_direct_pickup", "no_elevated_pickup_ready", "recommended_not_pick"),
        description="Elevated pickup-looking objects must be blocked in floor-only tidy mode.",
    ),
    "shelf_context_should_not_pick": RegressionCase(
        name="shelf_context_should_not_pick",
        bug="bug: shelf/rack objects were treated as pickup targets because the support was not modeled",
        scenario="regression_shelf_context_no_pick",
        expects=("support_visible", "no_elevated_pickup_ready", "recommended_not_pick"),
        description="Visible Shelf/ShelvingUnit context should prevent shelf objects from becoming direct pickups.",
    ),
    "mixed_floor_and_non_floor_prioritize_floor": RegressionCase(
        name="mixed_floor_and_non_floor_prioritize_floor",
        bug="bug class: elevated decoy must not distract from a real floor target",
        scenario="mixed_floor_and_non_floor",
        expects=("floor_pickup_ready", "no_elevated_pickup_ready"),
        description="When floor and elevated candidates coexist, only the floor target should be actionable.",
    ),
    "far_receptacle_should_approach_not_place": RegressionCase(
        name="far_receptacle_should_approach_not_place",
        bug="bug: distant CounterTop was treated as immediately placeable",
        scenario="regression_far_receptacle_no_place",
        expects=("receptacle_visible", "no_direct_place", "recommended_not_place"),
        description="A visible but distant receptacle must be approached/aligned, not used for immediate place.",
    ),
}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def request_json(method: str, url: str, *, payload: Optional[JsonDict] = None, timeout: float = 10.0) -> JsonDict:
    if method.upper() == "GET":
        response = requests.get(url, timeout=timeout)
    else:
        response = requests.post(url, json=payload or {}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Non-object JSON from {url}")
    return data


def strip_data_url_prefix(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value


def capture_observation(base_url: str, output_dir: Path, case_name: str, timeout: float) -> Path:
    data = request_json("GET", f"{base_url.rstrip('/')}/observation", timeout=timeout)
    if data.get("status") != "success":
        raise RuntimeError(f"/observation failed: {data}")
    image_b64 = str(data.get("vision_base64") or "")
    if not image_b64:
        raise RuntimeError("/observation did not return vision_base64")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{case_name}_rgb.jpg"
    image_path.write_bytes(base64.b64decode(strip_data_url_prefix(image_b64)))
    return image_path


def run_analyzer(
    *,
    image_path: Path,
    weights: Path,
    ontology: Path,
    output_dir: Path,
    case_name: str,
    save_vis: bool,
    timeout: float,
) -> JsonDict:
    cmd = [
        sys.executable,
        str(ANALYZER),
        "--image",
        str(image_path),
        "--weights",
        str(weights),
        "--ontology",
        str(ontology),
    ]
    if save_vis:
        cmd.extend(["--save-vis", str(output_dir / f"{case_name}_yolo.jpg")])
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"YOLO analyzer failed rc={completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("YOLO analyzer returned no JSON")
    return json.loads(lines[-1])


def all_candidates(analysis: JsonDict) -> List[JsonDict]:
    result: List[JsonDict] = []
    for key in ("service_candidates", "trash_candidates", "ignored_candidates", "top_obstacle_candidates"):
        value = analysis.get(key)
        if isinstance(value, list):
            result.extend(item for item in value if isinstance(item, dict))
    for key in ("best_pickup_candidate", "best_receptacle_candidate", "best_obstacle_candidate"):
        value = analysis.get(key)
        if isinstance(value, dict):
            result.append(value)
    return result


def pickup_candidates(analysis: JsonDict) -> List[JsonDict]:
    return [c for c in all_candidates(analysis) if c.get("task_semantic_class") == "pickup_target"]


def support_candidates(analysis: JsonDict) -> List[JsonDict]:
    support_labels = {"shelf", "shelvingunit", "shelving_unit", "countertop", "counter_top", "diningtable", "dining_table"}
    return [
        c
        for c in all_candidates(analysis)
        if normalize_label(c.get("label") or c.get("raw_label")) in support_labels
        or c.get("task_semantic_class") == "place_receptacle"
    ]


def receptacle_candidates(analysis: JsonDict) -> List[JsonDict]:
    return [c for c in all_candidates(analysis) if c.get("task_semantic_class") == "place_receptacle"]


def check_floor_pickup_ready(analysis: JsonDict) -> Tuple[bool, str]:
    candidate = analysis.get("best_pickup_candidate")
    if not isinstance(candidate, dict):
        return False, "missing best_pickup_candidate"
    ok = bool(
        truthy(analysis.get("direct_pickup_detected"))
        and candidate.get("task_semantic_class") == "pickup_target"
        and truthy(candidate.get("pickup_now"))
        and truthy(candidate.get("reachable"))
        and truthy(candidate.get("is_floor_level"))
        and str(candidate.get("surface_hint") or "") == "floor"
    )
    return ok, f"best_pickup={short_candidate(candidate)} direct_pickup={analysis.get('direct_pickup_detected')}"


def check_best_pickup_approach_or_pick(analysis: JsonDict) -> Tuple[bool, str]:
    candidate = analysis.get("best_pickup_candidate")
    if not isinstance(candidate, dict):
        return False, "missing best_pickup_candidate"
    ok = bool(
        candidate.get("task_semantic_class") == "pickup_target"
        and truthy(candidate.get("reachable"))
        and truthy(candidate.get("is_floor_level"))
        and str(candidate.get("surface_hint") or "") == "floor"
        and str(candidate.get("position_hint") or "") == "front-center"
        and (truthy(candidate.get("pickup_now")) or truthy(candidate.get("needs_approach")))
    )
    return ok, f"best_pickup={short_candidate(candidate)}"


def check_no_direct_pickup(analysis: JsonDict) -> Tuple[bool, str]:
    value = truthy(analysis.get("direct_pickup_detected"))
    return not value, f"direct_pickup_detected={analysis.get('direct_pickup_detected')}"


def check_recommended_not_pick(analysis: JsonDict) -> Tuple[bool, str]:
    action = str(analysis.get("recommended_action") or "")
    return action != "pick-object", f"recommended_action={action}"


def check_recommended_not_place(analysis: JsonDict) -> Tuple[bool, str]:
    action = str(analysis.get("recommended_action") or "")
    return action != "place-object", f"recommended_action={action}"


def check_no_direct_place(analysis: JsonDict) -> Tuple[bool, str]:
    value = truthy(analysis.get("direct_place_detected"))
    ready = [short_candidate(c) for c in receptacle_candidates(analysis) if truthy(c.get("place_now"))]
    return (not value and not ready), f"direct_place_detected={analysis.get('direct_place_detected')} place_now={ready}"


def check_no_elevated_pickup_ready(analysis: JsonDict) -> Tuple[bool, str]:
    offenders = []
    for candidate in pickup_candidates(analysis):
        if not truthy(candidate.get("pickup_now")):
            continue
        elevated = (
            str(candidate.get("surface_hint") or "") != "floor"
            or not truthy(candidate.get("is_floor_level"))
            or truthy(candidate.get("support_context_blocked"))
        )
        if elevated:
            offenders.append(short_candidate(candidate))
    return not offenders, f"elevated_pickup_now={offenders}"


def check_support_visible(analysis: JsonDict) -> Tuple[bool, str]:
    supports = support_candidates(analysis)
    shelf_like = [
        short_candidate(c)
        for c in supports
        if normalize_label(c.get("label") or c.get("raw_label")) in {"shelf", "shelvingunit", "shelving_unit"}
    ]
    return bool(shelf_like), f"shelf_like_supports={shelf_like[:5]}"


def check_receptacle_visible(analysis: JsonDict) -> Tuple[bool, str]:
    candidates = receptacle_candidates(analysis)
    return bool(candidates), f"receptacles={[short_candidate(c) for c in candidates[:5]]}"


EXPECTATIONS: Dict[str, Callable[[JsonDict], Tuple[bool, str]]] = {
    "floor_pickup_ready": check_floor_pickup_ready,
    "best_pickup_approach_or_pick": check_best_pickup_approach_or_pick,
    "no_direct_pickup": check_no_direct_pickup,
    "no_direct_place": check_no_direct_place,
    "recommended_not_pick": check_recommended_not_pick,
    "recommended_not_place": check_recommended_not_place,
    "no_elevated_pickup_ready": check_no_elevated_pickup_ready,
    "support_visible": check_support_visible,
    "receptacle_visible": check_receptacle_visible,
}


def short_candidate(candidate: JsonDict) -> JsonDict:
    return {
        "label": candidate.get("label"),
        "raw_label": candidate.get("raw_label"),
        "task_semantic_class": candidate.get("task_semantic_class"),
        "confidence": candidate.get("confidence"),
        "position_hint": candidate.get("position_hint"),
        "surface_hint": candidate.get("surface_hint"),
        "is_floor_level": candidate.get("is_floor_level"),
        "reachable": candidate.get("reachable"),
        "pickup_now": candidate.get("pickup_now"),
        "place_now": candidate.get("place_now"),
        "needs_approach": candidate.get("needs_approach"),
        "support_context_blocked": candidate.get("support_context_blocked"),
    }


def run_case(case: RegressionCase, args: argparse.Namespace, run_dir: Path) -> JsonDict:
    base_url = str(args.base_url).rstrip("/")
    load_result = request_json(
        "POST",
        f"{base_url}/scenario/load",
        payload={"name": case.scenario},
        timeout=args.timeout,
    )
    if load_result.get("status") != "success":
        raise RuntimeError(f"scenario load failed: {load_result}")

    for action in case.setup_actions:
        move_result = request_json("POST", f"{base_url}/move", payload={"action": action}, timeout=args.timeout)
        if move_result.get("status") != "success":
            raise RuntimeError(f"setup action failed: {action}: {move_result}")

    image_path = capture_observation(base_url, run_dir, case.name, args.timeout)
    analysis = run_analyzer(
        image_path=image_path,
        weights=Path(args.weights),
        ontology=Path(args.ontology),
        output_dir=run_dir,
        case_name=case.name,
        save_vis=not args.no_save_vis,
        timeout=args.analyze_timeout,
    )

    checks = []
    passed = True
    for expectation in case.expects:
        checker = EXPECTATIONS[expectation]
        ok, detail = checker(analysis)
        checks.append({"name": expectation, "passed": ok, "detail": detail})
        passed = passed and ok

    return {
        "case": case.name,
        "scenario": case.scenario,
        "bug": case.bug,
        "description": case.description,
        "passed": passed,
        "checks": checks,
        "image_path": str(image_path),
        "annotated_path": str(run_dir / f"{case.name}_yolo.jpg") if not args.no_save_vis else None,
        "recommended_action": analysis.get("recommended_action"),
        "best_pickup_candidate": short_candidate(analysis.get("best_pickup_candidate") or {}),
        "best_receptacle_candidate": short_candidate(analysis.get("best_receptacle_candidate") or {}),
        "notes": analysis.get("notes"),
    }


def select_cases(value: str) -> List[RegressionCase]:
    if value.strip().lower() == "all":
        return list(CASES.values())
    selected = []
    for name in [item.strip() for item in value.split(",") if item.strip()]:
        if name not in CASES:
            raise SystemExit(f"Unknown case: {name}. Use --list to inspect cases.")
        selected.append(CASES[name])
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run service robot regression checks.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--analyze-timeout", type=float, default=60.0)
    parser.add_argument("--no-save-vis", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for case in CASES.values():
            print(f"{case.name}: {case.description} [{case.scenario}]")
        return 0

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = select_cases(args.cases)

    results = []
    for case in selected_cases:
        try:
            result = run_case(case, args, run_dir)
        except Exception as exc:
            result = {
                "case": case.name,
                "scenario": case.scenario,
                "bug": case.bug,
                "description": case.description,
                "passed": False,
                "error": str(exc),
            }
        results.append(result)
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"{status} {case.name}")
        for check in result.get("checks", []):
            check_status = "PASS" if check.get("passed") else "FAIL"
            print(f"  {check_status} {check.get('name')}: {check.get('detail')}")
        if result.get("error"):
            print(f"  ERROR {result['error']}")

    summary = {
        "status": "success" if all(item.get("passed") for item in results) else "failed",
        "result_type": "service_regression_finished",
        "run_dir": str(run_dir),
        "total": len(results),
        "passed": sum(1 for item in results if item.get("passed")),
        "failed": sum(1 for item in results if not item.get("passed")),
        "results": results,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
