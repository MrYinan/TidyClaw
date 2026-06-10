#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///

from __future__ import annotations

import argparse
import json
import os
import sys

import requests


DEFAULT_BACKEND_URL = os.getenv("ROBOT_BACKEND_URL", "http://127.0.0.1:5000")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a simulated place action.")
    parser.add_argument(
        "--candidate-json",
        default=None,
        help="Sanitized visual receptacle candidate JSON from the decision module.",
    )
    parser.add_argument(
        "--strict-visual-grounding",
        action="store_true",
        help="Require the backend executor to ground the supplied visual candidate.",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="Run the online-safe executor precheck without executing placement.",
    )
    return parser


def parse_candidate(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    args = build_parser().parse_args()
    endpoint = "place-precheck" if bool(args.precheck_only) else "place"
    url = f"{DEFAULT_BACKEND_URL.rstrip('/')}/{endpoint}"
    payload = {
        "visual_candidate": parse_candidate(args.candidate_json),
        "strict_visual_grounding": bool(args.strict_visual_grounding),
        "interaction_contract": (
            "visual_candidate_place_precheck_v1"
            if bool(args.precheck_only)
            else "visual_candidate_grounding_v1"
        ),
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        try:
            data = response.json()
        except Exception:
            data = {
                "status": "error",
                "result_type": "error_place_invalid_response",
                "message": response.text,
                "http_status": response.status_code,
            }

        if response.status_code >= 500:
            data.setdefault("status", "error")
            data.setdefault("result_type", "error_place_service_unavailable")
            print(json.dumps(data, ensure_ascii=False))
            sys.exit(1)

        data.setdefault("http_status", response.status_code)
        print(json.dumps(data, ensure_ascii=False))

    except requests.exceptions.ConnectionError as exc:
        print(json.dumps({"status": "error", "result_type": "error_place_service_unavailable", "message": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    except requests.exceptions.Timeout as exc:
        print(json.dumps({"status": "error", "result_type": "error_place_timeout", "message": str(exc)}, ensure_ascii=False))
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"status": "error", "result_type": "error_place_unknown", "message": str(exc)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
