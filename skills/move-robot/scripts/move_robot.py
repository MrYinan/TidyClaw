#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///

import argparse
import json
import sys

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="控制 AI2-THOR 扫地机器人移动")
    parser.add_argument(
        "--action",
        required=True,
        choices=["MoveAhead", "MoveBack", "RotateLeft", "RotateRight"],
        help="执行的动作",
    )
    args = parser.parse_args()

    try:
        response = requests.post(
            "http://127.0.0.1:5000/move",
            json={"action": args.action},
            timeout=10,
        )

        try:
            data = response.json()
        except Exception:
            data = {
                "status": "error",
                "result_type": "error_move_invalid_response",
                "message": response.text,
                "http_status": response.status_code,
            }

        if response.status_code >= 500:
            data.setdefault("status", "error")
            data.setdefault("result_type", "error_move_service_unavailable")
            print(json.dumps(data, ensure_ascii=False))
            sys.exit(1)

        data.setdefault("http_status", response.status_code)
        print(json.dumps(data, ensure_ascii=False))

    except requests.exceptions.ConnectionError as e:
        print(json.dumps({"status": "error", "result_type": "error_move_service_unavailable", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print(json.dumps({"status": "error", "result_type": "error_move_timeout", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "result_type": "error_move_unknown", "message": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
