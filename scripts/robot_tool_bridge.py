#!/usr/bin/env python3
"""Local HTTP bridge for OpenClaw robot_cleaner_* tools.

The OpenClaw plugin is intentionally not allowed to spawn Python directly. This
bridge owns the fixed backend script mappings and exposes a small local JSON API
for the plugin.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_HOST = os.getenv("ROBOT_CLEANER_TOOL_BRIDGE_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("ROBOT_CLEANER_TOOL_BRIDGE_PORT", "8765"))
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("ROBOT_CLEANER_TOOL_BRIDGE_TIMEOUT", "120"))
MAX_REQUEST_BYTES = 64 * 1024

PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_decision_turn.py"
EXECUTE_SCRIPT = REPO_ROOT / "scripts" / "execute_option.py"
STATUS_SCRIPT = REPO_ROOT / "scripts" / "robot_status.py"
REPORT_SCRIPT = REPO_ROOT / "scripts" / "robot_report.py"
STOP_SCRIPT = REPO_ROOT / "scripts" / "robot_stop.py"


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolCommand:
    tool_name: str
    script: Path
    args: list[str]
    timeout_seconds: int


class ToolRequestError(ValueError):
    def __init__(self, result_type: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.result_type = result_type
        self.status_code = status_code


def now_epoch() -> float:
    return time.time()


def hidden_startupinfo() -> Any:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def parse_json_output(text: str) -> JsonDict:
    stripped = str(text or "").strip()
    if not stripped:
        return {}
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        pass
    for line in reversed([line.strip() for line in stripped.splitlines() if line.strip()]):
        try:
            data = json.loads(line)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            continue
    return {}


def safe_timeout(value: Any, default: int = DEFAULT_TIMEOUT_SECONDS) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(1, min(numeric, 600))


def safe_task_mode(value: Any) -> str:
    text = str(value or "tidy").strip().lower()
    return text if text in {"auto", "tidy", "clean"} else "tidy"


def safe_positive_int(value: Any, default: int, *, upper: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(1, min(numeric, upper))


def safe_reason(value: Any) -> str:
    text = str(value or "user_stop").strip()
    if not text or any(char in text for char in "\r\n\t"):
        return "user_stop"
    return text[:200]


def require_option_id(value: Any) -> str:
    option_id = str(value or "")
    if not option_id or option_id != option_id.strip() or any(char in option_id for char in "\r\n\t"):
        raise ToolRequestError(
            "invalid_option_id",
            "option_id must be a non-empty single-line selected option id.",
            status_code=400,
        )
    return option_id


def command_for_request(path: str, body: JsonDict) -> ToolCommand:
    timeout = safe_timeout(body.get("timeout_seconds"))
    if path == "/tools/prepare-decision-turn":
        return ToolCommand(
            tool_name="robot_cleaner_prepare_decision_turn",
            script=PREPARE_SCRIPT,
            args=[
                "--task-mode",
                safe_task_mode(body.get("task_mode")),
                "--output",
                "memory/decision-context.json",
                "--timeout",
                str(timeout),
                "--observe-retries",
                str(safe_positive_int(body.get("observe_retries"), 1, upper=3)),
                "--max-candidates",
                str(safe_positive_int(body.get("max_candidates"), 6, upper=24)),
                "--max-options",
                str(safe_positive_int(body.get("max_options"), 12, upper=32)),
                "--format",
                "compact",
            ],
            timeout_seconds=timeout + 15,
        )
    if path == "/tools/execute-option":
        option_id = require_option_id(body.get("option_id"))
        return ToolCommand(
            tool_name="robot_cleaner_execute_option",
            script=EXECUTE_SCRIPT,
            args=[
                "--option-id",
                option_id,
                "--context",
                "memory/decision-context.json",
                "--timeout",
                str(timeout),
                "--format",
                "compact",
            ],
            timeout_seconds=timeout + 15,
        )
    if path == "/tools/status":
        return ToolCommand("robot_cleaner_status", STATUS_SCRIPT, ["--format", "compact"], timeout)
    if path == "/tools/report":
        return ToolCommand("robot_cleaner_report", REPORT_SCRIPT, ["--format", "compact"], timeout)
    if path == "/tools/stop":
        return ToolCommand(
            "robot_cleaner_stop",
            STOP_SCRIPT,
            ["--reason", safe_reason(body.get("reason")), "--format", "compact"],
            timeout,
        )
    raise ToolRequestError("unknown_tool_endpoint", f"Unknown tool endpoint: {path}", status_code=404)


def run_tool_command(command: ToolCommand) -> tuple[int, JsonDict]:
    if not command.script.exists():
        return (
            HTTPStatus.NOT_IMPLEMENTED,
            {
                "status": "error",
                "result_type": "robot_cleaner_backend_script_missing",
                "tool": command.tool_name,
                "script": str(command.script),
                "required_next": "implement_backend_script",
            },
        )

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started_at = now_epoch()
    process = subprocess.run(
        [sys.executable, str(command.script), *command.args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=command.timeout_seconds,
        env=env,
        check=False,
        startupinfo=hidden_startupinfo(),
        creationflags=hidden_creationflags(),
    )
    data = parse_json_output(process.stdout) or parse_json_output(process.stderr)
    data.setdefault("status", "success" if process.returncode == 0 else "error")
    data.setdefault("result_type", "robot_cleaner_tool_script_executed")
    data.setdefault("tool", command.tool_name)
    data["bridge"] = {
        "script": str(command.script.relative_to(REPO_ROOT)),
        "returncode": process.returncode,
        "elapsed_ms": round((now_epoch() - started_at) * 1000, 1),
        "stderr_tail": process.stderr[-2000:] if process.stderr else "",
    }
    status_code = HTTPStatus.OK if process.returncode == 0 and data.get("status") != "error" else HTTPStatus.INTERNAL_SERVER_ERROR
    return status_code, data


def error_payload(result_type: str, message: str) -> JsonDict:
    return {"status": "error", "result_type": result_type, "message": message}


class RobotToolBridgeHandler(BaseHTTPRequestHandler):
    server_version = "RobotCleanerToolBridge/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        print(f"[robot-tool-bridge] {self.address_string()} - {format % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self.write_json(
                HTTPStatus.OK,
                {
                    "status": "success",
                    "result_type": "robot_cleaner_tool_bridge_health",
                    "service": "robot-cleaner-tool-bridge",
                    "available_endpoints": [
                        "/tools/prepare-decision-turn",
                        "/tools/execute-option",
                        "/tools/status",
                        "/tools/report",
                        "/tools/stop",
                    ],
                },
            )
            return
        self.write_json(HTTPStatus.NOT_FOUND, error_payload("not_found", f"Unknown path: {path}"))

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        path = urlparse(self.path).path
        try:
            body = self.read_json_body()
            command = command_for_request(path, body)
            status_code, payload = run_tool_command(command)
            self.write_json(status_code, payload)
        except ToolRequestError as exc:
            self.write_json(exc.status_code, error_payload(exc.result_type, str(exc)))
        except subprocess.TimeoutExpired as exc:
            self.write_json(
                HTTPStatus.GATEWAY_TIMEOUT,
                {
                    "status": "error",
                    "result_type": "robot_cleaner_backend_script_timeout",
                    "message": str(exc),
                    "stdout_tail": str(exc.stdout or "")[-2000:],
                    "stderr_tail": str(exc.stderr or "")[-2000:],
                },
            )
        except Exception as exc:
            self.write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                error_payload("robot_cleaner_tool_bridge_error", str(exc)),
            )

    def read_json_body(self) -> JsonDict:
        raw_length = self.headers.get("content-length")
        length = int(raw_length or "0")
        if length > MAX_REQUEST_BYTES:
            raise ToolRequestError("request_too_large", "Request body is too large.", status_code=413)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ToolRequestError("invalid_json_body", str(exc), status_code=400) from exc
        if not isinstance(data, dict):
            raise ToolRequestError("invalid_json_body", "Request JSON must be an object.", status_code=400)
        return data

    def write_json(self, status_code: int, payload: JsonDict) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status_code))
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), RobotToolBridgeHandler)
    print(f"robot-cleaner tool bridge listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve local HTTP tools for robot-cleaner OpenClaw plugin.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(str(args.host), int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
