from __future__ import annotations

import base64
import os
from typing import Any, Dict, Tuple

from flask import Flask, Response, jsonify, request

from back.api_sanitizers import (
    sanitize_clean_result,
    sanitize_inventory_result,
    sanitize_move_result,
    sanitize_service_action_result,
)
from back.eval_adapter import EvalAdapter
from back.observation_adapter import ObservationAdapter
from back.robot_env import RobotEnvironment
from back.scenario_manager import ScenarioManager


JsonDict = Dict[str, Any]

app = Flask(__name__)

print("=== 🤖 正在拉起 AI2-THOR 后台常驻环境 ===", flush=True)
env = RobotEnvironment()
obs_adapter = ObservationAdapter(env)
eval_adapter = EvalAdapter(env)
scenario_manager = ScenarioManager()
print("=== 🟢 环境就绪，API 监听服务器已启动 ===", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bool_query(name: str, default: bool = False) -> bool:
    value = request.args.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def json_error(result_type: str, message: str, status_code: int = 500) -> Tuple[Response, int]:
    return jsonify({"status": "error", "result_type": result_type, "message": message}), status_code


# ---------------------------------------------------------------------------
# V2 observation/evaluation boundary
# ---------------------------------------------------------------------------


@app.route("/observation", methods=["GET"])
def observation():
    """Online RGB-only observation endpoint.

    Agent/Skill code should use this endpoint in V2. It returns first-person RGB
    plus last-action feedback only. It intentionally excludes privileged state:
    exact pose, object metadata, depth and segmentation.
    """
    try:
        return jsonify(obs_adapter.get_rgb_observation())
    except Exception as e:
        return json_error("error_observation_failed", str(e), 500)


@app.route("/eval/state", methods=["GET"])
def eval_state():
    """Offline evaluation/debug endpoint with privileged simulator state."""
    try:
        return jsonify(eval_adapter.get_eval_state())
    except Exception as e:
        return json_error("error_eval_state_failed", str(e), 500)


@app.route("/eval/map", methods=["GET"])
def eval_map():
    """Offline-only AI2-THOR reachable-position map endpoint."""
    try:
        return jsonify(env.get_groundtruth_map_snapshot())
    except Exception as e:
        return json_error("error_eval_map_failed", str(e), 500)


@app.route("/vision", methods=["GET"])
def get_vision():
    """Backward-compatible alias for /observation."""
    try:
        data = obs_adapter.get_rgb_observation()
        data["result_type"] = "rgb_observation_deprecated_vision_alias"
        data["deprecated"] = True
        data["replacement_endpoint"] = "/observation"
        return jsonify(data)
    except Exception as e:
        return json_error("error_vision_failed", str(e), 500)


@app.route("/state", methods=["GET"])
def state():
    """Backward-compatible alias for /eval/state. Eval/debug only."""
    try:
        data = eval_adapter.get_eval_state()
        data["deprecated"] = True
        data["replacement_endpoint"] = "/eval/state"
        return jsonify(data)
    except Exception as e:
        return json_error("error_state_failed", str(e), 500)


# ---------------------------------------------------------------------------
# V2 benchmark scenario endpoints, offline/debug control only
# ---------------------------------------------------------------------------


@app.route("/scenario/list", methods=["GET"])
def scenario_list():
    """List V2 benchmark scenarios.

    This endpoint is for experiment setup, not online Agent perception.
    """
    try:
        data = scenario_manager.list_scenarios()
        data["online_safe"] = False
        data["usage_scope"] = "offline_benchmark_setup_only"
        return jsonify(data)
    except Exception as e:
        return json_error("error_scenario_list_failed", str(e), 500)


@app.route("/scenario/current", methods=["GET"])
def scenario_current():
    """Return current backend scenario metadata. Offline/debug only."""
    try:
        return jsonify(
            {
                "status": "success",
                "schema_version": 2,
                "result_type": "current_scenario",
                "scene": env.scene,
                "mode": env.mode,
                "scenario": env.current_scenario,
                "seeded_objects": env.seeded_objects,
                "online_safe": False,
                "usage_scope": "offline_benchmark_debug_only",
            }
        )
    except Exception as e:
        return json_error("error_current_scenario_failed", str(e), 500)


@app.route("/scenario/load", methods=["POST"])
def scenario_load():
    """Load a named benchmark scenario or an inline scenario object.

    Request examples:
      {"name": "front_floor_trash"}
      {"scenario": { ... inline scenario config ... }}

    This is intentionally an offline/debug endpoint. The online Agent should not
    call it during ordinary decision-making.
    """
    try:
        data = request.get_json(silent=True) or {}
        scenario = data.get("scenario") if isinstance(data.get("scenario"), dict) else None
        name = data.get("name")
        if scenario is None:
            if not name:
                return json_error("error_missing_scenario_name", "Provide 'name' or inline 'scenario'.", 400)
            scenario = scenario_manager.get_scenario(str(name))
            if scenario is None:
                return json_error("error_unknown_scenario", f"Unknown scenario: {name}", 404)

        result = env.apply_scenario(scenario)
        result["online_safe"] = False
        result["usage_scope"] = "offline_benchmark_setup_only"
        return jsonify(result)
    except Exception as e:
        return json_error("error_scenario_load_failed", str(e), 500)


# ---------------------------------------------------------------------------
# Action endpoints
# ---------------------------------------------------------------------------


@app.route("/move", methods=["POST"])
def move():
    """Control movement and return online-safe action feedback.

    By default this endpoint does not expose exact position/rotation. For
    controlled offline debug only, call /move?include_eval=1 to get raw result.
    """
    try:
        data = request.get_json(silent=True) or {}
        action = data.get("action")
        valid_actions = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
        if action not in valid_actions:
            return (
                jsonify(
                    {
                        "status": "error",
                        "result_type": "error_invalid_action",
                        "message": f"无效动作: {action}",
                        "valid_actions": sorted(valid_actions),
                    }
                ),
                400,
            )

        result = env.execute_action(action)
        if bool_query("include_eval", default=False):
            result = dict(result)
            result["online_safe"] = False
            result["usage_scope"] = "offline_evaluation_debug_only"
            return jsonify(result)
        return jsonify(sanitize_move_result(result))
    except Exception as e:
        return json_error("error_move_service_failed", str(e), 500)


@app.route("/clean", methods=["POST"])
def clean():
    """Simulated cleaning actuator.

    The backend uses AI2-THOR state internally to implement the actuator. The
    default response is sanitized so Agent code does not receive objectId/pose
    metadata. For offline debugging, call /clean?include_eval=1.
    """
    try:
        print("[/clean] 收到清扫请求", flush=True)
        result = env.clean_trash_in_front()
        print(f"[/clean] result={result}", flush=True)
        if bool_query("include_eval", default=False):
            result = dict(result)
            result["online_safe"] = False
            result["usage_scope"] = "offline_evaluation_debug_only"
            return jsonify(result)
        return jsonify(sanitize_clean_result(result))
    except Exception as e:
        print(f"[/clean] exception={e}", flush=True)
        return json_error("error_clean_service_failed", str(e), 500)


@app.route("/pick", methods=["POST"])
def pick():
    """Pickup actuator for household service tasks."""
    try:
        payload = request.get_json(silent=True) or {}
        payload = payload if isinstance(payload, dict) else {}
        result = env.pickup_object_in_front(
            visual_candidate=payload.get("visual_candidate"),
            strict_visual_grounding=bool(payload.get("strict_visual_grounding", False)),
        )
        if bool_query("include_eval", default=False):
            result = dict(result)
            result["online_safe"] = False
            result["usage_scope"] = "offline_evaluation_debug_only"
            return jsonify(result)
        return jsonify(sanitize_service_action_result(result, action_name="pickup"))
    except Exception as e:
        return json_error("error_pick_service_failed", str(e), 500)


@app.route("/place", methods=["POST"])
def place():
    """Place the currently held object onto a visible receptacle."""
    try:
        payload = request.get_json(silent=True) or {}
        payload = payload if isinstance(payload, dict) else {}
        result = env.place_held_object(
            visual_candidate=payload.get("visual_candidate"),
            strict_visual_grounding=bool(payload.get("strict_visual_grounding", False)),
        )
        if bool_query("include_eval", default=False):
            result = dict(result)
            result["online_safe"] = False
            result["usage_scope"] = "offline_evaluation_debug_only"
            return jsonify(result)
        return jsonify(sanitize_service_action_result(result, action_name="place"))
    except Exception as e:
        return json_error("error_place_service_failed", str(e), 500)


@app.route("/place-precheck", methods=["POST"])
def place_precheck():
    """Online-safe dry precheck for a visual place candidate."""
    try:
        payload = request.get_json(silent=True) or {}
        payload = payload if isinstance(payload, dict) else {}
        result = env.precheck_place_candidate(
            visual_candidate=payload.get("visual_candidate"),
            strict_visual_grounding=bool(payload.get("strict_visual_grounding", False)),
        )
        if bool_query("include_eval", default=False):
            result = dict(result)
            result["online_safe"] = False
            result["usage_scope"] = "offline_evaluation_debug_only"
            return jsonify(result)
        return jsonify(sanitize_service_action_result(result, action_name="place_precheck"))
    except Exception as e:
        return json_error("error_place_precheck_service_failed", str(e), 500)


@app.route("/inventory", methods=["GET"])
def inventory():
    """Online-safe inventory state: whether the robot is holding an object."""
    try:
        result = env.get_inventory_state()
        if bool_query("include_eval", default=False):
            result = dict(result)
            result["online_safe"] = False
            result["usage_scope"] = "offline_evaluation_debug_only"
            return jsonify(result)
        return jsonify(sanitize_inventory_result(result))
    except Exception as e:
        return json_error("error_inventory_service_failed", str(e), 500)


# ---------------------------------------------------------------------------
# Debug-only endpoints
# ---------------------------------------------------------------------------


@app.route("/debug/seed_trash", methods=["POST"])
def debug_seed_trash():
    """Debug-only: place a task object near the robot.

    Supports both old V1 payload {"distance": 0.6} and V2 payloads such as:
      {"kind":"floor_trash", "layout":"front-left", "distance":0.65}
      {"kind":"non_floor_decoy", "layout":"front-center", "y":0.85}
    """
    try:
        if os.getenv("ROBOT_DISABLE_DEBUG_ENDPOINTS", "0") == "1":
            return json_error("error_debug_disabled", "Debug endpoints are disabled.", 403)
        data = request.get_json(silent=True) or {}
        result = env.spawn_task_object(
            kind=str(data.get("kind", "floor_trash")),
            layout=str(data.get("layout", "front-center")),
            distance=float(data.get("distance", 0.6)),
            object_type=data.get("object_type"),
            object_types=data.get("object_types"),
            dataset_label=data.get("dataset_label"),
            task_semantic_class=data.get("task_semantic_class"),
            y=data.get("y"),
            lateral=data.get("lateral"),
        )
        result["usage_scope"] = "debug_or_benchmark_setup_only"
        result["online_safe"] = False
        return jsonify(result)
    except Exception as e:
        return json_error("error_debug_seed_failed", str(e), 500)


@app.route("/debug/reset", methods=["POST"])
def debug_reset():
    """Debug-only: reset environment, optionally with scene/mode/pose."""
    try:
        if os.getenv("ROBOT_DISABLE_DEBUG_ENDPOINTS", "0") == "1":
            return json_error("error_debug_disabled", "Debug endpoints are disabled.", 403)
        data = request.get_json(silent=True) or {}
        result = env.reset_env(
            scene=data.get("scene"),
            mode=data.get("mode"),
            seed_trash=data.get("seed_trash") if "seed_trash" in data else None,
            initial_pose=data.get("initial_pose") if isinstance(data.get("initial_pose"), dict) else None,
            look_down=bool(data.get("look_down", True)),
        )
        result["usage_scope"] = "debug_only"
        result["online_safe"] = False
        return jsonify(result)
    except Exception as e:
        return json_error("error_reset_failed", str(e), 500)


@app.route("/vision_img")
def vision_img():
    """Pure image stream for the browser monitor."""
    try:
        base64_img = env.get_first_person_view_base64()
        if "," in base64_img:
            base64_img = base64_img.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_img)
        return Response(img_bytes, mimetype="image/jpeg")
    except Exception as e:
        return f"获取画面失败: {e}", 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "success",
            "result_type": "backend_health",
            "scene": env.scene,
            "mode": env.mode,
            "scenario": env.current_scenario,
        }
    )


@app.route("/")
def index():
    """Monitoring page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 扫地机器人控制台</title>
        <meta charset="utf-8">
        <style>
            body { background-color: #1e1e1e; color: white; text-align: center; font-family: Arial, sans-serif; }
            #monitor { margin-top: 20px; border: 4px solid #4CAF50; border-radius: 10px; box-shadow: 0 0 20px rgba(76,175,80,0.5); max-width: 600px;}
            .status { margin-top: 10px; color: #aaa; font-size: 14px; }
            code { color: #9cdcfe; }
        </style>
        <script>
            function refreshImage() {
                var img = document.getElementById('monitor');
                img.src = '/vision_img?' + new Date().getTime();
            }
            setInterval(refreshImage, 1000);
        </script>
    </head>
    <body>
        <h2>📡 扫地机器人第一人称动态视野</h2>
        <img id="monitor" src="/vision_img" />
        <div class="status">🟢 实时监控中... (1 FPS)</div>
        <div class="status">V2 在线接口：<code>GET /observation</code></div>
        <div class="status">离线评测接口：<code>GET /eval/state</code></div>
        <div class="status">Benchmark：<code>GET /scenario/list</code>, <code>POST /scenario/load</code></div>
        <div class="status">调试接口：<code>POST /debug/seed_trash</code>，<code>POST /debug/reset</code></div>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
