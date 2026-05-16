---
name: get-vision
description: 获取 AI2-THOR 当前第一视角 RGB 图像。V2 在线链路只返回 RGB 图像路径与动作反馈，不返回机器人真实位姿或物体 metadata。
metadata:
  {
    "openclaw":
      {
        "emoji": "vision",
        "requires": { "bins": ["uv", "python3"] }
      }
  }
---

# 获取机器人当前 RGB 视野

当需要了解机器人当前环境时，执行：

```bash
uv run C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\get-vision\scripts\get_vision.py
```

本 skill 只负责获取在线安全观测，不负责决策或执行动作。

## V2 在线观测原则

1. 本 skill 调用后端 `GET /observation`，而不是旧版 `GET /vision`。
2. 在线 Agent 只允许使用第一视角 RGB 图像和上一步动作反馈。
3. 在线 Agent 不允许使用 AI2-THOR 的 object metadata、objectId、真实 position、真实 rotation、cameraHorizon、depth、instance segmentation。
4. 如果需要离线评测或调试，必须由评测脚本单独调用 `GET /eval/state`，不能把该接口结果喂给 Agent 决策。

## 使用规则

1. 在任何移动或清扫前，优先调用本 skill 获取最新环境图像。
2. 不要假设旧视野仍然有效；每次关键动作前都应重新感知。
3. 返回的 `image_path` 是后续 `analyze-scene-opencv` 或 V2 `perceive-scene-yolo` 的输入。

## Runtime JSON Contract

`get_vision.py` returns one JSON object.

Success fields:

- `status`: usually `"success"`
- `schema_version`: V2 observation schema version
- `result_type`: `"vision_captured_rgb_only"`
- `observation_contract`: usually `"rgb_only_action_feedback_v2"`
- `online_safe`: `true`
- `image_path`: local absolute path of the saved first-person RGB image
- `last_action_feedback`: action feedback from the previous backend step
- `last_action`: compatibility flat field for previous action name
- `last_action_success`: compatibility flat field for previous action success
- `last_action_error`: compatibility flat field for previous action error message

Forbidden online fields:

- `position`
- `rotation`
- `cameraHorizon`
- `objectId`
- `objectType`
- AI2-THOR `metadata.objects`
- depth / instance segmentation / instance masks

Error fields:

- `status`: `"error"`
- `result_type`: short machine-readable failure type
- `message`: short failure reason

Agent rule:

- Use `image_path` as the input for scene analysis.
- Do not use an old `image_path` after any physical action.
- If `status != "success"` or `image_path` is missing, do not move or clean in this round.
