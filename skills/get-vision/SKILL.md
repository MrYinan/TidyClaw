---
name: get-vision
description: 获取 AI2-THOR 当前第一视角 RGB 图像，并在后端支持时同步保存 depth_frame，供 YOLO 后处理生成几何候选。
metadata:
  {
    "openclaw":
      {
        "emoji": "vision",
        "requires": { "bins": ["uv", "python3"] }
      }
  }
---

# 获取机器人当前 RGB-D 观测

需要了解机器人当前环境时，执行：

```bash
uv run C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\get-vision\scripts\get_vision.py
```

本 skill 只负责获取在线安全观测，不负责决策或执行动作。

## 在线观测原则

1. 调用后端 `GET /observation`，不使用旧版 `GET /vision`。
2. 返回第一视角 RGB 图片路径 `image_path`。
3. 如果后端启用了 AI2-THOR depth 渲染，同时返回 `depth_path`，格式为 `.npy` float32，单位米。
4. depth 只作为第一视角传感器数据，用于几何判断，例如距离、粗略高度、可放置区域候选。
5. 在线 Agent 仍然不能使用 AI2-THOR 的 object metadata、objectId、真实 object position、instance segmentation 或 instance masks。

## Runtime JSON Contract

成功字段：

- `status`: `"success"`
- `schema_version`: `3`
- `result_type`: `"vision_captured_rgbd"` 或 `"vision_captured_rgb_only"`
- `observation_contract`: `"rgbd_action_feedback_v3"`
- `online_safe`: `true`
- `image_path`: 保存后的第一视角 RGB 图片路径
- `depth_path`: 保存后的 depth `.npy` 路径；如果后端没有 depth，则不存在
- `depth`: depth 编码、单位、尺寸说明
- `camera`: 相机内参和用于深度几何估计的相机高度/俯仰信息
- `last_action_feedback`: 上一次动作反馈
- `last_action`, `last_action_success`, `last_action_error`: 兼容字段

后续使用：

```powershell
python skills\perceive-scene-yolo\scripts\perceive_scene_yolo.py --image <image_path> --depth <depth_path>
```

在 patrol-runner 中，`depth_path` 和 `camera` 会自动传给 YOLO 感知层。

## 安全规则

- 每次物理动作前都应重新调用 `get-vision`。
- 不要在物理动作后复用旧的 `image_path` 或 `depth_path`。
- 如果 `status != "success"` 或缺少 `image_path`，本轮不能移动、清扫、拾取或放置。
- 如果缺少 `depth_path`，YOLO 会退回 RGB-only 后处理；这不是致命错误。
