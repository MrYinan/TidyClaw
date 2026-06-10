---
name: move-robot
description: 控制 AI2-THOR 中的机器人执行基础移动。
metadata:
  {
    "openclaw":
      {
        "emoji": "move",
        "requires": { "bins": ["uv", "python3"] }
      }
  }
---

# 控制机器人移动

当已经完成环境感知并决定移动时，执行：

```bash
uv run C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\move-robot\scripts\move_robot.py --action "动作"
```

本技能只负责执行移动，不负责判断是否应该移动。

## 允许动作

- `MoveAhead`
- `MoveBack`
- `MoveLeft`
- `MoveRight`
- `RotateLeft`
- `RotateRight`
- `LookUp`
- `LookDown`

## 使用规则

1. 只能在完成感知后调用。
2. 一次只执行一个移动动作。
3. 平移动作必须经过 RGB-D local costmap 的 swept-volume 安全检查；如果局部障碍明确阻塞目标方向，不要盲目执行。
4. 拿着物体时使用膨胀后的 footprint，侧移和旋转也必须做局部碰撞检查。
5. `LookUp` / `LookDown` 只改变相机俯仰，用于观察地面物体或桌面，不改变地图 cell。
6. 动作后必须重新感知并验证视野变化或动作反馈；在线链路不读取真实位姿。

## 运行时 JSON 契约

`move_robot.py --action <action>` 会从后端 `/move` 接口返回一个 JSON 对象，并额外加入 `http_status`。

成功、碰撞或错误相关字段：

- `status`: `"success"`、`"collision"` 或 `"error"`
- `result_type`: 通常为 `move_executed` 或 `error_action_invalid_or_collision`
- `action`: 实际执行的动作
- `lastActionSuccess`: AI2-THOR 动作执行结果
- `message`: 后端返回的摘要信息
- `error_message`: 后端错误信息，存在错误时返回
- `state_changed`: 可选的后端动作变化标记；V2 在线链路不依赖真实位姿
- `online_safe`: 当返回内容已移除精确位姿字段时为 `true`
- `http_status`: HTTP 状态码

## Agent 使用规则

- `MoveAhead`、`MoveBack`、`MoveLeft`、`MoveRight` 和旋转动作都应服从最新 `navigation-costmap.json` 的局部安全判断。
- 拿着物体时必须使用膨胀 footprint；不要把空手状态的安全判断直接复用于持物移动。
- 任意移动或相机俯仰动作后，都必须重新感知并验证。
- 如果 `status != "success"` 或 `lastActionSuccess = false`，应计为动作失败。
- 当 `state_changed = false` 存在时，也应视为可疑反馈；但 V2 不要求在线返回真实 `position` / `rotation`。
