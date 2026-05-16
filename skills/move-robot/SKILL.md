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

本 skill 只负责执行移动，不负责判断是否应该移动。

## 允许动作

- `MoveAhead`
- `MoveBack`
- `RotateLeft`
- `RotateRight`

## 使用规则

1. 只能在完成感知后调用。
2. 一次只执行一个移动动作。
3. 如果前方存在障碍，不要执行 `MoveAhead`。
4. 动作后必须重新感知并验证视野变化或动作反馈；V2 在线链路不读取真实位姿。

## Runtime JSON Contract

`move_robot.py --action <action>` returns one JSON object from the backend `/move` endpoint and adds `http_status`.

Success or collision fields:

- `status`: `"success"` or `"collision"` / `"error"`
- `result_type`: usually `move_executed` or `error_action_invalid_or_collision`
- `action`: executed action
- `lastActionSuccess`: AI2-THOR action result
- `message`: backend summary
- `error_message`: backend error when present
- `state_changed`: optional backend action-change flag; V2 在线链路不依赖真实位姿
- `online_safe`: true when the payload has removed exact pose fields
- `http_status`: HTTP status code

Agent rule:

- Do not call `MoveAhead` unless the latest structured perception has `obstacle_ahead = false`.
- After any movement, verify with fresh perception.
- If `status != "success"` or `lastActionSuccess = false`, count it as an action failure. `state_changed = false` is also suspicious when present, but V2 不要求在线返回真实 position / rotation。
