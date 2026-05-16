---
name: clean-garbage
description: 清理当前机器人前方可清扫垃圾或污渍。
metadata:
  {
    "openclaw":
      {
        "emoji": "clean",
        "requires": { "bins": ["uv", "python3"] }
      }
  }
---

# 执行清扫

当已经完成环境感知，并确认前方存在可清理目标时，执行：

```bash
uv run C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\clean-garbage\scripts\clean_garbage.py
```

本 skill 只负责执行清扫，不负责高层决策。

## 使用规则

1. 只有在结构化感知确认存在地面、可达、当前可清扫目标时才调用。
2. 一轮只执行清扫，不要和移动同时执行。
3. 清扫后必须重新感知并验证目标是否消失。

## Runtime JSON Contract

`clean_garbage.py` returns one JSON object from the backend `/clean` endpoint and adds `http_status`.

Success fields:

- `status`: `"success"`
- `result_type`: `"clean_executed"`
- `message`: backend summary
- `lastActionSuccess`: AI2-THOR actuator result
- `online_safe`: true when object metadata has been removed from the payload
- `eval_details_available`: true means offline debug can call `/clean?include_eval=1` for raw simulator details
- `http_status`: HTTP status code

Business failure fields:

- `status`: `"error"`
- `result_type`: one of:
  - `error_no_target_in_front`
  - `error_target_not_centered`
  - `error_not_reachable`
  - `error_not_floor_level`
  - `error_no_cleanable_target`
  - `error_ai2thor_clean_failed`
  - `error_clean_verify_failed`
- `message`: reason
- `http_status`: often `200` for business failures

V2 默认在线响应不会返回 `objectId`、`objectType`、`position`、`target`、`target_after` 或 `candidates` 等 simulator metadata。需要离线 debug 时，单独调用 `/clean?include_eval=1`。

Agent rule:

- Treat `clean_executed` with `status = "success"` as a simulated actuator success, then verify with fresh RGB perception. Do not rely on metadata disappearance in the online Agent chain.
- Do not treat `http_status = 200` as success by itself; read `status` and `result_type`.
- For `error_target_not_centered`, rotate toward the target if perception agrees, then re-sense.
- For `error_not_reachable`, move or align only after fresh perception says the path is safe.
- For `error_not_floor_level`, do not retry cleaning the same target.
- For `error_clean_verify_failed`, count it as an action verification failure.
