---
name: pick-object
description: 通过后端 /pick 执行器拾取视野前方居中的可见家庭服务物体。
metadata:
  {
    "openclaw": {
      "emoji": "pick",
      "requires": { "bins": ["python3"] }
    }
  }
---

# 拾取物体

当最新 RGB 感知结果表明前方居中存在可达的 `pickup_target` 时，使用本技能。

执行：

```bash
python skills/pick-object/scripts/pick_object.py
```

本技能会调用 `POST /pick`。后端内部可能使用模拟器 metadata 来执行 `PickupObject`，但默认响应是在线安全的，不暴露 `objectId`、精确位置或原始 metadata。

成功字段：

- `status = "success"`
- `result_type = "pickup_executed"`
- `lastActionSuccess`
- `holding_object = true`

可能的失败 `result_type` 包括：

- `error_no_pickup_target_in_front`
- `error_pickup_target_not_centered`
- `error_pickup_target_too_far`
- `error_target_not_pickupable`
- `error_already_holding_object`
