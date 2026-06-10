---
name: place-object
description: 通过后端 /place 执行器，把当前持有物体放到视野前方居中的可见容器或承载面上。
metadata:
  {
    "openclaw": {
      "emoji": "place",
      "requires": { "bins": ["python3"] }
    }
  }
---

# 放置物体

当机器人已经持有物体，并且最新 RGB 感知结果表明前方居中存在可达的 `place_receptacle` 时，使用本技能。

执行：

```bash
python skills/place-object/scripts/place_object.py
```

本技能会调用 `POST /place`。后端内部根据候选契约执行受控屏幕放置或精确三维放置，但默认响应是在线安全的。

V2 默认行为会通过容器或承载面前方可见区域附近的受控屏幕点进行放置，并验证最终物体仍位于目标 `receptacle` 上，且处在机器人前方可达区域内。旧版“整块容器/承载面”放置回退已移除。

成功字段：

- `status = "success"`
- `result_type = "place_executed"`
- `lastActionSuccess`
- `holding_object = false`
- `placement_verified = true`
- `object_reachable_from_agent = true`

可能的失败 `result_type` 包括：

- `error_no_held_object`
- `error_no_receptacle_in_front`
- `error_receptacle_not_centered`
- `error_receptacle_too_far`
- `error_target_not_receptacle`
- `error_place_no_reachable_point`
- `error_place_position_unreachable`
- `error_place_receptacle_mismatch`

## Surface Point Contract

`pointcloud_plane_grid_completion` placement does not require the chosen
screen point to be at the receptacle center. When the candidate carries
`placement_safety_contract.version=plane_local_grid_v1`, `placement_points`
are ranked free-space points already checked in the plane-local occupancy
grid. The executor reuses that metric clearance decision, still rejects direct
visible-object overlap, and validates an exact `PlaceObjectAtPoint` target.

Additional precheck result types:

- `error_place_point_clearance`: no contracted placement point remains safe.
- `error_place_pose_not_interactable`: the current pose cannot execute on the
  grounded receptacle; do not treat this alone as an occupied surface patch.

## Exact Grid Target Execution

For `pointcloud_plane_grid_completion`, the safe grid cell is an execution
target, not only a screen hint. The backend resolves a legal
AI2-THOR position close to the RGB-D `center_3d`, executes
`PlaceObjectAtPoint`, and verifies the final horizontal target error.

- `ROBOT_PLACE_EXACT_RESOLVE_MAX_ERROR_M=0.10` controls legal-target snapping.
- `ROBOT_PLACE_VERIFY_TARGET_MAX_ERROR_M=0.10` controls post-place acceptance.
- Grid candidates never fall back to nearby screen-guided placement.
- Blocked-plane free-space candidates come only from the plane-local grid;
  bbox-interval fallback completion is not supported.

Additional result types:

- `error_place_exact_target_unavailable`
- `error_place_target_deviation`
