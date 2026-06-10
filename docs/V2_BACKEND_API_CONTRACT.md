# V2 后端接口契约：RGB-D 在线观测与离线评测隔离

## 设计目标

在线 Agent 可以使用第一视角传感器数据和动作反馈：

- RGB 图像
- depth frame
- 相机内参和用于深度几何估计的相机高度/俯仰信息
- 上一步动作反馈

在线 Agent 仍然不能直接使用 AI2-THOR object metadata、objectId、真实 object position、instance segmentation 或 instance masks。后端执行器可以在内部使用这些信息完成仿真动作和验证，但不能把它们作为在线决策输入泄露给 Agent。

## 在线接口

### `GET /observation`

用途：在线 Agent / `get-vision` skill 使用。

允许返回：

- `vision_base64`: 第一视角 RGB JPEG
- `depth_base64`: 可选，第一视角 depth `.npy` float32，单位米
- `depth`: depth 编码、单位、尺寸说明
- `camera`: 相机内参、`camera_horizon_deg`、`camera_height_m`
- `last_action`, `last_action_success`, `last_action_error`, `last_action_feedback`
- 图像尺寸、时间戳、schema 信息

禁止返回：

- `metadata.objects`
- `objectId`
- `objectType`
- 真实物体位置
- instance segmentation frame
- instance masks

### `POST /move`

用途：在线移动执行。

默认返回在线安全动作反馈，不返回真实位姿。离线 debug 才能使用：

```text
POST /move?include_eval=1
```

### `POST /clean`

用途：在线仿真清扫执行器。

默认返回在线安全执行反馈，不返回目标 objectId、position、candidate metadata 等 privileged 字段。离线 debug 才能使用：

```text
POST /clean?include_eval=1
```

### `POST /pick` / `POST /place`

用途：在线服务动作执行器。

输入可以包含来自 YOLO/RGB-D 感知的 sanitized `visual_candidate`。其中 `surface_candidate_source=depth_geometry` 的候选只包含第一视角几何推断结果，不包含 AI2-THOR object metadata。

## 离线评测接口

### `GET /eval/state`

用途：离线评测、debug、数据集标注、metadata baseline 对照。

该接口返回 privileged simulator state。在线 Agent 普通决策链路不得调用它。

### `GET /state`

保留为兼容旧脚本的 alias，但返回中应包含：

```json
{
  "deprecated": true,
  "replacement_endpoint": "/eval/state",
  "usage_scope": "offline_evaluation_debug_only",
  "online_safe": false
}
```

新代码不应再依赖 `/state`。

## 感知链路

```text
GET /observation
  -> get-vision 保存 image_path + depth_path
  -> perceive-scene-yolo 使用 YOLO 识别物体类别
  -> depth_geometry 在 receptacle bbox 内生成 surface_candidates
  -> patrol-runner 对 surface_candidates 做 place 决策
```

如果没有 `depth_path`，YOLO 感知层会回退到 RGB-only 后处理。

## 修改文件

- `back/robot_env.py`
- `back/observation_adapter.py`
- `skills/get-vision/scripts/get_vision.py`
- `skills/perceive-scene-yolo/scripts/perceive_scene_yolo.py`
- `skills/perceive-scene-yolo/scripts/perceive_scene_yolo_core.py`
- `scripts/yolo_service.py`
- `scripts/patrol_runner.py`

## Surface Placement Safety Contract

For `surface_candidate_source=pointcloud_plane_grid_completion`, the online
candidate may include:

- `placement_points`: ranked screen points selected inside the same reachable
  free-space connected component.
- `placement_safety_contract.version=plane_local_grid_v1`.
- `placement_safety_contract.clearance_owner=pointcloud_plane_local_grid`.
- `free_space_completion.mode=plane_local_2d_grid`.

The grid layer is authoritative for metric obstacle inflation, support-edge
erosion, held-object footprint allowance, and reach-distance filtering. The
`/place` executor must still ground the receptacle, reject a point that is
directly inside a visible occupied-object box, and validate actuator
interactability. It must not reapply a second expanded pixel-margin rule to a
contracted grid point, because that produces inconsistent collision models.

Precheck failure semantics:

- `error_place_point_clearance`: contracted safe points are unusable at the
  final hard-contradiction check.
- `error_place_pose_not_interactable`: the grounded receptacle cannot be
  executed from the current robot pose; this is not evidence that the
  free-space component is occupied.
- `error_place_no_reachable_point`: legacy/non-contracted visual point search
  produced no usable point.

Grid safety configuration:

- `ROBOT_PC_STRUCTURAL_CORE_BBOX_SCALE` defaults to `1.0`; both initial
  region occupancy and grid completion use the same structural blocker
  footprint.
- `ROBOT_PC_FREE_GRID_HELD_FOOTPRINT_RADIUS_M` optionally overrides the
  carried-object radius included in edge and blocker clearance.
- `ROBOT_PC_FREE_GRID_PLACEMENT_CLEARANCE_M` defaults to `0.01`.
- `ROBOT_PC_FREE_GRID_MAX_POINTS_PER_COMPONENT` defaults to `4`.
- `ROBOT_PC_FREE_GRID_POINT_SEPARATION_M` defaults to `0.10`.

Exact grid-target execution:

- `pointcloud_plane_grid_completion` candidates carry
  `placement_safety_contract.requires_exact_target_execution=true` and a
  perception-derived `center_3d` for each ranked placement point.
- The backend transforms that camera-relative RGB-D target into world
  coordinates, resolves the nearest legal AI2-THOR spawn coordinate on the
  grounded receptacle, and executes `PlaceObjectAtPoint`.
- `ROBOT_PLACE_EXACT_RESOLVE_MAX_ERROR_M` defaults to `0.10`; a legal
  coordinate farther from the RGB-D target is rejected before execution.
- `ROBOT_PLACE_VERIFY_TARGET_MAX_ERROR_M` defaults to the resolution
  tolerance and is checked after release. Exact placement is not successful
  unless the final object remains within that horizontal error of the
  selected target.
- Grid candidates do not fall back to screen-guided placement. If no legal
  coordinate is sufficiently close to the RGB-D target, execution is rejected.
- Blocked support planes use plane-local grid free-space extraction only; the
  former bbox-interval completion fallback is removed.

Exact-target feedback fields:

- `placement_execution_mode=world_point_exact`
- `placement_target_required`
- `placement_target_verified`
- `placement_target_error_m`
- `placement_target_tolerance_m`
- `placement_target_resolution_error_m`
- `placement_target_resolution_tolerance_m`

Exact-target rejection types:

- `error_place_exact_target_unavailable`: no legal simulator coordinate is
  close enough to the selected RGB-D target.
- `error_place_target_deviation`: the object was released but did not settle
  within the verified target tolerance.
