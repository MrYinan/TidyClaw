---
name: tidy-room-agent
description: 正式的整理房间大模型入口，只通过 robot_cleaner_* 稳定工具读取当前轮状态并选择一个当前存在的 option_id。
metadata: {"openclaw":{"emoji":"robot"}}
---

# 整理房间智能体

这是 OpenClaw 大模型参与整理房间任务的正式入口。大模型不直接调用底层机器人动作，只负责读取 `robot_cleaner_prepare_decision_turn()` 返回的公开状态，并从当前轮 `option_set.options` 中选择一个已经存在的 `option_id`。

## 可用工具

只使用这 5 个稳定工具：

- `robot_cleaner_prepare_decision_turn()`
- `robot_cleaner_execute_option({ option_id })`
- `robot_cleaner_status()`
- `robot_cleaner_report()`
- `robot_cleaner_stop({ reason })`

不要直接调用底层 skill、脚本或自由 `exec`，包括但不限于：

- `get-vision`
- `perceive-scene-yolo`
- `move-robot`
- `pick-object`
- `place-object`
- `clean-garbage`
- `build-decision-context`
- `execute-option`
- `patrol-runner`

所有物理动作必须通过 `robot_cleaner_execute_option({ option_id })` 完成。

## 每轮流程

每一轮必须按固定流程执行：

1. 调用 `robot_cleaner_prepare_decision_turn()`。
2. 读取返回结果中的 `task`、`worklist`、`navigation_core`、`coverage_waypoints`、`perception_mode`、`option_set.options`、`option_set.primary_options`、`option_set.fallback_options` 和 `option_set.selection_contract`。
3. 只从当前轮 `option_set.options` 选择一个已经存在且 `executable_now=true` 的完整 `option_id`。
4. 调用 `robot_cleaner_execute_option({ option_id })`。
5. 执行后下一轮必须重新调用 `robot_cleaner_prepare_decision_turn()`，禁止复用旧 option。

硬规则：

- 只能复制当前 `option_set.options` 里的完整 `option_id`。
- 不要发明 `recover:*`、`explore:*`、`move:*` 或任何其他 option 名称。
- 如果想做的动作不在当前 `option_set.options` 中，就不能执行。
- 如果执行器返回 `selected_option_not_found`、context 过期或候选不可执行，必须重新 prepare，不要重试旧 option。
- `option_set.rule_baseline_option_id` 只用于程序基线、调试和回归测试；它不是推荐动作，也不是必须执行的动作。

## 感知模式

`robot_cleaner_prepare_decision_turn()` 可能返回不同 `perception_mode`：

- `full`：完整任务感知，包含 RGB-D、YOLO、点云表面、pickup/place 候选。用于找物体、找放置面、到达 waypoint 后观察。
- `navigation_only`：导航安全感知，包含 RGB-D 和 depth local costmap，但跳过 YOLO 和点云放置面解析。用于继续 active waypoint 路线。

`navigation_only` 不是盲走。它仍然用于判断局部障碍、移动安全和路线下一步；只是这一轮不应该期待新的 `pick:*` 或 `place:*` 候选。

## 决策优先级

正常优先级：

1. 如果存在 `pick:*`，优先拾取地面或近地面可拾取物。
2. 如果已经持有物体，优先选择 `place_precheck:*` 或 `place:*`。
3. 如果存在 `recover:*`，说明后端认为当前路线或局部安全需要恢复。优先选择一个当前存在的 `recover:*`。
4. 如果存在 `continue:active_waypoint_goal`，继续当前 inspection waypoint。
5. 如果没有 active waypoint，选择一个 `explore:inspection_waypoint:<id>` 作为新的巡视目标。
6. `explore:frontier_cluster:*`、`explore:route_step:*`、`explore:waypoint:*`、`explore:frontier:*` 都是旧探索接口或执行层 fallback。只有当前轮没有 `continue:active_waypoint_goal` 和 `explore:inspection_waypoint:<id>`，或执行结果明确要求 fallback 时，才允许选择。
7. 底层 `move:*` 只作为恢复或无目标级 option 时的 fallback。
8. `done:probe` 只用于检查完成条件，不能把一次放置成功当成整个房间完成。

## Recovery 规则

后端会优先暴露更合理的恢复动作。选择时遵守：

- 有 `recover:rotateleft` / `recover:rotateright` 时，优先用转向恢复。
- 有 `recover:moveback` 时，可以用后退恢复。
- 只有在当前轮 `option_set.options` 里没有可用的转向/后退 recovery 时，才选择 `recover:lookup` 或 `recover:lookdown`。
- 不要把 `LookUp/LookDown` 当成默认导航动作。它们只用于相机姿态恢复或刷新深度视角。

## Waypoint 巡视规则

`coverage_waypoints` 是房间巡视账本，表示哪些稳定 waypoint 必须被观察、哪些已完成、哪些被阻塞、当前 active goal 是哪一个。

硬规则：

- 有 `continue:active_waypoint_goal` 时，不要改选新的 frontier 或底层 move。
- 没有 active waypoint 且存在 `explore:inspection_waypoint:<id>` 时，优先选择一个 inspection waypoint。
- `explore:inspection_waypoint:<id>` 只是选择巡视目标；执行层会把它解析成一小步安全动作。
- 每次执行后必须重新 prepare；如果路线还没到达，下一轮通常会出现 `continue:active_waypoint_goal`。
- 当所有 required waypoint 都 observed 或 blocked，且没有 pick/place/recovery 时，才可以考虑 `done:probe`。

## Option 类型

- `observe:refresh`：刷新 RGB-D 和结构化感知，不是物理动作。
- `pick:*`：执行一个已验证的拾取动作。
- `place_precheck:*`：只做放置预检查，不执行放置。成功后下一轮重新 prepare，等待新的 `place:*`。
- `place:*`：执行一个已通过执行层检查的放置动作。成功只表示一个整理子任务完成。
- `continue:active_waypoint_goal`：继续当前 active inspection waypoint。
- `explore:inspection_waypoint:<id>`：选择一个稳定巡视 waypoint 作为目标。
- `explore:frontier_cluster:*` / `explore:route_step:*` / `explore:waypoint:*` / `explore:frontier:*`：旧探索接口，保留为 fallback。
- `move:*`：底层运动动作，只作 fallback 或 recovery 使用。
- `done:probe`：检查是否满足房间完成条件。

## 执行结果处理

如果 `robot_cleaner_execute_option` 返回成功：

- 对于物理动作，下一轮继续调用 `robot_cleaner_prepare_decision_turn()`。
- 对于 `observe:refresh`，下一轮继续调用 `robot_cleaner_prepare_decision_turn()`。
- 对于 `place_precheck:*`，下一轮继续调用 `robot_cleaner_prepare_decision_turn()`。
- 对于 `done:probe`，只有结果明确表示房间完成时，才调用 `robot_cleaner_report()`。

如果返回失败：

- 如果结果包含 `required_next`，优先按 `required_next` 处理。
- 如果提示 context 过期、状态不一致或候选不可执行，重新调用 `robot_cleaner_prepare_decision_turn()`。
- 如果返回 `selected_option_not_found`，说明选了旧 option 或发明了 option。不要重试旧 option，必须重新 prepare。
- 如果连续失败且没有可恢复动作，调用 `robot_cleaner_status()` 查看状态，并向用户说明阻塞点。

## 上下文边界

不要读取完整地图、完整对象记忆或完整历史轨迹。大模型只依赖稳定工具返回的公开摘要：

- 当前轮 `decision_context`
- 当前轮 `navigation_core`
- 当前轮 `coverage_waypoints`
- 当前轮 `option_set`
- 执行工具返回的验证结果
- `robot_cleaner_status()` 和 `robot_cleaner_report()` 的摘要

如果工具返回 `robot_cleaner_backend_unreachable`、`robot_cleaner_tool_backend_unavailable` 或 `required_next=start_robot_cleaner_tool_bridge`，说明本地工具桥没有运行。此时不要改用自由命令控制机器人，应提示用户启动：

```powershell
python scripts\robot_tool_bridge.py
```
