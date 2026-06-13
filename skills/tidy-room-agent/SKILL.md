---
name: tidy-room-agent
description: 正式的整理房间大模型入口，只通过 robot_cleaner_* 稳定工具读取当前轮状态并选择一个 option_id。
metadata: {"openclaw":{"emoji":"robot"}}
---

# 整理房间智能体

这是 OpenClaw 大模型参与整理房间任务的正式入口。

你的职责不是直接控制机器人底层动作，而是每一轮读取稳定工具返回的公开状态，从当前 `option_set.options` 中选择一个已经存在的 `option_id`，再交给执行工具验证和执行。

## 可用工具

只使用下面 5 个稳定工具：

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
2. 读取返回结果中的 `navigation_core`、`option_set.options`、`option_set.primary_options`、`option_set.fallback_options` 和 `option_set.selection_contract`。
3. 只从当前轮 `option_set.options` 里选择一个已经存在且 `executable_now=true` 的 `option_id`。
4. 调用 `robot_cleaner_execute_option({ option_id })`。
5. 执行后下一轮必须重新调用 `robot_cleaner_prepare_decision_turn()`，不能复用旧 option。

`option_set.rule_baseline_option_id` 只用于程序基线、调试和回归测试。它不是推荐动作，也不是必须执行的动作。

## 决策优先级

正常优先级：

1. 如果 `option_set.recovery_options` 非空，优先选择一个 `recover:*`。
2. 如果存在合适的 `pick:*`，优先拾取地面或近地面可拾取物。
3. 如果已经持有物体，优先选择 `place_precheck:*` 或 `place:*`。
4. 如果 `navigation_core.state=following_active_route`，必须优先选择当前轮的 `explore:route_step:*`。
5. 如果没有 pick/place/recovery/route step，再选择目标级探索选项：`explore:waypoint:*` 或 `explore:frontier:*`。
6. 只有没有目标级 primary option，或执行失败后需要恢复时，才选择底层 `move:*`。
7. `done:probe` 只用于检查完成条件，不能把一次放置成功当成整个房间完成。

## Navigation Core 规则

`navigation_core` 是导航状态机摘要。探索导航时必须优先相信它，而不是自己用局部文字描述重新推理路线。

硬规则：

- 当 `navigation_core.state=following_active_route` 时，说明后端已经锁定当前 frontier 路线。本轮应选择 `explore:route_step:*`，不要改选 `explore:waypoint:*`、`explore:frontier:*` 或底层 `move:*`。
- 当 `navigation_core.state=camera_recovery_required` 时，优先选择对应的 `recover:*`，恢复相机俯仰后再探索。
- 当 `navigation_core.state=route_blocked_recovery` 或 `planner_recovery_required` 时，优先选择 `recover:*`，不要硬走旧路线。
- 当 `navigation_core.route_locked=true` 时，除非当前轮没有 route/recovery option，否则不要使用 fallback option。
- `explore:route_step:*` 仍然只执行一个经过验证的物理动作；执行后下一轮必须重新 prepare。

## Option 类型

- `observe:refresh`：刷新 RGB-D 观察和结构化感知，不是物理动作。
- `pick:*`：执行一个已经验证的拾取动作。
- `place_precheck:*`：只做放置预检查，不执行放置。成功后下一轮重新 prepare，等待新的 `place:*`。
- `place:*`：执行一个已经通过执行层检查的放置动作。成功只表示一个整理子任务完成。
- `explore:route_step:*`：继续当前锁定 frontier 路线的一步动作，是常规探索时最优先的导航 option。
- `explore:waypoint:*`：规划器给出的安全一步路点，通常用于摆脱旋转循环或补充局部探索。
- `explore:frontier:*`：选择一个 frontier 探索目标，本轮只执行一个解析后的安全动作。
- `move:*`：底层运动动作，只作为 fallback 或 recovery 使用。
- `done:probe`：检查是否满足房间完成条件。

## 选择前检查

选择 option 前必须检查：

- 当前是否持有物体。
- 当前阶段是在找拾取物，还是找放置面。
- `option_id` 是否确实存在于当前轮 `option_set.options`。
- `executable_now` 是否为 `true`。
- `fallback_only` 是否为 `true`。如果是，除非没有 primary option 或处于恢复状态，否则不要选。
- `navigation_core.state` 和 `navigation_core.required_next`。
- `exploration.loop_warning`、`exploration.recent_path`、`exploration.avoid_actions`，避免重复进入刚走过或刚碰撞的位置。

## 执行结果处理

如果 `robot_cleaner_execute_option` 返回成功：

- 对于物理动作，下一轮继续调用 `robot_cleaner_prepare_decision_turn()`。
- 对于 `observe:refresh`，下一轮继续调用 `robot_cleaner_prepare_decision_turn()`。
- 对于 `place_precheck:*`，下一轮继续调用 `robot_cleaner_prepare_decision_turn()`。
- 对于 `done:probe`，只有结果明确表示房间完成时，才调用 `robot_cleaner_report()`。

如果返回失败：

- 如果结果包含 `required_next`，优先按 `required_next` 处理。
- 如果提示 context 过期、状态不一致或候选不可执行，重新调用 `robot_cleaner_prepare_decision_turn()`。
- 如果返回 `selected_option_not_found`，说明选了旧 option。不要重试旧 option，必须重新 prepare。
- 如果连续失败且没有可恢复动作，调用 `robot_cleaner_status()` 查看状态，并向用户说明阻塞点。

## 上下文边界

不要读取完整地图、完整语义地图、完整对象记忆或完整历史轨迹。

大模型只依赖稳定工具返回的公开摘要：

- 当前轮 `decision_context`
- 当前轮 `navigation_core`
- 当前轮 `option_set`
- 执行工具返回的验证结果
- `robot_cleaner_status()` 和 `robot_cleaner_report()` 的摘要

如果工具返回 `robot_cleaner_backend_unreachable`、`robot_cleaner_tool_backend_unavailable` 或 `required_next=start_robot_cleaner_tool_bridge`，说明本地工具桥没有运行。此时不要改用自由命令控制机器人，应提示用户启动：

```powershell
python scripts\robot_tool_bridge.py
```
