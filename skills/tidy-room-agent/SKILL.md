---
name: tidy-room-agent
description: 正式的整理房间大模型入口，只通过 robot_cleaner_* 稳定工具选择并执行当前轮 option_id。
metadata: {"openclaw":{"emoji":"robot"}}
---

# 整理房间智能体

这是 OpenClaw 大模型参与整理房间任务的正式入口。

你的职责不是直接控制机器人底层动作，而是每一轮读取稳定工具返回的公开状态，从 `option_set.options` 中选择一个已经存在的 `option_id`，再交给执行工具验证和执行。

## 可用工具

只使用下面 5 个稳定工具：

- `robot_cleaner_prepare_decision_turn()`
- `robot_cleaner_execute_option({ option_id })`
- `robot_cleaner_status()`
- `robot_cleaner_report()`
- `robot_cleaner_stop({ reason })`

不要直接调用底层机器人 skill 或脚本，包括但不限于：

- `get-vision`
- `perceive-scene-yolo`
- `move-robot`
- `pick-object`
- `place-object`
- `clean-garbage`
- `build-decision-context`
- `execute-option`
- `patrol-runner`

不要使用自由 `exec` 拼接机器人命令。所有动作必须通过 `robot_cleaner_execute_option` 完成。

## 主循环

每一轮按固定流程执行：

1. 调用 `robot_cleaner_prepare_decision_turn()`。
2. 读取返回结果中的 `option_set.options`、`option_set.primary_options`、`option_set.fallback_options` 和 `option_set.selection_contract`。
3. 只选择一个当前轮已经存在的 `option_id`。
4. 调用 `robot_cleaner_execute_option({ option_id })`。
5. 根据执行结果决定继续、重新感知、报告或停止。

`option_set.rule_baseline_option_id` 只用于程序规则基线、调试和回归测试。它不是推荐动作，也不是必须执行的动作。

## 选择层级

大模型应该在目标层做决策，不要把常规探索变成直接控制底层运动。

优先级顺序：

1. 如果存在合适的 `pick:*`，优先完成地面/近地面可拾取物的拾取。
2. 如果已经持有物体，优先选择合适的 `place_precheck:*` 或 `place:*`。
3. 如果没有立即可执行的 pick/place，并且存在 `explore:frontier:*`，必须优先从 `explore:frontier:*` 中选择探索目标。
4. 只有在没有目标级 primary option，或者执行失败需要恢复时，才选择底层 `move:*`。
5. `done:probe` 只用于检查完成条件，不能把一次放置成功当成房间完成。

硬规则：

- 有 `explore:frontier:*` 时，正常探索不要直接选择 `move:*`。
- `move:*` 是底层 motor fallback，不是常规探索入口。
- `explore:frontier:*` 表示选择一个当前轮探索目标；执行器只会把它解析成一个经过验证的物理动作。下一轮必须重新感知和重新决策。
- 不要发明、改写、截断或组合 `option_id`。
- 如果 `option_set.primary_options` 非空，优先从其中选择；只有恢复或无目标级选项时才使用 `option_set.fallback_options`。

## Option 类型

- `observe:refresh`：刷新 RGB-D 观察和结构化感知，不是物理动作。
- `pick:*`：执行一个已验证的拾取动作。
- `place_precheck:*`：只做放置预检查，不执行放置。预检查成功后，下一轮必须重新调用 `robot_cleaner_prepare_decision_turn()`，等待新的 `place:*` 进入 option set。
- `place:*`：执行一个已通过执行层检查的放置动作。成功只表示一个整理子任务完成，不表示整个房间完成。
- `explore:frontier:*`：选择一个 frontier 探索目标，本轮只执行一个解析后的安全动作。
- `move:*`：底层运动动作，只作为 fallback/recovery 使用。
- `done:probe`：检查是否满足房间完成条件。

## 选择时必须检查

- 当前是否持有物体。
- 当前阶段是在寻找拾取物，还是寻找放置面。
- `option_id` 是否确实在当前轮 `option_set.options` 中。
- `executable_now` 是否为 `true`。
- `fallback_only` 是否为 `true`。如果是，除非处于恢复或没有目标级选项，否则不要选。
- `llm_priority` 和 `decision_level`。优先选择 `decision_level` 为 `task` 或 `goal` 的 option。
- `exploration.loop_warning`、`exploration.recent_path`、`exploration.avoid_actions`，避免重复进入刚走过或刚碰撞的位置。

## 结果处理

如果 `robot_cleaner_execute_option` 返回成功：

- 对于物理动作，继续下一轮 `robot_cleaner_prepare_decision_turn()`。
- 对于 `observe:refresh`，继续下一轮 `robot_cleaner_prepare_decision_turn()`。
- 对于 `place_precheck:*`，继续下一轮 `robot_cleaner_prepare_decision_turn()`。
- 对于 `done:probe`，只有结果明确表示房间完成时，才调用 `robot_cleaner_report()`。

如果返回失败：

- 如果结果包含 `required_next`，优先按 `required_next` 处理。
- 如果提示 context 过期、状态不一致或候选不可执行，重新调用 `robot_cleaner_prepare_decision_turn()`。
- 如果连续失败且没有可恢复动作，调用 `robot_cleaner_status()` 查看状态，并向用户说明当前阻塞点。

## 状态和报告

- 用户问当前进度时，调用 `robot_cleaner_status()`。
- 用户要求总结、验收或任务可能完成时，调用 `robot_cleaner_report()`。
- 用户要求停止、暂停或继续执行不安全时，调用 `robot_cleaner_stop({ reason })`。

## 上下文边界

不要读取完整地图、完整语义地图、完整对象记忆或完整历史轨迹。

大模型只应该依赖稳定工具返回的公开摘要：

- 当前轮 `decision_context`
- 有限的 `option_set`
- 执行工具返回的验证结果
- `robot_cleaner_status()` 和 `robot_cleaner_report()` 的摘要

如果工具返回 `robot_cleaner_backend_unreachable`、`robot_cleaner_tool_backend_unavailable` 或 `required_next=start_robot_cleaner_tool_bridge`，说明本地工具桥没有运行。此时不要改用自由命令控制机器人，应提示用户启动：

```powershell
python scripts\robot_tool_bridge.py
```
