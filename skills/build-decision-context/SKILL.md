---
name: build-decision-context
description: 为 OpenClaw/大模型整理决策生成有边界的公开上下文，不暴露完整地图或完整历史记忆。
disable-model-invocation: true
metadata: {"openclaw":{"emoji":"context","requires":{"bins":["python3"]}}}
---

# 构建决策上下文

当 OpenClaw / 大模型需要为家庭服务机器人做单轮决策时，使用这个 skill 生成当前决策上下文。

这个 skill 的目标不是执行动作，而是把当前视觉、任务状态、导航状态和对象记忆压缩成一个小 JSON，让大模型只基于单轮必要信息选择下一步 `option_id`。

运行：

```powershell
python skills\build-decision-context\scripts\build_decision_context.py --task-mode tidy
```

如果需要把结果保存下来便于检查：

```powershell
python skills\build-decision-context\scripts\build_decision_context.py --task-mode tidy --output memory\decision-context.json
```

## 契约

这个 skill 是只读的：

- 不调用机器人后端。
- 不执行移动、清扫、拾取或放置。
- 不修改 `memory/*.json`。
- 只读取已有状态并生成压缩后的公开 agent view。

输出 schema 是 `robot_cleaner_decision_context_v1`，主要包含：

- `task`：tidy 阶段、是否持物、锁定的 pickup/receptacle 目标、任务进度计数。
- `perception`：最新结构化 YOLO 感知结果的压缩状态。
- `navigation`：位姿摘要、frontier 摘要、local costmap 安全判断、global plan 摘要。
- `worklist`：当前视野候选和对象记忆候选的压缩列表。
- `option_set`：有限动作选项，大模型应该只从中选择一个 `option_id`。
- `done_readiness`：房间完成的建议性 blockers；最终完成判断仍以后端/状态门控为准。
- `consistency_warnings`：状态来源不一致或过期时的警告。

## Agent 规则

- 当这个 skill 可用时，不要把完整 `position-map.json`、`semantic-map.json` 或 `object-memory.json` 读进聊天上下文。
- 主坐标和主地图来自 AI2-THOR groundtruth。`action_odometry` / `position-map` 只作为 fallback/debug，不应进入大模型主决策链。
- 大模型不要自由生成机器人动作，只能从 `option_set.options` 中选择一个 `option_id`。
- 大模型的输出应该只包含 `selected_option_id` 和一句简短理由。
- 不要直接执行 `option_set` 中的动作；必须交给 runner 或执行层重新校验。
- `observe:refresh` 不是物理动作。
- `place_precheck:*` 不是物理动作，只用于让后端验证当前点云放置候选是否真的 executor-ready。
- `orient:waypoint_floor_scan` 是 waypoint 到达后的原地安全信息增益转向；执行后必须重新构建 context。
- `continue:active_waypoint_goal` 和 `explore:inspection_waypoint:<id>` 是当前巡视主线；`frontier_cluster` / `route_step` 是 fallback。
- `pursue:pickup_target:<handle>` 表示 full observe 已看到地面 pickup target，但还没到可拾取位姿；它应优先于 waypoint 巡视。
- `move:*`、`pick:*`、`place:*`、`clean:*` 最多对应一个最终物理动作。
- `place-object` 成功只表示一个 tidy 子任务完成，不表示整个房间完成。

## 放置候选规则

点云放置候选分两级：

- `place_precheck:*`：候选已经 `visual_place_ready`，并带有 `interaction_point` / `placement_points` / `placement_safety_contract`，但还没有后端 precheck 结果。
- `place:*`：后端 precheck 已经通过，候选具备 `precheck_ok=true`、`place_now=true` 或 `final_place_ready=true`。

不要因为候选在视野内就直接选择 `place:*`。如果当前只有 `place_precheck:*`，应先选择 precheck，重新构建 context 后再选择后续 `place:*`。

## 推荐决策格式

大模型后续应返回类似：

```json
{
  "selected_option_id": "move:moveahead",
  "brief_reason": "front path is clear and no pickup/place option is executable now"
}
```

执行层必须根据最新感知和 executor 反馈再次验证该 `option_id`，不能把大模型选择直接等同于动作成功。

## 长期定位

这个 skill 是当前脚本式项目和未来 MCP / OpenClaw 工具面的过渡层。

长期目标是保持 `robot_cleaner_decision_context_v1` 这类公开上下文稳定，即使内部 memory、地图、对象记忆继续增长，也不要把原始后端状态直接暴露给大模型。

如果未来需要新增字段，优先通过 schema 版本演进，而不是让大模型直接读取完整后端文件。
