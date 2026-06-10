---
name: execute-option
description: 执行 OpenClaw/大模型从 decision_context 中选择的一个 option_id，并在执行前重新做本地安全校验。
disable-model-invocation: true
metadata: {"openclaw":{"emoji":"option","requires":{"bins":["python3"]}}}
---

# 执行决策选项

当 OpenClaw / 大模型已经根据 `build-decision-context` 选择了一个 `option_id` 时，使用这个 skill 把选择交给现有机器人执行层。

运行示例：

```powershell
python skills\execute-option\scripts\execute_option.py --option-id move:moveahead --context memory\decision-context.json
```

也可以传入大模型返回的 JSON：

```powershell
python skills\execute-option\scripts\execute_option.py --selection-json "{\"selected_option_id\":\"move:moveahead\",\"brief_reason\":\"front path is clear\"}"
```

## 契约

这个 skill 不允许大模型自由运行脚本。它只接受 `decision_context.option_set.options` 中已经存在的 `option_id`。

执行前会检查：

- context schema 必须是 `robot_cleaner_decision_context_v1`。
- `option_id` 必须存在于 `option_set.options`。
- `executable_now` 不能是 `false`。
- 物理动作必须有结构化感知。
- 移动动作必须通过 `navigation-costmap` 安全门控。
- pick/place 必须符合 holding 状态。
- `place_precheck:*` 必须引用可见、可达、带点云放置契约的候选；它只调用 `/place-precheck`，不执行放置。
- 如果 context 有 `consistency_warnings`，默认禁止物理动作，要求先 `observe:refresh`。

## 支持的 option_id

- `observe:refresh`：刷新 RGB-D 观察和 YOLO 结构化感知，不是物理动作。
- `place_precheck:*`：调用 `place-object --precheck-only`，验证点云候选是否可执行；成功后写入 `memory/place-precheck-cache.json`。
- `move:*`：调用 `move-robot`。
- `pick:*`：调用 `pick-object`。
- `place:*`：调用 `place-object`。
- `clean:*`：调用 `clean-garbage`。
- `done:probe`：返回 `done_readiness`，不执行物理动作。

## Agent 规则

- 不要绕过这个 skill 直接运行 move/pick/place/clean 脚本。
- 每次只提交一个 `option_id`。
- 如果返回 `error_option_validation_failed`，优先按 `required_next` 重新感知或重新构建 context。
- 如果 `place_precheck:*` 成功，下一步先重新运行 `build-decision-context`，再选择新的 `place:*`。
- 执行成功不等于房间完成；`place-object` 成功只表示一个整理子任务完成。

## 调试

只验证不执行：

```powershell
python skills\execute-option\scripts\execute_option.py --option-id move:moveahead --dry-run
```

允许使用带 warning 的旧 context 执行物理动作：

```powershell
python skills\execute-option\scripts\execute_option.py --option-id move:moveahead --allow-stale-context
```

`--allow-stale-context` 只用于调试，不建议正式运行使用。
