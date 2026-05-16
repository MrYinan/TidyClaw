# V2 后端接口契约：RGB-only 在线观测与离线评测隔离

## 设计目标

V2 的核心边界是：**在线 Agent 只能使用第一视角 RGB 图像与动作反馈；AI2-THOR metadata 只允许用于离线评测、debug、数据集标注和仿真执行器内部实现。**

这次改造参考了 thortils 的工程组织思想：把 AI2-THOR Controller 细节封装在后端工具层，再用 adapter 向外暴露稳定接口。但本项目没有把 thortils 整包复制进来，也没有把 visible objects 作为在线 Agent 决策输入。

## 在线接口

### `GET /observation`

用途：在线 Agent / get-vision skill 使用。

允许返回：

- 第一视角 RGB 图像：`vision_base64`
- 上一步动作反馈：`last_action`, `last_action_success`, `last_action_error`, `last_action_feedback`
- 图像尺寸、场景名、时间戳、schema 信息

禁止返回：

- `position`
- `rotation`
- `cameraHorizon`
- `objectId`
- `objectType`
- `metadata.objects`
- depth frame
- instance segmentation
- instance masks

### `POST /move`

用途：在线移动执行。

默认返回在线安全动作反馈，不返回真实位姿。

如需离线 debug 原始位姿信息，可使用：

```text
POST /move?include_eval=1
```

### `POST /clean`

用途：在线仿真清扫执行器。

默认返回在线安全执行反馈，不返回目标 objectId、position、target、candidate 等 metadata。

如需离线 debug 原始清扫细节，可使用：

```text
POST /clean?include_eval=1
```

## 离线评测接口

### `GET /eval/state`

用途：离线评测、debug、数据集标注、V1 metadata baseline 对照。

该接口返回 privileged simulator state，包括 robot pose 与 visible trash candidate metadata。在线 Agent 不得调用。

### `GET /state`

保留为兼容旧脚本的 alias，但返回中包含：

```json
{
  "deprecated": true,
  "replacement_endpoint": "/eval/state",
  "usage_scope": "offline_evaluation_debug_only",
  "online_safe": false
}
```

新代码不应再依赖 `/state`。

## Runner 清扫前校验策略

`scripts/patrol_runner.py` 新增：

```bash
--clean-validation visual|metadata|off
```

默认：

```bash
--clean-validation visual
```

含义：

- `visual`：V2 默认，只依据 RGB 感知输出字段判断是否允许清扫，不调用 `/eval/state`。
- `metadata`：V1 baseline / 离线 debug，对视觉目标再调用 `/eval/state` 做 metadata oracle 校验。
- `off`：保留给快速调试，不建议正式实验使用。

## Navigation Memory 改造

V1 的 navigation memory 会优先读取真实 `position / rotation`。V2 在线链路移除位姿后，navigation memory 已改为：

- 若 action result 显式包含 pose，则可在离线 debug 中使用；
- 默认在线情况下，基于上一格状态 + action success 做相对 odometry 更新；
- `MoveAhead` 成功：前进到相邻格；
- `MoveBack` 成功：后退到相邻格；
- `RotateLeft / RotateRight` 成功：只更新 heading；
- 移动失败：标记 blocked edge / collision。

这使得在线 Agent 不再依赖 AI2-THOR 真实位姿。

## 修改文件

- `back/observation_adapter.py`
- `back/eval_adapter.py`
- `back/api_sanitizers.py`
- `back/robot_server.py`
- `skills/get-vision/scripts/get_vision.py`
- `skills/get-vision/SKILL.md`
- `skills/move-robot/SKILL.md`
- `skills/clean-garbage/SKILL.md`
- `scripts/patrol_runner.py`
- `scripts/navigation_memory_core.py`
- `scripts/perception_action_validator.py`
- `skills/heartbeat-watchdog/scripts/heartbeat_watchdog.py`
- `skills/patrol-runner/scripts/patrol_runner_skill.py`
- `scripts/collect_alignment_samples.py`

## 推荐启动方式

正式 V2 在线巡视：

```bash
python scripts/patrol_runner.py --start --continuous --clean-validation visual
```

V1 metadata baseline 对照：

```bash
python scripts/patrol_runner.py --start --continuous --clean-validation metadata
```

OpenClaw patrol skill 默认也会传入 `--clean-validation visual`。
