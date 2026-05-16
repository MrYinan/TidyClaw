# V2 代码参考说明

## thortils

本次没有把 thortils 源码整包复制进项目。实际借鉴的是其工程思想：

1. 将 AI2-THOR Controller 操作集中封装；
2. 将环境状态、调试工具、地图/导航工具分层；
3. 避免上层 Agent 到处直接访问 AI2-THOR Event metadata；
4. 保持后端接口稳定，避免每次 V2 扩展都改一堆调用点。

因此本项目新增了：

- `back/observation_adapter.py`：在线 RGB-only 观测；
- `back/eval_adapter.py`：离线 metadata 评测；
- `back/api_sanitizers.py`：移动/清扫响应脱敏；
- `/observation` 与 `/eval/state` 双接口。

## AI2-THOR 官方代码

主要参考了 AI2-THOR 的基本 Event 使用方式：

- Controller 初始化；
- `controller.step(action=...)`；
- `event.frame` 获取第一视角 RGB；
- `event.metadata["lastActionSuccess"]` 获取动作反馈；
- `event.metadata["errorMessage"]` 获取失败原因。

V2 在线 Agent 只使用 RGB 与动作反馈，不使用 object metadata / depth / segmentation / exact pose。

## 后续视觉模块

当前改造为 V2 的环境边界与接口基础。后续可以在此基础上新增：

- `skills/perceive-scene-yolo/`
- `scripts/collect_rgb_dataset.py`
- `scripts/episode_logger.py`
- `scripts/evaluate_episode.py`

这些模块不需要再改 `/observation` 和 `/eval/state` 的基础设计。
