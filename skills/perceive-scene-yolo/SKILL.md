---
name: perceive-scene-yolo
description: 使用 YOLO11 分析当前 RGB 图像，并返回 patrol-runner 可直接使用的结构化场景感知 JSON。
metadata:
  {
    "openclaw": {
      "emoji": "yolo",
      "requires": { "bins": ["python3"] }
    }
  }
---

# YOLO11 家庭服务感知

运行命令：

```bash
python skills/perceive-scene-yolo/scripts/perceive_scene_yolo.py --image "<image_path>"
```

使用该 skill 前，建议先启动常驻 YOLO 服务（默认已经开启）：

```bash
python scripts/yolo_service.py
```

如果权重文件缺失，脚本会回退到 `yolo11n.pt`，主要用于冒烟测试，不建议作为正式服务巡视的长期配置。

## 语义映射

脚本会将 AI2-THOR 家庭物体标签，通过 `configs/service_task_ontology_v2.json` 映射到任务语义：

- `pickup_target`：可整理、可拾取物体
- `place_receptacle`：可放置台面、容器或支撑面
- `obstacle`：影响导航或交互的障碍物/结构物
- `cleanable_object`：兼容旧清扫逻辑的地面可清理目标
- `ignored_object`：当前服务任务不处理的物体

## 返回字段

该脚本主要服务于：

```bash
python scripts/patrol_runner.py --task-mode tidy --perception-backend yolo
```

它会返回 tidy 服务巡视所需字段：

- `pickup_target_detected`
- `place_receptacle_detected`
- `direct_pickup_detected`
- `direct_place_detected`
- `service_candidates`
- `receptacle_candidates`

每个服务候选会保留 YOLO 原始标签和任务语义：

- `raw_label`
- `task_semantic_class`
- `position_hint`
- `reachable`
- `pickup_now`
- `place_now`
- `area`
- `center_y_ratio`
- `bottom_y_ratio`

## 兼容旧清扫合同

为了让旧的 `clean` 巡视模式仍能运行，脚本也保留以下兼容字段：

- `floor_trash_detected`
- `direct_cleanable_detected`
- `alignment_needed`
- `trash_candidates`
- `ignored_candidates`
- `obstacle_ahead`
- `open_directions`
- `frontier_exists`
- `floor_clean`
- `analysis_confidence`
- `occupancy`
- `recommended_action`
- `notes`

## 使用规则

- 正式服务机器人主线应使用 `--perception-backend yolo`。
- `heartbeat-watchdog` 和 `patrol-runner` 默认应走 tidy 服务语义，而不是旧 clean 基线。
- 感知层只负责提供候选和结构化属性，不直接决定最终动作。
- 是否拾取、放置、靠近、转向，应由 `patrol-runner` 的状态机和执行反馈共同决定。

可通过环境变量或命令行启用该后端：

```bash
ROBOT_PERCEPTION_BACKEND=yolo
```

或：

```bash
python scripts/patrol_runner.py --perception-backend yolo --task-mode tidy
```
