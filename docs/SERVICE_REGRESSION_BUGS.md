# Service Regression Bug List

本文件记录 V2/V3 household service robot 已暴露过的典型 bug，以及对应的回归测试目标。

核心原则：

```text
一次 bug -> 一个固定场景 -> 一个 PASS/FAIL 检查
```

这样后续更换 YOLO 权重、修改 `patrol_runner.py` 或调整决策阈值时，可以快速确认旧问题没有回来。

## 已知 Bug

### 1. 架子/置物架上的物体被当成拾取目标

现象：

```text
机器人看到 Shelf/ShelvingUnit 上的物体，例如 Mug/Vase/Apple。
YOLO 或后处理没有识别出支撑结构上下文。
runner 锁定该物体并尝试 pick-object。
结果多次 error_pickup_target_not_centered 或错误恢复。
```

修复思路：

- V3 ontology 加入 `Shelf`、`ShelvingUnit`。
- YOLO 训练数据加入架子和架子上的物体。
- `perceive_scene_yolo.py` 增加 `support_context_blocked`。
- `patrol_runner.py` 的 `service_pick_ready()` 拒绝 `support_context_blocked=true` 的候选。

回归用例：

```text
shelf_context_should_not_pick
```

期望：

```text
识别到 Shelf/ShelvingUnit。
不产生可直接执行的 elevated pickup。
recommended_action 不能是 pick-object。
```

### 2. 桌面/台面物体被当成地面物体

现象：

```text
Book/Apple/Tomato 位于 CounterTop 或非地面高度。
2D bbox 位置靠下，看起来像 near/floor。
runner 误以为可以拾取。
```

修复思路：

- `--pickup-surface-policy floor-only` 作为 tidy 默认策略。
- `pickup_surface_allowed()` 要求 floor surface 和足够靠近底部。
- V3 数据加入桌面负例。

回归用例：

```text
tabletop_decoy_should_not_pick
```

期望：

```text
direct_pickup_detected=false。
没有 elevated pickup_now。
recommended_action 不能是 pick-object。
```

### 3. 小的地面 Potato 被忽略

现象：

```text
Potato 目标很小，bbox 面积小。
如果只用通用 near-area 阈值，可能被判为不可拾取或直接忽略。
```

修复思路：

- 保留小地面物体的 approach 分支。
- 只要它是 front-center、floor、reachable，就允许 `needs_approach=true` 或 `pickup_now=true`。

回归用例：

```text
floor_potato_far_should_approach_or_pick
```

期望：

```text
best_pickup_candidate 存在。
候选是 floor / reachable / front-center。
结果应为 pickup_now=true 或 needs_approach=true。
```

### 4. 修复负例后误伤真实地面拾取

现象：

```text
为了防止桌面/架子误拾取，阈值调得过保守。
结果地面 Apple/Tomato/Potato 也不能拾取。
```

修复思路：

- 正例和负例必须一起回归。
- 不能只检查“不拿错”，还要检查“该拿的仍然能拿”。

回归用例：

```text
floor_apple_should_pick
mixed_floor_and_non_floor_prioritize_floor
```

期望：

```text
明确地面 pickup target 应通过 pickup gate。
混合场景中只拿地面目标，不拿 elevated decoy。
```

### 5. HTTP 200 被误当成动作成功

现象：

```text
接口返回 http_status=200。
但 result_type 是 error_pickup_target_not_centered / error_no_target。
旧逻辑如果只看 HTTP，就会误记为成功。
```

修复思路：

- 移动看 `lastActionSuccess` 和 `state_changed`。
- 拾取看 `result_type=pickup_executed` 和 `holding_object=true`。
- 放置看 `result_type=place_executed`、`holding_object=false`、`placement_verified`。

回归状态：

```text
当前脚本主要覆盖 perception/decision gate。
后续可扩展 --execute-actions，加入真实 pick/place 动作验证。
```

## 运行方式

列出用例：

```powershell
python scripts\run_service_regression.py --list
```

运行全部用例：

```powershell
python scripts\run_service_regression.py --weights skills\perceive-scene-yolo\weights\best.pt --ontology configs\service_task_ontology_v3.json
```

用正在训练的新权重测试：

```powershell
python scripts\run_service_regression.py --weights runs\detect\service_v3_general_20260515_full\weights\best.pt --ontology configs\service_task_ontology_v3.json
```

只跑架子误拾取回归：

```powershell
python scripts\run_service_regression.py --cases shelf_context_should_not_pick --weights runs\detect\service_v3_general_20260515_full\weights\best.pt --ontology configs\service_task_ontology_v3.json
```

输出位置：

```text
memory/service-regression/YYYYMMDD_HHMMSS/report.json
memory/service-regression/YYYYMMDD_HHMMSS/*_rgb.jpg
memory/service-regression/YYYYMMDD_HHMMSS/*_yolo.jpg
```

## 当前限制

当前回归脚本默认不执行真实 `pick-object` / `place-object`，只检查视觉和决策准入字段。这是为了避免测试脚本在未确认场景时改变机器人持有状态。

后续可以加入执行型回归：

```text
floor target -> 执行 pick -> holding_object=true
held object + near CounterTop -> 执行 place -> holding_object=false
elevated target -> 确认 runner 不执行 pick
```
