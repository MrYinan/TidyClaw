# V2 YOLO 数据集采集与训练协议

## 目标

V2 的视觉模块不是普通物体识别，而是面向主动清扫任务的 RGB-only 任务语义感知。在线 Agent 不读取 AI2-THOR metadata，只使用第一视角 RGB 图像和动作反馈。

## 类别定义

第一版数据集只保留 3 个任务语义类：

```yaml
names:
  0: cleanable_floor_trash
  1: non_floor_object
  2: obstacle
```

含义：

- `cleanable_floor_trash`：位于地面、尺寸较小、允许清扫的目标，例如地面小物体、纸团、袜子、苹果等。
- `non_floor_object`：不应清扫的物体，尤其是桌面/台面/柜子上的物体。
- `obstacle`：会影响移动路径的障碍物，例如桌腿、椅子、沙发、柜子、大箱子等。

## 采集命令

启动 AI2-THOR 后端后：

```powershell
python back\robot_server.py
```

初始化数据集目录：

```powershell
python scripts\collect_rgb_dataset.py --init-only
```

采集 100 张 RGB 图像：

```powershell
python scripts\collect_rgb_dataset.py --count 100 --action-policy random --reset-before
```

调试时可每隔 N 张用 debug 接口种一个垃圾代理：

```powershell
python scripts\collect_rgb_dataset.py --count 100 --action-policy random --reset-before --seed-trash-every 10
```

严格正式实验数据集不建议使用 debug seed；debug seed 只用于快速补充正样本和冒烟测试。

## 输出目录

```text
datasets/trash_ai2thor_v2/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  meta/
    frames.jsonl
    collection-runs.jsonl
    collection-errors.jsonl
  trash_ai2thor.yaml
```

默认会为每张图片创建一个空 YOLO 标签文件。空标签文件表示“未标注/无目标”，不是最终训练标签。训练前必须用 LabelImg / CVAT / Roboflow 等工具标注并保存为 YOLO 格式。

## 标注规范

YOLO 标签格式：

```text
<class_id> <x_center> <y_center> <width> <height>
```

坐标均为 0~1 归一化。

标注规则：

1. 只标注对任务有意义的目标。
2. 桌面或台面物体标为 `non_floor_object`，不要标为可清扫垃圾。
3. 地面小物体标为 `cleanable_floor_trash`。
4. 会阻碍前进的大物体或家具边缘标为 `obstacle`。
5. 模糊不确定目标宁可不标，不要强行标错。

## 训练命令

安装：

```powershell
pip install ultralytics
```

训练：

```powershell
yolo detect train model=yolo11n.pt data=datasets/trash_ai2thor_v2/trash_ai2thor.yaml imgsz=640 epochs=80 batch=8
```

训练完成后复制权重：

```powershell
copy runs\detect\train\weights\best.pt skills\perceive-scene-yolo\weights\best.pt
```

## 接入闭环

单图测试：

```powershell
python skills\perceive-scene-yolo\scripts\perceive_scene_yolo.py --image "D:\photos\openclaw_robot_vision.jpg"
```

YOLO 闭环测试：

```powershell
python scripts\patrol_runner.py --start --segment-steps 3 --max-steps 200 --clean-validation visual --perception yolo
```

OpenCV baseline 对比：

```powershell
python scripts\patrol_runner.py --start --segment-steps 3 --max-steps 200 --clean-validation visual --perception opencv
```

## 在线/离线边界

- 在线感知：`/observation` + YOLO RGB 推理。
- 离线评测/标注辅助：可选 `/eval/state`，但不得进入 Agent 在线决策。

`collect_rgb_dataset.py --include-eval-state` 只用于离线标注和评测，严禁把 eval_state 喂给 patrol runner 或 Agent。
