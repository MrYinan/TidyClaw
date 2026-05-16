---
name: analyze-scene-opencv
description: 对当前视野图片进行结构化场景分析，返回垃圾、障碍、可探索方向等结果。
metadata:
  {
    "openclaw":
      {
        "emoji": "opencv",
        "requires": { "bins": ["uv", "python3"] }
      }
  }
---

# 结构化分析当前视野

当已经通过 `get-vision` 拿到 `image_path` 后，执行：

```bash
uv run C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\analyze-scene-opencv\scripts\analyze_scene_opencv.py --image "图片路径"
```

本 skill 只负责把图像转换为结构化分析结果，不负责高层决策。

## 使用规则

1. 每次使用前，应保证图片来自最新一次 `get-vision`。
2. 如果分析置信度低，应采取更保守动作。
3. 如果结果显示存在可清扫地面目标，优先考虑进入 CLEAN。
4. 如果结果显示前方存在障碍，禁止直接 `MoveAhead`。

## Runtime JSON Contract

`analyze_scene_opencv.py --image <image_path>` returns one JSON object.

Top-level success fields:

- `status`: `"success"`
- `image_path`: analyzed image path
- `floor_trash_detected`: true when at least one reachable floor-level candidate exists
- `direct_cleanable_detected`: true when at least one candidate is floor-level, reachable, and centered enough for immediate cleaning
- `alignment_needed`: true when a floor-level target exists but needs rotation/alignment before cleaning
- `trash_candidates`: reachable floor-level candidates
- `ignored_candidates`: non-floor, uncertain, or otherwise rejected visual candidates
- `obstacle_ahead`: true when the forward sector appears blocked
- `open_directions`: possible exploration directions, e.g. `["forward"]`
- `frontier_exists`: true when at least one open direction exists
- `floor_clean`: true when no reachable floor-level candidate is detected
- `analysis_confidence`: numeric confidence score
- `occupancy`: visual clutter ratio per direction
- `recommended_action`: conservative suggestion only; final decision stays with the Agent
- `notes`: human-readable diagnostic notes

`trash_candidates[]` fields:

- `label`: conservative visual label, e.g. `red_small_object`
- `bbox`: image bounding box
- `center`: image center point
- `position_hint`: `front-left`, `front-center`, or `front-right`
- `reachable`: true if floor-level and in the front reachable sector
- `surface_hint`: `floor` or `countertop_or_nonfloor`
- `is_floor_level`: true only for floor-level targets
- `cleanable_now`: true only when direct `clean-garbage` is allowed by visual rule
- `needs_alignment`: true when rotation/alignment is needed before cleaning
- `area`, `center_y_ratio`, `bottom_y_ratio`: diagnostics for perception tuning

Decision rule:

- Call `clean-garbage` only when `floor_trash_detected = true` and a candidate has `is_floor_level = true`, `reachable = true`, and `cleanable_now = true`.
- If `alignment_needed = true` but `direct_cleanable_detected = false`, rotate toward the candidate and re-run perception before cleaning.
- Treat `recommended_action` as advisory; safety and task rules in `AGENTS.md` take priority.
- If `status = "error"` or required fields are missing, do not move ahead or clean.
