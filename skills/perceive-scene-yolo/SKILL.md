---
name: perceive-scene-yolo
description: Analyze the current RGB image with YOLO11 and return the same scene-analysis JSON contract used by the patrol runner.
metadata:
  {
    "openclaw": {
      "emoji": "yolo",
      "requires": { "bins": ["python3"] }
    }
  }
---

# YOLO11 Household Service Perception

Run:

```bash
python skills/perceive-scene-yolo/scripts/perceive_scene_yolo.py --image "<image_path>"
```

Default weights are loaded from `skills/perceive-scene-yolo/weights/best.pt`.
If that file is missing, the script falls back to `yolo11n.pt` for smoke tests.

The script maps AI2-THOR household object labels through
`configs/service_task_ontology_v2.json` into task semantics:

- `pickup_target`
- `place_receptacle`
- `obstacle`
- `cleanable_object`
- `ignored_object`

It returns the service-task fields used by `patrol_runner.py --task-mode tidy`:

- `pickup_target_detected`
- `place_receptacle_detected`
- `direct_pickup_detected`
- `direct_place_detected`
- `service_candidates`
- `receptacle_candidates`

Each service candidate keeps the raw YOLO label and the mapped task class:

- `raw_label`
- `task_semantic_class`
- `position_hint`
- `reachable`
- `pickup_now` / `place_now`
- `area`
- `center_y_ratio`
- `bottom_y_ratio`

The script also preserves the runner-compatible cleaning contract so the old
`clean` patrol mode continues to work:

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

Use `ROBOT_PERCEPTION_BACKEND=yolo` or `--perception-backend yolo` to make
`patrol-runner` and `heartbeat-watchdog` use this backend.
