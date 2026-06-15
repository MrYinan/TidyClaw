# TOOLS.md - Runtime Notes

## Environment

- Framework: OpenClaw
- Simulator: AI2-THOR
- Local control service: `http://127.0.0.1:5000`
- Python: `D:\Anaconda\envs\robot\python.exe`
- Default RGB: `D:\photos\openclaw_robot_vision.jpg`
- Default depth: `D:\photos\openclaw_robot_depth.npy`
- Main map/pose: AI2-THOR groundtruth grid
- Fallback/debug only: `action_odometry`, `position-map`

## Stable OpenClaw Tools

Formal tidy turns should use only:

- `robot_cleaner_prepare_decision_turn()`
- `robot_cleaner_execute_option({ option_id })`
- `robot_cleaner_status()`
- `robot_cleaner_report()`
- `robot_cleaner_stop({ reason })`

Do not let the model directly call low-level scripts such as `move-robot`, `pick-object`, `place-object`, `get-vision`, or `perceive-scene-yolo`.

## Main Loop

```text
robot_cleaner_prepare_decision_turn
-> choose one current option_id
-> robot_cleaner_execute_option
-> verify result
-> next turn prepare again
```

## Perception Modes

- `full`: RGB-D + YOLO + depth/pointcloud geometry for waypoint observation and pickup/place decisions.
- `navigation_only`: RGB-D + depth local costmap for continuing an active waypoint route; skips YOLO.

## Waypoint Patrol

```text
explore:inspection_waypoint:<id>
-> continue:active_waypoint_goal
-> reach waypoint
-> orient:waypoint_floor_scan
-> full observe
-> pick/pursue/place if executable
-> otherwise choose next inspection waypoint
```

`frontier_cluster`, `route_step`, and raw `move:*` are fallback/internal navigation options.

## Tidy Semantics

- Default pickup policy: `floor-only`.
- `pick:*`: visible target is pickup-ready.
- `pursue:pickup_target:<handle>`: visible floor target needs one safe approach/alignment step, then re-observe.
- `place_precheck:*`: validate a placement candidate; not a physical placement.
- `place:*`: execute placement.
- A successful placement completes one subtask, not the whole room.

## Useful Local Check

```powershell
D:\Anaconda\envs\robot\python.exe scripts\prepare_decision_turn.py --task-mode tidy --format pretty
```

## Safety

- Do not bypass `robot_cleaner_execute_option` validation.
- Do not move without fresh structured perception or navigation costmap.
- Do not execute pickup/place unless the candidate is visible, reachable, and executor-ready.
- Do not choose fallback frontier/route/raw move while `pick:*`, `pursue:*`, `orient:*`, `continue:*`, or `explore:inspection_waypoint:*` is available.
- Repeated unrecoverable failures should be reported as recover failed, not room complete.
