# TidyClaw

TidyClaw is an OpenClaw-based household robot agent workspace. The project builds a robot intelligence layer on top of OpenClaw and AI2-THOR for autonomous single-room tidying.

The robot is not a simple scripted controller. OpenClaw hosts the robot agent, the language model makes high-level decisions, and the local robot tools validate and execute those decisions in the simulator.

## Research Goal

This project studies an active task-planning method for a household service robot based on OpenClaw. The current research focus is the `tidy-room-agent`, especially the inspection waypoint patrol layer:

- actively patrol the current room;
- observe the environment with RGB-D and YOLO;
- detect floor-level objects that need tidying;
- pick up valid objects;
- place them on valid receptacles;
- continue room coverage until completion conditions are met.

## OpenClaw Agent Loop

The OpenClaw language model does not directly call low-level robot scripts. It only chooses one currently exposed `option_id` from the decision context.

Each formal decision turn follows this loop:

```text
robot_cleaner_prepare_decision_turn
-> OpenClaw LLM chooses one executable option_id
-> robot_cleaner_execute_option
-> executor validates and performs the action
-> memory/state are updated
-> next turn prepares again
```

This makes the model responsible for task-level decision making, while the executor remains responsible for safety, action validation, navigation constraints, pickup/place checks, collision memory, and state synchronization.

## Current Main Task

The current main task mode is:

```text
tidy
```

The intended behavior is:

```text
idle heartbeat
-> detect whether the room needs service
-> request a tidy-room-agent run
-> prepare decision context
-> choose and execute one option per turn
-> patrol with inspection waypoints
-> pick/place valid floor-level objects
-> report when done, stopped, or blocked
```

## Main Components

```text
AGENTS.md / SOUL.md / IDENTITY.md / TOOLS.md
  OpenClaw-facing agent identity, policy, and tool rules.

HEARTBEAT.md
  Heartbeat policy. The heartbeat only detects whether a tidy task should start;
  it should not directly run the long tidy loop.

plugins/robot-cleaner-tools/
  OpenClaw plugin exposing stable robot_cleaner_* tools.

skills/tidy-room-agent/
  Main OpenClaw model-facing tidy agent instructions.

skills/heartbeat-watchdog/
  Idle watchdog used by OpenClaw heartbeat to detect service targets.

skills/get-vision/
  RGB-D observation capture.

skills/perceive-scene-yolo/
  YOLO + depth perception for task candidates.

scripts/
  Decision context building, option execution, waypoint planning, state memory,
  route management, map backends, object memory, and robot status/report logic.

back/
  AI2-THOR backend server and environment wrapper.

configs/
  Runtime, scenario, and service-task ontology configs.

tests/
  Unit and regression tests.

memory/
  Runtime state and logs. This directory is ignored by git.
```

## Stable OpenClaw Tools

Formal robot decisions should go through these stable tools:

```text
robot_cleaner_prepare_decision_turn()
robot_cleaner_execute_option({ option_id })
robot_cleaner_status()
robot_cleaner_report()
robot_cleaner_stop({ reason })
```

The model should not bypass them by directly calling low-level scripts such as movement, pickup, placement, vision capture, or YOLO perception scripts.

## Decision Priorities

The current tidy policy is:

1. Handle safety and recovery first.
2. If a valid floor-level pickup target is available, pick or pursue it.
3. If holding an object, run placement precheck and place it.
4. After reaching an inspection waypoint, run `orient:waypoint_floor_scan`, then perform full RGB-D + YOLO/depth observation.
5. Continue the active waypoint route when no higher-priority service action exists.
6. Select a new `explore:inspection_waypoint:<id>` when there is no active waypoint.
7. Use frontier, route-step, or raw movement only as fallback.

Default pickup policy is `floor-only`: tabletop and countertop objects can be recorded as visual candidates, but they are not tidy pickup targets by default.

## Map And Perception

- Primary pose/map frame: AI2-THOR groundtruth grid.
- `action_odometry` and `position-map`: fallback/debug only.
- `full` perception: RGB-D + YOLO + depth geometry, used for waypoint observation and pickup/place decisions.
- `navigation_only` perception: RGB-D + depth local costmap, used for continuing active waypoint routes without running full YOLO.
- Collision or failed translation writes a blocked edge so the planner can avoid repeating the same failure.

## Important Configs

```text
configs/robot_cleaner_runtime.json
  Runtime map backend configuration.

configs/scenarios_v2.json
  AI2-THOR benchmark and backend scenario definitions.

configs/service_task_ontology_v2.json
  Online YOLO task ontology.

configs/service_task_ontology_v3.json
  Training and regression ontology.
```

## Runtime Notes

This project is developed and tested in a local OpenClaw + AI2-THOR setup.

Typical local assumptions:

```text
AI2-THOR backend: http://127.0.0.1:5000
Python env: robot conda environment
Main task mode: tidy
```

Useful local commands:

```powershell
python scripts\prepare_decision_turn.py --task-mode tidy --format pretty
python scripts\robot_stop.py --reason user_stop
python skills\patrol-runner\scripts\patrol_runner_skill.py --command status
python skills\heartbeat-watchdog\scripts\heartbeat_watchdog.py --task-mode tidy --launch-mode agent-request --timeout 60
```

## Tests

Run all tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Common focused tests:

```powershell
python -m unittest discover -s tests -p "test_global_planner.py"
python -m unittest discover -s tests -p "test_decision_context_builder.py"
python -m unittest discover -s tests -p "test_object_memory_core.py"
```

## Status

The project is under active development. The current focus is a stable OpenClaw decision loop for single-room tidy service, with inspection waypoint patrol, RGB-D/YOLO observation, grounded pickup/place execution, and memory-backed safety recovery.

