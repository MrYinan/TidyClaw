# TidyClaw

TidyClaw is an OpenClaw-based embodied robot agent framework for autonomous household task planning.

The goal of this project is to explore how household robots can move from command-driven execution to state-driven autonomy.

Instead of waiting for continuous human instructions, the robot periodically monitors the environment, determines whether a service task is required, and autonomously initiates task execution.

The system is built on:

- OpenClaw Agent Runtime
- AI2-THOR simulation environment
- RGB-D perception
- YOLO-based object understanding
- Tool-constrained robot execution
- Memory-based state management


## Motivation

Traditional household robots usually follow a reactive paradigm:

User command
      |
      v
Robot execution


TidyClaw explores a proactive paradigm:

Environment
      |
      v
Heartbeat monitoring
      |
      v
Task necessity estimation
      |
      v
Agent planning
      |
      v
Robot execution


The robot is not only an executor, but also a task initiator.


## Research Objective

This project investigates a state-driven autonomous task planning approach for household service robots.

The current prototype focuses on single-room tidying:

- Detect whether the room requires service
- Start a tidy-room-agent workflow
- Inspect the environment through active patrol
- Identify task-relevant objects
- Execute pickup/place operations
- Verify results and update memory


## Autonomous Operation Loop

Heartbeat
    |
    v
Environment Check
    |
    v
Need Service?
    |
    +---- No ----> Sleep
    |
    v
Start Tidy Agent
    |
    v
Decision Loop
    |
    v
Execute Option
    |
    v
Verify
    |
    v
Update State
    |
    v
Continue


Heartbeat does not execute robot actions.

It only determines when the agent should wake up.

The LLM decides task-level actions, while deterministic tools execute and validate actions.


## Decision Loop

Observation
    |
    v
Decision Context
    |
    v
LLM Option Selection
    |
    v
Executor Validation
    |
    v
Robot Action
    |
    v
State Update


## Design Philosophy

TidyClaw does not let the LLM directly control robot actions.

LLM responsibilities:

- understand task context
- select high-level options
- perform task planning

Executor responsibilities:

- validate actions
- handle navigation constraints
- perform manipulation
- update robot state


## Why OpenClaw

OpenClaw provides the agent runtime layer:

- Agent lifecycle management
- Tool orchestration
- Context management
- Memory integration
- Heartbeat-driven execution


## System Components

- OpenClaw Agent Layer
- Robot Tool Layer
- Skill Layer
- AI2-THOR Simulation Backend
- Memory State Layer


## Evaluation

The system is evaluated in AI2-THOR scenarios:

- Empty room
- Single object tidying
- Multiple objects
- Obstacle + object scenarios

Metrics:

- Task completion rate
- Cleaning success rate
- Navigation steps
- Collision count
- Detection accuracy
- Autonomous trigger success rate


## Status

The project is under active development.

Current focus:

- heartbeat-driven task activation
- structured decision making
- RGB-D and YOLO perception
- grounded robot execution
- memory-backed recovery
