# TidyClaw

TidyClaw is an OpenClaw-based embodied robot agent framework for
autonomous household task planning.

The goal of this project is to explore how household service robots can
move from **command-driven execution** to **state-driven autonomy**.

Instead of waiting for continuous human instructions, TidyClaw enables
the robot to monitor the environment, determine whether a service task
is required, and autonomously initiate task execution.

The system is built on:

-   OpenClaw Agent Runtime
-   AI2-THOR simulation environment
-   RGB-D perception
-   YOLO-based object understanding
-   Tool-grounded robot execution
-   Memory-based state management

## Motivation

Traditional household robots usually follow a reactive paradigm:

    Human command
          |
          v
    Robot execution

The robot performs tasks only after receiving explicit instructions.

TidyClaw explores a proactive paradigm:

    Environment state
            |
            v
    Heartbeat monitoring
            |
            v
    Task necessity estimation
            |
            v
    Agent activation
            |
            v
    Task planning
            |
            v
    Robot execution

The robot is not only an executor of commands, but also a task initiator
driven by environmental state.

## Key Ideas

TidyClaw focuses on three aspects:

### 1. State-driven autonomy

The robot decides whether a task should start based on environment state
instead of continuous human commands.

Heartbeat transfers task initiation from user instructions to autonomous
state evaluation.

### 2. LLM-based high-level planning

The language model performs task-level reasoning and selects executable
options.

The LLM does not directly generate low-level robot commands.

### 3. Tool-grounded execution

Robot behaviors are executed through validated tools.

The executor layer handles:

-   action validation
-   navigation constraints
-   manipulation execution
-   state synchronization

This separation improves reliability and reproducibility.

## Research Objective

This project investigates a state-driven autonomous task planning
approach for household service robots.

The current prototype focuses on single-room tidying:

-   Detect whether the room requires service
-   Start the tidy-room-agent workflow
-   Actively patrol the environment
-   Identify task-relevant objects
-   Execute pickup/place operations
-   Verify results
-   Update memory state

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
    Prepare Decision Context
        |
        v
    LLM Selects Option
        |
        v
    Execute and Validate
        |
        v
    Update State
        |
        v
    Continue Patrol

Heartbeat does not directly execute robot actions.

It determines when the agent should become active.

The LLM decides task-level actions, while deterministic tools execute
and validate robot behaviors.

## Decision Loop

    Observation

        |

        v

    Decision Context Builder

        |

        v

    Option Space Generation

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

The LLM selects from validated executable options instead of directly
controlling robot actions.

## Design Philosophy

TidyClaw separates reasoning from execution.

### Agent Layer

Responsible for:

-   understanding task context
-   selecting high-level options
-   task planning

### Executor Layer

Responsible for:

-   validating actions
-   handling navigation constraints
-   performing manipulation
-   updating robot state

The LLM decides **what should happen**.

The executor determines **how it can safely happen**.

## Why OpenClaw

OpenClaw provides the agent runtime layer:

-   Agent lifecycle management
-   Tool orchestration
-   Context management
-   Memory integration
-   Heartbeat-driven execution

OpenClaw allows the robot system to maintain a continuous agent loop
instead of a single command-response interaction.

## Tool Interface

The robot agent interacts with the system through stable tools:

    robot_cleaner_prepare_decision_turn()

    robot_cleaner_execute_option({
        option_id
    })

    robot_cleaner_status()

    robot_cleaner_report()

    robot_cleaner_stop({
        reason
    })

The model should not bypass these interfaces by directly calling
low-level robot scripts.

## System Components

-   OpenClaw Agent Layer
-   Robot Tool Layer
-   Skill Layer
-   AI2-THOR Simulation Backend
-   Memory State Layer

## Evaluation

The system is evaluated in AI2-THOR scenarios:

-   Empty room
-   Single object tidying
-   Multiple objects
-   Obstacle + object scenarios

Metrics:

-   Task completion rate
-   Cleaning success rate
-   Navigation steps
-   Collision count
-   Detection accuracy
-   Autonomous trigger success rate
-   Recovery success rate

## Current Status

The project is under active development.

Implemented / developing:

-   heartbeat-driven task activation
-   structured decision making
-   RGB-D and YOLO perception
-   option-based execution
-   grounded robot execution
-   memory-backed recovery

## Future Work

Future improvements include:

-   richer long-term memory
-   more complex household task planning
-   dynamic multi-room environments
-   real robot deployment
