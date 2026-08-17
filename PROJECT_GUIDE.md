# PROJECT_GUIDE.md

## Purpose

A lightweight Python research sandbox for autonomous-agent experiments using:

* behavior trees
* reinforcement learning
* hybrid BT + RL control

The goal is rapid experimentation, visualization, and algorithm validation rather than production software development.

## Core Loop

```text
Environment
    ↓
Observation
    ↓
Controller
    ↓
Action
    ↓
Environment.step()
    ↓
State / Reward
```

The controller may be:

```text
Behavior Tree
RL Policy
Hybrid BT + RL
```

Different controllers should use the same environment where practical.

## Technology

* `pygame` — 2D environment and visualization
* `numpy` — numerical operations
* `py_trees` — behavior-tree execution
* `gymnasium` — standard RL environment interface
* `stable-baselines3` — RL training when required

Use existing library capabilities before implementing custom infrastructure.

## Core Structure

Keep three concerns separated:

```text
Environment
    ↓ state / observation

Controller
    ↓ action

Agent / Environment Dynamics
```

The environment owns world state and dynamics.

The controller decides actions.

Behavior-tree-specific and RL-specific logic should not be tightly coupled to the environment.

## Behavior Trees

Use `py_trees` as the runtime.

Project code should focus on task-specific nodes such as:

```text
Conditions:
- IsObstacleNear
- IsTargetVisible
- IsTargetReached

Actions:
- MoveToTarget
- AvoidObstacle
- Patrol
```

Do not build a general-purpose behavior-tree framework.

## Project Structure

Keep the codebase small.

Typical modules may include:

```text
main.py
environment.py
agent.py
behaviors.py
gym_env.py
train.py
experiments/
```

This is only a guideline.

Do not create modules until they are actually needed.

## Development Stages

Current direction:

```text
Pygame environment
    ↓
Behavior Tree control
    ↓
Gymnasium interface
    ↓
RL experiments
    ↓
BT + RL hybrid experiments
```

Implement only the currently requested stage.

Do not build later stages in advance.

## Current Milestone

Milestone 1.2 completed:

* definition-driven Behavior Tree visualization
* automatic topology traversal from the real `py_trees` root
* lightweight hierarchical layout with ordered children
* runtime status and active-path highlighting from `SnapshotVisitor`
* automatic layout rebuild when the tree structure signature changes

Milestone 2 completed:

* episode-based, controller-independent experiment recording
* simulation-time, travelled-path, collision-event, and trajectory metrics
* optional BT tick and active-action transition counts
* `SUCCESS`, `TIMEOUT`, `manual_reset`, and `window_closed` termination handling
* detailed per-Episode JSON logs and append-only CSV summaries
* existing simulation and Behavior Tree visualization retained
