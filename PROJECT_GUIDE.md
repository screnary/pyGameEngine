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

Milestone 3 completed (current):

* `Environment.step(command, dt)` is the single simulation entry shared by
  Manual, Behavior Tree, and Gym control
* each step applies command, motion, collision/boundary resolution,
  simulation-time/termination updates, then refreshes perception
* simulation uses a fixed `1/60` second timestep independent of render FPS
* `PygameRenderer` owns display, assets, fonts, drawing, and human pacing while
  only reading World state
* `AgentGymEnv` wraps the existing World and follows the Gymnasium reset/step API
* the continuous action is `[turn, throttle]` and reuses the existing Command
* the fixed 13-value `float32` observation uses Agent state, perceived Target /
  obstacle information, and boundary clearances without unavailable ground truth
* baseline reward uses a step cost, one penalty per new collision event, and a
  Target completion reward
* Target arrival is `terminated`; the scene time limit is `truncated`
* `render_mode=None` creates no window or Renderer; `human` reuses the same
  read-only Pygame Renderer
* Gym episodes can reuse `ExperimentRecorder`; BT-only metrics remain empty
* `gym_demo.py` provides a random-Action headless smoke test

Gym registration, PPO, Stable-Baselines3, vectorized environments, reward
optimization, and other RL training work remain intentionally deferred to M4.

Behavior Tree JSON definition refactor completed (current):

* `bt_configs/default.json` defines the migrated 16-node tree topology
* lightweight `bt-lab/v1` supports Selector, Sequence, Condition, and Action
* a recursive Loader validates JSON and constructs the real `py_trees` runtime
* a direct Class Registry maps Behavior names to existing Python node classes
* all registered leaves use `Behavior(context, name, **params)`
* JSON node parameters override scene defaults; omitted values use scene config
* `python main.py --bt default` selects a definition by filename
* the Visualizer continues to inspect runtime topology, not raw JSON
* experiment JSON/CSV output records `bt_config_id`
* all Python modules now include Chinese teaching-oriented Docstrings and
  block comments covering data flow, units, lifecycle, and algorithm intent

XML/YAML, SubTree, Decorator/Parallel expansion, plugins, hot reload, editors,
and automatic BT/RL generation remain intentionally out of scope.

Milestone 1.3 completed:

* perception-aware Behavior Tree decisions through a synchronous `PerceptionSnapshot`
* configurable perceived-target and ground-truth target-information modes
* target Range, FOV, and line-of-sight visibility checks
* FOV-aware obstacle perception and `Obstacle Threat?` decisions
* reactive Target Pursuit, low-speed Search Target, and timed avoidance preemption
* command lifecycle cleanup with no stale action command after preemption
* lightweight FOV and runtime feedback visualization
* existing M2 JSON/CSV metrics and active-action transition counting retained
* radius-aware FOV ray sampling for local traversable-gap perception
* relative-open-distance filtering so short wall-facing rays are not treated as gaps
* target-aware Gap Navigation when ground-truth or perceived target information is available
* fixed gap-entry waypoints so the Agent commits to entering a selected opening
* reduced emergency-only avoidance threshold while a gap entry is committed
* target-unknown Gap Exploration through `Traversable Gap?` and `Move Through Exploration Gap`
* in-place Search Target fallback when the current FOV has no safe local opening

Gap navigation is intentionally local. It does not build a map, remember visited
openings, guarantee complete coverage, or solve maze-like global navigation. A
committed entry waypoint stabilizes each local crossing, but later crossings can
still require new local gap selections and emergency avoidance.

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
