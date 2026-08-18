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

Keep the codebase small. The current responsibility-based layout is:

```text
main.py
gym_demo.py
autonomy_lab/
├── agent.py
├── environment.py
├── perception.py
├── scene_config.py
├── bt/
├── rendering/
├── gym/
└── experiment/
```

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

Milestone 4.1b Phase A completed (acceptance met):

* `AgentGymEnv` now supports lightweight action repeat while keeping World
  dynamics fixed at 60 Hz; defaults remain `action_repeat=1` and
  `contact_penalty_per_step=0.0` for M4.0/M4.1 compatibility
* Phase A used `action_repeat=6`, so PPO controlled at 10 Hz while the same
  Command was applied to six `1/60 s` internal simulation steps
* macro-step progress uses start/end Ground Truth distance; step cost,
  collision events, simulation time, contact duration, and Recorder metrics use
  the number of internal steps actually executed
* an internal termination/truncation immediately stops the remaining repeats,
  and the returned Observation represents the last executed World state
* training independently initialized from `models/ppo_m40.zip`, used seed 44,
  retained the existing PPO network, gamma, learning rate, Observation, Action,
  scene, Perception, dynamics, BT, and Renderer, and reached 200,704 decisions
* on evaluation seeds 3001-3010, PPO achieved 100% success versus Random's 0%;
  PPO averaged 3.60 s, 792.0 px, and 0 collision events, while Random averaged
  20.0 s, 2202.3 px, and 2.1 collision events
* deterministic human rendering and the retained trajectory show the Agent
  steering above the obstacle, clearing it without contact, then turning back
  toward the Target
* diagnostics changed from M4.1a's mean absolute turn 0.0163 / contact after one
  collision to mean absolute turn 0.3344 / zero contact duration; mean throttle
  remained 1.0
* Phase A met both acceptance thresholds, so Phase B was not executed and no
  `ppo_m41b_contact_penalty` model was trained

The Adapter supports the approved Phase B definition
`contact_penalty_per_step=-0.002`: collision events still cost `-0.05` only on
False-to-True transitions, while each actually executed contact step would add
`-0.002`. This mechanism was tested but remained disabled in Phase A. M2
`collision_count` semantics are unchanged.

M4.1b commands:

```text
conda run -n pygame_lab python train_ppo.py --scenario ppo_simple_obstacle --seed 44 --target-timesteps 200000 --model-path models/ppo_m41b_control10hz.zip --log-label m41b_phase_a_training --init-model-path models/ppo_m40.zip --action-repeat 6
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41b_phase_a_200k --render-mode none --action-repeat 6
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41b_phase_a_human --render-mode human --action-repeat 6
```

This controlled result supports excessive 60 Hz policy-decision frequency as
the primary cause of the M4.1a failure on the fixed single-obstacle task. It
does not establish whether contact shaping would help other tasks, because
Phase B was intentionally skipped after Phase A passed.

Milestone 4.1a experiment executed (acceptance not met):

* the original `ppo_simple_obstacles` experiment, checkpoints, and evaluations
  remain the **hard narrow-gap baseline** and were not retrained or overwritten
* `ppo_simple_obstacle` is a fixed `850x600` **simplified beacon-target
  baseline** with one `(350, 100, 80, 240)` obstacle, Agent at `(100, 300)`,
  Target at `(750, 300)`, and a wide lower bypass
* this scene sets `los_enabled=False` only to skip Target obstacle occlusion;
  Target still obeys the 700 px sensor range and 160-degree FOV, while obstacle
  perception remains active
* Observation remains 13-D, Action remains `[turn, throttle]`, and reward,
  dynamics, collision semantics, Perception, Renderer, BT, and default PPO
  network are unchanged
* training started from `models/ppo_m40.zip`, reached 200,704 steps, then
  resumed the same model to 501,760 because the 200k checkpoint failed
* on evaluation seeds 3001-3010, Random and PPO both achieved 0% success at
  200k and 500k; all episodes timed out at 20 s
* at 500k, deterministic PPO averaged 288.5 px path length and 1 collision,
  versus Random's 2203.7 px and 0 collisions; PPO did not satisfy the success
  criterion and training stopped at the fixed limit
* diagnostic output now records target-distance extrema, visibility ratios,
  mean controls, reward components, collision count, termination reason, and a
  reference to a Recorder trajectory
* the 500k typical episode kept Target and obstacle visible for 100% of steps,
  used mean throttle 1.0 and mean absolute turn 0.0163, approached from 650.0 px
  to a minimum 417.7 px, collided once, and then remained blocked until timeout

This result rules out Target occlusion as the immediate M4.1a failure. It does
not demonstrate that the current Observation is sufficient for learned basic
avoidance. The clearest observed local optimum is direct full-throttle approach:
the event-based collision penalty is paid only once, while the current nearest
obstacle distance/bearing does not describe obstacle extent or a bypass edge.
Reward and Observation remain frozen here so this conclusion can be investigated
in a later, explicitly scoped milestone.

M4.1a reproduction/evaluation commands:

```text
conda run -n pygame_lab python train_ppo.py --scenario ppo_simple_obstacle --seed 44 --target-timesteps 200000 --model-path models/ppo_m41a_simple_obstacle.zip --log-label m41a_training --init-model-path models/ppo_m40.zip
conda run -n pygame_lab python train_ppo.py --scenario ppo_simple_obstacle --seed 44 --target-timesteps 500000 --model-path models/ppo_m41a_simple_obstacle.zip --log-label m41a_training --resume
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41a_500k --render-mode none
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41a_human --render-mode human
```

Milestone 4.1 experiment executed (acceptance not met):

* `ppo_simple_obstacles` is a fixed `850x600` scene with two static rectangles
  separated by a 30 px line-of-sight slit; the 32 px diameter Agent cannot pass
  through the slit and must go around the combined obstacle
* Target, Agent, obstacles, and seeds are fixed; no curriculum or reset
  randomization was added
* the 13-value Observation, continuous `[turn, throttle]` Action, World dynamics,
  collision semantics, Perception, Renderer, and Behavior Tree remain unchanged
* Ground-truth target distance is used only as privileged reward shaping during
  training; it is not exposed through Observation or info
* `train_ppo.py` and `eval_ppo.py` now accept scenario, model/checkpoint, log, and
  evaluation-label inputs while preserving their M4.0 defaults
* obstacle fine-tuning initialized from `models/ppo_m40.zip`, saved to
  `models/ppo_m41_obstacles.zip`, evaluated at 200k, then resumed without
  reinitialization to the fixed 500k limit
* deterministic PPO and Random both achieved 0% success at 200k and 500k;
  training stopped at the required limit without reward or hyperparameter tuning
* diagnostics confirmed correct Action mapping, sufficient 20 s horizon, and a
  collision-free 3.33 s geometric route; the learned policy loses Target
  visibility near the obstacle and enters a local control loop

M4.0 therefore remains the latest passed RL milestone. M4.2 work must not start
automatically from this failed baseline.

M4.1 reproduction commands:

```text
conda run -n pygame_lab python train_ppo.py --scenario ppo_simple_obstacles --seed 43 --target-timesteps 200000 --model-path models/ppo_m41_obstacles.zip --log-label m41_training --init-model-path models/ppo_m40.zip
conda run -n pygame_lab python train_ppo.py --scenario ppo_simple_obstacles --seed 43 --target-timesteps 500000 --model-path models/ppo_m41_obstacles.zip --log-label m41_training --resume
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacles --model-path models/ppo_m41_obstacles.zip --evaluation-seed-start 2001 --tag m41_500k
```

Milestone 4.0 completed (current):

* `rl_sanity` is an isolated, reproducible no-obstacle scenario whose Target is
  visible from the initial Agent pose
* Gym action remains continuous `[turn, throttle]`; the perception-aware
  Observation remains the same 13-value `float32` vector
* Gym reward adds normalized Ground Truth target-distance progress to the M3
  step cost, collision-event penalty, and Target completion bonus
* Ground Truth progress is used only by reward calculation and is not exposed in
  Observation
* `train_ppo.py` trains one default `PPO("MlpPolicy")` environment, records SB3
  Monitor episode evidence, and can resume the same checkpoint to a cumulative
  timestep target
* `eval_ppo.py` compares Random and deterministic PPO with the same fixed seeds
  and reuses `ExperimentRecorder` metric definitions
* the 50k evaluation checkpoint is retained; training resumed from that model to
  100k because the 50k deterministic policy did not yet outperform Random
* the final model is saved at `models/ppo_m40.zip`; headless and human evaluation
  both use the existing Gym Adapter and read-only Pygame Renderer

Minimal commands:

```text
conda run -n pygame_lab python train_ppo.py --target-timesteps 50000
conda run -n pygame_lab python train_ppo.py --target-timesteps 100000 --resume
conda run -n pygame_lab python eval_ppo.py --controller both --tag 100k
conda run -n pygame_lab python eval_ppo.py --controller ppo --episodes 1 --render-mode human
```

Milestone 3.1 completed (current):

* core simulation remains in `agent.py`, `environment.py`, `perception.py`, and
  `scene_config.py`
* Behavior Tree nodes, construction, Controller, and visualization are grouped
  under `autonomy_lab/bt/`
* Pygame rendering and optional assets are grouped under
  `autonomy_lab/rendering/`
* the Gymnasium Adapter is located at `autonomy_lab/gym/env.py`
* `ExperimentRecorder` is located at `autonomy_lab/experiment/recorder.py`
* Core imports no BT, Gym, Rendering, or Experiment package
* obsolete root-package module copies and their imports were removed
* Behavior Tree, Gym spaces/reward, simulation dynamics, perception semantics,
  and experiment metric definitions remain unchanged

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

Gym registration, vectorized/multiprocess training, reward tuning, obstacle RL,
BT+RL hybrid control, and other M4.1 work remain intentionally deferred.

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
