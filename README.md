# Autonomy Lab

A lightweight Pygame research prototype for two-dimensional autonomous-agent
experiments.

## Current structure

- `main.py` - direct launcher
- `autonomy_lab/agent.py`, `environment.py`, `perception.py`, `scene_config.py` -
  core simulation
- `autonomy_lab/bt/` - Behavior nodes, Context, Registry, JSON Loader,
  Runtime Controller, and Visualizer
- `autonomy_lab/rendering/` - read-only Pygame Renderer and optional PNG loading
- `autonomy_lab/gym/env.py` - Gymnasium adapter around the same World
- `autonomy_lab/experiment/recorder.py` - controller-independent Episode metrics
- `bt_configs/` - Behavior Tree definitions in `bt-lab/v1` JSON
- `assets/` - optional tactical icons with primitive drawing fallback
- `gym_demo.py` - headless random-Action Gym smoke test

## Run

Behavior Tree control is the default:

```powershell
conda run -n pygame_lab python main.py
```

Choose controller, Behavior Tree definition, or scenario:

```powershell
conda run -n pygame_lab python main.py --controller bt --bt default
conda run -n pygame_lab python main.py --controller manual
conda run -n pygame_lab python main.py --scenario simple
conda run -n pygame_lab python main.py --scenario obstacle_course
conda run -n pygame_lab python main.py --scenario dense_obstacles
```

Add or edit scenarios in `autonomy_lab/scene_config.py`. Each preset defines the
world size, random seed, Agent parameters, Target, Obstacles, display options,
and the small set of Behavior Tree experiment parameters.

Controls: `W/S` or Up/Down to move, `A/D` or Left/Right to turn, and `R` to
reset.

Run the Gymnasium adapter without a window:

```powershell
conda run -n pygame_lab python gym_demo.py
```

Use human rendering from Python when visual inspection is needed:

```python
from autonomy_lab.gym.env import AgentGymEnv

env = AgentGymEnv(scenario="simple", render_mode="human")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

`render_mode=None` does not create a Renderer, window, event loop, or Clock.
Both modes use a fixed simulation step of `1/60` second and call the same
`Environment.step()` implementation.

## Gymnasium interface

The continuous `Box(shape=(2,))` action is `[turn, throttle]`, with both values
in `[-1, 1]`. Manual input and Behavior Tree Actions produce the same Command
keys before entering the World.

The fixed 13-value `float32` observation contains, in order:

```text
speed,
heading_sin, heading_cos,
target_visible, target_distance, target_bearing,
perceived_obstacle_available, obstacle_distance, obstacle_bearing,
left_clearance, right_clearance, top_clearance, bottom_clearance
```

Distances and clearances are normalized. An invisible Target or missing
perceived obstacle uses zero distance/bearing plus its visibility/availability
flag, so the vector never exposes unavailable World ground truth.

The M3 baseline reward is `-0.001` per step, `-0.05` for a new collision event,
and `+1.0` when the Target is reached. Target arrival sets `terminated`; the
scene time limit sets `truncated`. Reward tuning and PPO remain future work.

## Behavior Tree

The current topology is defined in `bt_configs/default.json`. The v1 format
supports `selector`, `sequence`, `condition`, and `action` nodes with the common
fields `type`, `name`, `behavior`, `memory`, `params`, and `children`.

```text
bt_configs/default.json
        -> BT Loader
        -> Behavior Registry
        -> py_trees runtime
        -> Visualizer / Experiment Log
```

JSON defines tree organization and optional parameter overrides. Python Behavior
classes define node execution. Parameters omitted from JSON fall back to the
selected scene's `behavior_tree` settings.

Every registered leaf follows the same construction contract:

```python
Behavior(context=context, name=name, **params)
```

The Registry therefore maps JSON names directly to classes; adding a conforming
Behavior requires one Registry entry but no dedicated factory or Loader branch.

The right-side panel reads topology, status, feedback, and the current visited
path from the real `py_trees` runtime. Changing JSON node order or composition
therefore changes both execution and visualization without editing the
Visualizer.

## Recommended reading order

源码中的中文教学注释按以下顺序阅读最容易理解：

```text
main.py
  -> scene_config.py / environment.py / agent.py
  -> perception.py
  -> bt/context.py / bt/behaviors.py
  -> bt/registry.py / bt/loader.py / bt/controller.py
  -> rendering/renderer.py / bt/visualizer.py
  -> experiment/recorder.py / gym/env.py
```

一帧自主控制的完整数据流是：

```text
Environment 当前状态
  -> py_trees tick Conditions / Actions
  -> Action 写入 context.command
  -> Environment.step(command, 1/60)
  -> motion / collision / termination / AgentPerception.update()
  -> ExperimentRecorder.update() 记录实际结果
  -> PygameRenderer / BTVisualizer 只读绘制画面
```

## Rendering assets

The renderer automatically uses `assets/agent.png`, `target.png`, and
`obstacle.png` when they load successfully. Missing or invalid images fall back
to Pygame primitives. `threat.png` and `waypoint.png` are reserved visual assets
and do not add behavior.

## Experiment recording

Every run starts an Episode and records controller-independent metrics. Target
arrival saves `SUCCESS`, the scene's `max_episode_time` saves `TIMEOUT`, `R`
saves `manual_reset` before starting a new Episode, and closing the window saves
`window_closed`.

```text
experiments/runs/episode_0001.json
experiments/results.csv
```

Metrics include simulation time, travelled distance, collision contacts,
trajectory samples, and optional BT tick/action-transition counts. BT-controlled
episodes also record the definition's `bt_config_id`; manual episodes leave it
empty. Runtime-generated experiment files are ignored by Git.
