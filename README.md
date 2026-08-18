# Autonomy Lab

A lightweight Pygame research prototype for two-dimensional autonomous-agent
experiments.

## Current structure

- `main.py` - direct launcher
- `autonomy_lab/agent.py` - agent state, controls, and rendering
- `autonomy_lab/assets.py` - optional project-local PNG loading
- `autonomy_lab/environment.py` - scene state, collision, reset, and rendering
- `autonomy_lab/scene_config.py` - editable scenario presets
- `autonomy_lab/behaviors.py` - task-specific `py_trees` nodes
- `autonomy_lab/behavior_context.py` - shared runtime dependencies for BT leaves
- `autonomy_lab/behavior_registry.py` - direct Behavior Class mapping
- `autonomy_lab/bt_loader.py` - JSON validation and `py_trees` construction
- `bt_configs/` - Behavior Tree definitions in `bt-lab/v1` JSON
- `assets/` - optional tactical icons with primitive drawing fallback

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
  -> behavior_context.py / behaviors.py
  -> behavior_registry.py / bt_loader.py
  -> behavior_tree.py
  -> bt_visualizer.py / experiment.py
```

一帧自主控制的完整数据流是：

```text
Environment 当前状态
  -> AgentPerception.update() 生成 PerceptionSnapshot
  -> py_trees tick Conditions / Actions
  -> Action 写入 context.command
  -> Environment.update_command() 更新 Agent
  -> ExperimentRecorder.update() 记录实际结果
  -> Environment / BTVisualizer 绘制画面
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
