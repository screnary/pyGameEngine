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
- `train_ppo.py` - single-environment PPO training/fine-tuning entry
- `eval_ppo.py` - fixed-seed Random/PPO evaluation and human rendering entry

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

The M4 reward adds normalized Target progress to `-0.001` per step, `-0.05`
for a new collision event, and `+1.0` when the Target is reached. Ground-truth
Target distance is used only as privileged reward shaping during training; it
is not exposed through Observation or `info`. Target arrival sets `terminated`;
the scene time limit sets `truncated`.

## PPO models and rendering

The current saved models are:

```text
models/ppo_m40.zip               M4.0 no-obstacle sanity model (passed)
models/ppo_m41_obstacles.zip     M4.1 static-obstacle experiment (not passed)
models/ppo_m41a_simple_obstacle.zip  M4.1a beacon-target experiment (not passed)
models/ppo_m41b_control10hz.zip  M4.1b 10 Hz control experiment (passed)
```

`ppo_simple_obstacles` is the original **hard narrow-gap baseline**: two
rectangles leave a 30 px visual slit that the 32 px Agent cannot traverse.
`ppo_simple_obstacle` is the separate **simplified beacon-target baseline**:
one rectangle blocks the direct route while leaving a wide lower route. In the
latter scene, `los_enabled=False` skips only Target occlusion by obstacles.
Target visibility still requires sensor range and FOV, and obstacle perception
remains enabled. The beacon does not expose Target ground-truth coordinates or
bearing to the policy.

The simplest way to view M4.1 is to let `eval_ppo.py` load the model and run one
deterministic Episode with the existing Pygame Renderer:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacles --model-path models/ppo_m41_obstacles.zip --controller ppo --episodes 1 --evaluation-seed-start 2001 --tag m41_manual_view --render-mode human
```

This command performs the complete inference path:

```text
PPO.load(model)
  -> AgentGymEnv("ppo_simple_obstacles", render_mode="human")
  -> model.predict(observation, deterministic=True)
  -> Environment.step([turn, throttle])
  -> PygameRenderer
```

The window closes when the Episode reaches its 20-second simulation limit. Use
`Ctrl+C` in the terminal to stop earlier. The current 500k M4.1 checkpoint did
**not** meet the obstacle-navigation acceptance criterion: the rendered Agent
approaches the obstacle, enters a local control loop, and times out instead of
reliably going around it. The command is therefore useful for inspecting the
recorded failure mode, not for demonstrating successful obstacle navigation.

For comparison, the passed M4.0 model can be rendered with:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario rl_sanity --model-path models/ppo_m40.zip --controller ppo --episodes 1 --evaluation-seed-start 1001 --tag m40_manual_view --render-mode human
```

Run Random and PPO headlessly on the same ten M4.1 evaluation seeds with:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacles --model-path models/ppo_m41_obstacles.zip --controller both --episodes 10 --evaluation-seed-start 2001 --tag m41_manual_check --render-mode none
```

The terminal prints success rate, mean elapsed time, mean path length, and mean
collision count. Detailed Recorder output is written under:

```text
experiments/m40_eval/<tag>/summary.json
experiments/m40_eval/<tag>/<controller>/results.csv
experiments/m40_eval/<tag>/<controller>/runs/episode_*.json
```

The retained experiment checkpoints are `m41_200k` and `m41_500k`. At both
checkpoints Random and PPO achieved 0% success; see `PROJECT_GUIDE.md` for the
bounded failure diagnosis. M4.0 remains the latest passed RL milestone.

To inspect the M4.1a single-obstacle model visually, run:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41a_manual_view --render-mode human
```

To reproduce its fixed-seed headless comparison, run:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41a_manual_check --render-mode none
```

M4.1a also did not meet acceptance: Random and PPO both scored 0% at 200k and
500k. The deterministic PPO approaches the obstacle, collides once, then keeps
full throttle with almost no steering until timeout. Evaluation now writes
`diagnostics.json` beside each controller's `results.csv`; it includes distance,
visibility/action ratios, reward components, termination reason, and a reference
to a typical trajectory under `runs/`. These checkpoints are retained as failure
evidence rather than advertised as successful navigation models.

M4.1b keeps the World simulation at 60 Hz but lets PPO choose one Action every
six internal steps (`action_repeat=6`, 10 Hz control). The same Command is held
for those steps; progress, step cost, collision events, simulation time, and
Recorder metrics are aggregated from the actual internal steps. The 200k Phase
A model reached 100% success versus Random's 0% on seeds 3001-3010, so the
optional contact-penalty Phase B was not run.

Render the successful M4.1b model with:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41b_manual_view --render-mode human --action-repeat 6
```

Reproduce the fixed-seed Random/PPO comparison with:

```powershell
conda run -n pygame_lab python eval_ppo.py --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41b_manual_check --render-mode none --action-repeat 6
```

The Adapter defaults remain `action_repeat=1` and
`contact_penalty_per_step=0.0`, so existing M4.0/M4.1 commands keep their
original one-decision/one-simulation-step behavior.

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
