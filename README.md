# Autonomy Lab

Autonomy Lab 是一个基于 Python、Pygame、`py_trees`、Gymnasium 和
Stable-Baselines3 的二维自主智能体科研原型，用于快速验证：

- Behavior Tree 自主决策；
- Reinforcement Learning；
- 后续的 Behavior Tree + RL 混合控制。

项目以可运行实验和可观察行为为优先，不以生产级框架或通用游戏引擎为目标。

## Current Project Structure

```text
project_root/
├── main.py                         # BT / manual Pygame 主入口
│
├── autonomy_lab/
│   ├── agent.py                    # Agent 状态与运动更新
│   ├── environment.py              # 唯一 World / Simulation Core
│   ├── perception.py               # Target 与 Obstacle 感知快照
│   ├── scene_config.py             # Scenario 与实验参数
│   │
│   ├── bt/
│   │   ├── behaviors.py            # 可执行 Condition / Action 节点
│   │   ├── context.py              # BT 节点共享依赖
│   │   ├── registry.py             # Behavior 名称到 Python class 的映射
│   │   ├── loader.py               # JSON Definition → py_trees Runtime
│   │   ├── controller.py           # BT tick 与 Command 输出
│   │   └── visualizer.py           # Runtime Tree 可视化
│   │
│   ├── rendering/
│   │   ├── renderer.py             # 只读 Pygame Renderer
│   │   └── assets.py               # 可选图片加载与 primitive fallback
│   │
│   ├── gym/
│   │   └── env.py                  # Environment 的 Gymnasium Adapter
│   │
│   └── experiment/
│       ├── recorder.py             # Controller-independent Episode metrics
│       └── runners.py              # BT / PPO 公共 Episode execution
│
├── scripts/
│   ├── gym_demo.py                 # headless Gymnasium smoke test
│   ├── train_ppo.py                # PPO training / fine-tuning 入口
│   ├── eval_ppo.py                 # Random / PPO evaluation 与 rendering
│   ├── compare_bt_ppo.py           # M4.2 frozen BT vs PPO comparison
│   └── eval_m43_generalization.py  # M4.3 geometry generalization evaluation
│
├── bt_configs/
│   └── default.json                # bt-lab/v1 Behavior Tree Definition
├── assets/                         # 可选本地 PNG 图标
├── models/                         # 本地 PPO checkpoints
├── experiments/                    # 运行日志、trajectory 与统计结果
├── tests/                          # 单元测试与轻量集成测试
├── docs/                           # 已保留的设计与实现记录
├── environment.yml                 # pygame_lab Conda Environment
├── AGENTS.md                       # 项目工作约束
└── PROJECT_GUIDE.md                # Milestone 与架构边界
```

依赖方向保持为：

```text
Core Simulation
      ↓
BT / Gym / Rendering
      ↓
Experiment
      ↓
Scripts
```

`Environment` 是唯一 World。Core 不反向依赖 BT、Gym、Rendering、Experiment
或 Scripts；`autonomy_lab` 内部模块也不依赖 `scripts/`。

`assets/`、`models/` 和 `experiments/` 保存可选本地资源或生成的实验产物，
因此默认被 Git 忽略。

## Quick Start

项目使用 Conda Environment `pygame_lab`。默认启动 Behavior Tree Controller：

```powershell
conda run -n pygame_lab python main.py
```

选择 Controller、BT Definition 或 Scenario：

```powershell
conda run -n pygame_lab python main.py --controller bt --bt default
conda run -n pygame_lab python main.py --controller manual
conda run -n pygame_lab python main.py --scenario simple
conda run -n pygame_lab python main.py --scenario obstacle_course
conda run -n pygame_lab python main.py --scenario dense_obstacles
```

Manual Controller 操作：

- `W/S` 或 `↑/↓`：加速、减速或反向移动；
- `A/D` 或 `←/→`：转向；
- `R`：重置当前 Scenario；
- 关闭窗口：结束运行。

在 [scene_config.py](autonomy_lab/scene_config.py) 中新增或调整 Scenario。
每个 preset 包含 World 尺寸、seed、Agent、Target、Obstacle、显示参数和少量
Behavior Tree 实验参数。

## Script Entry Points

非交互式入口位于 `scripts/`：

```powershell
conda run -n pygame_lab python -m scripts.gym_demo
conda run -n pygame_lab python -m scripts.train_ppo --help
conda run -n pygame_lab python -m scripts.eval_ppo --help
conda run -n pygame_lab python -m scripts.compare_bt_ppo --help
conda run -n pygame_lab python -m scripts.eval_m43_generalization --help
```

## Simulation and Rendering Boundary

`Environment.step(command, simulation_dt)` 是统一的 Simulation 入口。Manual、
Behavior Tree 和 Gym 最终都生成同一种 Command：

```text
{"turn": float, "throttle": float}
```

Simulation 使用固定 `simulation_dt = 1/60 s`，不由真实 FPS 决定。
`PygameRenderer` 只读取 World State，不修改 Simulation State。

`render_mode=None` 不创建 Renderer、窗口、event loop 或 Clock；
`render_mode="human"` 使用现有只读 Pygame Renderer。两种模式调用同一个
`Environment.step()`。

最小 Gymnasium human rendering 示例：

```python
from autonomy_lab.gym.env import AgentGymEnv

env = AgentGymEnv(scenario="simple", render_mode="human")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

## Gymnasium Interface

### Action

连续 `Box(shape=(2,))` Action 为：

```text
[turn, throttle]
```

两个值均位于 `[-1, 1]`。Manual input 和 Behavior Tree Action 在进入 World 前
也会转换为相同的 Command 表达。

### Observation

固定 13-D `float32` Observation 依次为：

```text
speed,
heading_sin, heading_cos,
target_visible, target_distance, target_bearing,
perceived_obstacle_available, obstacle_distance, obstacle_bearing,
left_clearance, right_clearance, top_clearance, bottom_clearance
```

距离和 clearance 均经过归一化。Target 不可见或没有感知到 Obstacle 时，
对应 distance/bearing 使用约定的中性值，并通过 visibility/availability flag
区分。Observation 不泄露当前不可用的 World Ground Truth。

### Reward

M4 Reward 保持为：

```text
normalized target progress
- 0.001 × executed simulation steps
- 0.05 × new collision events
+ 1.0 when Target is reached
```

Ground-truth Target distance 只作为 training-only privileged reward shaping，
不进入 Observation，也不通过 `info` 提供给 Policy。Target reached 对应
`terminated=True`；Scenario time limit 对应 `truncated=True`。

## PPO Models and Scenarios

当前保留的 checkpoints：

```text
models/ppo_m40.zip                    M4.0 no-obstacle sanity（通过）
models/ppo_m41_obstacles.zip          M4.1 hard narrow-gap（未通过）
models/ppo_m41a_simple_obstacle.zip   M4.1a beacon-target（未通过）
models/ppo_m41b_control10hz.zip       M4.1b 10 Hz control（通过）
```

主要 RL Scenario：

- `rl_sanity`：无障碍、Target 初始可见，用于验证 PPO pipeline；
- `ppo_simple_obstacles`：两个矩形形成不可通行的 30 px 狭缝，是
  **hard narrow-gap baseline**；
- `ppo_simple_obstacle`：单矩形障碍与宽裕绕行空间，是
  **simplified beacon-target baseline**。

在 `ppo_simple_obstacle` 中，`los_enabled=False` 只跳过 Target 的障碍遮挡；
Target 仍受 sensor range 与 FOV 约束，Obstacle perception 仍然启用。
Policy 不会获得 Target Ground Truth position 或 bearing。

## M4.0–M4.1b PPO Pipeline

### M4.0 — PPO Pipeline Sanity

M4.0 在 `rl_sanity` 上证明当前 Environment、13-D Observation、连续 Action
和 Reward 能够学习最简单的 Target Pursuit。最终模型：

```text
models/ppo_m40.zip
```

Human evaluation：

```powershell
conda run -n pygame_lab python -m scripts.eval_ppo --scenario rl_sanity --model-path models/ppo_m40.zip --controller ppo --episodes 1 --evaluation-seed-start 1001 --tag m40_manual_view --render-mode human
```

### M4.1 — Hard Narrow-gap Baseline

M4.1 使用 `ppo_simple_obstacles`。200k 和 500k checkpoints 的 Random/PPO
success rate 均为 0%，因此该模型被保留为失败证据，不作为成功导航模型。

查看典型失败轨迹：

```powershell
conda run -n pygame_lab python -m scripts.eval_ppo --scenario ppo_simple_obstacles --model-path models/ppo_m41_obstacles.zip --controller ppo --episodes 1 --evaluation-seed-start 2001 --tag m41_manual_view --render-mode human
```

该 deterministic PPO 接近障碍后进入局部控制循环，并在 20 秒 Simulation
horizon 后 timeout。

### M4.1a — Simple Obstacle Diagnosis

M4.1a 将任务降级为 `ppo_simple_obstacle`，但保持 60 Hz PPO decision frequency。
200k 与 500k 仍为 0% success：Policy 接近障碍、发生一次 collision event，
随后几乎保持 full throttle 且缺少有效转向，最终 timeout。

查看 M4.1a：

```powershell
conda run -n pygame_lab python -m scripts.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41a_manual_view --render-mode human
```

Evaluation 会在 `results.csv` 旁生成 `diagnostics.json`，记录 Target distance、
visibility ratio、Action 统计、Reward components、termination reason 和典型
trajectory 文件。

### M4.1b — 10 Hz Control Baseline

M4.1b 保持 World Simulation 为 60 Hz，但设置 `action_repeat=6`，使 PPO 每
6 个 internal simulation steps 决策一次，即约 10 Hz。相同 Command 在这 6 步
内保持不变；simulation time、Reward、collision event 和 Recorder metrics
仍按实际执行的 internal steps 计算。

Phase A 在 seeds 3001–3010 上达到：

```text
Random success rate = 0%
PPO success rate    = 100%
```

因此可选的 Phase B contact penalty 没有执行。

查看成功的 M4.1b 模型：

```powershell
conda run -n pygame_lab python -m scripts.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41b_manual_view --render-mode human --action-repeat 6
```

复现 Random/PPO fixed-seed comparison：

```powershell
conda run -n pygame_lab python -m scripts.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41b_manual_check --render-mode none --action-repeat 6
```

Adapter 默认值仍为 `action_repeat=1`、`contact_penalty_per_step=0.0`，保证
M4.0/M4.1 的旧调用语义不变。

## M4.2 — BT vs PPO Baseline

`scripts.compare_bt_ppo` 在相同 Environment、Scenario、seed 和 World 初态下
比较 frozen default Behavior Tree 与 `models/ppo_m41b_control10hz.zip`：

```text
World Simulation = 60 Hz
BT tick          = 60 Hz
PPO decision     ≈ 10 Hz（action_repeat=6）
```

这是当前完整 Controller baseline 的比较，不是 decision-frequency-matched
algorithm ablation。

运行 batch comparison：

```powershell
conda run -n pygame_lab python -m scripts.compare_bt_ppo
```

运行独立 human demos：

```powershell
conda run -n pygame_lab python -m scripts.compare_bt_ppo --human-demo
```

冻结结果：

```text
scenario                controller  success  time    path      collisions
rl_sanity               BT          100%     1.683s  370.3px   0
rl_sanity               PPO         100%     1.983s  436.3px   0
ppo_simple_obstacle     BT          100%     6.100s  783.2px   0
ppo_simple_obstacle     PPO         100%     3.600s  792.0px   0
ppo_simple_obstacles*   BT          100%     5.100s  772.9px   0
ppo_simple_obstacles*   PPO         100%     3.350s  737.0px   0
```

`*` 表示单独解释的 hard stress test，不与 primary baseline 混合计算总体
success rate。固定布局下 seeds 4001–4010 的初态等价，仅用于统一 evaluation
pipeline，不表示随机泛化能力。

输出位置：

```text
experiments/comparisons/m42_bt_vs_ppo.csv
experiments/comparisons/m42_bt_vs_ppo_summary.json
experiments/comparisons/runs/<scenario>/<controller>/
experiments/comparisons/human_demos/<scenario>/<controller>/
```

## M4.3 — Zero-shot Geometry Generalization

`scripts.eval_m43_generalization` 复用公共 Episode runners，在不重新训练或调参
的前提下评估 frozen BT 与 PPO：

- `seen`：`rl_sanity`、`ppo_simple_obstacle`；
- `unseen_mild`：`m43_target_shift`、`m43_obstacle_shift`、
  `m43_reverse_detour`、`m43_combined_shift`；
- `ood_hard`：`ppo_simple_obstacles`。

运行 headless evaluation：

```powershell
conda run -n pygame_lab python -m scripts.eval_m43_generalization
```

运行 8 个独立 human observation Episodes：

```powershell
conda run -n pygame_lab python -m scripts.eval_m43_generalization --human-demo
```

固定布局仅运行 seed 5001；统计单位是 Scenario，不是随机重复 Episode。
因此 `4/4` 表示“4 个手工构造的 mild unseen 场景全部成功”，不能解释为随机
trial 的 success probability。

```text
group         controller  successful scenes  mean time  mean path  mean collisions
seen          BT          2/2                3.892 s    576.8 px   0.0
seen          PPO         2/2                2.792 s    614.2 px   0.0
unseen_mild   BT          4/4                6.050 s    781.3 px   0.0
unseen_mild   PPO         4/4                3.613 s    788.9 px   1.0
ood_hard      BT          1/1                5.100 s    772.9 px   0.0
ood_hard      PPO         1/1                3.350 s    737.0 px   0.0
```

在 `m43_reverse_detour` 中，Agent 直径为 32 px：上方净余量从 baseline 的
68 px 缩小到 28 px，但真实 collision probe 确认仍可通过；下方净余量为
268 px。BT 选择下方且无碰撞；PPO 仍保持训练中形成的上方绕行偏好，发生
4 次 collision events 后成功到达。

这组结果说明 frozen PPO 通过了 4/4 个有限的 mild geometry probes，但没有
根据 Reverse Detour 的新几何切换绕行方向。它是对 route-selection bias 的
行为证据，不是任意地图上的 generalization 证明。

输出位置：

```text
experiments/comparisons/m43_generalization.csv
experiments/comparisons/m43_generalization_summary.json
experiments/comparisons/m43_runs/<scenario>/<controller>/
experiments/comparisons/m43_human_demos/<scenario>/<controller>/
```

## Behavior Tree Definition

当前 topology 定义在 `bt_configs/default.json`。`bt-lab/v1` 支持：

```text
selector
sequence
condition
action
```

通用字段包括 `type`、`name`、`behavior`、`memory`、`params` 和 `children`。

```text
bt_configs/default.json
        ↓
BT Loader
        ↓
Behavior Registry
        ↓
py_trees Runtime
        ↓
Visualizer / Experiment Log
```

JSON 决定 Tree topology、priority、组合和少量参数覆盖；Python Behavior class
负责节点的具体执行。JSON 未提供的参数继续使用当前 Scenario 的
`behavior_tree` 配置。

所有已注册 leaf 使用统一构建约定：

```python
Behavior(context=context, name=name, **params)
```

因此新增符合约定的 Behavior 只需实现 class 并添加一条 Registry 映射，
不需要专用 factory 或 Loader branch。

Visualizer 从真实 `py_trees` Runtime Tree 读取 topology、status、feedback 和
active path，而不是直接绘制 JSON。调整 JSON node 顺序或组合时，Runtime 与
Visualizer 会同步变化。

## Recommended Reading Order

源码包含面向学习的中文 Docstring 与 block comments，建议按以下顺序阅读：

```text
main.py
  → autonomy_lab/scene_config.py
  → autonomy_lab/environment.py / agent.py
  → autonomy_lab/perception.py
  → autonomy_lab/bt/context.py / behaviors.py
  → autonomy_lab/bt/registry.py / loader.py / controller.py
  → autonomy_lab/rendering/renderer.py / bt/visualizer.py
  → autonomy_lab/experiment/recorder.py / runners.py
  → autonomy_lab/gym/env.py
  → scripts/
```

一帧自主控制的数据流：

```text
Environment 当前 State
  → py_trees tick Conditions / Actions
  → Action 写入 context.command
  → Environment.step(command, 1/60)
  → motion / collision / termination / AgentPerception.update()
  → ExperimentRecorder.update() 记录真实结果
  → PygameRenderer / BTVisualizer 只读绘制
```

## Rendering Assets

Renderer 会尝试加载：

```text
assets/agent.png
assets/target.png
assets/obstacle.png
```

图片不存在或加载失败时自动 fallback 到 Pygame primitives。
`threat.png` 与 `waypoint.png` 目前只是预留视觉素材，不会增加对应业务行为。

## Experiment Recording

每次运行都会启动一个 Episode，并记录 Controller-independent metrics：

- Simulation time；
- path length；
- collision event count；
- trajectory samples；
- 可选 BT tick 与 active-action transition count。

结果状态包括：

- Target reached → `SUCCESS`；
- 达到 `max_episode_time` → `TIMEOUT`；
- 按 `R` 重置 → `manual_reset`；
- 关闭窗口 → `window_closed`。

默认输出：

```text
experiments/runs/episode_0001.json
experiments/results.csv
```

BT-controlled Episode 还会记录 `bt_config_id`；Manual Episode 该字段为空。
运行时生成的实验文件默认被 Git 忽略。
