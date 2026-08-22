# Autonomy Lab

Autonomy Lab 是一个基于 Python、Pygame、`py_trees`、Gymnasium 和
Stable-Baselines3 的二维自主智能体科研原型，用于快速验证：

- Behavior Tree 自主决策；
- Reinforcement Learning；
- Behavior Tree + RL 混合控制。

项目以可运行实验和可观察行为为优先，不以生产级框架或通用游戏引擎为目标。

## Current Project Structure

```text
project_root/
├── main.py                         # BT / manual Pygame 主入口
│
├── autonomy_lab/
│   ├── core/
│   │   ├── agent.py                # Agent 状态与运动更新
│   │   ├── environment.py          # 唯一 World / Simulation Core
│   │   └── observation.py          # Gym / Frozen PPO 共用的 13-D 编码
│   ├── perception/
│   │   ├── semantic_perception.py  # Goal / Hazard / Boundary 公共语义层
│   │   └── pygame_perception.py    # 当前 Pygame 感知 Provider
│   ├── scenarios/
│   │   ├── config.py               # 固定 Scenario 与实验参数
│   │   └── scenario_distribution.py # 可复现 Research Scenario Families
│   ├── bt/
│   │   ├── behaviors.py            # 可执行 Condition / Action 节点
│   │   ├── context.py              # BT 节点共享依赖
│   │   ├── parameters.py           # Research BT 运行时 Condition 参数
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
│   │   ├── env.py                  # Environment 的标准 Gymnasium Adapter
│   │   └── hybrid_env.py           # BT 调度上下文中的 PPO decision-step Adapter
│   │
│   └── experiment/
│       ├── recorder.py             # Controller-independent Episode metrics
│       └── runners.py              # BT / PPO 公共 Episode execution
│
├── scripts/
│   ├── demo/
│   │   ├── gym_demo.py             # headless Gymnasium smoke test
│   │   └── demo_scenario_distribution.py # sampled scene human demo
│   ├── training/
│   │   ├── train_ppo.py            # PPO training / fine-tuning
│   │   └── train_hybrid_ppo.py     # Hybrid-context training smoke
│   └── evaluation/
│       ├── eval_ppo.py             # Random / PPO evaluation
│       ├── compare_bt_ppo.py       # frozen BT vs PPO comparison
│       ├── eval_m43_generalization.py
│       ├── eval_m51_hybrid.py
│       ├── eval_m52_hybrid.py
│       ├── eval_m53_final.py
│       ├── analyze_hazard_sensing_range.py
│       ├── eval_action_competence.py  # R0.9/R0.10 fixed Action competence
│       ├── eval_condition_threshold_sensitivity.py  # R0.11 switching sweep
│       └── eval_context_threshold_necessity.py  # R0.12 paired contexts
│
├── bt_configs/
│   ├── default.json                # 原手工 BT baseline
│   ├── hybrid_ppo.json             # M5.0 Hybrid BT + Frozen PPO
│   └── condition_research.json      # R0.3 Parameterized Research BT
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
conda run -n pygame_lab python main.py --controller bt --bt hybrid_ppo --scenario ppo_simple_obstacle
conda run -n pygame_lab python main.py --controller bt --bt condition_research --scenario rl_sanity
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

在 [config.py](autonomy_lab/scenarios/config.py) 中新增或调整 Scenario。
每个 preset 包含 World 尺寸、seed、Agent、Target、Obstacle、显示参数和少量
Behavior Tree 实验参数。

## R0.1 Perception & Navigation Regression

R0.1 将两类几何语义明确分开：Target sensing 继续只回答目标是否处于
range/FOV/光学 LOS 内；Agent 是否能通过则由 footprint-aware obstacle/free-space
clearance 判断。`AgentPerception` 额外提供 12 个全方向 sector clearances 给 BT
Safety Actions 使用，但冻结的 PPO 13-D Observation 字段、顺序与数值语义不变。

默认 BT 现在先执行 `Boundary Risk? → Safe Boundary Recovery`。Boundary Recovery
与 Obstacle Avoidance 通过同一个轻量 safe-steering 评分同时考虑 obstacle clearance、
boundary clearance 和期望方向，避免两条安全分支把 Agent 互相推回风险区。

两个固定 Regression Scenario 可直接观察：

```powershell
conda run -n pygame_lab python main.py --controller bt --bt default --scenario r01_narrow_passage
conda run -n pygame_lab python main.py --controller bt --bt default --scenario r01_boundary_obstacle
```

- `r01_narrow_passage`：24 px 光学狭缝小于约 32 px Agent 直径；Target 可感知，
  但中心方向被 free-space sensing 判定为不可通。
- `r01_boundary_obstacle`：Agent 从 Boundary 与 Obstacle 的夹区启动，用于观察
  Boundary Recovery 绕开内侧障碍并脱离风险区。

这两个 Scenario 仅用于 R0.1 回归，不参与 PPO 训练。

## R0.2 Semantic Perception

`AgentPerception` 现在只执行一套几何计算，并以 `SemanticPerception` 作为主要
输出：

```text
Environment Ground Truth
        ↓
AgentPerception
        ↓
SemanticPerception
├── AgentState
├── GoalPerception
├── HazardPerception
└── BoundaryPerception
        ↓
Legacy 13-D PPO / current BT / future research adapters
```

- Goal 只表达 sensed/visible/available、距离和方位，不表达路径能否通过。
- Hazard 包含局部障碍 clearance、legacy 12 / Research 16 方向 sector ranges
  和可通行 gaps。
- Boundary 提供 Agent 圆形碰撞体到四条 World 边界的像素净空。
- 所有 semantic objects 都是 frozen 纯数据，不携带 `pygame.Rect`、World 或
  Surface。
- `target_visible`、`nearest_obstacle` 等历史字段仅是同一快照上的只读属性映射，
  不会触发第二次 perception computation。

冻结的 M4/M5 13-D Observation 仍保持原 shape、顺序、归一化、`float32` 和
不可见对象中性值。历史 Scenario、checkpoint、CSV 与 BT JSON 命名继续保留
Target/Obstacle，以免破坏已有实验资产。

## R0.3 Parameterized Research BT

`condition_research.json` 是独立于 legacy BT 和 Hybrid BT 的固定研究树：

```text
Priority Selector
├── Boundary Recovery: BoundaryRisk(theta_boundary) → SafeBoundaryRecovery
├── Hazard Avoidance:  HazardRisk(theta_hazard) → AvoidHazard
├── Goal Reached:      GoalReached(theta_goal) → Stop
└── Move To Goal
```

Topology 和 handcrafted Actions 保持固定，只有三个连续 Condition threshold 可在
运行时调整。默认值分别为 `hazard=90 px`、`boundary=40 px`、`goal=30 px`；
允许设置任意有限非负数值。Condition 每次 tick 都读取同一个
`ConditionParameters`，不会把构建时数值缓存到节点中：

```python
from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.bt.parameters import ConditionParameters

parameters = ConditionParameters(hazard_threshold=60.0)
controller = BehaviorTreeController(
    world,
    bt_config="condition_research",
    condition_parameters=parameters,
)

controller.tick(1.0 / 60.0)
parameters.hazard_threshold = 100.0
controller.tick(1.0 / 60.0)  # 此 tick 立即使用 100 px
```

三个 Research Condition 只读取当前 `SemanticPerception.goal`、`.hazard` 和
`.boundary`。Visualizer feedback 会显示观测距离/净空及当前 `theta`，便于比较
同一 observation 在不同 threshold 下的分支选择。本阶段不包含 Condition-RL、
参数平滑、hysteresis 或动态 topology。

## R0.4 Scenario Distribution

R0.4 在固定 M4/M5/R0 regression scenarios 之外新增独立研究采样路径：

```python
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution

scene = ScenarioDistribution("dynamic_hazard").sample(seed=42)
world = Environment(scene)
```

当前 family：

- `static_random`：随机 Agent pose、Goal 与两个静态 Hazard；
- `dense_hazard`：五个更紧凑、但保留外侧通路的静态 Hazard；
- `dynamic_hazard`：增加一个恒速、边界反射的移动 Hazard；
- `noisy_perception`：只扰动 Semantic Hazard range measurement；
- `context_shift`：按 simulation time 执行 low-risk → high-risk → recovery；
- `narrow_passage` / `boundary_hazard`：R0.1 固定 regression geometry alias。

`sample(seed)` 使用局部 RNG；相同 `family + seed` 会还原相同起点、Goal、
Hazard geometry、动态初态、噪声序列和 schedule。静态与动态 Hazard 都进入
Environment 的同一个 `obstacles` Rect 列表，共用 collision、LOS、sector、gap
和 rendering 逻辑。

轻量诊断可直接读取：

```python
world.scenario_metadata
world.dynamic_hazard_states
world.current_noise_level
world.current_context_phase
```

本阶段没有 Condition-RL、curriculum、procedural maze 或通用事件编排。

直接打开 R0.4 可视化窗口：

```powershell
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dynamic_hazard --seed 42
```

替换 `--family` 即可观察其他场景族，例如：

```powershell
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family static_random --seed 42
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dense_hazard --seed 42
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family noisy_perception --seed 42
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family context_shift --seed 42
```

窗口使用 `condition_research` BT，状态栏显示 Family、Hazard 数量、Noise 和
Context Phase。按 `R` 重放同一 seed，关闭窗口退出。

手工设置 R0.3 Condition threshold：

```powershell
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dynamic_hazard --seed 5001 --hazard-threshold 110 --boundary-threshold 45 --goal-threshold 30
```

## R0.5 Stabilized Research Interface

R0.5 冻结两条用途不同、但共享 World 和 perception computation 的数据流：

```text
Legacy M4/M5
Environment → AgentPerception → frozen 13-D Observation → PPO / Hybrid

Research R0/M6+
ScenarioDistribution → Environment → SemanticPerceptionProvider
→ SemanticPerception → Parameterized Research BT
→ AgentCommand → Environment.step()
```

`SemanticPerceptionProvider` 的最小 contract 是每个 control tick 调用一次
`observe()`，随后所有 Method/BT leaf 只读同一份 `snapshot`。当前
`AgentPerception` 是 Pygame provider；未来 simulator 只需实现相同 contract，
不需要复制 Parameterized Conditions 或 handcrafted Actions。

跨 simulator 的 Core semantics：

- `AgentState`：speed、heading、radius；
- `GoalPerception`：availability、distance、bearing；
- `HazardPerception`：nearest/visible Hazard 与 sector ranges。

Optional semantics 包括 `BoundaryPerception` 和 Pygame-derived traversable gaps，
分别使用 `available`、`gaps_available` 明确表示缺失，不伪造可用观测。动态 Hazard
和 noise/context metadata 仍属于 simulator/Scenario diagnostics；其当前测量结果
通过普通 Hazard semantics 提供给 Research Method。

`ConditionParameters` 的稳定接口：

```python
parameters.get_values()
parameters.set_values(hazard_threshold=100.0)
parameters.reset_defaults()
parameters.get_bounds()
```

本阶段仅冻结接口，没有 Safety-Gym dependency、SafetyGym adapter、Condition-RL
或 PPO → delta-theta。

## R0.6 Generic Parameter Interface

`autonomy_lab.bt.parameters` 现在用同一套薄接口描述和保存连续参数：

```python
from autonomy_lab.bt.parameters import ParameterSpec, ParameterStore

store = ParameterStore([
    ParameterSpec(
        name="avoid_turn_gain",
        value=1.0,
        default=1.0,
        min_value=0.25,
        max_value=2.0,
    )
])

store.set("avoid_turn_gain", 1.4)
gain = store.get("avoid_turn_gain")
store.reset("avoid_turn_gain")
```

通用 API 为 `get(name)`、`set(name, value)`、`reset(name)`、`reset_all()`、
`bounds(name)` 和 `spec(name)`。越界、非数值与非有限 current value 会显式报错，
不会被静默 clip。`spec()` 返回副本，外部不能绕过 `set()` 修改 Store 内部状态。

现有 `ConditionParameters` 是该 Store 的 Research BT 兼容入口，继续保留
`hazard_threshold`、`boundary_threshold`、`goal_threshold` 属性以及 R0.5 的批量
方法。三个 Condition 每次 tick 按参数 key 调用 `get()`；节点不知道参数来自
Manual、未来 PPO 或 CMA-ES。Store 本身也不区分 Condition 与 Action 参数，但
R0.6 尚未迁移或新增任何 Action 参数，也没有实现 optimizer。

## R0.7 Safety-Gym-aligned Finite-Range Sensing

R0.7 首次通过同一 `AgentPerception` 的 `research` profile 引入有限距离 360°
sensing；固定 M4/M5 Scenario 未声明 profile，仍走冻结的 legacy range/FOV/LOS
与 12-sector 路径。R0.7 的初始 Research 配置为 Goal/Hazard 均 `700 px`，随后由
R0.8 的固定样本统计校准。

```text
Research Goal lidar       360° / 16 bins / finite range
Research Hazard lidar     360° / 16 bins / finite range
Legacy M4/M5 sensing      unchanged
```

- Goal 在配置量程内暴露物理 distance、bearing 和 `sector_index`；超距时
  `sensed=False`、`available=False`，distance/bearing/sector 均为 `None`。
- Hazard 只在配置量程内进入 nearest/visible semantics；空 sector 返回当前
  `hazard_range`，命中 sector 返回 footprint-aware clearance。
- Research Hazard lidar 不编码 World Boundary；四向边界净空仍由独立
  `BoundaryPerception` 提供。
- Goal sensing 不检查 FOV、LOS 或路径可通行性。狭缝中可以同时成立
  `Goal sensed` 与 `Goal direction blocked`。
- noisy family 的顺序为 true geometry → finite-range gate → seeded noise →
  `SemanticPerception`；超距对象不会被噪声泄漏，World Rect 也不会被修改。

Research human demo 会绘制 Goal/Hazard sensing radius、16 条 Hazard sector 射线，
并在状态栏显示 Goal sensed 与 nearest Hazard clearance：

```powershell
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family narrow_passage --seed 901
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family noisy_perception --seed 42
```

本阶段只对齐 finite-range/object-specific lidar 机制；没有引入 Safety-Gymnasium
dependency、adapter、normalized lidar Observation、Condition-RL 或 M6 功能。

## R0.8 Research Sensing Semantics & Hazard Range Calibration

第一篇论文路径现在明确采用：

```text
Goal       → 360° / 850 px long-range task signal → distance / bearing
Hazard     → 360° / 300 px local safety coverage → nearest semantics
16 sectors → local free-space representation      → handcrafted steering
```

校准使用 `static_random`、`dense_hazard`、`dynamic_hazard`、
`noisy_perception`、`context_shift`，每个 family 使用 seeds `1001–1050`，共
250 个 `family + seed` 初态。结果如下：

同一 seed 在五个 family 中复用相同 Agent/Goal 初态，因此 Goal sensing rate 对应
50 个唯一 Goal geometries；Hazard 统计则继续以 250 个 family-seed 初态为单位。

| Hazard range | Availability | Mean visible count | All hazards visible |
|---:|---:|---:|---:|
| 200 px | 25.2% | 0.252 | 0.0% |
| 300 px | 92.0% | 1.100 | 0.0% |
| 400 px | 100.0% | 1.788 | 0.0% |
| 500 px | 100.0% | 2.712 | 71.6% |
| 700 px | 100.0% | 3.000 | 100.0% |

因此原 `700 px` 实际近似全局 Hazard sensing；默认改为 `300 px`，仍保留较高
局部输入可用率，但不会在 reset 时看见全部 Hazard。原 `goal_range=700 px` 在同一
样本集的 initial sensing rate 只有 `54.0%`；为避免把第一篇论文变成 Search/Memory
研究，Research Goal 长程范围提高到 `850 px`，initial sensing rate 为 `100.0%`。

感知数据流按职责分为：

```text
Sensor Coverage
    ↓
visible Hazards
    ↓
nearest clearance / bearing       # HazardRisk 只读这里
    ↓
16-sector local free-space        # AvoidHazard / steering 使用
```

运行校准分析（不启动 Controller，不训练模型）：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.analyze_hazard_sensing_range
```

默认结果写入 `experiments/analysis/r08_hazard_sensing_range.json`。Pygame Research
Testbed 使用 long-range Goal + local Hazard；未来 Safety-Gym external benchmark
保留其原生 sensing。两者只要求在 `SemanticPerception`、Condition 参数、BT 语义、
adaptation algorithm 和评价指标层对齐，不强求 raw lidar、量程、尺度或物理一致。

## R0.9 Fixed Action Competence Validation

R0.9 不训练模型，也不修改 Action/Condition/BT。评估入口直接复用真实
`SemanticPerception`、Action Node、`AgentCommand` 和 `Environment.step()`：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_action_competence
```

如只需快速复查确定性的 Action micro-scenarios：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_action_competence --isolated-only
```

隔离结果为：`MoveToGoal 5/5`、`Stop 2/2`、`AvoidHazard 3/6`、
`SafeBoundaryRecovery 0/7`。Boundary Recovery 用例最终可以回到 40 px 安全净空，
但从贴近边界且朝外的状态恢复时产生 collision event，因此按严格无碰撞标准失败；
AvoidHazard 在近距离/宽 Hazard 及靠近下边界的用例中同样发生碰撞。

默认 Research BT 在五个 family、seeds `1001–1050` 上的 success rate 为：
`static_random 76%`、`dense_hazard 64%`、`dynamic_hazard 82%`、
`noisy_perception 78%`、`context_shift 82%`；未成功 episode 均因 20 s timeout。
因此 R0.9 validation 已完成，但当前 fixed Action substrate 尚不建议冻结为 M6.1
前置 baseline。详细结果写入：

```text
experiments/analysis/r09_action_competence.json
experiments/analysis/r09_action_competence.csv
```

## R0.10 Fixed Action Safety Stabilization

R0.10 只调整 Research 路径中的 `AvoidHazard` 与 `SafeBoundaryRecovery`：

- `SafeBoundaryRecovery` 在安全 bearing 与当前 heading 相差超过 35° 时令
  `throttle=0`，先完成转向，再恢复固定 recovery throttle。
- `AvoidHazard` 在 Action 层将独立的 Hazard sector clearance 与 directional
  Boundary clearance 合并评分，锁定本次安全绕行侧；未对齐时先转向，对齐后推进。
- 固定参数集中在 scene behavior config：Avoid/Recovery alignment threshold 均为
  35°，turn gain 均为 45°。Condition、BT topology、Perception 与 Scenario 未改。

原 R0.9 微场景复测结果：

```text
MoveToGoal            5/5, collisions=0
Stop                  2/2, collisions=0
AvoidHazard           6/6, collisions=0
SafeBoundaryRecovery  7/7, collisions=0
```

相同五个 family、seeds `1001–1050` 的端到端结果：

| Family | R0.9 success | R0.10 success | R0.10 collision episode rate |
|---|---:|---:|---:|
| static_random | 76% | 86% | 2% |
| dense_hazard | 64% | 18% | 10% |
| dynamic_hazard | 82% | 70% | 16% |
| noisy_perception | 78% | 68% | 0% |
| context_shift | 82% | 68% | 14% |

安全 Action 的局部碰撞缺陷已消除，但完整 reactive BT 的 overall success 从
`191/250` 降为 `155/250`，尤其 dense family 长期被保守避障抢占并 timeout。
因此 R0.10 未满足完整验收条件，fixed Action substrate 暂不冻结，也不进入 M6.1。
R0.10 输出位于：

```text
experiments/analysis/r010_action_competence.json
experiments/analysis/r010_action_competence.csv
```

## R0.13 Hazard Action Commitment Fix

Research BT 的 `Hazard Avoidance` Sequence 现在使用 `memory=true`。这只改变
`py_trees` 的 running-child execution semantics：`HazardRisk` 仍以单阈值决定一次
maneuver 是否启动；一旦 `AvoidHazard` 返回 `RUNNING`，Sequence 会继续 tick 该
Action，直到它成功完成。更高优先级的 `Boundary Recovery` 仍可在任意 tick 立即
抢占，并把运行中的 `AvoidHazard` 正常置为 `INVALID`。

已确认的 `dynamic_hazard / seed=1001 / hazard_threshold=45` episode 对比：

| Metric | 修复前 | 修复后 |
|---|---:|---:|
| Success / collision | True / 0 | True / 0 |
| Branch switches | 26 | 6 |
| HazardRisk activations | 13 | 2 |
| Avoid maneuvers | 约 13 次被反复重启 | 2 次完整 maneuver |
| Longest continuous Avoid | 1.25 s | 1.27 s |
| Path length | 1096.15 px | 1090.83 px |

因此原案例中的高频切换已被消除，但这不意味着所有场景性能都会自动提高：完整
R0.11/R0.12 复跑显示，真正执行完避障 commitment 会增加总体 Avoid occupancy，
部分 episode 因而更容易 timeout。旧结果保存在 `*_pre_r013.*` 快照中，当前标准
结果文件均为 R0.13 semantics 下重新生成的数据。

## R0.11 Switching Bottleneck Attribution

R0.11 不修改 Action、Condition、BT topology、Perception 或场景，只在每个
evaluation episode 构建 Research BT 时覆盖运行时 `hazard_threshold`：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_condition_threshold_sensitivity
```

固定 sweep 为 `45, 63, 76.5, 90, 103.5, 117, 135 px`，每个阈值复用五个
Research family 和 seeds `1001–1050`，即每个阈值 250 个 paired episodes。
`boundary_threshold=40 px`、`goal_threshold=30 px` 与默认参数文件始终不变。

| θ_hazard | Success | Timeout | Collision episode | AvoidHazard ratio |
|---:|---:|---:|---:|---:|
| 45.0 | 82.0% | 18.0% | 8.0% | 48.4% |
| 63.0 | 68.8% | 31.2% | 8.4% | 55.3% |
| 76.5 | 52.4% | 47.6% | 9.6% | 61.9% |
| 90.0 | 36.0% | 64.0% | 8.0% | 66.4% |
| 103.5 | 27.2% | 72.8% | 5.6% | 68.3% |
| 117.0 | 22.4% | 77.6% | 3.6% | 69.7% |
| 135.0 | 18.4% | 81.6% | 4.4% | 72.2% |

R0.13 semantics 下结果仍支持 **Case A — Switching bottleneck supported**：阈值
45→135 px 时 success 由 82.0% 降到 18.4%，Avoid occupancy 由 48.4% 升到
72.2%，H2 Condition Sensitivity 仍成立。与此同时，平均 branch switches 从修复前
的 37.5–92.2 降为 12.2–25.6，HazardRisk activations 从 18.3–41.7 降为
3.1–5.0，而 longest continuous Avoid 增至 1.38–1.53 s，说明测到的是阈值对
完整 maneuver commitment 的影响，不再主要是 Action 被重复中断。45 px 仍是五个
family 的最佳 success 选择，但该值仅作为分析结果，没有写回默认配置。

输出：

```text
experiments/analysis/r011_threshold_sensitivity.json
experiments/analysis/r011_threshold_sensitivity.csv
experiments/analysis/r011_threshold_sensitivity_episodes.csv
```

## R0.12 Context-Dependent Threshold Necessity

R0.12 复用同一 `dynamic_hazard` seed 的 Agent、Goal、静态/动态 Hazard 几何，
只把动态 Hazard speed 设为 `36 px/s`（Low Risk）或 `180 px/s`（High Dynamic
Risk）。两个 context 使用完全相同的 `20/30/40/45/60/75/90 px` threshold grid
和 seeds `1001–1050`：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_context_threshold_necessity
```

`hazard_exposure` 是只用于 evaluation 的连续 Hazard Proximity Exposure：

```text
q_t = 0                                           if no Hazard is sensed
q_t = max(0, 1 - max(clearance, 0) / 300)^2       otherwise
hazard_exposure = mean(q_t over simulation steps)
```

它读取 Semantic nearest Hazard clearance，不进入 Reward、Condition、Action 或
参数更新。主要静态 context 结果：

| Context | θ | Success | Collision | Exposure | Min clearance | Avoid ratio | Mean time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 20 | 96% | 6% | 0.5624 | 12.7 | 34.9% | 7.49 s |
| Low | 30 | 92% | 10% | 0.5440 | 14.5 | 40.4% | 8.52 s |
| Low | 40 | 94% | 12% | 0.5379 | 17.6 | 43.2% | 8.99 s |
| Low | 45 | 96% | 6% | 0.5188 | 19.5 | 43.3% | 8.91 s |
| Low | 60 | 90% | 0% | 0.4866 | 26.4 | 50.9% | 10.24 s |
| Low | 75 | 74% | 6% | 0.4659 | 27.1 | 57.6% | 13.00 s |
| Low | 90 | 48% | 10% | 0.4610 | 20.2 | 63.7% | 16.01 s |
| High | 20 | 98% | 38% | 0.5616 | 11.4 | 35.7% | 7.45 s |
| High | 30 | 92% | 34% | 0.5524 | 13.6 | 40.2% | 8.15 s |
| High | 40 | 98% | 32% | 0.5307 | 15.9 | 42.7% | 7.58 s |
| High | 45 | 94% | 34% | 0.5263 | 15.0 | 45.8% | 8.59 s |
| High | 60 | 90% | 36% | 0.5042 | 15.5 | 52.7% | 10.84 s |
| High | 75 | 54% | 50% | 0.4984 | 10.1 | 61.3% | 14.10 s |
| High | 90 | 62% | 54% | 0.4644 | 9.3 | 63.3% | 15.28 s |

固定 diagnostic score 为 `success − collision_episode − 1.0 × exposure`。Low
Risk 首选 `60 px`（score 0.4134），High Dynamic Risk 首选 `40 px`（score
0.1293）；两者对对方 threshold 的 advantage 分别为 0.1313 和 0.0935，均超过
固定 `0.02` crossing criterion。方向仍不能简化成“风险越高 θ 越大”。

静态 crossing 成立后按原 gate 执行了 Low→High→Low。该 within-episode
diagnostic 没有独立复现同样清晰的 `60/40 px` preference reversal：High phase
中 40 px 的 collision/exposure 为 29.5%/0.5735，60 px 为 25.0%/0.5228；Low
phase 1 中 40 px 为 2.0%/0.5261，60 px 为 0%/0.4884。各 threshold 进入后续
phase 的 episode 数还会受提前成功影响，因此 H3 结论严格来自固定 paired static
contexts 的预注册 balanced-score crossing，而不是把 phase 数据解释为独立重复。

因此 R0.12 归类为 **Case A — Context Dependence Supported**：不存在统一支配
两个 tested contexts 的固定 threshold，H3 得到支持。输出：

```text
experiments/analysis/r012_context_threshold_necessity.json
experiments/analysis/r012_context_threshold_necessity.csv
experiments/analysis/r012_context_threshold_episodes.csv
```

| 假设                             | 证据                                                            | 状态            |
| ------------------------------ | ------------------------------------------------------------- | ------------- |
| H1 Action Competence           | AvoidHazard 6/6、BoundaryRecovery 7/7，零碰撞                      | **SUPPORTED** |
| H2 Condition Sensitivity       | θ 从 45→135 导致 Avoid occupancy 48.4%→72.2%，success 82.0%→18.4% | **SUPPORTED** |
| H3 Context Dependence          | paired static contexts 中 Low 首选 60 px，High Dynamic 首选 40 px；phase test 未独立复现反转 | **SUPPORTED（有限）** |
| H4 Online Adaptation Advantage | RL 能否在线逼近 context-appropriate θ                               | **待 M6 验证**   |

```BASH
# 较晚触发 AvoidHazard
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dynamic_hazard --seed 1001 --hazard-threshold 20

# Low Risk 中综合表现较好的阈值
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dynamic_hazard --seed 1001 --hazard-threshold 45

# 默认保守阈值
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dynamic_hazard --seed 1001 --hazard-threshold 90
```

### R0.10/R0.11 场景可视化

现有场景 demo 会加载同一棵 Research BT。可分别指定 45 px 与 135 px，直观比较
较晚/较早进入 `AvoidHazard` 时的分支占用差异：

```powershell
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dense_hazard --seed 1001 --hazard-threshold 45
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dense_hazard --seed 1001 --hazard-threshold 135
```

其他可观察 family：

```bash
# 普通静态障碍导航
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family static_random --seed 1001

# 密集障碍，适合观察 R0.10 的保守/卡住失败模式
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dense_hazard --seed 1001

# 动态障碍
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family dynamic_hazard --seed 1001

# 上下文变化
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family context_shift --seed 1001

# 感知噪声
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family noisy_perception --seed 1001

# 狭窄通道
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --family narrow_passage --seed 901

```

## Script Entry Points

非交互式入口位于 `scripts/`：

```powershell
conda run -n pygame_lab python -m scripts.demo.gym_demo
conda run -n pygame_lab python -m scripts.demo.demo_scenario_distribution --help
conda run -n pygame_lab python -m scripts.training.train_ppo --help
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --help
conda run -n pygame_lab python -m scripts.evaluation.compare_bt_ppo --help
conda run -n pygame_lab python -m scripts.evaluation.eval_m43_generalization --help
conda run -n pygame_lab python -m scripts.evaluation.eval_m51_hybrid --help
conda run -n pygame_lab python -m scripts.training.train_hybrid_ppo --help
conda run -n pygame_lab python -m scripts.evaluation.eval_m52_hybrid --help
conda run -n pygame_lab python -m scripts.evaluation.eval_m53_final --help
conda run -n pygame_lab python -m scripts.evaluation.analyze_hazard_sensing_range --help
conda run -n pygame_lab python -m scripts.evaluation.eval_action_competence --help
conda run -n pygame_lab python -m scripts.evaluation.eval_condition_threshold_sensitivity --help
conda run -n pygame_lab python -m scripts.evaluation.eval_context_threshold_necessity --help
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
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario rl_sanity --model-path models/ppo_m40.zip --controller ppo --episodes 1 --evaluation-seed-start 1001 --tag m40_manual_view --render-mode human
```

### M4.1 — Hard Narrow-gap Baseline

M4.1 使用 `ppo_simple_obstacles`。200k 和 500k checkpoints 的 Random/PPO
success rate 均为 0%，因此该模型被保留为失败证据，不作为成功导航模型。

查看典型失败轨迹：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacles --model-path models/ppo_m41_obstacles.zip --controller ppo --episodes 1 --evaluation-seed-start 2001 --tag m41_manual_view --render-mode human
```

该 deterministic PPO 接近障碍后进入局部控制循环，并在 20 秒 Simulation
horizon 后 timeout。

### M4.1a — Simple Obstacle Diagnosis

M4.1a 将任务降级为 `ppo_simple_obstacle`，但保持 60 Hz PPO decision frequency。
200k 与 500k 仍为 0% success：Policy 接近障碍、发生一次 collision event，
随后几乎保持 full throttle 且缺少有效转向，最终 timeout。

查看 M4.1a：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41a_manual_view --render-mode human
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
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41b_manual_view --render-mode human --action-repeat 6
```

复现 Random/PPO fixed-seed comparison：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41b_manual_check --render-mode none --action-repeat 6
```

Adapter 默认值仍为 `action_repeat=1`、`contact_penalty_per_step=0.0`，保证
M4.0/M4.1 的旧调用语义不变。

## M4.2 — BT vs PPO Baseline

`scripts.evaluation.compare_bt_ppo` 在相同 Environment、Scenario、seed 和 World 初态下
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
conda run -n pygame_lab python -m scripts.evaluation.compare_bt_ppo
```

运行独立 human demos：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.compare_bt_ppo --human-demo
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

`scripts.evaluation.eval_m43_generalization` 复用公共 Episode runners，在不重新训练或调参
的前提下评估 frozen BT 与 PPO：

- `seen`：`rl_sanity`、`ppo_simple_obstacle`；
- `unseen_mild`：`m43_target_shift`、`m43_obstacle_shift`、
  `m43_reverse_detour`、`m43_combined_shift`；
- `ood_hard`：`ppo_simple_obstacles`。

运行 headless evaluation：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_m43_generalization
```

运行 8 个独立 human observation Episodes：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_m43_generalization --human-demo
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

## M5.0 — Frozen PPO Action Integration

M5.0 不重新训练模型，而是把冻结的
`models/ppo_m41b_control10hz.zip` 作为标准 `py_trees` Action Node
`PPONavigate` 接入独立的 `bt_configs/hybrid_ppo.json`：

```text
Priority Selector
├── Boundary Recovery
│   ├── Boundary Risk?
│   └── Safe Boundary Recovery
├── Learned Navigation
│   ├── Target Visible?
│   └── PPO Navigate
└── Search Target
```

启动 Hybrid human demo：

```powershell
conda run -n pygame_lab python main.py --controller bt --bt hybrid_ppo --scenario rl_sanity
conda run -n pygame_lab python main.py --controller bt --bt hybrid_ppo --scenario ppo_simple_obstacle
```

World 与 BT tick 保持 60 Hz。`PPO Navigate` 复用 Gym 训练时完全相同的
13-D Observation 编码，只约每 0.1 s 调用一次 deterministic `model.predict()`，
其余约 6 个 simulation ticks 继续输出缓存的 `[turn, throttle]`。Action Node
不会调用 `Environment.step()`。

Boundary Safety 每个 BT tick 都先于 PPO 检查，因此可在 60 Hz 抢占 Learned
Navigation；Target 不可见时则回退到原 `Search Target`。Hybrid Tree 刻意不含
手工 `Obstacle Avoidance` 分支，单障碍绕行由 Frozen PPO 本身完成。

固定 seeds 6001–6003 的最小回归中，Pure PPO 与 Hybrid 在 `rl_sanity` 和
`ppo_simple_obstacle` 都为 3/3 成功、0 collision，并分别得到完全一致的
1.983 s / 436.3 px 与 3.600 s / 792.0 px。Episode 日志位于：

```text
experiments/m50_regression/<scenario>/<hybrid|ppo>/
```

## M5.1 — Hybrid Evaluation & Behavior Analysis

M5.1 冻结 `default.json`、`hybrid_ppo.json` 与
`models/ppo_m41b_control10hz.zip`，在同一 World 初态和公共指标下比较手工 BT、
Pure PPO 与 Hybrid BT + PPO。World 和 BT supervision 为 60 Hz；PPO 仅在活跃时
约 10 Hz 决策。固定布局各使用一次 seed 5001，统计单位是 Scenario，不代表随机
重复试验成功概率。

运行 batch evaluation：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_m51_hybrid
python -m scripts.evaluation.eval_m51_hybrid --human-demo

# Pure BT
conda run -n pygame_lab python main.py --controller bt --bt default --scenario ppo_simple_obstacle

# Hybrid BT + PPO
conda run -n pygame_lab python main.py --controller bt --bt hybrid_ppo --scenario ppo_simple_obstacle

```

独立运行 `ppo_simple_obstacle`、`m43_reverse_detour` 与
`ppo_simple_obstacles` 的三 Controller Human Demo（分别观察普通导航、Boundary
抢占和 Search 切换；不会写入或污染 batch 结果）：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_m51_hybrid --human-demo
```

本次冻结评估结果：

| Controller | seen | unseen_mild | ood_hard | mild 平均耗时 | mild 平均路径 | mild 平均碰撞 |
|---|---:|---:|---:|---:|---:|---:|
| BT | 2/2 | 4/4 | 1/1 | 6.050 s | 781.3 px | 0.00 |
| PPO | 2/2 | 4/4 | 1/1 | 3.613 s | 788.9 px | 1.00 |
| Hybrid | 2/2 | 3/4 | 0/1 | 7.625 s | 669.2 px | 1.25 |

Hybrid 在 seen 以及三个 mild 场景中全程由 PPO 分支控制，轨迹与 Pure PPO
一致。在 `m43_reverse_detour` 中，Hybrid 最初继承 PPO 的上方绕行选择，随后
Boundary Recovery 激活并抢占 PPO；最终在 20 s horizon 超时。该 Episode 的 PPO
active ratio 为 0.084，发生 1 次 Boundary Recovery activation 和 1 次 PPO
preemption。在 hard `ppo_simple_obstacles` 中，Target 很快不可见，Search
activation 1 次，PPO active ratio 仅 0.014，最终同样超时。

这说明当前 Hybrid 在普通 visible-target 条件下等价于 PPO，但高优先级 BT 分支
并未在这两个压力场景中形成有效恢复，反而改变了原本成功的 PPO 轨迹。该结论只
适用于列出的固定场景，不能解释为任意地图 generalization。

输出位置：

```text
experiments/comparisons/m51/m51_bt_ppo_hybrid.csv
experiments/comparisons/m51/m51_bt_ppo_hybrid_summary.json
experiments/comparisons/m51/runs/<scenario>/<controller>/
experiments/comparisons/m51/human_demos/<scenario>/<controller>/
```

## M5.2 — Hybrid PPO Lab Adapter

当前阶段聚焦 Hybrid Lab 框架是否正确，不以长时间 PPO 训练或策略收敛为目标。
`HybridPPOEnv` 复用真实 `hybrid_ppo.json` 和 `py_trees` Runtime，把一个 Gym
`step(action)` 定义为“一次 PPO decision 到下一个 PPO decision point”：

```text
PPO 提交 [turn, throttle]
  → 最多控制 6 个 1/60 s World steps
  → BT 仍在每个 World step 以 60 Hz supervision
  → Boundary Recovery / Search 可接管并输出自己的 Command
  → PPO 重新获得控制、Episode 结束或 horizon 到达时返回
```

因此 reward、elapsed time、trajectory、collision 和 simulation time 都按实际
World internal steps 聚合；BT 自主接管期间的动作不会被错误计作 PPO action。
Observation 仍是同一份 13-D 感知编码，Ground Truth target distance 仍只用于
已有 privileged progress reward，不进入 Policy Observation。

默认训练命令现在只运行 2,048 个 PPO decision steps，用于检查 Env、SB3、模型
save/load 和 logging 接线。需要长实验时必须显式提供 `--target-timesteps`：

```powershell
# 推荐的当前用法：短 smoke test
conda run -n pygame_lab python -m scripts.training.train_hybrid_ppo `
  --model-path models/ppo_m52_smoke.zip `
  --log-label m52_hybrid_smoke

# 读取已有 checkpoint 做结构化评估；不触发训练
conda run -n pygame_lab python -m scripts.evaluation.eval_m52_hybrid `
  --model-path models/ppo_m52_hybrid_trained_200k.zip `
  --checkpoint-label 200k --human-demo
```

框架验证已确认：同一个 frozen M4.1b Policy 经旧 Hybrid runner 和新的
external-action Adapter 运行 7 个固定场景时，success、elapsed time、path length
和 collision count 一致；Boundary 抢占测试也确认 PPO action 会立即停止，BT
恢复完成后才返回新的 PPO decision point。

在切换为框架优先之前已完成一次 200,704-step 试跑，Hybrid-relevant 场景仍为
1/2 成功，与 Frozen Hybrid 相同，因此该结果只作为诊断产物保留，不视为本阶段
验收目标。后续 500k 续训已停止，且没有保存不完整 checkpoint。

输出位置：

```text
experiments/comparisons/m52/200k/m52_frozen_vs_trained.csv
experiments/comparisons/m52/200k/m52_frozen_vs_trained_summary.json
experiments/comparisons/m52/200k/runs/<scenario>/<controller>/
```

## M5.3 — Hybrid BT-RL Final Evaluation

M5.3 不训练模型，统一冻结并评价：

```text
BT                  = default.json
Pure PPO            = ppo_m41b_control10hz.zip
Frozen Hybrid       = hybrid_ppo.json + frozen M4.1b PPO
Hybrid-trained PPO  = ppo_m52_hybrid_trained_200k.zip
```

运行最终 headless evaluation：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_m53_final
```

运行两个代表场景 × 四 Controller 的独立 human demo：

```powershell
conda run -n pygame_lab python -m scripts.evaluation.eval_m53_final --human-demo
```

固定 seed 5001 下的最终结果如下。均值按组内 **all episodes** 计算，timeout
不会被排除；固定 Scenario 是统计单位，不代表随机地图泛化率。

| Controller | Seen | Mild-Unseen | OOD-Hard | Mild 平均耗时 | Mild 平均路径 | Mild 平均碰撞 |
|---|---:|---:|---:|---:|---:|---:|
| BT | 2/2 | 4/4 | 1/1 | 6.050 s | 781.3 px | 0.00 |
| Pure PPO | 2/2 | 4/4 | 1/1 | 3.613 s | 788.9 px | 1.00 |
| Frozen Hybrid | 2/2 | 3/4 | 0/1 | 7.625 s | 669.2 px | 1.25 |
| Hybrid-trained PPO | 2/2 | 3/4 | 0/1 | 7.617 s | 668.7 px | 1.50 |

最终结论：

- Frozen Hybrid 在两个 Seen 场景保留了 Pure PPO 的成功能力和公共指标；
- `m43_reverse_detour` 中，Pure PPO 虽有 4 次 collision 仍成功，Frozen Hybrid
  在一次 Boundary preemption 后没有重新进入 PPO，20 s timeout；
- Hybrid-trained 在 Reverse Detour 同样 timeout，发生 6 次 collision，未证明
  约 200k Hybrid-context adaptation 带来成功率提升；
- hard narrow-gap 中，两种 Hybrid 都在 Target lost 后进入 Search，PPO active
  ratio 仅约 0.014，说明主要限制来自当前 Search / Perception / task design；
- 同一冻结 PPO 通过旧 Hybrid runtime 与 `HybridPPOEnv` external-action path
  运行时，7/7 场景的 success、elapsed time、path length、collision count 在
  `1e-6` tolerance 内一致。

M5 至此标记为 **COMPLETE**。当前项目已经验证同一 World/Gym/Experiment 基础
设施可稳定运行 BT、PPO 与 Hybrid BT-RL，并支持 60 Hz BT supervision、PPO
preemption 和 external-action ownership。Target memory、PPO Search、richer
perception 与其他 M6 研究能力仍未实现。

输出位置：

```text
experiments/comparisons/m53/m53_final.csv
experiments/comparisons/m53/m53_final_summary.json
experiments/comparisons/m53/runs/<scenario>/<controller>/
experiments/comparisons/m53/adapter_equivalence_runs/<scenario>/
experiments/comparisons/m53/human_demos/<scenario>/<controller>/
```

将独立学习策略嵌入 BT 并进行局部再训练，并不能保证 Hybrid Controller 获得更好的泛化性能；高层行为切换、感知条件和控制权恢复本身可能成为新的性能瓶颈。

## Behavior Tree Definition

手工 baseline 与 Hybrid topology 分别定义在 `bt_configs/default.json` 和
`bt_configs/hybrid_ppo.json`。`bt-lab/v1` 支持：

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
  → autonomy_lab/scenarios/config.py
  → autonomy_lab/core/environment.py / agent.py
  → autonomy_lab/perception/pygame_perception.py
  → autonomy_lab/core/observation.py
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
