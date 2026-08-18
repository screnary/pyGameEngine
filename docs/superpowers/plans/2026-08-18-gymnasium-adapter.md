# Milestone 3 Gymnasium Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留唯一 `Environment` 仿真内核的前提下，增加固定时间步、无窗口运行和轻量 Gymnasium Adapter。

**Architecture:** Manual、BT 与 Gym 都产生同一种 `{"turn", "throttle"}` Command，并调用 `Environment.step(command, dt)`。显示职责迁移到只读的 `PygameRenderer`；`AgentGymEnv` 仅负责 Gym action/observation/reward/lifecycle 适配。

**Tech Stack:** Python 3.11、pygame、numpy、py_trees、gymnasium、unittest

**Spec:** 本会话中已批准的 M3 设计及四项补充约束；用户要求不再创建更大的独立设计文档。

## Global Constraints

- `Environment` 是唯一 World，不复制运动、碰撞、边界、感知或终止逻辑。
- Simulation Core 可使用 `pygame.Vector2/Rect`，但不使用 display/event/draw/Surface/font/image/Clock。
- `simulation_dt = 1 / 60`，不得由 FPS 或 `clock.tick()` 返回值决定。
- Observation 为固定 13 维 `float32`，只来自 Agent state、PerceptionSnapshot 和自身边界余量。
- Renderer 只读取 World；`render_mode=None` 不创建 Renderer 或窗口。
- 不实现 PPO、SB3、Gym 注册、新 BT 行为或额外 RL 架构。
- 新代码继续使用中文教学型 Docstring 和关键逻辑注释。

---

### Task 1: 唯一 World 步进入口与只读 Renderer

**Files:**
- Modify: `autonomy_lab/environment.py`
- Modify: `autonomy_lab/agent.py`
- Modify: `autonomy_lab/perception.py`
- Create: `autonomy_lab/renderer.py`
- Modify: `main.py`
- Test: `tests/test_environment_step.py`
- Modify: `tests/test_main_lifecycle.py`

**Interfaces:**
- Produces: `Environment.reset(seed: int | None = None) -> None`
- Produces: `Environment.step(command: dict[str, float], dt: float) -> None`
- Produces: `PygameRenderer.render(environment, controller=None, controller_name="manual") -> None`

- [ ] 写失败测试：同一 Command 经 `Environment.step()` 更新运动、碰撞状态、时间、到达状态和 action 后感知，并验证非法 `dt`/缺失 Command。
- [ ] 运行 `conda run -n pygame_lab python -m unittest tests.test_environment_step -v`，确认因接口尚不存在而失败。
- [ ] 实现 `reset/step`，严格采用 command → motion → collision/boundary → time/termination → perception 顺序；让 BT 复用 `environment.perception`。
- [ ] 把 Agent/Environment 的绘图和图片状态迁移到 `PygameRenderer`，把键盘输入在 `main.py` 转成现有 dict Command。
- [ ] 固定主循环物理步长为 `1 / 60`，保留 `clock.tick(60)` 仅用于显示节奏。
- [ ] 运行 Environment 与主循环生命周期定向测试并修复回归。

### Task 2: Gymnasium Adapter 与 13 维感知 Observation

**Files:**
- Create: `autonomy_lab/gym_env.py`
- Modify: `autonomy_lab/__init__.py`
- Modify: `environment.yml`
- Test: `tests/test_gym_env.py`

**Interfaces:**
- Produces: `AgentGymEnv(gymnasium.Env)`，支持 `reset(seed=None, options=None)`、`step(action)`、`render()`、`close()`。
- Consumes: `Environment.step({"turn": float, "throttle": float}, simulation_dt)`。

- [ ] 在 `pygame_lab` 中仅安装 `gymnasium`，并把依赖加入 `environment.yml`。
- [ ] 写失败测试：Box action 映射、13 维 `float32` Observation、不泄露不可见 Target、action 后 Observation、terminated/truncated 和非法 render mode。
- [ ] 运行 `conda run -n pygame_lab python -m unittest tests.test_gym_env -v`，确认因 Adapter 尚不存在而失败。
- [ ] 实现最小 Adapter：`reset()` 调用 `super().reset(seed=seed)`；Observation 使用中性缺失值；每步固定 `1 / 60`。
- [ ] 实现集中 reward：每步 `-0.001`、新碰撞事件 `-0.05`、目标到达 `+1.0`；持续碰撞不重复扣分。
- [ ] 实现精简 info、自然终止和时间截断，并运行测试至通过。

### Task 3: Renderer 独立性与 Recorder 兼容

**Files:**
- Modify: `autonomy_lab/gym_env.py`
- Test: `tests/test_gym_env.py`
- Create: `gym_demo.py`

**Interfaces:**
- Produces: `render_mode=None` 无 Renderer；`render_mode="human"` 懒创建 `PygameRenderer`。
- Produces: `record_experiments=True` 时复用 `ExperimentRecorder`，BT 指标为 `None`。

- [ ] 写失败测试：相同 scenario/seed/actions 在 headless 与 SDL dummy human 模式产生相同 position、heading、speed、collision、termination 和 Observation。
- [ ] 写失败测试：Gym Recorder 使用现有 path/collision/trajectory 定义，并且碰撞奖励事件语义与 collision_count 一致。
- [ ] 实现 human 渲染与可选 Recorder 生命周期；`render()` 不调用 `Environment.step()`。
- [ ] 增加只运行随机 Action 的 `gym_demo.py` headless smoke 入口。
- [ ] 运行 Gym 定向测试至通过。

### Task 4: API 检查、BT 回归与文档

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_GUIDE.md`

**Interfaces:**
- Verifies: Gymnasium 1.x reset/step API、三个场景、BT/Pygame 旧入口和 M2 日志兼容。

- [ ] 运行 Gymnasium `check_env`、`observation_space.contains(obs)` 和 `action_space.contains(action)`。
- [ ] 运行三个场景的 headless 随机 Action smoke test，以及 SDL dummy human render smoke test。
- [ ] 运行 `tests.test_environment_step`、`tests.test_gym_env`、`tests.test_main_lifecycle`、`tests.test_perception_bt`、`tests.test_experiment`。
- [ ] 更新 README 的 headless/human 使用方式和 13 维 Observation 约定。
- [ ] 更新 `PROJECT_GUIDE.md` 的 Current Milestone，明确 M4 尚未实现 PPO/SB3。
- [ ] 运行 `compileall` 与 `git diff --check`，确认无阻止运行的问题后停止。
