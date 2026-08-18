# 中文教学型代码注释实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完全不改变运行逻辑的前提下，为项目 Python 代码补充中文教学型 Docstring 和逻辑块注释。

**Architecture:** 注释按运行数据流分三组补充：Pygame/环境基础层、感知与 BT 决策层、可视化与实验记录层。每组修改后运行对应现有测试，最终验证三个场景和全部测试。

**Tech Stack:** Python 3.11、pygame 2.6.1、py_trees 2.5.0、`unittest`。

## Global Constraints

- 只修改注释、Docstring、README 和 PROJECT_GUIDE；不得修改表达式、分支、签名或配置值。
- Docstring 使用中文，代码标识符和库名保持英文。
- 重点解释输入来源、状态所有权、单位、处理原因、生命周期和输出去向。
- 不逐行翻译显而易见的赋值、导入和直接绘图调用。
- JSON 不支持注释，`bt_configs/default.json` 保持不变。
- 使用现有 `pygame_lab` Conda 环境，不安装依赖。
- 当前为包含用户改动的脏工作区，不提交实现文件。

---

### Task 1: Pygame 主循环、Agent、Environment 和场景配置

**Files:**
- Modify: `main.py`
- Modify: `autonomy_lab/agent.py`
- Modify: `autonomy_lab/assets.py`
- Modify: `autonomy_lab/environment.py`
- Modify: `autonomy_lab/scene_config.py`

**Interfaces:**
- 不产生接口变化。
- 解释 `main → controller/environment → recorder → draw` 帧数据流。

- [x] **Step 1: 将模块、类和函数 Docstring 扩展为中文**

为命令行解析、主循环、Agent 状态、手动/自动命令统一入口、环境 reset、
碰撞、绘制、素材 fallback 和场景深拷贝说明职责、输入与副作用。

- [x] **Step 2: 按逻辑块解释关键变量和执行顺序**

在代码旁明确：`dt` 单位与上限、`heading` 弧度、`turn/throttle` 范围、
位移计算、分轴碰撞、圆-矩形判定、FOV 仅可视化、Episode 成功后冻结，
以及 reset 前先结束旧 Episode 的原因。

- [x] **Step 3: 运行生命周期和绘制测试**

```powershell
conda run -n pygame_lab python -m unittest tests.test_main_lifecycle tests.test_bt_visualizer -v
```

要求现有生命周期和 Environment 绘制结果全部通过。

### Task 2: Context、Perception、Behavior、Registry、Loader 和 Controller

**Files:**
- Modify: `autonomy_lab/behavior_context.py`
- Modify: `autonomy_lab/perception.py`
- Modify: `autonomy_lab/behaviors.py`
- Modify: `autonomy_lab/behavior_registry.py`
- Modify: `autonomy_lab/bt_loader.py`
- Modify: `autonomy_lab/behavior_tree.py`

**Interfaces:**
- 不产生接口变化。
- 解释 `Environment → PerceptionSnapshot → BT tick → command → Environment`。

- [x] **Step 1: 详细解释 BehaviorBuildContext 和构建期数据流**

逐字段说明 `perception`、`command`、`behavior_config`、`nodes_by_name` 的
创建者、共享者和用途；解释 Registry 只选类、Loader 只构建结构、Behavior
自己解释 params。

- [x] **Step 2: 详细解释感知算法**

说明角度归一化、相对 bearing、Ground Truth/perceived 差异、Range/FOV/LOS、
可见障碍物最近点、射线安全空间、连续开放射线分组、探索缺口和目标缺口
排序，以及固定世界坐标入口。

- [x] **Step 3: 详细解释 Behavior 生命周期**

说明 Python `__init__()` 与 py_trees `initialise()`、`update()`、
`terminate()` 的区别；逐个 Condition/Action 描述读取的 snapshot 字段、
返回状态、command 副作用、参数回退和节点引用。

- [x] **Step 4: 详细解释 Controller tick 和抢占**

说明节点分类缓存、缺口承诺时的紧急距离、每帧清空 command、向定时 Action
注入 `dt`、SnapshotVisitor、active behavior 和 reset。

- [x] **Step 5: 运行 BT 与感知测试**

```powershell
conda run -n pygame_lab python -m unittest tests.test_bt_loader tests.test_perception tests.test_perception_bt -v
```

要求 JSON、感知、Behavior 和 Controller 结果不变。

### Task 3: Visualizer、Experiment、测试说明和最终验证

**Files:**
- Modify: `autonomy_lab/bt_visualizer.py`
- Modify: `autonomy_lab/experiment.py`
- Modify: `tests/__init__.py`
- Modify: `tests/test_bt_loader.py`
- Modify: `tests/test_bt_visualizer.py`
- Modify: `tests/test_experiment.py`
- Modify: `tests/test_main_lifecycle.py`
- Modify: `tests/test_perception.py`
- Modify: `tests/test_perception_bt.py`
- Modify: `README.md`
- Modify: `PROJECT_GUIDE.md`

**Interfaces:**
- 不产生接口变化。
- 解释 Runtime Tree 到面板，以及一帧数据到 JSON/CSV 的流向。

- [x] **Step 1: 详细解释 Visualizer**

说明结构签名缓存、递归提取、父子连接、叶节点纵坐标、父节点居中、状态与
visited 的区别、文字换行和截断。

- [x] **Step 2: 详细解释 ExperimentRecorder**

说明 Episode 状态、仿真时间与墙钟时间、路径长度、碰撞接触边沿计数、BT
转移、轨迹采样、Episode ID、JSON 明细、CSV 摘要和旧表头迁移。

- [x] **Step 3: 将测试模块说明改为中文**

仅解释 fixture、临时 JSON、dummy SDL 帧和 CSV 迁移场景，不为直接断言
逐行加注释。

- [x] **Step 4: 更新阅读指南**

在 README 增加建议阅读顺序和一条完整数据流；在 PROJECT_GUIDE 记录本轮
教学型注释完成状态，不增加新里程碑能力。

- [x] **Step 5: 运行完整测试和三场景验证**

```powershell
conda run -n pygame_lab python -m unittest discover -s tests -v
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
git diff --check
```

另外在 SDL dummy 下对三个场景分别构建默认 BT、tick、绘制并 reset。完成后
立即停止，不修改任何算法。
