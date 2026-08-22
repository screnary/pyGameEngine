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
autonomy_lab/
├── core/
│   ├── agent.py
│   ├── environment.py
│   └── observation.py
├── perception/
│   ├── semantic_perception.py
│   └── pygame_perception.py
├── scenarios/
│   ├── config.py
│   └── scenario_distribution.py
├── bt/
│   └── parameters.py
├── rendering/
├── gym/
│   ├── env.py
│   └── hybrid_env.py
└── experiment/
    ├── recorder.py
    └── runners.py
scripts/
├── demo/
├── training/
└── evaluation/
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

R0.13 Hazard Action Commitment Fix completed:

* only the Research BT `Hazard Avoidance` Sequence changed from `memory=false` to
  `memory=true`; the Loader already supported this field, so no Loader code changed
* `HazardRisk` remains the unchanged single-threshold start Condition; after it succeeds,
  the existing `AvoidHazard` RUNNING child now keeps control until the maneuver completes
* the top-level Priority Selector remains reactive: `Boundary Recovery` immediately
  preempts a committed Avoid maneuver and terminates `AvoidHazard` as `INVALID`
* after normal Sequence success, the next visit starts again at `HazardRisk`, so a
  persistent or newly encountered Hazard can start another maneuver
* the diagnosed `dynamic_hazard`, seed 1001, 45 px episode changes from 26 to 6 branch
  switches and 13 to 2 HazardRisk activations; success remains true, collisions remain
  zero, and path length changes from 1096.15 to 1090.83 px
* R0.9/R0.10 micro competence remains MoveToGoal 5/5, Stop 2/2, AvoidHazard 6/6, and
  SafeBoundaryRecovery 7/7 with zero collision events

The full R0.11 and R0.12 evaluators were rerun without changing their threshold grids,
contexts, families, or seeds. Historical pre-fix artifacts are retained with the
`_pre_r013` suffix; the canonical result files now use committed execution semantics.

R0.13 is **COMPLETE**. The execution bug is fixed, H2 remains supported, and H3 remains
supported under the existing paired-static-context criterion. Fixed Actions and the
Research BT execution semantics are freeze-ready, so the stated M6.1 preconditions are
met; no M6 code, RL training, threshold tuning, hysteresis, Action, Perception, or
Scenario change was introduced.

### Previous Milestone Record — R0.12

R0.12 Context-Dependent Threshold Necessity, rerun after R0.13:

* paired contexts still differ only in dynamic Hazard speed: 36 px/s Low Risk versus
  180 px/s High Dynamic Risk, with the unchanged 20/30/40/45/60/75/90 px grid and
  seeds 1001-1050
* the analysis-only score remains `success - collision_episode - 1.0 * exposure`;
  Low Risk now prefers 60 px (90% success, 0% collision, 0.4866 exposure), while High
  Dynamic Risk prefers 40 px (98% success, 32% collision, 0.5307 exposure)
* the two cross-threshold score advantages are 0.1313 and 0.0935, both above the fixed
  0.02 gate; no tested static threshold dominates both paired contexts
* Low→High→Low was executed after the static gate passed, but its phase safety metrics
  did not independently reproduce a clean 60/40 reversal; unequal later-phase episode
  counts also reflect earlier successes, so that diagnostic is not treated as a second
  proof of context dependence

R0.12 remains **Case A — Context Dependence Supported** under its predefined static
crossing rule. H3 is supported with the explicit within-episode limitation above.

### Previous Milestone Record — R0.11

R0.11 Switching Bottleneck Attribution, rerun after R0.13:

* the unchanged sweep covers 45/63/76.5/90/103.5/117/135 px over five Research
  families and seeds 1001-1050, producing 1750 episodes
* overall success falls from 82.0% at 45 px to 18.4% at 135 px, timeout rises from
  18.0% to 81.6%, and AvoidHazard occupancy rises from 48.4% to 72.2%
* compared with the pre-fix run, mean branch switches across thresholds fall from
  37.5-92.2 to 12.2-25.6, HazardRisk activations fall from 18.3-41.7 to 3.1-5.0,
  and longest continuous avoidance rises from 0.85-1.42 s to 1.38-1.53 s
* the evaluator still returns **Case A — Switching bottleneck supported**: parameter
  sensitivity remains material and the best fixed threshold reaches 82% success
* the new run also meets the evaluator's aggregate safety-efficiency correlation rule,
  while family preferences still do not establish context dependence in this sweep

R0.11 remains **COMPLETE** and H2 Condition Sensitivity remains supported. The lower
post-fix success rates are retained rather than tuned away: completing each maneuver
reduces chattering but increases reactive Avoid occupancy and timeout in some episodes.

### Previous Milestone Record — R0.10

R0.10 Fixed Action Safety Stabilization assessed:

* Research `SafeBoundaryRecovery` now suppresses throttle while the selected safe
  bearing differs from Agent heading by more than 35 degrees, then resumes its fixed
  recovery throttle after alignment
* Research `AvoidHazard` combines independent Hazard sector clearance and directional
  Boundary clearance when selecting a local escape bearing; it locks one safe side,
  turns before driving, and rechecks nearby candidate sectors each tick
* both Actions consume only `SemanticPerception` and shared BT context; Hazard sectors
  still exclude Boundary, and no World obstacle list or privileged geometry is read
* fixed parameters are centralized in scene behavior config: 35-degree alignment
  gates and 45-degree turn gains; legacy `ObstacleThreat` avoidance and 12-sector
  boundary behavior retain their previous command path
* unchanged R0.9 micro-scenarios now produce `MoveToGoal 5/5`, `Stop 2/2`,
  `AvoidHazard 6/6`, and `SafeBoundaryRecovery 7/7`, with zero collision events and
  no steering-sign oscillation in the two safety Actions
* the same five families and seeds 1001-1050 reduce mean collision events to
  static 0.02, dense 0.32, dynamic 0.48, noisy 0.00, and context shift 0.48
* however end-to-end success changes from 191/250 to 155/250: static 86%, dense 18%,
  dynamic 70%, noisy 68%, context shift 68%; the conservative reactive avoidance
  dominates dense episodes and causes timeout rather than collision

R0.10 is **NOT COMPLETE** under its full acceptance criteria. Local safety Action
competence is stabilized, but the complete fixed Action substrate is **not freeze-ready**
because end-to-end navigation regressed substantially. M6 and Condition-RL remain
blocked and were not started.

### Previous Milestone Record — R0.9

R0.9 established the unchanged micro and 250-episode competence baseline:
`MoveToGoal 5/5`, `Stop 2/2`, `AvoidHazard 3/6`, `SafeBoundaryRecovery 0/7`, and
overall Research BT success 191/250. Its independent evaluation entry remains
`scripts.evaluation.eval_action_competence`.

### Previous Milestone Record — R0.8

R0.8 calibrated Research Goal/Hazard sensing to 850/300 px over five families and
seeds 1001-1050. Goal remains a stable long-range task signal, Hazard sensing remains
local, 16 sectors remain the handcrafted steering representation, and legacy M4/M5
perception and 13-D PPO Observation remain frozen.

### Previous Milestone Record — R0.7

R0.7 Safety-Gym-aligned Finite-Range Sensing completed:

* one configurable `AgentPerception` implementation now selects either the
  frozen `legacy` profile or the Research `research` profile
* centralized Research defaults use equal `goal_range=hazard_range=700 px`
  and 16 bins for both Goal direction and Hazard sectors
* Research Goal sensing is 360-degree and range-only; it ignores FOV, LOS, and
  `target_information_mode`, and exposes no distance/bearing/sector outside range
* Research Hazard sensing is 360-degree and footprint-aware; out-of-range objects
  do not enter nearest/visible data, while empty lidar sectors report max range
* Research Hazard sectors intentionally exclude World Boundary; the existing
  optional `BoundaryPerception` remains the independent boundary semantic
* seeded Hazard noise is applied after the finite-range gate and cannot leak an
  out-of-range object or modify collision geometry
* `condition_research.json` consumes the new finite-range semantics without
  topology, Action, or threshold changes
* legacy M4/M5 FOV/LOS behavior, 12-sector safety data, 13-D Observation values,
  Pure PPO, Frozen Hybrid, and HybridPPOEnv remain unchanged
* the Research human renderer now shows sensing radii, Hazard sector rays, Goal
  sensing state, and nearest Hazard clearance without changing World state

R0.7 is **COMPLETE**. Safety-Gymnasium dependencies/adapters, normalized lidar
observations, Condition-RL, Search/memory changes, new rewards, and M6 were not started.

### Previous Milestone Record — Pre-M6 Responsibility Grouping

Pre-M6 Responsibility Grouping completed:

* World, Agent dynamics, and the frozen 13-D encoder live under `core/`
* simulator-neutral semantics and the current Pygame provider live under
  `perception/`; `SemanticPerception` is not research-only infrastructure
* fixed scene presets and R0.4 distributions live under `scenarios/`, with the
  former `scene_config.py` renamed to `config.py`
* the generic `ParameterSpec/ParameterStore` remains under `bt/parameters.py`
* runnable modules are grouped under `scripts/demo`, `scripts/training`, and
  `scripts/evaluation` without changing their algorithms or default output roots
* flat legacy import paths were removed rather than retained through a
  compatibility layer

M4/M5/R0 semantics remain frozen and M6 has not started.

### Previous Milestone Record — R0.6

R0.6 Generic Parameter Interface completed:

* `ParameterSpec` stores a continuous parameter's name, current value, default,
  and inclusive minimum/maximum bounds as plain Python data
* `ParameterStore` provides one RL-independent `get/set/reset/reset_all/bounds/spec`
  interface and rejects unknown, non-numeric, non-finite, or out-of-range updates
* `ConditionParameters` now delegates to the generic Store while preserving the
  R0.3/R0.5 properties, constructor, defaults, ranges, and batch compatibility API
* Hazard, Boundary, and Goal Conditions read their named Store value every tick;
  they do not cache build-time thresholds or know how parameter updates are produced
* the Store treats a future Action parameter exactly like a Condition parameter;
  no Action behavior or learnable-node hierarchy was added
* the fixed Research BT produces the same branch decisions for the same semantic
  observation and theta values, while legacy M4/M5 paths remain unchanged

R0.1 through R0.6 are **COMPLETE**. Parameter optimizers, PPO/CMA-ES updates,
Action parameter migration, smoothing/hysteresis, and all M6 features were not started.

### Previous Milestone Record — R0.5

R0.5 Research Interface Stabilization completed:

* `SemanticPerceptionProvider` defines the minimal simulator adapter contract:
  one `observe()` per control tick followed by read-only access to `snapshot`
* Pygame `AgentPerception` satisfies that contract while retaining `update()`
  and legacy properties for frozen M4/M5 compatibility
* semantic dataclasses contain only plain immutable values; Agent radius is a
  scalar core field, while Boundary and Pygame-derived gaps expose explicit
  availability metadata
* all Conditions and Actions used by `condition_research.json` execute from
  `SemanticPerception + ConditionParameters` without a simulator World reference
* `AgentCommand` is the shared typed control shape used by Research, BT, Manual,
  Gym, and `Environment.step()`; no second research action representation exists
* `ConditionParameters` now exposes copy-safe get/set/reset/bounds methods while
  remaining a plain RL-independent mutable Store
* all five R0.4 families traverse the same Scenario → Environment → Provider →
  SemanticPerception → Research BT pipeline without family-specific method code
* `scripts.demo.demo_scenario_distribution` accepts family, seed, and three manual
  Condition thresholds as the minimal human research smoke entry
* the frozen legacy 13-D Observation, PPO checkpoints, Hybrid runtime, BT configs,
  and historical results remain separate and compatible

R0.5 was **COMPLETE**. At that stage the generic `ParameterSpec/ParameterStore`,
parameter optimizers, Condition-RL, and all M6 features had not started.

### Previous Milestone Record — R0.4

R0.4 Scenario Distribution completed:

* `ScenarioDistribution(family).sample(seed)` returns a normal scene dict that
  the existing `Environment` consumes without a parallel simulation framework
* five research families cover static random geometry, denser static risk,
  boundary-reflecting dynamic Hazard, seeded Hazard-range noise, and a fixed
  three-phase within-episode context shift
* R0.1 narrow-passage and boundary-obstacle scenes remain unchanged and are
  also exposed as deterministic research grouping aliases
* dynamic and static Hazards share the existing obstacle Rect list, so current
  collision, LOS, sector/gap perception, and rendering paths remain unified
* perception noise is applied only when constructing Semantic Hazard clearance;
  collision geometry, sector truth, Goal sensing, and legacy fixed scenes stay clean
* family, seed, Hazard count, dynamic state, noise level, and current phase are
  available through lightweight scene/World diagnostics
* sampled geometry, dynamic replay, noise sequence, and schedule are controlled
  by local seed state and do not use global random state
* the Parameterized Research BT runs all generated families with its fixed
  handcrafted Actions and manually configured thresholds
* `scripts.demo.demo_scenario_distribution` provides a 60 Hz human visualization
  with family/seed selection, same-seed reset, and context diagnostics

R0.4 is **COMPLETE**. At that stage Condition-RL, delta-theta policies,
curriculum systems, Safety-Gym integration, procedural mazes, and R0.5/M6 had
not started.

### Previous Milestone Record — R0.3

R0.3 Parameterized Research BT completed:

* `bt_configs/condition_research.json` defines a fixed handcrafted tree with
  Boundary Recovery, Hazard Avoidance, Goal Reached/Stop, and Move To Goal
* a shared mutable `ConditionParameters` holds finite non-negative Hazard,
  Boundary, and Goal thresholds with stable defaults of 90/40/30 px
* each parameterized Condition reads the current Store value on every tick, so
  changing a threshold affects the next decision without rebuilding the tree
* Research Conditions consume only nested `SemanticPerception.goal`, `.hazard`,
  and `.boundary` values; World Ground Truth is not read directly
* runtime feedback reports the observed distance/clearance and current theta
* existing R0.1 safe steering and handcrafted Actions are reused unchanged
* `default.json`, `hybrid_ppo.json`, the 13-D Observation, checkpoints, and
  frozen PPO/Hybrid paths remain compatible

R0.3 is **COMPLETE**. At that stage Condition-RL, learned delta-theta output,
smoothing, hysteresis, switching penalties, dynamic topology, and R0.4 had not
started.

### Previous Milestone Record — R0.2

R0.2 Semantic Perception Refactor completed:

* `SemanticPerception` is now the single `AgentPerception` output and groups
  immutable `AgentState`, `GoalPerception`, `HazardPerception`, and
  `BoundaryPerception` values
* semantic values contain plain Python scalars/tuples only; pygame geometry and
  World objects remain local to the Pygame perception computation
* Goal sensing remains range/FOV/raw-obstacle optical LOS, while Hazard owns
  footprint-aware local clearance, 12 sector ranges, and traversable gaps
* read-only legacy properties map existing Target/Obstacle names to the same
  nested values without recomputation or a parallel perception implementation
* the legacy Observation Builder now reads semantic Agent/Goal/Hazard/Boundary
  fields and remains numerically identical for the three frozen M4 scenarios
* current BT, R0.1 regressions, Pure PPO, Frozen Hybrid, and HybridPPOEnv remain
  operational without PPO training or checkpoint changes

R0.2 is **COMPLETE**. At that stage Parameterized Conditions, Condition-RL,
Safety-Gym, Observation expansion, Search redesign, and R0.3 had not started.

### Previous Milestone Record — R0.1

R0.1 Perception & Navigation Bug Fix completed:

* Goal sensing now uses range/FOV and raw-obstacle optical LOS only; Agent
  footprint inflation is reserved for traversability and free-space sensing
* `AgentPerception` exposes 12 full-circle, footprint-aware sector clearances
  to BT/navigation without changing the frozen legacy 13-D PPO observation
* `choose_safe_steering()` scores desired direction together with obstacle and
  boundary clearance; `SafeBoundaryRecovery` locks a detour side while the
  inward route is blocked, and `AvoidObstacle` compares both lateral sides
* the default BT now places `Boundary Recovery` above `Obstacle Avoidance`;
  existing target pursuit, gap navigation, exploration, and Search branches
  retain their prior order and behavior
* fixed test-only `r01_narrow_passage` and `r01_boundary_obstacle` scenarios
  protect optical sensing/traversability separation and safety-action recovery
* frozen M4/M5 checkpoints were evaluated without training; Pure PPO results
  remained unchanged and the Frozen Hybrid/external Adapter remained equivalent

R0.1 is **COMPLETE**. It does not introduce semantic perception, target memory,
new PPO observations, planners, Condition-RL, or R0.2 work.

### Previous Milestone Record — M5.3

Milestone 5.3 Hybrid BT-RL Final Evaluation & Milestone Closure completed:

* `scripts/evaluation/eval_m53_final.py` reuses the common Episode runners to evaluate
  BT, Pure PPO, Frozen Hybrid, and the preserved 200,704-step Hybrid-trained
  PPO on the seven fixed M5.1 scenarios; it does not call PPO training
* seed 5001 remains a fixed-pipeline identifier. The statistical unit is one
  fixed scenario, not repeated stochastic trials or randomized geometry
* BT and Pure PPO passed 2/2 seen, 4/4 mild-unseen, and 1/1 hard scenes;
  Frozen Hybrid and Hybrid-trained PPO passed 2/2, 3/4, and 0/1 respectively
* on all mild-unseen episodes, BT averaged 6.050 s / 781.3 px / 0 collisions,
  Pure PPO 3.613 s / 788.9 px / 1.00, Frozen Hybrid 7.625 s / 669.2 px /
  1.25, and Hybrid-trained PPO 7.617 s / 668.7 px / 1.50
* Reverse Detour remained the Hybrid failure: Frozen used one Boundary
  activation/preemption, 17 PPO decisions, 0.084 active ratio, five collisions,
  and timed out; trained used one activation/preemption, 18 decisions, 0.087
  ratio, six collisions, no re-entry, and also timed out
* on the hard scene both Hybrid variants activated Search once, used only three
  PPO decisions with 0.014 active ratio, and timed out. This remains a Search /
  Perception / task-design limitation rather than a PPONavigate training result
* Hybrid-context training did not improve mild-unseen success (3/4 versus 3/4),
  so M5 records the explicit negative result rather than continuing training
* a seven-scenario Adapter regression matched old Frozen Hybrid execution and
  `HybridPPOEnv` external-action execution on success, elapsed time, path length,
  and collision count within `1e-6` absolute tolerance
* independent human demos completed for `ppo_simple_obstacle` and
  `m43_reverse_detour` across all four Controllers without modifying batch data

M5.0 through M5.3 are now **COMPLETE**. M5 confirms that BT, PPO, Frozen PPO as
a BT Action, 60 Hz BT supervision/preemption, and Hybrid external-action
ownership can share the same Environment, Gym, Renderer, and Experiment
infrastructure. It does not claim that the current Hybrid improves navigation,
nor does it solve Target memory, Search, richer Perception, or hard navigation.
No M6 work was started.

M5.3 structured outputs and commands:

```text
experiments/comparisons/m53/m53_final.csv
experiments/comparisons/m53/m53_final_summary.json
experiments/comparisons/m53/runs/<scenario>/<controller>/
experiments/comparisons/m53/adapter_equivalence_runs/<scenario>/
experiments/comparisons/m53/human_demos/<scenario>/<controller>/

conda run -n pygame_lab python -m scripts.evaluation.eval_m53_final
conda run -n pygame_lab python -m scripts.evaluation.eval_m53_final --human-demo
```

Milestone 5.2 Hybrid PPO Lab Adapter framework validation completed:

* `autonomy_lab/gym/hybrid_env.py` exposes PPO decisions only while the real
  `PPONavigate` node owns control; it does not introduce another scheduler or
  duplicate the Hybrid tree
* World simulation and BT supervision remain 60 Hz; one PPO decision controls
  at most six actual World steps, while Boundary Recovery and Search retain
  their own BT Commands and can take ownership between PPO decisions
* Gym returns at the next PPO decision point, Episode end, or horizon; reward,
  simulation time, trajectory, collision events, and Recorder metrics aggregate
  actual internal World steps rather than PPO decision steps
* the Adapter uses the unchanged 13-D perception Observation and existing
  Reward semantics; privileged ground-truth target distance remains confined to
  reward shaping
* external-action mode is opt-in. Existing frozen Hybrid execution remains the
  default, and a seven-scenario adapter equivalence smoke check matched its
  success, elapsed time, path length, and collisions
* `scripts/training/train_hybrid_ppo.py` now defaults to a 2,048-decision smoke run.
  Longer training requires an explicit `--target-timesteps` value because the
  current project focus is Lab framework correctness, not PPO optimization
* a completed 200,704-step checkpoint is preserved as diagnostic evidence. It
  did not improve the two Hybrid-relevant scenario pass count (both Frozen and
  trained Hybrid passed 1/2); the requested 500k continuation was stopped and
  no incomplete checkpoint was saved

M5.2 did not change the Hybrid topology, Observation, Action, Reward, World
dynamics, Perception, Renderer, collision semantics, termination, or public
ExperimentRecorder metric definitions. M5.3 subsequently performed the frozen
final evaluation above.

M5.2 framework commands:

```text
conda run -n pygame_lab python -m scripts.training.train_hybrid_ppo --model-path models/ppo_m52_smoke.zip --log-label m52_hybrid_smoke
conda run -n pygame_lab python -m scripts.evaluation.eval_m52_hybrid --model-path models/ppo_m52_hybrid_trained_200k.zip --checkpoint-label 200k
```

Milestone 5.1 Hybrid Evaluation & Behavior Analysis completed:

* `scripts/evaluation/eval_m51_hybrid.py` evaluates the frozen default BT, deterministic
  `models/ppo_m41b_control10hz.zip`, and `hybrid_ppo.json` on the same seven
  fixed scenarios and verifies matching World initial states
* World simulation and BT supervision remain 60 Hz; Pure PPO and the active
  Hybrid PPO node infer at approximately 10 Hz; common metrics remain based on
  actual World simulation time and steps rather than Controller decisions
* read-only instrumentation records BT ticks/transitions, PPO decisions and
  active time/ratio, Boundary Recovery and Search activations, and PPO
  preemptions without changing topology, commands, or metric semantics
* seed 5001 standardizes the fixed-layout pipeline; the statistical unit is
  the scenario, not repeated stochastic trials
* BT and PPO passed 2/2 seen, 4/4 mild unseen, and 1/1 hard scenarios; Hybrid
  passed 2/2 seen, 3/4 mild unseen, and 0/1 hard scenarios
* on mild unseen scenes, BT averaged 6.050 s / 781.3 px / 0 collisions, PPO
  3.613 s / 788.9 px / 1.00 collisions, and Hybrid 7.625 s / 669.2 px / 1.25
  collisions across all episodes, including the Hybrid timeout
* Hybrid matched PPO exactly while PPO Navigation remained active. In Reverse
  Detour it selected the PPO upper route, then one Boundary Recovery activation
  preempted PPO and the episode timed out; PPO active ratio was 0.084
* in the hard narrow-gap scene, one Search activation followed target loss;
  PPO active ratio fell to 0.014 and Hybrid timed out. These results show an
  unresolved high-level handoff/recovery interaction, not a new capability
* Human demos cover the simple obstacle, Reverse Detour Boundary preemption,
  and hard-scene Search activation; they write only to the separate human-demo
  directory and do not enter the batch summary

M5.1 did not retrain PPO or modify BT topology/parameters, Reward, Observation,
Action, dynamics, Perception, scenarios, World stepping, Renderer, termination,
collision semantics, or Recorder schema. M4.2/M4.3 outputs were not overwritten,
M5.2 subsequently added the opt-in external-policy Lab Adapter described above.

M5.1 structured outputs:

```text
experiments/comparisons/m51/m51_bt_ppo_hybrid.csv
experiments/comparisons/m51/m51_bt_ppo_hybrid_summary.json
experiments/comparisons/m51/runs/<scenario>/<controller>/
experiments/comparisons/m51/human_demos/<scenario>/<controller>/
```

Reproduction commands:

```text
conda run -n pygame_lab python -m scripts.evaluation.eval_m51_hybrid
conda run -n pygame_lab python -m scripts.evaluation.eval_m51_hybrid --human-demo
```

Milestone 5.0 Frozen PPO Action Integration completed:

* `bt_configs/hybrid_ppo.json` defines a separate Hybrid tree with priority
  Boundary Recovery, visible-target Learned Navigation, and Search fallback;
  the original `default.json` baseline is unchanged
* `PPONavigate` is registered through the existing Registry/Loader and loads
  `models/ppo_m41b_control10hz.zip` once per Controller instance without
  calling `learn()`, saving a model, or stepping the World
* Gym and Hybrid inference now share
  `autonomy_lab.observation.build_navigation_observation`, preserving the
  frozen 13-D field order, normalization, neutral values, and `float32` dtype
* World simulation and BT supervision remain 60 Hz; the PPO node uses
  simulation `dt` to call deterministic `predict()` at approximately 10 Hz
  and reapplies the cached Command on the intervening ticks
* the new Boundary Safety branch can preempt PPO at 60 Hz; PPO termination
  clears its own stale Command without overwriting a new higher-priority
  Action command, and target invisibility returns control to Search Target
* `main.py --bt hybrid_ppo` reuses the existing Controller, Visualizer, World,
  Renderer, and ExperimentRecorder, with controller label `hybrid_bt_ppo`
* on fixed seeds 6001-6003, Hybrid and Pure PPO both reached 3/3 targets with
  zero collisions in `rl_sanity` and `ppo_simple_obstacle`; their elapsed time
  and path length matched exactly at 1.983 s / 436.3 px and
  3.600 s / 792.0 px respectively

M5.0 did not retrain PPO or change Reward, Observation schema, Action, Agent
dynamics, Perception, World stepping, Renderer, the default BT topology, or
Recorder metric semantics. M5.1 later evaluated this frozen implementation;
M5.2 subsequently added the opt-in external-policy Lab Adapter described above.

Milestone 4.4 project structure cleanup completed:

* root keeps `main.py` as the primary interactive launcher; Gym/PPO/evaluation
  entry points moved to the importable `scripts/` package
* the supported commands include `python -m scripts.demo.gym_demo`,
  `python -m scripts.training.train_ppo`, `python -m scripts.evaluation.eval_ppo`,
  `python -m scripts.evaluation.compare_bt_ppo`, and
  `python -m scripts.evaluation.eval_m43_generalization`; M5.1 later added
  `python -m scripts.evaluation.eval_m51_hybrid`
* M4.2/M4.3 shared BT/PPO Episode execution, initial-state capture/matching, and
  decision-frequency constants moved to `autonomy_lab/experiment/runners.py`
* scenario/seed orchestration, CSV paths, milestone grouping, and summary logic
  remain in their specific scripts; M4.3 no longer imports the M4.2 script
* the distinct `eval_ppo` Random/PPO diagnostic loop remains local because it
  records reward components and visibility/control diagnostics rather than the
  M4.2/M4.3 public comparison-row contract
* Core Simulation continues to import no BT, Gym, Rendering, Experiment, or
  Scripts modules; no `autonomy_lab` module depends on `scripts/`
* M4.0-M4.3 behavior, checkpoints, scenarios, metrics, and frozen result files
  remain unchanged; M4.4 introduced no new experiment or training

M4.4 changed structure only. It did not change BT/PPO behavior, Observation,
Reward, Action, dynamics, Perception, collision/termination semantics, Recorder
metrics, or M4.2/M4.3 statistics. M4.4 itself did not start M5 work.

Milestone 4.3 zero-shot geometry generalization completed:

* `scripts/evaluation/eval_m43_generalization.py` reuses the shared BT/PPO episode runners and keeps
  the default BT plus `models/ppo_m41b_control10hz.zip` frozen
* evaluation keeps World simulation at 60 Hz, BT tick at 60 Hz, and
  deterministic PPO at 10 Hz with `action_repeat=6`
* each fixed scenario runs once with seed 5001; the statistical unit is the
  scenario, so mild-unseen success is reported as successful scenes out of four
  rather than as repeated-episode probability
* seen scenarios are `rl_sanity` and `ppo_simple_obstacle`; test-only mild
  variants are `m43_target_shift`, `m43_obstacle_shift`,
  `m43_reverse_detour`, and `m43_combined_shift`; `ppo_simple_obstacles` remains
  a separately interpreted hard stress test
* BT and PPO each passed 2/2 seen, 4/4 mild unseen, and 1/1 hard scenes; the
  seen-to-mild success-rate drop was therefore zero for both Controllers
* on mild unseen scenes BT averaged 6.050 s, 781.3 px, and zero collisions;
  PPO averaged 3.613 s, 788.9 px, and one collision event per scene because all
  four events occurred in Reverse Detour
* Reverse Detour uses obstacle `(350, 60, 80, 240)`: for the 32 px Agent its
  upper net clearance is 28 px versus 68 px in the seen baseline, while lower
  net clearance is 268 px; real collision checks confirmed both routes remain
  physically traversable
* the trajectory-derived direction metric and human demo agree: BT selected
  the wide lower route, whereas PPO retained its learned upper preference,
  collided four times, then still reached the Target
* this is evidence that the current frozen policies solve all four bounded
  geometry probes, but PPO's unchanged detour side exposes a route-selection
  bias; it does not establish arbitrary-map or distributional generalization

M4.3 structured outputs:

```text
experiments/comparisons/m43_generalization.csv
experiments/comparisons/m43_generalization_summary.json
experiments/comparisons/m43_runs/<scenario>/<controller>/
experiments/comparisons/m43_human_demos/<scenario>/<controller>/
```

Reproduction commands:

```text
conda run -n pygame_lab python -m scripts.evaluation.eval_m43_generalization
conda run -n pygame_lab python -m scripts.evaluation.eval_m43_generalization --human-demo
```

M4.3 did not modify or retrain BT/PPO, and did not change Observation, Reward,
Action, World dynamics, Perception, termination, collision semantics, Renderer,
or Recorder schema. M4.2 outputs were kept intact and no later milestone was
started.

Milestone 4.2 BT vs PPO baseline completed:

* `scripts/evaluation/compare_bt_ppo.py` runs the frozen default BT and
  `models/ppo_m41b_control10hz.zip` on the same Environment scenario and seed
* each BT/PPO pair verifies Agent, Target, obstacle, radius, heading, speed, and
  World geometry initial state with a small floating-point tolerance
* World simulation remains 60 Hz; the current BT ticks at 60 Hz, while PPO uses
  deterministic inference at 10 Hz with `action_repeat=6`
* this is a comparison of the current complete Controller baselines, not a
  decision-frequency-matched algorithm ablation
* elapsed time, trajectory, path length, collision events, and termination are
  recorded from actual World internal simulation steps; decision frequency and
  decision count are additional comparison fields
* evaluation used seeds 4001-4010; current fixed layouts make all ten initial
  states equivalent, so these repetitions standardize the pipeline and do not
  establish random generalization
* `rl_sanity` and `ppo_simple_obstacle` are primary baselines;
  `ppo_simple_obstacles` is a separately labeled hard stress test and is not
  combined into a mixed overall success rate
* both Controllers achieved 100% success and zero collisions in all three
  scenarios; all termination reasons were `target_reached`
* on `rl_sanity`, BT was faster and shorter: 1.683 s / 370.3 px versus PPO's
  1.983 s / 436.3 px
* on `ppo_simple_obstacle`, PPO was faster (3.600 s versus 6.100 s), while BT's
  path was slightly shorter (783.2 px versus 792.0 px)
* on the hard stress test, PPO was faster and shorter: 3.350 s / 737.0 px versus
  BT's 5.100 s / 772.9 px
* mean decision counts reflect each frozen Controller clock: BT used 101, 366,
  and 306 decisions across the three scenarios; PPO used 20, 36, and 34
* human trajectories show BT taking a straight line in `rl_sanity` while PPO
  makes an unnecessary arc; in the single-obstacle scene BT routes below the
  obstacle and PPO routes above it, both without collision or visible trapping

M4.2 structured outputs:

```text
experiments/comparisons/m42_bt_vs_ppo.csv
experiments/comparisons/m42_bt_vs_ppo_summary.json
experiments/comparisons/runs/<scenario>/<controller>/
experiments/comparisons/human_demos/<scenario>/<controller>/
```

Reproduction commands:

```text
conda run -n pygame_lab python -m scripts.evaluation.compare_bt_ppo
conda run -n pygame_lab python -m scripts.evaluation.compare_bt_ppo --human-demo
```

M4.2 did not retrain PPO or change BT topology/parameters, reward, Observation,
action repeat, dynamics, Perception, scenarios, termination, collision semantics,
Renderer, or Recorder schema. No Hybrid BT+RL work was started.

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
conda run -n pygame_lab python -m scripts.training.train_ppo --scenario ppo_simple_obstacle --seed 44 --target-timesteps 200000 --model-path models/ppo_m41b_control10hz.zip --log-label m41b_phase_a_training --init-model-path models/ppo_m40.zip --action-repeat 6
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41b_phase_a_200k --render-mode none --action-repeat 6
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41b_control10hz.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41b_phase_a_human --render-mode human --action-repeat 6
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
conda run -n pygame_lab python -m scripts.training.train_ppo --scenario ppo_simple_obstacle --seed 44 --target-timesteps 200000 --model-path models/ppo_m41a_simple_obstacle.zip --log-label m41a_training --init-model-path models/ppo_m40.zip
conda run -n pygame_lab python -m scripts.training.train_ppo --scenario ppo_simple_obstacle --seed 44 --target-timesteps 500000 --model-path models/ppo_m41a_simple_obstacle.zip --log-label m41a_training --resume
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller both --episodes 10 --evaluation-seed-start 3001 --tag m41a_500k --render-mode none
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacle --model-path models/ppo_m41a_simple_obstacle.zip --controller ppo --episodes 1 --evaluation-seed-start 3001 --tag m41a_human --render-mode human
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
* `scripts.training.train_ppo` and `scripts.evaluation.eval_ppo` accept scenario, model/checkpoint, log, and
  evaluation-label inputs while preserving their M4.0 defaults
* obstacle fine-tuning initialized from `models/ppo_m40.zip`, saved to
  `models/ppo_m41_obstacles.zip`, evaluated at 200k, then resumed without
  reinitialization to the fixed 500k limit
* deterministic PPO and Random both achieved 0% success at 200k and 500k;
  training stopped at the required limit without reward or hyperparameter tuning
* diagnostics confirmed correct Action mapping, sufficient 20 s horizon, and a
  collision-free 3.33 s geometric route; the learned policy loses Target
  visibility near the obstacle and enters a local control loop

This M4.1 checkpoint remains retained failure evidence. The later M4.1b 10 Hz
baseline passed before the explicitly requested M4.2 comparison was started.

M4.1 reproduction commands:

```text
conda run -n pygame_lab python -m scripts.training.train_ppo --scenario ppo_simple_obstacles --seed 43 --target-timesteps 200000 --model-path models/ppo_m41_obstacles.zip --log-label m41_training --init-model-path models/ppo_m40.zip
conda run -n pygame_lab python -m scripts.training.train_ppo --scenario ppo_simple_obstacles --seed 43 --target-timesteps 500000 --model-path models/ppo_m41_obstacles.zip --log-label m41_training --resume
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --scenario ppo_simple_obstacles --model-path models/ppo_m41_obstacles.zip --evaluation-seed-start 2001 --tag m41_500k
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
* `scripts.training.train_ppo` trains one default `PPO("MlpPolicy")` environment, records SB3
  Monitor episode evidence, and can resume the same checkpoint to a cumulative
  timestep target
* `scripts.evaluation.eval_ppo` compares Random and deterministic PPO with the same fixed seeds
  and reuses `ExperimentRecorder` metric definitions
* the 50k evaluation checkpoint is retained; training resumed from that model to
  100k because the 50k deterministic policy did not yet outperform Random
* the final model is saved at `models/ppo_m40.zip`; headless and human evaluation
  both use the existing Gym Adapter and read-only Pygame Renderer

Minimal commands:

```text
conda run -n pygame_lab python -m scripts.training.train_ppo --target-timesteps 50000
conda run -n pygame_lab python -m scripts.training.train_ppo --target-timesteps 100000 --resume
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --controller both --tag 100k
conda run -n pygame_lab python -m scripts.evaluation.eval_ppo --controller ppo --episodes 1 --render-mode human
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
* `scripts.demo.gym_demo` provides a random-Action headless smoke test

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
