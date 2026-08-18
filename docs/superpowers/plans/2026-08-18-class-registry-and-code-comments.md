# Class Registry and Code Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-Behavior factory functions with a direct class registry and uniform constructor, then document the current application with concise Docstrings and targeted Chinese comments.

**Architecture:** A small `BehaviorBuildContext` module carries runtime dependencies. Task-specific leaves inherit from `AgentBehaviour`, `AgentCondition`, or `AgentAction`, accept `(context, name, **params)`, and own their parameter interpretation. The Loader validates JSON structure and generically instantiates the class selected by the Registry.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, standard-library `dataclasses`/`json`, `unittest`.

## Global Constraints

- Use only the existing `pygame_lab` Conda environment and add no dependency.
- Keep JSON `bt-lab/v1` and `bt_configs/default.json` backward compatible.
- Preserve all current Behavior `update()` logic, BT topology, visualization, and experiment metrics.
- Add no reflection, automatic discovery, plugin system, parameter-schema framework, RL, or Gymnasium.
- Use English Docstrings and Chinese inline comments only for non-obvious algorithms or control flow.
- Preserve unrelated files and existing dirty-worktree changes. Do not commit implementation changes from the dirty worktree.

---

### Task 1: Standard Behavior context, base classes, and constructors

**Files:**
- Create: `autonomy_lab/behavior_context.py`
- Modify: `autonomy_lab/behaviors.py`
- Modify: `tests/test_perception_bt.py`
- Modify: `tests/test_bt_loader.py`

**Interfaces:**
- Produces: `BehaviorBuildContext(perception, command, behavior_config, nodes_by_name={})`.
- Produces: `AgentBehaviour(context, name, **params)`, `AgentCondition`, and `AgentAction`.
- Produces: uniform constructors for all nine registered task-specific Behavior classes.

- [x] **Step 1: Add failing tests for the common constructor contract**

Add a loader test Behavior and constructor assertions:

```python
class ProbeAction(AgentAction):
    allowed_params = frozenset({"amount"})

    def update(self):
        return py_trees.common.Status.SUCCESS


def test_agent_action_uses_uniform_context_name_and_params(self):
    context = make_context()
    node = ProbeAction(context=context, name="Probe", amount=2.0)
    self.assertIs(node.context, context)
    self.assertEqual(node.name, "Probe")
    self.assertEqual(node.params, {"amount": 2.0})
    self.assertEqual(node.visual_type, "action")
```

Add cases asserting an unknown parameter reports `unknown params for
SearchTarget` and a non-numeric override reports `parameter 'turn' for
SearchTarget must be numeric`.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
conda run -n pygame_lab python -m unittest tests.test_bt_loader tests.test_perception_bt -v
```

Expected: import or constructor failures because the common bases and context
module do not exist yet.

- [x] **Step 3: Implement the context and lightweight base classes**

Create the context without construction policy:

```python
@dataclass
class BehaviorBuildContext:
    perception: AgentPerception
    command: dict[str, float]
    behavior_config: dict
    nodes_by_name: dict[str, py_trees.behaviour.Behaviour] = field(default_factory=dict)
```

Add the three bases to `behaviors.py`. `AgentBehaviour.__init__` stores context
and params, rejects keys outside `allowed_params`, and exposes these helpers:

```python
def number_param(self, param_name: str, config_name: str) -> float
def condition_param(self, param_name: str, expected_type: type | tuple[type, ...]) -> py_trees.behaviour.Behaviour
```

`number_param` checks `self.params` first and reads `context.behavior_config`
only when the JSON parameter is absent. `condition_param` resolves an existing
name from `context.nodes_by_name` and validates its concrete type.

- [x] **Step 4: Convert current leaf classes without changing execution logic**

Make every registered class accept exactly:

```python
def __init__(self, context: BehaviorBuildContext, name: str, **params: object) -> None:
```

Use class-level `allowed_params` and the base helpers. Conditions with no state
can inherit the base constructor directly. Stateful Conditions initialize only
their `threat` or `gap` field. Actions retrieve `command` and `perception` from
context and preserve existing `initialise`, `update`, and `terminate` bodies.

- [x] **Step 5: Adapt direct Behavior tests to construct one shared context**

Update `tests/test_perception_bt.py` helpers so direct nodes are created with a
`BehaviorBuildContext`, a readable JSON-style node name, and keyword params.
Keep all existing status, command, feedback, and preemption assertions.

- [x] **Step 6: Run focused Behavior tests and require GREEN**

Run the Task 1 command again. Require every existing perception/Behavior result
to remain unchanged.

### Task 2: Direct class Registry and generic Loader

**Files:**
- Modify: `autonomy_lab/behavior_registry.py`
- Modify: `autonomy_lab/bt_loader.py`
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `tests/test_bt_loader.py`

**Interfaces:**
- Produces: `BehaviorClass = type[AgentBehaviour]`.
- Produces: `BEHAVIOR_REGISTRY: dict[str, BehaviorClass]` mapping names directly to classes.
- Consumes: the uniform `Behavior(context, name, **params)` contract from Task 1.

- [x] **Step 1: Add failing Registry and generic Loader assertions**

Require every value to be an `AgentBehaviour` subclass:

```python
for behavior_class in BEHAVIOR_REGISTRY.values():
    self.assertTrue(issubclass(behavior_class, AgentBehaviour))
```

Build a single-node definition with `registry={"ProbeAction": ProbeAction}` and
assert its JSON `params` arrive in the constructed instance. This test proves a
new conforming class requires no new Loader branch or factory.

- [x] **Step 2: Run Loader tests and verify RED against factory values**

Run:

```powershell
conda run -n pygame_lab python -m unittest tests.test_bt_loader -v
```

Expected: the subclass assertion fails because Registry values are functions.

- [x] **Step 3: Replace factory functions with the class dictionary**

Reduce `behavior_registry.py` to the explicit imports, `BehaviorClass` alias,
and the existing nine-name dictionary:

```python
BEHAVIOR_REGISTRY = {
    "ObstacleThreat": ObstacleThreat,
    "AvoidObstacle": AvoidObstacle,
    "TargetAvailable": TargetAvailable,
    "TargetPathBlocked": TargetPathBlocked,
    "TargetAlignedGap": TargetAlignedGap,
    "MoveThroughGap": MoveThroughGap,
    "MoveToTarget": MoveToTarget,
    "TraversableGap": TraversableGap,
    "SearchTarget": SearchTarget,
}
```

Delete `_check_params`, `_number`, `_condition`, and every `build_xx()`.

- [x] **Step 4: Make Loader leaf construction fully generic**

Change the registry annotation to `dict[str, BehaviorClass]` and instantiate:

```python
behavior_class = registry.get(behavior_name)
if behavior_class is None:
    raise ValueError(f"{path}: unknown behavior '{behavior_name}'")
try:
    node = behavior_class(context=context, name=node_name, **params)
except (TypeError, ValueError) as error:
    raise ValueError(f"{path}: {error}") from error
```

Keep node-path errors and `visual_type` validation. Import the context from
`behavior_context.py` in both Loader and Controller.

- [x] **Step 5: Run Loader and runtime integration tests**

Run:

```powershell
conda run -n pygame_lab python -m unittest tests.test_bt_loader tests.test_perception_bt tests.test_bt_visualizer -v
```

Require the default 16-node topology, JSON reordering, parameter fallback,
Visualizer topology, and controller commands to remain unchanged.

### Task 3: Necessary project-wide code documentation

**Files:**
- Modify: `main.py`
- Modify: `autonomy_lab/__init__.py`
- Modify: `autonomy_lab/agent.py`
- Modify: `autonomy_lab/assets.py`
- Modify: `autonomy_lab/environment.py`
- Modify: `autonomy_lab/perception.py`
- Modify: `autonomy_lab/behaviors.py`
- Modify: `autonomy_lab/behavior_context.py`
- Modify: `autonomy_lab/behavior_registry.py`
- Modify: `autonomy_lab/bt_loader.py`
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `autonomy_lab/bt_visualizer.py`
- Modify: `autonomy_lab/experiment.py`
- Modify: `autonomy_lab/scene_config.py`
- Modify: `tests/test_bt_loader.py`
- Modify: `tests/test_bt_visualizer.py`
- Modify: `tests/test_experiment.py`
- Modify: `tests/test_main_lifecycle.py`
- Modify: `tests/test_perception.py`
- Modify: `tests/test_perception_bt.py`
- Modify: `README.md`
- Modify: `PROJECT_GUIDE.md`

**Interfaces:**
- No runtime interface changes.
- Produces: English Docstrings plus focused Chinese algorithm/control-flow comments.

- [x] **Step 1: Add missing module and public API Docstrings**

Give `main.py`, `agent.py`, `environment.py`, and every test module a concise
module Docstring. Add or improve class/method Docstrings where the caller needs
to know ownership, inputs, side effects, or lifecycle. Do not restate signatures.

- [x] **Step 2: Annotate only non-obvious runtime logic**

Add short Chinese comments immediately above these specific mechanisms:

- main-loop `dt` cap, reset/finish ordering, and post-success display loop;
- heading integration and visualization-only sprite rotation;
- optional-asset fallback;
- axis-separated collision sliding and circle/rectangle test;
- perception target-information modes, ray inflation by Agent clearance,
  contiguous open-ray grouping, and fixed world-space gap entries;
- command clearing before each BT tick, timed avoidance `dt`, and emergency gap
  commitment threshold;
- runtime-tree extraction and leaf-centered layout;
- collision contact-transition counting, trajectory sampling cadence, and old
  CSV header migration.

- [x] **Step 3: Document tests only where intent is otherwise unclear**

Add one-line module Docstrings and comments for synthetic snapshots, temporary
JSON definitions, dummy SDL lifecycle frames, and CSV migration fixtures. Avoid
comments on direct assertions.

- [x] **Step 4: Update user-facing architecture notes**

Update README and `PROJECT_GUIDE.md` from “explicit factory Registry” to “direct
Class Registry with standardized Behavior constructors”. Document the final
flow without adding future roadmap items.

- [x] **Step 5: Run the complete current test suite**

Run:

```powershell
conda run -n pygame_lab python -m unittest discover -s tests -v
```

Expected: all current tests pass with zero failures.

- [x] **Step 6: Verify all three scenarios and final source integrity**

Under SDL dummy, construct `BehaviorTreeController(..., bt_config="default")`
for `simple`, `obstacle_course`, and `dense_obstacles`; tick once, draw the
Environment and BT panel, then reset. Finally run:

```powershell
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
git diff --check
```

Stop after these checks. Do not add new Behavior capabilities.
