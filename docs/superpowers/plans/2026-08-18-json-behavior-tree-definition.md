# JSON Behavior Tree Definition v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the current `py_trees` runtime from selectable JSON v1 definitions and record the selected definition in M2 experiment output.

**Architecture:** A generic recursive Loader validates composite/leaf structure and delegates leaf construction to an explicit factory dictionary. The Controller loads the resulting real `py_trees` root and keeps only runtime housekeeping; the Visualizer continues to inspect that root.

**Tech Stack:** Python 3.11, standard-library `json`/`dataclasses`, pygame 2.6.1, py_trees 2.5.0, `unittest`.

## Global Constraints

- Use the `pygame_lab` Conda environment and add no dependency.
- Work inline under project FAST Mode; do not use multi-agent execution or a worktree.
- Support exactly selector, sequence, condition, and action in `bt-lab/v1` JSON.
- Keep `py_trees` as the only runtime and preserve all Behavior `update()` logic.
- Add no XML, YAML, ports, SubTree, Decorator, Parallel, editor, hot reload, plugin discovery, RL, or Gymnasium support.
- JSON params override scene BT parameters; omitted params fall back to the scene.
- Preserve the user's existing `.gitignore` and unrelated working-tree changes.
- Do not commit implementation changes from the already-dirty `main` worktree.

---

### Task 1: JSON definition, registry, and recursive loader

**Files:**
- Create: `bt_configs/default.json`
- Create: `autonomy_lab/behavior_registry.py`
- Create: `autonomy_lab/bt_loader.py`
- Create: `tests/test_bt_loader.py`

**Interfaces:**
- Produces: `BehaviorBuildContext`, `BEHAVIOR_REGISTRY`, `LoadedBehaviorTree`, `load_bt_definition(name, config_dir)`, `build_behavior_tree(definition, context)`, and `load_behavior_tree(name, context, config_dir)`.

- [x] **Step 1: Write failing Loader tests**

Cover the real default 16-node topology, unknown behavior, missing children,
invalid type, duplicate name, invalid Condition reference, child reordering, and
adding a second registered `SearchTarget` node. Representative assertions:

```python
loaded = load_behavior_tree("default", context)
self.assertEqual(loaded.config_id, "default_bt")
self.assertEqual(len(list(loaded.root.iterate())), 16)
self.assertEqual(
    [child.name for child in loaded.root.children],
    ["Obstacle Avoidance", "Target Gap Navigation", "Target Pursuit",
     "Gap Exploration", "Search Target"],
)

with self.assertRaisesRegex(ValueError, "unknown behavior 'MissingBehavior'"):
    build_behavior_tree(definition, context)
```

- [x] **Step 2: Run Loader tests and verify RED**

```text
conda run -n pygame_lab python -m unittest tests.test_bt_loader -v
```

Expected: import failure because Loader and Registry modules do not exist.

- [x] **Step 3: Create the exact default JSON**

Encode the current 16 nodes and their memory flags. Use `params.condition` for
the three Action-to-Condition references; do not add new Behaviors.

- [x] **Step 4: Implement the explicit factory Registry**

Use a plain context dataclass and callable dictionary:

```python
@dataclass
class BehaviorBuildContext:
    perception: AgentPerception
    command: dict[str, float]
    behavior_config: dict
    nodes_by_name: dict[str, py_trees.behaviour.Behaviour] = field(default_factory=dict)

BEHAVIOR_REGISTRY = {
    "ObstacleThreat": build_obstacle_threat,
    "AvoidObstacle": build_avoid_obstacle,
    "TargetAvailable": build_target_available,
    "TargetPathBlocked": build_target_path_blocked,
    "TargetAlignedGap": build_target_aligned_gap,
    "MoveThroughGap": build_move_through_gap,
    "MoveToTarget": build_move_to_target,
    "TraversableGap": build_traversable_gap,
    "SearchTarget": build_search_target,
}
```

Factories resolve JSON-first numeric params and named Condition dependencies.

- [x] **Step 5: Implement minimum recursive validation and construction**

Return a `LoadedBehaviorTree` dataclass. Require `format == "bt-lab/v1"`,
non-empty metadata, unique names, Boolean memory, non-empty composite children,
registered leaf behavior, object params, and declared leaf type matching
`visual_type`.

- [x] **Step 6: Run Loader tests and require GREEN**

Run the Task 1 module and require all definition, error, ordering, and extension
tests to pass.

### Task 2: Controller, CLI, and runtime Visualizer integration

**Files:**
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `main.py`
- Modify: `tests/test_perception_bt.py`
- Modify: `tests/test_bt_visualizer.py`
- Modify: `tests/test_main_lifecycle.py`

**Interfaces:**
- Produces: `BehaviorTreeController(environment, bt_config="default")`, `controller.bt_config_id`, `controller.nodes_by_name`, and CLI `--bt`.

- [x] **Step 1: Write failing controller/CLI integration tests**

Require the default loaded topology, unchanged first command `(1.0, 0.5)`,
`bt_config_id == "default_bt"`, config selection independent of working
directory, and Visualizer topology from the loaded runtime root.

- [x] **Step 2: Verify controller tests fail before migration**

Run the Loader, perception-BT, visualizer, and lifecycle modules. Expected:
failure because the Controller has no `bt_config` or `nodes_by_name` interface
and the CLI has no `--bt`.

- [x] **Step 3: Replace hard-coded topology with Loader output**

Create perception/command/context, call `load_behavior_tree`, assign its root,
and derive action/condition collections from real node classes. Tick all
`AvoidObstacle` actions with `dt`, apply emergency distance to all
`ObstacleThreat` nodes while any `MoveThroughGap` is committed, determine the
active Action generically, and preserve reset/summary behavior.

- [x] **Step 4: Add `--bt default` and pass selection into the Controller**

Use project-root config resolution already provided by the Loader. Manual mode
accepts the argument but does not load a tree.

- [x] **Step 5: Update tests to inspect `nodes_by_name` and require GREEN**

Do not preserve topology construction attributes in the Controller. Existing
behavior assertions should retrieve the real loaded nodes by JSON name.

### Task 3: M2 `bt_config_id` JSON/CSV recording

**Files:**
- Modify: `autonomy_lab/experiment.py`
- Modify: `main.py`
- Modify: `tests/test_perception_bt.py`
- Modify: `tests/test_main_lifecycle.py`

**Interfaces:**
- Produces: `ExperimentRecorder.start_episode(..., bt_config_id=None)` and a `bt_config_id` field in detailed JSON and CSV summaries.

- [x] **Step 1: Write failing recorder tests**

Assert BT episodes persist `default_bt`, manual episodes persist `None`/blank,
and an old known CSV header is upgraded without losing its historical row.

- [x] **Step 2: Run recorder/lifecycle tests and verify RED**

Expected: `start_episode` rejects `bt_config_id` and payload/CSV omit the field.

- [x] **Step 3: Add the field and one-time old-header upgrade**

Add `bt_config_id` to `SUMMARY_FIELDS`, episode state, JSON payload, summary
print, and both `start_episode` calls in `main.py`. Before appending, rewrite
only the exact previous known header with an empty new column; reject no other
schema.

- [x] **Step 4: Run M2 and lifecycle tests and require GREEN**

Confirm no existing metric values or termination behavior change.

### Task 4: Documentation and runtime verification

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_GUIDE.md`
- Modify: `docs/superpowers/plans/2026-08-18-json-behavior-tree-definition.md`

- [x] **Step 1: Document JSON v1 and `--bt default`**

Record the definition/Loader/Registry/runtime flow, JSON fields, current
migration, and explicit non-goals. Keep the description at research-prototype
scope.

- [x] **Step 2: Run the four existing modules plus Loader tests**

```text
conda run -n pygame_lab python -m unittest tests.test_bt_loader tests.test_perception tests.test_perception_bt tests.test_bt_visualizer tests.test_main_lifecycle -v
```

- [x] **Step 3: Run all three scenes under SDL dummy**

For each scenario, load `default`, tick, draw the Environment and loaded
Visualizer, and reset. Also run a fixed-step `simple` episode to confirm target
arrival remains possible.

- [x] **Step 4: Compile and check the patch**

```text
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
git diff --check
```

Stop after these checks; do not add future BT features.
