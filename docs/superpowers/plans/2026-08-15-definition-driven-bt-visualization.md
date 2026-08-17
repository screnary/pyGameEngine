# Definition-driven Behavior Tree Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded Behavior Tree panel topology with a cached layout derived from the live `py_trees` tree while retaining current runtime highlighting and experiment metrics.

**Architecture:** A new `BTVisualizer` traverses `tree.root`, caches a signature and lightweight visual-node adapters, computes a post-order hierarchical layout, and draws from real node statuses plus `SnapshotVisitor`. `BehaviorTreeController` remains the runtime owner and delegates only panel rendering; task behavior nodes expose non-semantic `visual_type` metadata.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, standard-library `unittest`.

## Global Constraints

- Use the existing Conda environment `pygame_lab`.
- Add no graph, GUI, web, Gymnasium, RL, or other third-party dependency.
- Do not change Environment dynamics, Agent motion, or Milestone 2 metric definitions.
- Keep the implementation to one lightweight visualizer module and targeted tests.
- Do not add Behavior Tree editing, generation, evolution, zoom, pan, or scrolling.

---

### Task 1: Topology extraction, classification, and cached layout

**Files:**
- Create: `autonomy_lab/bt_visualizer.py`
- Create: `tests/test_bt_visualizer.py`

**Interfaces:**
- Consumes: a real `py_trees.behaviour.Behaviour` root.
- Produces: `BTVisualizer(root, snapshot)`, `sync(panel_rect) -> bool`, `visual_nodes`, `connections`, `signature`, and `rebuild_count`.

- [ ] **Step 1: Write failing topology tests**

Create a real in-memory tree containing Selector, Sequence, a metadata-tagged condition, and an action. Assert that `sync()` discovers all five nodes, preserves parent/child order, classifies the node types, calculates distinct depth rows, and returns `True` for the first rebuild.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
conda run -n pygame_lab python -m unittest tests.test_bt_visualizer -v
```

Expected: import failure because `autonomy_lab.bt_visualizer` does not exist.

- [ ] **Step 3: Implement minimal traversal and layout**

Create a `VisualNode` dataclass referencing the real node and a `BTVisualizer` that:

```python
signature = tuple((node.id, parent.id if parent else None, order) ...)
```

Traverses `children` recursively, classifies native composite/decorator types with `isinstance`, falls back to `visual_type`, allocates leaf X slots in post-order, centers parents over their first and last child, and maps logical positions into the supplied panel rectangle.

- [ ] **Step 4: Run the tests and confirm GREEN**

Run the same `unittest` command and expect all topology tests to pass.

- [ ] **Step 5: Add and verify cache/topology-change tests**

Assert a second unchanged `sync()` returns `False` and does not increase `rebuild_count`. Add a real node to the test tree, run `sync()` again, and assert it returns `True`, adds the node and connection, and increments `rebuild_count` exactly once.

### Task 2: Runtime rendering and controller integration

**Files:**
- Modify: `autonomy_lab/bt_visualizer.py`
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `autonomy_lab/behaviors.py`
- Modify: `tests/test_bt_visualizer.py`

**Interfaces:**
- Consumes: `SnapshotVisitor.visited`, each real node's `status`, and the existing panel surface/font/world width.
- Produces: `BTVisualizer.draw(surface, font, world_width, summaries)` and metadata-driven condition/action shapes.

- [ ] **Step 1: Write failing runtime-state tests**

Tick a real test tree with `SnapshotVisitor`, then assert `runtime_state(node)` reports the real status and whether its ID is in the current visited mapping. Stop the root with `INVALID`, clear the visitor as controller reset does, and assert the node is inactive without requiring a layout rebuild.

- [ ] **Step 2: Run the targeted tests and confirm RED**

Run the single `unittest` module and confirm failure due to missing runtime-state/draw behavior.

- [ ] **Step 3: Implement drawing and integrate it**

Move generic panel drawing into `BTVisualizer`: generate all edges from cached parent links, highlight visited nodes and incoming edges using actual status, and render native/custom types without checking business names. In `BehaviorTreeController`, construct one visualizer and replace the hard-coded node/connection block with one delegation call. Add `visual_type = "condition"` or `"action"` in the three custom node constructors.

- [ ] **Step 4: Run the targeted tests and confirm GREEN**

Run the single `unittest` module and expect all tests to pass.

- [ ] **Step 5: Check for business-node rendering assumptions**

Inspect `bt_visualizer.py` and verify it contains no imports or conditionals for `ObstacleNear`, `AvoidObstacle`, `MoveToTarget`, or their display labels.

### Task 3: Integration verification and milestone documentation

**Files:**
- Modify: `PROJECT_GUIDE.md`

**Interfaces:**
- Consumes: the completed visualizer/controller integration.
- Produces: verified Milestone 1.2 behavior and an updated Current Milestone record.

- [ ] **Step 1: Run targeted automated verification**

Run:

```bash
conda run -n pygame_lab python -m unittest tests.test_bt_visualizer -v
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
```

Expected: all targeted tests pass and compilation exits successfully.

- [ ] **Step 2: Run the existing Pygame BT scenario**

Run with SDL dummy video so the current autonomous scenario completes without interaction:

```powershell
$env:SDL_VIDEODRIVER = "dummy"
conda run -n pygame_lab python main.py --scenario simple --controller bt
```

Expected: stable execution, `SUCCESS`, and a new JSON/CSV experiment record with positive BT tick and transition counts.

- [ ] **Step 3: Verify reset and dynamic topology behavior**

Use the targeted test tree to confirm reset clears active state but retains the cached layout, while adding/removing/reordering a child changes the signature and rebuilds once.

- [ ] **Step 4: Update Current Milestone**

Record only the completed Milestone 1.2 capabilities in `PROJECT_GUIDE.md`: real-tree traversal, hierarchical layout, runtime status/active path, and signature-triggered rebuilds. Retain the existing Milestone 2 completion record.

- [ ] **Step 5: Run final checks**

Run the targeted tests, compile check, `git diff --check`, and inspect the final diff for scope. Do not run an unrelated full test suite or add further features.

