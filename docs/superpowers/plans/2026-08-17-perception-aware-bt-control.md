# Perception-aware Behavior Tree Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Behavior Tree act on a synchronous perception snapshot with configurable perceived or ground-truth target information.

**Architecture:** A small `AgentPerception` derives target and obstacle observations from Environment ground truth before every BT tick. Conditions and actions consume only its `PerceptionSnapshot`; the existing controller owns the shared command and real `py_trees` tree, while Environment and the generic BT visualizer remain renderers of ground truth and runtime state.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, standard-library `dataclasses` and `unittest`.

## Global Constraints

- Use the existing Conda environment `pygame_lab`.
- Keep BT ticking frame-synchronous at approximately 60 Hz.
- Add no third-party dependency and do not modify `environment.yml`.
- Do not add Blackboard, target memory, world model, sensor framework, path planner, Gymnasium, or RL.
- Preserve M2 metric definitions and JSON/CSV formats.
- Preserve the existing Assets, reset lifecycle, collision model, and manual controller.

---

### Task 1: Perception snapshot, target modes, and obstacle sensing

**Files:**
- Create: `autonomy_lab/perception.py`
- Modify: `autonomy_lab/scene_config.py`
- Create: `tests/test_perception.py`

**Interfaces:**
- Consumes: `Environment`, root scene keys `sensor` and `target_information_mode`.
- Produces: `PerceivedObstacle`, `PerceptionSnapshot`, and `AgentPerception.update() -> PerceptionSnapshot`.

- [x] **Step 1: Write failing target-perception tests**

Use real `Environment(get_scene("simple"))` objects with controlled Agent/Target
positions. Assert literal outcomes for a target ahead, behind, outside 300 px,
across the `+/-180` heading boundary, and behind a blocking `pygame.Rect`.
Assert that perceived mode hides distance/bearing when invisible, while
ground-truth mode keeps exact distance/bearing but reports sensor visibility
independently.

- [x] **Step 2: Verify RED**

```bash
conda run -n pygame_lab python -m unittest tests.test_perception -v
```

Expected: import failure because `autonomy_lab.perception` does not exist.

- [x] **Step 3: Implement target perception and configuration**

Add defaults:

```python
DEFAULT_SENSOR_CONFIG = {
    "range": 300.0,
    "fov_degrees": 120.0,
    "los_enabled": True,
}
```

and root scene key `target_information_mode = "perceived"`. Implement relative
bearing normalization, Range/FOV checks, `Rect.clipline()` LOS checks, mode
validation, and unavailable reasons without exposing target measurements in
perceived mode.

- [x] **Step 4: Verify target tests GREEN**

Run the same test module and expect all target cases to pass.

- [x] **Step 5: Write failing obstacle-perception tests**

Assert that only rectangle nearest points inside Range and FOV produce
`PerceivedObstacle` entries, entries are sorted by clearance, and
`nearest_obstacle` is the first entry.

- [x] **Step 6: Implement obstacle perception and verify GREEN**

Calculate closest rectangle points, clearance minus Agent radius, relative
bearing, filtering and ordering. Re-run `tests.test_perception`.

### Task 2: Perception-driven reactive Behavior Tree

**Files:**
- Modify: `autonomy_lab/behaviors.py`
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `autonomy_lab/scene_config.py`
- Create: `tests/test_perception_bt.py`

**Interfaces:**
- Consumes: `AgentPerception.snapshot` and the existing shared command dict.
- Produces: `ObstacleThreat`, `TargetAvailable`, `AvoidObstacle`, `MoveToTarget`, `SearchTarget`, and the three-branch real `py_trees` topology.

- [x] **Step 1: Write failing node tests**

With real perception snapshots, assert:

```text
TargetAvailable: available -> SUCCESS, unavailable -> FAILURE
ObstacleThreat: visible and near/front -> SUCCESS, otherwise FAILURE
MoveToTarget: uses snapshot bearing/distance to write turn/throttle
SearchTarget: writes turn=0.25, throttle=0.25 and returns RUNNING
```

Also assert concise feedback matches visible, unavailable, threat, pursuit, and
search decisions.

- [x] **Step 2: Verify RED**

```bash
conda run -n pygame_lab python -m unittest tests.test_perception_bt -v
```

Expected: missing new behavior classes and old `MoveToTarget` signature.

- [x] **Step 3: Implement minimal perception-driven nodes**

Replace `ObstacleNear` with `ObstacleThreat`, make `MoveToTarget` consume only
`AgentPerception`, add `TargetAvailable` and `SearchTarget`, and retain
`visual_type` metadata. Conditions never write commands. Avoidance uses the
selected perceived bearing and resets only its local timing state in
`terminate()`.

- [x] **Step 4: Verify node tests GREEN**

Run `tests.test_perception_bt` and fix only behavior required by the assertions.

- [x] **Step 5: Write failing controller/preemption tests**

Construct a real `BehaviorTreeController` and assert its topology is:

```text
Priority Selector
  Obstacle Avoidance -> Obstacle Threat?, Avoid Obstacle
  Target Pursuit -> Target Available?, Move To Target
  Search Target
```

Verify Search becomes RUNNING when target unavailable, switches to Pursuit when
target enters FOV, and a newly perceived obstacle invalidates MoveToTarget while
the returned command equals AvoidObstacle's command rather than stale pursuit
or zero values.

- [x] **Step 6: Implement controller topology and verify GREEN**

Create `AgentPerception` in the controller, call `update()` before clearing and
ticking commands, keep the root Selector and Target Pursuit reactive, let the
short timed Obstacle Avoidance sequence retain action state, extend active-action
labels to SearchTarget, and keep reset/M2 interfaces.

### Task 3: FOV/feedback rendering and integration verification

**Files:**
- Modify: `autonomy_lab/environment.py`
- Modify: `autonomy_lab/bt_visualizer.py`
- Modify: `tests/test_bt_visualizer.py`
- Modify: `PROJECT_GUIDE.md`

**Interfaces:**
- Consumes: scene sensor configuration and real node `feedback_message`.
- Produces: low-opacity FOV sector and generic visited-node feedback text.

- [x] **Step 1: Write failing renderer integration tests**

Draw a real Environment and controller panel to a Pygame Surface. Assert drawing
does not fail with the expanded topology, the visualizer discovers all eight
nodes without a second layout model, repeated draws reuse the layout, and reset
clears active state. Exercise generic feedback truncation through a visited real
node rather than checking source text.

- [x] **Step 2: Verify RED**

Run the focused visualizer test module and confirm failure because the new
topology/feedback rendering is absent.

- [x] **Step 3: Implement FOV and feedback rendering**

Draw a translucent polygon sector from Agent heading before world objects.
Render a short, truncated `feedback_message` for visited nodes in the existing
generic `_draw_node()` method. Do not import or name business behaviors in the
visualizer.

- [x] **Step 4: Run focused automated verification**

```bash
conda run -n pygame_lab python -m unittest tests.test_perception tests.test_perception_bt tests.test_bt_visualizer tests.test_main_lifecycle -v
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
```

Expected: all focused tests and compilation pass.

- [x] **Step 5: Verify M2 and Pygame lifecycle**

Run a bounded SDL dummy scenario or fixed-step controller script. Confirm a
saved Episode retains existing JSON/CSV fields, has positive BT tick count, and
counts SearchTarget -> MoveToTarget / MoveToTarget -> AvoidObstacle transitions
through the existing active-action metric. Confirm reset and manual mode still
construct and draw.

- [x] **Step 6: Update documentation and final checks**

Update only `PROJECT_GUIDE.md` Current Milestone with completed M1.3 facts. Run
focused tests, compilation, `git diff --check`, and inspect scope. Do not add
future Target Memory, Blackboard, Gymnasium, or RL work.
