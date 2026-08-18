# Simple Scene Gap FOV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the simple-scene Agent perceive the known 70-pixel obstacle channel at the observed runtime approach pose.

**Architecture:** Preserve the current radius-safe ray algorithm and apply one scene-local sensor override. Protect the behavior with a real `AgentPerception` regression test at the recorded failing pose.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, `unittest`.

## Global Constraints

- Use the `pygame_lab` Conda environment and add no dependency.
- Work inline under project FAST Mode; do not use multi-agent execution or a worktree.
- Change only the simple scene FOV from 120 to 140 degrees.
- Keep radius, gap safety margin, ray count, minimum distance, open ratio, BT topology, and other scenarios unchanged.
- Preserve the user's existing `.gitignore` modification.

---

### Task 1: Detect the simple-scene channel at the failing pose

**Files:**
- Modify: `tests/test_perception.py`
- Modify: `autonomy_lab/scene_config.py`

**Interfaces:**
- Consumes: `get_scene("simple")`, `Environment`, and `AgentPerception.snapshot`.
- Produces: a non-`None` `best_target_gap` at the recorded approach pose.

- [x] **Step 1: Write the failing behavior test**

```python
def test_simple_wide_fov_detects_narrow_channel_at_runtime_approach(self):
    scene = get_scene("simple")
    environment = Environment(scene)
    environment.agent.position.update(674.7, 439.9)
    environment.agent.heading = math.radians(-26.7)

    snapshot = AgentPerception(environment).update()

    self.assertTrue(snapshot.target_path_blocked)
    self.assertIsNotNone(snapshot.best_target_gap)
    self.assertLess(snapshot.best_target_gap.bearing, math.radians(-60.0))
    self.assertGreater(snapshot.best_target_gap.free_distance, 290.0)
```

- [x] **Step 2: Run the test and verify RED**

Run:

```text
conda run -n pygame_lab python -m unittest tests.test_perception.GapPerceptionTests.test_simple_wide_fov_detects_narrow_channel_at_runtime_approach -v
```

Expected: FAIL because `best_target_gap` is `None` under the current 120-degree
simple-scene FOV.

- [x] **Step 3: Apply the scene-local FOV override**

```python
"sensor": {**DEFAULT_SENSOR_CONFIG, "fov_degrees": 140.0},
```

Change only the `simple` scene entry in `autonomy_lab/scene_config.py`.

- [x] **Step 4: Verify the fix and regressions**

Run:

```text
conda run -n pygame_lab python -m unittest tests.test_perception -v
conda run -n pygame_lab python -m unittest tests.test_perception tests.test_perception_bt tests.test_bt_visualizer tests.test_main_lifecycle -v
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
git diff --check
```

Expected: the new test and all existing tests pass, compilation succeeds, and
the diff check reports no whitespace errors.
