# Unknown-target Gap Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an Agent without target information explore through locally safe openings instead of following one closed circular command.

**Architecture:** `AgentPerception` ray-samples FOV free space against radius-inflated rectangles and contracted world bounds, then publishes immutable `PerceivedGap` observations. Two small BT nodes select and follow the best gap below Target Pursuit and above an in-place Search fallback; Environment keeps final collision authority.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, standard-library `dataclasses` and `unittest`.

## Global Constraints

- Use the existing Conda environment `pygame_lab`.
- Keep frame-synchronous perception and BT ticks; add no thread or async loop.
- Gap selection must not read Target Ground Truth in `perceived` mode.
- Add no map, target memory, coverage planner, A*, SLAM, Gymnasium, or RL.
- Preserve Environment collision resolution and M2 JSON/CSV formats.
- Work inline in the current tree because project FAST Mode forbids worktrees and multi-agent execution.
- Preserve the user's existing `.gitignore` modification and do not commit implementation unless requested.

---

### Task 1: Ray-sampled local gap perception

**Files:**
- Modify: `autonomy_lab/perception.py`
- Modify: `autonomy_lab/scene_config.py`
- Modify: `tests/test_perception.py`

**Interfaces:**
- Consumes: `Environment.agent`, `Environment.obstacles`, `Environment.world_size`, sensor FOV/range, and four `behavior_tree` gap settings.
- Produces: `PerceivedGap(bearing, free_distance, angular_width)`, `PerceptionSnapshot.traversable_gaps`, and `PerceptionSnapshot.best_exploration_gap`.

- [x] **Step 1: Write failing gap and configuration tests**

Add tests using real `Environment` objects that assert:

```python
snapshot.best_exploration_gap.bearing == 0.0  # open forward space
snapshot.best_exploration_gap.free_distance == 300.0
all(abs(gap.bearing) >= radians(20) for gap in snapshot.traversable_gaps)
# two inflated rectangles close the narrow central gap, while side space remains
```

Also assert a world boundary caps free distance and invalid `gap_ray_count`,
`gap_min_travel_distance`, or `gap_safety_margin` raises `ValueError`.

- [x] **Step 2: Run tests to verify RED**

```bash
conda run -n pygame_lab python -m unittest tests.test_perception -v
```

Expected: failures because `PerceivedGap` and snapshot gap fields do not exist.

- [x] **Step 3: Add minimal configuration and immutable gap types**

Extend `DEFAULT_BT_CONFIG` with:

```python
"gap_ray_count": 31,
"gap_min_travel_distance": 100.0,
"gap_safety_margin": 8.0,
"gap_throttle": 0.5,
```

Add the frozen dataclass and snapshot fields exactly as declared by the
Interfaces block. Validate count `>= 3`, travel distance `> 0`, and safety
margin `>= 0` in `AgentPerception.__init__()`.

- [x] **Step 4: Implement ray distance and grouping**

Sample inclusive relative bearings from `-half_fov` to `+half_fov`. Contract
world bounds and inflate each obstacle by `ceil(agent.radius + safety_margin)`.
For each ray, use `pygame.Rect.clipline()` to find the closest intersection.
Group consecutive rays whose free distance is at least the configured minimum,
recast each group midpoint, discard unsafe midpoints, and sort results by
bearing. Select with:

```python
max(gaps, key=lambda gap: (gap.free_distance, -abs(gap.bearing), -gap.bearing))
```

- [x] **Step 5: Run perception tests GREEN**

Run the Task 1 command and require all target, obstacle, and gap tests to pass.

### Task 2: Perception-driven Gap Exploration branch

**Files:**
- Modify: `autonomy_lab/behaviors.py`
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `autonomy_lab/scene_config.py`
- Modify: `tests/test_perception_bt.py`

**Interfaces:**
- Consumes: `AgentPerception.snapshot.best_exploration_gap` and the existing shared command dict.
- Produces: `TraversableGap`, `MoveThroughGap`, `BehaviorTreeController.gap_exploration`, and active action label `Move Through Gap`.

- [x] **Step 1: Write failing Condition and Action tests**

Assert `TraversableGap` succeeds with a snapshot gap, fails without one, never
changes commands, and reports bearing/distance feedback. Assert
`MoveThroughGap` writes a normalized turn from bearing, configured throttle,
and returns RUNNING. Assert `SearchTarget` now writes zero throttle.

- [x] **Step 2: Verify behavior tests RED**

```bash
conda run -n pygame_lab python -m unittest tests.test_perception_bt -v
```

Expected: import failures for the two new nodes or topology assertion failures.

- [x] **Step 3: Implement nodes and real topology**

Add the Condition and Action with `visual_type` metadata and concise
`feedback_message`. Build this real `py_trees` branch:

```text
Gap Exploration (Sequence, memory=False)
├── Traversable Gap?
└── Move Through Gap
```

Insert it after Target Pursuit and before Search Target. Extend
`active_behavior` and `decision_label`. Keep the root reactive. Change the
default `search_throttle` to `0.0` so no-gap fallback scans in place.

- [x] **Step 4: Verify preemption and lifecycle GREEN**

Test target appearance invalidates Move Through Gap and starts Move To Target
in the same tick. Test a new obstacle threat invalidates exploration and returns
Avoid Obstacle's complete command. Run the Task 2 command to GREEN.

### Task 3: Generic visualization, M2, and integration verification

**Files:**
- Modify: `tests/test_bt_visualizer.py`
- Modify: `PROJECT_GUIDE.md`
- Modify: `docs/superpowers/plans/2026-08-17-unknown-target-gap-exploration.md`

**Interfaces:**
- Consumes: the real 11-node tree, generic `visual_type`/feedback, and `active_behavior`.
- Produces: no new runtime framework or log fields.

- [x] **Step 1: Update focused integration assertions**

Assert the definition-driven visualizer discovers 11 nodes, includes the new
Condition/Action types, and same-depth rectangles do not overlap. Extend the M2
integration sequence to count `Move Through Gap -> Move To Target -> Avoid
Obstacle` through the existing transition metric.

- [x] **Step 2: Run focused verification**

```bash
conda run -n pygame_lab python -m unittest tests.test_perception tests.test_perception_bt tests.test_bt_visualizer tests.test_main_lifecycle -v
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
git diff --check
```

Require zero failures and zero compile/diff errors.

- [x] **Step 3: Exercise all three scenes headlessly**

With `SDL_VIDEODRIVER=dummy`, construct every scene, tick its controller, draw
Environment and the BT panel, reset both, and require exit code zero.

- [x] **Step 4: Update milestone documentation and stop**

Add only completed gap perception/exploration facts and known local-navigation
limitations to `PROJECT_GUIDE.md`. Mark this plan complete. Do not add target
memory, visited-space memory, global planning, Gymnasium, or RL.
