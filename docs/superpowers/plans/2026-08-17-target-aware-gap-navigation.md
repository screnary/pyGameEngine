# Target-aware Gap Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route globally known targets through locally safe gaps and commit each gap action to a fixed entry waypoint.

**Architecture:** Perception extracts relatively open radius-safe ray groups, publishes target-path blockage and target-aligned/exploration gap choices, and gives every gap an absolute entry point. The reactive root checks emergency avoidance first, then a committed target-gap branch, direct pursuit, committed unknown-target exploration, and scan fallback.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, standard-library `dataclasses` and `unittest`.

## Global Constraints

- Use Conda environment `pygame_lab`; add no dependency.
- Work inline under project FAST Mode; no worktree or multi-agent execution.
- Add no target memory, map, global planner, Gymnasium, or RL.
- Preserve Environment collision authority and M2 formats.
- Preserve the user's `simple` ground-truth configuration and `.gitignore` edit.

---

### Task 1: Target-aware gap perception and fixed entry points

**Files:**
- Modify: `autonomy_lab/perception.py`
- Modify: `autonomy_lab/scene_config.py`
- Modify: `tests/test_perception.py`

**Interfaces:**
- Produces: `PerceivedGap.entry_position: tuple[float, float]`, `PerceptionSnapshot.target_path_blocked: bool`, and `PerceptionSnapshot.best_target_gap: PerceivedGap | None`.

- [x] Write failing tests proving the simple wall selects a side gap, not bearing zero; its entry point lies beyond the inflated wall edge; clear/out-of-FOV target rays are not blocked; invalid ratios fail fast.
- [x] Run `conda run -n pygame_lab python -m unittest tests.test_perception -v` and verify RED for missing fields/wrong forward gap.
- [x] Add config values `gap_open_ratio=0.85`, `gap_entry_ratio=0.8`, and `gap_entry_reached_distance=24.0` with exact range validation.
- [x] Extract groups using `max(gap_min_travel_distance, max_sample_distance * gap_open_ratio)`, compute immutable entry tuples, derive target blockage only inside FOV, and select the smallest target-bearing gap error.
- [x] Re-run the perception module and require GREEN.

### Task 2: Committed target/exploration gap branches

**Files:**
- Modify: `autonomy_lab/behaviors.py`
- Modify: `autonomy_lab/behavior_tree.py`
- Modify: `tests/test_perception_bt.py`

**Interfaces:**
- Produces: `TargetPathBlocked`, `TargetAlignedGap`, two committed `MoveThroughGap` instances, `target_gap_navigation`, and `gap_exploration` with memory enabled.

- [x] Write failing tests for target-blocked and target-gap Conditions, fixed waypoint capture, SUCCESS near waypoint, INVALID cleanup, ground-truth simple first-tick target-gap navigation, clear direct pursuit, and unknown-target preemption.
- [x] Run `conda run -n pygame_lab python -m unittest tests.test_perception_bt -v` and verify RED.
- [x] Make `TargetAvailable` name-configurable; add the two Conditions without command writes.
- [x] Refactor `MoveThroughGap` to capture `entry_position` in `initialise()`, steer toward the absolute waypoint, finish inside the configured distance, and clear only local state on INVALID.
- [x] Build the approved five-branch root, use distinct gap Action instances, set both gap Sequences to `memory=True`, reduce avoidance to a true emergency distance during commitment, and update active/decision labels.
- [x] Re-run the BT tests and require GREEN, including emergency and target preemption.

### Task 3: Runtime integration and documentation

**Files:**
- Modify: `tests/test_bt_visualizer.py`
- Modify: `PROJECT_GUIDE.md`
- Modify: `docs/superpowers/plans/2026-08-17-target-aware-gap-navigation.md`

- [x] Update the definition-driven visualizer assertions for the real 16-node tree and verify no same-depth overlap.
- [x] Run all four test modules, compileall, and `git diff --check` with zero failures.
- [x] Run fixed-step `simple + ground_truth`; require Target Gap Navigation to activate before forced avoidance and inspect whether the Agent enters the selected waypoint corridor rather than repeating direct-pursuit/avoidance immediately.
- [x] Construct, tick, draw, and reset all three scenes under SDL dummy.
- [x] Record completed behavior and local-navigation limitations in `PROJECT_GUIDE.md`; stop without adding future capabilities.
