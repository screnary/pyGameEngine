# Success Window Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the Pygame window open after target success while finishing each Episode exactly once.

**Architecture:** Add one `episode_finished` flag to the existing `main.py` loop. The flag gates termination recording and simulation updates; `R` resets it when starting the next Episode.

**Tech Stack:** Python 3.11, pygame, standard-library `unittest` and `unittest.mock`.

## Global Constraints

- Use Conda environment `pygame_lab`.
- Modify only the main-loop lifecycle and targeted tests.
- Do not change BT, Environment dynamics, Agent motion, or M2 metric definitions.
- Timeout continues to exit automatically.

---

### Task 1: Freeze successful Episode until close or reset

**Files:**
- Modify: `main.py`
- Create: `tests/test_main_lifecycle.py`

**Interfaces:**
- Consumes: current Pygame events and `ExperimentRecorder.finish_episode()`.
- Produces: an internal `episode_finished: bool` controlling loop updates.

- [ ] **Step 1: Write failing lifecycle tests**

Run the real `main.main()` with SDL dummy video, a scene whose target starts at
the Agent position, a temporary real recorder, and deterministic event
sequences. Verify that a close event is processed after success and produces
only one SUCCESS record. Verify that `R` after success starts a new Episode and
does not overwrite or interrupt the completed Episode.

- [ ] **Step 2: Verify RED**

```bash
conda run -n pygame_lab python -m unittest tests.test_main_lifecycle -v
```

Expected: the loop exits immediately on first success, so the later close/reset
event is never processed.

- [ ] **Step 3: Implement minimal lifecycle flag**

Initialize `episode_finished = False`. On QUIT, record interruption only when
the Episode is active. On `R`, finish only an active Episode, reset runtime
objects, start a fresh Episode, and clear the flag. Skip controller,
environment, recorder, and termination updates while finished. On SUCCESS,
save once and set the flag instead of setting `running = False`.

- [ ] **Step 4: Verify GREEN and regression behavior**

```bash
conda run -n pygame_lab python -m unittest tests.test_main_lifecycle tests.test_bt_visualizer -v
conda run -n pygame_lab python -m compileall -q main.py autonomy_lab tests
```

Expected: lifecycle and BT visualization tests pass; compilation succeeds.

- [ ] **Step 5: Run the real application path and final checks**

Use a short event-driven dummy-video run to confirm SUCCESS remains open until
QUIT without duplicating JSON/CSV rows. Run `git diff --check` and stop without
adding unrelated features.

