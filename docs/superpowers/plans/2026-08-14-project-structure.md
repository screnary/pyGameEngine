# Project Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This project's FAST Mode explicitly prohibits multi-agent execution.

**Goal:** Move the Milestone 0 domain code into the `autonomy_lab` package without changing runtime behavior.

**Architecture:** Keep `main.py` as the root launcher. Place Agent and Environment responsibilities in a flat `autonomy_lab` package, using explicit absolute and relative imports. Add no future-facing packages until a milestone needs them.

**Tech Stack:** Python 3.11, Pygame 2.6.1, Conda environment `pygame_lab`

## Global Constraints

- Keep the project a lightweight research prototype; do not adopt a `src/` layout.
- Do not implement Behavior Tree, RL, Gymnasium, editors, plugins, or future milestone features.
- Do not create empty `controllers/`, `behaviors/`, `rl/`, or `experiments/` directories.
- Preserve the existing `.vscode/` directory without modification.
- Run every Python command through the `pygame_lab` Conda environment.
- Preserve all Milestone 0 controls, collision handling, reset behavior, and rendering.

---

### Task 1: Move domain modules into `autonomy_lab`

**Files:**
- Create: `autonomy_lab/__init__.py`
- Move: `agent.py` -> `autonomy_lab/agent.py`
- Move: `environment.py` -> `autonomy_lab/environment.py`
- Modify: `autonomy_lab/environment.py:3`
- Modify: `main.py:3`

**Interfaces:**
- Consumes: existing `Agent` and `Environment` classes without signature changes
- Produces: `autonomy_lab.agent.Agent` and `autonomy_lab.environment.Environment`

- [x] **Step 1: Verify the target package does not exist yet**

Run:

```powershell
conda run -n pygame_lab python -c "import autonomy_lab.environment"
```

Expected: FAIL with `ModuleNotFoundError` before the package is created.

- [x] **Step 2: Create the package and move the modules**

Create `autonomy_lab/__init__.py` with only:

```python
"""Two-dimensional autonomous-agent research prototype."""
```

Move the complete contents of root `agent.py` to `autonomy_lab/agent.py` without
behavior changes. Move the complete contents of root `environment.py` to
`autonomy_lab/environment.py`, changing only its Agent import to:

```python
from .agent import Agent
```

Delete the two original root modules after their contents are present in the package.

- [x] **Step 3: Update the launcher import**

In `main.py`, replace the existing Environment import with:

```python
from autonomy_lab.environment import Environment
```

- [x] **Step 4: Verify imports and compilation**

Run:

```powershell
conda run -n pygame_lab python -m py_compile main.py autonomy_lab/__init__.py autonomy_lab/agent.py autonomy_lab/environment.py
conda run -n pygame_lab python -c "from autonomy_lab.agent import Agent; from autonomy_lab.environment import Environment; print(Agent.__name__, Environment.__name__)"
```

Expected: both commands exit 0, and the second prints `Agent Environment`.

### Task 2: Document and verify the reorganized prototype

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: root `main.py` and the `pygame_lab` Conda environment
- Produces: documented direct-run command `conda run -n pygame_lab python main.py`

- [x] **Step 1: Replace the minimal README with current project instructions**

Use this content:

````markdown
# Autonomy Lab

A lightweight Pygame research prototype for two-dimensional autonomous-agent experiments.

## Current structure

- `main.py` — direct launcher
- `autonomy_lab/agent.py` — agent state, controls, and rendering
- `autonomy_lab/environment.py` — scene state, collision, reset, and rendering

## Run

```powershell
conda run -n pygame_lab python main.py
```

Controls: `W/S` or Up/Down to move, `A/D` or Left/Right to turn, and `R` to reset.
````

- [x] **Step 2: Verify control, collision, and reset paths**

Run:

```powershell
conda run -n pygame_lab python -c "from collections import defaultdict; import pygame; from autonomy_lab.environment import Environment; keys=defaultdict(bool); keys[pygame.K_w]=True; e=Environment(); start=e.agent.position.copy(); e.update(0.1, keys); assert e.agent.position.x > start.x; e.agent.position=pygame.Vector2(233,350); e._move_agent(pygame.Vector2(10,0)); assert e.agent.position==pygame.Vector2(233,350); e.reset(); assert e.agent.position==pygame.Vector2(100,350); print('control, collision, reset: ok')"
```

Expected: exit 0 with `control, collision, reset: ok`.

- [x] **Step 3: Smoke-test the Pygame main loop**

Set `SDL_VIDEODRIVER=dummy`, schedule one `pygame.QUIT` event, call `main.main()`,
and remove the environment variable afterward:

```powershell
$env:SDL_VIDEODRIVER='dummy'
conda run -n pygame_lab python -c "import threading, pygame, main; threading.Timer(0.5, lambda: pygame.event.post(pygame.event.Event(pygame.QUIT))).start(); main.main(); print('pygame main loop: ok')"
Remove-Item Env:SDL_VIDEODRIVER
```

Expected: exit 0 with `pygame main loop: ok`.

- [x] **Step 4: Review the final scope**

Run:

```powershell
rg --files -g '!__pycache__/**'
git diff --check
git status --short
```

Expected: the source tree matches the approved design, there are no whitespace errors,
and `.vscode/` remains unmodified and untracked.
