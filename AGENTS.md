# AGENTS.md

This is a **research prototype**, not a production system.

## Working Principles

* Prefer the simplest runnable implementation.
* Implement only the explicitly requested task.
* Do not design for hypothetical future requirements.
* Do not add unrelated features or refactor unrelated code.
* Prefer small modules, plain Python classes, and simple functions.
* Reuse lightweight existing libraries instead of rebuilding infrastructure.
* Keep dependencies and abstractions minimal.

## Preferred Stack

* `pygame` — simulation and visualization
* `numpy` — numerical operations
* `py_trees` — behavior-tree runtime
* `gymnasium` — RL environment interface
* `stable-baselines3` — RL algorithms when needed

Do not implement a custom behavior-tree framework unless explicitly requested.

## Superpowers: FAST Mode by Default

Use Superpowers in **FAST mode** for this project.

FAST mode means:

1. Understand the requested task briefly.
2. Inspect only relevant files.
3. Make the smallest implementation needed.
4. Run targeted tests or the relevant demo.
5. Fix blocking problems.
6. Report the result and stop.

Unless clearly necessary, **do not** perform:

* lengthy brainstorming
* extensive design/specification work
* multi-agent development
* parallel-agent dispatch
* full test-suite execution
* exhaustive TDD
* repeated code-review loops
* unrelated architecture review
* git worktrees or complex branch workflows

Use heavier Superpowers workflows only when the task is genuinely complex, risky, cross-cutting, or explicitly requested by the user.

When uncertain, prefer **FAST mode first**.

## Avoid Unless Explicitly Required

* plugin architecture
* event bus
* dependency injection
* ECS
* database or networking
* service/repository layers
* complex configuration systems
* unnecessary factories, registries, or framework abstractions

## Scope and Stop Condition

Implement only the current requested milestone.

After the requested feature works, **STOP**.

Do not automatically add future features, speculative improvements, or unrelated refactoring.

Priority:

**working experiment > observable behavior > algorithm correctness > architectural elegance**
