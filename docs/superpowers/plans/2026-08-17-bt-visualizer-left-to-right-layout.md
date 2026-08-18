# BT Visualizer Left-to-Right Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the complete 16-node Behavior Tree left to right with compact, two-line node labels inside the existing panel.

**Architecture:** Transpose the current leaf-order tree layout so depth maps to horizontal columns and leaf order maps to vertical rows. Keep the real `py_trees` topology and runtime styling, while adding a small pixel-width text wrapper for at most two name lines.

**Tech Stack:** Python 3.11, pygame 2.6.1, py_trees 2.5.0, `unittest`.

## Global Constraints

- Use the `pygame_lab` Conda environment and add no dependency.
- Work inline under project FAST Mode; do not create a worktree or use multi-agent execution.
- Keep `PANEL_WIDTH = 480` and the current window dimensions.
- Do not change Behavior Tree topology, controller behavior, perception, navigation, or Environment logic.
- Preserve the user's existing `.gitignore` modification.

---

### Task 1: Left-to-right topology layout

**Files:**
- Modify: `tests/test_bt_visualizer.py`
- Modify: `autonomy_lab/bt_visualizer.py`

**Interfaces:**
- Consumes: `BTVisualizer.sync(panel_rect: pygame.Rect)` and `VisualNode.depth`.
- Produces: left-to-right `VisualNode.x/y` positions and right-to-left connection endpoints.

- [x] **Step 1: Write failing layout tests**

Update the topology test to require increasing `x` with depth, and update the
controller test to require all 16 node rectangles to remain inside the layout
panel without any pair overlapping.

```python
self.assertLess(nodes["Root"].x, nodes["Branch"].x)
self.assertLess(nodes["Branch"].x, nodes["Condition"].x)

for rect in rectangles:
    self.assertTrue(layout_panel.contains(rect))
for index, rect in enumerate(rectangles):
    for other in rectangles[index + 1:]:
        self.assertFalse(rect.colliderect(other))
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```text
conda run -n pygame_lab python -m unittest tests.test_bt_visualizer -v
```

Expected: failure because the current layout maps depth to `y`, and the
16-node layout does not satisfy the new orientation assertions.

- [x] **Step 3: Transpose layout and connections**

In `_calculate_layout`, assign sequential logical vertical positions to leaves,
center parents between their first and last children, map depth to `x`, and
derive compact node heights from same-column vertical spacing. In
`_draw_connection`, connect `parent_rect.midright` to `child_rect.midleft` and
route through a midpoint `x`.

```python
x = panel_rect.left + horizontal_margin + round(
    visual.depth * horizontal_span / max_depth
)
y = panel_rect.top + vertical_margin + round(
    logical_y[node_id] * vertical_span / (leaf_count - 1)
)

start = parent_rect.midright
end = child_rect.midleft
midpoint_x = (start[0] + end[0]) // 2
```

- [x] **Step 4: Run the visualizer tests and require GREEN for layout**

Run the same module and require the orientation, bounds, connections, and
non-overlap assertions to pass.

### Task 2: Compact two-line labels

**Files:**
- Modify: `tests/test_bt_visualizer.py`
- Modify: `autonomy_lab/bt_visualizer.py`

**Interfaces:**
- Produces: `BTVisualizer._wrap_text(text, font, max_width, max_lines=2) -> list[str]`.

- [x] **Step 1: Write a failing long-label wrapping test**

```python
font = pygame.font.Font(None, 12)
lines = BTVisualizer._wrap_text(
    "Move Through Exploration Gap", font, 120, max_lines=2
)
self.assertLessEqual(len(lines), 2)
self.assertEqual(" ".join(lines), "Move Through Exploration Gap")
self.assertTrue(all(font.size(line)[0] <= 120 for line in lines))
```

- [x] **Step 2: Run the wrapping test and verify RED**

Run the visualizer module and require failure because `_wrap_text` does not yet
exist.

- [x] **Step 3: Implement compact rendering**

Add a word-boundary, pixel-width wrapper with character fallback for a word
wider than the node. Reduce cached fonts to 16 and 12 pixels. Render type, up to
two centered name lines, and status; render feedback only when the node height
has spare room.

- [x] **Step 4: Run focused and regression verification**

Run:

```text
conda run -n pygame_lab python -m unittest tests.test_bt_visualizer -v
conda run -n pygame_lab python -m compileall -q autonomy_lab tests
git diff --check
```

Expected: all visualizer tests pass, compilation succeeds, and the diff check
reports no whitespace errors.
