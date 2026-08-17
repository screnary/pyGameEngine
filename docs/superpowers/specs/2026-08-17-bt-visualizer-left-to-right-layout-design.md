# BT Visualizer Left-to-Right Layout Design

## Goal

Display the complete 16-node Behavior Tree legibly inside the existing
480-pixel-wide panel without changing controller behavior or window size.

## Approved Direction

- Draw the tree from left to right: tree depth maps to columns.
- Place ordered leaf nodes from top to bottom and center each parent over its
  first and last child.
- Use smaller node text and wrap node names onto at most two lines.
- Keep the current panel width, runtime colors, node shapes, connections,
  summary text, and legend.

## Layout

The existing leaf-order layout algorithm is transposed:

1. Assign every leaf a sequential logical vertical position.
2. Place a parent halfway between its first and last child.
3. Map node depth to evenly spaced horizontal columns.
4. Limit node width to the available column spacing and node height to the
   available same-column row spacing.

This preserves real `py_trees` child ordering and keeps related nodes visually
grouped. The 16-node tree currently has three depth levels, so each node retains
enough horizontal space for meaningful labels.

## Text Rendering

- Use a compact font for all node text.
- Wrap names by measured pixel width, preferring word boundaries.
- Render at most two name lines and apply an ellipsis only if the second line
  still cannot contain the remaining text.
- Keep runtime status visible through border/fill colors and a short status
  label.
- Omit per-node feedback text when the compact node height cannot contain it;
  the existing panel-level Decision summary remains available.

## Scope

Change only `BTVisualizer` layout and rendering plus focused visualizer tests.
Do not change Behavior Tree topology, behavior execution, perception,
navigation, the Environment, panel width, or overall window dimensions.

## Verification

- Test that depth increases from left to right.
- Test that sibling/leaf order increases from top to bottom.
- Test that representative long node names wrap to no more than two lines.
- Verify all 16 node rectangles stay inside the layout panel and no nodes in
  the same depth column overlap.
- Run the visualizer test module in the `pygame_lab` Conda environment.

## Limitation

If a future tree adds substantially more leaves than the panel height can hold,
labels may again need abbreviation or a separate scrolling/zoom feature. Those
features are outside this task.
