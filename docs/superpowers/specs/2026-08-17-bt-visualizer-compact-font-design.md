# BT Visualizer Compact Font Design

## Goal

Reduce text crowding and clipping inside the existing Behavior Tree nodes after
the tree expanded to 16 nodes.

## Approved Scope

- Keep the current top-down layout and 480-pixel panel width.
- Reduce the visualizer's normal and compact node font sizes.
- Keep the existing name truncation behavior for labels that still cannot fit.
- Do not change Behavior Tree topology, runtime behavior, node geometry, panel
  width, or window size.

## Implementation

Change only the cached fonts created by `BTVisualizer`. The normal node font
will be reduced from 20 to 16 pixels, and the compact-node font from 16 to 12
pixels. Existing status, type, name, and feedback rendering continues to use
the same selection logic.

## Verification

- Add a focused test for the two visualizer font sizes.
- Run the visualizer test module in the `pygame_lab` Conda environment.
- Confirm the existing 16-node tree still renders without same-depth node
  overlap.

## Limitation

Very long labels in extremely narrow nodes remain abbreviated. Fully displaying
every label would require a layout or panel-width change, which was explicitly
not selected for this task.
