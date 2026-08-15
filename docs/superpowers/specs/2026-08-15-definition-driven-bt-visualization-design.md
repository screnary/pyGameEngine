# Definition-driven Behavior Tree Visualization

## Scope

Milestone 1.2 replaces the hard-coded Behavior Tree panel topology with a view
derived directly from the live `py_trees` tree. It keeps the existing Pygame
panel, runtime behavior, controller commands, and Milestone 2 experiment
metrics unchanged.

This milestone does not add a tree editor, dynamic tree generation, new agent
behaviors, external graph libraries, Gymnasium, or reinforcement learning.

## Components

### `behavior_tree.py`

`BehaviorTreeController` remains responsible for building and ticking the real
`py_trees` tree, owning the `SnapshotVisitor`, exposing the current action to
the experiment recorder, and asking the visualizer to draw the panel.

### `behaviors.py`

The three task-specific behavior nodes receive only lightweight display
metadata:

- `ObstacleNear.visual_type = "condition"`
- `AvoidObstacle.visual_type = "action"`
- `MoveToTarget.visual_type = "action"`

This metadata has no execution semantics.

### `bt_visualizer.py`

A single lightweight module will:

1. Traverse the live tree from `tree.root`.
2. Adapt each real node into minimal drawing data containing its reference,
   parent, ordered children, type, depth, and calculated position.
3. Calculate a hierarchical layout without business-node knowledge.
4. Draw parent/child connections and nodes in the existing Pygame panel.
5. Apply runtime colors and active highlighting from real node statuses and
   `SnapshotVisitor` data.

The drawing data is only a view of real nodes. It is not another Behavior Tree
runtime or execution model.

## Topology and Structure Signature

Traversal is depth-first and preserves each parent's `children` order. Each
entry in the structure signature is:

```text
(node.id, parent.id or None, child_order)
```

The visualizer compares the current signature with the cached signature before
drawing. An unchanged signature reuses the existing topology and coordinates.
A changed signature rebuilds the adapted node list and recalculates layout.
This also covers runtime additions, removals, reparenting, and child reordering.

## Node Type Recognition

Native `py_trees` types are detected with `isinstance` in this order:

- Selector
- Sequence
- Parallel
- Decorator

Other behavior nodes use `node.visual_type` when present and otherwise fall
back to `"behaviour"`. No concrete business class or node name is referenced
by the visualizer.

## Layout

The layout is a small post-order hierarchical algorithm:

1. Each leaf receives the next horizontal slot from left to right.
2. A parent is positioned at the midpoint between its first and last child.
3. Vertical position is derived from node depth.
4. Logical positions are scaled into the available panel area with fixed
   margins and simple node-size limits.

This is sufficient for the current tree and modest trees with tens of nodes.
It intentionally provides no zooming, panning, scrolling, overlap solver, or
general graph layout capabilities.

## Runtime State and Active Path

Topology and coordinates are cached separately from runtime state. Each draw
reads the current status directly from the real `py_trees` node.

The current-tick visited set comes from `SnapshotVisitor.visited`. A node is
active when it was visited during the current tick; its incoming connection is
highlighted using the child's actual runtime status. `RUNNING`, `SUCCESS`,
`FAILURE`, and `INVALID` retain the existing color semantics. Reset continues
to clear the visitor and invalidate the real tree, so the panel returns to its
inactive state without rebuilding an unchanged layout.

## Error Handling

An empty root is outside the controller's valid state. Unknown custom node
types render as generic behavior nodes. A topology change is handled on the
next draw through signature comparison; no observer or event system is added.

## Targeted Verification

Verification will remain small and focused:

- Assert traversal returns every real node with correct parents and order.
- Assert native composites and custom condition/action metadata are classified.
- Assert parent/child edges and calculated positions come from the tree.
- Assert repeated synchronization of an unchanged tree does not rebuild.
- Add a temporary node to an in-memory tree and assert the signature, topology,
  edge set, and rebuild count update without visualizer changes.
- Tick and reset the existing controller to verify SnapshotVisitor-driven
  highlighting and statuses.
- Run the Pygame program with the existing BT controller and confirm a normal
  Milestone 2 experiment record is still produced.

