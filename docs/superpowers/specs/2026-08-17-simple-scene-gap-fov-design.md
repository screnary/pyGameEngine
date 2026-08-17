# Simple Scene Gap FOV Design

## Goal

Allow the Agent to perceive the physically traversable 70-pixel channel in the
`simple` scene when the required gap bearing is slightly outside the current
120-degree sensor field of view.

## Approved Change

- Override only the `simple` scene sensor FOV from 120 to 140 degrees.
- Keep Agent radius 16, gap safety margin 8, ray count 31, minimum travel
  distance 100, and open ratio 0.85 unchanged.
- Keep the default sensor FOV and the other two scenarios unchanged.
- Do not change gap extraction, Behavior Tree topology, collision handling, or
  navigation actions.

## Verification

At the observed runtime position `(674.7, 439.9)` and heading `-26.7` degrees,
the direct target path remains blocked, but the simple-scene configuration must
produce a target-aligned gap with about 300 pixels of free distance. Existing
perception, BT, visualizer, and lifecycle tests must remain green.

## Limitation

This is a scene-level perception adjustment, not a global planner. Multi-turn
passages can still be missed by the local straight-ray gap model.
