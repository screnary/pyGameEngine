# R0.2 Semantic Perception Design

## Goal

Make `SemanticPerception` the single output of the existing geometric
perception computation while preserving the frozen M4/M5 13-D observation and
current BT behavior.

## Data Model

`autonomy_lab/semantic_perception.py` contains simulator-neutral frozen
dataclasses:

- `AgentState`: speed and heading.
- `GoalPerception`: sensed/visible/available state, source, distance, bearing,
  and unavailable reason. It contains no path-traversability meaning.
- `HazardObservation`: clearance and bearing only, with a read-only legacy
  `distance` alias.
- `SectorRange`: bearing and footprint-aware clearance.
- `NavigationGap`: pure numeric local-gap data.
- `HazardPerception`: visible hazards, nearest hazard data, sector ranges,
  local gaps, and goal-direction free-space results.
- `BoundaryPerception`: left/right/top/bottom clearance.
- `SemanticPerception`: groups Agent, Goal, Hazard, and Boundary.

No semantic object contains `pygame`, World, geometry objects, or mutable
simulation state.

## Data Flow and Compatibility

`AgentPerception.update()` performs the existing geometry computation once and
constructs one `SemanticPerception`. Read-only properties on that object map
historical names such as `target_visible`, `nearest_obstacle`, and
`sector_clearances` to nested semantic values; they never recompute perception.

The legacy 13-D builder reads the semantic Agent/Goal/Hazard/Boundary fields and
keeps its existing shape, order, normalization, neutral values, and `float32`
dtype. Existing scenario/result/checkpoint names remain unchanged.

Goal range/FOV/optical-LOS sensing remains independent from footprint-aware
Hazard/free-space clearance. No planner, memory, new Observation, training, or
R0.3 functionality is added.

## Verification

Tests cover semantic construction, Goal/Hazard separation, sector and boundary
propagation, read-only legacy mappings, deterministic 13-D equivalence, both
R0.1 regression scenarios, existing BT/Gym/Hybrid behavior, and diff hygiene.
