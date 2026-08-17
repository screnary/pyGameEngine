# Unknown-target Gap Exploration

## Scope

Extend the current perception-aware Behavior Tree so an Agent with no target
information explores through locally traversable openings instead of following
one constant circular command. The feature remains a local reactive prototype,
not a map, global planner, SLAM system, or target-memory implementation.

Target information modes keep their existing meaning. Gap selection must never
read Target Ground Truth in `perceived` mode.

## Local Gap Perception

`AgentPerception` samples a small fixed number of rays across the configured
Agent FOV. For each ray it calculates the free distance before the Agent center
would reach an obstacle or the safe world boundary.

Before intersection checks, obstacle rectangles are inflated by:

```text
Agent radius + configured safety margin
```

The world is contracted by the same clearance. This converts the circular Agent
into a point for the local ray checks and rejects openings narrower than the
configured safe diameter.

Consecutive rays whose free distance meets the configured minimum travel
distance form one local gap. A `PerceivedGap` contains:

```text
bearing
free_distance
angular_width
```

Each gap bearing is the midpoint of its angular interval. The selected
`best_exploration_gap` has the greatest free distance at that midpoint; ties
prefer the smaller absolute bearing and then the leftmost bearing for a stable
result. An empty result means the current FOV contains no safe local opening.

`PerceptionSnapshot` gains the detected gap tuple and selected best gap. Existing
target and obstacle fields remain unchanged.

## Behavior Tree

The real tree becomes:

```text
Priority Selector (memory=False)
├── Obstacle Avoidance
│   ├── Obstacle Threat?
│   └── Avoid Obstacle
├── Target Pursuit
│   ├── Target Available?
│   └── Move To Target
├── Gap Exploration (memory=False)
│   ├── Traversable Gap?
│   └── Move Through Gap
└── Search Target
```

`TraversableGap` is a Condition. It reads only the current PerceptionSnapshot,
selects no world objects, writes no command, and reports concise feedback.

`MoveThroughGap` is a reactive Action. It steers from the selected relative
bearing, uses a conservative exploration throttle, returns RUNNING, and writes
the complete turn/throttle command every tick.

Target Pursuit remains above Gap Exploration, so a newly visible target
immediately preempts exploration. Existing Obstacle Avoidance remains highest
priority. If no gap is available, Search Target rotates in place so subsequent
ticks can inspect a different FOV without tracing another closed movement circle.

## Configuration

Add only small Behavior Tree settings following the existing flat configuration:

```python
"gap_ray_count": 31,
"gap_min_travel_distance": 100.0,
"gap_safety_margin": 8.0,
"gap_throttle": 0.5,
```

The existing sensor range and FOV define the maximum local sensing region. The
ray count must be at least three, minimum travel distance must be positive, and
safety margin must be non-negative. Invalid values fail fast during controller
construction rather than silently changing navigation behavior.

## Data Flow and Ownership

```text
Environment rectangles and bounds
        -> AgentPerception ray sampling
        -> PerceptionSnapshot gaps
        -> Traversable Gap?
        -> Move Through Gap command
        -> existing Environment collision resolution
```

Environment remains the authority for collision and dynamics. Perception only
derives local observations. BT nodes select behavior and issue commands. This
increment uses existing generic BT feedback to show the selected bearing and
distance; it adds no separate gap overlay or visualization dependency.

## Verification

Focused tests cover:

- open forward space selects a near-forward gap;
- a sufficiently wide opening between inflated rectangles remains traversable;
- an opening narrower than the Agent safe diameter is rejected;
- world boundaries limit ray distance and are never treated as exits;
- target appearance preempts a running Move Through Gap action;
- an imminent obstacle still preempts exploration;
- no-gap fallback rotates in place;
- existing target Range/FOV/LOS, M2 recording, reset, scenes, and visualization
  continue to work.

## Known Limitations

This is local, memoryless free-space steering. It may revisit areas, oscillate in
symmetric layouts, or fail in maze-like environments requiring a global route.
It does not rank gaps by an unknown target direction, build a map, remember
visited gaps, or guarantee complete environment coverage.
