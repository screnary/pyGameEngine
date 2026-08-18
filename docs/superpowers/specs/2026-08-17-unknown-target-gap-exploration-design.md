# Target-aware Gap Navigation and Commitment

## Scope

Extend the local gap prototype so it serves both unknown-target exploration and
obstacle bypass when target coordinates are globally available. A selected safe
gap is followed to a fixed world-space entry waypoint instead of being replaced
on every tick.

This remains local reactive navigation. It adds no map, target memory, visited
space, global planner, SLAM, Gymnasium, or reinforcement learning.

## Gap Perception

`AgentPerception` continues sampling rays across the configured FOV. Obstacles
are inflated by `Agent radius + gap_safety_margin`, and world bounds are
contracted by the same amount. A ray through the inflated geometry is therefore
a safe center path for the circular Agent.

Gap extraction uses relative openness instead of only a fixed distance:

```python
opening_threshold = max(
    gap_min_travel_distance,
    max_sample_distance * gap_open_ratio,
)
```

Consecutive rays meeting this threshold form a candidate. This prevents a short
wall-facing corridor from joining a much longer side opening.

`PerceivedGap` contains:

```text
bearing
free_distance
angular_width
entry_position
```

`entry_position` is an immutable `(x, y)` tuple on the gap center ray at
`free_distance * gap_entry_ratio`. The ratio remains below one so the waypoint
stays inside the observed free corridor.

`PerceptionSnapshot` exposes:

```text
traversable_gaps
best_exploration_gap
target_path_blocked
best_target_gap
```

`best_exploration_gap` prefers the greatest free distance and then the smallest
turn. `best_target_gap` is available only when target information exists, the
target bearing is inside the current FOV, and the radius-safe ray toward the
target is locally blocked. It selects the candidate with the smallest angular
difference from the target bearing, breaking ties by greater free distance.

If the target bearing is outside FOV, `target_path_blocked` is false. The normal
Move To Target action is then allowed to turn the Agent toward the target before
local blockage is assessed.

## Behavior Tree

The real tree becomes:

```text
Priority Selector (memory=False)
├── Obstacle Avoidance
│   ├── Obstacle Threat?
│   └── Avoid Obstacle
├── Target Gap Navigation (Sequence, memory=True)
│   ├── Target Available?
│   ├── Target Path Blocked?
│   ├── Target-aligned Gap?
│   └── Move Through Target Gap
├── Target Pursuit
│   ├── Target Available?
│   └── Move To Target
├── Gap Exploration (Sequence, memory=True)
│   ├── Traversable Gap?
│   └── Move Through Exploration Gap
└── Search Target
```

The reactive root always checks emergency avoidance first. Target Gap Navigation
precedes direct Target Pursuit, so ground-truth target availability can no longer
hide the gap branch.

`TargetPathBlocked` and both gap nodes are Conditions and never write commands.
The target-aligned Condition reads only `best_target_gap`; the exploration
Condition reads only `best_exploration_gap` and therefore does not leak unknown
target direction.

## Gap Commitment

Each Move Through Gap Action captures its Condition's `entry_position` in
`initialise()`. The Action steers toward that fixed absolute point and remains
RUNNING until the Agent is within `gap_entry_reached_distance`, then returns
SUCCESS.

Both gap Sequences use `memory=True`, so their Conditions are not re-run while
the Action is entering the selected opening. This prevents left/right candidate
flicker. Higher-priority root branches still provide the required preemption:

- Emergency Obstacle Avoidance invalidates either gap action immediately.
- A newly available target preempts unknown-target Gap Exploration.
- Target Gap Navigation intentionally completes its short entry commitment even
  if the direct target ray clears part-way through; it re-evaluates after reaching
  the waypoint.

An invalidated gap Action clears only its local waypoint. The shared command is
still reset by the controller before every tree tick, preventing stale commands.

## Configuration

Keep the existing flat Behavior Tree configuration and add:

```python
"gap_open_ratio": 0.85,
"gap_entry_ratio": 0.8,
"gap_entry_reached_distance": 24.0,
"gap_commit_emergency_distance": 4.0,
```

Validation requires `0 < gap_open_ratio <= 1`, `0 < gap_entry_ratio < 1`, a
positive reached distance, and an emergency distance between zero and the normal
avoidance distance. Existing ray count, minimum travel distance, safety
margin, and throttle settings remain unchanged. During a committed gap Action,
Obstacle Threat temporarily uses `gap_commit_emergency_distance` instead of the
normal proactive avoidance distance. This lets a radius-safe path finish while
retaining preemption for an actual near-collision deviation.

## Data Flow

```text
Environment geometry and optional target information
        -> AgentPerception safe rays and gap candidates
        -> target_path_blocked / selected PerceivedGap
        -> BT Conditions
        -> committed Move Through Gap waypoint
        -> existing Environment collision resolution
```

Environment remains authoritative for collision and dynamics. Perception derives
observations. BT selects intent and commands. No visualization data feeds back
into navigation.

## Verification

Focused tests cover:

- the simple-scene wall produces a side opening rather than a forward false gap;
- an opening closed by radius/safety inflation is not selected through its center;
- ground-truth target plus blocked direct ray runs Target Gap Navigation;
- a clear direct ray runs Move To Target;
- a target outside FOV turns through Move To Target rather than falsely reporting
  a blocked path;
- a gap action retains one absolute waypoint while perception candidates change;
- reaching the waypoint returns SUCCESS and allows tree re-evaluation;
- Emergency Avoidance invalidates committed target/exploration gap actions;
- a newly perceived target preempts committed unknown-target exploration;
- existing M2 fields, visualization, reset, and three scenes remain operational.

## Known Limitations

The fixed waypoint is a short local commitment, not route planning. Navigation
can still revisit openings or fail in maze-like layouts. Target-aligned selection
uses only the currently sensed FOV, and no history is kept after completion or
preemption.
