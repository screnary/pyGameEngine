# Perception-aware Behavior Tree Control

## Scope

Milestone 1.3 separates Environment ground truth from information available to
the Behavior Tree. The BT will consume one lightweight perception snapshot,
while rendering retains access to the full scene for research observation.

This milestone does not add target memory, a world model, Blackboard, sensor
noise, path planning, SLAM, Gymnasium, reinforcement learning, threads, or a
general sensor framework. Existing M2 metric definitions and log formats remain
unchanged.

## Runtime Data Flow

```text
Environment ground truth
        -> AgentPerception.update()
        -> PerceptionSnapshot
        -> Behavior Tree conditions/actions
        -> turn/throttle command
        -> Environment dynamics
```

The existing frame-synchronous runtime is retained. Simulation, rendering, and
BT ticking remain approximately 60 Hz. `AgentPerception.update()` runs at the
start of every controller tick, so each BT evaluation reads the newest scene
state without threads or a separate accumulator.

## Configuration

Each scene receives the same lightweight defaults:

```python
"target_information_mode": "perceived",
"sensor": {
    "range": 300.0,
    "fov_degrees": 120.0,
    "los_enabled": True,
},
```

`target_information_mode` accepts:

- `"perceived"`: target direction and distance are available only when the
  target satisfies Range, FOV, and LOS checks.
- `"ground_truth"`: exact target direction and distance are always available
  as mission/global information. Sensor visibility is still calculated and
  reported independently for comparison.

Existing scenes default to `"perceived"`. Changing this single scene field is
sufficient for a ground-truth baseline experiment.

Behavior-tree configuration also contains the existing avoidance parameters
and small Search Target commands:

```python
"search_throttle": 0.25,
"search_turn": 0.25,
```

These low positive values produce a slow forward arc.

## Perception Model

`autonomy_lab/perception.py` contains two small dataclasses and one plain class.

### `PerceivedObstacle`

Contains the detected rectangle reference, clearance distance from the Agent,
and relative bearing.

### `PerceptionSnapshot`

Contains:

```text
target_visible
target_available
target_source              # ground_truth / perception / None
target_distance            # None when unavailable
target_bearing             # None when unavailable
target_unavailable_reason  # concise feedback reason
visible_obstacles
nearest_obstacle
```

No target position or direction is exposed in perceived mode while the target
is unavailable.

### `AgentPerception`

Owns the current snapshot and derives a new snapshot from the Environment each
tick. It validates `target_information_mode` once and otherwise remains a
direct, synchronous calculator.

## Target Visibility

Target center distance must be no greater than `sensor.range`. Relative bearing
is normalized to `[-pi, +pi]` and must satisfy:

```text
abs(relative_bearing) <= radians(fov_degrees) / 2
```

When LOS is enabled, the segment from Agent center to Target center is tested
against each obstacle using `pygame.Rect.clipline()`. Any intersection blocks
visibility. The snapshot records one short reason: `out of range`, `outside
FOV`, or `occluded`.

In ground-truth mode, these checks still determine `target_visible`, but target
distance and bearing remain available with source `ground_truth`.

## Obstacle Perception

For every obstacle, perception calculates the closest point on its rectangle to
the Agent center. Clearance subtracts the Agent radius. The obstacle is visible
when that closest point is within Sensor Range and its relative bearing is
inside the FOV. Visible obstacles are ordered by clearance, and the first is
also exposed as `nearest_obstacle`.

The first version does not model obstacle-to-obstacle occlusion or partial
rectangle coverage beyond this closest-point rule.

## Behavior Tree Definition

The real `py_trees` definition becomes:

```text
Priority Selector (memory=False)
├── Obstacle Avoidance (Sequence, memory=False)
│   ├── Obstacle Threat? (Condition)
│   └── Avoid Obstacle (Action)
├── Target Pursuit (Sequence, memory=False)
│   ├── Target Available? (Condition)
│   └── Move To Target (Action)
└── Search Target (Action)
```

Both Sequences are reactive so their conditions are checked every tick.

### Conditions

`ObstacleThreat` reads only perceived obstacles and succeeds when the nearest
relevant obstacle is within the configured avoidance distance and forward
bearing. It retains the selected `PerceivedObstacle` for `AvoidObstacle`.

`TargetAvailable` reads only `PerceptionSnapshot.target_available`. Its feedback
distinguishes ground truth, sensor-visible target, and the concise unavailable
reason.

Conditions do not write controller commands.

### Actions

`MoveToTarget` reads only perceived/available target bearing and distance. It
uses relative bearing for turn strength, reduces throttle for large bearing
errors, stops within the configured reached distance, and never reads the
Environment target.

`SearchTarget` emits configured low turn and throttle commands and remains
RUNNING. This is a simple moving scan, not a coverage planner.

`AvoidObstacle` selects a turn direction from the perceived obstacle bearing,
emits its existing timed avoidance command, and does not scan all Environment
obstacles.

## Preemption and Action Lifecycle

The root Selector remains reactive. A new obstacle threat can therefore
invalidate a running Target Pursuit or Search Target action during the same BT
tick.

The controller clears the shared command before each tree tick. The action
chosen during that tick writes the complete new command, so a previous action's
command cannot persist. Action `terminate()` methods clear only local temporary
state when needed; they do not clear the shared command, because an invalidated
lower-priority action could otherwise erase a higher-priority action's command
written earlier in the same tick.

## Feedback and Visualization

Conditions and actions set short `py_trees.feedback_message` strings containing
the current reason or command, such as:

```text
visible: 183 px, +24 deg
outside FOV
threat: 48 px, -17 deg
search arc
```

The existing definition-driven visualizer requires no business-node layout
changes. A generic renderer addition displays truncated feedback only for
current-tick visited nodes when space permits.

The Environment draws a low-opacity FOV sector from the Agent heading using the
configured range and angle. This visualization never participates in perception
calculations, and the Ground Truth Target remains visible to the researcher.

## Experiment Recording

The controller's active action now includes `SearchTarget`, `MoveToTarget`, and
`AvoidObstacle`. The existing ExperimentRecorder continues to count transitions
whenever this active action changes. Elapsed time, path length, collisions,
trajectory, BT tick count, JSON fields, and CSV fields are unchanged.

## Targeted Verification

Focused tests will verify:

- Range, FOV, and angle wrapping around `+/-180` degrees.
- Target ahead, behind, outside range, occluded, and entering FOV.
- Different availability results for `perceived` and `ground_truth` modes.
- Obstacle visibility and Obstacle Threat decisions from snapshot data.
- Search Target commands and Search-to-Pursuit switching.
- Pursuit preemption by avoidance and no stale command after preemption.
- Obstacle-cleared re-evaluation.
- Feedback messages matching decisions.
- Automatic visualizer topology expansion without business-name layout logic.
- Reset, existing assets/scenes, and M2 JSON/CSV recording remain operational.

## Known Simplifications

Perception is exact and instantaneous inside its configured limits. It has no
noise, latency, target recognition uncertainty, target memory, obstacle
occlusion model, or map. Search Target follows one fixed slow arc and does not
guarantee efficient discovery in every obstacle arrangement.
