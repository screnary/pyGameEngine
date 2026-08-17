# Class Registry and Code Comments Design

## Goal

Replace the per-Behavior factory functions with a direct class registry and a
uniform constructor while preserving the current JSON format, runtime behavior,
visualization, and experiment metrics. Add concise documentation to the project
code so the simulation and BT data flow are easier to understand.

## Scope

This change covers the current Python application and tests. It does not add new
Behavior nodes, change BT topology, change scene behavior, or introduce plugin
discovery, reflection, dependency injection, schema frameworks, RL, or
Gymnasium.

JSON remains `bt-lab/v1`. Existing definitions remain valid without migration.

## Behavior Construction

All task-specific leaf nodes use one constructor shape:

```python
Behavior(context=context, name=name, **params)
```

Three lightweight bases provide the shared runtime contract:

```text
AgentBehaviour
├── AgentCondition   visual_type = "condition"
└── AgentAction      visual_type = "action"
```

`AgentBehaviour` stores the construction context and raw node parameters. It
also provides small helpers for:

- rejecting unknown parameters;
- resolving a JSON parameter before a scene-config fallback;
- validating numeric values;
- resolving a named, already-built Condition dependency.

Individual Behavior classes still own their task-specific parameter names and
execution logic. This avoids moving business knowledge into the Loader or
creating a general parameter schema system.

## Build Context and References

`BehaviorBuildContext` continues to carry only the runtime dependencies needed
by the current nodes:

```text
perception
command
behavior_config
nodes_by_name
```

Action nodes such as `AvoidObstacle` and `MoveThroughGap` resolve their
`params.condition` reference from `nodes_by_name`. The referenced Condition
must already have been built, so JSON definitions continue to place the
Condition before its dependent Action.

## Registry and Loader

The registry becomes a direct mapping from JSON Behavior name to Python class:

```python
BEHAVIOR_REGISTRY = {
    "ObstacleThreat": ObstacleThreat,
    "AvoidObstacle": AvoidObstacle,
    "MoveToTarget": MoveToTarget,
}
```

The Loader remains responsible for JSON structure and composite construction.
For leaf nodes it performs one generic operation:

```python
behavior_class = registry[behavior_name]
node = behavior_class(context=context, name=node_name, **params)
```

It continues to wrap construction errors with the JSON node path and verifies
that the created node's `visual_type` matches the declared `condition` or
`action` type.

## Commenting Strategy

Code documentation uses concise English Docstrings for modules, public classes,
and non-obvious methods. Chinese inline comments explain difficult control flow
and algorithms, especially:

- the Pygame main-loop and Episode lifecycle;
- BT construction, tick ordering, command clearing, and gap commitment;
- collision resolution;
- target visibility, ray clearance, and traversable-gap grouping;
- definition-driven Visualizer extraction and layout;
- experiment metric accumulation and legacy CSV migration;
- optional asset fallback and heading-based sprite rotation.

Obvious assignments, simple loops, and direct drawing calls are not annotated.
Tests receive comments or Docstrings only where they clarify fixture intent or a
non-obvious regression. JSON files cannot contain comments and remain unchanged.

## Error Handling

Existing clear failures remain available for unknown Behavior names, invalid
node types, missing children, duplicate names, invalid references, unknown node
parameters, and non-numeric values. Direct class construction must not expose a
less informative error than the current factory-based implementation.

## Verification

Targeted tests will verify:

1. every registry value is a Behavior class rather than a per-node factory;
2. the Loader builds the unchanged 16-node default tree through the generic
   constructor;
3. JSON overrides and scene-config fallbacks remain unchanged;
4. a registered test Behavior class can be added without Loader changes;
5. invalid parameters and Condition references retain clear messages;
6. current perception, controller, Visualizer, lifecycle, and experiment tests
   continue to pass;
7. all three existing scenarios can construct, tick, draw, and reset.

The implementation stops after this refactor and documentation pass.
