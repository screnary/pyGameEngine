# JSON Behavior Tree Definition v1 Design

## Goal

Move the current Behavior Tree topology out of `BehaviorTreeController` and
into a lightweight JSON definition while retaining `py_trees` as the only
runtime and preserving current Agent behavior, visualization, and experiment
metrics.

## Data Flow

```text
bt_configs/default.json
    -> bt_loader.py
    -> behavior_registry.py
    -> py_trees runtime root
    -> BTVisualizer / ExperimentRecorder
```

The JSON controls topology, priority, node names, composite memory, and small
node parameters. Python Behavior classes continue to own node execution.

## BT Definition v1

Every file contains:

```json
{
  "format": "bt-lab/v1",
  "id": "default_bt",
  "name": "Default Agent Behavior",
  "root": {}
}
```

Supported node types are exactly `selector`, `sequence`, `condition`, and
`action`. Node fields are limited to `type`, `name`, `behavior`, `memory`,
`params`, and `children`.

Composite nodes require a non-empty `children` list and accept Boolean
`memory`. Leaf nodes require a registered `behavior`. Node names must be unique
within one definition because action parameters can refer to a previously built
Condition by name:

```json
{
  "type": "action",
  "name": "Avoid Obstacle",
  "behavior": "AvoidObstacle",
  "params": {"condition": "Obstacle Threat?"}
}
```

The default definition migrates the current real 16-node tree exactly. No
Boundary Behavior or other new capability is added.

## Behavior Registry

`behavior_registry.py` exposes one explicit `BEHAVIOR_REGISTRY` dictionary.
Each value is a small factory callable rather than a discovered plugin or a
reflection-based constructor. This is necessary because the current Behavior
constructors require runtime objects such as perception, command output, scene
parameters, and sometimes a previously built Condition.

Factories receive a plain build context, the JSON node name, and `params`.
JSON parameters override matching values from
`scene_config["behavior_tree"]`; omitted parameters fall back to the scene
configuration. Unknown referenced Conditions produce a clear error.

Adding another instance of an already registered Behavior changes only JSON.
Adding a new Python Behavior implementation requires one explicit registry
entry, but no Loader or Visualizer changes.

## Loader

`bt_loader.py` uses ordinary `json` and recursive functions. It validates the
format identifier, top-level metadata, supported node type, node name,
composite children, leaf behavior, params object, memory type, unique name, and
registry lookup. Errors include the JSON node path and a direct explanation.

The Loader returns a small result containing:

- `config_id`
- definition name
- real `py_trees` root
- nodes indexed by JSON name

The Visualizer receives only the real runtime root and remains unaware of JSON.

## Controller and CLI

`BehaviorTreeController(environment, bt_config="default")` loads the selected
definition. `main.py` adds `--bt default`; paths resolve from the project root,
not the current working directory.

The Controller performs only runtime housekeeping: perception update, command
reset, Action `dt`, committed-gap emergency thresholds, tree tick/reset, and
panel summaries. It derives relevant action/condition collections from loaded
Python node types instead of constructing topology.

## Experiment Log

`ExperimentRecorder.start_episode` accepts `bt_config_id`. Detailed JSON and
CSV summaries include that field without changing existing metric meanings.
Manual control records `null`/blank. If an existing CSV has the previous known
header, it is rewritten once with an empty `bt_config_id` column so historical
rows remain usable.

## Verification

- Load `default.json` and reproduce the current 16-node topology and first-tick
  command.
- Reorder JSON children and confirm the runtime root and Visualizer connection
  order both follow the definition.
- Add another existing Behavior instance and build it without Loader or
  Visualizer changes.
- Reject invalid type, missing children, unknown behavior, duplicate name, and
  invalid Condition reference with clear errors.
- Confirm `--bt default`, M2 JSON/CSV `bt_config_id`, reset, visualization, and
  all three scenarios still work.

## Explicit Non-goals

No XML, YAML, schema framework, BehaviorTree.CPP compatibility, ports,
SubTree, Decorator, Parallel, blackboard redesign, editor, hot reload, plugin
discovery, automatic generation, evolution, RL, or Gymnasium support is added.
