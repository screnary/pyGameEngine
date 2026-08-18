"""读取、校验 BT JSON，并递归构建真正的 py_trees Runtime Tree。

Loader 只理解 selector、sequence、condition 和 action 的结构语义。叶节点
具体如何执行、需要哪些运行对象，由 Registry 选出的 Behavior 类自己负责。
"""

from dataclasses import dataclass
import json
from pathlib import Path

import py_trees

from .registry import (
    BEHAVIOR_REGISTRY,
    BehaviorClass,
)
from .context import BehaviorBuildContext


# 配置路径以源码位置为基准，因此从任意 working directory 启动都一致。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BT_CONFIG_DIR = PROJECT_ROOT / "bt_configs"
FORMAT = "bt-lab/v1"
NODE_TYPES = {"selector", "sequence", "condition", "action"}
NODE_FIELDS = {"type", "name", "behavior", "memory", "params", "children"}


@dataclass(frozen=True)
class LoadedBehaviorTree:
    """Loader 的构建结果，同时提供运行根节点和按名称访问的节点索引。"""

    config_id: str
    name: str
    root: py_trees.behaviour.Behaviour
    nodes_by_name: dict[str, py_trees.behaviour.Behaviour]


def load_bt_definition(
    name: str, config_dir: Path = DEFAULT_BT_CONFIG_DIR
) -> dict:
    """按名称读取项目内 BT JSON，返回尚未构建的普通字典。

    ``name`` 可以是 ``default`` 或 ``default.json``，但不能包含目录，以免
    一个 CLI 参数绕过固定的 ``bt_configs`` 根目录。
    """
    filename = name if name.endswith(".json") else f"{name}.json"
    if Path(filename).name != filename:
        raise ValueError("BT config name must not contain a directory path")
    path = Path(config_dir) / filename
    try:
        # read_text 明确使用 UTF-8，使节点名称可以安全包含非 ASCII 字符。
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"BT config not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid BT JSON in {path} at line {error.lineno}: {error.msg}"
        ) from error


def build_behavior_tree(
    definition: dict,
    context: BehaviorBuildContext,
    registry: dict[str, BehaviorClass] = BEHAVIOR_REGISTRY,
) -> LoadedBehaviorTree:
    """校验 v1 定义并构建可直接 tick 的 py_trees 节点树。

    结构错误使用 ``root.children[0]`` 一类路径定位；Behavior 构造错误也会
    附带同一路径，便于从异常直接找到 JSON 中的问题位置。
    """
    # 第一层只校验整个文件的元数据；节点字段由内部 build() 递归校验。
    if not isinstance(definition, dict):
        raise ValueError("BT definition must be a JSON object")
    if definition.get("format") != FORMAT:
        raise ValueError(f"BT format must be '{FORMAT}'")
    config_id = definition.get("id")
    if not isinstance(config_id, str) or not config_id:
        raise ValueError("BT definition requires a non-empty id")
    definition_name = definition.get("name")
    if not isinstance(definition_name, str) or not definition_name:
        raise ValueError("BT definition requires a non-empty name")
    if "root" not in definition:
        raise ValueError("BT definition is missing root")

    # 每次构建都清空索引，防止复用 Context 时残留上一棵树的节点引用。
    context.nodes_by_name = {}
    seen_names: set[str] = set()

    def build(node_definition: object, path: str) -> py_trees.behaviour.Behaviour:
        """Build one node recursively while preserving a precise JSON error path."""
        if not isinstance(node_definition, dict):
            raise ValueError(f"{path}: node must be an object")
        unknown_fields = sorted(set(node_definition) - NODE_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{path}: unknown node fields: {', '.join(unknown_fields)}"
            )
        # type 决定使用 py_trees Composite，还是从 Registry 查找叶节点类。
        node_type = node_definition.get("type")
        if node_type not in NODE_TYPES:
            raise ValueError(f"{path}: invalid node type '{node_type}'")
        node_name = node_definition.get("name")
        if not isinstance(node_name, str) or not node_name:
            raise ValueError(f"{path}: node requires a non-empty name")
        # name 既用于界面显示，也作为 Condition 引用键，因此必须全树唯一。
        if node_name in seen_names:
            raise ValueError(f"{path}: duplicate node name '{node_name}'")
        seen_names.add(node_name)

        # params 不由 Loader 解释，原样展开给统一 Behavior 构造函数。
        params = node_definition.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"{path}: params must be an object")

        if node_type in {"selector", "sequence"}:
            if "children" not in node_definition:
                raise ValueError(f"{path}: missing children")
            children_definition = node_definition["children"]
            if not isinstance(children_definition, list) or not children_definition:
                raise ValueError(f"{path}: children must be a non-empty list")
            # memory 控制 Composite 在 RUNNING 后是否从上次子节点继续。
            memory = node_definition.get("memory", False)
            if not isinstance(memory, bool):
                raise ValueError(f"{path}: memory must be a boolean")
            if params:
                raise ValueError(f"{path}: composite params are not supported")
            # 子节点严格按 JSON 顺序构建，顺序同时决定 BT 优先级和可视化顺序。
            children = [
                build(child, f"{path}.children[{index}]")
                for index, child in enumerate(children_definition)
            ]
            # Composite 由 py_trees 原生类实现，本项目不另造 BT Runtime。
            composite_type = (
                py_trees.composites.Selector
                if node_type == "selector"
                else py_trees.composites.Sequence
            )
            node = composite_type(name=node_name, memory=memory, children=children)
        else:
            if "children" in node_definition:
                raise ValueError(f"{path}: leaf nodes cannot have children")
            if "memory" in node_definition:
                raise ValueError(f"{path}: memory is only valid for composites")
            behavior_name = node_definition.get("behavior")
            if not isinstance(behavior_name, str) or not behavior_name:
                raise ValueError(f"{path}: leaf node requires behavior")
            behavior_class = registry.get(behavior_name)
            if behavior_class is None:
                raise ValueError(f"{path}: unknown behavior '{behavior_name}'")
            try:
                # 所有叶节点共享同一构造协议，Loader 不再了解各节点的依赖差异。
                node = behavior_class(
                    context=context,
                    name=node_name,
                    **params,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: {error}") from error
            # 防止 JSON 把 Action 类错误声明为 Condition，造成语义与图形不一致。
            if getattr(node, "visual_type", None) != node_type:
                raise ValueError(
                    f"{path}: behavior '{behavior_name}' is not a {node_type}"
                )

        # 后续 Action 可通过 JSON 中的名称引用已构建 Condition。
        context.nodes_by_name[node_name] = node
        return node

    root = build(definition["root"], "root")
    return LoadedBehaviorTree(
        config_id=config_id,
        name=definition_name,
        root=root,
        nodes_by_name=dict(context.nodes_by_name),
    )


def load_behavior_tree(
    name: str,
    context: BehaviorBuildContext,
    config_dir: Path = DEFAULT_BT_CONFIG_DIR,
) -> LoadedBehaviorTree:
    """组合“读取 JSON”和“构建 Runtime Tree”两个步骤。"""
    return build_behavior_tree(load_bt_definition(name, config_dir), context)
