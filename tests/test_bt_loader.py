"""验证 BT JSON、Class Registry、递归构建和 Runtime 子节点顺序。"""

import copy
import os
import tempfile
import unittest
from pathlib import Path

import pygame
import py_trees

from autonomy_lab.bt.behaviors import AgentAction, AgentBehaviour
from autonomy_lab.bt.context import BehaviorBuildContext
from autonomy_lab.bt.loader import (
    build_behavior_tree,
    load_behavior_tree,
    load_bt_definition,
)
from autonomy_lab.bt.registry import BEHAVIOR_REGISTRY
from autonomy_lab.bt.visualizer import BTVisualizer
from autonomy_lab.environment import Environment
from autonomy_lab.perception import AgentPerception
from autonomy_lab.scene_config import get_scene


def make_context() -> BehaviorBuildContext:
    """使用真实场景依赖创建 Context，避免 Loader 测试依赖 mock 行为。"""
    environment = Environment(get_scene("simple"))
    return BehaviorBuildContext(
        perception=AgentPerception(environment),
        command={"turn": 0.0, "throttle": 0.0},
        behavior_config=environment.scene_config["behavior_tree"],
    )


def leaf_definition(behavior: str = "SearchTarget", name: str = "Search") -> dict:
    """返回最小合法单节点定义，用于错误配置和扩展性测试。"""
    return {
        "format": "bt-lab/v1",
        "id": "test_bt",
        "name": "Test Tree",
        "root": {
            "type": "action",
            "name": name,
            "behavior": behavior,
        },
    }


class ProbeAction(AgentAction):
    """只用于验证统一构造接口的最小 Action。"""

    allowed_params = frozenset({"amount"})

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class BTLoaderTests(unittest.TestCase):
    def test_registry_maps_names_directly_to_behavior_classes(self):
        for behavior_class in BEHAVIOR_REGISTRY.values():
            self.assertTrue(issubclass(behavior_class, AgentBehaviour))

    def test_agent_action_uses_uniform_context_name_and_params(self):
        context = make_context()

        node = ProbeAction(context=context, name="Probe", amount=2.0)

        self.assertIs(node.context, context)
        self.assertEqual(node.name, "Probe")
        self.assertEqual(node.params, {"amount": 2.0})
        self.assertEqual(node.visual_type, "action")

    def test_default_definition_builds_current_runtime_topology(self):
        loaded = load_behavior_tree("default", make_context())

        self.assertEqual(loaded.config_id, "default_bt")
        self.assertEqual(loaded.name, "Default Agent Behavior")
        self.assertEqual(loaded.root.name, "Priority Selector")
        self.assertEqual(len(list(loaded.root.iterate())), 16)
        self.assertEqual(
            [child.name for child in loaded.root.children],
            [
                "Obstacle Avoidance",
                "Target Gap Navigation",
                "Target Pursuit",
                "Gap Exploration",
                "Search Target",
            ],
        )
        self.assertFalse(loaded.root.memory)
        self.assertTrue(loaded.nodes_by_name["Obstacle Avoidance"].memory)
        self.assertTrue(loaded.nodes_by_name["Target Gap Navigation"].memory)
        self.assertFalse(loaded.nodes_by_name["Target Pursuit"].memory)
        self.assertTrue(loaded.nodes_by_name["Gap Exploration"].memory)

    def test_json_params_override_scene_defaults_and_omissions_fall_back(self):
        definition = leaf_definition(name="Fast Search")
        definition["root"]["params"] = {"turn": 0.75}

        loaded = build_behavior_tree(definition, make_context())
        search = loaded.nodes_by_name["Fast Search"]

        self.assertEqual(search.turn, 0.75)
        self.assertEqual(search.throttle, 0.0)

    def test_registered_conforming_class_needs_no_loader_factory(self):
        definition = leaf_definition(behavior="ProbeAction", name="Probe")
        definition["root"]["params"] = {"amount": 3.0}

        loaded = build_behavior_tree(
            definition,
            make_context(),
            registry={"ProbeAction": ProbeAction},
        )

        self.assertEqual(loaded.nodes_by_name["Probe"].params, {"amount": 3.0})

    def test_reordered_json_drives_runtime_and_visualizer_order(self):
        definition = load_bt_definition("default")
        definition["root"]["children"][0], definition["root"]["children"][1] = (
            definition["root"]["children"][1],
            definition["root"]["children"][0],
        )
        loaded = build_behavior_tree(definition, make_context())
        tree = py_trees.trees.BehaviourTree(loaded.root)
        snapshot = py_trees.visitors.SnapshotVisitor()
        tree.visitors.append(snapshot)
        visualizer = BTVisualizer(loaded.root, snapshot)

        visualizer.sync(pygame.Rect(0, 0, 480, 500))

        self.assertEqual(loaded.root.children[0].name, "Target Gap Navigation")
        root_connections = [
            child_id
            for parent_id, child_id in visualizer.connections
            if parent_id == loaded.root.id
        ]
        self.assertEqual(root_connections, [child.id for child in loaded.root.children])

    def test_existing_registered_behavior_can_be_added_without_loader_changes(self):
        definition = load_bt_definition("default")
        definition["root"]["children"].append(
            {
                "type": "action",
                "name": "Backup Search",
                "behavior": "SearchTarget",
                "params": {"turn": -0.25},
            }
        )

        loaded = build_behavior_tree(definition, make_context())
        tree = py_trees.trees.BehaviourTree(loaded.root)
        snapshot = py_trees.visitors.SnapshotVisitor()
        visualizer = BTVisualizer(loaded.root, snapshot)
        visualizer.sync(pygame.Rect(0, 0, 480, 500))

        self.assertEqual(len(list(loaded.root.iterate())), 17)
        self.assertIn("Backup Search", loaded.nodes_by_name)
        self.assertIn(loaded.nodes_by_name["Backup Search"].id, visualizer.visual_nodes)

    def test_default_path_does_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                definition = load_bt_definition("default")
            finally:
                os.chdir(previous)

        self.assertEqual(definition["id"], "default_bt")

    def test_invalid_definitions_raise_clear_errors(self):
        cases = []

        unknown = leaf_definition(behavior="MissingBehavior")
        cases.append((unknown, "unknown behavior 'MissingBehavior'"))

        invalid_type = leaf_definition()
        invalid_type["root"]["type"] = "parallel"
        cases.append((invalid_type, "invalid node type 'parallel'"))

        missing_children = leaf_definition()
        missing_children["root"] = {
            "type": "selector",
            "name": "Empty Selector",
            "memory": False,
        }
        cases.append((missing_children, "missing children"))

        duplicate = leaf_definition()
        duplicate["root"] = {
            "type": "sequence",
            "name": "Duplicate Sequence",
            "memory": False,
            "children": [
                {"type": "action", "name": "Same", "behavior": "SearchTarget"},
                {"type": "action", "name": "Same", "behavior": "SearchTarget"},
            ],
        }
        cases.append((duplicate, "duplicate node name 'Same'"))

        missing_condition = leaf_definition(behavior="AvoidObstacle")
        missing_condition["root"]["params"] = {"condition": "Not Built"}
        cases.append((missing_condition, "unknown condition 'Not Built'"))

        unknown_param = leaf_definition()
        unknown_param["root"]["params"] = {"unexpected": 1}
        cases.append((unknown_param, "unknown params for SearchTarget: unexpected"))

        non_numeric = leaf_definition()
        non_numeric["root"]["params"] = {"turn": "left"}
        cases.append(
            (non_numeric, "parameter 'turn' for SearchTarget must be numeric")
        )

        for definition, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_behavior_tree(copy.deepcopy(definition), make_context())


if __name__ == "__main__":
    unittest.main()
