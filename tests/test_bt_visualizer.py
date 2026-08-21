"""验证面板结构与状态始终来自真实 py_trees Runtime Tree。"""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import py_trees

from autonomy_lab.bt.controller import PANEL_WIDTH, BehaviorTreeController
from autonomy_lab.bt.visualizer import BTVisualizer
from autonomy_lab.core.environment import Environment
from autonomy_lab.rendering.renderer import PygameRenderer
from autonomy_lab.scenarios.config import get_scene


def build_test_tree():
    """创建带叶节点语义标记的小树，隔离测试提取和布局逻辑。"""
    condition = py_trees.behaviours.Success(name="Condition")
    condition.visual_type = "condition"
    action = py_trees.behaviours.Running(name="Branch Action")
    action.visual_type = "action"
    fallback = py_trees.behaviours.Running(name="Fallback Action")
    fallback.visual_type = "action"
    branch = py_trees.composites.Sequence(
        name="Branch", memory=False, children=[condition, action]
    )
    root = py_trees.composites.Selector(
        name="Root", memory=False, children=[branch, fallback]
    )
    tree = py_trees.trees.BehaviourTree(root)
    snapshot = py_trees.visitors.SnapshotVisitor()
    tree.visitors.append(snapshot)
    return tree, snapshot


class BTVisualizerTopologyTests(unittest.TestCase):
    def test_extracts_real_tree_topology_types_and_hierarchical_layout(self):
        tree, snapshot = build_test_tree()
        visualizer = BTVisualizer(tree.root, snapshot)

        rebuilt = visualizer.sync(pygame.Rect(0, 140, 480, 500))

        self.assertTrue(rebuilt)
        self.assertEqual(len(visualizer.visual_nodes), 5)
        nodes = {item.name: item for item in visualizer.visual_nodes.values()}
        self.assertEqual(nodes["Root"].node_type, "selector")
        self.assertEqual(nodes["Branch"].node_type, "sequence")
        self.assertEqual(nodes["Condition"].node_type, "condition")
        self.assertEqual(nodes["Branch Action"].node_type, "action")
        self.assertEqual(nodes["Fallback Action"].node_type, "action")
        self.assertEqual(nodes["Root"].depth, 0)
        self.assertEqual(nodes["Branch"].depth, 1)
        self.assertEqual(nodes["Condition"].depth, 2)
        self.assertLess(nodes["Root"].x, nodes["Branch"].x)
        self.assertLess(nodes["Branch"].x, nodes["Condition"].x)
        self.assertLess(nodes["Condition"].y, nodes["Branch Action"].y)
        self.assertLess(nodes["Branch Action"].y, nodes["Fallback Action"].y)
        self.assertEqual(
            visualizer.connections,
            [
                (tree.root.id, tree.root.children[0].id),
                (tree.root.children[0].id, tree.root.children[0].children[0].id),
                (tree.root.children[0].id, tree.root.children[0].children[1].id),
                (tree.root.id, tree.root.children[1].id),
            ],
        )

    def test_rebuilds_only_when_real_tree_structure_changes(self):
        tree, snapshot = build_test_tree()
        visualizer = BTVisualizer(tree.root, snapshot)
        panel = pygame.Rect(0, 140, 480, 500)

        self.assertTrue(visualizer.sync(panel))
        signature_before = visualizer.signature
        self.assertFalse(visualizer.sync(panel))
        self.assertEqual(visualizer.rebuild_count, 1)

        added = py_trees.behaviours.Success(name="Added Condition")
        added.visual_type = "condition"
        tree.root.add_child(added)

        self.assertTrue(visualizer.sync(panel))
        self.assertNotEqual(visualizer.signature, signature_before)
        self.assertIn(added.id, visualizer.visual_nodes)
        self.assertIn((tree.root.id, added.id), visualizer.connections)
        self.assertEqual(visualizer.rebuild_count, 2)

        tree.root.remove_child(added)
        self.assertTrue(visualizer.sync(panel))
        self.assertNotIn(added.id, visualizer.visual_nodes)
        self.assertEqual(visualizer.rebuild_count, 3)

        original_first = tree.root.children[0]
        tree.root.remove_child(original_first)
        tree.root.add_child(original_first)
        self.assertTrue(visualizer.sync(panel))
        self.assertEqual(visualizer.visual_nodes[original_first.id].parent_id, tree.root.id)
        self.assertEqual(visualizer.rebuild_count, 4)

    def test_wraps_long_node_name_to_two_pixel_bounded_lines(self):
        pygame.font.init()
        font = pygame.font.Font(None, 12)

        lines = BTVisualizer._wrap_text(
            "Move Through Exploration Gap", font, 120, max_lines=2
        )

        self.assertLessEqual(len(lines), 2)
        self.assertEqual(" ".join(lines), "Move Through Exploration Gap")
        self.assertTrue(all(font.size(line)[0] <= 120 for line in lines))


class BTVisualizerRuntimeTests(unittest.TestCase):
    def test_runtime_state_comes_from_real_node_and_snapshot_visitor(self):
        tree, snapshot = build_test_tree()
        visualizer = BTVisualizer(tree.root, snapshot)
        panel = pygame.Rect(0, 140, 480, 500)
        visualizer.sync(panel)

        tree.tick()

        status, visited = visualizer.runtime_state(tree.root.children[0].children[1])
        self.assertEqual(status, py_trees.common.Status.RUNNING)
        self.assertTrue(visited)

        tree.root.stop(py_trees.common.Status.INVALID)
        snapshot.visited = {}
        snapshot.previously_visited = {}
        self.assertFalse(visualizer.sync(panel))
        status, visited = visualizer.runtime_state(tree.root.children[0].children[1])
        self.assertEqual(status, py_trees.common.Status.INVALID)
        self.assertFalse(visited)

    def test_controller_delegates_panel_to_metadata_driven_visualizer(self):
        pygame.font.init()
        environment = Environment(get_scene("simple"))
        controller = BehaviorTreeController(environment)

        self.assertIs(controller.visualizer.root, controller.tree.root)
        self.assertEqual(
            controller.nodes_by_name["Obstacle Threat?"].visual_type,
            "condition",
        )
        self.assertEqual(
            controller.nodes_by_name["Target Available for Gap?"].visual_type,
            "condition",
        )
        self.assertEqual(
            controller.nodes_by_name["Target Path Blocked?"].visual_type,
            "condition",
        )
        self.assertEqual(
            controller.nodes_by_name["Target-aligned Gap?"].visual_type,
            "condition",
        )
        self.assertEqual(
            controller.nodes_by_name["Target Available?"].visual_type,
            "condition",
        )
        self.assertEqual(
            controller.nodes_by_name["Traversable Gap?"].visual_type,
            "condition",
        )
        self.assertEqual(
            controller.nodes_by_name["Avoid Obstacle"].visual_type, "action"
        )
        self.assertEqual(
            controller.nodes_by_name["Move Through Target Gap"].visual_type,
            "action",
        )
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].visual_type, "action"
        )
        self.assertEqual(
            controller.nodes_by_name["Move Through Exploration Gap"].visual_type,
            "action",
        )
        self.assertEqual(
            controller.nodes_by_name["Search Target"].visual_type, "action"
        )

        controller.tick(1.0 / 60.0)
        surface = pygame.Surface(
            (environment.world_size[0] + PANEL_WIDTH, environment.world_size[1])
        )
        controller.draw_panel(
            surface, pygame.font.Font(None, 26), environment.world_size[0]
        )
        self.assertEqual(controller.visualizer.rebuild_count, 1)
        self.assertEqual(len(controller.visualizer.visual_nodes), 19)
        layout_panel = pygame.Rect(
            environment.world_size[0],
            132,
            PANEL_WIDTH,
            environment.world_size[1] - 132 - 62,
        )
        rectangles = [
            controller.visualizer._node_rect(visual)
            for visual in controller.visualizer.visual_nodes.values()
        ]
        for rect in rectangles:
            self.assertTrue(layout_panel.contains(rect))
        for index, rect in enumerate(rectangles):
            for other_rect in rectangles[index + 1 :]:
                self.assertFalse(rect.colliderect(other_rect))

        controller.tick(1.0 / 60.0)
        controller.draw_panel(
            surface, pygame.font.Font(None, 26), environment.world_size[0]
        )
        self.assertEqual(controller.visualizer.rebuild_count, 1)

        controller.reset()
        controller.draw_panel(
            surface, pygame.font.Font(None, 26), environment.world_size[0]
        )
        self.assertEqual(controller.visualizer.rebuild_count, 1)
        self.assertFalse(
            any(
                controller.visualizer.runtime_state(node)[1]
                for node in controller.tree.root.iterate()
            )
        )

    def test_feedback_text_is_only_exposed_for_visited_nodes_and_is_truncated(self):
        tree, snapshot = build_test_tree()
        visualizer = BTVisualizer(tree.root, snapshot)
        action = tree.root.children[0].children[1]
        action.feedback_message = "searching for a distant target"

        self.assertEqual(visualizer.feedback_text(action, max_chars=14), "")

        tree.tick()
        action.feedback_message = "searching for a distant target"

        self.assertEqual(
            visualizer.feedback_text(action, max_chars=14), "searching f..."
        )

    def test_environment_draws_fov_in_front_but_not_behind_agent(self):
        scene = get_scene("simple")
        scene["agent"]["position"] = (300, 300)
        scene["agent"]["heading_degrees"] = 0.0
        scene["target"]["position"] = (800, 600)
        scene["obstacles"] = []
        scene["sensor"] = {
            "range": 100.0,
            "fov_degrees": 90.0,
            "los_enabled": True,
        }
        environment = Environment(scene)
        renderer = PygameRenderer(environment)
        try:
            renderer.render(environment)
            surface = renderer.screen

            background = scene["display"]["background_color"]
            self.assertNotEqual(surface.get_at((350, 300))[:3], background)
            self.assertEqual(surface.get_at((250, 300))[:3], background)
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()
