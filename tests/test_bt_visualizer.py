import unittest

import pygame
import py_trees

from autonomy_lab.behavior_tree import PANEL_WIDTH, BehaviorTreeController
from autonomy_lab.bt_visualizer import BTVisualizer
from autonomy_lab.environment import Environment
from autonomy_lab.scene_config import get_scene


def build_test_tree():
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
        self.assertLess(nodes["Root"].y, nodes["Branch"].y)
        self.assertLess(nodes["Branch"].y, nodes["Condition"].y)
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
        self.assertEqual(controller.obstacle_near.visual_type, "condition")
        self.assertEqual(controller.avoid_obstacle.visual_type, "action")
        self.assertEqual(controller.move_to_target.visual_type, "action")

        controller.tick(1.0 / 60.0)
        surface = pygame.Surface(
            (environment.world_size[0] + PANEL_WIDTH, environment.world_size[1])
        )
        controller.draw_panel(
            surface, pygame.font.Font(None, 26), environment.world_size[0]
        )
        self.assertEqual(controller.visualizer.rebuild_count, 1)

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


if __name__ == "__main__":
    unittest.main()
