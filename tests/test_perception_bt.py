"""使用真实同步 snapshot 验证 Behavior 状态、命令和 Controller 抢占。"""

import math
from pathlib import Path
import tempfile
import unittest

import pygame
import py_trees

import autonomy_lab.behaviors as behavior_nodes
from autonomy_lab.behavior_context import BehaviorBuildContext
from autonomy_lab.behavior_tree import BehaviorTreeController
from autonomy_lab.behaviors import (
    MoveToTarget,
    ObstacleThreat,
    SearchTarget,
    TargetAvailable,
)
from autonomy_lab.environment import Environment
from autonomy_lab.experiment import ExperimentRecorder
from autonomy_lab.perception import AgentPerception
from autonomy_lab.scene_config import get_scene


def make_environment(
    *,
    target_position=(400.0, 300.0),
    obstacles=(),
    mode="perceived",
    behavior_tree_updates=None,
):
    """创建紧凑、确定性的测试场景，便于控制目标和障碍物相对位置。"""
    scene = get_scene("simple")
    scene["agent"]["position"] = (300.0, 300.0)
    scene["agent"]["heading_degrees"] = 0.0
    scene["target"]["position"] = target_position
    scene["obstacles"] = list(obstacles)
    scene["target_information_mode"] = mode
    if behavior_tree_updates:
        scene["behavior_tree"].update(behavior_tree_updates)
    return Environment(scene)


def make_behavior_context(
    perception: AgentPerception,
    command: dict[str, float] | None = None,
) -> BehaviorBuildContext:
    """为直接构造 Behavior 的测试组装共享感知、命令、配置和节点索引。"""
    return BehaviorBuildContext(
        perception=perception,
        command=command or {"turn": 0.0, "throttle": 0.0},
        behavior_config=perception.environment.scene_config["behavior_tree"],
    )


class PerceptionConditionTests(unittest.TestCase):
    def test_target_available_reads_visible_perception_without_changing_command(self):
        perception = AgentPerception(make_environment())
        command = {"turn": 0.4, "throttle": 0.6}
        condition = TargetAvailable(
            context=make_behavior_context(perception, command),
            name="Target Available?",
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.SUCCESS)
        self.assertEqual(command, {"turn": 0.4, "throttle": 0.6})
        self.assertIn("visible", condition.feedback_message)

    def test_target_available_reports_sensor_failure(self):
        perception = AgentPerception(
            make_environment(target_position=(200.0, 300.0))
        )
        condition = TargetAvailable(
            context=make_behavior_context(perception),
            name="Target Available?",
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.FAILURE)
        self.assertEqual(condition.feedback_message, "outside FOV")

    def test_target_available_distinguishes_ground_truth_source(self):
        perception = AgentPerception(
            make_environment(target_position=(200.0, 300.0), mode="ground_truth")
        )
        condition = TargetAvailable(
            context=make_behavior_context(perception),
            name="Target Available?",
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.SUCCESS)
        self.assertIn("ground truth", condition.feedback_message)

    def test_obstacle_threat_uses_visible_clearance_and_bearing(self):
        perception = AgentPerception(
            make_environment(obstacles=[(400, 280, 20, 40)])
        )
        condition = ObstacleThreat(
            context=make_behavior_context(perception),
            name="Obstacle Threat?",
            avoidance_distance=90.0,
            half_angle_degrees=45.0,
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.SUCCESS)
        self.assertIs(condition.threat, perception.snapshot.nearest_obstacle)
        self.assertIn("84 px", condition.feedback_message)

    def test_obstacle_outside_avoidance_distance_is_not_a_threat(self):
        perception = AgentPerception(
            make_environment(obstacles=[(400, 280, 20, 40)])
        )
        condition = ObstacleThreat(
            context=make_behavior_context(perception),
            name="Obstacle Threat?",
            avoidance_distance=80.0,
            half_angle_degrees=45.0,
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.FAILURE)
        self.assertIsNone(condition.threat)
        self.assertEqual(condition.feedback_message, "no nearby threat")

    def test_traversable_gap_selects_snapshot_gap_and_reports_feedback(self):
        condition_type = getattr(behavior_nodes, "TraversableGap", None)
        self.assertIsNotNone(condition_type)
        perception = AgentPerception(
            make_environment(target_position=(200.0, 300.0), obstacles=[])
        )
        condition = condition_type(
            context=make_behavior_context(perception),
            name="Traversable Gap?",
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.SUCCESS)
        self.assertIs(condition.gap, perception.snapshot.best_exploration_gap)
        self.assertEqual(condition.feedback_message, "gap: 300 px, +0 deg")

    def test_traversable_gap_fails_when_fov_has_no_safe_opening(self):
        condition_type = getattr(behavior_nodes, "TraversableGap", None)
        self.assertIsNotNone(condition_type)
        perception = AgentPerception(
            make_environment(
                target_position=(200.0, 300.0),
                obstacles=[(350, 0, 30, 700)],
            )
        )
        condition = condition_type(
            context=make_behavior_context(perception),
            name="Traversable Gap?",
        )

        status = condition.update()

        self.assertEqual(status, py_trees.common.Status.FAILURE)
        self.assertIsNone(condition.gap)
        self.assertEqual(condition.feedback_message, "no traversable gap")

    def test_target_path_blocked_and_target_aligned_gap_read_snapshot(self):
        blocked_type = getattr(behavior_nodes, "TargetPathBlocked", None)
        gap_type = getattr(behavior_nodes, "TargetAlignedGap", None)
        self.assertIsNotNone(blocked_type)
        self.assertIsNotNone(gap_type)
        scene = get_scene("simple")
        scene["target_information_mode"] = "ground_truth"
        perception = AgentPerception(Environment(scene))
        context = make_behavior_context(perception)
        blocked = blocked_type(context=context, name="Target Path Blocked?")
        target_gap = gap_type(context=context, name="Target-aligned Gap?")

        self.assertEqual(blocked.update(), py_trees.common.Status.SUCCESS)
        self.assertEqual(blocked.feedback_message, "target path blocked")
        self.assertEqual(target_gap.update(), py_trees.common.Status.SUCCESS)
        self.assertIs(target_gap.gap, perception.snapshot.best_target_gap)
        self.assertIn("target gap", target_gap.feedback_message)

    def test_target_path_blocked_fails_for_clear_direct_path(self):
        blocked_type = getattr(behavior_nodes, "TargetPathBlocked", None)
        self.assertIsNotNone(blocked_type)
        perception = AgentPerception(
            make_environment(
                target_position=(700.0, 300.0), obstacles=[], mode="ground_truth"
            )
        )
        condition = blocked_type(
            context=make_behavior_context(perception),
            name="Target Path Blocked?",
        )

        self.assertEqual(condition.update(), py_trees.common.Status.FAILURE)
        self.assertEqual(condition.feedback_message, "target path clear")


class PerceptionActionTests(unittest.TestCase):
    def test_move_to_target_uses_snapshot_relative_bearing(self):
        angle = math.radians(22.5)
        target_position = (
            300.0 + math.cos(angle) * 100.0,
            300.0 + math.sin(angle) * 100.0,
        )
        perception = AgentPerception(make_environment(target_position=target_position))
        command = {"turn": 0.0, "throttle": 0.0}
        action = MoveToTarget(
            context=make_behavior_context(perception, command),
            name="Move To Target",
            reached_distance=30.0,
        )

        status = action.update()

        self.assertEqual(status, py_trees.common.Status.RUNNING)
        self.assertAlmostEqual(command["turn"], 0.5)
        self.assertEqual(command["throttle"], 1.0)
        self.assertIn("pursuit", action.feedback_message)

    def test_move_to_target_fails_safely_without_target_information(self):
        perception = AgentPerception(
            make_environment(target_position=(200.0, 300.0))
        )
        command = {"turn": 0.7, "throttle": 0.8}
        action = MoveToTarget(
            context=make_behavior_context(perception, command),
            name="Move To Target",
            reached_distance=30.0,
        )

        status = action.update()

        self.assertEqual(status, py_trees.common.Status.FAILURE)
        self.assertEqual(command, {"turn": 0.0, "throttle": 0.0})
        self.assertEqual(action.feedback_message, "target unavailable")

    def test_move_through_gap_commits_to_entry_waypoint_until_reached(self):
        condition_type = getattr(behavior_nodes, "TraversableGap", None)
        action_type = getattr(behavior_nodes, "MoveThroughGap", None)
        self.assertIsNotNone(condition_type)
        self.assertIsNotNone(action_type)
        environment = make_environment(target_position=(700.0, 300.0), obstacles=[])
        perception = AgentPerception(environment)
        command = {"turn": 0.0, "throttle": 0.0}
        context = make_behavior_context(perception, command)
        condition = condition_type(context=context, name="Traversable Gap?")
        context.nodes_by_name[condition.name] = condition
        self.assertEqual(condition.update(), py_trees.common.Status.SUCCESS)
        action = action_type(
            context=context,
            name="Move Through Test Gap",
            condition=condition.name,
            throttle=0.5,
            reached_distance=24.0,
        )

        action.initialise()
        committed_entry = action.entry_position
        condition.gap = None

        status = action.update()

        self.assertEqual(status, py_trees.common.Status.RUNNING)
        self.assertEqual(action.entry_position, committed_entry)
        self.assertEqual(command, {"turn": 0.0, "throttle": 0.5})
        self.assertIn("gap entry", action.feedback_message)

        environment.agent.position.update(committed_entry)
        self.assertEqual(action.update(), py_trees.common.Status.SUCCESS)
        self.assertEqual(command, {"turn": 0.0, "throttle": 0.0})

        action.terminate(py_trees.common.Status.INVALID)
        self.assertIsNone(action.entry_position)

    def test_search_target_rotates_in_place_without_target_information(self):
        for target_position in ((700.0, 300.0), (200.0, 300.0)):
            with self.subTest(target_position=target_position):
                perception = AgentPerception(
                    make_environment(target_position=target_position)
                )
                command = {"turn": 0.0, "throttle": 0.0}
                action = SearchTarget(
                    context=make_behavior_context(perception, command),
                    name="Search Target",
                    throttle=0.0,
                    turn=0.25,
                )

                status = action.update()

                self.assertEqual(status, py_trees.common.Status.RUNNING)
                self.assertEqual(command, {"turn": 0.25, "throttle": 0.0})
                self.assertEqual(action.feedback_message, "search scan")


class PerceptionControllerTests(unittest.TestCase):
    def test_real_tree_has_reactive_perception_driven_topology(self):
        environment = make_environment(target_position=(800.0, 300.0))

        controller = BehaviorTreeController(environment, bt_config="default")

        self.assertEqual(controller.bt_config_id, "default_bt")
        self.assertEqual(controller.bt_definition_name, "Default Agent Behavior")
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].name,
            "Move To Target",
        )
        self.assertEqual(controller.root.name, "Priority Selector")
        self.assertEqual(
            [node.name for node in controller.root.children],
            [
                "Obstacle Avoidance",
                "Target Gap Navigation",
                "Target Pursuit",
                "Gap Exploration",
                "Search Target",
            ],
        )
        self.assertEqual(
            [
                node.name
                for node in controller.nodes_by_name["Obstacle Avoidance"].children
            ],
            ["Obstacle Threat?", "Avoid Obstacle"],
        )
        self.assertEqual(
            [
                node.name
                for node in controller.nodes_by_name[
                    "Target Gap Navigation"
                ].children
            ],
            [
                "Target Available for Gap?",
                "Target Path Blocked?",
                "Target-aligned Gap?",
                "Move Through Target Gap",
            ],
        )
        self.assertEqual(
            [
                node.name
                for node in controller.nodes_by_name["Target Pursuit"].children
            ],
            ["Target Available?", "Move To Target"],
        )
        self.assertEqual(
            [
                node.name
                for node in controller.nodes_by_name["Gap Exploration"].children
            ],
            ["Traversable Gap?", "Move Through Exploration Gap"],
        )
        self.assertEqual(len(list(controller.root.iterate())), 16)
        self.assertFalse(controller.root.memory)
        self.assertTrue(controller.nodes_by_name["Obstacle Avoidance"].memory)
        self.assertTrue(controller.nodes_by_name["Target Gap Navigation"].memory)
        self.assertFalse(controller.nodes_by_name["Target Pursuit"].memory)
        self.assertTrue(controller.nodes_by_name["Gap Exploration"].memory)

    def test_simple_ground_truth_uses_target_gap_before_direct_pursuit(self):
        scene = get_scene("simple")
        scene["target_information_mode"] = "ground_truth"
        controller = BehaviorTreeController(Environment(scene))

        command = controller.tick(1.0 / 60.0)

        self.assertEqual(command, (1.0, 0.5))
        self.assertEqual(controller.active_behavior, "Move Through Target Gap")
        self.assertEqual(
            controller.nodes_by_name["Move Through Target Gap"].status,
            py_trees.common.Status.RUNNING,
        )
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].status,
            py_trees.common.Status.INVALID,
        )

    def test_simple_target_gap_commitment_reaches_entry_without_early_avoidance(self):
        scene = get_scene("simple")
        scene["target_information_mode"] = "ground_truth"
        environment = Environment(scene)
        controller = BehaviorTreeController(environment)
        reached_entry = False

        for _ in range(180):
            turn, throttle = controller.tick(1.0 / 60.0)
            self.assertNotEqual(controller.active_behavior, "Avoid Obstacle")
            environment.step(
                {"turn": turn, "throttle": throttle},
                1.0 / 60.0,
            )
            if (
                controller.nodes_by_name["Move Through Target Gap"].status
                == py_trees.common.Status.SUCCESS
            ):
                reached_entry = True
                break

        self.assertTrue(reached_entry)
        self.assertGreater(environment.agent.position.y, 480.0)

    def test_ground_truth_clear_or_out_of_fov_path_uses_direct_pursuit(self):
        cases = (
            ((700.0, 300.0), (0.0, 1.0)),
            ((200.0, 300.0), (-1.0, 0.3)),
        )
        for target_position, expected_command in cases:
            with self.subTest(target_position=target_position):
                controller = BehaviorTreeController(
                    make_environment(
                        target_position=target_position,
                        obstacles=[],
                        mode="ground_truth",
                    )
                )

                command = controller.tick(1.0 / 60.0)

                self.assertEqual(command, expected_command)
                self.assertEqual(controller.active_behavior, "Move To Target")

    def test_gap_exploration_switches_to_pursuit_when_target_enters_fov(self):
        environment = make_environment(target_position=(700.0, 300.0))
        controller = BehaviorTreeController(environment)

        exploration_command = controller.tick(1.0 / 60.0)
        self.assertEqual(exploration_command, (0.0, 0.5))
        self.assertEqual(
            controller.nodes_by_name["Move Through Exploration Gap"].status,
            py_trees.common.Status.RUNNING,
        )
        self.assertEqual(controller.active_behavior, "Move Through Exploration Gap")

        environment.target.update(400.0, 300.0)
        pursuit_command = controller.tick(1.0 / 60.0)

        self.assertEqual(pursuit_command, (0.0, 1.0))
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].status,
            py_trees.common.Status.RUNNING,
        )
        self.assertEqual(
            controller.nodes_by_name["Move Through Exploration Gap"].status,
            py_trees.common.Status.INVALID,
        )
        self.assertEqual(
            controller.nodes_by_name["Search Target"].status,
            py_trees.common.Status.INVALID,
        )
        self.assertEqual(controller.active_behavior, "Move To Target")

    def test_no_gap_fallback_rotates_in_place(self):
        environment = make_environment(
            target_position=(200.0, 300.0),
            obstacles=[(350, 0, 30, 700)],
            behavior_tree_updates={"obstacle_detection_distance": 10.0},
        )
        controller = BehaviorTreeController(environment)

        command = controller.tick(1.0 / 60.0)

        self.assertEqual(command, (0.25, 0.0))
        self.assertEqual(
            controller.nodes_by_name["Search Target"].status,
            py_trees.common.Status.RUNNING,
        )
        self.assertEqual(controller.active_behavior, "Search Target")

    def test_obstacle_avoidance_preempts_gap_exploration(self):
        environment = make_environment(target_position=(700.0, 300.0))
        controller = BehaviorTreeController(environment)
        self.assertEqual(controller.tick(1.0 / 60.0), (0.0, 0.5))
        self.assertEqual(controller.active_behavior, "Move Through Exploration Gap")

        environment.obstacles = [pygame.Rect(317, 280, 20, 40)]
        avoid_command = controller.tick(1.0 / 60.0)

        self.assertEqual(avoid_command, (1.0, 0.75))
        self.assertEqual(controller.active_behavior, "Avoid Obstacle")
        self.assertEqual(
            controller.nodes_by_name["Move Through Exploration Gap"].status,
            py_trees.common.Status.INVALID,
        )

    def test_obstacle_avoidance_preempts_pursuit_without_stale_command(self):
        environment = make_environment(target_position=(500.0, 300.0))
        controller = BehaviorTreeController(environment)
        self.assertEqual(controller.tick(1.0 / 60.0), (0.0, 1.0))
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].status,
            py_trees.common.Status.RUNNING,
        )

        environment.obstacles = [pygame.Rect(400, 280, 20, 40)]
        avoid_command = controller.tick(1.0 / 60.0)

        self.assertEqual(avoid_command, (1.0, 0.75))
        self.assertEqual(
            controller.nodes_by_name["Avoid Obstacle"].status,
            py_trees.common.Status.RUNNING,
        )
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].status,
            py_trees.common.Status.INVALID,
        )
        self.assertEqual(controller.active_behavior, "Avoid Obstacle")

        environment.obstacles = []
        self.assertEqual(controller.tick(1.0 / 60.0), (1.0, 0.75))
        self.assertEqual(
            controller.nodes_by_name["Avoid Obstacle"].status,
            py_trees.common.Status.RUNNING,
        )

        for _ in range(60):
            resumed_command = controller.tick(1.0 / 60.0)
            if (
                controller.nodes_by_name["Move To Target"].status
                == py_trees.common.Status.RUNNING
            ):
                break
        else:
            self.fail("pursuit did not resume after the timed avoidance action")

        self.assertEqual(resumed_command, (0.0, 1.0))
        self.assertEqual(
            controller.nodes_by_name["Move To Target"].status,
            py_trees.common.Status.RUNNING,
        )

    def test_m2_records_gap_pursuit_and_avoidance_transitions(self):
        environment = make_environment(target_position=(700.0, 300.0))
        controller = BehaviorTreeController(environment)

        with tempfile.TemporaryDirectory() as output_dir:
            recorder = ExperimentRecorder(Path(output_dir))
            recorder.start_episode(environment, "test", "bt-v1", track_bt=True)

            controller.tick(1.0 / 60.0)
            self.assertEqual(
                controller.active_behavior, "Move Through Exploration Gap"
            )
            recorder.update(
                1.0 / 60.0,
                environment,
                active_action=controller.active_behavior,
                bt_ticked=True,
            )

            environment.target.update(400.0, 300.0)
            controller.tick(1.0 / 60.0)
            self.assertEqual(controller.active_behavior, "Move To Target")
            recorder.update(
                1.0 / 60.0,
                environment,
                active_action=controller.active_behavior,
                bt_ticked=True,
            )

            environment.obstacles = [pygame.Rect(340, 280, 20, 40)]
            controller.tick(1.0 / 60.0)
            self.assertEqual(controller.active_behavior, "Avoid Obstacle")
            recorder.update(
                1.0 / 60.0,
                environment,
                active_action=controller.active_behavior,
                bt_ticked=True,
            )

            payload = recorder.finish_episode("FAILURE", "test_complete")

            self.assertEqual(payload["bt_tick_count"], 3)
            self.assertEqual(payload["bt_transition_count"], 2)
            self.assertIn("trajectory", payload)
            self.assertTrue((Path(output_dir) / "results.csv").exists())
            self.assertTrue(
                (Path(output_dir) / "runs" / "episode_0001.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
