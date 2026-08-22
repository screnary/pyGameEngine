"""R0.7 Research sensing 的有限距离、360 度与 legacy 隔离回归。"""

from copy import deepcopy
import math
import unittest

import py_trees

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.core.environment import Environment
from autonomy_lab.perception.pygame_perception import sector_index_for_bearing
from autonomy_lab.scenarios.config import (
    DEFAULT_RESEARCH_SENSOR_CONFIG,
    get_scene,
)
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution


def make_research_scene(
    *,
    agent_position: tuple[float, float] = (100.0, 300.0),
    target_position: tuple[float, float] = (200.0, 300.0),
    obstacles: list[tuple[int, int, int, int]] | None = None,
    hazard_noise: float = 0.0,
) -> dict:
    """从冻结场景复制几何接口，只给测试场景启用 Research sensing profile。"""
    scene = get_scene("rl_sanity")
    scene["world_size"] = (850, 600)
    scene["agent"]["position"] = agent_position
    scene["agent"]["heading_degrees"] = 0.0
    scene["target"]["position"] = target_position
    scene["obstacles"] = list(obstacles or ())
    # 故意保留 ground_truth 模式，验证 Research profile 不会借它绕过 range gate。
    scene["target_information_mode"] = "ground_truth"
    scene["sensor"] = {
        **scene["sensor"],
        **DEFAULT_RESEARCH_SENSOR_CONFIG,
    }
    scene["perception_noise"] = {
        "hazard_range_std": hazard_noise,
    }
    return scene


class ResearchGoalSensingTests(unittest.TestCase):
    def test_research_defaults_keep_goal_long_range_and_hazard_local(self):
        """把 Goal/Hazard 重新绑成同一量程会破坏 R0.8 的职责分离。"""
        config = DEFAULT_RESEARCH_SENSOR_CONFIG
        self.assertTrue(math.isfinite(config["goal_range"]))
        self.assertGreater(config["goal_range"], 0.0)
        self.assertGreater(config["goal_range"], config["hazard_range"])
        self.assertEqual(config["goal_num_bins"], 16)
        self.assertEqual(config["hazard_num_bins"], 16)

    def test_longest_calibration_goal_is_initially_sensed(self):
        """Goal range 过短会使正常 Research episode 一开始就失去导航信号。"""
        world = Environment(ScenarioDistribution("static_random").sample(1020))
        self.assertTrue(world.perception.snapshot.goal.sensed)

    def test_goal_just_inside_and_outside_range_has_no_ground_truth_leak(self):
        goal_range = float(DEFAULT_RESEARCH_SENSOR_CONFIG["goal_range"])
        inside = Environment(
            make_research_scene(target_position=(100.0 + goal_range - 1.0, 300.0))
        ).perception.snapshot.goal
        outside = Environment(
            make_research_scene(target_position=(100.0 + goal_range + 1.0, 300.0))
        ).perception.snapshot.goal

        self.assertTrue(inside.sensed)
        self.assertTrue(inside.available)
        self.assertIsNotNone(inside.distance)
        self.assertIsNotNone(inside.bearing)
        self.assertIsNotNone(inside.sector_index)
        self.assertFalse(outside.sensed)
        self.assertFalse(outside.available)
        self.assertIsNone(outside.distance)
        self.assertIsNone(outside.bearing)
        self.assertIsNone(outside.sector_index)

    def test_goal_is_sensed_in_front_rear_left_and_right(self):
        positions = {
            "front": ((525.0, 300.0), 0.0, 8),
            "rear": ((325.0, 300.0), -math.pi, 0),
            "agent_left": ((425.0, 200.0), -math.pi / 2.0, 4),
            "agent_right": ((425.0, 400.0), math.pi / 2.0, 12),
        }
        for label, (target, expected_bearing, expected_sector) in positions.items():
            with self.subTest(direction=label):
                goal = Environment(
                    make_research_scene(
                        agent_position=(425.0, 300.0),
                        target_position=target,
                    )
                ).perception.snapshot.goal
                self.assertTrue(goal.sensed)
                self.assertAlmostEqual(goal.bearing, expected_bearing)
                self.assertEqual(goal.sector_index, expected_sector)


class ResearchHazardLidarTests(unittest.TestCase):
    def test_sector_index_mapping_includes_wrap_and_bin_boundaries(self):
        step = 2.0 * math.pi / 16.0
        self.assertEqual(sector_index_for_bearing(0.0, 16), 8)
        self.assertEqual(sector_index_for_bearing(math.pi / 2.0, 16), 12)
        self.assertEqual(sector_index_for_bearing(-math.pi / 2.0, 16), 4)
        self.assertEqual(sector_index_for_bearing(math.pi, 16), 0)
        self.assertEqual(sector_index_for_bearing(-math.pi, 16), 0)
        self.assertEqual(sector_index_for_bearing(step / 2.0 - 1e-9, 16), 8)
        self.assertEqual(sector_index_for_bearing(step / 2.0, 16), 9)

    def test_hazard_just_inside_and_outside_range(self):
        hazard_range = int(DEFAULT_RESEARCH_SENSOR_CONFIG["hazard_range"])
        radius = 16
        inside_x = 100 + radius + hazard_range - 1
        outside_x = 100 + radius + hazard_range + 1
        inside = Environment(
            make_research_scene(obstacles=[(inside_x, 290, 10, 20)])
        ).perception.snapshot.hazard
        outside = Environment(
            make_research_scene(obstacles=[(outside_x, 290, 10, 20)])
        ).perception.snapshot.hazard

        self.assertIsNotNone(inside.nearest_hazard)
        self.assertAlmostEqual(
            inside.nearest_hazard.clearance, hazard_range - 1.0
        )
        self.assertIsNone(outside.nearest_hazard)
        self.assertEqual(outside.visible_hazards, ())

    def test_default_hazard_coverage_is_local_at_three_hundred_pixels(self):
        """默认量程退回 700 px 时，301 px 的 Hazard 会错误进入 Research BT。"""
        inside = Environment(
            make_research_scene(obstacles=[(100 + 16 + 299, 290, 10, 20)])
        ).perception.snapshot.hazard
        outside = Environment(
            make_research_scene(obstacles=[(100 + 16 + 301, 290, 10, 20)])
        ).perception.snapshot.hazard
        self.assertIsNotNone(inside.nearest_hazard)
        self.assertIsNone(outside.nearest_hazard)

    def test_hazard_is_detected_in_all_four_directions(self):
        cases = {
            "front": ((521, 290, 20, 20), 0.0),
            "rear": ((309, 290, 20, 20), -math.pi),
            "agent_left": ((415, 184, 20, 20), -math.pi / 2.0),
            "agent_right": ((415, 396, 20, 20), math.pi / 2.0),
        }
        for label, (obstacle, expected_bearing) in cases.items():
            with self.subTest(direction=label):
                hazard = Environment(
                    make_research_scene(
                        agent_position=(425.0, 300.0),
                        obstacles=[obstacle],
                    )
                ).perception.snapshot.hazard.nearest_hazard
                self.assertIsNotNone(hazard)
                self.assertAlmostEqual(hazard.bearing, expected_bearing)
                self.assertAlmostEqual(hazard.clearance, 80.0)

    def test_empty_research_lidar_does_not_encode_world_boundary(self):
        hazard_range = float(DEFAULT_RESEARCH_SENSOR_CONFIG["hazard_range"])
        snapshot = Environment(
            make_research_scene(agent_position=(20.0, 20.0))
        ).perception.snapshot

        self.assertEqual(len(snapshot.hazard.sector_ranges), 16)
        self.assertTrue(
            all(
                sector.clearance == hazard_range
                for sector in snapshot.hazard.sector_ranges
            )
        )
        self.assertLess(snapshot.boundary.left, hazard_range)
        self.assertLess(snapshot.boundary.top, hazard_range)

    def test_range_gate_precedes_noise_and_preserves_world_geometry(self):
        hazard_range = int(DEFAULT_RESEARCH_SENSOR_CONFIG["hazard_range"])
        outside_x = 100 + 16 + hazard_range + 1
        scene = make_research_scene(
            obstacles=[(outside_x, 290, 10, 20)],
            hazard_noise=1000.0,
        )
        original_obstacles = deepcopy(scene["obstacles"])
        world = Environment(scene)

        for _ in range(12):
            self.assertIsNone(world.perception.snapshot.hazard.nearest_hazard)
            self.assertTrue(
                all(
                    sector.clearance == hazard_range
                    for sector in world.perception.snapshot.hazard.sector_ranges
                )
            )
            world.perception.observe()
        self.assertEqual([tuple(rect) for rect in world.obstacles], original_obstacles)

    def test_noise_is_applied_after_detection_to_research_sector_measurement(self):
        obstacle = (196, 290, 20, 20)
        clean = Environment(make_research_scene(obstacles=[obstacle]))
        noisy = Environment(
            make_research_scene(obstacles=[obstacle], hazard_noise=25.0)
        )

        clean_front = clean.perception.snapshot.hazard.sector_ranges[8].clearance
        noisy_front = noisy.perception.snapshot.hazard.sector_ranges[8].clearance
        self.assertNotEqual(clean_front, noisy_front)
        self.assertEqual(
            [tuple(rect) for rect in clean.obstacles],
            [tuple(rect) for rect in noisy.obstacles],
        )

    def test_dynamic_hazard_switches_when_crossing_local_range(self):
        """动态 Hazard 更新后未重新 range-gate 会让 available 状态滞留。"""
        scene = make_research_scene(
            agent_position=(425.0, 300.0),
            target_position=(425.0, 100.0),
        )
        scene["dynamic_hazards"] = [
            {
                "position": (755.0, 300.0),
                "size": (20, 20),
                "speed": 120.0,
                "heading_degrees": 180.0,
            }
        ]
        world = Environment(scene)
        command = {"turn": 0.0, "throttle": 0.0}

        self.assertFalse(world.perception.snapshot.hazard.available)
        for _ in range(10):
            world.step(command, 1.0 / 60.0)
            if world.perception.snapshot.hazard.available:
                break
        self.assertTrue(world.perception.snapshot.hazard.available)

        for _ in range(400):
            world.step(command, 1.0 / 60.0)
            if not world.perception.snapshot.hazard.available:
                break
        self.assertFalse(world.perception.snapshot.hazard.available)


class ResearchIntegrationTests(unittest.TestCase):
    def test_narrow_passage_goal_sensed_but_front_sector_not_traversable(self):
        scene = ScenarioDistribution("narrow_passage").sample(901)
        world = Environment(scene)
        snapshot = world.perception.snapshot
        front = snapshot.hazard.sector_ranges[8]

        self.assertEqual(world.perception.sensing_profile, "research")
        self.assertTrue(snapshot.goal.sensed)
        self.assertTrue(snapshot.goal.available)
        self.assertLess(front.clearance, snapshot.goal.distance)

    def test_research_bt_conditions_obey_finite_range_semantics(self):
        inside_world = Environment(
            make_research_scene(obstacles=[(176, 290, 20, 20)])
        )
        inside_controller = BehaviorTreeController(
            inside_world, bt_config="condition_research"
        )
        inside_controller.tick(1.0 / 60.0)
        self.assertEqual(inside_controller.active_behavior, "Avoid Hazard")

        hazard_range = int(DEFAULT_RESEARCH_SENSOR_CONFIG["hazard_range"])
        outside_hazard_world = Environment(
            make_research_scene(
                obstacles=[(100 + 16 + hazard_range + 1, 290, 10, 20)]
            )
        )
        outside_hazard_controller = BehaviorTreeController(
            outside_hazard_world, bt_config="condition_research"
        )
        outside_hazard_controller.tick(1.0 / 60.0)
        self.assertEqual(
            outside_hazard_controller.nodes_by_name["Hazard Risk?"].status,
            py_trees.common.Status.FAILURE,
        )
        self.assertEqual(
            outside_hazard_controller.active_behavior,
            "Move To Goal",
        )

        goal_range = float(DEFAULT_RESEARCH_SENSOR_CONFIG["goal_range"])
        outside_world = Environment(
            make_research_scene(
                target_position=(100.0 + goal_range + 1.0, 300.0),
            )
        )
        outside_controller = BehaviorTreeController(
            outside_world, bt_config="condition_research"
        )
        outside_controller.tick(1.0 / 60.0)
        self.assertEqual(
            outside_controller.nodes_by_name["Goal Reached?"].status,
            py_trees.common.Status.FAILURE,
        )

    def test_research_boundary_hazard_regression_still_escapes(self):
        world = Environment(ScenarioDistribution("boundary_hazard").sample(902))
        controller = BehaviorTreeController(
            world, bt_config="condition_research"
        )
        escaped_at = None
        for step in range(600):
            turn, throttle = controller.tick(1.0 / 60.0)
            world.step({"turn": turn, "throttle": throttle}, 1.0 / 60.0)
            right_clearance = (
                world.world_size[0]
                - world.agent.radius
                - world.agent.position.x
            )
            if right_clearance > 45.0 and not world.collision_this_step:
                escaped_at = step
                break
        self.assertIsNotNone(escaped_at)
        self.assertLess(escaped_at, 300)

    def test_legacy_profile_keeps_fov_and_twelve_boundary_aware_sectors(self):
        scene = get_scene("rl_sanity")
        scene["target"]["position"] = (20.0, 250.0)
        world = Environment(scene)
        snapshot = world.perception.snapshot

        self.assertEqual(world.perception.sensing_profile, "legacy")
        self.assertFalse(snapshot.goal.sensed)
        self.assertEqual(len(snapshot.hazard.sector_ranges), 12)


if __name__ == "__main__":
    unittest.main()
