"""验证目标、障碍物、射线安全距离和局部缺口感知。"""

import math
import unittest

import pygame

from autonomy_lab.environment import Environment
from autonomy_lab.perception import AgentPerception
from autonomy_lab.scene_config import get_scene


def make_environment(
    *,
    agent_position=(300.0, 300.0),
    heading_degrees=0.0,
    target_position=(400.0, 300.0),
    obstacles=(),
    mode="perceived",
    behavior_tree_updates=None,
):
    """围绕固定测试 Agent 创建可精确控制几何关系的场景。"""
    scene = get_scene("simple")
    scene["agent"]["position"] = agent_position
    scene["agent"]["heading_degrees"] = heading_degrees
    scene["target"]["position"] = target_position
    scene["obstacles"] = list(obstacles)
    scene["sensor"] = {
        "range": 300.0,
        "fov_degrees": 120.0,
        "los_enabled": True,
    }
    scene["target_information_mode"] = mode
    if behavior_tree_updates:
        scene["behavior_tree"].update(behavior_tree_updates)
    return Environment(scene)


class TargetPerceptionTests(unittest.TestCase):
    def test_target_ahead_is_visible_and_available_in_perceived_mode(self):
        environment = make_environment()

        snapshot = AgentPerception(environment).update()

        self.assertTrue(snapshot.target_visible)
        self.assertTrue(snapshot.target_available)
        self.assertEqual(snapshot.target_source, "perception")
        self.assertAlmostEqual(snapshot.target_distance, 100.0)
        self.assertAlmostEqual(snapshot.target_bearing, 0.0)
        self.assertEqual(snapshot.target_unavailable_reason, "")

    def test_target_behind_is_hidden_without_leaking_measurements(self):
        environment = make_environment(target_position=(200.0, 300.0))

        snapshot = AgentPerception(environment).update()

        self.assertFalse(snapshot.target_visible)
        self.assertFalse(snapshot.target_available)
        self.assertIsNone(snapshot.target_source)
        self.assertIsNone(snapshot.target_distance)
        self.assertIsNone(snapshot.target_bearing)
        self.assertEqual(snapshot.target_unavailable_reason, "outside FOV")

    def test_target_outside_range_is_hidden(self):
        environment = make_environment(target_position=(601.0, 300.0))

        snapshot = AgentPerception(environment).update()

        self.assertFalse(snapshot.target_visible)
        self.assertFalse(snapshot.target_available)
        self.assertEqual(snapshot.target_unavailable_reason, "out of range")

    def test_relative_bearing_wraps_across_plus_minus_180_degrees(self):
        target_angle = math.radians(-179.0)
        target_position = (
            300.0 + math.cos(target_angle) * 100.0,
            300.0 + math.sin(target_angle) * 100.0,
        )
        environment = make_environment(
            heading_degrees=179.0, target_position=target_position
        )

        snapshot = AgentPerception(environment).update()

        self.assertTrue(snapshot.target_visible)
        self.assertAlmostEqual(snapshot.target_bearing, math.radians(2.0), places=6)

    def test_obstacle_blocks_target_line_of_sight(self):
        environment = make_environment(
            agent_position=(100.0, 100.0),
            target_position=(300.0, 100.0),
            obstacles=[(190, 50, 20, 100)],
        )

        snapshot = AgentPerception(environment).update()

        self.assertFalse(snapshot.target_visible)
        self.assertFalse(snapshot.target_available)
        self.assertEqual(snapshot.target_unavailable_reason, "occluded")

    def test_ground_truth_mode_keeps_target_available_when_sensor_cannot_see_it(self):
        environment = make_environment(
            target_position=(200.0, 300.0), mode="ground_truth"
        )

        snapshot = AgentPerception(environment).update()

        self.assertFalse(snapshot.target_visible)
        self.assertTrue(snapshot.target_available)
        self.assertEqual(snapshot.target_source, "ground_truth")
        self.assertAlmostEqual(snapshot.target_distance, 100.0)
        self.assertAlmostEqual(abs(snapshot.target_bearing), math.pi)
        self.assertEqual(snapshot.target_unavailable_reason, "outside FOV")

    def test_unknown_target_information_mode_is_rejected(self):
        environment = make_environment(mode="telepathy")

        with self.assertRaisesRegex(ValueError, "target_information_mode"):
            AgentPerception(environment)


class ObstaclePerceptionTests(unittest.TestCase):
    def test_visible_obstacles_use_nearest_points_and_are_sorted_by_clearance(self):
        environment = make_environment(
            obstacles=[
                (500, 280, 20, 40),
                (400, 280, 20, 40),
                (200, 280, 20, 40),
                (650, 280, 20, 40),
            ]
        )

        snapshot = AgentPerception(environment).update()

        self.assertEqual(
            [item.rect for item in snapshot.visible_obstacles],
            [environment.obstacles[1], environment.obstacles[0]],
        )
        self.assertAlmostEqual(snapshot.visible_obstacles[0].distance, 84.0)
        self.assertAlmostEqual(snapshot.visible_obstacles[0].bearing, 0.0)
        self.assertEqual(snapshot.nearest_obstacle, snapshot.visible_obstacles[0])

    def test_obstacle_outside_fov_is_not_visible(self):
        environment = make_environment(obstacles=[(280, 400, 40, 20)])

        snapshot = AgentPerception(environment).update()

        self.assertEqual(snapshot.visible_obstacles, ())
        self.assertIsNone(snapshot.nearest_obstacle)


class GapPerceptionTests(unittest.TestCase):
    def test_open_forward_space_produces_a_full_range_centered_gap(self):
        environment = make_environment(
            target_position=(100.0, 300.0),
            obstacles=[],
        )

        snapshot = AgentPerception(environment).update()
        gap = getattr(snapshot, "best_exploration_gap", None)

        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap.bearing, 0.0, places=6)
        self.assertAlmostEqual(gap.free_distance, 300.0, places=6)
        self.assertGreater(gap.angular_width, 0.0)
        self.assertTrue(hasattr(gap, "entry_position"))
        entry_distance = (
            pygame.Vector2(gap.entry_position) - environment.agent.position
        ).length()
        self.assertAlmostEqual(entry_distance, 240.0, places=5)

    def test_simple_ground_truth_wall_selects_a_target_aligned_side_gap(self):
        scene = get_scene("simple")
        scene["target_information_mode"] = "ground_truth"
        environment = Environment(scene)

        snapshot = AgentPerception(environment).update()
        target_path_blocked = getattr(snapshot, "target_path_blocked", None)
        target_gap = getattr(snapshot, "best_target_gap", None)

        self.assertTrue(target_path_blocked)
        self.assertIsNotNone(target_gap)
        self.assertGreater(target_gap.bearing, math.radians(30.0))
        self.assertGreater(target_gap.free_distance, 250.0)
        self.assertGreater(target_gap.entry_position[1], 454.0)

    def test_simple_wide_fov_detects_narrow_channel_at_runtime_approach(self):
        scene = get_scene("simple")
        environment = Environment(scene)
        environment.agent.position.update(674.7, 439.9)
        environment.agent.heading = math.radians(-26.7)

        snapshot = AgentPerception(environment).update()

        self.assertTrue(snapshot.target_path_blocked)
        self.assertIsNotNone(snapshot.best_target_gap)
        self.assertLess(snapshot.best_target_gap.bearing, math.radians(-60.0))
        self.assertGreater(snapshot.best_target_gap.free_distance, 290.0)

    def test_clear_or_out_of_fov_target_path_is_not_reported_blocked(self):
        cases = (
            make_environment(
                target_position=(700.0, 300.0), obstacles=[], mode="ground_truth"
            ),
            make_environment(
                target_position=(200.0, 300.0), obstacles=[], mode="ground_truth"
            ),
        )

        for environment in cases:
            with self.subTest(target=tuple(environment.target)):
                snapshot = AgentPerception(environment).update()
                self.assertTrue(hasattr(snapshot, "target_path_blocked"))
                self.assertFalse(snapshot.target_path_blocked)
                self.assertIsNone(getattr(snapshot, "best_target_gap", None))

    def test_wide_opening_is_detected_but_safe_diameter_closes_narrow_opening(self):
        wide_environment = make_environment(
            target_position=(100.0, 300.0),
            obstacles=[(400, 0, 30, 250), (400, 350, 30, 350)],
        )
        narrow_environment = make_environment(
            target_position=(100.0, 300.0),
            obstacles=[(400, 0, 30, 287), (400, 313, 30, 387)],
        )

        wide_snapshot = AgentPerception(wide_environment).update()
        narrow_snapshot = AgentPerception(narrow_environment).update()

        wide_gap = getattr(wide_snapshot, "best_exploration_gap", None)
        self.assertIsNotNone(wide_gap)
        self.assertAlmostEqual(wide_gap.bearing, 0.0, places=6)
        narrow_gaps = getattr(narrow_snapshot, "traversable_gaps", ())
        self.assertFalse(
            any(abs(gap.bearing) < math.radians(20.0) for gap in narrow_gaps)
        )
        self.assertGreater(
            abs(narrow_snapshot.best_exploration_gap.bearing),
            math.radians(20.0),
        )

    def test_safe_world_boundary_caps_gap_distance(self):
        environment = make_environment(
            agent_position=(820.0, 350.0),
            target_position=(100.0, 350.0),
            obstacles=[],
            behavior_tree_updates={"gap_min_travel_distance": 40.0},
        )

        snapshot = AgentPerception(environment).update()
        gap = getattr(snapshot, "best_exploration_gap", None)

        self.assertIsNotNone(gap)
        self.assertGreater(abs(gap.bearing), math.radians(50.0))
        self.assertGreater(gap.free_distance, 90.0)
        self.assertLess(gap.free_distance, 120.0)

    def test_agent_inside_boundary_margin_can_still_select_an_inward_gap(self):
        environment = make_environment(
            agent_position=(300.0, 680.0),
            heading_degrees=-90.0,
            target_position=(100.0, 100.0),
            obstacles=[],
        )

        snapshot = AgentPerception(environment).update()
        gap = snapshot.best_exploration_gap

        self.assertIsNotNone(gap)
        self.assertAlmostEqual(gap.bearing, 0.0, places=6)
        self.assertAlmostEqual(gap.free_distance, 300.0, places=6)

    def test_invalid_gap_perception_settings_are_rejected(self):
        invalid_settings = (
            ({"gap_ray_count": 2}, "gap_ray_count"),
            ({"gap_min_travel_distance": 0.0}, "gap_min_travel_distance"),
            ({"gap_safety_margin": -1.0}, "gap_safety_margin"),
            ({"gap_open_ratio": 0.0}, "gap_open_ratio"),
            ({"gap_open_ratio": 1.1}, "gap_open_ratio"),
            ({"gap_entry_ratio": 0.0}, "gap_entry_ratio"),
            ({"gap_entry_ratio": 1.0}, "gap_entry_ratio"),
            (
                {"gap_entry_reached_distance": 0.0},
                "gap_entry_reached_distance",
            ),
        )
        for updates, message in invalid_settings:
            with self.subTest(updates=updates):
                environment = make_environment(behavior_tree_updates=updates)
                with self.assertRaisesRegex(ValueError, message):
                    AgentPerception(environment)


if __name__ == "__main__":
    unittest.main()
