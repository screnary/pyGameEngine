"""复现并保护 R0.1 的感知分离与局部安全转向问题。"""

import math
import unittest

import py_trees

from autonomy_lab.bt.behaviors import AvoidObstacle, ObstacleThreat
from autonomy_lab.bt.context import BehaviorBuildContext
from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.config import SCENES, get_scene


SIMULATION_DT = 1.0 / 60.0


def make_boundary_obstacle_world() -> Environment:
    """构造会让旧 Boundary Recovery 朝障碍物持续推进的精确几何。"""
    scene = get_scene("ppo_simple_obstacle")
    scene["target_information_mode"] = "ground_truth"
    scene["obstacles"] = [(670, 200, 120, 200)]
    scene["agent"]["position"] = (810, 300)
    scene["agent"]["heading_degrees"] = 180.0
    return Environment(scene)


class R01PerceptionRegressionTests(unittest.TestCase):
    def test_optical_goal_sensing_is_independent_from_footprint_clearance(self):
        """若 Goal LOS 又使用 footprint 膨胀障碍或 sector 未实现，本测试失败。"""
        scene = get_scene("ppo_simple_obstacles")
        scene["sensor"]["los_enabled"] = True
        # 24 px 光学狭缝小于 32 px Agent 直径，但中心线仍能看到 Goal。
        scene["obstacles"] = [(350, 80, 80, 208), (350, 312, 80, 208)]
        world = Environment(scene)

        snapshot = world.perception.snapshot
        sectors = getattr(snapshot, "sector_clearances", ())

        self.assertTrue(snapshot.target_visible)
        self.assertTrue(snapshot.target_available)
        self.assertTrue(snapshot.target_path_blocked)
        self.assertEqual(len(sectors), 12)
        center = min(sectors, key=lambda sector: abs(sector.bearing))
        self.assertAlmostEqual(center.bearing, 0.0, places=6)
        self.assertLess(center.clearance, snapshot.target_distance)
        self.assertFalse(
            any(
                abs(gap.bearing) < math.radians(10.0)
                for gap in snapshot.traversable_gaps
            )
        )

        controller = BehaviorTreeController(world, bt_config="default")
        controller.tick(SIMULATION_DT)
        self.assertNotEqual(controller.active_behavior, "Search Target")

    def test_requested_regression_scenarios_are_runnable_fixed_presets(self):
        """场景遗漏或 reset 后不能生成稳定 World 时，本测试失败。"""
        for name in ("r01_narrow_passage", "r01_boundary_obstacle"):
            with self.subTest(name=name):
                self.assertIn(name, SCENES)
                first = Environment(get_scene(name))
                second = Environment(get_scene(name))
                self.assertEqual(tuple(first.agent.position), tuple(second.agent.position))
                self.assertEqual(tuple(first.target), tuple(second.target))
                self.assertEqual(first.obstacles, second.obstacles)

    def test_narrow_passage_does_not_fall_into_a_search_deadlock(self):
        """光学可见但不可穿越的狭缝不能把已感知 Goal 错判成长期未知。"""
        world = Environment(get_scene("r01_narrow_passage"))
        controller = BehaviorTreeController(world, bt_config="default")
        longest_search_run = 0
        current_search_run = 0
        sensed_steps = 0

        for _ in range(600):
            turn, throttle = controller.tick(SIMULATION_DT)
            sensed_steps += int(world.perception.snapshot.target_visible)
            if controller.active_behavior == "Search Target":
                current_search_run += 1
                longest_search_run = max(longest_search_run, current_search_run)
            else:
                current_search_run = 0
            world.step({"turn": turn, "throttle": throttle}, SIMULATION_DT)

        self.assertGreater(sensed_steps, 0)
        self.assertLess(longest_search_run, 120)


class R01SafeSteeringRegressionTests(unittest.TestCase):
    def test_boundary_recovery_turns_around_blocked_inward_direction(self):
        """若 Recovery 仍只朝 World 中心，Agent 会在 x≈806 永久碰撞。"""
        world = make_boundary_obstacle_world()
        controller = BehaviorTreeController(world, bt_config="hybrid_ppo")

        first_turn, _ = controller.tick(SIMULATION_DT)
        self.assertEqual(controller.active_behavior, "Safe Boundary Recovery")
        self.assertGreater(abs(first_turn), 0.5)

        escaped = False
        collision_steps = 0
        for _ in range(300):
            turn, throttle = controller.tick(SIMULATION_DT)
            world.step({"turn": turn, "throttle": throttle}, SIMULATION_DT)
            collision_steps += int(world.collision_this_step)
            right_clearance = (
                world.world_size[0] - world.agent.radius - world.agent.position.x
            )
            if right_clearance > 40.0:
                escaped = True
                break

        self.assertTrue(escaped)
        self.assertLess(collision_steps, 120)

    def test_obstacle_avoidance_does_not_choose_the_nearby_top_boundary(self):
        """若 Avoid 仍只看 threat bearing，它会输出 -1 主动转向上边界。"""
        scene = get_scene("simple")
        scene["agent"]["position"] = (500, 35)
        scene["agent"]["heading_degrees"] = 0.0
        scene["obstacles"] = [(550, 40, 60, 80)]
        scene["target_information_mode"] = "ground_truth"
        world = Environment(scene)
        command = {"turn": 0.0, "throttle": 0.0}
        context = BehaviorBuildContext(
            perception=world.perception,
            command=command,
            behavior_config=world.scene_config["behavior_tree"],
        )
        condition = ObstacleThreat(context=context, name="Obstacle Threat?")
        context.nodes_by_name[condition.name] = condition
        action = AvoidObstacle(
            context=context,
            name="Avoid Obstacle",
            condition=condition.name,
        )

        self.assertEqual(condition.update(), py_trees.common.Status.SUCCESS)
        action.initialise()
        action.update()

        self.assertGreater(command["turn"], 0.0)
        self.assertIn("safe steering", action.feedback_message)

    def test_default_bt_boundary_obstacle_scene_leaves_the_trap_without_oscillation(self):
        """若 Boundary 分支缺失或两个 Safety Action 互推，场景不能稳定脱困。"""
        self.assertIn("r01_boundary_obstacle", SCENES)
        world = Environment(get_scene("r01_boundary_obstacle"))
        controller = BehaviorTreeController(world, bt_config="default")
        branches: list[str | None] = []
        escaped_at: int | None = None

        for step in range(600):
            turn, throttle = controller.tick(SIMULATION_DT)
            branches.append(controller.active_behavior)
            world.step({"turn": turn, "throttle": throttle}, SIMULATION_DT)
            right_clearance = (
                world.world_size[0] - world.agent.radius - world.agent.position.x
            )
            if right_clearance > 45.0 and not world.collision_this_step:
                escaped_at = step
                break

        switches = sum(left != right for left, right in zip(branches, branches[1:]))
        self.assertIsNotNone(escaped_at)
        self.assertLess(escaped_at, 300)
        self.assertLessEqual(switches, 6)
        self.assertEqual(branches[0], "Safe Boundary Recovery")


if __name__ == "__main__":
    unittest.main()
