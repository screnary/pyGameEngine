"""验证唯一 World 步进入口的状态顺序和重置语义。"""

import unittest

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.environment import Environment
from autonomy_lab.scene_config import get_scene


class EnvironmentStepTests(unittest.TestCase):
    """这些测试防止 Gym/BT 各自绕过 World 实现第二套运动逻辑。"""

    def test_step_updates_motion_time_termination_then_perception(self):
        world = Environment(get_scene("simple"))
        initial_distance = world.perception.snapshot.target_distance

        world.step({"turn": 0.0, "throttle": 1.0}, 0.1)

        self.assertAlmostEqual(world.agent.position.x, 122.0)
        self.assertAlmostEqual(world.agent.position.y, 350.0)
        self.assertAlmostEqual(world.simulation_time, 0.1)
        self.assertFalse(world.target_reached)
        # 目标距离必须来自 action 应用后的最终位置，而不是本步开始时的旧快照。
        self.assertAlmostEqual(
            world.perception.snapshot.target_distance,
            initial_distance - 22.0,
        )

    def test_step_resolves_collision_before_refreshing_perception(self):
        scene = get_scene("simple")
        scene["agent"]["position"] = (230, 350)
        scene["target"]["position"] = (850, 350)
        world = Environment(scene)

        world.step({"turn": 0.0, "throttle": 1.0}, 0.1)

        self.assertTrue(world.collision_this_step)
        self.assertAlmostEqual(world.agent.position.x, 230.0)
        self.assertAlmostEqual(
            world.perception.snapshot.target_distance,
            620.0,
        )

    def test_reset_clears_episode_state_and_applies_seed(self):
        world = Environment(get_scene("simple"))
        world.step({"turn": 0.0, "throttle": 1.0}, 0.1)

        world.reset(seed=123)

        self.assertEqual(world.seed, 123)
        self.assertEqual(world.simulation_time, 0.0)
        self.assertFalse(world.collision_this_step)
        self.assertEqual(world.agent.position, (100, 350))
        self.assertIsNotNone(world.perception.snapshot)

    def test_step_rejects_incomplete_command_and_negative_dt(self):
        world = Environment(get_scene("simple"))

        with self.assertRaisesRegex(ValueError, "turn.*throttle"):
            world.step({"turn": 0.0}, 0.1)
        with self.assertRaisesRegex(ValueError, "dt"):
            world.step({"turn": 0.0, "throttle": 0.0}, -0.1)

    def test_bt_controller_reuses_world_perception(self):
        world = Environment(get_scene("simple"))

        controller = BehaviorTreeController(world)

        self.assertIs(controller.perception, world.perception)


if __name__ == "__main__":
    unittest.main()
