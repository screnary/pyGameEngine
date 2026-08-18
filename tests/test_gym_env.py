"""验证 Gym Adapter 映射现有 World，而不是复制一套仿真。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from autonomy_lab.experiment import ExperimentRecorder
from autonomy_lab.gym_env import AgentGymEnv


class AgentGymEnvCoreTests(unittest.TestCase):
    """每个断言都针对 Gym/World 边界上的可观察契约。"""

    def tearDown(self):
        env = getattr(self, "env", None)
        if env is not None:
            env.close()

    def test_headless_reset_returns_fixed_perception_observation(self):
        self.env = AgentGymEnv(scenario="simple", render_mode=None)

        observation, info = self.env.reset(seed=42)

        self.assertEqual(observation.shape, (13,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertIsNone(self.env._renderer)
        self.assertEqual(info["scenario"], "simple")
        self.assertEqual(info["simulation_time"], 0.0)

    def test_invisible_target_does_not_leak_ground_truth_distance_or_bearing(self):
        # simple 使用 ground_truth BT 模式，但起点到目标的视线被左侧墙体遮挡。
        self.env = AgentGymEnv(scenario="simple", render_mode=None)

        observation, _ = self.env.reset(seed=42)

        self.assertEqual(observation[3], 0.0)  # target_visible
        self.assertEqual(observation[4], 0.0)  # neutral target_distance
        self.assertEqual(observation[5], 0.0)  # neutral target_bearing

    def test_action_maps_turn_then_throttle_to_existing_world_command(self):
        self.env = AgentGymEnv(scenario="simple", render_mode=None)
        self.env.reset(seed=42)

        observation, _, _, _, _ = self.env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )

        expected_x = 100.0 + 220.0 / 60.0
        self.assertAlmostEqual(self.env.world.agent.position.x, expected_x, places=5)
        self.assertAlmostEqual(self.env.world.agent.position.y, 350.0, places=5)
        self.assertAlmostEqual(self.env.world.simulation_time, 1.0 / 60.0)
        self.assertEqual(observation[0], 1.0)  # normalized speed

    def test_observation_is_built_after_action_final_state(self):
        self.env = AgentGymEnv(scenario="simple", render_mode=None)
        self.env.reset(seed=42)

        observation, _, _, _, _ = self.env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )

        expected_heading = np.deg2rad(150.0) / 60.0
        self.assertAlmostEqual(observation[1], np.sin(expected_heading), places=6)
        self.assertAlmostEqual(observation[2], np.cos(expected_heading), places=6)

    def test_target_reached_is_terminated_not_truncated(self):
        self.env = AgentGymEnv(scenario="simple", render_mode=None)
        self.env.world.scene_config["target"]["position"] = (100, 350)
        self.env.reset(seed=42)

        _, reward, terminated, truncated, info = self.env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 0.999)
        self.assertTrue(info["target_reached"])

    def test_time_limit_is_truncated_not_terminated(self):
        self.env = AgentGymEnv(scenario="simple", render_mode=None)
        self.env.world.scene_config["experiment"]["max_episode_time"] = 1.0 / 60.0
        self.env.reset(seed=42)

        _, _, terminated, truncated, _ = self.env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_collision_penalty_is_charged_once_per_contact_event(self):
        self.env = AgentGymEnv(scenario="simple", render_mode=None)
        self.env.world.scene_config["agent"]["position"] = (233, 350)
        self.env.reset(seed=42)
        action = np.array([0.0, 1.0], dtype=np.float32)

        _, first_reward, _, _, first_info = self.env.step(action)
        _, second_reward, _, _, second_info = self.env.step(action)

        self.assertTrue(first_info["collision"])
        self.assertTrue(second_info["collision"])
        self.assertAlmostEqual(first_reward, -0.051)
        self.assertAlmostEqual(second_reward, -0.001)

    def test_reset_with_same_seed_reproduces_initial_observation(self):
        self.env = AgentGymEnv(scenario="dense_obstacles", render_mode=None)

        first, _ = self.env.reset(seed=7)
        self.env.step(np.array([0.5, 1.0], dtype=np.float32))
        second, _ = self.env.reset(seed=7)

        np.testing.assert_array_equal(first, second)

    def test_invalid_render_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "render_mode"):
            AgentGymEnv(render_mode="rgb_array")

    def test_human_render_and_headless_produce_identical_simulation(self):
        headless = AgentGymEnv(scenario="dense_obstacles", render_mode=None)
        human = AgentGymEnv(scenario="dense_obstacles", render_mode="human")
        actions = [
            np.array([turn, throttle], dtype=np.float32)
            for turn, throttle in (
                (0.4, 1.0),
                (0.0, 0.6),
                (-0.25, 0.8),
                (0.0, 1.0),
            )
        ] * 3
        try:
            headless_observation, _ = headless.reset(seed=7)
            human_observation, _ = human.reset(seed=7)
            np.testing.assert_array_equal(headless_observation, human_observation)

            for action in actions:
                headless_result = headless.step(action)
                human_result = human.step(action)
                np.testing.assert_array_equal(headless_result[0], human_result[0])
                self.assertEqual(headless_result[1:], human_result[1:])

            headless_state = (
                tuple(headless.world.agent.position),
                headless.world.agent.heading,
                headless.world.agent.speed,
                headless.world.collision_this_step,
                headless.world.target_reached,
            )
            human_state = (
                tuple(human.world.agent.position),
                human.world.agent.heading,
                human.world.agent.speed,
                human.world.collision_this_step,
                human.world.target_reached,
            )
            self.assertEqual(headless_state, human_state)
        finally:
            headless.close()
            human.close()

    def test_render_does_not_advance_world_state(self):
        self.env = AgentGymEnv(scenario="simple", render_mode="human")
        self.env.reset(seed=42)
        before = (
            tuple(self.env.world.agent.position),
            self.env.world.agent.heading,
            self.env.world.simulation_time,
            self.env.world.perception.snapshot,
        )

        self.env.render()
        self.env.render()

        after = (
            tuple(self.env.world.agent.position),
            self.env.world.agent.heading,
            self.env.world.simulation_time,
            self.env.world.perception.snapshot,
        )
        self.assertEqual(before, after)

    def test_gym_episode_reuses_recorder_collision_event_metrics(self):
        with tempfile.TemporaryDirectory() as output_dir:
            recorder = ExperimentRecorder(Path(output_dir))
            self.env = AgentGymEnv(
                scenario="simple",
                render_mode=None,
                recorder=recorder,
            )
            self.env.world.scene_config["agent"]["position"] = (233, 350)
            self.env.world.scene_config["experiment"]["max_episode_time"] = 2.0 / 60.0
            self.env.reset(seed=42)
            action = np.array([0.0, 1.0], dtype=np.float32)

            _, first_reward, _, first_truncated, _ = self.env.step(action)
            _, second_reward, _, second_truncated, _ = self.env.step(action)

            self.assertFalse(first_truncated)
            self.assertTrue(second_truncated)
            self.assertAlmostEqual(first_reward, -0.051)
            self.assertAlmostEqual(second_reward, -0.001)
            run_path = next((Path(output_dir) / "runs").glob("episode_*.json"))
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["controller"], "gym")
            self.assertEqual(payload["collision_count"], 1)
            self.assertIsNone(payload["bt_tick_count"])
            self.assertIsNone(payload["bt_transition_count"])


if __name__ == "__main__":
    unittest.main()
