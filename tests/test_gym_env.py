"""验证 Gym Adapter 映射现有 World，而不是复制一套仿真。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from autonomy_lab.experiment.recorder import ExperimentRecorder
from autonomy_lab.environment import Environment
from autonomy_lab.gym.env import AgentGymEnv
from autonomy_lab.observation import build_navigation_observation
from autonomy_lab.scene_config import get_scene


class AgentGymEnvCoreTests(unittest.TestCase):
    """每个断言都针对 Gym/World 边界上的可观察契约。"""

    def test_shared_navigation_observation_matches_known_rl_sanity_state(self):
        """共享编码器必须保留 M4 训练时的 13-D 顺序、归一化和 dtype。"""
        world = Environment(get_scene("rl_sanity"))

        observation = build_navigation_observation(world)

        expected = np.array(
            [
                0.0,
                0.0,
                1.0,
                1.0,
                400.0 / np.hypot(700.0, 500.0),
                0.0,
                0.0,
                0.0,
                0.0,
                104.0 / 700.0,
                564.0 / 700.0,
                234.0 / 500.0,
                234.0 / 500.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observation, expected, rtol=0.0, atol=1e-7)
        self.assertEqual(observation.dtype, np.float32)

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

    def test_rl_sanity_starts_with_visible_target_and_no_obstacles(self):
        """M4.0 场景必须让 PPO 从第一帧就获得合法的感知目标信息。"""
        self.env = AgentGymEnv(scenario="rl_sanity", render_mode=None)

        observation, _ = self.env.reset(seed=1042)

        self.assertEqual(observation.shape, (13,))
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(len(self.env.world.obstacles), 0)
        self.assertEqual(observation[3], 1.0)  # target_visible
        self.assertGreater(observation[4], 0.0)  # perceived target distance

    def test_ppo_obstacle_scene_has_visible_target_and_impassable_gap(self):
        """狭缝允许中心 LOS 穿过，但净宽必须小于 Agent 碰撞直径。"""
        self.env = AgentGymEnv(
            scenario="ppo_simple_obstacles", render_mode=None
        )

        observation, _ = self.env.reset(seed=2001)
        upper, lower = sorted(self.env.world.obstacles, key=lambda rect: rect.top)
        gap_width = lower.top - upper.bottom

        self.assertEqual(self.env.world.world_size, (850, 600))
        self.assertEqual(len(self.env.world.obstacles), 2)
        self.assertEqual(gap_width, 30)
        self.assertEqual(self.env.world.agent.radius * 2, 32)
        self.assertLess(gap_width, self.env.world.agent.radius * 2)
        self.assertEqual(observation.shape, (13,))
        self.assertEqual(observation[3], 1.0)  # target_visible through the slit

    def test_ppo_obstacle_scene_straight_action_cannot_cross_gap(self):
        """若圆形碰撞退化成点碰撞，持续直行会错误穿过 30 px 狭缝。"""
        self.env = AgentGymEnv(
            scenario="ppo_simple_obstacles", render_mode=None
        )
        self.env.reset(seed=2001)

        collided = False
        for _ in range(120):
            _, _, terminated, _, info = self.env.step(
                np.array([0.0, 1.0], dtype=np.float32)
            )
            collided = collided or info["collision"]
            self.assertFalse(terminated)

        obstacle_left = min(rect.left for rect in self.env.world.obstacles)
        self.assertTrue(collided)
        # 分帧移动会停在最后一个安全位置；只要求圆心未越过障碍左边界。
        # 若碰撞错误退化为点模型，圆心会穿过狭缝并越过整个障碍。
        self.assertLess(self.env.world.agent.position.x, obstacle_left)

    def test_simple_beacon_scene_keeps_target_and_obstacle_perception(self):
        """关闭 Target 遮挡后，单障碍仍须正常进入最近障碍 Observation。"""
        self.env = AgentGymEnv(scenario="ppo_simple_obstacle", render_mode=None)

        observation, _ = self.env.reset(seed=3001)
        obstacle = self.env.world.obstacles[0]
        target_line = (
            tuple(self.env.world.agent.position),
            tuple(self.env.world.target),
        )

        self.assertEqual(self.env.world.world_size, (850, 600))
        self.assertEqual(len(self.env.world.obstacles), 1)
        self.assertTrue(obstacle.clipline(*target_line))  # direct path is blocked
        self.assertEqual(observation.shape, (13,))
        self.assertEqual(observation[3], 1.0)  # non-occluded beacon target
        self.assertEqual(observation[6], 1.0)  # obstacle perception remains active

    def test_non_occluded_beacon_still_respects_range_and_fov(self):
        """los_enabled=False 不能退化成无条件 Ground Truth Target Observation。"""
        out_of_range_scene = get_scene("ppo_simple_obstacle")
        out_of_range_scene["target"]["position"] = (810, 300)
        out_of_range_world = Environment(out_of_range_scene)

        outside_fov_scene = get_scene("ppo_simple_obstacle")
        outside_fov_scene["agent"]["heading_degrees"] = 180.0
        outside_fov_world = Environment(outside_fov_scene)

        self.assertFalse(out_of_range_world.perception.snapshot.target_visible)
        self.assertEqual(
            out_of_range_world.perception.snapshot.target_unavailable_reason,
            "out of range",
        )
        self.assertFalse(outside_fov_world.perception.snapshot.target_visible)
        self.assertEqual(
            outside_fov_world.perception.snapshot.target_unavailable_reason,
            "outside FOV",
        )

    def test_progress_reward_prefers_motion_toward_visible_target(self):
        """Ground Truth 距离只塑造 reward，不改变 Observation 字段。"""
        moving_env = AgentGymEnv(scenario="rl_sanity", render_mode=None)
        stationary_env = AgentGymEnv(scenario="rl_sanity", render_mode=None)
        try:
            moving_env.reset(seed=1042)
            stationary_env.reset(seed=1042)

            moving_observation, moving_reward, *_ = moving_env.step(
                np.array([0.0, 1.0], dtype=np.float32)
            )
            stationary_observation, stationary_reward, *_ = stationary_env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )

            self.assertGreater(moving_reward, stationary_reward)
            self.assertEqual(moving_observation.shape, stationary_observation.shape)
            self.assertEqual(moving_observation.shape, (13,))
        finally:
            moving_env.close()
            stationary_env.close()

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

    def test_default_control_settings_preserve_one_simulation_step(self):
        """默认 Gym 行为仍是一决策对应一个 1/60 s World step。"""
        self.env = AgentGymEnv(scenario="rl_sanity", render_mode=None)
        self.env.reset(seed=44)

        _, reward, _, _, _ = self.env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )

        self.assertEqual(self.env.action_repeat, 1)
        self.assertEqual(self.env.contact_penalty_per_step, 0.0)
        self.assertAlmostEqual(self.env.world.simulation_time, 1.0 / 60.0)
        self.assertAlmostEqual(
            reward,
            self.env.last_reward_components["progress_reward"] - 0.001,
        )

    def test_action_repeat_advances_six_internal_simulation_steps(self):
        """10 Hz 决策只重复 Command，不改变 60 Hz World 动力学。"""
        self.env = AgentGymEnv(
            scenario="rl_sanity",
            render_mode=None,
            action_repeat=6,
        )
        self.env.reset(seed=44)

        observation, reward, terminated, truncated, _ = self.env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(self.env.world.simulation_time, 6.0 / 60.0)
        self.assertAlmostEqual(self.env.world.agent.position.x, 142.0)
        self.assertEqual(observation[0], 1.0)
        components = self.env.last_reward_components
        self.assertEqual(components["internal_simulation_steps"], 6)
        self.assertAlmostEqual(components["step_reward"], -0.006)
        self.assertAlmostEqual(reward, sum(
            components[name]
            for name in (
                "progress_reward",
                "step_reward",
                "collision_event_reward",
                "contact_penalty_reward",
                "goal_reward",
            )
        ))

    def test_action_repeat_stops_immediately_at_time_limit(self):
        """macro step 不得在 World 已 truncated 后继续消耗 repeat。"""
        self.env = AgentGymEnv(
            scenario="rl_sanity",
            render_mode=None,
            action_repeat=6,
        )
        self.env.world.scene_config["experiment"]["max_episode_time"] = 2.0 / 60.0
        self.env.reset(seed=44)

        _, _, terminated, truncated, _ = self.env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(self.env.world.simulation_time, 2.0 / 60.0)
        self.assertEqual(
            self.env.last_reward_components["internal_simulation_steps"], 2
        )
        self.assertAlmostEqual(
            self.env.last_reward_components["step_reward"], -0.002
        )

    def test_contact_penalty_accumulates_per_internal_contact_step(self):
        """事件罚分只记首次接触，contact shaping 则按接触物理时间累计。"""
        self.env = AgentGymEnv(
            scenario="simple",
            render_mode=None,
            action_repeat=6,
            contact_penalty_per_step=-0.002,
        )
        self.env.world.scene_config["agent"]["position"] = (233, 350)
        self.env.reset(seed=42)

        _, reward, _, _, info = self.env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )

        components = self.env.last_reward_components
        self.assertTrue(info["collision"])
        self.assertEqual(components["collision_event_count"], 1)
        self.assertEqual(components["contact_steps"], 6)
        self.assertAlmostEqual(components["contact_duration"], 0.1)
        self.assertAlmostEqual(components["collision_event_reward"], -0.05)
        self.assertAlmostEqual(components["contact_penalty_reward"], -0.012)
        self.assertAlmostEqual(reward, -0.068)

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
