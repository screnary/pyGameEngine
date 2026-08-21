"""验证 Hybrid-context Gym Adapter 的 Action ownership 与时间语义。"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autonomy_lab.gym.hybrid_env import HybridPPOEnv
from autonomy_lab.experiment.runners import run_hybrid_policy_episode


class ConstantPolicy:
    def predict(self, observation, deterministic=False):
        del observation
        if not deterministic:
            raise AssertionError("M5.2 evaluation must be deterministic")
        return np.asarray([0.0, 1.0], dtype=np.float32), None


class HybridPPOEnvTests(unittest.TestCase):
    def test_common_runner_records_trained_hybrid_episode_and_diagnostics(self):
        """若评价绕开 Recorder/Hybrid Env 或漏记 PPO diagnostics，应失败。"""
        with tempfile.TemporaryDirectory() as output_dir:
            row, initial_state = run_hybrid_policy_episode(
                "rl_sanity",
                7001,
                Path(output_dir),
                ConstantPolicy(),
            )

        self.assertTrue(row["success"])
        self.assertEqual(row["controller"], "hybrid_trained_ppo")
        self.assertEqual(row["ppo_decision_count"], 17)
        self.assertAlmostEqual(row["ppo_active_ratio"], 1.0)
        self.assertEqual(row["ppo_reentry_count"], 0)
        self.assertEqual(initial_state["agent_position"], [120.0, 250.0])

    def test_visible_target_action_controls_exactly_six_world_steps(self):
        """机械执行错误步数或按 decision step 计时/奖励都应使本测试失败。"""
        env = HybridPPOEnv(scenarios=("rl_sanity",))
        try:
            observation, reset_info = env.reset(seed=52)
            self.assertEqual(observation.shape, (13,))
            self.assertTrue(reset_info["ppo_action_required"])

            _, reward, terminated, truncated, info = env.step(
                np.asarray([0.0, 1.0], dtype=np.float32)
            )

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(env.world.simulation_time, 6.0 / 60.0)
            self.assertEqual(info["internal_simulation_steps"], 6)
            self.assertEqual(info["ppo_controlled_steps"], 6)
            self.assertEqual(info["ppo_decision_count"], 1)
            self.assertAlmostEqual(info["ppo_active_ratio"], 1.0)
            self.assertEqual(
                env.last_reward_components["internal_simulation_steps"], 6
            )
            self.assertAlmostEqual(
                reward,
                sum(
                    float(env.last_reward_components[name])
                    for name in (
                        "progress_reward",
                        "step_reward",
                        "collision_event_reward",
                        "contact_penalty_reward",
                        "goal_reward",
                    )
                ),
            )
        finally:
            env.close()

    def test_boundary_preemption_stops_ppo_action_and_returns_after_reentry(self):
        """若 PPO Action 在 Boundary 接管期间仍被执行，最终位置和计数会失败。"""
        env = HybridPPOEnv(scenarios=("ppo_simple_obstacle",))
        try:
            env.reset(seed=52)
            # 初始 PPO branch 已获控制权；把 Agent 放到风险线外 4 px，并令
            # 负 throttle 沿反向驶向右边界，第二个物理步后应触发 Recovery。
            env.world.agent.position.update(790.0, 300.0)
            env.world.agent.heading = math.pi
            env.world.agent.speed = 0.0
            env.world.perception.update()

            _, _, terminated, truncated, info = env.step(
                np.asarray([0.0, -1.0], dtype=np.float32)
            )

            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["ppo_decision_count"], 1)
            self.assertEqual(info["ppo_controlled_steps"], 2)
            self.assertEqual(info["internal_simulation_steps"], 4)
            self.assertEqual(info["boundary_recovery_activation_count"], 1)
            self.assertEqual(info["ppo_preemption_count"], 1)
            self.assertEqual(info["ppo_reentry_count"], 1)
            self.assertAlmostEqual(info["ppo_active_ratio"], 0.5)
            self.assertIsNotNone(info["observation_before_preemption"])
            self.assertIsNotNone(info["observation_after_reentry"])
            self.assertLess(env.world.agent.position.x, 797.0)
            self.assertTrue(info["ppo_action_required"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
