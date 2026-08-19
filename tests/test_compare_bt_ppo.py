"""验证 M4.2 公平比较入口的初态、时钟和输出契约。"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO

from autonomy_lab.environment import Environment
from autonomy_lab.gym.env import AgentGymEnv
from autonomy_lab.scene_config import get_scene
from compare_bt_ppo import (
    BT_DECISION_FREQUENCY,
    PPO_ACTION_REPEAT,
    PPO_DECISION_FREQUENCY,
    assert_initial_states_match,
    capture_initial_state,
    run_bt_episode,
    run_ppo_episode,
    summarize_rows,
    write_comparison_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m41b_control10hz.zip"


class BTvsPPOComparisonTests(unittest.TestCase):
    def test_initial_state_match_uses_tolerance_for_float_state(self):
        bt_world = Environment(get_scene("ppo_simple_obstacle"))
        ppo_env = AgentGymEnv(scenario="ppo_simple_obstacle")
        try:
            bt_world.reset(seed=4001)
            ppo_env.reset(seed=4001)
            bt_state = capture_initial_state(bt_world)
            ppo_state = capture_initial_state(ppo_env.world)

            ppo_state["agent_position"][0] += 1e-10
            assert_initial_states_match(bt_state, ppo_state)
            ppo_state["agent_position"][0] += 1e-3
            with self.assertRaisesRegex(AssertionError, "agent_position"):
                assert_initial_states_match(bt_state, ppo_state)
        finally:
            ppo_env.close()

    def test_bt_and_ppo_episode_metrics_use_world_simulation_time(self):
        model = PPO.load(MODEL_PATH, device="cpu")
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            bt_row, bt_state = run_bt_episode(
                "rl_sanity", 4001, output_path / "bt"
            )
            ppo_row, ppo_state = run_ppo_episode(
                "rl_sanity", 4001, output_path / "ppo", model
            )

        assert_initial_states_match(bt_state, ppo_state)
        self.assertEqual(bt_row["decision_frequency_hz"], BT_DECISION_FREQUENCY)
        self.assertEqual(ppo_row["decision_frequency_hz"], PPO_DECISION_FREQUENCY)
        self.assertEqual(PPO_ACTION_REPEAT, 6)
        self.assertAlmostEqual(
            bt_row["elapsed_time"],
            bt_row["decision_count"] / BT_DECISION_FREQUENCY,
            places=6,
        )
        # PPO 最后一个 macro step 可能提前到达，因此实际时间不超过 decisions/10。
        self.assertGreater(ppo_row["elapsed_time"], 0.0)
        self.assertLessEqual(
            ppo_row["elapsed_time"],
            ppo_row["decision_count"] / PPO_DECISION_FREQUENCY + 1e-9,
        )

    def test_outputs_group_each_scenario_without_mixed_overall_rate(self):
        rows = [
            {
                "controller": controller,
                "scenario": scenario,
                "scenario_role": role,
                "seed": 4001,
                "success": success,
                "elapsed_time": 2.0,
                "path_length": 100.0,
                "collision_count": 0,
                "termination_reason": "target_reached" if success else "timeout",
                "decision_frequency_hz": 60.0 if controller == "bt" else 10.0,
                "decision_count": 120 if controller == "bt" else 20,
            }
            for scenario, role in (
                ("rl_sanity", "primary"),
                ("ppo_simple_obstacles", "hard_stress_test"),
            )
            for controller, success in (("bt", True), ("ppo", False))
        ]

        summary_rows = summarize_rows(rows)
        self.assertEqual(len(summary_rows), 4)
        self.assertFalse(any(row["scenario"] == "overall" for row in summary_rows))

        with tempfile.TemporaryDirectory() as output_dir:
            csv_path, summary_path = write_comparison_outputs(
                rows,
                Path(output_dir),
                MODEL_PATH,
                seeds=(4001,),
            )
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(saved_rows), 4)
        self.assertEqual(len(summary["controller_scenario_summary"]), 4)
        self.assertIn("fixed layouts", summary["seed_interpretation"])


if __name__ == "__main__":
    unittest.main()
