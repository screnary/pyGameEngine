"""验证 M4 PPO 训练/评估入口的最小稳定契约。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from eval_ppo import (
    EVALUATION_SEEDS,
    is_clearly_better,
    parse_args as parse_evaluation_args,
    run_evaluation,
    summarize_results,
)
from train_ppo import (
    additional_timesteps_to_target,
    parse_args as parse_training_args,
)


class PPOSanityPipelineTests(unittest.TestCase):
    def test_training_cli_selects_obstacle_scenario_checkpoint_and_log_label(self):
        args = parse_training_args(
            [
                "--scenario",
                "ppo_simple_obstacles",
                "--seed",
                "43",
                "--target-timesteps",
                "200000",
                "--model-path",
                "models/ppo_m41_obstacles.zip",
                "--log-label",
                "m41_training",
                "--init-model-path",
                "models/ppo_m40.zip",
            ]
        )

        self.assertEqual(args.scenario, "ppo_simple_obstacles")
        self.assertEqual(args.seed, 43)
        self.assertEqual(args.log_label, "m41_training")
        self.assertEqual(args.init_model_path, Path("models/ppo_m40.zip"))

    def test_evaluation_cli_selects_scenario_and_m41_seed_range(self):
        args = parse_evaluation_args(
            [
                "--scenario",
                "ppo_simple_obstacles",
                "--evaluation-seed-start",
                "2001",
                "--tag",
                "m41_200k",
            ]
        )

        self.assertEqual(args.scenario, "ppo_simple_obstacles")
        self.assertEqual(args.evaluation_seed_start, 2001)
        self.assertEqual(args.tag, "m41_200k")

    def test_m41b_clis_select_action_repeat_and_contact_penalty(self):
        training_args = parse_training_args(
            [
                "--scenario",
                "ppo_simple_obstacle",
                "--action-repeat",
                "6",
                "--contact-penalty-per-step",
                "-0.002",
            ]
        )
        evaluation_args = parse_evaluation_args(
            [
                "--scenario",
                "ppo_simple_obstacle",
                "--action-repeat",
                "6",
                "--contact-penalty-per-step",
                "-0.002",
            ]
        )

        self.assertEqual(training_args.action_repeat, 6)
        self.assertEqual(training_args.contact_penalty_per_step, -0.002)
        self.assertEqual(evaluation_args.action_repeat, 6)
        self.assertEqual(evaluation_args.contact_penalty_per_step, -0.002)

    def test_resume_only_trains_remaining_timesteps(self):
        self.assertEqual(additional_timesteps_to_target(51_200, 100_000), 48_800)
        self.assertEqual(additional_timesteps_to_target(100_352, 100_000), 0)

    def test_evaluation_seeds_are_fixed_and_distinct_from_training_seed(self):
        self.assertEqual(len(EVALUATION_SEEDS), 10)
        self.assertEqual(len(set(EVALUATION_SEEDS)), 10)
        self.assertNotIn(42, EVALUATION_SEEDS)

    def test_summary_uses_experiment_recorder_metric_fields(self):
        payloads = [
            {
                "result": "SUCCESS",
                "elapsed_time": 2.0,
                "path_length": 390.0,
                "collision_count": 0,
            },
            {
                "result": "TIMEOUT",
                "elapsed_time": 10.0,
                "path_length": 50.0,
                "collision_count": 1,
            },
        ]

        summary = summarize_results(payloads)

        self.assertEqual(summary["episodes"], 2)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_elapsed_time"], 6.0)
        self.assertAlmostEqual(summary["mean_path_length"], 220.0)
        self.assertAlmostEqual(summary["mean_collision_count"], 0.5)

    def test_clear_improvement_requires_stable_success_and_large_margin(self):
        random_summary = {"success_rate": 0.1}

        self.assertTrue(
            is_clearly_better(random_summary, {"success_rate": 0.8})
        )
        self.assertFalse(
            is_clearly_better(random_summary, {"success_rate": 0.6})
        )

    def test_random_evaluation_produces_recorder_payload(self):
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            payloads = run_evaluation(
                controller="random",
                seeds=(EVALUATION_SEEDS[0],),
                output_dir=output_path,
                render_mode=None,
            )

            diagnostics = json.loads(
                (output_path / "diagnostics.json").read_text(encoding="utf-8")
            )
            episode = diagnostics["episodes"][0]
            required_fields = {
                "initial_target_distance",
                "min_target_distance",
                "final_target_distance",
                "target_visible_ratio",
                "obstacle_visible_ratio",
                "mean_abs_turn",
                "mean_throttle",
                "progress_reward_sum",
                "step_reward_sum",
                "collision_reward_sum",
                "collision_event_reward_sum",
                "contact_penalty_sum",
                "contact_duration",
                "goal_reward_sum",
                "total_reward",
                "collision_count",
                "termination_reason",
                "trajectory_file",
            }
            self.assertTrue(required_fields.issubset(episode))
            component_total = sum(
                episode[field]
                for field in (
                    "progress_reward_sum",
                    "step_reward_sum",
                    "collision_event_reward_sum",
                    "contact_penalty_sum",
                    "goal_reward_sum",
                )
            )
            self.assertAlmostEqual(component_total, episode["total_reward"], places=6)
            self.assertTrue((output_path / episode["trajectory_file"]).exists())

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["scenario"], "rl_sanity")
        self.assertEqual(payloads[0]["controller"], "random")
        self.assertIn(payloads[0]["result"], {"SUCCESS", "TIMEOUT"})


if __name__ == "__main__":
    unittest.main()
