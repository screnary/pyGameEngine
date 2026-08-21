"""验证 M5.2 训练参数和 Frozen-vs-trained 汇总契约。"""

import unittest
from pathlib import Path

from scripts.evaluation.eval_m52_hybrid import M52_SCENARIO_GROUPS, build_m52_summary
from scripts.training.train_hybrid_ppo import parse_args


class M52PipelineTests(unittest.TestCase):
    def test_training_defaults_use_bounded_hybrid_context_setup(self):
        """若训练集、初始化模型或首节点预算漂移，本测试应失败。"""
        args = parse_args([])

        self.assertEqual(
            tuple(args.scenarios),
            (
                "ppo_simple_obstacle",
                "m43_obstacle_shift",
                "m43_reverse_detour",
            ),
        )
        # M5.2 当前用于验证 Lab 的训练接线，而不是追求策略收敛。
        # 长训练仍可显式传参，但默认命令必须快速结束。
        self.assertEqual(args.target_timesteps, 2_048)
        self.assertEqual(args.seed, 52)
        self.assertEqual(args.init_model_path.name, "ppo_m41b_control10hz.zip")
        self.assertEqual(args.model_path.name, "ppo_m52_smoke.zip")

    def test_summary_separates_hybrid_relevant_improvement_from_hard_search(self):
        """若 hard 场景混入成功标准或两种 Hybrid 未分开，应失败。"""
        rows = []
        for scenario, group in M52_SCENARIO_GROUPS.items():
            for controller in ("frozen_hybrid", "hybrid_trained_ppo"):
                success = group != "ood_hard"
                if controller == "frozen_hybrid" and scenario == "m43_reverse_detour":
                    success = False
                rows.append(
                    {
                        "controller": controller,
                        "scenario": scenario,
                        "scenario_group": group,
                        "seed": 7001,
                        "success": success,
                        "elapsed_time": 4.0 if success else 20.0,
                        "path_length": 700.0,
                        "collision_count": 0,
                        "termination_reason": "target_reached" if success else "timeout",
                        "decision_frequency_hz": 60.0,
                        "decision_count": 240,
                        "bt_tick_count": 240,
                        "bt_transition_count": 1,
                        "ppo_decision_count": 40,
                        "ppo_active_time": 4.0,
                        "ppo_active_ratio": 1.0,
                        "boundary_recovery_activation_count": 0,
                        "search_activation_count": 0,
                        "ppo_preemption_count": 0,
                        "ppo_reentry_count": 0,
                        "observation_before_preemption": None,
                        "observation_after_reentry": None,
                    }
                )

        summary = build_m52_summary(
            rows,
            Path("models/ppo_m52_hybrid_trained.zip"),
            seed=7001,
            checkpoint_label="200k",
        )
        groups = {
            (row["controller"], row["scenario_group"]): row
            for row in summary["controller_group_summary"]
        }

        self.assertEqual(groups[("frozen_hybrid", "seen")]["success_rate"], 1.0)
        self.assertEqual(
            groups[("frozen_hybrid", "hybrid_relevant")]["success_rate"],
            0.5,
        )
        self.assertEqual(
            groups[("hybrid_trained_ppo", "hybrid_relevant")]["success_rate"],
            1.0,
        )
        self.assertEqual(summary["hybrid_relevant_success_improvement"], 0.5)
        self.assertEqual(summary["checkpoint_label"], "200k")


if __name__ == "__main__":
    unittest.main()
