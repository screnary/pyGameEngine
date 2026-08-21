"""验证 M5.3 四 Controller 汇总、Adapter 等价和独立输出契约。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.eval_m53_final import (
    CONTROLLERS,
    M53_SCENARIO_GROUPS,
    assert_adapter_equivalence,
    build_m53_summary,
    write_m53_outputs,
)


class M53FinalEvaluationTests(unittest.TestCase):
    """若漏掉 Controller、混淆 hard 结果或弱化负结果，这些测试应失败。"""

    def make_rows(self) -> list[dict]:
        rows = []
        for scenario, group in M53_SCENARIO_GROUPS.items():
            for controller in CONTROLLERS:
                success = True
                if controller in {"frozen_hybrid", "hybrid_trained_ppo"}:
                    success = scenario not in {
                        "m43_reverse_detour",
                        "ppo_simple_obstacles",
                    }
                rows.append(
                    {
                        "controller": controller,
                        "scenario": scenario,
                        "scenario_group": group,
                        "seed": 5001,
                        "success": success,
                        "elapsed_time": 4.0 if success else 20.0,
                        "path_length": 700.0,
                        "collision_count": 1,
                        "termination_reason": (
                            "target_reached" if success else "timeout"
                        ),
                        "decision_frequency_hz": 60.0,
                        "decision_count": 240,
                        "bt_tick_count": 240,
                        "ppo_decision_count": 40,
                        "ppo_active_ratio": 0.8,
                        "boundary_recovery_activation_count": 1,
                        "search_activation_count": 0,
                        "ppo_preemption_count": 1,
                        "ppo_reentry_count": 1,
                    }
                )
        return rows

    def test_summary_keeps_four_controllers_and_records_negative_training_result(self):
        """删除一种 Controller 或把 0 improvement 包装成提升都应被捕获。"""
        summary = build_m53_summary(
            self.make_rows(),
            frozen_model_path=Path("models/ppo_m41b_control10hz.zip"),
            trained_model_path=Path("models/ppo_m52_hybrid_trained_200k.zip"),
            seed=5001,
            adapter_equivalence={"passed": True, "scenarios": []},
        )
        groups = {
            (row["controller"], row["scenario_group"]): row
            for row in summary["controller_group_summary"]
        }

        self.assertEqual(len(groups), 12)
        self.assertEqual(groups[("bt", "seen")]["successful_scenarios"], 2)
        self.assertEqual(groups[("pure_ppo", "unseen_mild")]["success_rate"], 1.0)
        self.assertEqual(
            groups[("frozen_hybrid", "unseen_mild")]["success_rate"], 0.75
        )
        self.assertEqual(
            groups[("hybrid_trained_ppo", "ood_hard")]["success_rate"], 0.0
        )
        self.assertEqual(summary["generalization_drop"]["frozen_hybrid"], 0.25)
        self.assertEqual(summary["hybrid_training_mild_success_improvement"], 0.0)
        self.assertFalse(summary["hybrid_training_improvement_demonstrated"])
        self.assertTrue(summary["adapter_equivalence"]["passed"])

    def test_adapter_equivalence_accepts_float_tolerance_and_rejects_metric_drift(self):
        """公共 metrics 漂移必须使 regression 失败，微小浮点误差则允许。"""
        reference = [
            {
                "scenario": "rl_sanity",
                "success": True,
                "elapsed_time": 1.983333,
                "path_length": 436.3000000,
                "collision_count": 0,
            }
        ]
        adapter = [
            {
                "scenario": "rl_sanity",
                "success": True,
                "elapsed_time": 1.9833334,
                "path_length": 436.3000004,
                "collision_count": 0,
            }
        ]

        report = assert_adapter_equivalence(reference, adapter, absolute_tolerance=1e-6)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scenarios"][0]["scenario"], "rl_sanity")

        adapter[0]["collision_count"] = 1
        with self.assertRaisesRegex(AssertionError, "collision_count"):
            assert_adapter_equivalence(reference, adapter, absolute_tolerance=1e-6)

    def test_outputs_use_m53_names_and_keep_28_final_rows(self):
        """输出若覆盖旧 Milestone 文件或少写 Controller 行，应失败。"""
        with tempfile.TemporaryDirectory() as output_dir:
            csv_path, summary_path = write_m53_outputs(
                self.make_rows(),
                Path(output_dir),
                Path("models/ppo_m41b_control10hz.zip"),
                Path("models/ppo_m52_hybrid_trained_200k.zip"),
                seed=5001,
                adapter_equivalence={"passed": True, "scenarios": []},
            )
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(csv_path.name, "m53_final.csv")
        self.assertEqual(summary_path.name, "m53_final_summary.json")
        self.assertEqual(len(saved_rows), 28)
        self.assertIn("ppo_reentry_count", saved_rows[0])
        self.assertEqual(len(summary["controller_scenario_results"]), 28)


if __name__ == "__main__":
    unittest.main()
