"""验证 M5.1 三方法分组汇总、诊断字段和独立输出契约。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_m51_hybrid import (
    HUMAN_SCENARIOS,
    M51_SCENARIO_GROUPS,
    build_m51_summary,
    write_m51_outputs,
)


class M51HybridEvaluationTests(unittest.TestCase):
    def test_human_demos_include_boundary_and_search_switch_cases(self):
        self.assertEqual(
            HUMAN_SCENARIOS,
            (
                "ppo_simple_obstacle",
                "m43_reverse_detour",
                "ppo_simple_obstacles",
            ),
        )

    def make_rows(self) -> list[dict]:
        rows = []
        for scenario, group in M51_SCENARIO_GROUPS.items():
            for controller in ("bt", "ppo", "hybrid_bt_ppo"):
                success = not (
                    controller == "ppo"
                    and group == "unseen_mild"
                    and scenario in {"m43_reverse_detour", "m43_combined_shift"}
                )
                is_ppo = controller in {"ppo", "hybrid_bt_ppo"}
                is_bt = controller in {"bt", "hybrid_bt_ppo"}
                rows.append(
                    {
                        "controller": controller,
                        "scenario": scenario,
                        "scenario_group": group,
                        "seed": 5001,
                        "success": success,
                        "elapsed_time": 2.0,
                        "path_length": 100.0,
                        "collision_count": 0,
                        "termination_reason": (
                            "target_reached" if success else "timeout"
                        ),
                        "decision_frequency_hz": 60.0 if is_bt else 10.0,
                        "decision_count": 120 if is_bt else 20,
                        "bt_tick_count": 120 if is_bt else None,
                        "bt_transition_count": 1 if is_bt else None,
                        "ppo_decision_count": 20 if is_ppo else 0,
                        "ppo_active_time": 2.0 if is_ppo else 0.0,
                        "ppo_active_ratio": 1.0 if is_ppo else 0.0,
                        "boundary_recovery_activation_count": 0,
                        "search_activation_count": 0,
                        "ppo_preemption_count": 0,
                    }
                )
        return rows

    def test_summary_uses_three_frozen_controllers_and_scenario_groups(self):
        """遗漏 Hybrid、混入 hard drop 或按 Episode 重复统计都会使结果失败。"""
        summary = build_m51_summary(
            self.make_rows(), Path("models/ppo_m41b_control10hz.zip"), seed=5001
        )
        groups = {
            (row["controller"], row["scenario_group"]): row
            for row in summary["controller_group_summary"]
        }

        self.assertEqual(len(groups), 9)
        self.assertEqual(groups[("bt", "seen")]["successful_scenarios"], 2)
        self.assertEqual(groups[("ppo", "unseen_mild")]["success_rate"], 0.5)
        self.assertEqual(
            groups[("hybrid_bt_ppo", "unseen_mild")]["success_rate"], 1.0
        )
        self.assertEqual(summary["generalization_drop"]["bt"], 0.0)
        self.assertEqual(summary["generalization_drop"]["ppo"], 0.5)
        self.assertEqual(summary["generalization_drop"]["hybrid_bt_ppo"], 0.0)
        self.assertEqual(summary["hybrid_diagnostics"]["total_ppo_decisions"], 140)
        self.assertEqual(summary["statistical_unit"], "scenario")

    def test_output_files_keep_public_and_hybrid_diagnostic_fields(self):
        rows = self.make_rows()
        with tempfile.TemporaryDirectory() as output_dir:
            csv_path, summary_path = write_m51_outputs(
                rows,
                Path(output_dir),
                Path("models/ppo_m41b_control10hz.zip"),
                seed=5001,
            )
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(csv_path.name, "m51_bt_ppo_hybrid.csv")
        self.assertEqual(summary_path.name, "m51_bt_ppo_hybrid_summary.json")
        self.assertEqual(len(saved_rows), 21)
        self.assertIn("ppo_active_ratio", saved_rows[0])
        self.assertIn("ppo_preemption_count", saved_rows[0])
        self.assertEqual(len(summary["controller_scenario_results"]), 21)


if __name__ == "__main__":
    unittest.main()
