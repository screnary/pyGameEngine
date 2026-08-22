"""R0.11 Condition threshold sensitivity evaluator regression tests."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autonomy_lab.bt.parameters import ConditionParameters
from scripts.evaluation.eval_condition_threshold_sensitivity import (
    _attribution,
    build_hazard_threshold_grid,
    evaluate_threshold_sensitivity,
    write_results,
)


class ConditionThresholdSensitivityTests(unittest.TestCase):
    def test_grid_spans_aggressive_default_and_conservative_values(self):
        """A grid that omits the default or clips incorrectly breaks attribution."""
        parameters = ConditionParameters()

        thresholds = build_hazard_threshold_grid(parameters)

        self.assertEqual(
            thresholds,
            (45.0, 63.0, 76.5, 90.0, 103.5, 117.0, 135.0),
        )

    def test_episode_sweep_changes_only_runtime_hazard_threshold(self):
        """A sweep must not mutate the frozen Boundary/Goal/default parameters."""
        payload = evaluate_threshold_sensitivity(
            families=("static_random",),
            seeds=(1001,),
            thresholds=(45.0, 90.0),
            episode_horizon=0.05,
        )

        self.assertEqual(payload["default_parameters"], {
            "hazard_threshold": 90.0,
            "boundary_threshold": 40.0,
            "goal_threshold": 30.0,
        })
        self.assertEqual(
            [row["threshold"] for row in payload["episodes"]],
            [45.0, 90.0],
        )
        self.assertTrue(
            all(row["boundary_threshold"] == 40.0 for row in payload["episodes"])
        )
        self.assertTrue(
            all(row["goal_threshold"] == 30.0 for row in payload["episodes"])
        )
        self.assertEqual(ConditionParameters().hazard_threshold, 90.0)
        timeout_groups = payload["summary"]["timeout_by_threshold_and_family"]
        self.assertEqual(
            {(row["threshold"], row["family"]) for row in timeout_groups},
            {(45.0, "all"), (45.0, "static_random"),
             (90.0, "all"), (90.0, "static_random")},
        )

        for row in payload["episodes"]:
            self.assertEqual(row["simulation_steps"], 3)
            self.assertIn("hazard_risk_activation_count", row)
            self.assertIn("avoid_active_ratio", row)
            self.assertIn("move_to_goal_active_ratio", row)
            self.assertIn("boundary_active_ratio", row)
            self.assertIn("branch_switch_count", row)
            self.assertIn("longest_avoid_duration", row)

    def test_material_switching_gain_is_not_mislabeled_as_action_failure(self):
        """A dominant less-conservative threshold still supports switching attribution."""
        overall = [
            {
                "threshold": 45.0,
                "success_rate": 0.96,
                "timeout_rate": 0.04,
                "collision_episode_rate": 0.05,
                "mean_avoid_active_ratio": 0.34,
            },
            {
                "threshold": 90.0,
                "success_rate": 0.62,
                "timeout_rate": 0.38,
                "collision_episode_rate": 0.08,
                "mean_avoid_active_ratio": 0.53,
            },
            {
                "threshold": 135.0,
                "success_rate": 0.31,
                "timeout_rate": 0.69,
                "collision_episode_rate": 0.06,
                "mean_avoid_active_ratio": 0.68,
            },
        ]
        preferences = [
            {"family": family, "best_balanced_threshold": 45.0}
            for family in ("static", "dense", "dynamic")
        ]

        result = _attribution(overall, preferences)

        self.assertEqual(result["case"], "Case A — Switching bottleneck supported")
        self.assertTrue(result["switching_material"])
        self.assertFalse(result["safety_efficiency_tradeoff"])
        self.assertFalse(result["action_bottleneck_remains"])

    def test_context_shift_records_real_simulation_phase_diagnostics(self):
        """Aggregate-only logging would hide within-episode threshold behavior."""
        payload = evaluate_threshold_sensitivity(
            families=("context_shift",),
            seeds=(1001,),
            thresholds=(90.0,),
            episode_horizon=4.05,
        )

        phases = {
            row["phase"]: row
            for row in payload["episodes"][0]["phase_diagnostics"]
        }
        self.assertEqual(
            set(phases), {"low_risk", "high_risk", "recovery"}
        )
        # World 按累计 float simulation_time 切 phase；边界 tick 允许落在相邻
        # phase，但总计必须是 4.05 s × 60 Hz 的 243 个真实仿真步。
        self.assertEqual(
            sum(row["simulation_steps"] for row in phases.values()), 243
        )
        self.assertIn(phases["low_risk"]["simulation_steps"], (120, 121))
        self.assertIn(phases["high_risk"]["simulation_steps"], (119, 120, 121))
        self.assertIn(phases["recovery"]["simulation_steps"], (2, 3))
        self.assertIn("goal_distance_change", phases["high_risk"])
        self.assertIn("avoid_active_ratio", phases["high_risk"])

    def test_outputs_preserve_summary_and_episode_diagnostics(self):
        """R0.11 evidence must remain inspectable outside console output."""
        payload = evaluate_threshold_sensitivity(
            families=("static_random",),
            seeds=(1001,),
            thresholds=(90.0,),
            episode_horizon=0.05,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = write_results(
                payload,
                json_path=root / "result.json",
                summary_csv_path=root / "summary.csv",
                episode_csv_path=root / "episodes.csv",
            )
            stored = json.loads(paths[0].read_text(encoding="utf-8"))
            with paths[1].open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            with paths[2].open(encoding="utf-8", newline="") as handle:
                episode_rows = list(csv.DictReader(handle))

        self.assertEqual(stored["milestone"], "R0.11")
        self.assertEqual(summary_rows[0]["scope"], "overall")
        self.assertEqual(episode_rows[0]["family"], "static_random")
        self.assertIn("avoid_active_ratio", episode_rows[0])


if __name__ == "__main__":
    unittest.main()
