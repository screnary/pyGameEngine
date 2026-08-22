"""R0.12 paired operating-context necessity evaluator tests."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluation.eval_context_threshold_necessity import (
    CONTEXTS,
    THRESHOLDS,
    _phase_summary,
    build_context_scene,
    evaluate_context_threshold_necessity,
    hazard_proximity_exposure,
    write_results,
)


class ContextThresholdNecessityTests(unittest.TestCase):
    def test_exposure_uses_clearance_and_calibrated_range(self):
        """Center distance or a binary near-miss gate would produce different values."""
        self.assertEqual(hazard_proximity_exposure(None, 300.0), 0.0)
        self.assertEqual(hazard_proximity_exposure(300.0, 300.0), 0.0)
        self.assertEqual(hazard_proximity_exposure(150.0, 300.0), 0.25)
        self.assertEqual(hazard_proximity_exposure(0.0, 300.0), 1.0)
        self.assertEqual(hazard_proximity_exposure(-20.0, 300.0), 1.0)

    def test_contexts_are_paired_and_change_only_dynamic_hazard_speed(self):
        """Different maps would confound threshold preference with geometry."""
        low = build_context_scene(1001, "low_risk")
        high = build_context_scene(1001, "high_dynamic_risk")

        self.assertEqual(low["agent"], high["agent"])
        self.assertEqual(low["target"], high["target"])
        self.assertEqual(low["obstacles"], high["obstacles"])
        self.assertEqual(
            [item["position"] for item in low["dynamic_hazards"]],
            [item["position"] for item in high["dynamic_hazards"]],
        )
        self.assertEqual(
            [item["heading_degrees"] for item in low["dynamic_hazards"]],
            [item["heading_degrees"] for item in high["dynamic_hazards"]],
        )
        self.assertEqual(low["dynamic_hazards"][0]["speed"], 36.0)
        self.assertEqual(high["dynamic_hazards"][0]["speed"], 180.0)
        self.assertEqual(low["sensor"]["hazard_range"], 300.0)
        self.assertEqual(high["sensor"]["hazard_range"], 300.0)

    def test_short_evaluation_keeps_grid_and_fixed_parameters(self):
        """R0.12 may vary context speed and theta_hazard, but no other control input."""
        payload = evaluate_context_threshold_necessity(
            contexts=CONTEXTS,
            thresholds=(20.0, 45.0),
            seeds=(1001,),
            episode_horizon=0.05,
        )

        self.assertEqual(THRESHOLDS, (20.0, 30.0, 40.0, 45.0, 60.0, 75.0, 90.0))
        self.assertEqual(len(payload["episodes"]), 4)
        self.assertEqual(payload["fixed_parameters"], {
            "boundary_threshold": 40.0,
            "goal_threshold": 30.0,
            "hazard_range": 300.0,
        })
        for row in payload["episodes"]:
            self.assertEqual(row["simulation_steps"], 3)
            self.assertGreaterEqual(row["hazard_exposure"], 0.0)
            self.assertLessEqual(row["hazard_exposure"], 1.0)
            self.assertIn("avoid_active_ratio", row)
            self.assertIn("minimum_clearance", row)

    def test_results_write_independent_json_and_csv_files(self):
        """Necessity evidence must remain reproducible without console output."""
        payload = evaluate_context_threshold_necessity(
            contexts=("low_risk",),
            thresholds=(20.0,),
            seeds=(1001,),
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
                summary = list(csv.DictReader(handle))
            with paths[2].open(encoding="utf-8", newline="") as handle:
                episodes = list(csv.DictReader(handle))

        self.assertEqual(stored["milestone"], "R0.12")
        self.assertEqual(summary[0]["context"], "low_risk")
        self.assertIn("hazard_exposure", episodes[0])

    def test_phase_summary_reports_collision_episode_rate(self):
        """Raw collision totals alone cannot compare phases with different samples."""
        episodes = [
            {
                "threshold": 20.0,
                "phase_safety_diagnostics": [
                    {
                        "phase": "high_dynamic_risk",
                        "collision_count": count,
                        "hazard_exposure": exposure,
                        "minimum_clearance": clearance,
                        "avoid_active_ratio": 0.2,
                        "move_to_goal_active_ratio": 0.8,
                        "goal_distance_change": 100.0,
                    }
                ],
            }
            for count, exposure, clearance in (
                (1, 0.6, 10.0),
                (0, 0.4, 30.0),
            )
        ]

        row = _phase_summary(episodes)[0]

        self.assertEqual(row["collision_episodes"], 1)
        self.assertEqual(row["collision_episode_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
