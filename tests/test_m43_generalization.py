"""验证 M4.3 test-only 场景、分组统计和绕行方向指标。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from autonomy_lab.scenarios.config import get_scene
from scripts.evaluation.eval_m43_generalization import (
    M43_SCENARIO_GROUPS,
    build_generalization_summary,
    initial_detour_direction,
    write_generalization_outputs,
)


class M43GeneralizationTests(unittest.TestCase):
    def test_unseen_scenes_are_marked_and_keep_frozen_beacon_interface(self):
        for scenario in (
            "m43_target_shift",
            "m43_obstacle_shift",
            "m43_reverse_detour",
            "m43_combined_shift",
        ):
            scene = get_scene(scenario)

            self.assertTrue(scene["test_only"])
            self.assertTrue(scene["unseen_during_ppo_training"])
            self.assertEqual(scene["evaluation_group"], "unseen_mild")
            self.assertEqual(scene["target_information_mode"], "perceived")
            self.assertEqual(scene["sensor"]["range"], 700.0)
            self.assertEqual(scene["sensor"]["fov_degrees"], 160.0)
            self.assertFalse(scene["sensor"]["los_enabled"])
            self.assertEqual(len(scene["obstacles"]), 1)

    def test_reverse_detour_is_passable_above_but_clearer_below(self):
        baseline = get_scene("ppo_simple_obstacle")
        reverse = get_scene("m43_reverse_detour")
        _, reverse_y, _, reverse_height = reverse["obstacles"][0]
        _, baseline_y, _, _ = baseline["obstacles"][0]
        diameter = 2.0 * reverse["agent"]["radius"]
        world_height = reverse["world_size"][1]

        baseline_upper_margin = baseline_y - diameter
        reverse_upper_margin = reverse_y - diameter
        reverse_lower_margin = (
            world_height - (reverse_y + reverse_height) - diameter
        )

        self.assertEqual(reverse["obstacles"], [(350, 60, 80, 240)])
        self.assertGreater(reverse_upper_margin, 0.0)
        self.assertLess(reverse_upper_margin, baseline_upper_margin)
        self.assertGreater(reverse_lower_margin, reverse_upper_margin)

    def test_initial_detour_direction_uses_first_material_vertical_deviation(self):
        obstacle = (350, 60, 80, 240)
        upper = [[0.0, 100.0, 300.0], [0.5, 260.0, 275.0], [1.0, 440.0, 100.0]]
        lower = [[0.0, 100.0, 300.0], [0.5, 260.0, 325.0], [1.0, 440.0, 400.0]]
        straight = [[0.0, 100.0, 300.0], [0.5, 260.0, 305.0], [1.0, 440.0, 300.0]]

        self.assertEqual(initial_detour_direction(upper, obstacle, 16), "upper")
        self.assertEqual(initial_detour_direction(lower, obstacle, 16), "lower")
        self.assertEqual(
            initial_detour_direction(straight, obstacle, 16), "ambiguous"
        )

    def test_summary_uses_scenarios_as_units_and_excludes_hard_from_drop(self):
        rows = []
        for scenario, group in M43_SCENARIO_GROUPS.items():
            for controller in ("bt", "ppo"):
                success = not (
                    group == "unseen_mild"
                    and controller == "ppo"
                    and scenario in {"m43_reverse_detour", "m43_combined_shift"}
                )
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
                        "decision_frequency_hz": 60.0 if controller == "bt" else 10.0,
                        "decision_count": 120 if controller == "bt" else 20,
                        "initial_detour_direction": "ambiguous",
                    }
                )

        summary = build_generalization_summary(
            rows, Path("models/ppo_m41b_control10hz.zip"), seed=5001
        )
        groups = {
            (row["controller"], row["scenario_group"]): row
            for row in summary["controller_group_summary"]
        }

        self.assertEqual(groups[("bt", "unseen_mild")]["successful_scenarios"], 4)
        self.assertEqual(groups[("ppo", "unseen_mild")]["successful_scenarios"], 2)
        self.assertEqual(summary["generalization_drop"]["bt"], 0.0)
        self.assertEqual(summary["generalization_drop"]["ppo"], 0.5)

        with tempfile.TemporaryDirectory() as output_dir:
            csv_path, summary_path = write_generalization_outputs(
                rows,
                Path(output_dir),
                Path("models/ppo_m41b_control10hz.zip"),
                seed=5001,
            )
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(saved_rows), 14)
        self.assertEqual(saved_summary["statistical_unit"], "scenario")


if __name__ == "__main__":
    unittest.main()
