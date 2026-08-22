"""R0.9 固定 Action 能力评估入口的回归测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import unittest

from scripts.evaluation.eval_action_competence import (
    ISOLATED_ACTIONS,
    REQUIRED_FAMILIES,
    evaluate_action_competence,
    write_results,
)


class ActionCompetenceEvaluationTests(unittest.TestCase):
    def test_isolated_suite_exercises_every_research_action(self):
        """评估不能只跑整树，否则无法把失败归因到具体 Action。"""
        payload = evaluate_action_competence(
            families=(),
            seeds=(),
            include_isolated=True,
        )

        isolated = payload["isolated_results"]
        self.assertEqual({row["action"] for row in isolated}, ISOLATED_ACTIONS)
        self.assertTrue(all(row["case_id"] for row in isolated))
        self.assertTrue(
            all(row["record_type"] == "isolated" for row in isolated)
        )
        for row in isolated:
            for field in (
                "success",
                "failure_reason",
                "elapsed_time",
                "path_length",
                "collision_count",
                "minimum_clearance",
            ):
                self.assertIn(field, row)

    def test_end_to_end_uses_real_research_bt_and_world_steps(self):
        """短 horizon 即使超时，也必须留下真实 Research BT episode 记录。"""
        payload = evaluate_action_competence(
            families=("static_random",),
            seeds=(1001,),
            include_isolated=False,
            episode_horizon=1.0 / 60.0,
        )

        self.assertEqual(REQUIRED_FAMILIES[:1], ("static_random",))
        rows = payload["end_to_end_results"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "ResearchBT")
        self.assertEqual(rows[0]["scenario"], "static_random")
        self.assertEqual(rows[0]["seed"], 1001)
        self.assertAlmostEqual(rows[0]["elapsed_time"], 1.0 / 60.0, places=6)
        self.assertEqual(rows[0]["failure_reason"], "timeout")

    def test_writers_preserve_machine_readable_rows_and_summary(self):
        payload = evaluate_action_competence(
            families=(),
            seeds=(),
            include_isolated=True,
        )
        with TemporaryDirectory() as temp_dir:
            json_path, csv_path = write_results(
                payload,
                Path(temp_dir) / "result.json",
                Path(temp_dir) / "result.csv",
            )
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertIn("summary", saved)
        self.assertEqual(
            len(csv_rows),
            len(saved["isolated_results"]) + len(saved["end_to_end_results"]),
        )
        self.assertIn("action", csv_rows[0])
        self.assertIn("minimum_clearance", csv_rows[0])


if __name__ == "__main__":
    unittest.main()
