"""R0.8 Hazard range calibration 脚本的真实场景统计回归。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from scripts.evaluation.analyze_hazard_sensing_range import (
    analyze_sensing_ranges,
    write_analysis,
)


class HazardSensingRangeAnalysisTests(unittest.TestCase):
    def test_analysis_distinguishes_local_and_global_hazard_coverage(self):
        """若统计漏算可见对象，700 px 不会呈现 100% all-visible。"""
        result = analyze_sensing_ranges(
            families=("static_random",),
            seeds=(1001, 1002),
            hazard_ranges=(200.0, 700.0),
            goal_ranges=(700.0,),
        )
        rows = {row["hazard_range"]: row for row in result["hazard_ranges"]}

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(rows[200.0]["hazard_availability_rate"], 0.5)
        self.assertEqual(rows[200.0]["mean_sensed_hazard_count"], 0.5)
        self.assertEqual(rows[200.0]["all_hazards_visible_rate"], 0.0)
        self.assertEqual(rows[700.0]["hazard_availability_rate"], 1.0)
        self.assertEqual(rows[700.0]["mean_sensed_hazard_count"], 2.0)
        self.assertEqual(rows[700.0]["all_hazards_visible_rate"], 1.0)

    def test_json_output_preserves_inputs_and_statistics(self):
        """只打印临时结果会使校准依据无法复现或审阅。"""
        result = analyze_sensing_ranges(
            families=("static_random",),
            seeds=(1001,),
            hazard_ranges=(300.0,),
            goal_ranges=(850.0,),
        )
        with TemporaryDirectory() as temp_dir:
            path = write_analysis(result, Path(temp_dir) / "analysis.json")
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["families"], ["static_random"])
        self.assertEqual(payload["seeds"], [1001])
        self.assertEqual(payload["hazard_ranges"][0]["hazard_range"], 300.0)


if __name__ == "__main__":
    unittest.main()
