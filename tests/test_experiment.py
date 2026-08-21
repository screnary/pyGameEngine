"""验证 bt_config_id 记录以及旧版 CSV 表头的兼容迁移。"""

import csv
import tempfile
import unittest
from pathlib import Path

from autonomy_lab.core.environment import Environment
from autonomy_lab.experiment.recorder import ExperimentRecorder
from autonomy_lab.scenarios.config import get_scene


OLD_SUMMARY_FIELDS = (
    "episode_id",
    "scenario",
    "controller",
    "seed",
    "result",
    "termination_reason",
    "elapsed_time",
    "path_length",
    "collision_count",
    "bt_tick_count",
    "bt_transition_count",
)


class ExperimentBTConfigTests(unittest.TestCase):
    def test_bt_config_id_is_written_to_json_and_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            recorder = ExperimentRecorder(Path(output_dir))
            environment = Environment(get_scene("simple"))
            recorder.start_episode(
                environment,
                "simple",
                "bt-v1",
                track_bt=True,
                bt_config_id="default_bt",
            )

            payload = recorder.finish_episode("FAILURE", "test_finished")

            self.assertEqual(payload["bt_config_id"], "default_bt")
            with recorder.results_path.open(
                encoding="utf-8", newline=""
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["bt_config_id"], "default_bt")

    def test_previous_csv_header_is_upgraded_without_losing_rows(self):
        # 手工写入上一版表头，验证迁移真实历史行而不是只测试新文件。
        with tempfile.TemporaryDirectory() as output_dir:
            recorder = ExperimentRecorder(Path(output_dir))
            old_row = {field: "" for field in OLD_SUMMARY_FIELDS}
            old_row.update(
                episode_id="0007",
                scenario="simple",
                controller="bt-v1",
                result="SUCCESS",
            )
            with recorder.results_path.open(
                "w", encoding="utf-8", newline=""
            ) as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=OLD_SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerow(old_row)

            environment = Environment(get_scene("simple"))
            recorder.start_episode(
                environment,
                "simple",
                "bt-v1",
                track_bt=True,
                bt_config_id="default_bt",
            )
            recorder.finish_episode("FAILURE", "test_finished")

            with recorder.results_path.open(
                encoding="utf-8", newline=""
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["episode_id"], "0007")
            self.assertEqual(rows[0]["bt_config_id"], "")
            self.assertEqual(rows[1]["bt_config_id"], "default_bt")


if __name__ == "__main__":
    unittest.main()
