"""在 dummy SDL 下验证成功、重置、关闭和超时的 Episode 生命周期。"""

import argparse
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import main as app
from autonomy_lab.experiment.recorder import ExperimentRecorder
from autonomy_lab.scenarios.config import get_scene


class SuccessWindowLifecycleTests(unittest.TestCase):
    def test_cli_accepts_named_bt_config(self):
        with mock.patch("sys.argv", ["main.py", "--bt", "custom"]):
            args = app.parse_args()

        self.assertEqual(args.bt, "custom")

    def run_main_with_events(
        self,
        event_frames: list[str | None],
        *,
        target_at_start: bool = True,
        max_episode_time: float | None = None,
    ):
        """每个渲染帧注入一个事件名称，确定性驱动无窗口主循环。"""
        output_dir = Path(tempfile.mkdtemp())
        scene = get_scene("simple")
        if target_at_start:
            scene["target"]["position"] = scene["agent"]["position"]
        if max_episode_time is not None:
            scene["experiment"]["max_episode_time"] = max_episode_time
        event_calls = 0

        def next_events():
            nonlocal event_calls
            event_calls += 1
            event_name = (
                event_frames[event_calls - 1]
                if event_calls <= len(event_frames)
                else "quit"
            )
            if event_name == "quit":
                return [pygame.event.Event(pygame.QUIT)]
            if event_name == "reset":
                return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)]
            return []

        args = argparse.Namespace(scenario="simple", controller="manual")
        with (
            mock.patch.object(app, "parse_args", return_value=args),
            mock.patch.object(app, "get_scene", return_value=scene),
            mock.patch.object(
                app,
                "ExperimentRecorder",
                new=lambda: ExperimentRecorder(output_dir=output_dir),
            ),
            mock.patch.object(pygame.event, "get", side_effect=next_events),
        ):
            app.main()

        payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output_dir / "runs").glob("episode_*.json"))
        ]
        with (output_dir / "results.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        return event_calls, payloads, rows

    def test_success_waits_for_window_close_without_second_result(self):
        event_calls, payloads, rows = self.run_main_with_events([None, "quit"])

        self.assertEqual(event_calls, 2)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["result"], "SUCCESS")
        self.assertEqual(payloads[0]["termination_reason"], "target_reached")
        self.assertIsNone(payloads[0]["bt_config_id"])
        self.assertEqual(len(rows), 1)

    def test_reset_after_success_starts_a_fresh_episode(self):
        event_calls, payloads, rows = self.run_main_with_events(
            [None, "reset", "quit"]
        )

        self.assertEqual(event_calls, 3)
        self.assertEqual([item["episode_id"] for item in payloads], ["0001", "0002"])
        self.assertEqual([item["result"] for item in payloads], ["SUCCESS", "SUCCESS"])
        self.assertEqual(len(rows), 2)

    def test_active_episode_close_remains_interrupted(self):
        _, payloads, rows = self.run_main_with_events(
            ["quit"], target_at_start=False
        )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["result"], "INTERRUPTED")
        self.assertEqual(payloads[0]["termination_reason"], "window_closed")
        self.assertEqual(len(rows), 1)

    def test_timeout_still_exits_automatically(self):
        event_calls, payloads, rows = self.run_main_with_events(
            [None], target_at_start=False, max_episode_time=0.0
        )

        self.assertEqual(event_calls, 1)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["result"], "TIMEOUT")
        self.assertEqual(payloads[0]["termination_reason"], "timeout")
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
