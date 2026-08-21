"""验证 R0.4 human demo 的 CLI、运行时接线与状态诊断。"""

import os
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame


class ScenarioDistributionDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from scripts.demo import demo_scenario_distribution
        except ImportError as error:
            self.fail(f"R0.4 visualization entry is missing: {error}")
        self.demo = demo_scenario_distribution

    def test_cli_selects_family_and_seed(self):
        """忽略 CLI 参数会让用户看到错误 family 或不可复现初态。"""
        args = self.demo.parse_args(
            [
                "--family", "context_shift", "--seed", "77",
                "--hazard-threshold", "120",
                "--boundary-threshold", "55",
                "--goal-threshold", "24",
            ]
        )
        self.assertEqual(args.family, "context_shift")
        self.assertEqual(args.seed, 77)
        self.assertEqual(args.hazard_threshold, 120.0)
        self.assertEqual(args.boundary_threshold, 55.0)
        self.assertEqual(args.goal_threshold, 24.0)

    def test_build_demo_injects_manual_condition_thresholds(self):
        """CLI θ 仅打印而未注入共享 Store 会产生误导性 research smoke。"""
        from autonomy_lab.bt.parameters import ConditionParameters

        parameters = ConditionParameters(120.0, 55.0, 24.0)
        _, controller = self.demo.build_demo(
            "static_random", 42, parameters=parameters
        )
        self.assertIs(controller.condition_parameters, parameters)

    def test_build_demo_uses_sampled_world_and_parameterized_research_bt(self):
        """接到 fixed Scene 或 default BT 会掩盖 R0.4/R0.3 实际效果。"""
        world, controller = self.demo.build_demo("dynamic_hazard", 42)
        self.assertEqual(world.scenario_metadata["family"], "dynamic_hazard")
        self.assertEqual(world.seed, 42)
        self.assertEqual(controller.bt_config_id, "condition_research_bt")
        self.assertEqual(len(world.dynamic_hazard_states), 1)

    def test_human_loop_renders_one_frame_then_closes(self):
        """入口缺少 event lifecycle 时窗口无法正常显示或手动退出。"""
        events = [[], [pygame.event.Event(pygame.QUIT)]]
        with mock.patch.object(pygame.event, "get", side_effect=events):
            frames = self.demo.run_demo("dynamic_hazard", 42)
        self.assertEqual(frames, 1)

    def test_status_lines_expose_research_context(self):
        """不显示 family/noise/phase 会使 noisy/context-shift demo 不可观察。"""
        from autonomy_lab.rendering.renderer import PygameRenderer

        world, _ = self.demo.build_demo("context_shift", 51)
        lines = PygameRenderer.status_lines(world, "condition-research")
        self.assertTrue(any("Family: context_shift" in line for line in lines))
        self.assertTrue(any("Noise: 2.0 px" in line for line in lines))
        self.assertTrue(any("Phase: low_risk" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
