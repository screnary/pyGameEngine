"""验证 Frozen PPO 作为 BT Action 时的频率、生命周期与高层抢占。"""

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import py_trees

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.core.environment import Environment
from autonomy_lab.experiment.runners import run_bt_episode
from autonomy_lab.core.observation import build_navigation_observation
from autonomy_lab.scenarios.config import get_scene


class FakeFrozenModel:
    """替代磁盘 PPO 推理；保留真实 Controller/Loader/py_trees 生命周期。"""

    def __init__(self, action: tuple[float, float] = (0.25, 0.8)) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.calls: list[tuple[np.ndarray, bool]] = []

    def predict(self, observation: np.ndarray, deterministic: bool = False):
        self.calls.append((observation.copy(), deterministic))
        return self.action.copy(), None


class HybridPPOTreeTests(unittest.TestCase):
    def make_controller(
        self,
        scenario: str = "rl_sanity",
        action: tuple[float, float] = (0.25, 0.8),
    ) -> tuple[Environment, BehaviorTreeController, FakeFrozenModel, mock.Mock]:
        world = Environment(get_scene(scenario))
        model = FakeFrozenModel(action)
        loader = mock.Mock(return_value=model)
        with mock.patch("stable_baselines3.PPO.load", loader):
            controller = BehaviorTreeController(world, bt_config="hybrid_ppo")
        return world, controller, model, loader

    def test_hybrid_definition_builds_boundary_navigation_and_search_branches(self):
        """缺失分支、普通 AvoidObstacle 混入或 Registry 未注册都应使本测试失败。"""
        _, controller, _, _ = self.make_controller()

        self.assertEqual(controller.bt_config_id, "hybrid_bt_ppo")
        self.assertEqual(
            [child.name for child in controller.root.children],
            ["Boundary Recovery", "Learned Navigation", "Search Target"],
        )
        self.assertIn("PPO Navigate", controller.nodes_by_name)
        self.assertNotIn("Obstacle Avoidance", controller.nodes_by_name)

    def test_external_ppo_control_accepts_action_only_after_bt_grants_ownership(self):
        """若训练 Adapter 绕开真实 BT 或仍加载冻结模型，本测试应失败。"""
        world = Environment(get_scene("rl_sanity"))
        with mock.patch("stable_baselines3.PPO.load") as loader:
            controller = BehaviorTreeController(
                world,
                bt_config="hybrid_ppo",
                external_ppo_control=True,
            )

        controller.tick(0.0)
        self.assertTrue(controller.ppo_active)
        self.assertTrue(controller.ppo_action_required)
        controller.set_ppo_action(np.asarray([0.3, 0.7], dtype=np.float32))

        self.assertFalse(controller.ppo_action_required)
        self.assertEqual(controller.ppo_decision_count, 1)
        np.testing.assert_allclose(
            tuple(controller.command.values()),
            (0.3, 0.7),
            rtol=0.0,
            atol=1e-7,
        )
        loader.assert_not_called()

    def test_ppo_predicts_immediately_then_holds_each_action_for_six_bt_ticks(self):
        """若错误退化为 60 Hz predict 或丢失缓存 command，本测试会失败。"""
        world, controller, model, loader = self.make_controller(
            action=(0.4, 0.9)
        )
        initial_position = tuple(world.agent.position)

        commands = [controller.tick(1.0 / 60.0) for _ in range(12)]

        self.assertEqual(len(model.calls), 2)  # tick 1 与 tick 7
        self.assertEqual(controller.ppo_decision_count, 2)
        self.assertAlmostEqual(controller.ppo_active_time, 12.0 / 60.0)
        for command in commands:
            np.testing.assert_allclose(command, (0.4, 0.9), rtol=0.0, atol=1e-7)
        self.assertEqual(tuple(world.agent.position), initial_position)
        self.assertEqual(world.simulation_time, 0.0)  # Action Node 不得 step World
        loader.assert_called_once()
        for observation, deterministic in model.calls:
            np.testing.assert_array_equal(
                observation, build_navigation_observation(world)
            )
            self.assertTrue(deterministic)

    def test_reset_reenters_with_immediate_prediction_without_reloading_model(self):
        """Node initialise 可重新决策，但 Controller 生命周期内不得重复读 checkpoint。"""
        _, controller, model, loader = self.make_controller()
        controller.tick(1.0 / 60.0)

        controller.reset()
        controller.tick(1.0 / 60.0)

        self.assertEqual(len(model.calls), 2)
        loader.assert_called_once()

    def test_boundary_branch_preempts_running_ppo_and_replaces_stale_command(self):
        """Boundary 风险必须在下一次 60 Hz BT tick 抢占 10 Hz PPO Action。"""
        world, controller, _, _ = self.make_controller(action=(0.4, 0.9))
        np.testing.assert_allclose(
            controller.tick(1.0 / 60.0), (0.4, 0.9), rtol=0.0, atol=1e-7
        )
        ppo_node = controller.nodes_by_name["PPO Navigate"]
        self.assertEqual(ppo_node.status, py_trees.common.Status.RUNNING)

        world.agent.position.x = world.world_size[0] - world.agent.radius - 5.0
        turn, throttle = controller.tick(1.0 / 60.0)

        self.assertEqual(controller.active_behavior, "Safe Boundary Recovery")
        self.assertEqual(ppo_node.status, py_trees.common.Status.INVALID)
        self.assertEqual(controller.boundary_recovery_activation_count, 1)
        self.assertEqual(controller.ppo_preemption_count, 1)
        self.assertAlmostEqual(abs(turn), 1.0)
        self.assertAlmostEqual(throttle, 0.5)

        world.agent.position.x = world.world_size[0] / 2.0
        world.agent.heading = 0.0
        resumed = controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "PPO Navigate")
        np.testing.assert_allclose(resumed, (0.4, 0.9), rtol=0.0, atol=1e-7)

    def test_tree_stop_clears_running_ppo_command(self):
        """Tree reset/关闭使 Action INVALID 时不得留下旧 throttle/turn。"""
        _, controller, _, _ = self.make_controller(action=(-0.6, 1.0))
        controller.tick(1.0 / 60.0)

        controller.root.stop(py_trees.common.Status.INVALID)

        self.assertEqual(controller.command, {"turn": 0.0, "throttle": 0.0})

    def test_target_outside_fov_uses_search_then_switches_to_ppo_when_visible(self):
        """PPO 只负责可见目标导航；目标不可见时由原 SearchTarget 扫描。"""
        world = Environment(get_scene("rl_sanity"))
        world.agent.heading = np.pi
        world.perception.update()
        model = FakeFrozenModel()
        with mock.patch("stable_baselines3.PPO.load", return_value=model):
            controller = BehaviorTreeController(world, bt_config="hybrid_ppo")

        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "Search Target")
        self.assertEqual(len(model.calls), 0)
        self.assertEqual(controller.search_activation_count, 1)

        world.agent.heading = 0.0
        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "PPO Navigate")
        self.assertEqual(len(model.calls), 1)

    def test_hybrid_runner_reuses_recorder_with_hybrid_controller_label(self):
        """Hybrid Episode 必须保留 BT metrics，并与普通 bt label 区分。"""
        with tempfile.TemporaryDirectory() as output_dir:
            row, _ = run_bt_episode(
                "rl_sanity",
                6001,
                Path(output_dir),
                bt_config="hybrid_ppo",
                recorder_controller="hybrid_bt_ppo",
                collect_diagnostics=True,
            )

        self.assertEqual(row["controller"], "hybrid_bt_ppo")
        self.assertTrue(row["success"])
        self.assertEqual(row["bt_tick_count"], row["decision_count"])
        self.assertEqual(row["ppo_decision_count"], 20)
        self.assertAlmostEqual(row["ppo_active_time"], row["elapsed_time"], places=5)
        self.assertAlmostEqual(row["ppo_active_ratio"], 1.0)
        self.assertEqual(row["boundary_recovery_activation_count"], 0)
        self.assertEqual(row["search_activation_count"], 0)
        self.assertEqual(row["ppo_preemption_count"], 0)


if __name__ == "__main__":
    unittest.main()
