"""保护 R0.4 Scenario Distribution 的可复现性与 World/Perception 边界。"""

from copy import deepcopy
import unittest

import pygame

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.core.environment import Environment


def circle_intersects_rect(
    position: tuple[float, float], radius: float, bounds: tuple[int, int, int, int]
) -> bool:
    """测试夹具中的独立圆矩形相交计算，不调用 Environment 私有实现。"""
    rect = pygame.Rect(*bounds)
    x, y = position
    closest_x = max(rect.left, min(x, rect.right))
    closest_y = max(rect.top, min(y, rect.bottom))
    return (x - closest_x) ** 2 + (y - closest_y) ** 2 < radius**2


class ScenarioDistributionSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution
        except ImportError as error:
            self.fail(f"ScenarioDistribution is missing: {error}")
        self.distribution_type = ScenarioDistribution

    def test_same_family_and_seed_reproduce_complete_scene(self):
        """未使用局部 RNG 会让同一实验 seed 生成不同初态。"""
        first = self.distribution_type("static_random").sample(104)
        second = self.distribution_type("static_random").sample(104)
        self.assertEqual(first, second)
        self.assertEqual(first["research_metadata"]["family"], "static_random")
        self.assertEqual(first["research_metadata"]["seed"], 104)

    def test_different_seed_changes_sampled_geometry(self):
        """忽略 sample(seed) 会使 Distribution 退化为固定 Scenario。"""
        first = self.distribution_type("static_random").sample(104)
        second = self.distribution_type("static_random").sample(105)
        self.assertNotEqual(
            (first["agent"], first["target"], first["obstacles"]),
            (second["agent"], second["target"], second["obstacles"]),
        )

    def test_all_samples_have_valid_agent_goal_and_hazard_spawns(self):
        """生成器缺少几何约束时会把 Agent 或 Goal 初始化进 Hazard。"""
        for family in ("static_random", "dense_hazard", "noisy_perception"):
            with self.subTest(family=family):
                scene = self.distribution_type(family).sample(37)
                width, height = scene["world_size"]
                agent = scene["agent"]
                goal = scene["target"]
                ax, ay = agent["position"]
                gx, gy = goal["position"]
                self.assertGreaterEqual(ax, agent["radius"])
                self.assertLessEqual(ax, width - agent["radius"])
                self.assertGreaterEqual(ay, agent["radius"])
                self.assertLessEqual(ay, height - agent["radius"])
                self.assertGreaterEqual(gx, goal["radius"])
                self.assertLessEqual(gx, width - goal["radius"])
                self.assertGreaterEqual(gy, goal["radius"])
                self.assertLessEqual(gy, height - goal["radius"])
                for hazard in scene["obstacles"]:
                    self.assertFalse(
                        circle_intersects_rect(
                            agent["position"], agent["radius"], hazard
                        )
                    )
                    self.assertFalse(
                        circle_intersects_rect(
                            goal["position"], goal["radius"], hazard
                        )
                    )

    def test_metadata_summarises_each_research_family(self):
        """缺失追踪字段会使生成 Episode 无法还原 family/risk context。"""
        for family in (
            "static_random",
            "dense_hazard",
            "dynamic_hazard",
            "noisy_perception",
            "context_shift",
        ):
            with self.subTest(family=family):
                scene = self.distribution_type(family).sample(12)
                metadata = scene["research_metadata"]
                self.assertEqual(metadata["family"], family)
                self.assertEqual(metadata["seed"], 12)
                self.assertEqual(metadata["hazard_count"], len(scene["obstacles"]) + len(scene.get("dynamic_hazards", ())))
                self.assertIn("dynamic_hazard_enabled", metadata)
                self.assertIn("noise_level", metadata)


class DynamicHazardTests(unittest.TestCase):
    def setUp(self) -> None:
        from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution

        self.scene = ScenarioDistribution("dynamic_hazard").sample(73)

    def test_dynamic_hazard_moves_and_remains_a_shared_collision_obstacle(self):
        """只更新诊断数据而不更新共享 Rect 会让 perception/collision 看见旧位置。"""
        world = Environment(deepcopy(self.scene))
        before = world.dynamic_hazard_states[0]["position"]
        dynamic_rect = world.dynamic_hazards[0]["rect"]
        self.assertIn(dynamic_rect, world.obstacles)

        world.step({"turn": 0.0, "throttle": 0.0}, 0.5)

        after = world.dynamic_hazard_states[0]["position"]
        self.assertNotEqual(before, after)
        self.assertEqual(dynamic_rect.center, (round(after[0]), round(after[1])))

    def test_dynamic_hazard_replay_is_deterministic(self):
        """动态运动依赖墙钟或全局 RNG 时，同 seed trajectory 会漂移。"""
        first = Environment(deepcopy(self.scene))
        second = Environment(deepcopy(self.scene))
        command = {"turn": 0.0, "throttle": 0.0}
        first_states = []
        second_states = []
        for _ in range(20):
            first.step(command, 1.0 / 60.0)
            second.step(command, 1.0 / 60.0)
            first_states.append(first.dynamic_hazard_states)
            second_states.append(second.dynamic_hazard_states)
        self.assertEqual(first_states, second_states)


class PerceptionNoiseTests(unittest.TestCase):
    def setUp(self) -> None:
        from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution

        self.scene = ScenarioDistribution("noisy_perception").sample(88)

    def test_noise_sequence_replays_from_world_seed(self):
        """感知噪声 RNG 未在 reset 时重置会破坏同 seed replay。"""
        first = Environment(deepcopy(self.scene))
        second = Environment(deepcopy(self.scene))
        command = {"turn": 0.0, "throttle": 0.0}
        first_values = []
        second_values = []
        for _ in range(8):
            first_values.append(first.perception.snapshot.hazard.nearest_clearance)
            second_values.append(second.perception.snapshot.hazard.nearest_clearance)
            first.step(command, 1.0 / 60.0)
            second.step(command, 1.0 / 60.0)
        self.assertEqual(first_values, second_values)
        self.assertGreater(len(set(first_values)), 1)

    def test_noise_changes_semantic_measurement_not_ground_truth_geometry(self):
        """将 noise 写入 Rect 会污染 collision truth 与实验可复现性。"""
        noisy_scene = deepcopy(self.scene)
        clean_scene = deepcopy(self.scene)
        clean_scene["perception_noise"]["hazard_range_std"] = 0.0
        noisy = Environment(noisy_scene)
        clean = Environment(clean_scene)

        self.assertEqual(
            [tuple(rect) for rect in noisy.obstacles],
            [tuple(rect) for rect in clean.obstacles],
        )
        self.assertNotEqual(
            noisy.perception.snapshot.hazard.nearest_clearance,
            clean.perception.snapshot.hazard.nearest_clearance,
        )
        self.assertEqual(noisy._agent_collides(), clean._agent_collides())


class ContextShiftAndResearchBTTests(unittest.TestCase):
    def setUp(self) -> None:
        from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution

        self.distribution_type = ScenarioDistribution

    def test_context_phase_changes_at_reproducible_simulation_times(self):
        """使用 decision count 或墙钟切 phase 会在不同 Controller 下产生偏差。"""
        world = Environment(self.distribution_type("context_shift").sample(51))
        command = {"turn": 0.0, "throttle": 0.0}
        self.assertEqual(world.current_context_phase, "low_risk")
        initial_noise = world.current_noise_level

        for _ in range(121):
            world.step(command, 1.0 / 60.0)
        self.assertEqual(world.current_context_phase, "high_risk")
        self.assertGreater(world.current_noise_level, initial_noise)

        for _ in range(120):
            world.step(command, 1.0 / 60.0)
        self.assertEqual(world.current_context_phase, "recovery")
        self.assertLess(world.current_noise_level, 10.0)

    def test_parameterized_research_bt_ticks_all_generated_families(self):
        """输出不兼容现有 SceneConfig 时 Controller 构建或 tick 会失败。"""
        for family in (
            "static_random",
            "dense_hazard",
            "dynamic_hazard",
            "noisy_perception",
            "context_shift",
        ):
            with self.subTest(family=family):
                world = Environment(self.distribution_type(family).sample(9))
                controller = BehaviorTreeController(
                    world, bt_config="condition_research"
                )
                command = controller.tick(1.0 / 60.0)
                self.assertEqual(len(command), 2)
                self.assertEqual(world.scenario_metadata["family"], family)

    def test_regression_families_preserve_fixed_bug_reproduction_geometry(self):
        """错误随机化 regression family 会使原 bug 场景失去确定语义。"""
        narrow = self.distribution_type("narrow_passage").sample(901)
        boundary = self.distribution_type("boundary_hazard").sample(902)
        self.assertEqual(
            narrow["obstacles"],
            [(350, 80, 80, 208), (350, 312, 80, 208)],
        )
        self.assertEqual(boundary["agent"]["position"], (810, 300))
        self.assertEqual(narrow["research_metadata"]["family"], "narrow_passage")
        self.assertEqual(boundary["research_metadata"]["family"], "boundary_hazard")


if __name__ == "__main__":
    unittest.main()
