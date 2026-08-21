"""冻结 R0.5 semantic provider、参数 Store 与 Research BT 接口。"""

from dataclasses import fields, is_dataclass
import math
from types import SimpleNamespace
import unittest

import pygame
import py_trees

from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.config import get_scene
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution


def assert_simulator_independent(test: unittest.TestCase, value: object) -> None:
    """递归拒绝 semantic dataclass 中的 simulator runtime object。"""
    forbidden = (pygame.Rect, pygame.Vector2, pygame.Surface, Environment)
    test.assertNotIsInstance(value, forbidden)
    if is_dataclass(value):
        for field in fields(value):
            assert_simulator_independent(test, getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_simulator_independent(test, item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_simulator_independent(test, item)


class SemanticAdapterContractTests(unittest.TestCase):
    def test_pygame_perception_satisfies_runtime_provider_contract(self):
        """缺少统一 observe/snapshot contract 会迫使未来 Adapter 复制 BT。"""
        try:
            from autonomy_lab.perception.semantic_perception import (
                SemanticPerception,
                SemanticPerceptionProvider,
            )
        except ImportError as error:
            self.fail(f"semantic adapter contract is missing: {error}")
        world = Environment(get_scene("rl_sanity"))
        provider = world.perception

        self.assertIsInstance(provider, SemanticPerceptionProvider)
        observation = provider.observe()
        self.assertIsInstance(observation, SemanticPerception)
        self.assertIs(provider.snapshot, observation)

    def test_semantic_snapshot_contains_only_plain_simulator_independent_data(self):
        """把 Rect/Vector2/World 放入 semantic output 会破坏跨 simulator 合约。"""
        for family in (
            "static_random",
            "dynamic_hazard",
            "noisy_perception",
            "context_shift",
        ):
            with self.subTest(family=family):
                world = Environment(ScenarioDistribution(family).sample(31))
                assert_simulator_independent(self, world.perception.observe())

    def test_optional_semantics_report_availability_without_fake_values(self):
        """未来 simulator 缺少 Boundary/gaps 时必须能明确标记 unavailable。"""
        from autonomy_lab.perception.semantic_perception import (
            BoundaryPerception,
            HazardPerception,
        )

        boundary = BoundaryPerception(0.0, 0.0, 0.0, 0.0, available=False)
        hazard = HazardPerception(
            sector_available=False,
            gaps_available=False,
        )
        self.assertFalse(boundary.available)
        self.assertFalse(hazard.sector_available)
        self.assertFalse(hazard.gaps_available)


class ConditionParameterInterfaceTests(unittest.TestCase):
    def test_get_set_reset_and_bounds_form_a_stable_rl_independent_api(self):
        """缺少复制安全的 values/bounds/reset 会让 M6 直接耦合 dataclass internals。"""
        from autonomy_lab.bt.parameters import ConditionParameters

        parameters = ConditionParameters()
        self.assertEqual(
            parameters.get_values(),
            {
                "hazard_threshold": 90.0,
                "boundary_threshold": 40.0,
                "goal_threshold": 30.0,
            },
        )
        parameters.set_values(
            hazard_threshold=105.5,
            boundary_threshold=52.0,
            goal_threshold=25.0,
        )
        self.assertEqual(parameters.get_values()["hazard_threshold"], 105.5)
        bounds = parameters.get_bounds()
        self.assertEqual(bounds["goal_threshold"], (0.0, math.inf))
        bounds["goal_threshold"] = (-1.0, 1.0)
        self.assertEqual(parameters.get_bounds()["goal_threshold"], (0.0, math.inf))

        parameters.reset_defaults()
        self.assertEqual(
            tuple(parameters.get_values().values()),
            (90.0, 40.0, 30.0),
        )

    def test_set_values_rejects_unknown_parameter_name(self):
        """静默接受拼写错误会使实验日志与真实 θ 不一致。"""
        from autonomy_lab.bt.parameters import ConditionParameters

        with self.assertRaises(ValueError):
            ConditionParameters().set_values(unknown_threshold=1.0)


class SemanticOnlyResearchActionTests(unittest.TestCase):
    def make_context(self):
        """构造无 Environment 属性的 Provider，偷读 World 会立即失败。"""
        from autonomy_lab.bt.context import BehaviorBuildContext
        from autonomy_lab.bt.parameters import ConditionParameters
        from autonomy_lab.perception.semantic_perception import (
            AgentState,
            BoundaryPerception,
            GoalPerception,
            HazardObservation,
            HazardPerception,
            SectorRange,
            SemanticPerception,
        )

        threat = HazardObservation(clearance=35.0, bearing=0.2)
        snapshot = SemanticPerception(
            agent=AgentState(speed=0.0, heading=0.0, radius=16.0),
            goal=GoalPerception(
                sensed=True,
                visible=True,
                available=True,
                source="perception",
                distance=180.0,
                bearing=0.1,
                unavailable_reason="",
            ),
            hazard=HazardPerception(
                visible_hazards=(threat,),
                nearest_hazard=threat,
                sector_ranges=tuple(
                    SectorRange(bearing=bearing, clearance=120.0)
                    for bearing in (-math.pi, -math.pi / 2, 0.0, math.pi / 2)
                ),
                sector_available=True,
            ),
            boundary=BoundaryPerception(
                left=20.0,
                right=200.0,
                top=100.0,
                bottom=100.0,
                available=True,
            ),
        )

        class Provider:
            def __init__(self):
                self.snapshot = snapshot

            def observe(self):
                return self.snapshot

        return BehaviorBuildContext(
            perception=Provider(),
            command={"turn": 0.0, "throttle": 0.0},
            behavior_config={
                "target_reached_distance": 30.0,
                "avoid_duration": 0.9,
                "avoid_throttle": 0.75,
            },
            condition_parameters=ConditionParameters(),
        )

    def test_research_actions_execute_without_simulator_world_access(self):
        """SafeBoundaryRecovery 等节点访问 provider.environment 时本测试会报错。"""
        from autonomy_lab.bt.behaviors import (
            AvoidObstacle,
            GoalReached,
            HazardRisk,
            MoveToTarget,
            ResearchBoundaryRisk,
            SafeBoundaryRecovery,
            Stop,
        )

        context = self.make_context()
        hazard = HazardRisk(context, "Hazard Risk?")
        context.nodes_by_name[hazard.name] = hazard
        nodes = [
            ResearchBoundaryRisk(context, "Boundary Risk?"),
            hazard,
            GoalReached(context, "Goal Reached?"),
            MoveToTarget(context, "Move To Goal"),
            AvoidObstacle(context, "Avoid Hazard", condition=hazard.name),
            SafeBoundaryRecovery(context, "Safe Boundary Recovery"),
            Stop(context, "Stop"),
        ]

        for node in nodes:
            with self.subTest(node=node.name):
                node.initialise()
                status = node.update()
                self.assertIn(
                    status,
                    (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE, py_trees.common.Status.RUNNING),
                )


class ScenarioSemanticPipelineTests(unittest.TestCase):
    def test_research_action_uses_the_shared_agent_command_shape(self):
        """另建一套 Research action 表达会让 Environment/controller 接口分叉。"""
        try:
            from autonomy_lab.core.agent import AgentCommand
        except ImportError as error:
            self.fail(f"shared AgentCommand contract is missing: {error}")
        command: AgentCommand = {"turn": 0.25, "throttle": 0.5}
        world = Environment(get_scene("rl_sanity"))
        world.step(command, 1.0 / 60.0)
        self.assertGreater(world.agent.speed, 0.0)

    def test_all_r04_families_share_one_provider_to_research_bt_pipeline(self):
        """family-specific Adapter 分支会破坏统一 research method layer。"""
        from autonomy_lab.bt.controller import BehaviorTreeController
        from autonomy_lab.perception.semantic_perception import SemanticPerception

        for family in (
            "static_random",
            "dense_hazard",
            "dynamic_hazard",
            "noisy_perception",
            "context_shift",
        ):
            with self.subTest(family=family):
                world = Environment(ScenarioDistribution(family).sample(61))
                controller = BehaviorTreeController(
                    world, bt_config="condition_research"
                )
                controller.tick(1.0 / 60.0)
                self.assertIsInstance(
                    world.perception.snapshot, SemanticPerception
                )

    def test_dynamic_hazard_current_geometry_reaches_hazard_semantics(self):
        """动态 Rect 移动后若 semantic 仍缓存旧距离，Safety Condition 会失真。"""
        scene = ScenarioDistribution("dynamic_hazard").sample(62)
        scene["agent"]["position"] = (100.0, 300.0)
        scene["agent"]["heading_degrees"] = 0.0
        scene["obstacles"] = []
        scene["dynamic_hazards"] = [
            {
                "position": (250.0, 300.0),
                "size": (58, 58),
                "speed": 72.0,
                "heading_degrees": 0.0,
            }
        ]
        world = Environment(scene)
        before = world.perception.snapshot.hazard.nearest_clearance
        world.step({"turn": 0.0, "throttle": 0.0}, 0.5)
        after = world.perception.snapshot.hazard.nearest_clearance
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertGreater(after, before)

    def test_context_shift_changes_semantic_hazard_observation(self):
        """仅更新 metadata 而未进入 perception 时，θ(t) 无法观察 context shift。"""
        world = Environment(ScenarioDistribution("context_shift").sample(63))
        command = {"turn": 0.0, "throttle": 0.0}
        low = world.perception.snapshot.hazard.nearest_clearance
        for _ in range(121):
            world.step(command, 1.0 / 60.0)
        high = world.perception.snapshot.hazard.nearest_clearance
        self.assertEqual(world.current_context_phase, "high_risk")
        self.assertNotEqual(low, high)


if __name__ == "__main__":
    unittest.main()
