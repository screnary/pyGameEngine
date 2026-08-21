"""保护 R0.3 参数化 Condition 与固定 Research BT 的运行时边界。"""

import math
from types import SimpleNamespace
import unittest

import py_trees

from autonomy_lab.bt.context import BehaviorBuildContext
from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.config import get_scene
from autonomy_lab.perception.semantic_perception import (
    AgentState,
    BoundaryPerception,
    GoalPerception,
    HazardObservation,
    HazardPerception,
    SemanticPerception,
)


def make_semantic_context(
    parameters: ConditionParameters,
    *,
    hazard_clearance: float = 60.0,
    boundary_clearances: tuple[float, float, float, float] = (30, 80, 90, 100),
    goal_distance: float = 100.0,
) -> BehaviorBuildContext:
    """构造不带 Environment 的真实语义快照，防止 Condition 偷读 World。"""
    hazard = HazardObservation(clearance=hazard_clearance, bearing=0.2)
    left, right, top, bottom = boundary_clearances
    snapshot = SemanticPerception(
        agent=AgentState(speed=0.0, heading=0.0),
        goal=GoalPerception(
            sensed=True,
            visible=True,
            available=True,
            source="perception",
            distance=goal_distance,
            bearing=0.0,
            unavailable_reason="",
        ),
        hazard=HazardPerception(
            visible_hazards=(hazard,),
            nearest_hazard=hazard,
        ),
        boundary=BoundaryPerception(
            left=float(left),
            right=float(right),
            top=float(top),
            bottom=float(bottom),
        ),
    )
    try:
        return BehaviorBuildContext(
            perception=SimpleNamespace(snapshot=snapshot),
            command={"turn": 0.0, "throttle": 0.0},
            behavior_config={},
            condition_parameters=parameters,
        )
    except TypeError as error:
        raise AssertionError(f"Behavior context does not share parameters: {error}")


class ConditionParameterStoreTests(unittest.TestCase):
    def test_parameters_are_mutable_finite_non_negative_continuous_values(self):
        """缺少共享 Store 或接受无效阈值都会破坏后续连续参数实验。"""
        try:
            from autonomy_lab.bt.parameters import ConditionParameters
        except ImportError as error:
            self.fail(f"ConditionParameters is missing: {error}")

        parameters = ConditionParameters()
        self.assertEqual(
            (
                parameters.hazard_threshold,
                parameters.boundary_threshold,
                parameters.goal_threshold,
            ),
            (90.0, 40.0, 30.0),
        )
        parameters.hazard_threshold = 100.5
        parameters.boundary_threshold = 55
        parameters.goal_threshold = 24.25
        self.assertEqual(
            (
                parameters.hazard_threshold,
                parameters.boundary_threshold,
                parameters.goal_threshold,
            ),
            (100.5, 55.0, 24.25),
        )

        for name, value in (
            ("hazard_threshold", -1.0),
            ("boundary_threshold", math.inf),
            ("goal_threshold", True),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    setattr(parameters, name, value)


class ParameterizedConditionTests(unittest.TestCase):
    def test_hazard_threshold_switches_on_the_same_semantic_observation(self):
        """缓存构建期阈值或读取 World 都会使这次 runtime 切换失败。"""
        try:
            from autonomy_lab.bt.behaviors import HazardRisk
        except ImportError as error:
            self.fail(f"HazardRisk is missing: {error}")
        parameters = ConditionParameters(hazard_threshold=50.0)
        condition = HazardRisk(
            context=make_semantic_context(parameters),
            name="Hazard Risk?",
        )

        self.assertEqual(condition.update(), py_trees.common.Status.FAILURE)
        self.assertIn("d=60", condition.feedback_message)
        self.assertIn("theta=50", condition.feedback_message)
        parameters.hazard_threshold = 70.0
        self.assertEqual(condition.update(), py_trees.common.Status.SUCCESS)
        self.assertIn("theta=70", condition.feedback_message)

    def test_boundary_threshold_switches_on_the_same_semantic_observation(self):
        """Boundary Condition 必须每 tick 读取共享 θ，而不是 JSON 常量。"""
        try:
            from autonomy_lab.bt.behaviors import ResearchBoundaryRisk
        except ImportError as error:
            self.fail(f"ResearchBoundaryRisk is missing: {error}")
        parameters = ConditionParameters(boundary_threshold=20.0)
        condition = ResearchBoundaryRisk(
            context=make_semantic_context(parameters),
            name="Boundary Risk?",
        )

        self.assertEqual(condition.update(), py_trees.common.Status.FAILURE)
        self.assertIn("clearance=30", condition.feedback_message)
        self.assertIn("theta=20", condition.feedback_message)
        parameters.boundary_threshold = 40.0
        self.assertEqual(condition.update(), py_trees.common.Status.SUCCESS)
        self.assertIn("theta=40", condition.feedback_message)

    def test_goal_threshold_switches_on_the_same_semantic_observation(self):
        """GoalReached 必须只使用 sensed Goal distance 与当前 θgoal。"""
        try:
            from autonomy_lab.bt.behaviors import GoalReached
        except ImportError as error:
            self.fail(f"GoalReached is missing: {error}")
        parameters = ConditionParameters(goal_threshold=90.0)
        condition = GoalReached(
            context=make_semantic_context(parameters),
            name="Goal Reached?",
        )

        self.assertEqual(condition.update(), py_trees.common.Status.FAILURE)
        self.assertIn("distance=100", condition.feedback_message)
        self.assertIn("theta=90", condition.feedback_message)
        parameters.goal_threshold = 110.0
        self.assertEqual(condition.update(), py_trees.common.Status.SUCCESS)
        self.assertIn("theta=110", condition.feedback_message)


class ResearchBehaviorTreeTests(unittest.TestCase):
    def make_controller(
        self,
        scenario: str,
        parameters: ConditionParameters,
    ) -> BehaviorTreeController:
        try:
            return BehaviorTreeController(
                Environment(get_scene(scenario)),
                bt_config="condition_research",
                condition_parameters=parameters,
            )
        except (TypeError, FileNotFoundError, ValueError) as error:
            self.fail(f"Research BT cannot be constructed: {error}")

    def test_research_json_builds_fixed_handcrafted_topology(self):
        """复用 legacy JSON 或混入 PPO/Search 会破坏独立研究基线。"""
        parameters = ConditionParameters()
        controller = self.make_controller("rl_sanity", parameters)

        self.assertIs(controller.condition_parameters, parameters)
        self.assertEqual(controller.bt_config_id, "condition_research_bt")
        self.assertEqual(
            [child.name for child in controller.root.children],
            [
                "Boundary Recovery",
                "Hazard Avoidance",
                "Goal Reached",
                "Move To Goal",
            ],
        )
        self.assertNotIn("PPO Navigate", controller.nodes_by_name)
        self.assertNotIn("Search Target", controller.nodes_by_name)

    def test_runtime_hazard_update_changes_active_branch_without_world_step(self):
        """相同 Hazard observation 下增大 theta 必须更早触发 Avoid Hazard。"""
        parameters = ConditionParameters(
            hazard_threshold=200.0,
            boundary_threshold=0.0,
            goal_threshold=0.0,
        )
        controller = self.make_controller("ppo_simple_obstacle", parameters)

        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "Move To Goal")
        parameters.hazard_threshold = 250.0
        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "Avoid Hazard")

    def test_runtime_boundary_update_changes_active_branch_without_world_step(self):
        """相同 Boundary clearance 下增大 theta 必须触发安全恢复。"""
        parameters = ConditionParameters(
            hazard_threshold=0.0,
            boundary_threshold=20.0,
            goal_threshold=0.0,
        )
        controller = self.make_controller("r01_boundary_obstacle", parameters)

        controller.tick(1.0 / 60.0)
        self.assertNotEqual(controller.active_behavior, "Safe Boundary Recovery")
        parameters.boundary_threshold = 30.0
        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "Safe Boundary Recovery")

    def test_runtime_goal_update_changes_active_branch_without_world_step(self):
        """相同 Goal distance 下增大 theta 必须从移动切换到 Stop。"""
        parameters = ConditionParameters(
            hazard_threshold=0.0,
            boundary_threshold=0.0,
            goal_threshold=300.0,
        )
        controller = self.make_controller("rl_sanity", parameters)

        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "Move To Goal")
        parameters.goal_threshold = 450.0
        controller.tick(1.0 / 60.0)
        self.assertEqual(controller.active_behavior, "Stop")


if __name__ == "__main__":
    unittest.main()
