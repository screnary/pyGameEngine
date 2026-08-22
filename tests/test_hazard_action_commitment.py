"""R0.13 Research Hazard branch running-child commitment regressions."""

import unittest

import py_trees

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution
from scripts.evaluation.eval_condition_threshold_sensitivity import (
    evaluate_threshold_sensitivity,
)


DT = 1.0 / 60.0


def make_commitment_controller() -> BehaviorTreeController:
    """构造中部单 Hazard 场景，初始净空 44 px，小于 45 px threshold。"""
    scene = ScenarioDistribution("static_random").sample(1313)
    scene["agent"].update(
        {"position": (300.0, 300.0), "heading_degrees": 0.0, "initial_speed": 0.0}
    )
    scene["target"].update({"position": (760.0, 300.0)})
    scene["obstacles"] = [(360, 260, 60, 80)]
    scene["dynamic_hazards"] = []
    world = Environment(scene)
    return BehaviorTreeController(
        world,
        bt_config="condition_research",
        condition_parameters=ConditionParameters(hazard_threshold=45.0),
    )


class HazardActionCommitmentTests(unittest.TestCase):
    def test_only_research_hazard_sequence_has_running_child_memory(self):
        """Changing the root or sibling memory would alter unrelated BT semantics."""
        controller = make_commitment_controller()

        self.assertFalse(controller.root.memory)
        self.assertFalse(controller.nodes_by_name["Boundary Recovery"].memory)
        self.assertTrue(controller.nodes_by_name["Hazard Avoidance"].memory)
        self.assertFalse(controller.nodes_by_name["Goal Reached"].memory)

    def test_clearance_crossing_does_not_interrupt_running_avoidance(self):
        """A RUNNING maneuver must not be cancelled by its start Condition."""
        controller = make_commitment_controller()
        world = controller.environment

        controller.tick(DT)
        avoid = controller.nodes_by_name["Avoid Hazard"]
        self.assertEqual(avoid.status, py_trees.common.Status.RUNNING)

        # 将唯一 Hazard 移到 threshold 外；memory Sequence 应从 RUNNING child 继续。
        world.obstacles[0].x = 600
        controller.tick(DT)

        self.assertEqual(controller.active_behavior, "Avoid Hazard")
        self.assertEqual(avoid.status, py_trees.common.Status.RUNNING)

    def test_boundary_recovery_preempts_committed_avoidance_immediately(self):
        """Sequence memory must not turn into a top-level priority lock."""
        controller = make_commitment_controller()
        world = controller.environment
        controller.tick(DT)
        avoid = controller.nodes_by_name["Avoid Hazard"]
        self.assertEqual(avoid.status, py_trees.common.Status.RUNNING)

        world.agent.position.x = 20.0
        controller.tick(DT)

        self.assertEqual(controller.active_behavior, "Safe Boundary Recovery")
        self.assertEqual(avoid.status, py_trees.common.Status.INVALID)

    def test_completed_maneuver_can_reenter_through_hazard_condition(self):
        """Memory must reset after SUCCESS so a persistent Hazard can start again."""
        controller = make_commitment_controller()
        avoid = controller.nodes_by_name["Avoid Hazard"]

        for _ in range(120):
            turn, throttle = controller.tick(DT)
            controller.environment.step(
                {"turn": float(turn), "throttle": float(throttle)}, DT
            )
            if avoid.status == py_trees.common.Status.SUCCESS:
                break
        self.assertEqual(avoid.status, py_trees.common.Status.SUCCESS)

        # 完成后重新把同一 Hazard 放到当前 Agent 附近，明确满足 start Condition。
        position = controller.environment.agent.position
        obstacle = controller.environment.obstacles[0]
        obstacle.left = round(position.x + 30.0)
        obstacle.centery = round(position.y)
        controller.tick(DT)

        self.assertEqual(avoid.status, py_trees.common.Status.RUNNING)
        self.assertEqual(controller.active_behavior, "Avoid Hazard")

    def test_confirmed_seed_no_longer_chatters_at_single_static_hazard(self):
        """The diagnosed 26-switch episode must form a small number of maneuvers."""
        payload = evaluate_threshold_sensitivity(
            families=("dynamic_hazard",),
            seeds=(1001,),
            thresholds=(45.0,),
        )
        row = payload["episodes"][0]

        self.assertTrue(row["success"])
        self.assertEqual(row["collision_count"], 0)
        self.assertLessEqual(row["branch_switch_count"], 13)
        self.assertLessEqual(row["hazard_risk_activation_count"], 6)
        self.assertLessEqual(row["avoid_maneuver_count"], 6)
        self.assertLess(row["path_length"], 1096.15)


if __name__ == "__main__":
    unittest.main()
