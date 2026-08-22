"""R0.10 两个固定安全 Action 的真实 World 回归测试。"""

import unittest

from autonomy_lab.bt.behaviors import (
    AvoidObstacle,
    HazardRisk,
    SafeBoundaryRecovery,
)
from autonomy_lab.bt.context import BehaviorBuildContext
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.config import (
    DEFAULT_RESEARCH_SENSOR_CONFIG,
    get_scene,
)
from scripts.evaluation.eval_action_competence import run_isolated_suite


def make_research_world(
    *,
    position: tuple[float, float],
    heading_degrees: float,
    obstacles: tuple[tuple[int, int, int, int], ...],
) -> Environment:
    """构造只含本测试所需几何的正式 Research perception World。"""
    scene = get_scene("rl_sanity")
    scene["world_size"] = (850, 600)
    scene["agent"].update(
        {"position": position, "heading_degrees": heading_degrees}
    )
    scene["target"]["position"] = (760, 300)
    scene["obstacles"] = list(obstacles)
    scene["sensor"].update(DEFAULT_RESEARCH_SENSOR_CONFIG)
    return Environment(scene)


def make_context(world: Environment) -> tuple[BehaviorBuildContext, dict]:
    command = {"turn": 0.0, "throttle": 0.0}
    return (
        BehaviorBuildContext(
            perception=world.perception,
            command=command,
            behavior_config=world.scene_config["behavior_tree"],
            condition_parameters=ConditionParameters(),
        ),
        command,
    )


class FixedActionSafetyTests(unittest.TestCase):
    def test_boundary_recovery_turns_before_driving_from_outward_heading(self):
        """删除 heading gate 会让该状态在转正前继续撞向左边界。"""
        world = make_research_world(
            position=(30, 300), heading_degrees=180.0, obstacles=()
        )
        context, command = make_context(world)
        action = SafeBoundaryRecovery(context, "Safe Boundary Recovery")

        action.tick_once()

        self.assertNotEqual(command["turn"], 0.0)
        self.assertEqual(command["throttle"], 0.0)

    def test_avoid_hazard_rejects_turn_toward_near_bottom_boundary(self):
        """忽略独立 BoundaryPerception 会让正前方 Hazard 默认向下避让。"""
        world = make_research_world(
            position=(425, 500),
            heading_degrees=0.0,
            obstacles=((500, 455, 70, 90),),
        )
        context, command = make_context(world)
        condition = HazardRisk(context, "Hazard Risk?")
        context.nodes_by_name[condition.name] = condition
        condition.update()
        action = AvoidObstacle(
            context, "Avoid Hazard", condition=condition.name
        )

        action.tick_once()

        self.assertLess(command["turn"], 0.0)
        self.assertEqual(command["throttle"], 0.0)

    def test_original_r09_micro_cases_are_collision_free(self):
        """放宽或替换原微场景会掩盖 R0.9 已记录的真实失败。"""
        rows = run_isolated_suite()
        grouped = {
            action: [row for row in rows if row["action"] == action]
            for action in {row["action"] for row in rows}
        }

        self.assertEqual(sum(row["success"] for row in grouped["MoveToGoal"]), 5)
        self.assertEqual(sum(row["success"] for row in grouped["Stop"]), 2)
        self.assertEqual(sum(row["success"] for row in grouped["AvoidHazard"]), 6)
        self.assertEqual(
            sum(row["collision_count"] for row in grouped["AvoidHazard"]), 0
        )
        self.assertEqual(
            sum(row["success"] for row in grouped["SafeBoundaryRecovery"]), 7
        )
        self.assertEqual(
            sum(
                row["collision_count"]
                for row in grouped["SafeBoundaryRecovery"]
            ),
            0,
        )

    def test_avoid_hazard_does_not_flip_steering_at_control_rate(self):
        """软方向偏好失效时，相邻 sector 会让转向符号几乎逐帧互换。"""
        avoid_rows = [
            row
            for row in run_isolated_suite()
            if row["action"] == "AvoidHazard"
        ]

        self.assertLessEqual(
            max(row["steering_sign_changes"] for row in avoid_rows), 2
        )


if __name__ == "__main__":
    unittest.main()
