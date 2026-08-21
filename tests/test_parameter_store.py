"""验证 R0.6 通用连续参数接口及 Research BT 的运行时读取语义。"""

import math
from types import SimpleNamespace
import unittest

import py_trees

from autonomy_lab.bt.context import BehaviorBuildContext
from autonomy_lab.perception.semantic_perception import (
    AgentState,
    BoundaryPerception,
    GoalPerception,
    HazardObservation,
    HazardPerception,
    SemanticPerception,
)


class ParameterSpecTests(unittest.TestCase):
    def test_spec_normalizes_valid_continuous_metadata(self):
        """若 metadata 未标准化，Store 的数值比较会因 int/float 混用而不稳定。"""
        from autonomy_lab.bt.parameters import ParameterSpec

        spec = ParameterSpec(
            name="avoid_turn_gain",
            value=1,
            default=1,
            min_value=0,
            max_value=2,
        )

        self.assertEqual(spec.name, "avoid_turn_gain")
        self.assertEqual(
            (spec.value, spec.default, spec.min_value, spec.max_value),
            (1.0, 1.0, 0.0, 2.0),
        )

    def test_spec_rejects_invalid_bounds_default_or_current_value(self):
        """非法 metadata 若进入 Store，会绕过后续 set() 的范围约束。"""
        from autonomy_lab.bt.parameters import ParameterSpec

        invalid_specs = (
            {"name": "", "value": 1.0, "default": 1.0, "min_value": 0.0, "max_value": 2.0},
            {"name": "gain", "value": 1.0, "default": 1.0, "min_value": 2.0, "max_value": 0.0},
            {"name": "gain", "value": 1.0, "default": 3.0, "min_value": 0.0, "max_value": 2.0},
            {"name": "gain", "value": math.inf, "default": 1.0, "min_value": 0.0, "max_value": math.inf},
        )

        for kwargs in invalid_specs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ParameterSpec(**kwargs)


class ParameterStoreTests(unittest.TestCase):
    def make_store(self):
        from autonomy_lab.bt.parameters import ParameterSpec, ParameterStore

        return ParameterStore(
            (
                ParameterSpec("hazard_threshold", 90.0, 90.0, 0.0, math.inf),
                ParameterSpec("avoid_turn_gain", 1.0, 1.0, 0.25, 2.0),
            )
        )

    def test_get_set_bounds_and_spec_share_one_generic_interface(self):
        """若 Action 参数需要另一种 Store，本轮的通用接口目标就没有达成。"""
        store = self.make_store()

        self.assertEqual(store.get("hazard_threshold"), 90.0)
        self.assertEqual(store.get("avoid_turn_gain"), 1.0)
        self.assertEqual(store.bounds("avoid_turn_gain"), (0.25, 2.0))

        store.set("hazard_threshold", 105.5)
        store.set("avoid_turn_gain", 1.4)
        self.assertEqual(store.get("hazard_threshold"), 105.5)
        self.assertEqual(store.get("avoid_turn_gain"), 1.4)
        self.assertEqual(store.spec("avoid_turn_gain").value, 1.4)

    def test_spec_returns_a_copy_that_cannot_bypass_store_validation(self):
        """调用者若能直接改内部 spec.value，就能绕过 set() 的范围检查。"""
        store = self.make_store()

        exposed = store.spec("avoid_turn_gain")
        exposed.value = 99.0

        self.assertEqual(store.get("avoid_turn_gain"), 1.0)

    def test_reset_and_reset_all_restore_each_spec_default(self):
        """reset 若误用统一默认值，会破坏不同参数各自的 baseline。"""
        store = self.make_store()
        store.set("hazard_threshold", 120.0)
        store.set("avoid_turn_gain", 1.8)

        store.reset("hazard_threshold")
        self.assertEqual(store.get("hazard_threshold"), 90.0)
        self.assertEqual(store.get("avoid_turn_gain"), 1.8)

        store.reset_all()
        self.assertEqual(store.get("avoid_turn_gain"), 1.0)

    def test_set_rejects_unknown_non_numeric_non_finite_and_out_of_range_values(self):
        """错误参数名或越界值不能静默污染实验的真实 theta。"""
        store = self.make_store()

        with self.assertRaises(KeyError):
            store.set("missing", 1.0)
        for value in (True, "1.0", math.inf, 0.0, 2.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    store.set("avoid_turn_gain", value)


class ConditionParametersCompatibilityTests(unittest.TestCase):
    def test_existing_condition_api_delegates_to_the_generic_store(self):
        """R0.6 迁移不能改变 R0.3/R0.5 的默认值、范围或兼容方法。"""
        from autonomy_lab.bt.parameters import ConditionParameters, ParameterStore

        parameters = ConditionParameters(100.0, 50.0, 25.0)

        self.assertIsInstance(parameters, ParameterStore)
        self.assertEqual(parameters.get("hazard_threshold"), 100.0)
        self.assertEqual(parameters.boundary_threshold, 50.0)
        self.assertEqual(parameters.get_bounds()["goal_threshold"], (0.0, math.inf))

        parameters.goal_threshold = 35.0
        parameters.set_values(hazard_threshold=110.0)
        self.assertEqual(
            parameters.get_values(),
            {
                "hazard_threshold": 110.0,
                "boundary_threshold": 50.0,
                "goal_threshold": 35.0,
            },
        )

        parameters.reset("goal_threshold")
        self.assertEqual(parameters.goal_threshold, 30.0)
        parameters.reset_defaults()
        self.assertEqual(
            tuple(parameters.get_values().values()),
            (90.0, 40.0, 30.0),
        )


class GetOnlyParameterStore:
    """只暴露目标通用接口；Condition 若偷读旧属性会立即失败。"""

    def __init__(self):
        self.values = {
            "hazard_threshold": 50.0,
            "boundary_threshold": 20.0,
            "goal_threshold": 90.0,
        }

    def get(self, name: str) -> float:
        return self.values[name]


def make_get_only_context() -> BehaviorBuildContext:
    hazard = HazardObservation(clearance=60.0, bearing=0.0)
    snapshot = SemanticPerception(
        agent=AgentState(speed=0.0, heading=0.0),
        goal=GoalPerception(
            sensed=True,
            visible=True,
            available=True,
            source="perception",
            distance=100.0,
            bearing=0.0,
            unavailable_reason="",
        ),
        hazard=HazardPerception(
            visible_hazards=(hazard,),
            nearest_hazard=hazard,
        ),
        boundary=BoundaryPerception(30.0, 80.0, 90.0, 100.0),
    )
    return BehaviorBuildContext(
        perception=SimpleNamespace(snapshot=snapshot),
        command={"turn": 0.0, "throttle": 0.0},
        behavior_config={},
        condition_parameters=GetOnlyParameterStore(),
    )


class GenericConditionParameterReadTests(unittest.TestCase):
    def test_research_conditions_read_named_values_from_store_each_update(self):
        """节点若缓存构建时数值或读取属性，未来统一优化器更新不会在下一 tick 生效。"""
        from autonomy_lab.bt.behaviors import GoalReached, HazardRisk, ResearchBoundaryRisk

        context = make_get_only_context()
        conditions = (
            HazardRisk(context, "Hazard Risk?"),
            ResearchBoundaryRisk(context, "Boundary Risk?"),
            GoalReached(context, "Goal Reached?"),
        )

        self.assertEqual(
            tuple(condition.update() for condition in conditions),
            (
                py_trees.common.Status.FAILURE,
                py_trees.common.Status.FAILURE,
                py_trees.common.Status.FAILURE,
            ),
        )

        context.condition_parameters.values.update(
            hazard_threshold=70.0,
            boundary_threshold=40.0,
            goal_threshold=110.0,
        )
        self.assertEqual(
            tuple(condition.update() for condition in conditions),
            (
                py_trees.common.Status.SUCCESS,
                py_trees.common.Status.SUCCESS,
                py_trees.common.Status.SUCCESS,
            ),
        )


if __name__ == "__main__":
    unittest.main()
