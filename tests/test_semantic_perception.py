"""保护 R0.2 的 simulator-neutral 语义感知契约与历史兼容边界。"""

import unittest
import math
from types import SimpleNamespace

import numpy as np

from autonomy_lab.core.environment import Environment
from autonomy_lab.core.observation import build_navigation_observation
from autonomy_lab.perception import semantic_perception as semantic
from autonomy_lab.scenarios.config import get_scene


class SemanticDataModelTests(unittest.TestCase):
    def test_semantic_snapshot_groups_only_plain_immutable_values(self):
        """缺失语义层或把 pygame/World 对象塞进快照都会破坏未来 Adapter。"""
        self.assertIsNotNone(semantic, "SemanticPerception module is missing")

        gap = semantic.NavigationGap(
            bearing=0.25,
            free_distance=120.0,
            angular_width=0.5,
            entry_position=(10.0, 20.0),
        )
        nearest = semantic.HazardObservation(clearance=18.0, bearing=-0.2)
        snapshot = semantic.SemanticPerception(
            agent=semantic.AgentState(speed=12.0, heading=0.5),
            goal=semantic.GoalPerception(
                sensed=True,
                visible=True,
                available=True,
                source="perception",
                distance=100.0,
                bearing=0.1,
                unavailable_reason="",
            ),
            hazard=semantic.HazardPerception(
                visible_hazards=(nearest,),
                nearest_hazard=nearest,
                sector_ranges=(
                    semantic.SectorRange(bearing=0.0, clearance=40.0),
                ),
                traversable_gaps=(gap,),
                best_exploration_gap=gap,
                goal_direction_blocked=True,
                best_goal_gap=gap,
            ),
            boundary=semantic.BoundaryPerception(
                left=10.0,
                right=20.0,
                top=30.0,
                bottom=40.0,
            ),
        )

        self.assertEqual(snapshot.agent.speed, 12.0)
        self.assertTrue(snapshot.goal.sensed)
        self.assertEqual(snapshot.hazard.nearest_clearance, 18.0)
        self.assertEqual(snapshot.hazard.nearest_bearing, -0.2)
        self.assertTrue(snapshot.hazard.available)
        self.assertEqual(snapshot.boundary.bottom, 40.0)
        with self.assertRaises((AttributeError, TypeError)):
            snapshot.goal.distance = 5.0

        def assert_plain(value):
            if hasattr(value, "__dataclass_fields__"):
                for field_name in value.__dataclass_fields__:
                    assert_plain(getattr(value, field_name))
            elif isinstance(value, tuple):
                for item in value:
                    assert_plain(item)
            else:
                self.assertIsInstance(value, (bool, float, str, type(None)))

        assert_plain(snapshot)


class SemanticConstructionTests(unittest.TestCase):
    def test_agent_perception_emits_nested_semantics_and_read_only_legacy_views(self):
        """旧平铺快照继续作为主输出会让新研究代码无法使用语义边界。"""
        world = Environment(get_scene("ppo_simple_obstacle"))
        snapshot = world.perception.snapshot

        self.assertIsInstance(snapshot, semantic.SemanticPerception)
        self.assertEqual(snapshot.agent.speed, world.agent.speed)
        self.assertEqual(snapshot.agent.heading, world.agent.heading)
        self.assertTrue(snapshot.goal.sensed)
        self.assertTrue(snapshot.goal.visible)
        self.assertTrue(snapshot.goal.available)
        self.assertEqual(len(snapshot.hazard.sector_ranges), 12)
        self.assertTrue(snapshot.hazard.available)

        width, height = world.world_size
        radius = world.agent.radius
        self.assertEqual(
            snapshot.boundary,
            semantic.BoundaryPerception(
                left=world.agent.position.x - radius,
                right=width - radius - world.agent.position.x,
                top=world.agent.position.y - radius,
                bottom=height - radius - world.agent.position.y,
            ),
        )

        # Compatibility fields are property views over the exact nested values.
        self.assertEqual(snapshot.target_visible, snapshot.goal.visible)
        self.assertEqual(snapshot.target_available, snapshot.goal.available)
        self.assertEqual(snapshot.target_distance, snapshot.goal.distance)
        self.assertEqual(snapshot.target_bearing, snapshot.goal.bearing)
        self.assertIs(snapshot.nearest_obstacle, snapshot.hazard.nearest_hazard)
        self.assertIs(snapshot.visible_obstacles, snapshot.hazard.visible_hazards)
        self.assertIs(snapshot.sector_clearances, snapshot.hazard.sector_ranges)
        with self.assertRaises((AttributeError, TypeError)):
            snapshot.target_visible = False

    def test_narrow_passage_keeps_goal_sensing_separate_from_hazard_clearance(self):
        """把 footprint clearance 写入 Goal sensing 会重新引入狭缝误判。"""
        world = Environment(get_scene("r01_narrow_passage"))
        snapshot = world.perception.snapshot

        self.assertTrue(snapshot.goal.sensed)
        self.assertTrue(snapshot.goal.visible)
        self.assertTrue(snapshot.goal.available)
        self.assertTrue(snapshot.hazard.goal_direction_blocked)
        center = min(
            snapshot.hazard.sector_ranges,
            key=lambda sector: abs(sector.bearing),
        )
        self.assertLess(center.clearance, snapshot.goal.distance)


class LegacyObservationCompatibilityTests(unittest.TestCase):
    def test_builder_consumes_nested_semantics_without_legacy_properties(self):
        """Observation Builder 回读平铺 target/obstacle 字段会阻塞新 Adapter。"""
        nearest = semantic.HazardObservation(clearance=100.0, bearing=-math.pi / 2)
        snapshot = SimpleNamespace(
            agent=semantic.AgentState(speed=120.0, heading=math.pi / 2),
            goal=semantic.GoalPerception(
                sensed=True,
                visible=True,
                available=True,
                source="perception",
                distance=500.0,
                bearing=math.pi / 2,
                unavailable_reason="",
            ),
            hazard=semantic.HazardPerception(
                visible_hazards=(nearest,),
                nearest_hazard=nearest,
            ),
            boundary=semantic.BoundaryPerception(
                left=60.0,
                right=120.0,
                top=200.0,
                bottom=400.0,
            ),
        )
        world = SimpleNamespace(
            agent=SimpleNamespace(max_speed=240.0),
            world_size=(600, 800),
            perception=SimpleNamespace(snapshot=snapshot, sensor_range=400.0),
        )

        try:
            observation = build_navigation_observation(world)
        except AttributeError as error:
            self.fail(f"builder still depends on legacy/raw fields: {error}")

        np.testing.assert_allclose(
            observation,
            np.asarray(
                [
                    0.5,
                    1.0,
                    0.0,
                    1.0,
                    0.5,
                    0.5,
                    1.0,
                    0.25,
                    -0.5,
                    0.1,
                    0.2,
                    0.25,
                    0.5,
                ],
                dtype=np.float32,
            ),
            atol=1e-7,
        )
        self.assertEqual(observation.shape, (13,))
        self.assertEqual(observation.dtype, np.float32)

    def test_frozen_scenario_initial_observations_remain_numerically_identical(self):
        """字段顺序、归一化或中性值变化会使历史 PPO checkpoint 失配。"""
        expected = {
            "rl_sanity": [
                0.0, 0.0, 1.0, 1.0, 0.4649905562, 0.0, 0.0, 0.0, 0.0,
                0.1485714316, 0.8057143092, 0.4679999948, 0.4679999948,
            ],
            "ppo_simple_obstacle": [
                0.0, 0.0, 1.0, 1.0, 0.6247401237, 0.0, 1.0, 0.3342857063,
                0.0, 0.0988235325, 0.8635293841, 0.4733333290,
                0.4733333290,
            ],
            "ppo_simple_obstacles": [
                0.0, 0.0, 1.0, 1.0, 0.6247401237, 0.0, 1.0, 0.3206689954,
                -0.0198685247, 0.0988235325, 0.8635293841, 0.4650000036,
                0.4816666543,
            ],
        }
        for scenario, values in expected.items():
            with self.subTest(scenario=scenario):
                world = Environment(get_scene(scenario))
                np.testing.assert_allclose(
                    build_navigation_observation(world),
                    np.asarray(values, dtype=np.float32),
                    atol=1e-7,
                )


if __name__ == "__main__":
    unittest.main()
