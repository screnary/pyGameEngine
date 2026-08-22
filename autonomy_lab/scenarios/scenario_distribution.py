"""生成 R0.4 研究场景族，同时保留现有固定 Scenario 不变。

模块只负责把 ``family + seed`` 转换成普通 scene dict。返回值继续由现有
``Environment`` 消费，因此这里不持有 World、Renderer 或 Episode 状态。
"""

from __future__ import annotations

from copy import deepcopy
import math
import random

from .config import (
    DEFAULT_BT_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_RESEARCH_SENSOR_CONFIG,
    DEFAULT_SENSOR_CONFIG,
    get_scene,
)


RESEARCH_FAMILIES = frozenset(
    {
        "static_random",
        "dense_hazard",
        "dynamic_hazard",
        "noisy_perception",
        "context_shift",
        "narrow_passage",
        "boundary_hazard",
    }
)


class ScenarioDistribution:
    """一个绑定单一 family 的轻量、可复现场景采样器。"""

    def __init__(self, family: str) -> None:
        if family not in RESEARCH_FAMILIES:
            choices = ", ".join(sorted(RESEARCH_FAMILIES))
            raise ValueError(f"unknown research scenario family '{family}'; use {choices}")
        self.family = family

    def sample(self, seed: int) -> dict:
        """使用局部 RNG 采样 scene；不读取或修改全局 random 状态。"""
        numeric_seed = int(seed)
        if self.family in {"narrow_passage", "boundary_hazard"}:
            return self._fixed_regression(numeric_seed)

        rng = random.Random(numeric_seed)
        scene = self._base_scene(rng, numeric_seed)
        if self.family == "dense_hazard":
            scene["obstacles"] = self._sample_obstacles(rng, count=5, dense=True)
        else:
            scene["obstacles"] = self._sample_obstacles(rng, count=2, dense=False)

        if self.family in {"dynamic_hazard", "context_shift"}:
            # position 表示矩形中心；Environment 把它变成与静态 Hazard 共用的 Rect。
            heading = rng.choice((-90.0, 90.0))
            scene["dynamic_hazards"] = [
                {
                    "position": (470.0, float(rng.randint(190, 460))),
                    "size": (58, 58),
                    "speed": 72.0,
                    "heading_degrees": heading,
                }
            ]

        if self.family == "noisy_perception":
            scene["perception_noise"] = {"hazard_range_std": 10.0}

        if self.family == "context_shift":
            # 固定三阶段足以验证 within-episode context shift；不建立通用事件系统。
            scene["context_schedule"] = [
                {
                    "name": "low_risk",
                    "start_time": 0.0,
                    "hazard_speed_scale": 0.6,
                    "noise_level": 2.0,
                },
                {
                    "name": "high_risk",
                    "start_time": 2.0,
                    "hazard_speed_scale": 1.6,
                    "noise_level": 14.0,
                },
                {
                    "name": "recovery",
                    "start_time": 4.0,
                    "hazard_speed_scale": 0.8,
                    "noise_level": 3.0,
                },
            ]

        dynamic_count = len(scene.get("dynamic_hazards", ()))
        noise_level = float(
            scene.get("perception_noise", {}).get("hazard_range_std", 0.0)
        )
        if self.family == "context_shift":
            noise_level = float(scene["context_schedule"][0]["noise_level"])
        scene["research_metadata"] = {
            "family": self.family,
            "seed": numeric_seed,
            "dynamic_hazard_enabled": dynamic_count > 0,
            "noise_level": noise_level,
            "hazard_count": len(scene["obstacles"]) + dynamic_count,
            "context_schedule": tuple(
                phase["name"] for phase in scene.get("context_schedule", ())
            ),
        }
        return scene

    def _base_scene(self, rng: random.Random, seed: int) -> dict:
        """创建所有随机 family 共用的合法起终点与轻量显示配置。"""
        agent_position = (float(rng.randint(85, 135)), float(rng.randint(120, 530)))
        goal_position = (float(rng.randint(765, 815)), float(rng.randint(120, 530)))
        direction = math.degrees(
            math.atan2(
                goal_position[1] - agent_position[1],
                goal_position[0] - agent_position[0],
            )
        )
        return {
            "name": f"R0.4 {self.family}",
            "world_size": (900, 650),
            "seed": seed,
            "agent": {
                "position": agent_position,
                "heading_degrees": direction + rng.uniform(-12.0, 12.0),
                "initial_speed": 0.0,
                "max_speed": 220.0,
                "turn_speed_degrees": 150.0,
                "radius": 16,
            },
            "target": {"position": goal_position, "radius": 18},
            "obstacles": [],
            # Research profile 自身按 360° finite range 决定 Goal availability；本字段
            # 只为满足共享 Scene schema 保留，不能绕过 Research range gate。
            "target_information_mode": "perceived",
            "sensor": {
                **DEFAULT_SENSOR_CONFIG,
                **DEFAULT_RESEARCH_SENSOR_CONFIG,
                "range": 720.0,
                "fov_degrees": 220.0,
            },
            "behavior_tree": {**DEFAULT_BT_CONFIG},
            "experiment": {
                **DEFAULT_EXPERIMENT_CONFIG,
                "max_episode_time": 20.0,
            },
            "display": {
                "background_color": (22, 27, 36),
                "agent_color": (60, 170, 255),
                "target_color": (80, 220, 120),
                "obstacle_color": (105, 112, 125),
                "text_color": (225, 230, 238),
                "show_fps": True,
            },
        }

    @staticmethod
    def _sample_obstacles(
        rng: random.Random, *, count: int, dense: bool
    ) -> list[tuple[int, int, int, int]]:
        """在中部带采样不会封死上下通道的矩形 Hazard。"""
        x_slots = [260, 355, 450, 545, 640] if dense else [330, 570]
        hazards: list[tuple[int, int, int, int]] = []
        for index, base_x in enumerate(x_slots[:count]):
            width = rng.randint(48, 76) if dense else rng.randint(55, 88)
            height = rng.randint(80, 150) if dense else rng.randint(90, 175)
            # 交替靠上/靠下布置，但保留至少一条宽裕外侧路径。
            if index % 2 == 0:
                y = rng.randint(105, 235)
            else:
                y = rng.randint(315, 445)
            x_jitter = rng.randint(-18, 18)
            hazards.append((base_x + x_jitter, y, width, height))
        return hazards

    def _fixed_regression(self, seed: int) -> dict:
        """把 R0.1 固定几何暴露为 research grouping，不做任何随机化。"""
        source = (
            "r01_narrow_passage"
            if self.family == "narrow_passage"
            else "r01_boundary_obstacle"
        )
        scene = deepcopy(get_scene(source))
        scene["seed"] = seed
        # 固定 regression 只切换感知 profile，不改 R0.1 的任何几何与 BT 参数。
        scene["sensor"].update(DEFAULT_RESEARCH_SENSOR_CONFIG)
        scene["research_metadata"] = {
            "family": self.family,
            "seed": seed,
            "dynamic_hazard_enabled": False,
            "noise_level": 0.0,
            "hazard_count": len(scene["obstacles"]),
            "context_schedule": (),
        }
        return scene
