"""把 World/Perception 状态编码为冻结 PPO 使用的 13 维 Observation。"""

import math

import numpy as np

from .environment import Environment


OBSERVATION_SIZE = 13


def build_navigation_observation(world: Environment) -> np.ndarray:
    """返回与 M4 PPO 训练完全一致的低维 ``float32`` Observation。

    字段顺序固定为 speed、heading sin/cos、target visible/distance/bearing、
    obstacle available/distance/bearing、left/right/top/bottom clearance。
    不可见对象的 distance/bearing 使用中性值 0，相邻 available 位负责消歧。

    该函数只读取 World；Gym Adapter 与 Frozen PPO BT Node 共用它，避免两条
    推理路径的归一化或字段顺序随代码演进产生偏差。
    """
    snapshot = world.perception.snapshot
    agent_state = snapshot.agent
    goal = snapshot.goal
    hazard = snapshot.hazard
    boundary = snapshot.boundary
    speed = float(
        np.clip(agent_state.speed / world.agent.max_speed, -1.0, 1.0)
    )

    target_visible = float(goal.visible)
    if goal.visible:
        world_diagonal = math.hypot(*world.world_size)
        target_distance = float(
            np.clip((goal.distance or 0.0) / world_diagonal, 0.0, 1.0)
        )
        target_bearing = float(
            np.clip((goal.bearing or 0.0) / math.pi, -1.0, 1.0)
        )
    else:
        # 即使 BT ground_truth 模式在 Snapshot 内保留目标信息，PPO 也不读取。
        target_distance = 0.0
        target_bearing = 0.0

    nearest_hazard = hazard.nearest_hazard
    obstacle_available = float(nearest_hazard is not None)
    if nearest_hazard is None:
        obstacle_distance = 0.0
        obstacle_bearing = 0.0
    else:
        obstacle_distance = float(
            np.clip(
                nearest_hazard.clearance / world.perception.sensor_range,
                0.0,
                1.0,
            )
        )
        obstacle_bearing = float(
            np.clip(nearest_hazard.bearing / math.pi, -1.0, 1.0)
        )

    width, height = world.world_size
    clearances = (
        np.clip(boundary.left / width, 0.0, 1.0),
        np.clip(boundary.right / width, 0.0, 1.0),
        np.clip(boundary.top / height, 0.0, 1.0),
        np.clip(boundary.bottom / height, 0.0, 1.0),
    )
    return np.asarray(
        [
            speed,
            math.sin(agent_state.heading),
            math.cos(agent_state.heading),
            target_visible,
            target_distance,
            target_bearing,
            obstacle_available,
            obstacle_distance,
            obstacle_bearing,
            *clearances,
        ],
        dtype=np.float32,
    )
