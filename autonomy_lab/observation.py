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
    agent = world.agent
    snapshot = world.perception.snapshot
    speed = float(np.clip(agent.speed / agent.max_speed, -1.0, 1.0))

    target_visible = float(snapshot.target_visible)
    if snapshot.target_visible:
        world_diagonal = math.hypot(*world.world_size)
        target_distance = float(
            np.clip((snapshot.target_distance or 0.0) / world_diagonal, 0.0, 1.0)
        )
        target_bearing = float(
            np.clip((snapshot.target_bearing or 0.0) / math.pi, -1.0, 1.0)
        )
    else:
        # 即使 BT ground_truth 模式在 Snapshot 内保留目标信息，PPO 也不读取。
        target_distance = 0.0
        target_bearing = 0.0

    obstacle = snapshot.nearest_obstacle
    obstacle_available = float(obstacle is not None)
    if obstacle is None:
        obstacle_distance = 0.0
        obstacle_bearing = 0.0
    else:
        obstacle_distance = float(
            np.clip(obstacle.distance / world.perception.sensor_range, 0.0, 1.0)
        )
        obstacle_bearing = float(
            np.clip(obstacle.bearing / math.pi, -1.0, 1.0)
        )

    width, height = world.world_size
    radius = agent.radius
    clearances = (
        np.clip((agent.position.x - radius) / width, 0.0, 1.0),
        np.clip((width - radius - agent.position.x) / width, 0.0, 1.0),
        np.clip((agent.position.y - radius) / height, 0.0, 1.0),
        np.clip((height - radius - agent.position.y) / height, 0.0, 1.0),
    )
    return np.asarray(
        [
            speed,
            math.sin(agent.heading),
            math.cos(agent.heading),
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
