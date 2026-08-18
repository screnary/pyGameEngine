"""把唯一 World 适配为 Gymnasium API，不复制任何仿真动力学。"""

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..environment import Environment
from ..experiment.recorder import ExperimentRecorder
from ..scene_config import DEFAULT_SCENARIO, SCENES, get_scene


SIMULATION_DT = 1.0 / 60.0
OBSERVATION_SIZE = 13


class AgentGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """连续控制二维 Agent 的轻量 Gym Adapter。

    Action 固定为 ``[turn, throttle]``。Observation 只包含 Agent 自身状态、
    当前 ``PerceptionSnapshot`` 和自身到边界的余量；不可见目标不会因为场景
    使用 ground-truth BT 模式而泄漏给 Gym Agent。
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        scenario: str = DEFAULT_SCENARIO,
        render_mode: str | None = None,
        simulation_dt: float = SIMULATION_DT,
        recorder: ExperimentRecorder | None = None,
    ) -> None:
        if scenario not in SCENES:
            raise ValueError(f"unknown scenario: {scenario}")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive")

        self.scenario = scenario
        self.render_mode = render_mode
        self.simulation_dt = float(simulation_dt)
        self.world = Environment(get_scene(scenario))
        # 传入现有 Recorder 即启用日志；默认 None 避免每次训练 reset 都写磁盘。
        self.recorder = recorder
        # Renderer 必须保持 None，直到 human 模式真正需要显示；headless 路径
        # 不初始化 display、event、Clock 或任何 Surface。
        self._renderer = None
        self._collision_active = False

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            high=np.ones(OBSERVATION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置 Gym 与 World Episode，并返回第一份感知 Observation。"""
        # Gymnasium 官方约定：先初始化 Env.np_random，再重置自有状态。
        super().reset(seed=seed)
        del options  # 当前固定场景没有 reset options，显式说明而非静默使用。
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish_episode("INTERRUPTED", "gym_reset")
        self.world.reset(seed=seed)
        self._collision_active = False
        if self.recorder is not None:
            self.recorder.start_episode(
                self.world,
                self.scenario,
                "gym",
                track_bt=False,
            )
        observation = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """把连续 Action 转为现有 Command，再推进唯一 World 一次固定步长。"""
        action_array = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action_array):
            raise ValueError("action must be a length-2 vector within [-1, 1]")

        command = {
            "turn": float(action_array[0]),
            "throttle": float(action_array[1]),
        }
        self.world.step(command, self.simulation_dt)

        # 与 M2 collision_count 相同：只有接触状态 False→True 才是新事件。
        collision_event = self.world.collision_this_step and not self._collision_active
        self._collision_active = self.world.collision_this_step
        terminated = bool(self.world.target_reached)
        max_time = float(
            self.world.scene_config["experiment"]["max_episode_time"]
        )
        truncated = bool(
            not terminated and self.world.simulation_time >= max_time
        )
        reward = self._compute_reward(terminated, collision_event)
        observation = self._get_observation()
        info = self._get_info()

        if self.recorder is not None:
            self.recorder.update(self.simulation_dt, self.world)
            if terminated:
                self.recorder.finish_episode("SUCCESS", "target_reached")
            elif truncated:
                self.recorder.finish_episode("TIMEOUT", "timeout")

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """只读显示当前 World；不推进仿真或刷新感知。"""
        if self.render_mode != "human":
            return
        if self._renderer is None:
            # 延迟导入确保 headless 模式不触及 Renderer 的显示资源。
            from ..rendering.renderer import PygameRenderer

            self._renderer = PygameRenderer(self.world)
        self._renderer.render(self.world, controller_name="gym")
        self._renderer.pace(self.metadata["render_fps"])

    def close(self) -> None:
        """仅在 human Renderer 已创建时释放显示资源。"""
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish_episode("INTERRUPTED", "env_closed")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @staticmethod
    def _compute_reward(terminated: bool, collision_event: bool) -> float:
        """返回 M3 基线 reward；复杂 reward engineering 留到 M4。"""
        reward = -0.001
        if collision_event:
            reward -= 0.05
        if terminated:
            reward += 1.0
        return reward

    def _get_observation(self) -> np.ndarray:
        """把 action 后最终 Agent/Perception 状态编码为固定 13 维向量。

        字段顺序为 speed、heading sin/cos、target visible/distance/bearing、
        obstacle available/distance/bearing、left/right/top/bottom clearance。
        不可见对象的 distance/bearing 使用 0；相邻 available 位负责消歧。
        """
        agent = self.world.agent
        snapshot = self.world.perception.snapshot
        speed = float(np.clip(agent.speed / agent.max_speed, -1.0, 1.0))

        target_visible = float(snapshot.target_visible)
        if snapshot.target_visible:
            world_diagonal = math.hypot(*self.world.world_size)
            target_distance = float(
                np.clip((snapshot.target_distance or 0.0) / world_diagonal, 0.0, 1.0)
            )
            target_bearing = float(
                np.clip((snapshot.target_bearing or 0.0) / math.pi, -1.0, 1.0)
            )
        else:
            # 即使 ground_truth 模式在 Snapshot 内保留目标信息，Gym 也不读取。
            target_distance = 0.0
            target_bearing = 0.0

        obstacle = snapshot.nearest_obstacle
        obstacle_available = float(obstacle is not None)
        if obstacle is None:
            obstacle_distance = 0.0
            obstacle_bearing = 0.0
        else:
            obstacle_distance = float(
                np.clip(
                    obstacle.distance / self.world.perception.sensor_range,
                    0.0,
                    1.0,
                )
            )
            obstacle_bearing = float(
                np.clip(obstacle.bearing / math.pi, -1.0, 1.0)
            )

        width, height = self.world.world_size
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

    def _get_info(self) -> dict[str, Any]:
        """返回评价所需的少量真值，不把完整 World 塞入 info。"""
        return {
            "simulation_time": float(self.world.simulation_time),
            "target_reached": bool(self.world.target_reached),
            "collision": bool(self.world.collision_this_step),
            "scenario": self.scenario,
        }
