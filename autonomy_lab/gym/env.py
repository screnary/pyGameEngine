"""把唯一 World 适配为 Gymnasium API，不复制任何仿真动力学。"""

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..environment import Environment
from ..experiment.recorder import ExperimentRecorder
from ..observation import OBSERVATION_SIZE, build_navigation_observation
from ..scene_config import DEFAULT_SCENARIO, SCENES, get_scene


SIMULATION_DT = 1.0 / 60.0


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
        action_repeat: int = 1,
        contact_penalty_per_step: float = 0.0,
        recorder_controller: str = "gym",
    ) -> None:
        if scenario not in SCENES:
            raise ValueError(f"unknown scenario: {scenario}")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive")
        if isinstance(action_repeat, bool) or not isinstance(action_repeat, int):
            raise ValueError("action_repeat must be a positive integer")
        if action_repeat <= 0:
            raise ValueError("action_repeat must be a positive integer")
        if contact_penalty_per_step > 0.0:
            raise ValueError("contact_penalty_per_step must be zero or negative")

        self.scenario = scenario
        self.render_mode = render_mode
        self.simulation_dt = float(simulation_dt)
        # action_repeat 只改变 Policy 决策频率；World 仍逐次以 simulation_dt 推进。
        # 默认 1/0.0 保持 M4.0、M4.1 既有一步一决策与 reward 行为。
        self.action_repeat = action_repeat
        self.contact_penalty_per_step = float(contact_penalty_per_step)
        self.world = Environment(get_scene(scenario))
        # 传入现有 Recorder 即启用日志；默认 None 避免每次训练 reset 都写磁盘。
        self.recorder = recorder
        self.recorder_controller = recorder_controller
        self.last_episode_payload: dict[str, Any] | None = None
        # Renderer 必须保持 None，直到 human 模式真正需要显示；headless 路径
        # 不初始化 display、event、Clock 或任何 Surface。
        self._renderer = None
        self._collision_active = False
        self.last_reward_components: dict[str, float | int] = {}

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
        self.last_reward_components = {}
        self.last_episode_payload = None
        if self.recorder is not None:
            self.recorder.start_episode(
                self.world,
                self.scenario,
                self.recorder_controller,
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
        """保持一个连续 Action，按 60 Hz 推进一个或多个内部仿真步。"""
        action_array = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action_array):
            raise ValueError("action must be a length-2 vector within [-1, 1]")

        command = {
            "turn": float(action_array[0]),
            "throttle": float(action_array[1]),
        }
        # Ground-truth target distance is used only as privileged reward shaping
        # during training. 它不写入 Observation 或 info，Policy 推理也不读取它。
        # macro step 只取首尾距离，等价于各 internal step progress 的望远镜求和。
        previous_target_distance = (
            self.world.target - self.world.agent.position
        ).length()
        internal_steps = 0
        collision_event_count = 0
        contact_steps = 0
        terminated = False
        truncated = False
        max_time = float(
            self.world.scene_config["experiment"]["max_episode_time"]
        )

        for _ in range(self.action_repeat):
            self.world.step(command, self.simulation_dt)
            internal_steps += 1

            # collision event 与 contact 是两个独立语义：前者只记 False→True，
            # 后者记录每个实际仍接触障碍的 1/60 s 仿真步。
            colliding = bool(self.world.collision_this_step)
            if colliding and not self._collision_active:
                collision_event_count += 1
            if colliding:
                contact_steps += 1
            self._collision_active = colliding

            if self.recorder is not None:
                # Recorder 必须观察每个物理步，不能把 6 个步误记成 1 个决策步。
                self.recorder.update(self.simulation_dt, self.world)

            terminated = bool(self.world.target_reached)
            truncated = bool(
                not terminated and self.world.simulation_time >= max_time
            )
            if terminated or truncated:
                break

        current_target_distance = (
            self.world.target - self.world.agent.position
        ).length()
        world_diagonal = math.hypot(*self.world.world_size)
        target_progress = (
            previous_target_distance - current_target_distance
        ) / world_diagonal

        reward, self.last_reward_components = self._compute_reward(
            target_progress=target_progress,
            terminated=terminated,
            collision_event_count=collision_event_count,
            internal_steps=internal_steps,
            contact_steps=contact_steps,
            simulation_dt=self.simulation_dt,
            contact_penalty_per_step=self.contact_penalty_per_step,
        )
        observation = self._get_observation()
        info = self._get_info()

        if self.recorder is not None:
            if terminated:
                self.last_episode_payload = self.recorder.finish_episode(
                    "SUCCESS", "target_reached"
                )
            elif truncated:
                self.last_episode_payload = self.recorder.finish_episode(
                    "TIMEOUT", "timeout"
                )

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
        # 只降低显示采样频率来匹配 macro step 的物理时长；FPS 不参与动力学。
        control_fps = max(1, round(self.metadata["render_fps"] / self.action_repeat))
        self._renderer.pace(control_fps)

    def close(self) -> None:
        """仅在 human Renderer 已创建时释放显示资源。"""
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish_episode("INTERRUPTED", "env_closed")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @staticmethod
    def _compute_reward(
        target_progress: float,
        terminated: bool,
        collision_event_count: int,
        internal_steps: int,
        contact_steps: int,
        simulation_dt: float,
        contact_penalty_per_step: float,
    ) -> tuple[float, dict[str, float | int]]:
        """按实际 60 Hz 仿真步集中计算并拆分导航 reward。

        ``target_progress`` 是 action 前后 Ground Truth 距离差除以世界对角线。
        Ground-truth target distance is used only as privileged reward shaping
        during training. 归一化仅控制数值尺度，不把 Ground Truth 注入感知
        Observation 或 info。碰撞事件仍按 False→True 扣分，保持与 M2
        ``collision_count`` 一致；可选 contact penalty 则按接触物理步累计。
        """
        components: dict[str, float | int] = {
            "progress_reward": float(target_progress),
            "step_reward": -0.001 * internal_steps,
            "collision_event_reward": -0.05 * collision_event_count,
            "contact_penalty_reward": contact_penalty_per_step * contact_steps,
            "goal_reward": 1.0 if terminated else 0.0,
            "internal_simulation_steps": internal_steps,
            "collision_event_count": collision_event_count,
            "contact_steps": contact_steps,
            "contact_duration": contact_steps * simulation_dt,
        }
        reward = sum(
            float(components[name])
            for name in (
                "progress_reward",
                "step_reward",
                "collision_event_reward",
                "contact_penalty_reward",
                "goal_reward",
            )
        )
        return reward, components

    def _get_observation(self) -> np.ndarray:
        """委托共享编码器，确保 Gym 与 Hybrid BT 的 PPO 输入完全相同。"""
        return build_navigation_observation(self.world)

    def _get_info(self) -> dict[str, Any]:
        """返回评价所需的少量真值，不把完整 World 塞入 info。"""
        return {
            "simulation_time": float(self.world.simulation_time),
            "target_reached": bool(self.world.target_reached),
            "collision": bool(self.world.collision_this_step),
            "scenario": self.scenario,
        }
