"""把真实 Hybrid BT Runtime 适配为只在 PPO ownership 时决策的 Gym Env。"""

import math
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..bt.controller import PANEL_WIDTH, BehaviorTreeController
from ..core.environment import Environment
from ..experiment.recorder import ExperimentRecorder
from ..core.observation import OBSERVATION_SIZE, build_navigation_observation
from ..scenarios.config import SCENES, get_scene
from .env import AgentGymEnv, SIMULATION_DT


DEFAULT_TRAINING_SCENARIOS = (
    "ppo_simple_obstacle",
    "m43_obstacle_shift",
    "m43_reverse_detour",
)


class HybridPPOEnv(gym.Env[np.ndarray, np.ndarray]):
    """训练 Hybrid Tree 中 ``PPONavigate`` 的轻量 Gymnasium Adapter。

    一个 Gym ``step(action)`` 从当前 PPO decision point 开始，以 60 Hz 推进
    BT 和 World。PPO Action 最多控制 6 个物理步；若 Boundary/Search 接管，
    Adapter 改用真实 BT Command 继续推进，直到 PPO re-entry 或 Episode 结束。
    因而 SB3 不会把 BT 自主动作误当作自己刚输出的 Action。
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        scenarios: Sequence[str] = DEFAULT_TRAINING_SCENARIOS,
        render_mode: str | None = None,
        simulation_dt: float = SIMULATION_DT,
        ppo_hold_steps: int = 6,
        bt_config: str = "hybrid_ppo",
        recorder: ExperimentRecorder | None = None,
        recorder_controller: str = "hybrid_context_ppo",
    ) -> None:
        scenario_names = tuple(scenarios)
        if not scenario_names:
            raise ValueError("scenarios must contain at least one scene")
        unknown = [name for name in scenario_names if name not in SCENES]
        if unknown:
            raise ValueError(f"unknown scenario: {unknown[0]}")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive")
        if isinstance(ppo_hold_steps, bool) or not isinstance(ppo_hold_steps, int):
            raise ValueError("ppo_hold_steps must be a positive integer")
        if ppo_hold_steps <= 0:
            raise ValueError("ppo_hold_steps must be a positive integer")

        self.scenarios = scenario_names
        self.render_mode = render_mode
        self.simulation_dt = float(simulation_dt)
        self.ppo_hold_steps = ppo_hold_steps
        self.bt_config = bt_config
        self.recorder = recorder
        self.recorder_controller = recorder_controller
        self._renderer = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array(
                [-1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            high=np.ones(OBSERVATION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        self.last_reward_components: dict[str, float | int] = {}
        self.last_episode_payload: dict[str, Any] | None = None
        self._collision_active = False
        self.ppo_active_time = 0.0
        self.ppo_reentry_count = 0
        self._awaiting_reentry = False
        self.observation_before_preemption: list[float] | None = None
        self.observation_after_reentry: list[float] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """选择一个固定训练场景，并停在第一个 PPO decision point。"""
        super().reset(seed=seed)
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish_episode("INTERRUPTED", "gym_reset")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        requested = None if options is None else options.get("scenario")
        if requested is not None:
            if requested not in self.scenarios:
                raise ValueError("reset scenario must be one of configured scenarios")
            self.scenario = str(requested)
        elif len(self.scenarios) == 1:
            self.scenario = self.scenarios[0]
        else:
            index = int(self.np_random.integers(0, len(self.scenarios)))
            self.scenario = self.scenarios[index]

        self.world = Environment(get_scene(self.scenario))
        self.world.reset(seed=seed)
        self.controller = BehaviorTreeController(
            self.world,
            bt_config=self.bt_config,
            external_ppo_control=True,
        )
        self._collision_active = False
        self.last_reward_components = {}
        self.last_episode_payload = None
        self.ppo_active_time = 0.0
        self.ppo_reentry_count = 0
        self._awaiting_reentry = False
        self.observation_before_preemption = None
        self.observation_after_reentry = None
        if self.recorder is not None:
            self.recorder.start_episode(
                self.world,
                self.scenario,
                self.recorder_controller,
                track_bt=True,
                bt_config_id=self.controller.bt_config_id,
            )

        # t=0 的一次 BT supervision 只决定初始 ownership，不推进物理时间。
        self.controller.tick(0.0)
        if not self.controller.ppo_action_required:
            raise RuntimeError(
                "HybridPPOEnv reset requires an initial PPO decision point"
            )
        observation = build_navigation_observation(self.world)
        if self.render_mode == "human":
            self.render()
        return observation, self._get_info(0, 0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """推进至下一次 PPO decision、任务终止或时间截断。"""
        action_array = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action_array):
            raise ValueError("action must be a length-2 vector within [-1, 1]")
        if not self.controller.ppo_action_required:
            raise RuntimeError("step called while PPO does not own control")

        previous_distance = (
            self.world.target - self.world.agent.position
        ).length()
        self.controller.set_ppo_action(action_array)
        ppo_steps_this_action = 0
        ppo_controlled_steps = 0
        internal_steps = 0
        collision_events = 0
        contact_steps = 0
        terminated = False
        truncated = False
        max_time = float(
            self.world.scene_config["experiment"]["max_episode_time"]
        )

        while True:
            ppo_owned_step = self.controller.ppo_active
            active_action = self.controller.active_behavior
            command = dict(self.controller.command)
            self.world.step(command, self.simulation_dt)
            internal_steps += 1
            if ppo_owned_step:
                ppo_steps_this_action += 1
                ppo_controlled_steps += 1
                self.ppo_active_time += self.simulation_dt

            colliding = bool(self.world.collision_this_step)
            if colliding and not self._collision_active:
                collision_events += 1
            if colliding:
                contact_steps += 1
            self._collision_active = colliding

            if self.recorder is not None:
                self.recorder.update(
                    self.simulation_dt,
                    self.world,
                    active_action=active_action,
                    bt_ticked=True,
                )
            if self.render_mode == "human":
                self.render()

            terminated = bool(self.world.target_reached)
            truncated = bool(
                not terminated and self.world.simulation_time >= max_time
            )
            if terminated or truncated:
                break

            # 在 interval 末端先撤销旧 PPO Action，再由下一次 60 Hz BT tick
            # 判定是请求新 Action，还是让 Boundary/Search 接管。
            if ppo_owned_step and ppo_steps_this_action >= self.ppo_hold_steps:
                self.controller.clear_ppo_action()
            previous_ppo_active = ppo_owned_step
            self.controller.tick(self.simulation_dt)
            current_ppo_active = self.controller.ppo_active

            if previous_ppo_active and not current_ppo_active:
                self._awaiting_reentry = True
                self.observation_before_preemption = (
                    build_navigation_observation(self.world).tolist()
                )
            elif (
                not previous_ppo_active
                and current_ppo_active
                and self._awaiting_reentry
            ):
                self.ppo_reentry_count += 1
                self._awaiting_reentry = False
                self.observation_after_reentry = (
                    build_navigation_observation(self.world).tolist()
                )

            if self.controller.ppo_action_required:
                break

        current_distance = (
            self.world.target - self.world.agent.position
        ).length()
        target_progress = (previous_distance - current_distance) / math.hypot(
            *self.world.world_size
        )
        reward, self.last_reward_components = AgentGymEnv._compute_reward(
            target_progress=target_progress,
            terminated=terminated,
            collision_event_count=collision_events,
            internal_steps=internal_steps,
            contact_steps=contact_steps,
            simulation_dt=self.simulation_dt,
            contact_penalty_per_step=0.0,
        )
        observation = build_navigation_observation(self.world)
        info = self._get_info(internal_steps, ppo_controlled_steps)

        if self.recorder is not None:
            if terminated:
                self.last_episode_payload = self.recorder.finish_episode(
                    "SUCCESS", "target_reached"
                )
            elif truncated:
                self.last_episode_payload = self.recorder.finish_episode(
                    "TIMEOUT", "timeout"
                )
        return observation, reward, terminated, truncated, info

    def _get_info(
        self, internal_steps: int, ppo_controlled_steps: int
    ) -> dict[str, Any]:
        simulation_time = float(self.world.simulation_time)
        return {
            "scenario": self.scenario,
            "simulation_time": simulation_time,
            "target_reached": bool(self.world.target_reached),
            "collision": bool(self.world.collision_this_step),
            "internal_simulation_steps": internal_steps,
            "ppo_controlled_steps": ppo_controlled_steps,
            "ppo_action_required": self.controller.ppo_action_required,
            "ppo_decision_count": self.controller.ppo_decision_count,
            "ppo_active_time": self.ppo_active_time,
            "ppo_active_ratio": (
                self.ppo_active_time / simulation_time
                if simulation_time > 0.0
                else 0.0
            ),
            "boundary_recovery_activation_count": (
                self.controller.boundary_recovery_activation_count
            ),
            "search_activation_count": self.controller.search_activation_count,
            "ppo_preemption_count": self.controller.ppo_preemption_count,
            "ppo_reentry_count": self.ppo_reentry_count,
            # 这里只保存 13-D sensed state，不含 Ground Truth Target position。
            "observation_before_preemption": self.observation_before_preemption,
            "observation_after_reentry": self.observation_after_reentry,
        }

    def render(self) -> None:
        """显示当前 World/BT 状态；渲染不推进仿真。"""
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from ..rendering.renderer import PygameRenderer

            self._renderer = PygameRenderer(self.world, panel_width=PANEL_WIDTH)
        self._renderer.render(
            self.world, self.controller, self.recorder_controller
        )
        self._renderer.pace(self.metadata["render_fps"])

    def close(self) -> None:
        """结束活动日志并释放可选 Renderer。"""
        if self.recorder is not None and self.recorder.active:
            self.recorder.finish_episode("INTERRUPTED", "env_closed")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
