"""运行 BT/PPO Episode，并返回 Controller 间可比较的公共结果。

本模块只负责单个 Episode 的执行。场景列表、seed 编排、milestone 分组、
CSV 路径和 summary 统计仍由 ``scripts/`` 中的具体实验入口负责。
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from autonomy_lab.bt.controller import PANEL_WIDTH, BehaviorTreeController
from autonomy_lab.environment import Environment
from autonomy_lab.experiment.recorder import ExperimentRecorder
from autonomy_lab.gym.env import AgentGymEnv, SIMULATION_DT
from autonomy_lab.gym.hybrid_env import HybridPPOEnv
from autonomy_lab.scene_config import get_scene


BT_DECISION_FREQUENCY = 60.0
PPO_ACTION_REPEAT = 6
PPO_DECISION_FREQUENCY = 60.0 / PPO_ACTION_REPEAT


def capture_initial_state(world: Environment) -> dict[str, Any]:
    """复制公平比较所需的 World 初态，不保留运行时对象引用。"""
    return {
        "world_size": [float(value) for value in world.world_size],
        "agent_position": [
            float(world.agent.position.x),
            float(world.agent.position.y),
        ],
        "agent_heading": float(world.agent.heading),
        "agent_speed": float(world.agent.speed),
        "agent_radius": float(world.agent.radius),
        "target_position": [float(world.target.x), float(world.target.y)],
        "target_radius": float(world.target_radius),
        "obstacles": [
            [float(rect.x), float(rect.y), float(rect.width), float(rect.height)]
            for rect in world.obstacles
        ],
    }


def assert_initial_states_match(
    bt_state: dict[str, Any],
    ppo_state: dict[str, Any],
    absolute_tolerance: float = 1e-7,
) -> None:
    """用合理浮点容差确认 BT/PPO 从同一 World 初态启动。"""
    for field in (
        "world_size",
        "agent_position",
        "target_position",
        "obstacles",
    ):
        if not np.allclose(
            np.asarray(bt_state[field], dtype=float),
            np.asarray(ppo_state[field], dtype=float),
            rtol=0.0,
            atol=absolute_tolerance,
        ):
            raise AssertionError(f"initial {field} differs between BT and PPO")
    for field in (
        "agent_heading",
        "agent_speed",
        "agent_radius",
        "target_radius",
    ):
        if not math.isclose(
            float(bt_state[field]),
            float(ppo_state[field]),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(f"initial {field} differs between BT and PPO")


def _comparison_row(
    payload: dict[str, Any],
    decision_frequency: float,
    decision_count: int,
) -> dict[str, Any]:
    """把 Recorder payload 转换为 Controller 共用的性能字段。"""
    return {
        "controller": payload["controller"],
        "scenario": payload["scenario"],
        "seed": int(payload["seed"]),
        "success": payload["result"] == "SUCCESS",
        "elapsed_time": float(payload["elapsed_time"]),
        "path_length": float(payload["path_length"]),
        "collision_count": int(payload["collision_count"]),
        "termination_reason": payload["termination_reason"],
        "decision_frequency_hz": decision_frequency,
        "decision_count": decision_count,
    }


def _pygame_window_closed() -> bool:
    """Human demo 只响应关闭窗口；键盘输入不参与 Controller 决策。"""
    import pygame

    return any(event.type == pygame.QUIT for event in pygame.event.get())


def run_bt_episode(
    scenario: str,
    seed: int,
    output_dir: Path,
    render_mode: str | None = None,
    bt_config: str = "default",
    recorder_controller: str = "bt",
    collect_diagnostics: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按当前正式 60 Hz BT tick 运行一个完整 World Episode。"""
    world = Environment(get_scene(scenario))
    world.reset(seed=seed)
    initial_state = capture_initial_state(world)
    controller = BehaviorTreeController(world, bt_config=bt_config)
    recorder = ExperimentRecorder(output_dir)
    recorder.start_episode(
        world,
        scenario,
        recorder_controller,
        track_bt=True,
        bt_config_id=controller.bt_config_id,
    )
    renderer = None
    if render_mode == "human":
        from autonomy_lab.rendering.renderer import PygameRenderer

        renderer = PygameRenderer(world, panel_width=PANEL_WIDTH)

    decisions = 0
    payload: dict[str, Any] | None = None
    max_time = float(world.scene_config["experiment"]["max_episode_time"])
    try:
        while payload is None:
            if renderer is not None and _pygame_window_closed():
                payload = recorder.finish_episode("INTERRUPTED", "window_closed")
                break

            turn, throttle = controller.tick(SIMULATION_DT)
            world.step({"turn": turn, "throttle": throttle}, SIMULATION_DT)
            decisions += 1
            recorder.update(
                SIMULATION_DT,
                world,
                active_action=controller.active_behavior,
                bt_ticked=True,
            )

            if world.target_reached:
                payload = recorder.finish_episode("SUCCESS", "target_reached")
            elif world.simulation_time >= max_time:
                payload = recorder.finish_episode("TIMEOUT", "timeout")

            if renderer is not None:
                renderer.render(world, controller, recorder_controller)
                renderer.pace(round(BT_DECISION_FREQUENCY))
    finally:
        if renderer is not None:
            renderer.close()
        if recorder.active:
            recorder.finish_episode("INTERRUPTED", "runner_closed")

    if payload is None:
        raise RuntimeError("BT Episode ended without a Recorder payload")
    row = _comparison_row(payload, BT_DECISION_FREQUENCY, decisions)
    if collect_diagnostics:
        simulation_time = float(world.simulation_time)
        row.update(
            {
                "bt_tick_count": int(payload["bt_tick_count"]),
                "bt_transition_count": int(payload["bt_transition_count"]),
                "ppo_decision_count": controller.ppo_decision_count,
                "ppo_active_time": round(controller.ppo_active_time, 6),
                "ppo_active_ratio": (
                    controller.ppo_active_time / simulation_time
                    if simulation_time > 0.0
                    else 0.0
                ),
                "boundary_recovery_activation_count": (
                    controller.boundary_recovery_activation_count
                ),
                "search_activation_count": controller.search_activation_count,
                "ppo_preemption_count": controller.ppo_preemption_count,
            }
        )
    return row, initial_state


def run_ppo_episode(
    scenario: str,
    seed: int,
    output_dir: Path,
    model: PPO,
    render_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按 deterministic PPO 和固定 action-repeat 6 运行一个 Episode。"""
    recorder = ExperimentRecorder(output_dir)
    env = AgentGymEnv(
        scenario=scenario,
        render_mode=render_mode,
        recorder=recorder,
        recorder_controller="ppo",
        action_repeat=PPO_ACTION_REPEAT,
    )
    decisions = 0
    payload: dict[str, Any] | None = None
    try:
        observation, _ = env.reset(seed=seed)
        initial_state = capture_initial_state(env.world)
        while payload is None:
            if render_mode == "human" and _pygame_window_closed():
                payload = recorder.finish_episode("INTERRUPTED", "window_closed")
                break
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            decisions += 1
            if terminated or truncated:
                payload = env.last_episode_payload
    finally:
        env.close()

    if payload is None:
        raise RuntimeError("PPO Episode ended without a Recorder payload")
    return _comparison_row(payload, PPO_DECISION_FREQUENCY, decisions), initial_state


def run_hybrid_policy_episode(
    scenario: str,
    seed: int,
    output_dir: Path,
    model: Any,
    render_mode: str | None = None,
    recorder_controller: str = "hybrid_trained_ppo",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """用外部 deterministic Policy 驱动 Hybrid 的 PPO ownership 区间。"""
    recorder = ExperimentRecorder(output_dir)
    env = HybridPPOEnv(
        scenarios=(scenario,),
        render_mode=render_mode,
        recorder=recorder,
        recorder_controller=recorder_controller,
    )
    decisions = 0
    payload: dict[str, Any] | None = None
    last_info: dict[str, Any] = {}
    try:
        observation, last_info = env.reset(seed=seed)
        initial_state = capture_initial_state(env.world)
        while payload is None:
            if render_mode == "human" and _pygame_window_closed():
                payload = recorder.finish_episode("INTERRUPTED", "window_closed")
                break
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, last_info = env.step(action)
            decisions += 1
            if terminated or truncated:
                payload = env.last_episode_payload
    finally:
        env.close()

    if payload is None:
        raise RuntimeError("Hybrid PPO Episode ended without a Recorder payload")
    row = _comparison_row(payload, PPO_DECISION_FREQUENCY, decisions)
    row.update(
        {
            "bt_tick_count": int(payload["bt_tick_count"]),
            "bt_transition_count": int(payload["bt_transition_count"]),
            "ppo_decision_count": int(last_info["ppo_decision_count"]),
            "ppo_active_time": float(last_info["ppo_active_time"]),
            "ppo_active_ratio": float(last_info["ppo_active_ratio"]),
            "boundary_recovery_activation_count": int(
                last_info["boundary_recovery_activation_count"]
            ),
            "search_activation_count": int(
                last_info["search_activation_count"]
            ),
            "ppo_preemption_count": int(last_info["ppo_preemption_count"]),
            "ppo_reentry_count": int(last_info["ppo_reentry_count"]),
            "observation_before_preemption": last_info[
                "observation_before_preemption"
            ],
            "observation_after_reentry": last_info[
                "observation_after_reentry"
            ],
        }
    )
    return row, initial_state
