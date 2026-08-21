"""以 Pygame human 模式观察 R0.4 Scenario Distribution。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pygame

from autonomy_lab.bt.controller import BehaviorTreeController, PANEL_WIDTH
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.environment import Environment
from autonomy_lab.rendering.renderer import PygameRenderer
from autonomy_lab.scenarios.scenario_distribution import (
    RESEARCH_FAMILIES,
    ScenarioDistribution,
)


SIMULATION_DT = 1.0 / 60.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析需要观察的 family 与可复现 seed。"""
    parser = argparse.ArgumentParser(
        description="Visualize an R0.4 sampled scenario with the Research BT."
    )
    parser.add_argument(
        "--family",
        choices=sorted(RESEARCH_FAMILIES),
        default="dynamic_hazard",
        help="research scenario family (default: dynamic_hazard)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="reproducible sample seed (default: 42)",
    )
    parser.add_argument(
        "--hazard-threshold",
        type=float,
        default=ConditionParameters.DEFAULTS["hazard_threshold"],
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=ConditionParameters.DEFAULTS["boundary_threshold"],
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=ConditionParameters.DEFAULTS["goal_threshold"],
    )
    return parser.parse_args(argv)


def build_demo(
    family: str,
    seed: int,
    parameters: ConditionParameters | None = None,
) -> tuple[Environment, BehaviorTreeController]:
    """采样一个 World，并连接固定的 R0.3 Parameterized Research BT。"""
    scene = ScenarioDistribution(family).sample(seed)
    world = Environment(scene)
    controller = BehaviorTreeController(
        world,
        bt_config="condition_research",
        condition_parameters=parameters,
    )
    return world, controller


def run_demo(
    family: str,
    seed: int,
    parameters: ConditionParameters | None = None,
) -> int:
    """运行 60 Hz 可视化循环，返回实际绘制帧数便于 smoke test。"""
    world, controller = build_demo(family, seed, parameters=parameters)
    renderer = PygameRenderer(world, panel_width=PANEL_WIDTH, font_size=22)
    frames = 0
    running = True
    print(
        f"R0.4 demo: family={family}, seed={seed}, "
        f"hazards={world.scenario_metadata['hazard_count']}"
    )
    print(f"Condition parameters: {controller.condition_parameters.get_values()}")
    print("Close the window to exit; press R to replay the same sample.")
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    world.reset(seed=seed)
                    controller.reset()
            if not running:
                break

            # 仿真时间始终使用固定 1/60 s；renderer.pace() 只限制观看速度。
            turn, throttle = controller.tick(SIMULATION_DT)
            world.step(
                {"turn": turn, "throttle": throttle},
                SIMULATION_DT,
            )
            renderer.render(world, controller, "condition-research")
            renderer.pace(60)
            frames += 1
    finally:
        renderer.close()
    return frames


def main() -> None:
    args = parse_args()
    parameters = ConditionParameters(
        hazard_threshold=args.hazard_threshold,
        boundary_threshold=args.boundary_threshold,
        goal_threshold=args.goal_threshold,
    )
    run_demo(args.family, args.seed, parameters=parameters)


if __name__ == "__main__":
    main()
