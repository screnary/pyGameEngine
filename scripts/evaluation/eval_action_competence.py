"""R0.9：验证固定 Research BT Action 的基础能力，不修改任何行为算法。

隔离用例会直接 tick 项目中的真实 Action 节点，但仍通过同一份
``SemanticPerception -> AgentCommand -> Environment.step`` 链路推进 World。
整树用例则加载 ``condition_research.json``，用于区分 Action 局部能力与
Condition/BT 协同后的端到端表现。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import py_trees

from autonomy_lab.bt.behaviors import (
    AvoidObstacle,
    HazardRisk,
    MoveToTarget,
    SafeBoundaryRecovery,
    Stop,
)
from autonomy_lab.bt.context import BehaviorBuildContext
from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.agent import AgentCommand
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.config import (
    DEFAULT_RESEARCH_SENSOR_CONFIG,
    get_scene,
)
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = PROJECT_ROOT / "experiments" / "analysis" / "r09_action_competence.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "experiments" / "analysis" / "r09_action_competence.csv"
SIMULATION_DT = 1.0 / 60.0
REQUIRED_FAMILIES = (
    "static_random",
    "dense_hazard",
    "dynamic_hazard",
    "noisy_perception",
    "context_shift",
)
ISOLATED_ACTIONS = {
    "MoveToGoal",
    "AvoidHazard",
    "SafeBoundaryRecovery",
    "Stop",
}


@dataclass(frozen=True)
class MicroCase:
    """一个确定性的 Action 隔离用例。scene 仍使用正式 Environment schema。"""

    action: str
    case_id: str
    scene: dict
    horizon: float


class EpisodeMetrics:
    """按真实 World step 累积与 ExperimentRecorder 一致的核心指标。"""

    def __init__(self, world: Environment) -> None:
        self.elapsed_time = 0.0
        self.path_length = 0.0
        self.collision_count = 0
        self.minimum_clearance = math.inf
        self.steering_sign_changes = 0
        self.toward_hazard_steps = 0
        self._last_position = world.agent.position.copy()
        self._collision_active = False
        self._last_turn_sign = 0
        self._previous_clearance = self._clearance(world)
        self._sample_clearance(world)

    @staticmethod
    def _clearance(world: Environment) -> float | None:
        return world.perception.snapshot.hazard.nearest_clearance

    def _sample_clearance(self, world: Environment) -> None:
        clearance = self._clearance(world)
        if clearance is not None:
            self.minimum_clearance = min(self.minimum_clearance, clearance)

    def update(self, world: Environment, dt: float, turn: float) -> None:
        self.elapsed_time += dt
        self.path_length += self._last_position.distance_to(world.agent.position)
        self._last_position = world.agent.position.copy()

        colliding = bool(world.collision_this_step)
        if colliding and not self._collision_active:
            self.collision_count += 1
        self._collision_active = colliding

        turn_sign = 1 if turn > 0.05 else -1 if turn < -0.05 else 0
        if (
            turn_sign
            and self._last_turn_sign
            and turn_sign != self._last_turn_sign
        ):
            self.steering_sign_changes += 1
        if turn_sign:
            self._last_turn_sign = turn_sign

        clearance = self._clearance(world)
        if (
            clearance is not None
            and self._previous_clearance is not None
            and clearance < self._previous_clearance - 0.25
        ):
            self.toward_hazard_steps += 1
        self._previous_clearance = clearance
        self._sample_clearance(world)

    def common_row(self) -> dict[str, object]:
        minimum = (
            None
            if math.isinf(self.minimum_clearance)
            else round(self.minimum_clearance, 3)
        )
        return {
            "elapsed_time": round(self.elapsed_time, 6),
            "path_length": round(self.path_length, 3),
            "collision_count": self.collision_count,
            "minimum_clearance": minimum,
            "steering_sign_changes": self.steering_sign_changes,
            "toward_hazard_steps": self.toward_hazard_steps,
        }


def _micro_scene(
    *,
    agent_position: tuple[float, float],
    heading_degrees: float,
    target_position: tuple[float, float],
    obstacles: Sequence[tuple[int, int, int, int]] = (),
    initial_speed: float = 0.0,
) -> dict:
    """从正式 scene schema 派生只供评估使用的确定性微场景。"""
    scene = get_scene("rl_sanity")
    scene.update(
        {
            "name": "R0.9 Action Competence Micro Scenario",
            "world_size": (850, 600),
            "seed": 9009,
            "obstacles": list(obstacles),
            "target_information_mode": "perceived",
            "test_only": True,
        }
    )
    scene["agent"].update(
        {
            "position": agent_position,
            "heading_degrees": heading_degrees,
            "initial_speed": initial_speed,
            "radius": 16,
        }
    )
    scene["target"].update({"position": target_position, "radius": 18})
    scene["sensor"].update(DEFAULT_RESEARCH_SENSOR_CONFIG)
    scene["experiment"].update(
        {"max_episode_time": 12.0, "target_reached_distance": 30.0}
    )
    return scene


def _isolated_cases() -> tuple[MicroCase, ...]:
    """覆盖目标方位、Hazard 方位/尺度、四边/角落及停车的固定用例。"""
    move_specs = (
        ("goal_front", (425, 300), 0.0, (645, 300)),
        ("goal_left", (425, 300), 0.0, (425, 95)),
        ("goal_right", (425, 300), 0.0, (425, 535)),
        ("goal_rear_left", (425, 300), 15.0, (205, 125)),
        ("goal_rear_right", (425, 300), -15.0, (205, 475)),
    )
    cases = [
        MicroCase(
            "MoveToGoal",
            case_id,
            _micro_scene(
                agent_position=agent,
                heading_degrees=heading,
                target_position=target,
            ),
            10.0,
        )
        for case_id, agent, heading, target in move_specs
    ]

    avoid_specs = (
        ("hazard_ahead_60", (425, 300), 0.0, (501, 260, 50, 80)),
        ("hazard_front_left", (425, 300), 0.0, (490, 190, 60, 70)),
        ("hazard_front_right", (425, 300), 0.0, (490, 340, 60, 70)),
        ("hazard_close_small", (425, 300), 0.0, (475, 280, 35, 40)),
        ("hazard_wide", (425, 300), 0.0, (500, 230, 90, 140)),
        ("hazard_near_bottom_boundary", (425, 500), 0.0, (500, 455, 70, 90)),
    )
    cases.extend(
        MicroCase(
            "AvoidHazard",
            case_id,
            _micro_scene(
                agent_position=agent,
                heading_degrees=heading,
                target_position=(760, 300),
                obstacles=(obstacle,),
            ),
            5.0,
        )
        for case_id, agent, heading, obstacle in avoid_specs
    )

    boundary_specs = (
        ("boundary_left", (30, 300), 180.0, ()),
        ("boundary_right", (820, 300), 0.0, ()),
        ("boundary_top", (425, 30), -90.0, ()),
        ("boundary_bottom", (425, 570), 90.0, ()),
        ("boundary_top_left_corner", (32, 32), 225.0, ()),
        ("boundary_bottom_right_corner", (818, 568), 45.0, ()),
        (
            "boundary_right_with_hazard",
            (805, 300),
            0.0,
            ((665, 240, 95, 120),),
        ),
    )
    cases.extend(
        MicroCase(
            "SafeBoundaryRecovery",
            case_id,
            _micro_scene(
                agent_position=agent,
                heading_degrees=heading,
                target_position=(425, 300),
                obstacles=obstacles,
            ),
            6.0,
        )
        for case_id, agent, heading, obstacles in boundary_specs
    )
    cases.extend(
        (
            MicroCase(
                "Stop",
                "stop_stationary",
                _micro_scene(
                    agent_position=(425, 300),
                    heading_degrees=0.0,
                    target_position=(425, 300),
                ),
                0.5,
            ),
            MicroCase(
                "Stop",
                "stop_from_motion",
                _micro_scene(
                    agent_position=(425, 300),
                    heading_degrees=30.0,
                    target_position=(425, 300),
                    initial_speed=80.0,
                ),
                0.5,
            ),
        )
    )
    return tuple(cases)


def _context(world: Environment, command: AgentCommand) -> BehaviorBuildContext:
    return BehaviorBuildContext(
        perception=world.perception,
        command=command,
        behavior_config=world.scene_config["behavior_tree"],
        condition_parameters=ConditionParameters(),
    )


def _boundary_clearance(world: Environment) -> float:
    boundary = world.perception.snapshot.boundary
    return min(boundary.left, boundary.right, boundary.top, boundary.bottom)


def _failure_reason(reached: bool, collisions: int, fallback: str) -> str:
    if collisions:
        return "collision"
    return "" if reached else fallback


def _run_isolated_case(case: MicroCase) -> dict[str, object]:
    world = Environment(case.scene)
    command: AgentCommand = {"turn": 0.0, "throttle": 0.0}
    context = _context(world, command)
    metrics = EpisodeMetrics(world)
    initial_distance = world.agent.position.distance_to(world.target)
    initial_boundary = _boundary_clearance(world)
    action: py_trees.behaviour.Behaviour

    if case.action == "MoveToGoal":
        action = MoveToTarget(context, "Move To Goal")
    elif case.action == "AvoidHazard":
        condition = HazardRisk(context, "Hazard Risk?")
        context.nodes_by_name[condition.name] = condition
        condition.update()  # 仅把真实 nearest Hazard 交给 Action，不用于触发分支。
        action = AvoidObstacle(context, "Avoid Hazard", condition=condition.name)
    elif case.action == "SafeBoundaryRecovery":
        action = SafeBoundaryRecovery(context, "Safe Boundary Recovery")
    elif case.action == "Stop":
        action = Stop(context, "Stop")
    else:  # pragma: no cover - case inventory is local and closed
        raise ValueError(f"unknown isolated action: {case.action}")

    achieved = False
    while metrics.elapsed_time + 1e-12 < case.horizon:
        world.perception.observe()
        command["turn"] = 0.0
        command["throttle"] = 0.0
        if isinstance(action, AvoidObstacle):
            action.condition.threat = world.perception.snapshot.hazard.nearest_hazard
            action.dt = SIMULATION_DT
        action.tick_once()
        turn = float(command["turn"])
        world.step(command, SIMULATION_DT)
        metrics.update(world, SIMULATION_DT, turn)

        if case.action == "MoveToGoal" and world.target_reached:
            achieved = True
            break
        if case.action == "AvoidHazard":
            clearance = world.perception.snapshot.hazard.nearest_clearance
            if clearance is None or clearance >= 90.0:
                achieved = True
                break
        if case.action == "SafeBoundaryRecovery" and _boundary_clearance(world) >= 40.0:
            achieved = True
            break

    final_distance = world.agent.position.distance_to(world.target)
    if case.action == "Stop":
        achieved = metrics.path_length <= 1.0 and abs(world.agent.speed) <= 1e-9

    if case.action == "MoveToGoal":
        fallback = "goal_not_reached"
    elif case.action == "AvoidHazard":
        fallback = "hazard_risk_not_escaped"
    elif case.action == "SafeBoundaryRecovery":
        fallback = "safe_boundary_clearance_not_reached"
    else:
        fallback = "agent_moved_after_stop"
    success = achieved and metrics.collision_count == 0

    row: dict[str, object] = {
        "record_type": "isolated",
        "action": case.action,
        "scenario": "micro",
        "family": None,
        "case_id": case.case_id,
        "seed": int(case.scene["seed"]),
        "success": success,
        "failure_reason": _failure_reason(
            achieved, metrics.collision_count, fallback
        ),
        **metrics.common_row(),
        "initial_target_distance": round(initial_distance, 3),
        "final_target_distance": round(final_distance, 3),
        "initial_boundary_clearance": round(initial_boundary, 3),
        "final_boundary_clearance": round(_boundary_clearance(world), 3),
        "path_efficiency": (
            round(initial_distance / metrics.path_length, 4)
            if case.action == "MoveToGoal" and metrics.path_length > 0.0
            else None
        ),
    }
    return row


def run_isolated_suite() -> list[dict[str, object]]:
    """执行所有真实 Action 的确定性局部能力用例。"""
    return [_run_isolated_case(case) for case in _isolated_cases()]


def _run_research_bt_episode(
    family: str,
    seed: int,
    episode_horizon: float | None,
) -> dict[str, object]:
    world = Environment(ScenarioDistribution(family).sample(seed))
    controller = BehaviorTreeController(world, bt_config="condition_research")
    metrics = EpisodeMetrics(world)
    horizon = (
        float(episode_horizon)
        if episode_horizon is not None
        else float(world.scene_config["experiment"]["max_episode_time"])
    )
    initial_distance = world.agent.position.distance_to(world.target)

    while metrics.elapsed_time + 1e-12 < horizon and not world.target_reached:
        turn, throttle = controller.tick(SIMULATION_DT)
        world.step(
            {"turn": float(turn), "throttle": float(throttle)},
            SIMULATION_DT,
        )
        metrics.update(world, SIMULATION_DT, float(turn))

    success = bool(world.target_reached)
    return {
        "record_type": "end_to_end",
        "action": "ResearchBT",
        "scenario": family,
        "family": family,
        "case_id": None,
        "seed": int(seed),
        "success": success,
        "failure_reason": "" if success else "timeout",
        **metrics.common_row(),
        "initial_target_distance": round(initial_distance, 3),
        "final_target_distance": round(
            world.agent.position.distance_to(world.target), 3
        ),
        "initial_boundary_clearance": None,
        "final_boundary_clearance": round(_boundary_clearance(world), 3),
        "path_efficiency": (
            round(initial_distance / metrics.path_length, 4)
            if metrics.path_length > 0.0
            else None
        ),
        "bt_tick_count": controller.tick_count,
        "bt_transition_count": None,
    }


def run_end_to_end_suite(
    families: Iterable[str],
    seeds: Iterable[int],
    *,
    episode_horizon: float | None = None,
) -> list[dict[str, object]]:
    """用默认参数运行真实 ``condition_research`` BT。"""
    return [
        _run_research_bt_episode(family, int(seed), episode_horizon)
        for family in families
        for seed in seeds
    ]


def _mean(rows: Sequence[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(fmean(values), 4) if values else None


def _summarize(
    rows: Sequence[dict[str, object]], key: str
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return [
        {
            key: group,
            "cases": len(group_rows),
            "successes": sum(bool(row["success"]) for row in group_rows),
            "success_rate": round(
                sum(bool(row["success"]) for row in group_rows)
                / len(group_rows),
                4,
            ),
            "mean_elapsed_time": _mean(group_rows, "elapsed_time"),
            "mean_path_length": _mean(group_rows, "path_length"),
            "mean_collision_count": _mean(group_rows, "collision_count"),
            "mean_minimum_clearance": _mean(group_rows, "minimum_clearance"),
            "failure_reasons": sorted(
                {
                    str(row["failure_reason"])
                    for row in group_rows
                    if row["failure_reason"]
                }
            ),
        }
        for group, group_rows in sorted(groups.items())
    ]


def evaluate_action_competence(
    *,
    families: Iterable[str] = REQUIRED_FAMILIES,
    seeds: Iterable[int] = range(1001, 1051),
    include_isolated: bool = True,
    episode_horizon: float | None = None,
) -> dict[str, object]:
    """返回完整 R0.9 数据；该函数不写文件，便于测试与复现。"""
    family_tuple = tuple(families)
    seed_tuple = tuple(int(seed) for seed in seeds)
    isolated = run_isolated_suite() if include_isolated else []
    end_to_end = run_end_to_end_suite(
        family_tuple,
        seed_tuple,
        episode_horizon=episode_horizon,
    )
    return {
        "milestone": "R0.9",
        "simulation_dt": SIMULATION_DT,
        "bt_config": "condition_research",
        "condition_parameters": ConditionParameters().get_values(),
        "families": list(family_tuple),
        "seeds": list(seed_tuple),
        "isolated_results": isolated,
        "end_to_end_results": end_to_end,
        "summary": {
            "isolated_by_action": _summarize(isolated, "action"),
            "end_to_end_by_family": _summarize(end_to_end, "family"),
        },
    }


CSV_FIELDS = (
    "record_type",
    "action",
    "scenario",
    "family",
    "case_id",
    "seed",
    "success",
    "failure_reason",
    "elapsed_time",
    "path_length",
    "collision_count",
    "minimum_clearance",
    "steering_sign_changes",
    "toward_hazard_steps",
    "initial_target_distance",
    "final_target_distance",
    "initial_boundary_clearance",
    "final_boundary_clearance",
    "path_efficiency",
    "bt_tick_count",
    "bt_transition_count",
)


def write_results(
    payload: dict[str, object],
    json_path: Path = DEFAULT_JSON_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> tuple[Path, Path]:
    """写入独立 JSON 与扁平 CSV，不触碰历史 ExperimentRecorder 输出。"""
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = list(payload["isolated_results"]) + list(
        payload["end_to_end_results"]
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    return json_path, csv_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed Research BT Action competence (R0.9)."
    )
    parser.add_argument(
        "--seed-start", type=int, default=1001, help="inclusive seed start"
    )
    parser.add_argument(
        "--seed-end", type=int, default=1050, help="inclusive seed end"
    )
    parser.add_argument(
        "--json-output", type=Path, default=DEFAULT_JSON_PATH
    )
    parser.add_argument(
        "--csv-output", type=Path, default=DEFAULT_CSV_PATH
    )
    parser.add_argument(
        "--isolated-only",
        action="store_true",
        help="run only deterministic Action micro-scenarios",
    )
    return parser.parse_args(argv)


def _print_summary(payload: dict[str, object]) -> None:
    print("R0.9 isolated Action competence")
    for row in payload["summary"]["isolated_by_action"]:
        print(
            f"  {row['action']}: {row['successes']}/{row['cases']} "
            f"({row['success_rate']:.0%}), failures={row['failure_reasons']}"
        )
    print("R0.9 end-to-end Research BT")
    for row in payload["summary"]["end_to_end_by_family"]:
        print(
            f"  {row['family']}: {row['successes']}/{row['cases']} "
            f"({row['success_rate']:.0%}), collisions="
            f"{row['mean_collision_count']:.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be greater than or equal to seed-start")
    families = () if args.isolated_only else REQUIRED_FAMILIES
    seeds = range(args.seed_start, args.seed_end + 1)
    payload = evaluate_action_competence(families=families, seeds=seeds)
    json_path, csv_path = write_results(
        payload, args.json_output, args.csv_output
    )
    _print_summary(payload)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
