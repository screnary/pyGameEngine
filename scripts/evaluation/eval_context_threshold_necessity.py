"""R0.12：用 paired dynamic-risk contexts 检验固定 Hazard threshold 必要性。

本脚本是纯 evaluation 入口。两个静态 context 复用同一 seed 生成的完整场景，
只改变动态 Hazard speed；BT、Action、Condition、Perception 和默认参数均不变。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution
from scripts.evaluation.eval_action_competence import EpisodeMetrics, SIMULATION_DT
from scripts.evaluation.eval_condition_threshold_sensitivity import SwitchingDiagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "experiments" / "analysis" / "r012_context_threshold_necessity.json"
)
DEFAULT_SUMMARY_CSV_PATH = (
    PROJECT_ROOT / "experiments" / "analysis" / "r012_context_threshold_necessity.csv"
)
DEFAULT_EPISODE_CSV_PATH = (
    PROJECT_ROOT / "experiments" / "analysis" / "r012_context_threshold_episodes.csv"
)

CONTEXT_SPEEDS = {
    "low_risk": 36.0,
    "high_dynamic_risk": 180.0,
}
CONTEXTS = tuple(CONTEXT_SPEEDS)
THRESHOLDS = (20.0, 30.0, 40.0, 45.0, 60.0, 75.0, 90.0)
BALANCED_EXPOSURE_WEIGHT = 1.0


def hazard_proximity_exposure(
    clearance: float | None, hazard_range: float
) -> float:
    """按 calibrated range 把当前最近 Hazard clearance 映射到 [0, 1]。"""
    if hazard_range <= 0.0:
        raise ValueError("hazard_range must be positive")
    if clearance is None or clearance >= hazard_range:
        return 0.0
    normalized = max(float(clearance), 0.0) / hazard_range
    return (1.0 - normalized) ** 2


def build_context_scene(seed: int, context: str) -> dict:
    """创建 paired scene；context 之间只覆盖动态 Hazard 的速度。"""
    try:
        speed = CONTEXT_SPEEDS[context]
    except KeyError as error:
        choices = ", ".join(CONTEXTS)
        raise ValueError(f"unknown context '{context}'; use {choices}") from error
    scene = ScenarioDistribution("dynamic_hazard").sample(int(seed))
    for hazard in scene["dynamic_hazards"]:
        hazard["speed"] = speed
    scene["name"] = f"R0.12 {context}"
    scene["research_metadata"].update(
        {
            "family": "r012_paired_dynamic_risk",
            "operating_context": context,
            "dynamic_hazard_speed": speed,
        }
    )
    return scene


def _build_context_shift_scene(seed: int) -> dict:
    """仅在静态 context 已出现 crossing 后创建 Low→High→Low scene。"""
    scene = build_context_scene(seed, "low_risk")
    scene["name"] = "R0.12 Low-High-Low Context Shift"
    scene["context_schedule"] = [
        {
            "name": "low_risk_1",
            "start_time": 0.0,
            "hazard_speed_scale": 1.0,
            "noise_level": 0.0,
        },
        {
            "name": "high_dynamic_risk",
            "start_time": 4.0,
            "hazard_speed_scale": 5.0,
            "noise_level": 0.0,
        },
        {
            "name": "low_risk_2",
            "start_time": 8.0,
            "hazard_speed_scale": 1.0,
            "noise_level": 0.0,
        },
    ]
    scene["research_metadata"].update(
        {
            "operating_context": "low_high_low",
            "context_schedule": (
                "low_risk_1",
                "high_dynamic_risk",
                "low_risk_2",
            ),
        }
    )
    return scene


class ExposureDiagnostics:
    """累计 evaluation-only proximity exposure 和可选 phase 安全指标。"""

    def __init__(self, hazard_range: float) -> None:
        self.hazard_range = hazard_range
        self.exposure_sum = 0.0
        self.steps = 0
        self._collision_active = False
        self._phases: dict[str, dict[str, float | int]] = {}

    def update(
        self,
        world: Environment,
        *,
        phase: str,
        active_action: str,
        goal_distance_before_step: float,
    ) -> None:
        clearance = world.perception.snapshot.hazard.nearest_clearance
        exposure = hazard_proximity_exposure(clearance, self.hazard_range)
        collision_event = bool(world.collision_this_step) and not self._collision_active
        goal_distance_after = world.agent.position.distance_to(world.target)
        self.steps += 1
        self.exposure_sum += exposure

        row = self._phases.setdefault(
            phase,
            {
                "simulation_steps": 0,
                "exposure_sum": 0.0,
                "minimum_clearance": math.inf,
                "collision_count": 0,
                "avoid_active_steps": 0,
                "move_to_goal_active_steps": 0,
                "initial_goal_distance": goal_distance_before_step,
                "final_goal_distance": goal_distance_before_step,
            },
        )
        row["simulation_steps"] = int(row["simulation_steps"]) + 1
        row["exposure_sum"] = float(row["exposure_sum"]) + exposure
        if clearance is not None:
            row["minimum_clearance"] = min(
                float(row["minimum_clearance"]), float(clearance)
            )
        if collision_event:
            row["collision_count"] = int(row["collision_count"]) + 1
        if active_action == "Avoid Hazard":
            row["avoid_active_steps"] = int(row["avoid_active_steps"]) + 1
        if active_action == "Move To Goal":
            row["move_to_goal_active_steps"] = (
                int(row["move_to_goal_active_steps"]) + 1
            )
        row["final_goal_distance"] = goal_distance_after
        self._collision_active = bool(world.collision_this_step)

    @property
    def mean_exposure(self) -> float:
        return self.exposure_sum / self.steps if self.steps else 0.0

    def phase_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for phase, raw in self._phases.items():
            steps = int(raw["simulation_steps"])
            minimum = float(raw["minimum_clearance"])
            initial = float(raw["initial_goal_distance"])
            final = float(raw["final_goal_distance"])
            rows.append(
                {
                    "phase": phase,
                    "simulation_steps": steps,
                    "hazard_exposure": round(
                        float(raw["exposure_sum"]) / steps, 6
                    ),
                    "minimum_clearance": (
                        None if math.isinf(minimum) else round(minimum, 3)
                    ),
                    "collision_count": int(raw["collision_count"]),
                    "avoid_active_ratio": round(
                        int(raw["avoid_active_steps"]) / steps, 6
                    ),
                    "move_to_goal_active_ratio": round(
                        int(raw["move_to_goal_active_steps"]) / steps, 6
                    ),
                    "goal_distance_change": round(initial - final, 3),
                }
            )
        return rows


def _run_episode(
    scene: dict,
    context: str,
    threshold: float,
    episode_horizon: float | None,
) -> dict[str, object]:
    world = Environment(scene)
    parameters = ConditionParameters()
    parameters.set("hazard_threshold", threshold)
    controller = BehaviorTreeController(
        world,
        bt_config="condition_research",
        condition_parameters=parameters,
    )
    metrics = EpisodeMetrics(world)
    switching = SwitchingDiagnostics()
    hazard_range = float(world.scene_config["sensor"]["hazard_range"])
    exposure = ExposureDiagnostics(hazard_range)
    horizon = (
        float(episode_horizon)
        if episode_horizon is not None
        else float(world.scene_config["experiment"]["max_episode_time"])
    )

    while metrics.elapsed_time + 1e-12 < horizon and not world.target_reached:
        phase = world.current_context_phase
        distance = world.agent.position.distance_to(world.target)
        turn, throttle = controller.tick(SIMULATION_DT)
        world.step(
            {"turn": float(turn), "throttle": float(throttle)}, SIMULATION_DT
        )
        metrics.update(world, SIMULATION_DT, float(turn))
        switching.update(
            controller,
            world,
            phase=phase,
            goal_distance_before_step=distance,
        )
        exposure.update(
            world,
            phase=phase,
            active_action=controller.active_behavior,
            goal_distance_before_step=distance,
        )

    success = bool(world.target_reached)
    common = metrics.common_row()
    return {
        "context": context,
        "threshold": float(threshold),
        "seed": int(world.seed),
        "dynamic_hazard_speed": float(
            scene["dynamic_hazards"][0]["speed"]
        ),
        "boundary_threshold": parameters.boundary_threshold,
        "goal_threshold": parameters.goal_threshold,
        "hazard_range": hazard_range,
        "success": success,
        "timeout": not success,
        "termination_reason": "target_reached" if success else "timeout",
        "collision_episode": int(common["collision_count"]) > 0,
        "hazard_exposure": round(exposure.mean_exposure, 6),
        **common,
        **switching.episode_fields(),
        "phase_safety_diagnostics": exposure.phase_rows(),
    }


def _mean(rows: Sequence[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(fmean(values), 6) if values else None


def _summary_row(
    rows: Sequence[dict[str, object]], context: str, threshold: float
) -> dict[str, object]:
    episodes = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision_episode"]) for row in rows)
    success_rate = successes / episodes
    collision_rate = collisions / episodes
    exposure = float(_mean(rows, "hazard_exposure") or 0.0)
    return {
        "scope": "context",
        "context": context,
        "phase": "all",
        "threshold": threshold,
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(success_rate, 6),
        "timeouts": episodes - successes,
        "timeout_rate": round(1.0 - success_rate, 6),
        "collision_episodes": collisions,
        "collision_episode_rate": round(collision_rate, 6),
        "collision_count": sum(int(row["collision_count"]) for row in rows),
        "mean_hazard_exposure": round(exposure, 6),
        "mean_minimum_clearance": _mean(rows, "minimum_clearance"),
        "mean_elapsed_time": _mean(rows, "elapsed_time"),
        "mean_path_length": _mean(rows, "path_length"),
        "mean_avoid_active_ratio": _mean(rows, "avoid_active_ratio"),
        "mean_move_to_goal_active_ratio": _mean(
            rows, "move_to_goal_active_ratio"
        ),
        "mean_boundary_active_ratio": _mean(rows, "boundary_active_ratio"),
        "mean_branch_switch_count": _mean(rows, "branch_switch_count"),
        "mean_longest_avoid_duration": _mean(rows, "longest_avoid_duration"),
        "balanced_score": round(
            success_rate
            - collision_rate
            - BALANCED_EXPOSURE_WEIGHT * exposure,
            6,
        ),
    }


def _preference_rows(
    summary_rows: Sequence[dict[str, object]], contexts: Sequence[str]
) -> list[dict[str, object]]:
    preferences: list[dict[str, object]] = []
    for context in contexts:
        rows = [row for row in summary_rows if row["context"] == context]
        best_success = max(
            rows,
            key=lambda row: (
                float(row["success_rate"]),
                -float(row["collision_episode_rate"]),
                -float(row["mean_hazard_exposure"]),
            ),
        )
        best_safety = min(
            rows,
            key=lambda row: (
                float(row["collision_episode_rate"]),
                float(row["mean_hazard_exposure"]),
                -float(row["mean_minimum_clearance"] or -math.inf),
            ),
        )
        best_balanced = max(
            rows,
            key=lambda row: (
                float(row["balanced_score"]),
                float(row["success_rate"]),
                -float(row["mean_elapsed_time"]),
            ),
        )
        preferences.append(
            {
                "context": context,
                "best_success_threshold": best_success["threshold"],
                "best_safety_threshold": best_safety["threshold"],
                "preferred_balanced_threshold": best_balanced["threshold"],
                "preferred_balanced_score": best_balanced["balanced_score"],
            }
        )
    return preferences


def _score_map(
    summary_rows: Sequence[dict[str, object]], context: str
) -> dict[float, float]:
    return {
        float(row["threshold"]): float(row["balanced_score"])
        for row in summary_rows
        if row["context"] == context
    }


def _exposure_crossing(
    summary_rows: Sequence[dict[str, object]], contexts: Sequence[str]
) -> bool:
    if len(contexts) != 2:
        return False
    by_context = {
        context: {
            float(row["threshold"]): float(row["mean_hazard_exposure"])
            for row in summary_rows
            if row["context"] == context
        }
        for context in contexts
    }
    thresholds = sorted(by_context[contexts[0]])
    for index, first in enumerate(thresholds):
        for second in thresholds[index + 1 :]:
            low_difference = (
                by_context[contexts[0]][first] - by_context[contexts[0]][second]
            )
            high_difference = (
                by_context[contexts[1]][first] - by_context[contexts[1]][second]
            )
            if (
                low_difference * high_difference < 0.0
                and abs(low_difference) >= 0.01
                and abs(high_difference) >= 0.01
            ):
                return True
    return False


def _attribution(
    summary_rows: Sequence[dict[str, object]],
    preferences: Sequence[dict[str, object]],
    contexts: Sequence[str],
) -> dict[str, object]:
    preferred = {
        str(row["context"]): float(row["preferred_balanced_threshold"])
        for row in preferences
    }
    balanced_crossing = False
    if len(contexts) == 2 and preferred[contexts[0]] != preferred[contexts[1]]:
        first_scores = _score_map(summary_rows, contexts[0])
        second_scores = _score_map(summary_rows, contexts[1])
        first_choice = preferred[contexts[0]]
        second_choice = preferred[contexts[1]]
        balanced_crossing = (
            first_scores[first_choice] - first_scores[second_choice] >= 0.02
            and second_scores[second_choice] - second_scores[first_choice] >= 0.02
        )
    exposure_crossing = _exposure_crossing(summary_rows, contexts)
    safety_preferences = {
        float(row["best_safety_threshold"]) for row in preferences
    }
    safety_frontier_context_dependent = (
        len(safety_preferences) > 1 and exposure_crossing
    )
    if balanced_crossing:
        case = "Case A — Context Dependence Supported"
    elif safety_frontier_context_dependent:
        case = "Case C — Safety Frontier Reveals Context Dependence"
    else:
        case = "Case B — Static Threshold Dominates"
    h3_supported = case != "Case B — Static Threshold Dominates"
    dominant = (
        next(iter(set(preferred.values())))
        if len(set(preferred.values())) == 1 and not exposure_crossing
        else None
    )
    return {
        "case": case,
        "h3_context_dependence_supported": h3_supported,
        "balanced_threshold_crossing": balanced_crossing,
        "safety_exposure_crossing": exposure_crossing,
        "dominant_static_threshold": dominant,
        "execute_context_shift": h3_supported,
        "balanced_score": (
            "success_rate - collision_episode_rate - 1.0 * hazard_exposure"
        ),
        "crossing_rule": (
            "context preferences must differ and each preferred threshold must "
            "improve its own balanced score by at least 0.02 over the other choice"
        ),
    }


def _phase_summary(
    episodes: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[float, str], list[dict[str, object]]] = {}
    for episode in episodes:
        for phase in episode["phase_safety_diagnostics"]:
            groups.setdefault(
                (float(episode["threshold"]), str(phase["phase"])), []
            ).append(phase)
    rows: list[dict[str, object]] = []
    for (threshold, phase), group in sorted(groups.items()):
        collision_episodes = sum(
            int(row["collision_count"]) > 0 for row in group
        )
        rows.append(
            {
                "scope": "context_shift_phase",
                "context": "low_high_low",
                "phase": phase,
                "threshold": threshold,
                "episodes": len(group),
                "collision_episodes": collision_episodes,
                "collision_episode_rate": round(
                    collision_episodes / len(group), 6
                ),
                "collision_count": sum(int(row["collision_count"]) for row in group),
                "mean_hazard_exposure": _mean(group, "hazard_exposure"),
                "mean_minimum_clearance": _mean(group, "minimum_clearance"),
                "mean_avoid_active_ratio": _mean(group, "avoid_active_ratio"),
                "mean_move_to_goal_active_ratio": _mean(
                    group, "move_to_goal_active_ratio"
                ),
                "mean_goal_distance_change": _mean(group, "goal_distance_change"),
            }
        )
    return rows


def evaluate_context_threshold_necessity(
    *,
    contexts: Iterable[str] = CONTEXTS,
    thresholds: Iterable[float] = THRESHOLDS,
    seeds: Iterable[int] = range(1001, 1051),
    episode_horizon: float | None = None,
) -> dict[str, object]:
    """先运行 paired static contexts，有 crossing 时再自动执行 Low→High→Low。"""
    context_tuple = tuple(contexts)
    threshold_tuple = tuple(float(value) for value in thresholds)
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not context_tuple or not threshold_tuple or not seed_tuple:
        raise ValueError("contexts, thresholds, and seeds must not be empty")
    validator = ConditionParameters()
    for threshold in threshold_tuple:
        validator.set("hazard_threshold", threshold)

    episodes = [
        _run_episode(
            build_context_scene(seed, context),
            context,
            threshold,
            episode_horizon,
        )
        for context in context_tuple
        for threshold in threshold_tuple
        for seed in seed_tuple
    ]
    summary_rows = [
        _summary_row(
            [
                row
                for row in episodes
                if row["context"] == context
                and float(row["threshold"]) == threshold
            ],
            context,
            threshold,
        )
        for context in context_tuple
        for threshold in threshold_tuple
    ]
    preferences = _preference_rows(summary_rows, context_tuple)
    attribution = _attribution(summary_rows, preferences, context_tuple)

    shift_episodes: list[dict[str, object]] = []
    # H3 静态 evidence 是硬 gate；没有 evidence 时不会偷偷增加第三组场景。
    if attribution["execute_context_shift"]:
        shift_episodes = [
            _run_episode(
                _build_context_shift_scene(seed),
                "low_high_low",
                threshold,
                episode_horizon,
            )
            for threshold in threshold_tuple
            for seed in seed_tuple
        ]

    return {
        "milestone": "R0.12",
        "simulation_dt": SIMULATION_DT,
        "bt_config": "condition_research",
        "thresholds": list(threshold_tuple),
        "contexts": {
            context: {"dynamic_hazard_speed": CONTEXT_SPEEDS[context]}
            for context in context_tuple
        },
        "paired_base_family": "dynamic_hazard",
        "balanced_exposure_weight": BALANCED_EXPOSURE_WEIGHT,
        "default_parameters": ConditionParameters().get_values(),
        "fixed_parameters": {
            "boundary_threshold": ConditionParameters().boundary_threshold,
            "goal_threshold": ConditionParameters().goal_threshold,
            "hazard_range": 300.0,
        },
        "seeds": list(seed_tuple),
        "episodes": episodes,
        "context_shift_episodes": shift_episodes,
        "summary": {
            "by_context_and_threshold": summary_rows,
            "context_preferences": preferences,
            "attribution": attribution,
            "context_shift_by_threshold_and_phase": _phase_summary(
                shift_episodes
            ),
        },
    }


SUMMARY_CSV_FIELDS = (
    "scope",
    "context",
    "phase",
    "threshold",
    "episodes",
    "successes",
    "success_rate",
    "timeouts",
    "timeout_rate",
    "collision_episodes",
    "collision_episode_rate",
    "collision_count",
    "mean_hazard_exposure",
    "mean_minimum_clearance",
    "mean_elapsed_time",
    "mean_path_length",
    "mean_avoid_active_ratio",
    "mean_move_to_goal_active_ratio",
    "mean_boundary_active_ratio",
    "mean_branch_switch_count",
    "mean_longest_avoid_duration",
    "mean_goal_distance_change",
    "balanced_score",
)

EPISODE_CSV_FIELDS = (
    "context",
    "threshold",
    "seed",
    "dynamic_hazard_speed",
    "success",
    "timeout",
    "termination_reason",
    "collision_count",
    "collision_episode",
    "minimum_clearance",
    "hazard_exposure",
    "elapsed_time",
    "path_length",
    "avoid_active_ratio",
    "move_to_goal_active_ratio",
    "boundary_active_ratio",
    "branch_switch_count",
    "longest_avoid_duration",
    "avoid_maneuver_count",
    "simulation_steps",
)


def write_results(
    payload: dict[str, object],
    json_path: Path = DEFAULT_JSON_PATH,
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV_PATH,
    episode_csv_path: Path = DEFAULT_EPISODE_CSV_PATH,
) -> tuple[Path, Path, Path]:
    """保存完整 JSON、context summary CSV 与 episode-level CSV。"""
    paths = tuple(Path(path) for path in (json_path, summary_csv_path, episode_csv_path))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_rows = list(payload["summary"]["by_context_and_threshold"]) + list(
        payload["summary"]["context_shift_by_threshold_and_phase"]
    )
    with paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_CSV_FIELDS})
    all_episodes = list(payload["episodes"]) + list(
        payload["context_shift_episodes"]
    )
    with paths[2].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        for row in all_episodes:
            writer.writerow({field: row.get(field) for field in EPISODE_CSV_FIELDS})
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R0.12 paired context fixed-threshold necessity study."
    )
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--seed-end", type=int, default=1050)
    parser.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    parser.add_argument("--episode-horizon", type=float)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument(
        "--summary-csv-output", type=Path, default=DEFAULT_SUMMARY_CSV_PATH
    )
    parser.add_argument(
        "--episode-csv-output", type=Path, default=DEFAULT_EPISODE_CSV_PATH
    )
    return parser.parse_args(argv)


def _print_summary(payload: dict[str, object]) -> None:
    print("R0.12 context-dependent threshold necessity")
    for row in payload["summary"]["by_context_and_threshold"]:
        print(
            f"  {row['context']:<18} theta={row['threshold']:>4.0f}: "
            f"success={row['success_rate']:.1%}, "
            f"collision={row['collision_episode_rate']:.1%}, "
            f"exposure={row['mean_hazard_exposure']:.4f}, "
            f"avoid={row['mean_avoid_active_ratio']:.1%}"
        )
    print(payload["summary"]["attribution"]["case"])


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be greater than or equal to seed-start")
    if args.episode_horizon is not None and args.episode_horizon <= 0.0:
        raise ValueError("episode-horizon must be positive")
    payload = evaluate_context_threshold_necessity(
        thresholds=args.thresholds,
        seeds=range(args.seed_start, args.seed_end + 1),
        episode_horizon=args.episode_horizon,
    )
    paths = write_results(
        payload,
        args.json_output,
        args.summary_csv_output,
        args.episode_csv_output,
    )
    _print_summary(payload)
    print(f"Context shift executed: {bool(payload['context_shift_episodes'])}")
    print(f"JSON:        {paths[0]}")
    print(f"Summary CSV: {paths[1]}")
    print(f"Episode CSV: {paths[2]}")


if __name__ == "__main__":
    main()
