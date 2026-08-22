"""R0.11：仅改变运行时 Hazard threshold 的 switching sensitivity sweep。

评估器复用正式 ``ScenarioDistribution``、``Environment`` 和
``BehaviorTreeController``。它只观察真实 Runtime 每个 simulation tick 的节点
状态，不参与 BT 决策，也不修改默认 Condition 参数或任何 Action 实现。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import py_trees

from autonomy_lab.bt.controller import BehaviorTreeController
from autonomy_lab.bt.parameters import ConditionParameters
from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution
from scripts.evaluation.eval_action_competence import EpisodeMetrics, SIMULATION_DT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "experiments" / "analysis" / "r011_threshold_sensitivity.json"
)
DEFAULT_SUMMARY_CSV_PATH = (
    PROJECT_ROOT / "experiments" / "analysis" / "r011_threshold_sensitivity.csv"
)
DEFAULT_EPISODE_CSV_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "analysis"
    / "r011_threshold_sensitivity_episodes.csv"
)
RESEARCH_FAMILIES = (
    "static_random",
    "dense_hazard",
    "dynamic_hazard",
    "noisy_perception",
    "context_shift",
)
THRESHOLD_FACTORS = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)


def build_hazard_threshold_grid(
    parameters: ConditionParameters | None = None,
) -> tuple[float, ...]:
    """围绕冻结 default 建立 aggressive → conservative 的 7 点网格。"""
    store = parameters if parameters is not None else ConditionParameters()
    spec = store.spec("hazard_threshold")
    values: list[float] = []
    for factor in THRESHOLD_FACTORS:
        value = max(spec.min_value, min(spec.default * factor, spec.max_value))
        rounded = round(value, 6)
        if rounded not in values:
            values.append(rounded)
    return tuple(values)


class SwitchingDiagnostics:
    """按真实 BT/World tick 记录分支占用，不向 Controller 注入任何状态。"""

    def __init__(self) -> None:
        self.total_steps = 0
        self.hazard_risk_activation_count = 0
        self.avoid_active_steps = 0
        self.move_to_goal_active_steps = 0
        self.boundary_active_steps = 0
        self.branch_switch_count = 0
        self._previous_action: str | None = None
        self._hazard_active = False
        self._collision_active = False
        self._continuous_avoid_steps = 0
        self._longest_avoid_steps = 0
        self._phases: dict[str, dict[str, float | int]] = {}

    def update(
        self,
        controller: BehaviorTreeController,
        world: Environment,
        *,
        phase: str,
        goal_distance_before_step: float,
    ) -> None:
        """记录刚完成的一次 simulation step 及其对应的 BT decision。"""
        action = controller.active_behavior
        hazard_node = controller.nodes_by_name["Hazard Risk?"]
        hazard_active = hazard_node.status == py_trees.common.Status.SUCCESS
        collision_event = bool(world.collision_this_step) and not self._collision_active

        self.total_steps += 1
        if hazard_active and not self._hazard_active:
            self.hazard_risk_activation_count += 1
        if self._previous_action is not None and action != self._previous_action:
            self.branch_switch_count += 1

        if action == "Avoid Hazard":
            self.avoid_active_steps += 1
            self._continuous_avoid_steps += 1
            self._longest_avoid_steps = max(
                self._longest_avoid_steps, self._continuous_avoid_steps
            )
        else:
            self._continuous_avoid_steps = 0
        if action == "Move To Goal":
            self.move_to_goal_active_steps += 1
        if action == "Safe Boundary Recovery":
            self.boundary_active_steps += 1

        goal_distance_after_step = world.agent.position.distance_to(world.target)
        phase_row = self._phases.setdefault(
            phase,
            {
                "simulation_steps": 0,
                "hazard_risk_activation_count": 0,
                "avoid_active_steps": 0,
                "collision_count": 0,
                "initial_goal_distance": goal_distance_before_step,
                "final_goal_distance": goal_distance_before_step,
            },
        )
        phase_row["simulation_steps"] = int(phase_row["simulation_steps"]) + 1
        if hazard_active and not self._hazard_active:
            phase_row["hazard_risk_activation_count"] = (
                int(phase_row["hazard_risk_activation_count"]) + 1
            )
        if action == "Avoid Hazard":
            phase_row["avoid_active_steps"] = int(phase_row["avoid_active_steps"]) + 1
        if collision_event:
            phase_row["collision_count"] = int(phase_row["collision_count"]) + 1
        phase_row["final_goal_distance"] = goal_distance_after_step

        self._previous_action = action
        self._hazard_active = hazard_active
        self._collision_active = bool(world.collision_this_step)

    def episode_fields(self) -> dict[str, object]:
        steps = self.total_steps
        denominator = float(steps) if steps else 1.0
        return {
            "simulation_steps": steps,
            "hazard_risk_activation_count": self.hazard_risk_activation_count,
            "avoid_active_steps": self.avoid_active_steps,
            "avoid_active_time": round(self.avoid_active_steps * SIMULATION_DT, 6),
            "avoid_active_ratio": round(self.avoid_active_steps / denominator, 6),
            "move_to_goal_active_steps": self.move_to_goal_active_steps,
            "move_to_goal_active_ratio": round(
                self.move_to_goal_active_steps / denominator, 6
            ),
            "boundary_active_steps": self.boundary_active_steps,
            "boundary_active_ratio": round(
                self.boundary_active_steps / denominator, 6
            ),
            "branch_switch_count": self.branch_switch_count,
            "longest_avoid_duration": round(
                self._longest_avoid_steps * SIMULATION_DT, 6
            ),
            "phase_diagnostics": self.phase_rows(),
        }

    def phase_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for phase, raw in self._phases.items():
            steps = int(raw["simulation_steps"])
            initial = float(raw["initial_goal_distance"])
            final = float(raw["final_goal_distance"])
            rows.append(
                {
                    "phase": phase,
                    "simulation_steps": steps,
                    "hazard_risk_activation_count": int(
                        raw["hazard_risk_activation_count"]
                    ),
                    "avoid_active_steps": int(raw["avoid_active_steps"]),
                    "avoid_active_ratio": round(
                        int(raw["avoid_active_steps"]) / steps, 6
                    ),
                    "collision_count": int(raw["collision_count"]),
                    "initial_goal_distance": round(initial, 3),
                    "final_goal_distance": round(final, 3),
                    "goal_distance_change": round(initial - final, 3),
                }
            )
        return rows


def _run_episode(
    family: str,
    seed: int,
    threshold: float,
    episode_horizon: float | None,
) -> dict[str, object]:
    world = Environment(ScenarioDistribution(family).sample(seed))
    parameters = ConditionParameters()
    # 这是 sweep 唯一允许的运行时变量；另两个 threshold 保持冻结 default。
    parameters.set("hazard_threshold", threshold)
    controller = BehaviorTreeController(
        world,
        bt_config="condition_research",
        condition_parameters=parameters,
    )
    metrics = EpisodeMetrics(world)
    switching = SwitchingDiagnostics()
    horizon = (
        float(episode_horizon)
        if episode_horizon is not None
        else float(world.scene_config["experiment"]["max_episode_time"])
    )

    while metrics.elapsed_time + 1e-12 < horizon and not world.target_reached:
        phase = world.current_context_phase
        goal_distance = world.agent.position.distance_to(world.target)
        turn, throttle = controller.tick(SIMULATION_DT)
        world.step(
            {"turn": float(turn), "throttle": float(throttle)},
            SIMULATION_DT,
        )
        metrics.update(world, SIMULATION_DT, float(turn))
        switching.update(
            controller,
            world,
            phase=phase,
            goal_distance_before_step=goal_distance,
        )

    success = bool(world.target_reached)
    common = metrics.common_row()
    return {
        "threshold": float(threshold),
        "boundary_threshold": parameters.get("boundary_threshold"),
        "goal_threshold": parameters.get("goal_threshold"),
        "family": family,
        "seed": int(seed),
        "success": success,
        "timeout": not success,
        "termination_reason": "target_reached" if success else "timeout",
        "collision_episode": int(common["collision_count"]) > 0,
        **common,
        **switching.episode_fields(),
    }


def _mean(rows: Sequence[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(fmean(values), 6) if values else None


def _aggregate(
    rows: Sequence[dict[str, object]], *, scope: str, family: str = "all"
) -> dict[str, object]:
    episodes = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)
    collision_episodes = sum(bool(row["collision_episode"]) for row in rows)
    return {
        "scope": scope,
        "threshold": float(rows[0]["threshold"]),
        "family": family,
        "phase": "all",
        "episodes": episodes,
        "successes": successes,
        "success_rate": round(successes / episodes, 6),
        "timeouts": timeouts,
        "timeout_rate": round(timeouts / episodes, 6),
        "collision_episodes": collision_episodes,
        "collision_episode_rate": round(collision_episodes / episodes, 6),
        "collision_count": sum(int(row["collision_count"]) for row in rows),
        "mean_elapsed_time": _mean(rows, "elapsed_time"),
        "mean_path_length": _mean(rows, "path_length"),
        "mean_minimum_clearance": _mean(rows, "minimum_clearance"),
        "mean_hazard_risk_activation_count": _mean(
            rows, "hazard_risk_activation_count"
        ),
        "mean_avoid_active_ratio": _mean(rows, "avoid_active_ratio"),
        "mean_move_to_goal_active_ratio": _mean(
            rows, "move_to_goal_active_ratio"
        ),
        "mean_boundary_active_ratio": _mean(rows, "boundary_active_ratio"),
        "mean_branch_switch_count": _mean(rows, "branch_switch_count"),
        "mean_longest_avoid_duration": _mean(rows, "longest_avoid_duration"),
    }


def _timeout_aggregate(
    rows: Sequence[dict[str, object]], threshold: float, family: str
) -> dict[str, object]:
    timeout_rows = [row for row in rows if bool(row["timeout"])]
    return {
        "threshold": threshold,
        "family": family,
        "timeout_episodes": len(timeout_rows),
        "mean_avoid_active_ratio": _mean(timeout_rows, "avoid_active_ratio"),
        "mean_move_to_goal_active_ratio": _mean(
            timeout_rows, "move_to_goal_active_ratio"
        ),
        "mean_branch_switch_count": _mean(timeout_rows, "branch_switch_count"),
        "mean_longest_avoid_duration": _mean(
            timeout_rows, "longest_avoid_duration"
        ),
        "collision_episode_rate": (
            round(
                sum(bool(row["collision_episode"]) for row in timeout_rows)
                / len(timeout_rows),
                6,
            )
            if timeout_rows
            else None
        ),
    }


def _context_phase_summary(
    episodes: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[float, str], list[dict[str, object]]] = {}
    for episode in episodes:
        if episode["family"] != "context_shift":
            continue
        for phase in episode["phase_diagnostics"]:
            groups.setdefault(
                (float(episode["threshold"]), str(phase["phase"])), []
            ).append(phase)

    rows: list[dict[str, object]] = []
    for (threshold, phase), group in sorted(groups.items()):
        rows.append(
            {
                "scope": "context_phase",
                "threshold": threshold,
                "family": "context_shift",
                "phase": phase,
                "episodes": len(group),
                "mean_hazard_risk_activation_count": _mean(
                    group, "hazard_risk_activation_count"
                ),
                "mean_avoid_active_ratio": _mean(group, "avoid_active_ratio"),
                "collision_count": sum(int(row["collision_count"]) for row in group),
                "mean_goal_distance_change": _mean(group, "goal_distance_change"),
            }
        )
    return rows


def _family_preferences(
    family_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in family_rows:
        groups.setdefault(str(row["family"]), []).append(row)

    preferences: list[dict[str, object]] = []
    for family, rows in sorted(groups.items()):
        best_success = max(
            rows,
            key=lambda row: (
                float(row["success_rate"]),
                -float(row["collision_episode_rate"]),
                -float(row["mean_elapsed_time"]),
            ),
        )
        best_safety = min(
            rows,
            key=lambda row: (
                float(row["collision_episode_rate"]),
                -float(row["mean_minimum_clearance"] or -math.inf),
                -float(row["success_rate"]),
            ),
        )
        best_balanced = max(
            rows,
            key=lambda row: (
                float(row["success_rate"])
                - float(row["collision_episode_rate"]),
                float(row["success_rate"]),
                -float(row["mean_elapsed_time"]),
            ),
        )
        preferences.append(
            {
                "family": family,
                "best_success_threshold": best_success["threshold"],
                "best_safety_threshold": best_safety["threshold"],
                "best_balanced_threshold": best_balanced["threshold"],
                "best_balanced_score": round(
                    float(best_balanced["success_rate"])
                    - float(best_balanced["collision_episode_rate"]),
                    6,
                ),
            }
        )
    return preferences


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return round(numerator / denominator, 6) if denominator else 0.0


def _attribution(
    overall: Sequence[dict[str, object]],
    preferences: Sequence[dict[str, object]],
) -> dict[str, object]:
    ordered = sorted(overall, key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in ordered]
    success = [float(row["success_rate"]) for row in ordered]
    timeout = [float(row["timeout_rate"]) for row in ordered]
    collision = [float(row["collision_episode_rate"]) for row in ordered]
    avoid = [float(row["mean_avoid_active_ratio"]) for row in ordered]
    success_range = max(success) - min(success)
    avoid_range = max(avoid) - min(avoid)
    correlations = {
        "threshold_vs_success": _pearson(thresholds, success),
        "threshold_vs_timeout": _pearson(thresholds, timeout),
        "threshold_vs_collision_episode": _pearson(thresholds, collision),
        "threshold_vs_avoid_active_ratio": _pearson(thresholds, avoid),
    }
    switching_material = success_range >= 0.05 and avoid_range >= 0.05
    safety_efficiency_tradeoff = (
        correlations["threshold_vs_avoid_active_ratio"] >= 0.3
        and correlations["threshold_vs_collision_episode"] <= -0.3
        and correlations["threshold_vs_timeout"] >= 0.3
    )
    balanced_thresholds = {
        float(row["best_balanced_threshold"]) for row in preferences
    }
    context_dependent = len(balanced_thresholds) > 1
    best_success_rate = max(success)
    # 如果固定 Action 在某个 threshold 下已达到高成功率，同时只改 switching
    # boundary 就产生显著变化，则不能把默认阈值下的失败归因给 Action substrate。
    action_bottleneck_remains = not (
        switching_material and best_success_rate >= 0.8
    )
    if not action_bottleneck_remains and context_dependent:
        case = "Case C — Context-dependent static optimum"
    elif not action_bottleneck_remains:
        case = "Case A — Switching bottleneck supported"
    else:
        case = "Case B — Action bottleneck remains"
    return {
        "case": case,
        "switching_material": switching_material,
        "safety_efficiency_tradeoff": safety_efficiency_tradeoff,
        "context_dependent_preferences": context_dependent,
        "action_bottleneck_remains": action_bottleneck_remains,
        "fixed_actions_freeze_ready": not action_bottleneck_remains,
        "best_success_rate": round(best_success_rate, 6),
        "success_rate_range": round(success_range, 6),
        "avoid_active_ratio_range": round(avoid_range, 6),
        "correlations": correlations,
        "rule": (
            "material if success range >= 0.05 and Avoid occupancy range >= 0.05; "
            "Action bottleneck is rejected when switching is material and one fixed "
            "threshold reaches >= 0.8 success; tradeoff separately requires threshold "
            "correlations Avoid >= 0.3, collision <= -0.3, timeout >= 0.3; Case C "
            "takes precedence when family balanced optima differ"
        ),
    }


def _summarize_episodes(
    episodes: Sequence[dict[str, object]],
    thresholds: Sequence[float],
    families: Sequence[str],
) -> dict[str, object]:
    """从已完成 episode 重建全部汇总，避免重复运行昂贵仿真。"""
    overall: list[dict[str, object]] = []
    by_family: list[dict[str, object]] = []
    timeout: list[dict[str, object]] = []
    for threshold in thresholds:
        threshold_rows = [
            row for row in episodes if float(row["threshold"]) == threshold
        ]
        overall.append(_aggregate(threshold_rows, scope="overall"))
        timeout.append(_timeout_aggregate(threshold_rows, threshold, "all"))
        for family in families:
            family_group = [
                row for row in threshold_rows if row["family"] == family
            ]
            by_family.append(
                _aggregate(family_group, scope="family", family=family)
            )
            timeout.append(_timeout_aggregate(family_group, threshold, family))

    preferences = _family_preferences(by_family)
    return {
        "overall_by_threshold": overall,
        "by_threshold_and_family": by_family,
        "timeout_by_threshold_and_family": timeout,
        "context_shift_by_threshold_and_phase": _context_phase_summary(episodes),
        "family_preferences": preferences,
        "attribution": _attribution(overall, preferences),
    }


def evaluate_threshold_sensitivity(
    *,
    families: Iterable[str] = RESEARCH_FAMILIES,
    seeds: Iterable[int] = range(1001, 1051),
    thresholds: Iterable[float] | None = None,
    episode_horizon: float | None = None,
) -> dict[str, object]:
    """运行 paired sweep；返回数据但不写文件，便于测试和复现。"""
    family_tuple = tuple(families)
    seed_tuple = tuple(int(seed) for seed in seeds)
    defaults = ConditionParameters()
    threshold_tuple = (
        build_hazard_threshold_grid(defaults)
        if thresholds is None
        else tuple(float(value) for value in thresholds)
    )
    if not family_tuple or not seed_tuple or not threshold_tuple:
        raise ValueError("families, seeds, and thresholds must not be empty")
    for threshold in threshold_tuple:
        defaults.set("hazard_threshold", threshold)
    defaults.reset_all()

    episodes = [
        _run_episode(family, seed, threshold, episode_horizon)
        for threshold in threshold_tuple
        for family in family_tuple
        for seed in seed_tuple
    ]
    return {
        "milestone": "R0.11",
        "simulation_dt": SIMULATION_DT,
        "bt_config": "condition_research",
        "swept_parameter": "hazard_threshold",
        "thresholds": list(threshold_tuple),
        "default_parameters": ConditionParameters().get_values(),
        "fixed_parameters": {
            "boundary_threshold": ConditionParameters().boundary_threshold,
            "goal_threshold": ConditionParameters().goal_threshold,
        },
        "families": list(family_tuple),
        "seeds": list(seed_tuple),
        "episodes": episodes,
        "summary": _summarize_episodes(episodes, threshold_tuple, family_tuple),
    }


SUMMARY_CSV_FIELDS = (
    "scope",
    "threshold",
    "family",
    "phase",
    "episodes",
    "successes",
    "success_rate",
    "timeouts",
    "timeout_rate",
    "collision_episodes",
    "collision_episode_rate",
    "collision_count",
    "mean_elapsed_time",
    "mean_path_length",
    "mean_minimum_clearance",
    "mean_hazard_risk_activation_count",
    "mean_avoid_active_ratio",
    "mean_move_to_goal_active_ratio",
    "mean_boundary_active_ratio",
    "mean_branch_switch_count",
    "mean_longest_avoid_duration",
    "mean_goal_distance_change",
)

EPISODE_CSV_FIELDS = (
    "threshold",
    "boundary_threshold",
    "goal_threshold",
    "family",
    "seed",
    "success",
    "timeout",
    "termination_reason",
    "collision_episode",
    "collision_count",
    "elapsed_time",
    "path_length",
    "minimum_clearance",
    "simulation_steps",
    "hazard_risk_activation_count",
    "avoid_active_steps",
    "avoid_active_time",
    "avoid_active_ratio",
    "move_to_goal_active_steps",
    "move_to_goal_active_ratio",
    "boundary_active_steps",
    "boundary_active_ratio",
    "branch_switch_count",
    "longest_avoid_duration",
)


def write_results(
    payload: dict[str, object],
    json_path: Path = DEFAULT_JSON_PATH,
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV_PATH,
    episode_csv_path: Path = DEFAULT_EPISODE_CSV_PATH,
) -> tuple[Path, Path, Path]:
    """保存完整 JSON、汇总 CSV 和 episode-level CSV。"""
    paths = tuple(Path(path) for path in (json_path, summary_csv_path, episode_csv_path))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = payload["summary"]
    summary_rows = (
        list(summary["overall_by_threshold"])
        + list(summary["by_threshold_and_family"])
        + list(summary["context_shift_by_threshold_and_phase"])
    )
    with paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_CSV_FIELDS})

    with paths[2].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_CSV_FIELDS)
        writer.writeheader()
        for row in payload["episodes"]:
            writer.writerow({field: row.get(field) for field in EPISODE_CSV_FIELDS})
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R0.11 runtime Hazard threshold sensitivity sweep."
    )
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--seed-end", type=int, default=1050)
    parser.add_argument("--thresholds", nargs="+", type=float)
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
    print("R0.11 Hazard threshold sensitivity")
    for row in payload["summary"]["overall_by_threshold"]:
        print(
            f"  theta={row['threshold']:>5.1f}: success={row['success_rate']:.1%}, "
            f"timeout={row['timeout_rate']:.1%}, "
            f"collision={row['collision_episode_rate']:.1%}, "
            f"avoid={row['mean_avoid_active_ratio']:.1%}"
        )
    print(payload["summary"]["attribution"]["case"])


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be greater than or equal to seed-start")
    if args.episode_horizon is not None and args.episode_horizon <= 0.0:
        raise ValueError("episode-horizon must be positive")
    payload = evaluate_threshold_sensitivity(
        seeds=range(args.seed_start, args.seed_end + 1),
        thresholds=args.thresholds,
        episode_horizon=args.episode_horizon,
    )
    paths = write_results(
        payload,
        args.json_output,
        args.summary_csv_output,
        args.episode_csv_output,
    )
    _print_summary(payload)
    print(f"JSON:        {paths[0]}")
    print(f"Summary CSV: {paths[1]}")
    print(f"Episode CSV: {paths[2]}")


if __name__ == "__main__":
    main()
