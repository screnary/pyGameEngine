"""用固定 Research families/seeds 校准局部 Hazard sensing range。

本脚本只构建初始 World 与 SemanticPerception，不运行 Controller、Episode 或训练。
统计单位是一个 ``family + seed`` 初态；结果用于解释默认 sensing profile，而不是
声称策略性能或随机泛化。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from autonomy_lab.core.environment import Environment
from autonomy_lab.scenarios.config import DEFAULT_RESEARCH_SENSOR_CONFIG
from autonomy_lab.scenarios.scenario_distribution import ScenarioDistribution


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / "analysis" / "r08_hazard_sensing_range.json"
)
DEFAULT_FAMILIES = (
    "static_random",
    "dense_hazard",
    "dynamic_hazard",
    "noisy_perception",
    "context_shift",
)
DEFAULT_HAZARD_RANGES = (200.0, 300.0, 400.0, 500.0, 700.0)
DEFAULT_GOAL_RANGES = (700.0, 850.0)
DEFAULT_SEED_START = 1001
DEFAULT_SEED_COUNT = 50


def _validate_ranges(values: Sequence[float], name: str) -> tuple[float, ...]:
    ranges = tuple(float(value) for value in values)
    if not ranges or any(not math.isfinite(value) or value <= 0.0 for value in ranges):
        raise ValueError(f"{name} must contain positive finite values")
    return ranges


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    """对较小样本做线性插值 percentile；空样本显式返回 None。"""
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _clearance_distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered) if ordered else None,
        "min": ordered[0] if ordered else None,
        "p10": _percentile(ordered, 0.10),
        "p25": _percentile(ordered, 0.25),
        "p50": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1] if ordered else None,
    }


def analyze_sensing_ranges(
    *,
    families: Sequence[str],
    seeds: Sequence[int],
    hazard_ranges: Sequence[float],
    goal_ranges: Sequence[float],
) -> dict[str, Any]:
    """返回候选量程在固定 Research 初态上的可复现 coverage 统计。"""
    family_names = tuple(str(family) for family in families)
    numeric_seeds = tuple(int(seed) for seed in seeds)
    candidate_hazard_ranges = _validate_ranges(hazard_ranges, "hazard_ranges")
    candidate_goal_ranges = _validate_ranges(goal_ranges, "goal_ranges")
    if not family_names or not numeric_seeds:
        raise ValueError("families and seeds must not be empty")
    sample_count = len(family_names) * len(numeric_seeds)

    goal_rows: list[dict[str, float | int]] = []
    for goal_range in candidate_goal_ranges:
        sensed_count = 0
        for family in family_names:
            for seed in numeric_seeds:
                scene = ScenarioDistribution(family).sample(seed)
                scene["sensor"]["goal_range"] = goal_range
                world = Environment(scene)
                sensed_count += int(world.perception.snapshot.goal.sensed)
        goal_rows.append(
            {
                "goal_range": goal_range,
                "sensed_samples": sensed_count,
                "initial_sensing_rate": sensed_count / sample_count,
            }
        )

    hazard_rows: list[dict[str, Any]] = []
    for hazard_range in candidate_hazard_ranges:
        availability_count = 0
        nearest_available_count = 0
        sensed_counts: list[int] = []
        all_visible_count = 0
        nearest_clearances: list[float] = []
        family_rows: list[dict[str, float | int | str]] = []
        for family in family_names:
            family_availability = 0
            family_sensed_counts: list[int] = []
            family_all_visible = 0
            for seed in numeric_seeds:
                scene = ScenarioDistribution(family).sample(seed)
                scene["sensor"]["hazard_range"] = hazard_range
                world = Environment(scene)
                hazard = world.perception.snapshot.hazard
                sensed_count = len(hazard.visible_hazards)
                total_count = len(world.obstacles)
                available = hazard.available
                availability_count += int(available)
                nearest_available_count += int(hazard.nearest_hazard is not None)
                sensed_counts.append(sensed_count)
                all_visible_count += int(sensed_count == total_count)
                family_availability += int(available)
                family_sensed_counts.append(sensed_count)
                family_all_visible += int(sensed_count == total_count)
                if hazard.nearest_clearance is not None:
                    nearest_clearances.append(hazard.nearest_clearance)
            family_rows.append(
                {
                    "family": family,
                    "hazard_availability_rate": (
                        family_availability / len(numeric_seeds)
                    ),
                    "mean_sensed_hazard_count": mean(family_sensed_counts),
                    "all_hazards_visible_rate": (
                        family_all_visible / len(numeric_seeds)
                    ),
                }
            )
        hazard_rows.append(
            {
                "hazard_range": hazard_range,
                "hazard_availability_rate": availability_count / sample_count,
                "nearest_hazard_availability_rate": (
                    nearest_available_count / sample_count
                ),
                "mean_sensed_hazard_count": mean(sensed_counts),
                "all_hazards_visible_rate": all_visible_count / sample_count,
                "nearest_clearance_distribution": _clearance_distribution(
                    nearest_clearances
                ),
                "by_family": family_rows,
            }
        )

    return {
        "families": list(family_names),
        "seeds": list(numeric_seeds),
        "sample_count": sample_count,
        "statistical_unit": "family_seed_initial_state",
        "goal_geometry_interpretation": (
            "The five families reuse the same sampled Agent/Goal geometry for a "
            "given seed; Goal rates therefore cover 50 unique Goal geometries, "
            "represented across 250 family-seed initial states."
        ),
        "configured_goal_range": float(
            DEFAULT_RESEARCH_SENSOR_CONFIG["goal_range"]
        ),
        "configured_hazard_range": float(
            DEFAULT_RESEARCH_SENSOR_CONFIG["hazard_range"]
        ),
        "goal_ranges": goal_rows,
        "hazard_ranges": hazard_rows,
    }


def write_analysis(result: dict[str, Any], output_path: Path) -> Path:
    """把一次校准结果保存为独立 JSON，不接入 ExperimentRecorder schema。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _print_summary(result: dict[str, Any]) -> None:
    for row in result["goal_ranges"]:
        print(
            f"Goal {row['goal_range']:>5.0f} px: "
            f"initial sensing={row['initial_sensing_rate']:.1%}"
        )
    print()
    print("Hazard range | available | mean count | all visible | nearest p50")
    for row in result["hazard_ranges"]:
        p50 = row["nearest_clearance_distribution"]["p50"]
        p50_text = "none" if p50 is None else f"{p50:.1f} px"
        print(
            f"{row['hazard_range']:>12.0f} | "
            f"{row['hazard_availability_rate']:>9.1%} | "
            f"{row['mean_sensed_hazard_count']:>10.3f} | "
            f"{row['all_hazards_visible_rate']:>11.1%} | {p50_text}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument(
        "--hazard-ranges",
        type=float,
        nargs="+",
        default=DEFAULT_HAZARD_RANGES,
    )
    parser.add_argument(
        "--goal-ranges",
        type=float,
        nargs="+",
        default=DEFAULT_GOAL_RANGES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = analyze_sensing_ranges(
        families=DEFAULT_FAMILIES,
        seeds=range(args.seed_start, args.seed_start + args.seed_count),
        hazard_ranges=args.hazard_ranges,
        goal_ranges=args.goal_ranges,
    )
    _print_summary(result)
    print(f"Saved: {write_analysis(result, args.output)}")


if __name__ == "__main__":
    main()
