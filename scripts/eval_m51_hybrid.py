"""统一评价冻结的 BT、PPO 与 Hybrid BT + PPO 三种 Controller。"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from stable_baselines3 import PPO

from autonomy_lab.experiment.runners import (
    BT_DECISION_FREQUENCY,
    PPO_ACTION_REPEAT,
    PPO_DECISION_FREQUENCY,
    assert_initial_states_match,
    run_bt_episode,
    run_ppo_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m41b_control10hz.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparisons" / "m51"
DEFAULT_SEED = 5001
M51_SCENARIO_GROUPS = {
    "rl_sanity": "seen",
    "ppo_simple_obstacle": "seen",
    "m43_target_shift": "unseen_mild",
    "m43_obstacle_shift": "unseen_mild",
    "m43_reverse_detour": "unseen_mild",
    "m43_combined_shift": "unseen_mild",
    "ppo_simple_obstacles": "ood_hard",
}
CONTROLLERS = ("bt", "ppo", "hybrid_bt_ppo")
SCENARIO_GROUPS = ("seen", "unseen_mild", "ood_hard")
HUMAN_SCENARIOS = (
    "ppo_simple_obstacle",
    "m43_reverse_detour",
    "ppo_simple_obstacles",
)
CSV_FIELDS = (
    "controller",
    "scenario",
    "scenario_group",
    "seed",
    "success",
    "elapsed_time",
    "path_length",
    "collision_count",
    "termination_reason",
    "decision_frequency_hz",
    "decision_count",
    "bt_tick_count",
    "bt_transition_count",
    "ppo_decision_count",
    "ppo_active_time",
    "ppo_active_ratio",
    "boundary_recovery_activation_count",
    "search_activation_count",
    "ppo_preemption_count",
)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    """计算公共指标均值；空分组返回 None，避免制造无意义的零。"""
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def build_m51_summary(
    rows: list[dict[str, Any]], model_path: Path, seed: int
) -> dict[str, Any]:
    """按 Controller × Scenario Group 汇总性能和 Hybrid 介入程度。"""
    group_summary = []
    for controller in CONTROLLERS:
        for group in SCENARIO_GROUPS:
            group_rows = [
                row
                for row in rows
                if row["controller"] == controller
                and row["scenario_group"] == group
            ]
            successful = [row for row in group_rows if row["success"]]
            group_summary.append(
                {
                    "controller": controller,
                    "scenario_group": group,
                    "scenarios": len(group_rows),
                    "successful_scenarios": len(successful),
                    "success_rate": (
                        len(successful) / len(group_rows) if group_rows else 0.0
                    ),
                    "mean_elapsed_time": _mean(group_rows, "elapsed_time"),
                    "mean_path_length": _mean(group_rows, "path_length"),
                    "mean_collision_count": _mean(
                        group_rows, "collision_count"
                    ),
                    "timeout_scenarios": sum(
                        row["termination_reason"] == "timeout"
                        for row in group_rows
                    ),
                }
            )

    rates = {
        (row["controller"], row["scenario_group"]): row["success_rate"]
        for row in group_summary
    }
    generalization_drop = {
        controller: rates[(controller, "seen")]
        - rates[(controller, "unseen_mild")]
        for controller in CONTROLLERS
    }
    hybrid_rows = [
        row for row in rows if row["controller"] == "hybrid_bt_ppo"
    ]
    hybrid_diagnostics = {
        "episodes": len(hybrid_rows),
        "total_ppo_decisions": sum(
            int(row["ppo_decision_count"]) for row in hybrid_rows
        ),
        "mean_ppo_decisions": _mean(hybrid_rows, "ppo_decision_count"),
        "mean_ppo_active_time": _mean(hybrid_rows, "ppo_active_time"),
        "mean_ppo_active_ratio": _mean(hybrid_rows, "ppo_active_ratio"),
        "total_boundary_recovery_activations": sum(
            int(row["boundary_recovery_activation_count"])
            for row in hybrid_rows
        ),
        "total_search_activations": sum(
            int(row["search_activation_count"]) for row in hybrid_rows
        ),
        "total_ppo_preemptions": sum(
            int(row["ppo_preemption_count"]) for row in hybrid_rows
        ),
        "per_scenario": [
            {
                field: row[field]
                for field in (
                    "scenario",
                    "ppo_decision_count",
                    "ppo_active_time",
                    "ppo_active_ratio",
                    "boundary_recovery_activation_count",
                    "search_activation_count",
                    "ppo_preemption_count",
                )
            }
            for row in hybrid_rows
        ],
    }
    return {
        "ppo_checkpoint": str(model_path),
        "bt_config": "default",
        "hybrid_bt_config": "hybrid_ppo",
        "world_simulation_frequency_hz": 60.0,
        "ppo_action_repeat": PPO_ACTION_REPEAT,
        "controller_frequencies": {
            "bt": {"bt_tick_hz": BT_DECISION_FREQUENCY},
            "ppo": {"ppo_inference_hz": PPO_DECISION_FREQUENCY},
            "hybrid_bt_ppo": {
                "bt_supervision_hz": BT_DECISION_FREQUENCY,
                "ppo_inference_hz_when_active": PPO_DECISION_FREQUENCY,
            },
        },
        "comparison_scope": (
            "Frozen complete-controller baselines; this is not a matched-"
            "frequency ablation. Reward is not a public evaluation metric."
        ),
        "seed": seed,
        "statistical_unit": "scenario",
        "seed_interpretation": (
            "All layouts are fixed. Seed 5001 standardizes reset/evaluation "
            "and does not represent a random geometry sample."
        ),
        "scenario_groups": {
            group: [
                scenario
                for scenario, scenario_group in M51_SCENARIO_GROUPS.items()
                if scenario_group == group
            ]
            for group in SCENARIO_GROUPS
        },
        "controller_scenario_results": sorted(
            rows, key=lambda row: (row["scenario"], row["controller"])
        ),
        "controller_group_summary": group_summary,
        "generalization_drop": generalization_drop,
        "hybrid_diagnostics": hybrid_diagnostics,
        "interpretation_limit": (
            "Results describe only the listed fixed scenes and cannot establish "
            "arbitrary-map or randomized-distribution generalization."
        ),
    }


def write_m51_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    seed: int,
) -> tuple[Path, Path]:
    """写入 M5.1 独立 CSV/JSON，不触碰 M4.2/M4.3 历史文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m51_bt_ppo_hybrid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS} for row in rows
        )

    summary_path = output_dir / "m51_bt_ppo_hybrid_summary.json"
    summary_path.write_text(
        json.dumps(build_m51_summary(rows, model_path, seed), indent=2),
        encoding="utf-8",
    )
    return csv_path, summary_path


def _complete_ppo_diagnostics(row: dict[str, Any]) -> None:
    """为 Pure PPO 补齐统一列；其整个 Episode 都由 PPO Policy 控制。"""
    row.update(
        {
            "bt_tick_count": None,
            "bt_transition_count": None,
            "ppo_decision_count": int(row["decision_count"]),
            "ppo_active_time": float(row["elapsed_time"]),
            "ppo_active_ratio": 1.0,
            "boundary_recovery_activation_count": 0,
            "search_activation_count": 0,
            "ppo_preemption_count": 0,
        }
    )


def run_batch(
    model: PPO,
    model_path: Path,
    output_dir: Path,
    seed: int,
    bt_config: str,
    hybrid_bt_config: str,
) -> tuple[Path, Path]:
    """每个固定场景各运行一次 BT/PPO/Hybrid，并核对三者 World 初态。"""
    rows: list[dict[str, Any]] = []
    runs_root = output_dir / "runs"
    for scenario, group in M51_SCENARIO_GROUPS.items():
        bt_row, bt_state = run_bt_episode(
            scenario,
            seed,
            runs_root / scenario / "bt",
            bt_config=bt_config,
            collect_diagnostics=True,
        )
        ppo_row, ppo_state = run_ppo_episode(
            scenario, seed, runs_root / scenario / "ppo", model
        )
        hybrid_row, hybrid_state = run_bt_episode(
            scenario,
            seed,
            runs_root / scenario / "hybrid_bt_ppo",
            bt_config=hybrid_bt_config,
            recorder_controller="hybrid_bt_ppo",
            collect_diagnostics=True,
        )
        assert_initial_states_match(bt_state, ppo_state)
        assert_initial_states_match(bt_state, hybrid_state)
        _complete_ppo_diagnostics(ppo_row)
        for row in (bt_row, ppo_row, hybrid_row):
            row["scenario_group"] = group
            rows.append(row)
    return write_m51_outputs(rows, output_dir, model_path, seed)


def run_human_demos(
    model: PPO,
    output_dir: Path,
    seed: int,
    bt_config: str,
    hybrid_bt_config: str,
) -> None:
    """独立运行三个代表场景 × 三种 Controller，不污染 batch 结果。"""
    root = output_dir / "human_demos"
    for scenario in HUMAN_SCENARIOS:
        bt_row, _ = run_bt_episode(
            scenario,
            seed,
            root / scenario / "bt",
            render_mode="human",
            bt_config=bt_config,
            collect_diagnostics=True,
        )
        ppo_row, _ = run_ppo_episode(
            scenario,
            seed,
            root / scenario / "ppo",
            model,
            render_mode="human",
        )
        hybrid_row, _ = run_bt_episode(
            scenario,
            seed,
            root / scenario / "hybrid_bt_ppo",
            render_mode="human",
            bt_config=hybrid_bt_config,
            recorder_controller="hybrid_bt_ppo",
            collect_diagnostics=True,
        )
        print(
            f"Human {scenario}: BT={bt_row['termination_reason']}, "
            f"PPO={ppo_row['termination_reason']}, "
            f"Hybrid={hybrid_row['termination_reason']}, "
            f"Hybrid PPO ratio={hybrid_row['ppo_active_ratio']:.3f}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bt", default="default")
    parser.add_argument("--hybrid-bt", default="hybrid_ppo")
    parser.add_argument("--human-demo", action="store_true")
    args = parser.parse_args(argv)
    if not args.model_path.exists():
        parser.error(f"PPO model does not exist: {args.model_path}")
    return args


def _print_summary(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["controller_group_summary"]:
        print(
            f"{row['controller']:>13} {row['scenario_group']:>11}: "
            f"{row['successful_scenarios']}/{row['scenarios']} success, "
            f"time={row['mean_elapsed_time']:.3f}s "
            f"path={row['mean_path_length']:.1f}px "
            f"collisions={row['mean_collision_count']:.2f}"
        )
    print(f"Generalization drop: {payload['generalization_drop']}")
    print(f"Hybrid diagnostics: {payload['hybrid_diagnostics']}")


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path, device="cpu")
    if args.human_demo:
        run_human_demos(
            model,
            args.output_dir,
            args.seed,
            args.bt,
            args.hybrid_bt,
        )
        return
    csv_path, summary_path = run_batch(
        model,
        args.model_path,
        args.output_dir,
        args.seed,
        args.bt,
        args.hybrid_bt,
    )
    _print_summary(summary_path)
    print(f"Saved M5.1 CSV: {csv_path}")
    print(f"Saved M5.1 summary: {summary_path}")


if __name__ == "__main__":
    main()
