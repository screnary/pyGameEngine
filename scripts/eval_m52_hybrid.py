"""比较 Frozen Hybrid 与 Hybrid-context trained PPO。"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from stable_baselines3 import PPO

from autonomy_lab.experiment.runners import (
    assert_initial_states_match,
    run_bt_episode,
    run_hybrid_policy_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m52_hybrid_trained.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparisons" / "m52"
DEFAULT_SEED = 7001
CONTROLLERS = ("frozen_hybrid", "hybrid_trained_ppo")
SCENARIO_GROUPS = ("seen", "hybrid_relevant", "additional_mild", "ood_hard")
M52_SCENARIO_GROUPS = {
    "rl_sanity": "seen",
    "ppo_simple_obstacle": "seen",
    "m43_obstacle_shift": "hybrid_relevant",
    "m43_reverse_detour": "hybrid_relevant",
    "m43_target_shift": "additional_mild",
    "m43_combined_shift": "additional_mild",
    "ppo_simple_obstacles": "ood_hard",
}
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
    "ppo_reentry_count",
    "observation_before_preemption",
    "observation_after_reentry",
)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def build_m52_summary(
    rows: list[dict[str, Any]],
    model_path: Path,
    seed: int,
    checkpoint_label: str,
) -> dict[str, Any]:
    group_summary = []
    for controller in CONTROLLERS:
        for group in SCENARIO_GROUPS:
            selected = [
                row
                for row in rows
                if row["controller"] == controller
                and row["scenario_group"] == group
            ]
            successes = [row for row in selected if row["success"]]
            group_summary.append(
                {
                    "controller": controller,
                    "scenario_group": group,
                    "scenarios": len(selected),
                    "successful_scenarios": len(successes),
                    "success_rate": (
                        len(successes) / len(selected) if selected else 0.0
                    ),
                    "mean_elapsed_time": _mean(selected, "elapsed_time"),
                    "mean_path_length": _mean(selected, "path_length"),
                    "mean_collision_count": _mean(
                        selected, "collision_count"
                    ),
                }
            )
    rates = {
        (row["controller"], row["scenario_group"]): row["success_rate"]
        for row in group_summary
    }
    return {
        "checkpoint_label": checkpoint_label,
        "trained_checkpoint": str(model_path),
        "initial_checkpoint": "models/ppo_m41b_control10hz.zip",
        "seed": seed,
        "world_simulation_frequency_hz": 60.0,
        "bt_supervision_frequency_hz": 60.0,
        "ppo_decision_frequency_hz": 10.0,
        "reward_note": (
            "Ground-truth target distance is privileged training-only reward "
            "shaping and is absent from the 13-D policy Observation."
        ),
        "scenario_groups": {
            group: [
                scenario
                for scenario, scenario_group in M52_SCENARIO_GROUPS.items()
                if scenario_group == group
            ]
            for group in SCENARIO_GROUPS
        },
        "controller_scenario_results": rows,
        "controller_group_summary": group_summary,
        "hybrid_relevant_success_improvement": (
            rates[("hybrid_trained_ppo", "hybrid_relevant")]
            - rates[("frozen_hybrid", "hybrid_relevant")]
        ),
        "hard_scene_is_not_acceptance_target": True,
    }


def _complete_frozen_diagnostics(row: dict[str, Any]) -> None:
    row.update(
        {
            "ppo_reentry_count": 0,
            "observation_before_preemption": None,
            "observation_after_reentry": None,
        }
    )


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    seed: int,
    checkpoint_label: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m52_frozen_vs_trained.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS} for row in rows
        )
    summary_path = output_dir / "m52_frozen_vs_trained_summary.json"
    summary_path.write_text(
        json.dumps(
            build_m52_summary(rows, model_path, seed, checkpoint_label),
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


def run_batch(
    model: PPO,
    model_path: Path,
    output_dir: Path,
    seed: int,
    checkpoint_label: str,
) -> tuple[Path, Path]:
    rows = []
    root = output_dir / "runs"
    for scenario, group in M52_SCENARIO_GROUPS.items():
        frozen_row, frozen_state = run_bt_episode(
            scenario,
            seed,
            root / scenario / "frozen_hybrid",
            bt_config="hybrid_ppo",
            recorder_controller="frozen_hybrid",
            collect_diagnostics=True,
        )
        trained_row, trained_state = run_hybrid_policy_episode(
            scenario,
            seed,
            root / scenario / "hybrid_trained_ppo",
            model,
        )
        assert_initial_states_match(frozen_state, trained_state)
        _complete_frozen_diagnostics(frozen_row)
        for row in (frozen_row, trained_row):
            row["scenario_group"] = group
            rows.append(row)
    return write_outputs(
        rows, output_dir, model_path, seed, checkpoint_label
    )


def run_human_demos(model: PPO, output_dir: Path, seed: int) -> None:
    root = output_dir / "human_demos"
    for scenario in ("ppo_simple_obstacle", "m43_reverse_detour"):
        run_bt_episode(
            scenario,
            seed,
            root / scenario / "frozen_hybrid",
            render_mode="human",
            bt_config="hybrid_ppo",
            recorder_controller="frozen_hybrid",
            collect_diagnostics=True,
        )
        run_hybrid_policy_episode(
            scenario,
            seed,
            root / scenario / "hybrid_trained_ppo",
            model,
            render_mode="human",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-label", default="200k")
    parser.add_argument("--human-demo", action="store_true")
    args = parser.parse_args(argv)
    if not args.model_path.exists():
        parser.error(f"trained model does not exist: {args.model_path}")
    return args


def _print_summary(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["controller_group_summary"]:
        print(
            f"{row['controller']:>20} {row['scenario_group']:>15}: "
            f"{row['successful_scenarios']}/{row['scenarios']} success, "
            f"time={row['mean_elapsed_time']:.3f}s "
            f"path={row['mean_path_length']:.1f}px "
            f"collisions={row['mean_collision_count']:.2f}"
        )
    print(
        "Hybrid-relevant success improvement: "
        f"{summary['hybrid_relevant_success_improvement']:+.3f}"
    )


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path, device="cpu")
    output_dir = args.output_dir / args.checkpoint_label
    if args.human_demo:
        run_human_demos(model, output_dir, args.seed)
        return
    _, summary_path = run_batch(
        model,
        args.model_path,
        output_dir,
        args.seed,
        args.checkpoint_label,
    )
    _print_summary(summary_path)
    print(f"Saved M5.2 summary: {summary_path}")


if __name__ == "__main__":
    main()
