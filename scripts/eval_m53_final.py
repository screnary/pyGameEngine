"""统一评价 M5 的 BT、PPO、Frozen Hybrid 与 Hybrid-trained PPO。"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from stable_baselines3 import PPO

from autonomy_lab.experiment.runners import (
    BT_DECISION_FREQUENCY,
    PPO_ACTION_REPEAT,
    PPO_DECISION_FREQUENCY,
    assert_initial_states_match,
    run_bt_episode,
    run_hybrid_policy_episode,
    run_ppo_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m41b_control10hz.zip"
DEFAULT_TRAINED_MODEL_PATH = (
    PROJECT_ROOT / "models" / "ppo_m52_hybrid_trained_200k.zip"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparisons" / "m53"
DEFAULT_SEED = 5001

CONTROLLERS = ("bt", "pure_ppo", "frozen_hybrid", "hybrid_trained_ppo")
SCENARIO_GROUPS = ("seen", "unseen_mild", "ood_hard")
M53_SCENARIO_GROUPS = {
    "rl_sanity": "seen",
    "ppo_simple_obstacle": "seen",
    "m43_target_shift": "unseen_mild",
    "m43_obstacle_shift": "unseen_mild",
    "m43_reverse_detour": "unseen_mild",
    "m43_combined_shift": "unseen_mild",
    "ppo_simple_obstacles": "ood_hard",
}
HUMAN_SCENARIOS = ("ppo_simple_obstacle", "m43_reverse_detour")
PUBLIC_METRICS = (
    "success",
    "elapsed_time",
    "path_length",
    "collision_count",
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
    "ppo_active_ratio",
    "boundary_recovery_activation_count",
    "search_activation_count",
    "ppo_preemption_count",
    "ppo_reentry_count",
)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    """计算全 Episode 均值；空组使用 None，不伪造零值。"""
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def _complete_diagnostics(row: dict[str, Any], controller: str) -> None:
    """把不同 runner 的结果投影到统一 CSV，非适用字段保持为空。"""
    defaults: dict[str, Any] = {
        "bt_tick_count": None,
        "bt_transition_count": None,
        "ppo_decision_count": None,
        "ppo_active_ratio": None,
        "boundary_recovery_activation_count": None,
        "search_activation_count": None,
        "ppo_preemption_count": None,
        "ppo_reentry_count": None,
    }
    if controller == "pure_ppo":
        defaults.update(
            {
                "ppo_decision_count": int(row["decision_count"]),
                "ppo_active_ratio": 1.0,
            }
        )
    elif controller == "frozen_hybrid":
        # 旧 Hybrid runtime 没有显式 re-entry counter；其余诊断来自 Controller。
        defaults["ppo_reentry_count"] = None
    elif controller == "bt":
        # 纯 BT 没有 PPO lifecycle，BT tick/transition 已由 runner 提供。
        pass
    for field, value in defaults.items():
        row.setdefault(field, value)
    row["controller"] = controller


def assert_adapter_equivalence(
    reference_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """确认旧 Frozen Hybrid 与 external-action Adapter 的公共语义一致。

    success/collision 是离散量，必须完全一致；time/path 允许很小浮点误差。
    返回的报告进入最终 JSON，但任何 drift 都直接终止评价，避免生成误导结果。
    """
    reference = {str(row["scenario"]): row for row in reference_rows}
    adapter = {str(row["scenario"]): row for row in adapter_rows}
    if reference.keys() != adapter.keys():
        raise AssertionError("adapter scenario set differs from frozen reference")

    scenario_reports = []
    for scenario in M53_SCENARIO_GROUPS:
        if scenario not in reference:
            continue
        expected = reference[scenario]
        actual = adapter[scenario]
        for metric in ("success", "collision_count"):
            if actual[metric] != expected[metric]:
                raise AssertionError(
                    f"adapter {metric} differs in {scenario}: "
                    f"{actual[metric]!r} != {expected[metric]!r}"
                )
        for metric in ("elapsed_time", "path_length"):
            if not math.isclose(
                float(actual[metric]),
                float(expected[metric]),
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            ):
                raise AssertionError(
                    f"adapter {metric} differs in {scenario}: "
                    f"{actual[metric]!r} != {expected[metric]!r}"
                )
        scenario_reports.append(
            {
                "scenario": scenario,
                "matched_metrics": list(PUBLIC_METRICS),
                "elapsed_time_delta": (
                    float(actual["elapsed_time"])
                    - float(expected["elapsed_time"])
                ),
                "path_length_delta": (
                    float(actual["path_length"])
                    - float(expected["path_length"])
                ),
            }
        )
    return {
        "passed": True,
        "absolute_tolerance": absolute_tolerance,
        "scenarios": scenario_reports,
    }


def build_m53_summary(
    rows: list[dict[str, Any]],
    frozen_model_path: Path,
    trained_model_path: Path,
    seed: int,
    adapter_equivalence: dict[str, Any],
) -> dict[str, Any]:
    """按 Controller × 场景组汇总，并显式保留 Hybrid training 负结果。"""
    group_summary = []
    for controller in CONTROLLERS:
        for group in SCENARIO_GROUPS:
            selected = [
                row
                for row in rows
                if row["controller"] == controller
                and row["scenario_group"] == group
            ]
            successful = [row for row in selected if row["success"]]
            group_summary.append(
                {
                    "controller": controller,
                    "scenario_group": group,
                    "scenarios": len(selected),
                    "successful_scenarios": len(successful),
                    "success_rate": (
                        len(successful) / len(selected) if selected else 0.0
                    ),
                    # 统一按 all episodes 统计；失败的 timeout 不被悄悄删除。
                    "mean_elapsed_time": _mean(selected, "elapsed_time"),
                    "mean_path_length": _mean(selected, "path_length"),
                    "mean_collision_count": _mean(selected, "collision_count"),
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
    hybrid_improvement = (
        rates[("hybrid_trained_ppo", "unseen_mild")]
        - rates[("frozen_hybrid", "unseen_mild")]
    )
    return {
        "milestone": "M5.3",
        "status": "COMPLETE",
        "frozen_ppo_checkpoint": str(frozen_model_path),
        "hybrid_trained_checkpoint": str(trained_model_path),
        "seed": seed,
        "statistical_unit": "fixed scenario",
        "world_simulation_frequency_hz": 60.0,
        "bt_supervision_frequency_hz": BT_DECISION_FREQUENCY,
        "ppo_decision_frequency_hz": PPO_DECISION_FREQUENCY,
        "ppo_action_repeat": PPO_ACTION_REPEAT,
        "metric_aggregation": "all episodes, measured from World simulation steps",
        "scenario_groups": {
            group: [
                scenario
                for scenario, scenario_group in M53_SCENARIO_GROUPS.items()
                if scenario_group == group
            ]
            for group in SCENARIO_GROUPS
        },
        "controller_scenario_results": sorted(
            rows, key=lambda row: (row["scenario"], row["controller"])
        ),
        "controller_group_summary": group_summary,
        "generalization_drop": generalization_drop,
        "hybrid_training_mild_success_improvement": hybrid_improvement,
        "hybrid_training_improvement_demonstrated": hybrid_improvement > 0.0,
        "adapter_equivalence": adapter_equivalence,
        "reward_is_not_a_public_metric": True,
        "interpretation_limit": (
            "The statistical unit is one fixed scenario. Results do not "
            "establish randomized-map or distributional generalization."
        ),
    }


def write_m53_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    frozen_model_path: Path,
    trained_model_path: Path,
    seed: int,
    adapter_equivalence: dict[str, Any],
) -> tuple[Path, Path]:
    """只写 M5.3 目录，不覆盖 M5.1/M5.2 历史结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m53_final.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS} for row in rows
        )
    summary_path = output_dir / "m53_final_summary.json"
    summary_path.write_text(
        json.dumps(
            build_m53_summary(
                rows,
                frozen_model_path,
                trained_model_path,
                seed,
                adapter_equivalence,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


def run_batch(
    frozen_model: PPO,
    trained_model: PPO,
    frozen_model_path: Path,
    trained_model_path: Path,
    output_dir: Path,
    seed: int,
) -> tuple[Path, Path]:
    """运行 7 场景 × 4 Controller，并执行额外 Adapter regression。"""
    rows: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []
    runs_root = output_dir / "runs"
    adapter_root = output_dir / "adapter_equivalence_runs"

    for scenario, group in M53_SCENARIO_GROUPS.items():
        bt_row, bt_state = run_bt_episode(
            scenario,
            seed,
            runs_root / scenario / "bt",
            bt_config="default",
            collect_diagnostics=True,
        )
        ppo_row, ppo_state = run_ppo_episode(
            scenario, seed, runs_root / scenario / "pure_ppo", frozen_model
        )
        frozen_row, frozen_state = run_bt_episode(
            scenario,
            seed,
            runs_root / scenario / "frozen_hybrid",
            bt_config="hybrid_ppo",
            recorder_controller="frozen_hybrid",
            collect_diagnostics=True,
        )
        trained_row, trained_state = run_hybrid_policy_episode(
            scenario,
            seed,
            runs_root / scenario / "hybrid_trained_ppo",
            trained_model,
        )
        adapter_row, adapter_state = run_hybrid_policy_episode(
            scenario,
            seed,
            adapter_root / scenario,
            frozen_model,
            recorder_controller="adapter_external_frozen",
        )

        # 四类 Controller 和 equivalence probe 必须从同一 World 几何初态开始。
        for state in (ppo_state, frozen_state, trained_state, adapter_state):
            assert_initial_states_match(bt_state, state)

        for row, controller in (
            (bt_row, "bt"),
            (ppo_row, "pure_ppo"),
            (frozen_row, "frozen_hybrid"),
            (trained_row, "hybrid_trained_ppo"),
        ):
            _complete_diagnostics(row, controller)
            row["scenario_group"] = group
            rows.append(row)
        frozen_rows.append(frozen_row)
        adapter_rows.append(adapter_row)

    equivalence = assert_adapter_equivalence(frozen_rows, adapter_rows)
    return write_m53_outputs(
        rows,
        output_dir,
        frozen_model_path,
        trained_model_path,
        seed,
        equivalence,
    )


def run_human_demos(
    frozen_model: PPO,
    trained_model: PPO,
    output_dir: Path,
    seed: int,
) -> None:
    """独立显示两个代表场景 × 四 Controller，不污染 batch CSV。"""
    root = output_dir / "human_demos"
    for scenario in HUMAN_SCENARIOS:
        bt_row, _ = run_bt_episode(
            scenario,
            seed,
            root / scenario / "bt",
            render_mode="human",
            bt_config="default",
            collect_diagnostics=True,
        )
        ppo_row, _ = run_ppo_episode(
            scenario,
            seed,
            root / scenario / "pure_ppo",
            frozen_model,
            render_mode="human",
        )
        frozen_row, _ = run_bt_episode(
            scenario,
            seed,
            root / scenario / "frozen_hybrid",
            render_mode="human",
            bt_config="hybrid_ppo",
            recorder_controller="frozen_hybrid",
            collect_diagnostics=True,
        )
        trained_row, _ = run_hybrid_policy_episode(
            scenario,
            seed,
            root / scenario / "hybrid_trained_ppo",
            trained_model,
            render_mode="human",
        )
        print(
            f"Human {scenario}: BT={bt_row['termination_reason']}, "
            f"PPO={ppo_row['termination_reason']}, "
            f"Frozen={frozen_row['termination_reason']}, "
            f"Trained={trained_row['termination_reason']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-model-path", type=Path, default=DEFAULT_FROZEN_MODEL_PATH
    )
    parser.add_argument(
        "--trained-model-path", type=Path, default=DEFAULT_TRAINED_MODEL_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--human-demo", action="store_true")
    args = parser.parse_args(argv)
    for label, path in (
        ("frozen", args.frozen_model_path),
        ("hybrid-trained", args.trained_model_path),
    ):
        if not path.exists():
            parser.error(f"{label} model does not exist: {path}")
    return args


def _print_summary(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["controller_group_summary"]:
        print(
            f"{row['controller']:>20} {row['scenario_group']:>11}: "
            f"{row['successful_scenarios']}/{row['scenarios']} success, "
            f"time={row['mean_elapsed_time']:.3f}s "
            f"path={row['mean_path_length']:.1f}px "
            f"collisions={row['mean_collision_count']:.2f}"
        )
    print(f"Generalization drop: {summary['generalization_drop']}")
    print(
        "Hybrid training mild-success improvement: "
        f"{summary['hybrid_training_mild_success_improvement']:+.3f}"
    )
    print(
        "Adapter equivalence: "
        f"{summary['adapter_equivalence']['passed']}"
    )


def main() -> None:
    args = parse_args()
    frozen_model = PPO.load(args.frozen_model_path, device="cpu")
    trained_model = PPO.load(args.trained_model_path, device="cpu")
    if args.human_demo:
        run_human_demos(
            frozen_model, trained_model, args.output_dir, args.seed
        )
        return
    csv_path, summary_path = run_batch(
        frozen_model,
        trained_model,
        args.frozen_model_path,
        args.trained_model_path,
        args.output_dir,
        args.seed,
    )
    _print_summary(summary_path)
    print(f"Saved M5.3 CSV: {csv_path}")
    print(f"Saved M5.3 summary: {summary_path}")


if __name__ == "__main__":
    main()
