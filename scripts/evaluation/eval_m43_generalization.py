"""用冻结的 M4.2 BT/PPO runners 评价有限手工几何变化的 zero-shot 行为。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m41b_control10hz.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparisons"
DEFAULT_SEED = 5001
M43_SCENARIO_GROUPS = {
    "rl_sanity": "seen",
    "ppo_simple_obstacle": "seen",
    "m43_target_shift": "unseen_mild",
    "m43_obstacle_shift": "unseen_mild",
    "m43_reverse_detour": "unseen_mild",
    "m43_combined_shift": "unseen_mild",
    "ppo_simple_obstacles": "ood_hard",
}
HUMAN_SCENARIOS = (
    "ppo_simple_obstacle",
    "m43_target_shift",
    "m43_obstacle_shift",
    "m43_reverse_detour",
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
    "initial_detour_direction",
)


def initial_detour_direction(
    trajectory: Sequence[Sequence[float]],
    obstacle: Sequence[float],
    agent_radius: float,
) -> str:
    """返回首次越过障碍前的实质垂直偏移方向。

    Pygame 的 y 轴向下，因此 y 变小是 upper、y 变大是 lower。以 Agent
    radius 作为噪声阈值，避免把几像素控制抖动误判为路线选择。
    """
    if not trajectory:
        return "ambiguous"
    start_y = float(trajectory[0][2])
    obstacle_right = float(obstacle[0]) + float(obstacle[2])
    threshold = float(agent_radius)
    for point in trajectory[1:]:
        x = float(point[1])
        y = float(point[2])
        if x > obstacle_right + agent_radius:
            break
        if y <= start_y - threshold:
            return "upper"
        if y >= start_y + threshold:
            return "lower"
    return "ambiguous"


def _latest_episode_payload(output_dir: Path) -> dict[str, Any]:
    """读取 runner 刚追加的最大 Episode 编号，不修改 Recorder schema。"""
    paths = list((output_dir / "runs").glob("episode_*.json"))
    if not paths:
        raise RuntimeError(f"no Recorder Episode found under {output_dir}")
    path = max(paths, key=lambda item: int(item.stem.removeprefix("episode_")))
    return json.loads(path.read_text(encoding="utf-8"))


def _direction_for_episode(
    scenario: str,
    initial_state: dict[str, Any],
    output_dir: Path,
) -> str:
    """从已有 Recorder trajectory 派生路线方向，不向 World/Recorder 加字段。"""
    obstacles = initial_state["obstacles"]
    if not obstacles:
        return "ambiguous"
    payload = _latest_episode_payload(output_dir)
    return initial_detour_direction(
        payload["trajectory"],
        obstacles[0],
        float(initial_state["agent_radius"]),
    )


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def build_generalization_summary(
    rows: list[dict[str, Any]], model_path: Path, seed: int
) -> dict[str, Any]:
    """以场景为统计单位汇总 seen/mild/hard，并计算主要泛化下降。"""
    scenario_summary = []
    for row in sorted(rows, key=lambda item: (item["scenario"], item["controller"])):
        scenario_summary.append(dict(row))

    group_summary = []
    for controller in ("bt", "ppo"):
        for group in ("seen", "unseen_mild", "ood_hard"):
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
                    "success_rate": len(successful) / len(group_rows),
                    "mean_elapsed_time": _mean(group_rows, "elapsed_time"),
                    "mean_path_length": _mean(group_rows, "path_length"),
                    "mean_collision_count": _mean(group_rows, "collision_count"),
                    "timeout_scenarios": sum(
                        row["termination_reason"] == "timeout" for row in group_rows
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
        for controller in ("bt", "ppo")
    }

    # 路径/时间变化只与几何训练 baseline ppo_simple_obstacle 比较，避免把
    # 无障碍 rl_sanity 混入导航效率参考。
    reference = {
        row["controller"]: row
        for row in rows
        if row["scenario"] == "ppo_simple_obstacle"
    }
    geometry_deltas = []
    for row in rows:
        if row["scenario_group"] != "unseen_mild":
            continue
        baseline = reference[row["controller"]]
        geometry_deltas.append(
            {
                "controller": row["controller"],
                "scenario": row["scenario"],
                "elapsed_time_delta": (
                    float(row["elapsed_time"]) - float(baseline["elapsed_time"])
                ),
                "path_length_delta": (
                    float(row["path_length"]) - float(baseline["path_length"])
                ),
                "collision_count_delta": (
                    int(row["collision_count"]) - int(baseline["collision_count"])
                ),
            }
        )

    return {
        "ppo_checkpoint": str(model_path),
        "bt_config": "default",
        "world_simulation_frequency_hz": 60.0,
        "bt_decision_frequency_hz": BT_DECISION_FREQUENCY,
        "ppo_decision_frequency_hz": PPO_DECISION_FREQUENCY,
        "ppo_action_repeat": PPO_ACTION_REPEAT,
        "seed": seed,
        "statistical_unit": "scenario",
        "seed_interpretation": (
            "All layouts are fixed. Seed 5001 standardizes reset/evaluation and "
            "does not represent a random geometry sample."
        ),
        "scenario_groups": {
            group: [
                scenario
                for scenario, scenario_group in M43_SCENARIO_GROUPS.items()
                if scenario_group == group
            ]
            for group in ("seen", "unseen_mild", "ood_hard")
        },
        "controller_scenario_results": scenario_summary,
        "controller_group_summary": group_summary,
        "generalization_drop": generalization_drop,
        "geometry_deltas_vs_ppo_simple_obstacle": geometry_deltas,
        "interpretation_limit": (
            "Results describe only the listed hand-authored geometry variants; "
            "they do not establish arbitrary-map generalization."
        ),
    }


def write_generalization_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    seed: int,
) -> tuple[Path, Path]:
    """覆盖 M4.3 汇总文件，但不触碰 M4.2 CSV/summary 或历史 Episode。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m43_generalization.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)

    summary_path = output_dir / "m43_generalization_summary.json"
    summary_path.write_text(
        json.dumps(
            build_generalization_summary(rows, model_path, seed), indent=2
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


def run_batch(
    model: PPO,
    model_path: Path,
    output_dir: Path,
    seed: int,
    bt_config: str,
) -> tuple[Path, Path]:
    """每个固定场景只运行一次 BT/PPO，并验证同场景初态一致。"""
    rows: list[dict[str, Any]] = []
    runs_root = output_dir / "m43_runs"
    for scenario, group in M43_SCENARIO_GROUPS.items():
        bt_output = runs_root / scenario / "bt"
        ppo_output = runs_root / scenario / "ppo"
        bt_row, bt_state = run_bt_episode(
            scenario, seed, bt_output, bt_config=bt_config
        )
        ppo_row, ppo_state = run_ppo_episode(
            scenario, seed, ppo_output, model
        )
        assert_initial_states_match(bt_state, ppo_state)
        for row, state, run_output in (
            (bt_row, bt_state, bt_output),
            (ppo_row, ppo_state, ppo_output),
        ):
            # M4.2 runner 的 scenario_role 只认识当时的 primary/hard 两组；
            # M4.3 使用自己的 seen/unseen_mild/ood_hard 分组，移除旧标签以免误读。
            row.pop("scenario_role", None)
            row["scenario_group"] = group
            row["initial_detour_direction"] = _direction_for_episode(
                scenario, state, run_output
            )
            rows.append(row)
    return write_generalization_outputs(rows, output_dir, model_path, seed)


def run_human_demos(
    model: PPO,
    output_dir: Path,
    seed: int,
    bt_config: str,
) -> None:
    """运行 4 场景 × 2 Controller 的独立可视观察，不写 batch 汇总。"""
    root = output_dir / "m43_human_demos"
    for scenario in HUMAN_SCENARIOS:
        for controller in ("bt", "ppo"):
            run_output = root / scenario / controller
            if controller == "bt":
                row, state = run_bt_episode(
                    scenario,
                    seed,
                    run_output,
                    render_mode="human",
                    bt_config=bt_config,
                )
            else:
                row, state = run_ppo_episode(
                    scenario, seed, run_output, model, render_mode="human"
                )
            direction = _direction_for_episode(scenario, state, run_output)
            print(
                f"Human {scenario} {controller}: "
                f"{row['termination_reason']}, detour={direction}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bt", default="default")
    parser.add_argument("--human-demo", action="store_true")
    args = parser.parse_args(argv)
    if not args.model_path.exists():
        parser.error(f"PPO model does not exist: {args.model_path}")
    return args


def _print_summary(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["controller_group_summary"]:
        print(
            f"{row['controller']:>3} {row['scenario_group']:>11}: "
            f"{row['successful_scenarios']}/{row['scenarios']} success, "
            f"time={row['mean_elapsed_time']:.3f}s "
            f"path={row['mean_path_length']:.1f}px "
            f"collisions={row['mean_collision_count']:.2f}"
        )
    print(f"Generalization drop: {payload['generalization_drop']}")


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path, device="cpu")
    if args.human_demo:
        run_human_demos(model, args.output_dir, args.seed, args.bt)
        return
    csv_path, summary_path = run_batch(
        model, args.model_path, args.output_dir, args.seed, args.bt
    )
    _print_summary(summary_path)
    print(f"Saved M4.3 CSV: {csv_path}")
    print(f"Saved M4.3 summary: {summary_path}")


if __name__ == "__main__":
    main()
