"""在相同固定 World 初态下批量比较正式 BT 与已训练 PPO baseline。"""

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
from autonomy_lab.scene_config import SCENES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m41b_control10hz.zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "comparisons"
DEFAULT_SCENARIOS = (
    "rl_sanity",
    "ppo_simple_obstacle",
    "ppo_simple_obstacles",
)
PRIMARY_SCENARIOS = {"rl_sanity", "ppo_simple_obstacle"}
DEFAULT_SEEDS = tuple(range(4001, 4011))

CSV_FIELDS = (
    "controller",
    "scenario",
    "scenario_role",
    "seed",
    "success",
    "elapsed_time",
    "path_length",
    "collision_count",
    "termination_reason",
    "decision_frequency_hz",
    "decision_count",
)


def scenario_role(scenario: str) -> str:
    """区分主要 baseline 与不参与混合总体结论的 hard stress test。"""
    return "primary" if scenario in PRIMARY_SCENARIOS else "hard_stress_test"


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    """空 successful subset 返回 None，避免把“无成功样本”伪装为 0。"""
    if not rows:
        return None
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 scenario × controller 汇总，不生成混合 hard test 的总体成功率。"""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["controller"]), []).append(row)

    summaries = []
    for (scenario, controller), episodes in sorted(grouped.items()):
        successful = [row for row in episodes if row["success"]]
        summaries.append(
            {
                "controller": controller,
                "scenario": scenario,
                "scenario_role": scenario_role(scenario),
                "episodes": len(episodes),
                "success_rate": len(successful) / len(episodes),
                "mean_elapsed_time_all": _mean(episodes, "elapsed_time"),
                "mean_path_length_all": _mean(episodes, "path_length"),
                "mean_collision_count_all": _mean(episodes, "collision_count"),
                "successful_episodes": len(successful),
                "mean_elapsed_time_successful": _mean(successful, "elapsed_time"),
                "mean_path_length_successful": _mean(successful, "path_length"),
                "decision_frequency_hz": float(
                    episodes[0]["decision_frequency_hz"]
                ),
                "mean_decision_count": _mean(episodes, "decision_count"),
                "termination_reasons": {
                    reason: sum(
                        row["termination_reason"] == reason for row in episodes
                    )
                    for reason in sorted(
                        {row["termination_reason"] for row in episodes}
                    )
                },
            }
        )
    return summaries


def write_comparison_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    seeds: Sequence[int],
) -> tuple[Path, Path]:
    """写入 M4.2 结构化结果，不触碰 M4.0/M4.1 历史目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "m42_bt_vs_ppo.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)

    summary_path = output_dir / "m42_bt_vs_ppo_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "ppo_checkpoint": str(model_path),
                "world_simulation_frequency_hz": 60.0,
                "bt_decision_frequency_hz": BT_DECISION_FREQUENCY,
                "ppo_decision_frequency_hz": PPO_DECISION_FREQUENCY,
                "ppo_action_repeat": PPO_ACTION_REPEAT,
                "seeds": list(seeds),
                "seed_interpretation": (
                    "Current scenarios use fixed layouts; repeated seeds preserve a "
                    "uniform evaluation pipeline and do not demonstrate random "
                    "generalization."
                ),
                "no_mixed_overall_rate": (
                    "The hard stress test is summarized separately and is not mixed "
                    "with the two primary baselines."
                ),
                "controller_scenario_summary": summarize_rows(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, summary_path


def run_batch(
    scenarios: Sequence[str],
    seeds: Sequence[int],
    model: PPO,
    model_path: Path,
    output_dir: Path,
    bt_config: str,
) -> tuple[Path, Path]:
    """逐 scenario/seed 运行 BT 与 PPO，并在写结果前验证每对初态。"""
    rows: list[dict[str, Any]] = []
    runs_root = output_dir / "runs"
    for scenario in scenarios:
        for seed in seeds:
            bt_row, bt_state = run_bt_episode(
                scenario,
                seed,
                runs_root / scenario / "bt",
                bt_config=bt_config,
            )
            ppo_row, ppo_state = run_ppo_episode(
                scenario,
                seed,
                runs_root / scenario / "ppo",
                model,
            )
            assert_initial_states_match(bt_state, ppo_state)
            # scenario_role 是 M4.2 的编排语义，不属于公共 Episode runner。
            role = scenario_role(scenario)
            bt_row["scenario_role"] = role
            ppo_row["scenario_role"] = role
            rows.extend((bt_row, ppo_row))
    return write_comparison_outputs(rows, output_dir, model_path, seeds)


def run_human_demos(
    seed: int,
    model: PPO,
    output_dir: Path,
    bt_config: str,
) -> None:
    """在独立目录顺序展示两个 primary scenario 的 BT/PPO 典型轨迹。"""
    for scenario in ("rl_sanity", "ppo_simple_obstacle"):
        bt_row, bt_state = run_bt_episode(
            scenario,
            seed,
            output_dir / scenario / "bt",
            render_mode="human",
            bt_config=bt_config,
        )
        ppo_row, ppo_state = run_ppo_episode(
            scenario,
            seed,
            output_dir / scenario / "ppo",
            model,
            render_mode="human",
        )
        assert_initial_states_match(bt_state, ppo_state)
        print(
            f"Human demo {scenario}: BT={bt_row['termination_reason']}, "
            f"PPO={ppo_row['termination_reason']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bt", default="default")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--episodes", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENES),
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument(
        "--human-demo",
        action="store_true",
        help="run four visual demos in a separate directory; do not write batch CSV",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not args.model_path.exists():
        parser.error(f"PPO model does not exist: {args.model_path}")
    return args


def _print_summary(summary_path: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in payload["controller_scenario_summary"]:
        print(
            f"{row['scenario']:>22} | {row['controller']:>3} | "
            f"success={row['success_rate']:.0%} "
            f"time(all)={row['mean_elapsed_time_all']:.3f}s "
            f"path(all)={row['mean_path_length_all']:.1f}px "
            f"collisions={row['mean_collision_count_all']:.2f} "
            f"decisions={row['mean_decision_count']:.1f}@"
            f"{row['decision_frequency_hz']:.0f}Hz"
        )


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path, device="cpu")
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    if args.human_demo:
        run_human_demos(
            seed=seeds[0],
            model=model,
            output_dir=args.output_dir / "human_demos",
            bt_config=args.bt,
        )
        return

    csv_path, summary_path = run_batch(
        scenarios=args.scenarios,
        seeds=seeds,
        model=model,
        model_path=args.model_path,
        output_dir=args.output_dir,
        bt_config=args.bt,
    )
    _print_summary(summary_path)
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
