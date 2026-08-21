"""在指定场景用同一组固定 seed 比较 Random Policy 与 PPO。"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from stable_baselines3 import PPO

from autonomy_lab.experiment.recorder import ExperimentRecorder
from autonomy_lab.gym.env import AgentGymEnv
from autonomy_lab.scenarios.config import SCENES


# 脚本迁入 ``scripts/evaluation/`` 后，默认路径仍指向项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m40.zip"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "m40_eval"
DEFAULT_SCENARIO = "rl_sanity"
TRAINING_SEED = 42
# 评估 seed 与训练初始化 seed 分离；Random 和 PPO 始终共享这个有序集合。
EVALUATION_SEEDS = tuple(range(1001, 1011))


def summarize_results(payloads: list[dict]) -> dict[str, float | int]:
    """把 Recorder payload 汇总成控制器间可比较的四项 M2 指标。"""
    if not payloads:
        raise ValueError("at least one Episode payload is required")
    episode_count = len(payloads)
    return {
        "episodes": episode_count,
        "success_rate": sum(
            payload["result"] == "SUCCESS" for payload in payloads
        )
        / episode_count,
        "mean_elapsed_time": sum(
            float(payload["elapsed_time"]) for payload in payloads
        )
        / episode_count,
        "mean_path_length": sum(
            float(payload["path_length"]) for payload in payloads
        )
        / episode_count,
        "mean_collision_count": sum(
            int(payload["collision_count"]) for payload in payloads
        )
        / episode_count,
    }


def is_clearly_better(random_summary: dict, ppo_summary: dict) -> bool:
    """判定是否达到 M4.0 的稳定学习证据，而不是比较训练 reward。"""
    random_rate = float(random_summary["success_rate"])
    ppo_rate = float(ppo_summary["success_rate"])
    # 80% 表示固定简单任务已稳定完成；50 个百分点避免把偶然命中当作学习。
    return ppo_rate >= 0.8 and ppo_rate - random_rate >= 0.5


def run_evaluation(
    controller: str,
    seeds: Iterable[int],
    output_dir: Path,
    render_mode: str | None,
    model: PPO | None = None,
    scenario: str = DEFAULT_SCENARIO,
    action_repeat: int = 1,
    contact_penalty_per_step: float = 0.0,
) -> list[dict]:
    """按指定 seed 运行 Episode，并复用 ExperimentRecorder 的指标定义。

    Random Policy 为每个 Episode 创建独立 RNG；PPO 使用 deterministic predict。
    两条路径只产生同一种 ``[turn, throttle]`` Action，不直接修改 World State。
    """
    if controller not in {"random", "ppo"}:
        raise ValueError("controller must be 'random' or 'ppo'")
    if controller == "ppo" and model is None:
        raise ValueError("PPO evaluation requires a loaded model")

    output_dir = Path(output_dir)
    recorder = ExperimentRecorder(output_dir)
    env = AgentGymEnv(
        scenario=scenario,
        render_mode=render_mode,
        recorder=recorder,
        recorder_controller=controller,
        action_repeat=action_repeat,
        contact_penalty_per_step=contact_penalty_per_step,
    )
    payloads: list[dict] = []
    diagnostic_episodes: list[dict] = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            random_generator = np.random.default_rng(int(seed))

            # 诊断读取 World Ground Truth 只用于离线 evaluation 文件；这些值不会
            # 进入 Observation、info 或 model.predict()，也不改变训练 reward。
            initial_target_distance = (
                env.world.target - env.world.agent.position
            ).length()
            min_target_distance = initial_target_distance
            target_visible_steps = 0
            obstacle_visible_steps = 0
            abs_turn_sum = 0.0
            throttle_sum = 0.0
            progress_reward_sum = 0.0
            step_reward_sum = 0.0
            collision_event_reward_sum = 0.0
            contact_penalty_sum = 0.0
            goal_reward_sum = 0.0
            contact_duration = 0.0
            total_reward = 0.0
            step_count = 0

            while True:
                # 当前 Observation 是本步 Action 真正读取的输入，因此可见率按
                # decision samples 统计，而不是混入终止后的额外快照。
                target_visible_steps += int(observation[3] == 1.0)
                obstacle_visible_steps += int(observation[6] == 1.0)
                if controller == "random":
                    action = random_generator.uniform(-1.0, 1.0, size=2).astype(
                        np.float32
                    )
                else:
                    action, _ = model.predict(observation, deterministic=True)
                    action = np.asarray(action, dtype=np.float32)

                observation, reward, terminated, truncated, _ = env.step(action)
                current_target_distance = (
                    env.world.target - env.world.agent.position
                ).length()
                step_count += 1
                abs_turn_sum += abs(float(action[0]))
                throttle_sum += float(action[1])
                components = env.last_reward_components
                progress_reward_sum += float(components["progress_reward"])
                step_reward_sum += float(components["step_reward"])
                collision_event_reward_sum += float(
                    components["collision_event_reward"]
                )
                contact_penalty_sum += float(
                    components["contact_penalty_reward"]
                )
                goal_reward_sum += float(components["goal_reward"])
                contact_duration += float(components["contact_duration"])
                total_reward += float(reward)
                min_target_distance = min(
                    min_target_distance, current_target_distance
                )
                if terminated or truncated:
                    payload = env.last_episode_payload
                    if payload is None:
                        raise RuntimeError("Recorder did not finish the Episode")
                    payloads.append(payload)
                    diagnostic_episodes.append(
                        {
                            "episode_id": payload["episode_id"],
                            "seed": int(seed),
                            "initial_target_distance": initial_target_distance,
                            "min_target_distance": min_target_distance,
                            "final_target_distance": current_target_distance,
                            "target_visible_ratio": (
                                target_visible_steps / step_count
                            ),
                            "obstacle_visible_ratio": (
                                obstacle_visible_steps / step_count
                            ),
                            "mean_abs_turn": abs_turn_sum / step_count,
                            "mean_throttle": throttle_sum / step_count,
                            "progress_reward_sum": progress_reward_sum,
                            "step_reward_sum": step_reward_sum,
                            # 旧键保留兼容 M4.1a；新键明确其事件语义。
                            "collision_reward_sum": collision_event_reward_sum,
                            "collision_event_reward_sum": (
                                collision_event_reward_sum
                            ),
                            "contact_penalty_sum": contact_penalty_sum,
                            "contact_duration": contact_duration,
                            "goal_reward_sum": goal_reward_sum,
                            "total_reward": total_reward,
                            "collision_count": payload["collision_count"],
                            "termination_reason": payload["termination_reason"],
                            "trajectory_file": (
                                f"runs/episode_{payload['episode_id']}.json"
                            ),
                        }
                    )
                    break
    finally:
        env.close()

    # 一个普通 JSON 与现有 Recorder 文件并列即可，无需新增 logging framework。
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "scenario": scenario,
                "controller": controller,
                "typical_trajectory": (
                    diagnostic_episodes[0]["trajectory_file"]
                    if diagnostic_episodes
                    else None
                ),
                "episodes": diagnostic_episodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return payloads


def _print_summary(label: str, summary: dict) -> None:
    """打印一行足以人工比较的评估摘要。"""
    print(
        f"{label:>8} | success={summary['success_rate']:.0%} "
        f"time={summary['mean_elapsed_time']:.3f}s "
        f"path={summary['mean_path_length']:.1f}px "
        f"collisions={summary['mean_collision_count']:.2f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析评估参数；默认值继续对应 M4.0。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=sorted(SCENES), default=DEFAULT_SCENARIO
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--controller", choices=("both", "random", "ppo"), default="both"
    )
    parser.add_argument(
        "--render-mode", choices=("none", "human"), default="none"
    )
    parser.add_argument(
        "--episodes", type=int, default=len(EVALUATION_SEEDS)
    )
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--contact-penalty-per-step", type=float, default=0.0)
    parser.add_argument("--evaluation-seed-start", type=int, default=1001)
    parser.add_argument(
        "--tag",
        default="latest",
        help="output subdirectory, e.g. 50k or 100k",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= len(EVALUATION_SEEDS):
        parser.error(f"--episodes must be within 1..{len(EVALUATION_SEEDS)}")
    if args.action_repeat <= 0:
        parser.error("--action-repeat must be positive")
    if args.contact_penalty_per_step > 0.0:
        parser.error("--contact-penalty-per-step must be zero or negative")
    return args


def main() -> None:
    """执行评估并保存当前 checkpoint 的公共指标证据。"""
    args = parse_args()

    seeds = tuple(
        range(
            args.evaluation_seed_start,
            args.evaluation_seed_start + args.episodes,
        )
    )
    render_mode = None if args.render_mode == "none" else "human"
    checkpoint_dir = DEFAULT_OUTPUT_ROOT / args.tag
    summaries: dict[str, dict] = {}

    if args.controller in {"both", "random"}:
        random_payloads = run_evaluation(
            "random",
            seeds,
            checkpoint_dir / "random",
            render_mode,
            scenario=args.scenario,
            action_repeat=args.action_repeat,
            contact_penalty_per_step=args.contact_penalty_per_step,
        )
        summaries["random"] = summarize_results(random_payloads)

    if args.controller in {"both", "ppo"}:
        model = PPO.load(args.model_path, device="cpu")
        ppo_payloads = run_evaluation(
            "ppo",
            seeds,
            checkpoint_dir / "ppo",
            render_mode,
            model=model,
            scenario=args.scenario,
            action_repeat=args.action_repeat,
            contact_penalty_per_step=args.contact_penalty_per_step,
        )
        summaries["ppo"] = summarize_results(ppo_payloads)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_path = checkpoint_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "seeds": list(seeds),
                "action_repeat": args.action_repeat,
                "contact_penalty_per_step": args.contact_penalty_per_step,
                **summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for label, summary in summaries.items():
        _print_summary(label, summary)
    if set(summaries) == {"random", "ppo"}:
        print(
            "Clearly better than Random: "
            f"{is_clearly_better(summaries['random'], summaries['ppo'])}"
        )
    print(f"Saved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
