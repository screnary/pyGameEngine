"""在指定固定场景上训练、初始化微调或续训一个最小 PPO。"""

import argparse
from pathlib import Path
from typing import Sequence

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from autonomy_lab.gym.env import AgentGymEnv
from autonomy_lab.scenarios.config import SCENES


# 脚本位于 ``scripts/training/``；默认模型和日志仍相对项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m40.zip"
DEFAULT_SCENARIO = "rl_sanity"
DEFAULT_LOG_LABEL = "m40_training"


def additional_timesteps_to_target(
    current_timesteps: int,
    target_timesteps: int,
) -> int:
    """返回达到累计训练目标仍需执行的步数，绝不回退或重新初始化。"""
    if current_timesteps < 0 or target_timesteps < 0:
        raise ValueError("timesteps must be non-negative")
    return max(0, target_timesteps - current_timesteps)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析训练参数，同时保留 M4.0 的无参数默认调用。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=sorted(SCENES), default=DEFAULT_SCENARIO
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-timesteps", type=int, default=50_000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log-label", default=DEFAULT_LOG_LABEL)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--contact-penalty-per-step", type=float, default=0.0)
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--resume",
        action="store_true",
        help="load model-path and continue to the cumulative target",
    )
    start_group.add_argument(
        "--init-model-path",
        type=Path,
        help="initialize weights/optimizer from a checkpoint, then count training from zero",
    )
    args = parser.parse_args(argv)
    if args.target_timesteps <= 0:
        parser.error("--target-timesteps must be positive")
    if args.action_repeat <= 0:
        parser.error("--action-repeat must be positive")
    if args.contact_penalty_per_step > 0.0:
        parser.error("--contact-penalty-per-step must be zero or negative")
    if args.resume and not args.model_path.exists():
        parser.error(f"resume model does not exist: {args.model_path}")
    if args.init_model_path is not None and not args.init_model_path.exists():
        parser.error(f"initial model does not exist: {args.init_model_path}")
    return args


def main() -> None:
    """训练至累计 timestep 目标，并把模型及 Monitor episode reward 写入磁盘。"""
    args = parse_args()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / "experiments" / args.log_label / "train"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Monitor 提供 episode reward/length 证据；Recorder 留给固定 seed 性能评估。
    base_env = AgentGymEnv(
        scenario=args.scenario,
        render_mode=None,
        action_repeat=args.action_repeat,
        contact_penalty_per_step=args.contact_penalty_per_step,
    )
    env = Monitor(
        base_env,
        filename=str(log_path),
        override_existing=not args.resume,
    )
    try:
        if args.resume:
            # PPO.load 重建 Python 对象但恢复同一组权重与 optimizer state；不是新模型。
            model = PPO.load(args.model_path, env=env, device="cpu")
            current_timesteps = int(model.num_timesteps)
            reset_num_timesteps = False
        elif args.init_model_path is not None:
            # M4.1 继承 M4.0 的目标趋近权重与 optimizer state，但微调步数从零记录。
            model = PPO.load(args.init_model_path, env=env, device="cpu")
            model.set_random_seed(args.seed)
            current_timesteps = 0
            reset_num_timesteps = True
            print(f"Initialized PPO from: {args.init_model_path}")
        else:
            # 保留 SB3 PPO 的默认网络和近默认超参数，仅固定随机 seed 与 CPU。
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                verbose=1,
                device="cpu",
            )
            current_timesteps = 0
            reset_num_timesteps = True

        remaining = additional_timesteps_to_target(
            current_timesteps, args.target_timesteps
        )
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=reset_num_timesteps,
                log_interval=1,
            )
        model.save(args.model_path)
        print(
            f"Saved PPO model: {args.model_path} "
            f"(num_timesteps={model.num_timesteps})"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
