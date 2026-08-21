"""在固定 Hybrid BT 调度上下文中验证 PPONavigate 的训练接线。"""

import argparse
from pathlib import Path
from typing import Sequence

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from autonomy_lab.gym.hybrid_env import DEFAULT_TRAINING_SCENARIOS, HybridPPOEnv
from autonomy_lab.scenarios.config import SCENES
from scripts.training.train_ppo import additional_timesteps_to_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INIT_MODEL = PROJECT_ROOT / "models" / "ppo_m41b_control10hz.zip"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_m52_smoke.zip"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENES),
        default=list(DEFAULT_TRAINING_SCENARIOS),
    )
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument(
        "--target-timesteps",
        type=int,
        default=2_048,
        help=(
            "cumulative PPO decision-step target; defaults to a short Lab "
            "smoke run, and longer experiments must opt in explicitly"
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--init-model-path", type=Path, default=DEFAULT_INIT_MODEL
    )
    parser.add_argument("--log-label", default="m52_hybrid_smoke")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue model-path to the cumulative target instead of reinitializing",
    )
    args = parser.parse_args(argv)
    if args.target_timesteps <= 0:
        parser.error("--target-timesteps must be positive")
    if args.resume:
        if not args.model_path.exists():
            parser.error(f"resume model does not exist: {args.model_path}")
    elif not args.init_model_path.exists():
        parser.error(f"initial model does not exist: {args.init_model_path}")
    return args


def main() -> None:
    args = parse_args()
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / "experiments" / args.log_label / "train"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = HybridPPOEnv(scenarios=args.scenarios, render_mode=None)
    env = Monitor(
        base_env,
        filename=str(log_path),
        override_existing=not args.resume,
    )
    try:
        if args.resume:
            model = PPO.load(args.model_path, env=env, device="cpu")
            current_timesteps = int(model.num_timesteps)
            reset_num_timesteps = False
            print(f"Resuming Hybrid PPO: {args.model_path}")
        else:
            # 保留 M4.1b 已学习的导航权重与 PPO 配置，只更换 rollout Env。
            model = PPO.load(args.init_model_path, env=env, device="cpu")
            model.set_random_seed(args.seed)
            current_timesteps = 0
            reset_num_timesteps = True
            print(f"Initialized Hybrid PPO from: {args.init_model_path}")

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
            f"Saved Hybrid PPO model: {args.model_path} "
            f"(num_timesteps={model.num_timesteps})"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
