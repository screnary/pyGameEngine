"""运行一个不创建窗口的 Gymnasium 随机 Action smoke test。"""

from autonomy_lab.gym.env import AgentGymEnv

# 该模块通过 ``python -m scripts.gym_demo`` 从项目根目录运行。


def main() -> None:
    """用固定 seed 推进最多 1000 步，并打印终止时的精简摘要。"""
    env = AgentGymEnv(render_mode=None)
    try:
        observation, info = env.reset(seed=42)
        for step_index in range(1, 1001):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        print(
            "Gym smoke completed: "
            f"steps={step_index}, shape={observation.shape}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"time={info['simulation_time']:.3f}s, reward={reward:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
