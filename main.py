"""启动二维自主智能体实验，并管理 Pygame 主循环与 Episode 生命周期。

本文件只负责把各模块串起来：Environment 保存世界状态，Controller 产生
控制命令，ExperimentRecorder 记录结果，Pygame 负责事件和画面显示。
"""

import argparse

import pygame

from autonomy_lab.bt.controller import PANEL_WIDTH, BehaviorTreeController
from autonomy_lab.core.environment import Environment
from autonomy_lab.experiment.recorder import ExperimentRecorder
from autonomy_lab.rendering.renderer import PygameRenderer
from autonomy_lab.scenarios.config import DEFAULT_SCENARIO, SCENES, get_scene


SIMULATION_DT = 1.0 / 60.0


def manual_command(keys: pygame.key.ScancodeWrapper) -> dict[str, float]:
    """把键盘状态转换成与 BT/Gym 相同的 ``turn/throttle`` Command。"""
    # 相反方向相减后自然得到 -1、0 或 +1；World 不需要知道输入来自键盘。
    turn = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
        keys[pygame.K_a] or keys[pygame.K_LEFT]
    )
    throttle = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(
        keys[pygame.K_s] or keys[pygame.K_DOWN]
    )
    return {"turn": turn, "throttle": throttle}


def parse_args() -> argparse.Namespace:
    """解析启动参数。

    ``scenario`` 选择世界预设，``controller`` 决定手动或 BT 控制，``bt``
    只在 BT 模式下用于选择 ``bt_configs`` 中的 JSON 定义。
    """
    parser = argparse.ArgumentParser(description="Run a 2D agent experiment scene.")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENES),
        default=DEFAULT_SCENARIO,
        help="scene preset to run (default: %(default)s)",
    )
    parser.add_argument(
        "--controller",
        choices=("bt", "manual"),
        default="bt",
        help="control mode to run (default: %(default)s)",
    )
    parser.add_argument(
        "--bt",
        default="default",
        help="behavior-tree JSON config name for BT control (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """运行事件处理、仿真更新、实验记录和画面渲染循环。"""
    args = parse_args()

    # get_scene() 返回深拷贝，因此本次运行对配置的修改不会污染全局预设。
    environment = Environment(get_scene(args.scenario))

    # 手动模式不构建 BT；BT 模式通过 Loader 从 JSON 创建真正的 py_trees。
    controller = (
        BehaviorTreeController(environment, bt_config=args.bt)
        if args.controller == "bt"
        else None
    )
    controller_id = controller.controller_id if controller is not None else "manual"

    # Recorder 与 Controller 解耦：无论手动还是 BT，都记录相同的基础指标。
    recorder = ExperimentRecorder()
    recorder.start_episode(
        environment,
        args.scenario,
        controller_id,
        track_bt=controller is not None,
        bt_config_id=(controller.bt_config_id if controller is not None else None),
    )
    experiment_config = environment.scene_config["experiment"]

    panel_width = PANEL_WIDTH if controller is not None else 0
    renderer = PygameRenderer(environment, panel_width=panel_width)

    running = True
    # 成功到达目标后保持窗口，但不再更新仿真；R 可以开始新的 Episode。
    episode_finished = False
    while running:
        # 真实 FPS 只控制观看节奏；物理始终使用固定 SIMULATION_DT。
        renderer.pace(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 只结束仍在运行的 Episode；已成功保存的 Episode 不重复写入。
                if not episode_finished:
                    recorder.finish_episode("INTERRUPTED", "window_closed")
                running = False
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # 先封存旧 Episode，再重置环境和 BT，避免两次运行混入同一日志。
                if not episode_finished:
                    recorder.finish_episode("INTERRUPTED", "manual_reset")
                environment.reset()
                if controller is not None:
                    controller.reset()
                recorder.start_episode(
                    environment,
                    args.scenario,
                    controller_id,
                    track_bt=controller is not None,
                    bt_config_id=(
                        controller.bt_config_id if controller is not None else None
                    ),
                )
                episode_finished = False

        if not running:
            break

        # 成功后仅停止仿真，继续渲染结果，直到用户关闭或按 R 重开。
        if not episode_finished:
            if controller is None:
                command = manual_command(pygame.key.get_pressed())
            else:
                # BT 只输出归一化命令；位置、朝向仍由同一个 World 更新。
                turn, throttle = controller.tick(SIMULATION_DT)
                command = {"turn": turn, "throttle": throttle}

            environment.step(command, SIMULATION_DT)

            # 运动已经应用后再记录，因此 path_length 使用本帧的真实位移。
            recorder.update(
                SIMULATION_DT,
                environment,
                active_action=(
                    controller.active_behavior if controller is not None else None
                ),
                bt_ticked=controller is not None,
            )

            # World 在碰撞解决后更新自然终止状态，Recorder 只负责封存结果。
            if environment.target_reached:
                recorder.finish_episode("SUCCESS", "target_reached")
                episode_finished = True
            elif environment.simulation_time >= experiment_config["max_episode_time"]:
                recorder.finish_episode("TIMEOUT", "timeout")
                # TIMEOUT 用于批量实验，因此仍保持原有的自动退出语义。
                running = False

        # 无论 Episode 是否结束都持续绘制，使用户能观察最终状态。
        renderer.render(environment, controller, args.controller)

    renderer.close()


if __name__ == "__main__":
    main()
