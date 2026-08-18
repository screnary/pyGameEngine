"""启动二维自主智能体实验，并管理 Pygame 主循环与 Episode 生命周期。

本文件只负责把各模块串起来：Environment 保存世界状态，Controller 产生
控制命令，ExperimentRecorder 记录结果，Pygame 负责事件和画面显示。
"""

import argparse

import pygame

from autonomy_lab.behavior_tree import PANEL_WIDTH, BehaviorTreeController
from autonomy_lab.environment import Environment
from autonomy_lab.experiment import ExperimentRecorder
from autonomy_lab.scene_config import DEFAULT_SCENARIO, SCENES, get_scene


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
    controller_id = "bt-v1" if controller is not None else "manual"

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

    # 必须先初始化 Pygame 和显示模式，PNG 的 convert_alpha() 才能正确工作。
    pygame.init()
    panel_width = PANEL_WIDTH if controller is not None else 0
    screen = pygame.display.set_mode(
        (environment.world_size[0] + panel_width, environment.world_size[1])
    )
    pygame.display.set_caption(f"Autonomy Lab - {environment.scene_name}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 26)

    running = True
    # 成功到达目标后保持窗口，但不再更新仿真；R 可以开始新的 Episode。
    episode_finished = False
    while running:
        # tick 返回上一帧耗时；上限避免窗口卡顿后单帧位移过大。
        dt = min(clock.tick(60) / 1000.0, 0.05)
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
                # 手动和 BT 最终都进入 Agent.update_motion()，动力学保持一致。
                environment.update(dt, pygame.key.get_pressed())
            else:
                # BT 只输出归一化命令；位置、朝向仍由 Environment/Agent 更新。
                turn, throttle = controller.tick(dt)
                environment.update_command(dt, throttle, turn)

            # 运动已经应用后再记录，因此 path_length 使用本帧的真实位移。
            recorder.update(
                dt,
                environment,
                active_action=(
                    controller.active_behavior if controller is not None else None
                ),
                bt_ticked=controller is not None,
            )

            distance_to_target = (
                environment.target - environment.agent.position
            ).length()
            # 到达条件属于实验终止标准，不属于某个具体 Behavior 的内部逻辑。
            if distance_to_target <= experiment_config["target_reached_distance"]:
                recorder.finish_episode("SUCCESS", "target_reached")
                episode_finished = True
            elif recorder.elapsed_time >= experiment_config["max_episode_time"]:
                recorder.finish_episode("TIMEOUT", "timeout")
                # TIMEOUT 用于批量实验，因此仍保持原有的自动退出语义。
                running = False

        # 无论 Episode 是否结束都持续绘制，使用户能观察最终状态。
        environment.draw(screen, font, clock.get_fps(), args.controller)
        if controller is not None:
            controller.draw_panel(screen, font, environment.world_size[0])
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
