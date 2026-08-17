import argparse

import pygame

from autonomy_lab.behavior_tree import PANEL_WIDTH, BehaviorTreeController
from autonomy_lab.environment import Environment
from autonomy_lab.experiment import ExperimentRecorder
from autonomy_lab.scene_config import DEFAULT_SCENARIO, SCENES, get_scene


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = Environment(get_scene(args.scenario))
    controller = (
        BehaviorTreeController(environment) if args.controller == "bt" else None
    )
    controller_id = "bt-v1" if controller is not None else "manual"
    recorder = ExperimentRecorder()
    recorder.start_episode(
        environment, args.scenario, controller_id, track_bt=controller is not None
    )
    experiment_config = environment.scene_config["experiment"]

    pygame.init()
    panel_width = PANEL_WIDTH if controller is not None else 0
    screen = pygame.display.set_mode(
        (environment.world_size[0] + panel_width, environment.world_size[1])
    )
    pygame.display.set_caption(f"Autonomy Lab - {environment.scene_name}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 26)

    running = True
    episode_finished = False
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)  # delta time
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if not episode_finished:
                    recorder.finish_episode("INTERRUPTED", "window_closed")
                running = False
                break
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
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
                )
                episode_finished = False

        if not running:
            break

        if not episode_finished:
            if controller is None:
                environment.update(dt, pygame.key.get_pressed())
            else:
                turn, throttle = controller.tick(dt)
                environment.update_command(dt, throttle, turn)

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
            if distance_to_target <= experiment_config["target_reached_distance"]:
                recorder.finish_episode("SUCCESS", "target_reached")
                episode_finished = True
            elif recorder.elapsed_time >= experiment_config["max_episode_time"]:
                recorder.finish_episode("TIMEOUT", "timeout")
                running = False

        environment.draw(screen, font, clock.get_fps(), args.controller)
        if controller is not None:
            controller.draw_panel(screen, font, environment.world_size[0])
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
