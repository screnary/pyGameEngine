import random

import pygame

from .agent import Agent
from .assets import load_optional_image


class Environment:
    def __init__(self, scene_config: dict) -> None:
        self.scene_config = scene_config
        self.scene_name = str(scene_config["name"])
        self.world_size = tuple(scene_config["world_size"])
        self.seed = int(scene_config["seed"])
        self.random = random.Random(self.seed)
        self.display = scene_config["display"]
        self.images: dict[str, pygame.Surface | None] | None = None
        self.reset()

    def reset(self) -> None:
        self.random.seed(self.seed)
        agent_config = self.scene_config["agent"]
        target_config = self.scene_config["target"]
        self.agent = Agent(**agent_config)
        self.target = pygame.Vector2(target_config["position"])
        self.target_radius = int(target_config["radius"])
        self.obstacles = [pygame.Rect(*bounds) for bounds in self.scene_config["obstacles"]]
        self.collision_this_step = False

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper) -> None:
        movement = self.agent.update_controls(dt, keys)
        self._move_agent(movement)

    def update_command(self, dt: float, throttle: float, turn: float) -> None:
        """Advance simulation from a controller command independent of Pygame input."""
        movement = self.agent.update_motion(dt, throttle, turn)
        self._move_agent(movement)

    def _move_agent(self, movement: pygame.Vector2) -> None:
        # Resolve each axis independently so the circle can slide along walls.
        collided = False
        for axis in ("x", "y"):
            old_value = getattr(self.agent.position, axis)
            setattr(self.agent.position, axis, old_value + getattr(movement, axis))
            if self._agent_collides():
                setattr(self.agent.position, axis, old_value)
                collided = True
        self.collision_this_step = collided

    def _agent_collides(self) -> bool:
        radius = self.agent.radius
        position = self.agent.position

        if not (
            radius <= position.x <= self.world_size[0] - radius
            and radius <= position.y <= self.world_size[1] - radius
        ):
            return True

        for obstacle in self.obstacles:
            closest_x = max(obstacle.left, min(position.x, obstacle.right))
            closest_y = max(obstacle.top, min(position.y, obstacle.bottom))
            delta_x = position.x - closest_x
            delta_y = position.y - closest_y
            if delta_x * delta_x + delta_y * delta_y < radius * radius:
                return True
        return False

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        fps: float = 0.0,
        controller_name: str = "manual",
    ) -> None:
        if self.images is None:
            self.images = {
                name: load_optional_image(f"{name}.png")
                for name in ("agent", "target", "obstacle", "threat", "waypoint")
            }

        surface.fill(self.display["background_color"])

        for obstacle in self.obstacles:
            if self.images["obstacle"] is not None:
                obstacle_image = pygame.transform.smoothscale(
                    self.images["obstacle"], obstacle.size
                )
                surface.blit(obstacle_image, obstacle)
            else:
                pygame.draw.rect(
                    surface, self.display["obstacle_color"], obstacle, border_radius=4
                )
            pygame.draw.rect(surface, (170, 178, 192), obstacle, 2, border_radius=4)

        target_center = (round(self.target.x), round(self.target.y))
        if self.images["target"] is not None:
            target_size = self.target_radius * 3
            target_image = pygame.transform.smoothscale(
                self.images["target"], (target_size, target_size)
            )
            surface.blit(target_image, target_image.get_rect(center=target_center))
        else:
            pygame.draw.circle(
                surface, self.display["target_color"], target_center, self.target_radius
            )
            pygame.draw.circle(
                surface, (210, 255, 220), target_center, self.target_radius, 2
            )
            pygame.draw.line(
                surface,
                (20, 80, 35),
                (target_center[0] - 8, target_center[1]),
                (target_center[0] + 8, target_center[1]),
                2,
            )
            pygame.draw.line(
                surface,
                (20, 80, 35),
                (target_center[0], target_center[1] - 8),
                (target_center[0], target_center[1] + 8),
                2,
            )
        self.agent.draw(
            surface,
            image=self.images["agent"],
            fill_color=self.display["agent_color"],
        )

        heading_degrees = self.agent.heading * 180.0 / 3.141592653589793
        status_lines = [
            f"Scenario: {self.scene_name}    Controller: {controller_name.upper()}    Seed: {self.seed}",
            (
                "W/S or Up/Down: move    A/D or Left/Right: turn    R: reset"
                if controller_name == "manual"
                else "Behavior Tree autonomous control    R: reset"
            ),
            f"Position: ({self.agent.position.x:6.1f}, {self.agent.position.y:6.1f})",
            f"Heading: {heading_degrees:6.1f} deg    Speed: {self.agent.speed:6.1f}",
        ]
        if self.display["show_fps"]:
            status_lines[0] += f"    FPS: {fps:5.1f}"
        for index, text in enumerate(status_lines):
            label = font.render(text, True, self.display["text_color"])
            surface.blit(label, (16, 14 + index * 24))
