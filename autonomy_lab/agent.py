import math

import pygame


class Agent:
    """A keyboard-controlled circular agent."""

    def __init__(
        self,
        position: tuple[float, float],
        heading_degrees: float = 0.0,
        initial_speed: float = 0.0,
        max_speed: float = 220.0,
        turn_speed_degrees: float = 150.0,
        radius: int = 16,
    ) -> None:
        self.start_position = pygame.Vector2(position)
        self.start_heading = math.radians(heading_degrees)
        self.start_speed = initial_speed
        self.radius = radius
        self.turn_speed = math.radians(turn_speed_degrees)
        self.max_speed = max_speed
        self.reset()

    def reset(self) -> None:
        self.position = self.start_position.copy()
        self.heading = self.start_heading
        self.speed = self.start_speed

    def update_controls(self, dt: float, keys: pygame.key.ScancodeWrapper) -> pygame.Vector2:
        turn = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        throttle = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(
            keys[pygame.K_s] or keys[pygame.K_DOWN]
        )

        return self.update_motion(dt, throttle, turn)

    def update_motion(self, dt: float, throttle: float, turn: float) -> pygame.Vector2:
        """Advance heading and return displacement for a normalised command."""
        throttle = max(-1.0, min(1.0, throttle))
        turn = max(-1.0, min(1.0, turn))

        self.heading = (self.heading + turn * self.turn_speed * dt) % (2 * math.pi)
        self.speed = throttle * self.max_speed
        forward = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        return forward * self.speed * dt

    def draw(
        self,
        surface: pygame.Surface,
        image: pygame.Surface | None = None,
        fill_color: tuple[int, int, int] = (60, 170, 255),
        outline_color: tuple[int, int, int] = (220, 240, 255),
        heading_color: tuple[int, int, int] = (255, 245, 120),
    ) -> None:
        center = (round(self.position.x), round(self.position.y))

        if image is not None:
            size = self.radius * 3
            scaled = pygame.transform.smoothscale(image, (size, size))
            angle = -math.degrees(self.heading) - 90.0
            rotated = pygame.transform.rotozoom(scaled, angle, 1.0)
            surface.blit(rotated, rotated.get_rect(center=center))
        else:
            pygame.draw.circle(surface, fill_color, center, self.radius)
            pygame.draw.circle(surface, outline_color, center, self.radius, 2)

        direction = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        end = self.position + direction * (self.radius + 10)
        pygame.draw.line(surface, heading_color, center, end, 3)
