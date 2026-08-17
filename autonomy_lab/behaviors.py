"""Task-specific py_trees nodes for Milestone 1."""

import math

import pygame
import py_trees

from .environment import Environment


def normalise_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class ObstacleNear(py_trees.behaviour.Behaviour):
    """Succeeds when an obstacle is close to the Agent's forward direction."""

    def __init__(
        self,
        environment: Environment,
        detection_distance: float,
        half_angle_degrees: float,
    ) -> None:
        super().__init__(name="ObstacleNear")
        self.visual_type = "condition"
        self.environment = environment
        self.detection_distance = detection_distance
        self.half_angle = math.radians(half_angle_degrees)
        self.obstacle: pygame.Rect | None = None

    def update(self) -> py_trees.common.Status:
        agent = self.environment.agent
        nearest: tuple[float, pygame.Rect] | None = None

        for obstacle in self.environment.obstacles:
            closest = pygame.Vector2(
                max(obstacle.left, min(agent.position.x, obstacle.right)),
                max(obstacle.top, min(agent.position.y, obstacle.bottom)),
            )
            offset = closest - agent.position
            clearance = offset.length() - agent.radius
            if clearance > self.detection_distance:
                continue

            direction = offset if offset.length_squared() else pygame.Vector2(
                obstacle.center
            ) - agent.position
            angle_error = normalise_angle(
                math.atan2(direction.y, direction.x) - agent.heading
            )
            if abs(angle_error) <= self.half_angle and (
                nearest is None or clearance < nearest[0]
            ):
                nearest = (clearance, obstacle)

        self.obstacle = nearest[1] if nearest else None
        return (
            py_trees.common.Status.SUCCESS
            if self.obstacle is not None
            else py_trees.common.Status.FAILURE
        )


class AvoidObstacle(py_trees.behaviour.Behaviour):
    """Issues a short deterministic turn away from the detected obstacle."""

    def __init__(
        self,
        environment: Environment,
        condition: ObstacleNear,
        command: dict[str, float],
        duration: float,
        throttle: float,
    ) -> None:
        super().__init__(name="AvoidObstacle")
        self.visual_type = "action"
        self.environment = environment
        self.condition = condition
        self.command = command
        self.duration = duration
        self.throttle = throttle
        self.dt = 0.0
        self.remaining = 0.0
        self.turn_direction = 1.0

    def initialise(self) -> None:
        self.remaining = self.duration
        obstacle = self.condition.obstacle
        if obstacle is None:
            self.turn_direction = 1.0
            return

        offset = pygame.Vector2(obstacle.center) - self.environment.agent.position
        relative_angle = normalise_angle(
            math.atan2(offset.y, offset.x) - self.environment.agent.heading
        )
        if abs(relative_angle) > math.radians(5.0):
            self.turn_direction = -1.0 if relative_angle > 0.0 else 1.0
        else:
            self.turn_direction = (
                1.0 if self.environment.random.random() >= 0.5 else -1.0
            )

    def update(self) -> py_trees.common.Status:
        self.command["turn"] = self.turn_direction
        self.command["throttle"] = self.throttle
        self.remaining -= self.dt
        return (
            py_trees.common.Status.SUCCESS
            if self.remaining <= 0.0
            else py_trees.common.Status.RUNNING
        )


class MoveToTarget(py_trees.behaviour.Behaviour):
    """Steers toward Target until the configured reached distance is met."""

    def __init__(
        self,
        environment: Environment,
        command: dict[str, float],
        reached_distance: float,
    ) -> None:
        super().__init__(name="MoveToTarget")
        self.visual_type = "action"
        self.environment = environment
        self.command = command
        self.reached_distance = reached_distance

    def update(self) -> py_trees.common.Status:
        offset = self.environment.target - self.environment.agent.position
        if offset.length() <= self.reached_distance:
            self.command["turn"] = 0.0
            self.command["throttle"] = 0.0
            return py_trees.common.Status.SUCCESS

        desired_heading = math.atan2(offset.y, offset.x)
        error = normalise_angle(desired_heading - self.environment.agent.heading)
        self.command["turn"] = max(-1.0, min(1.0, error / math.radians(45.0)))
        self.command["throttle"] = 1.0 if abs(error) < math.radians(60.0) else 0.3
        return py_trees.common.Status.RUNNING
