"""Behavior Tree construction, ticking, and runtime access."""

import pygame
import py_trees

from .behaviors import AvoidObstacle, MoveToTarget, ObstacleNear
from .bt_visualizer import BTVisualizer
from .environment import Environment


PANEL_WIDTH = 480

class BehaviorTreeController:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.command = {"turn": 0.0, "throttle": 0.0}
        config = environment.scene_config["behavior_tree"]

        self.obstacle_near = ObstacleNear(
            environment,
            config["obstacle_detection_distance"],
            config["obstacle_detection_half_angle_degrees"],
        )
        self.avoid_obstacle = AvoidObstacle(
            environment,
            self.obstacle_near,
            self.command,
            config["avoid_duration"],
            config["avoid_throttle"],
        )
        self.move_to_target = MoveToTarget(
            environment, self.command, config["target_reached_distance"]
        )

        self.obstacle_avoidance = py_trees.composites.Sequence(
            name="ObstacleAvoidance",
            memory=True,
            children=[self.obstacle_near, self.avoid_obstacle],
        )
        self.root = py_trees.composites.Selector(
            name="Selector",
            memory=False,
            children=[self.obstacle_avoidance, self.move_to_target],
        )
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.snapshot = py_trees.visitors.SnapshotVisitor()
        self.tree.visitors.append(self.snapshot)
        self.visualizer = BTVisualizer(self.tree.root, self.snapshot)
        self.tick_count = 0

    def tick(self, dt: float) -> tuple[float, float]:
        self.command["turn"] = 0.0
        self.command["throttle"] = 0.0
        self.avoid_obstacle.dt = dt
        self.tree.tick()
        self.tick_count += 1
        return self.command["turn"], self.command["throttle"]

    def reset(self) -> None:
        self.command["turn"] = 0.0
        self.command["throttle"] = 0.0
        self.tree.root.stop(py_trees.common.Status.INVALID)
        self.snapshot.visited = {}
        self.snapshot.previously_visited = {}
        self.snapshot.changed = False
        self.tick_count = 0

    @property
    def active_behavior(self) -> str:
        if self.avoid_obstacle.status == py_trees.common.Status.RUNNING:
            return self.avoid_obstacle.name
        if self.move_to_target.status == py_trees.common.Status.RUNNING:
            return self.move_to_target.name
        if self.move_to_target.status == py_trees.common.Status.SUCCESS:
            return "Target Reached"
        return "None"

    @property
    def active_action_label(self) -> str:
        labels = {
            self.avoid_obstacle.name: "Avoid Obstacle",
            self.move_to_target.name: "Move To Target",
            "Target Reached": "Target Reached",
            "None": "None",
        }
        return labels[self.active_behavior]

    @property
    def decision_label(self) -> str:
        if self.obstacle_near.status == py_trees.common.Status.SUCCESS:
            return "Obstacle detected"
        if self.obstacle_near.status == py_trees.common.Status.FAILURE:
            return "No obstacle nearby"
        return "Awaiting condition"

    def draw_panel(
        self, surface: pygame.Surface, font: pygame.font.Font, world_width: int
    ) -> None:
        summaries = [
            f"Active Action: {self.active_action_label}",
            f"Decision: {self.decision_label}",
            f"Tick: {self.tick_count}",
        ]
        self.visualizer.draw(surface, font, world_width, summaries)
