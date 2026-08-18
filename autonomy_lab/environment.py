"""维护场景状态，并负责碰撞约束下的运动、重置和绘制。"""

import math
import random

import pygame

from .agent import Agent
from .assets import load_optional_image


class Environment:
    """拥有一次 Episode 中所有可变的二维世界状态。

    Controller 不能直接改 ``Agent.position``，而是提交 turn/throttle；
    Environment 将其转换为位移并执行边界与障碍物碰撞检查。
    """

    def __init__(self, scene_config: dict) -> None:
        # scene_config 是 get_scene() 的独立副本，可作为本 Episode 的配置源。
        self.scene_config = scene_config
        self.scene_name = str(scene_config["name"])
        self.world_size = tuple(scene_config["world_size"])
        self.seed = int(scene_config["seed"])
        # 使用局部随机数生成器，避免实验 seed 影响 Python 全局 random 状态。
        self.random = random.Random(self.seed)
        self.display = scene_config["display"]
        # None 表示尚未尝试加载；字典中的 None 表示该项素材加载失败。
        self.images: dict[str, pygame.Surface | None] | None = None
        self.reset()

    def reset(self) -> None:
        """根据场景预设重新创建确定性的运行时状态。

        Agent、Target 和障碍物都会恢复初始值；已加载图片可继续复用。
        """
        self.random.seed(self.seed)
        agent_config = self.scene_config["agent"]
        target_config = self.scene_config["target"]
        self.agent = Agent(**agent_config)
        self.target = pygame.Vector2(target_config["position"])
        self.target_radius = int(target_config["radius"])
        # 配置使用普通元组，运行时转换为 Pygame Rect 便于碰撞与绘制。
        self.obstacles = [pygame.Rect(*bounds) for bounds in self.scene_config["obstacles"]]
        self.collision_this_step = False

    def update(self, dt: float, keys: pygame.key.ScancodeWrapper) -> None:
        """推进一帧手动控制仿真。"""
        movement = self.agent.update_controls(dt, keys)
        self._move_agent(movement)

    def update_command(self, dt: float, throttle: float, turn: float) -> None:
        """根据 Controller 命令推进一帧，与键盘输入完全解耦。"""
        movement = self.agent.update_motion(dt, throttle, turn)
        self._move_agent(movement)

    def _move_agent(self, movement: pygame.Vector2) -> None:
        """尝试应用位移，并记录本帧是否发生碰撞。"""
        # 分轴尝试位移：某一轴碰撞时回退该轴，另一轴仍可沿墙滑动。
        collided = False
        for axis in ("x", "y"):
            old_value = getattr(self.agent.position, axis)
            setattr(self.agent.position, axis, old_value + getattr(movement, axis))
            if self._agent_collides():
                setattr(self.agent.position, axis, old_value)
                collided = True
        self.collision_this_step = collided

    def _agent_collides(self) -> bool:
        """检查 Agent 圆形碰撞体是否越界或进入任一矩形障碍物。"""
        radius = self.agent.radius
        position = self.agent.position

        # 圆心至少离世界四边一个 radius，才能保证整个圆留在场景中。
        if not (
            radius <= position.x <= self.world_size[0] - radius
            and radius <= position.y <= self.world_size[1] - radius
        ):
            return True

        for obstacle in self.obstacles:
            # 矩形上离圆心最近的点，用于精确的圆-矩形碰撞判定。
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
        """按 FOV、障碍物、目标、Agent、文字的顺序绘制当前场景。"""
        if self.images is None:
            # 首次绘制时一次性尝试素材；None 会自然走几何图形 fallback。
            self.images = {
                name: load_optional_image(f"{name}.png")
                for name in ("agent", "target", "obstacle", "threat", "waypoint")
            }

        # 先清屏再画半透明 FOV，之后实体会覆盖 FOV，层次更清晰。
        surface.fill(self.display["background_color"])
        self._draw_fov(surface)

        for obstacle in self.obstacles:
            # 图片只按碰撞 Rect 缩放显示；碰撞模型仍然是原始 Rect。
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

        # Target 目前只有位置和显示半径，不参与障碍物碰撞。
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

        # 内部 heading 为弧度，界面转换为更适合人阅读的角度。
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

    def _draw_fov(self, surface: pygame.Surface) -> None:
        """绘制传感器视场扇形；该图层不参与真实感知计算。"""
        sensor = self.scene_config.get("sensor")
        if not sensor:
            return
        sensor_range = float(sensor["range"])
        half_fov = math.radians(float(sensor["fov_degrees"])) / 2.0
        center = self.agent.position
        # 多边形第一个点是 Agent 圆心，后续点沿扇形外弧均匀采样。
        points = [(round(center.x), round(center.y))]
        segments = 24
        for index in range(segments + 1):
            angle = self.agent.heading - half_fov + 2.0 * half_fov * index / segments
            endpoint = center + pygame.Vector2(math.cos(angle), math.sin(angle)) * sensor_range
            points.append((round(endpoint.x), round(endpoint.y)))

        # 单独透明 Surface 可避免直接在主画面绘制时丢失 alpha 混合。
        overlay = pygame.Surface(self.world_size, pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (70, 155, 255, 30), points)
        pygame.draw.lines(overlay, (90, 175, 255, 75), False, points, 1)
        surface.blit(overlay, (0, 0))
