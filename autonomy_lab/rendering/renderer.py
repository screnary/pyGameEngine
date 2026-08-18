"""集中管理 Pygame 显示资源，并只读绘制当前 World State。"""

import math
from typing import TYPE_CHECKING

import pygame

from .assets import load_optional_image

if TYPE_CHECKING:
    from ..bt.controller import BehaviorTreeController
    from ..environment import Environment


class PygameRenderer:
    """Pygame 窗口、素材、字体和观看帧率的唯一拥有者。

    Renderer 不调用 ``Environment.step()``，也不修改 Agent、碰撞、感知或
    Episode 状态。``pace()`` 只延迟真实墙钟时间，不提供仿真 ``dt``。
    """

    def __init__(
        self,
        environment: "Environment",
        panel_width: int = 0,
        font_size: int = 26,
    ) -> None:
        pygame.init()
        self.panel_width = panel_width
        self.screen = pygame.display.set_mode(
            (environment.world_size[0] + panel_width, environment.world_size[1])
        )
        pygame.display.set_caption(f"Autonomy Lab - {environment.scene_name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, font_size)
        # convert_alpha() 需要已经初始化的显示模式，因此图片在窗口之后加载。
        self.images = {
            name: load_optional_image(f"{name}.png")
            for name in ("agent", "target", "obstacle", "threat", "waypoint")
        }

    @property
    def fps(self) -> float:
        """返回只用于界面显示的真实渲染帧率。"""
        return self.clock.get_fps()

    def pace(self, fps: int = 60) -> None:
        """限制 human 显示速度；返回耗时绝不能作为 Simulation dt。"""
        self.clock.tick(fps)

    def render(
        self,
        environment: "Environment",
        controller: "BehaviorTreeController | None" = None,
        controller_name: str = "manual",
    ) -> None:
        """绘制一个只读快照，并在 BT 模式下附加 Runtime Tree 面板。"""
        display = environment.scene_config["display"]
        self.screen.fill(display["background_color"])
        self._draw_fov(environment)
        self._draw_obstacles(environment)
        self._draw_target(environment)
        self._draw_agent(environment)
        self._draw_status(environment, controller_name)
        if controller is not None:
            controller.draw_panel(self.screen, self.font, environment.world_size[0])
        # Gym human 模式没有 main.py 的事件循环；pump 只维持窗口响应，不改 World。
        pygame.event.pump()
        pygame.display.flip()

    def close(self) -> None:
        """释放本 Renderer 创建的显示资源。"""
        pygame.display.quit()
        pygame.quit()

    def _draw_obstacles(self, environment: "Environment") -> None:
        display = environment.scene_config["display"]
        for obstacle in environment.obstacles:
            # 图片只适配碰撞 Rect 的显示尺寸，不反向改变碰撞模型。
            if self.images["obstacle"] is not None:
                obstacle_image = pygame.transform.smoothscale(
                    self.images["obstacle"], obstacle.size
                )
                self.screen.blit(obstacle_image, obstacle)
            else:
                pygame.draw.rect(
                    self.screen,
                    display["obstacle_color"],
                    obstacle,
                    border_radius=4,
                )
            pygame.draw.rect(
                self.screen, (170, 178, 192), obstacle, 2, border_radius=4
            )

    def _draw_target(self, environment: "Environment") -> None:
        display = environment.scene_config["display"]
        center = (round(environment.target.x), round(environment.target.y))
        if self.images["target"] is not None:
            size = environment.target_radius * 3
            image = pygame.transform.smoothscale(
                self.images["target"], (size, size)
            )
            self.screen.blit(image, image.get_rect(center=center))
            return

        pygame.draw.circle(
            self.screen, display["target_color"], center, environment.target_radius
        )
        pygame.draw.circle(
            self.screen, (210, 255, 220), center, environment.target_radius, 2
        )
        pygame.draw.line(
            self.screen,
            (20, 80, 35),
            (center[0] - 8, center[1]),
            (center[0] + 8, center[1]),
            2,
        )
        pygame.draw.line(
            self.screen,
            (20, 80, 35),
            (center[0], center[1] - 8),
            (center[0], center[1] + 8),
            2,
        )

    def _draw_agent(self, environment: "Environment") -> None:
        agent = environment.agent
        display = environment.scene_config["display"]
        center = (round(agent.position.x), round(agent.position.y))
        image = self.images["agent"]
        if image is not None:
            size = agent.radius * 3
            scaled = pygame.transform.smoothscale(image, (size, size))
            # PNG 朝上为零方向，而 World heading=0 指向右侧，所以额外减 90°。
            rotated = pygame.transform.rotozoom(
                scaled, -math.degrees(agent.heading) - 90.0, 1.0
            )
            self.screen.blit(rotated, rotated.get_rect(center=center))
        else:
            pygame.draw.circle(
                self.screen, display["agent_color"], center, agent.radius
            )
            pygame.draw.circle(self.screen, (220, 240, 255), center, agent.radius, 2)

        direction = pygame.Vector2(math.cos(agent.heading), math.sin(agent.heading))
        end = agent.position + direction * (agent.radius + 10)
        pygame.draw.line(self.screen, (255, 245, 120), center, end, 3)

    def _draw_status(
        self, environment: "Environment", controller_name: str
    ) -> None:
        display = environment.scene_config["display"]
        heading_degrees = math.degrees(environment.agent.heading)
        status_lines = [
            (
                f"Scenario: {environment.scene_name}    "
                f"Controller: {controller_name.upper()}    Seed: {environment.seed}"
            ),
            (
                "W/S or Up/Down: move    A/D or Left/Right: turn    R: reset"
                if controller_name == "manual"
                else "Behavior Tree autonomous control    R: reset"
            ),
            (
                f"Position: ({environment.agent.position.x:6.1f}, "
                f"{environment.agent.position.y:6.1f})"
            ),
            (
                f"Heading: {heading_degrees:6.1f} deg    "
                f"Speed: {environment.agent.speed:6.1f}"
            ),
        ]
        if display["show_fps"]:
            status_lines[0] += f"    FPS: {self.fps:5.1f}"
        for index, text in enumerate(status_lines):
            label = self.font.render(text, True, display["text_color"])
            self.screen.blit(label, (16, 14 + index * 24))

    def _draw_fov(self, environment: "Environment") -> None:
        """绘制感知配置的扇形提示；真实感知仍由 AgentPerception 计算。"""
        sensor = environment.scene_config.get("sensor")
        if not sensor:
            return
        sensor_range = float(sensor["range"])
        half_fov = math.radians(float(sensor["fov_degrees"])) / 2.0
        center = environment.agent.position
        points = [(round(center.x), round(center.y))]
        segments = 24
        for index in range(segments + 1):
            angle = (
                environment.agent.heading
                - half_fov
                + 2.0 * half_fov * index / segments
            )
            endpoint = center + pygame.Vector2(
                math.cos(angle), math.sin(angle)
            ) * sensor_range
            points.append((round(endpoint.x), round(endpoint.y)))

        overlay = pygame.Surface(environment.world_size, pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (70, 155, 255, 30), points)
        pygame.draw.lines(overlay, (90, 175, 255, 75), False, points, 1)
        self.screen.blit(overlay, (0, 0))
