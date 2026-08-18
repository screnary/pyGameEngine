"""定义 Agent 的运动状态、归一化控制接口和俯视图绘制。"""

import math

import pygame


class Agent:
    """由手动控制器和自主控制器共享的圆形 Agent。

    ``position`` 使用像素坐标，``heading`` 在内部使用弧度，``speed`` 使用
    像素/秒。碰撞半径 ``radius`` 与图片显示尺寸相互独立。
    """

    def __init__(
        self,
        position: tuple[float, float],
        heading_degrees: float = 0.0,
        initial_speed: float = 0.0,
        max_speed: float = 220.0,
        turn_speed_degrees: float = 150.0,
        radius: int = 16,
    ) -> None:
        # start_* 保存不可变的初始状态，R 重置时不必重新读取外部配置。
        self.start_position = pygame.Vector2(position)
        # 配置文件使用更直观的角度；运行时统一转换为弧度参与三角函数计算。
        self.start_heading = math.radians(heading_degrees)
        self.start_speed = initial_speed
        self.radius = radius
        self.turn_speed = math.radians(turn_speed_degrees)
        self.max_speed = max_speed
        self.reset()

    def reset(self) -> None:
        """恢复配置中的初始位置、朝向和速度。"""
        self.position = self.start_position.copy()
        self.heading = self.start_heading
        self.speed = self.start_speed

    def update_controls(self, dt: float, keys: pygame.key.ScancodeWrapper) -> pygame.Vector2:
        """把键盘状态转换为与 BT 相同的归一化命令。

        返回值是本帧建议位移，不直接修改位置；Environment 会在碰撞检测后
        决定是否真正接受该位移。
        """
        # 相反方向相减后自然得到 -1、0 或 +1，允许同时支持 WASD 和方向键。
        turn = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        throttle = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(
            keys[pygame.K_s] or keys[pygame.K_DOWN]
        )

        return self.update_motion(dt, throttle, turn)

    def update_motion(self, dt: float, throttle: float, turn: float) -> pygame.Vector2:
        """根据归一化命令更新朝向和速度，并返回本帧位移。

        ``throttle`` 和 ``turn`` 都限制到 [-1, 1]。方法更新运动学状态，但
        位置由 Environment 在碰撞约束下更新。
        """
        # 防止手动输入或未来控制器输出超过约定范围。
        throttle = max(-1.0, min(1.0, throttle))
        turn = max(-1.0, min(1.0, turn))

        # heading 统一使用弧度，取模后始终保持在 [0, 2π)。
        self.heading = (self.heading + turn * self.turn_speed * dt) % (2 * math.pi)
        self.speed = throttle * self.max_speed
        # heading=0 指向屏幕右侧；Pygame 的 y 正方向向下。
        forward = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        # 速度(像素/秒)乘以 dt(秒)得到本帧位移(像素)。
        return forward * self.speed * dt

    def draw(
        self,
        surface: pygame.Surface,
        image: pygame.Surface | None = None,
        fill_color: tuple[int, int, int] = (60, 170, 255),
        outline_color: tuple[int, int, int] = (220, 240, 255),
        heading_color: tuple[int, int, int] = (255, 245, 120),
    ) -> None:
        """在当前位置绘制图片或几何圆，并额外绘制朝向指示线。"""
        center = (round(self.position.x), round(self.position.y))

        if image is not None:
            size = self.radius * 3
            scaled = pygame.transform.smoothscale(image, (size, size))
            # PNG 仅是可视化；旋转不参与 Agent 状态或碰撞计算。
            angle = -math.degrees(self.heading) - 90.0
            rotated = pygame.transform.rotozoom(scaled, angle, 1.0)
            surface.blit(rotated, rotated.get_rect(center=center))
        else:
            # 没有可用图片时保留稳定的几何 fallback，仿真仍可运行。
            pygame.draw.circle(surface, fill_color, center, self.radius)
            pygame.draw.circle(surface, outline_color, center, self.radius, 2)

        # 朝向线始终绘制，便于确认图片旋转和真实 heading 是否一致。
        direction = pygame.Vector2(math.cos(self.heading), math.sin(self.heading))
        end = self.position + direction * (self.radius + 10)
        pygame.draw.line(surface, heading_color, center, end, 3)
