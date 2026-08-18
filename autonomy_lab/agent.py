"""定义 Agent 的运动状态以及由所有 Controller 共享的归一化动力学。"""

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
