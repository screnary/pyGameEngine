"""维护唯一二维 World，并在固定仿真步内推进全部运行状态。

本模块允许使用 ``pygame.Vector2`` 和 ``pygame.Rect`` 进行几何计算，但不
初始化窗口，也不依赖 Pygame 的事件、绘制、图片、字体或实时时钟系统。
"""

from copy import deepcopy
import math
import random

import pygame

from .agent import Agent, AgentCommand
from ..perception.pygame_perception import AgentPerception


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
        self.scenario_metadata = deepcopy(
            scene_config.get(
                "research_metadata",
                {
                    "family": "fixed",
                    "seed": self.seed,
                    "dynamic_hazard_enabled": False,
                    "noise_level": 0.0,
                    "hazard_count": len(scene_config["obstacles"]),
                    "context_schedule": (),
                },
            )
        )
        # 使用局部随机数生成器，避免实验 seed 影响 Python 全局 random 状态。
        self.random = random.Random(self.seed)
        # 感知噪声使用独立 RNG，避免增加传感采样次数改变其他 World 随机序列。
        self.perception_random = random.Random(self.seed + 104729)
        self.reset()
        # World 持有唯一感知器。BT 与 Gym 都读取这个实例产生的同步快照。
        self.perception = AgentPerception(self)

    def reset(self, seed: int | None = None) -> None:
        """根据场景预设重新创建确定性的运行时状态。

        ``seed`` 只控制本次 World 的局部随机源；当前三个固定场景不会随机
        摆放实体，但保留该入口可让 Gym 的标准 seeding 与 World 对齐。
        """
        if seed is not None:
            self.seed = int(seed)
        self.random.seed(self.seed)
        self.perception_random.seed(self.seed + 104729)
        agent_config = self.scene_config["agent"]
        target_config = self.scene_config["target"]
        self.agent = Agent(**agent_config)
        self.target = pygame.Vector2(target_config["position"])
        self.target_radius = int(target_config["radius"])
        # 静态和动态 Hazard 最终都进入同一个 Rect 列表，现有碰撞、LOS、感知和
        # Renderer 因此自然读取动态对象的当前位置，而不需要第二套接口。
        self.obstacles = [
            pygame.Rect(*bounds) for bounds in self.scene_config["obstacles"]
        ]
        self.dynamic_hazards: list[dict[str, object]] = []
        for config in self.scene_config.get("dynamic_hazards", ()):
            width, height = config["size"]
            position = pygame.Vector2(config["position"])
            heading = math.radians(float(config["heading_degrees"]))
            speed = float(config["speed"])
            rect = pygame.Rect(0, 0, int(width), int(height))
            rect.center = (round(position.x), round(position.y))
            state: dict[str, object] = {
                "rect": rect,
                "position": position,
                "base_velocity": pygame.Vector2(
                    math.cos(heading) * speed,
                    math.sin(heading) * speed,
                ),
            }
            self.dynamic_hazards.append(state)
            self.obstacles.append(rect)

        self.context_schedule = tuple(self.scene_config.get("context_schedule", ()))
        self.current_context_phase = "stationary"
        self.current_hazard_speed_scale = 1.0
        self.current_noise_level = float(
            self.scene_config.get("perception_noise", {}).get(
                "hazard_range_std", 0.0
            )
        )
        self.collision_this_step = False
        self.simulation_time = 0.0
        self._update_context_phase()
        self._update_termination_state()
        # __init__ 首次 reset 时感知器尚未创建；后续 Episode reset 必须立即
        # 刷新快照，保证 Controller/Gym 在第一步前读到新的 Agent 状态。
        if hasattr(self, "perception"):
            self.perception.update()

    def step(self, command: AgentCommand, dt: float) -> None:
        """使用统一 Command 推进一次完整仿真步。

        顺序固定为：Command → Agent motion → collision/boundary resolution →
        simulation time/termination → perception。这样调用者在方法返回后读取的
        Observation 一定对应 Action 已执行完毕的最终状态。
        """
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if "turn" not in command or "throttle" not in command:
            raise ValueError("command must contain turn and throttle")

        movement = self.agent.update_motion(
            dt,
            float(command["throttle"]),
            float(command["turn"]),
        )
        self._move_dynamic_hazards(dt)
        self._move_agent(movement)
        self.simulation_time += dt
        self._update_context_phase()
        self._update_termination_state()
        self.perception.update()

    @property
    def dynamic_hazard_states(self) -> tuple[dict[str, object], ...]:
        """返回只含普通数值的当前动态 Hazard 诊断快照。"""
        states: list[dict[str, object]] = []
        for hazard in self.dynamic_hazards:
            position = hazard["position"]
            base_velocity = hazard["base_velocity"]
            if not isinstance(position, pygame.Vector2) or not isinstance(
                base_velocity, pygame.Vector2
            ):
                raise TypeError("dynamic hazard runtime state is invalid")
            velocity = base_velocity * self.current_hazard_speed_scale
            states.append(
                {
                    "position": (float(position.x), float(position.y)),
                    "velocity": (float(velocity.x), float(velocity.y)),
                    "speed": float(velocity.length()),
                    "heading": float(math.atan2(velocity.y, velocity.x)),
                }
            )
        return tuple(states)

    def _update_context_phase(self) -> None:
        """按 simulation time 选择当前简单 schedule phase。"""
        if not self.context_schedule:
            return
        phase = self.context_schedule[0]
        for candidate in self.context_schedule:
            if self.simulation_time >= float(candidate["start_time"]):
                phase = candidate
            else:
                break
        self.current_context_phase = str(phase["name"])
        self.current_hazard_speed_scale = float(phase["hazard_speed_scale"])
        self.current_noise_level = float(phase["noise_level"])

    def _move_dynamic_hazards(self, dt: float) -> None:
        """以恒速推进动态矩形，并在 World 边界执行轴向反射。"""
        width, height = self.world_size
        for hazard in self.dynamic_hazards:
            rect = hazard["rect"]
            position = hazard["position"]
            base_velocity = hazard["base_velocity"]
            if not isinstance(rect, pygame.Rect) or not isinstance(
                position, pygame.Vector2
            ) or not isinstance(base_velocity, pygame.Vector2):
                raise TypeError("dynamic hazard runtime state is invalid")

            velocity = base_velocity * self.current_hazard_speed_scale
            position += velocity * dt
            half_width = rect.width / 2.0
            half_height = rect.height / 2.0
            if position.x < half_width or position.x > width - half_width:
                base_velocity.x *= -1.0
                position.x = max(half_width, min(position.x, width - half_width))
            if position.y < half_height or position.y > height - half_height:
                base_velocity.y *= -1.0
                position.y = max(half_height, min(position.y, height - half_height))
            rect.center = (round(position.x), round(position.y))

    def _update_termination_state(self) -> None:
        """根据当前最终位置更新自然任务终止状态，不处理外部时间截断。"""
        reached_distance = float(
            self.scene_config["experiment"]["target_reached_distance"]
        )
        self.target_reached = (
            self.target - self.agent.position
        ).length() <= reached_distance

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
