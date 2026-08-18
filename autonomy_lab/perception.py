"""从 Environment 当前真值同步生成供 Behavior 使用的只读感知快照。

这里的“感知”是科研原型中的确定性几何计算，不模拟噪声、延迟或跟踪器。
是否允许目标真值由 ``target_information_mode`` 单独控制。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from .environment import Environment


def normalise_angle(angle: float) -> float:
    """把任意弧度角包裹到 [-π, π)，便于计算最短转向误差。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class PerceivedObstacle:
    """一个进入传感器 FOV 的障碍物相对观测。

    ``distance`` 是 Agent 圆边缘到矩形最近点的像素距离；``bearing`` 是相对
    当前 heading 的有符号弧度，正值表示屏幕坐标系中的顺时针方向。
    """

    rect: pygame.Rect
    distance: float
    bearing: float


@dataclass(frozen=True)
class PerceivedGap:
    """由连续开放射线组成的局部缺口。

    ``entry_position`` 已转换成世界坐标，Action 可以在转向过程中保持同一个
    目标点，而不是每帧追逐不断变化的相对 bearing。
    """

    bearing: float
    free_distance: float
    angular_width: float
    entry_position: tuple[float, float]


@dataclass(frozen=True)
class PerceptionSnapshot:
    """一个 BT tick 内所有 Behavior 共享的不可变观测集合。

    目标不可用时 distance/bearing 为 ``None``，防止 perceived 模式意外泄漏
    Ground Truth。元组字段确保节点不会修改感知器产生的集合。
    """

    target_visible: bool
    target_available: bool
    target_source: str | None
    target_distance: float | None
    target_bearing: float | None
    target_unavailable_reason: str
    visible_obstacles: tuple[PerceivedObstacle, ...] = ()
    nearest_obstacle: PerceivedObstacle | None = None
    traversable_gaps: tuple[PerceivedGap, ...] = ()
    best_exploration_gap: PerceivedGap | None = None
    target_path_blocked: bool = False
    best_target_gap: PerceivedGap | None = None


class AgentPerception:
    """根据 Agent 位姿和场景几何生成最新 snapshot，不维护长期世界模型。"""

    VALID_TARGET_MODES = {"perceived", "ground_truth"}

    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        # sensor 决定目标与障碍物可见范围；behavior_tree 中的 gap 参数决定
        # 如何把有限条射线解释为可通行开口。
        self.sensor_config = environment.scene_config["sensor"]
        self.target_information_mode = environment.scene_config[
            "target_information_mode"
        ]
        # perceived 必须满足传感器约束；ground_truth 始终提供目标相对位置。
        if self.target_information_mode not in self.VALID_TARGET_MODES:
            raise ValueError(
                "target_information_mode must be 'perceived' or 'ground_truth'"
            )
        self.sensor_range = float(self.sensor_config["range"])
        self.half_fov = math.radians(float(self.sensor_config["fov_degrees"])) / 2.0
        self.los_enabled = bool(self.sensor_config["los_enabled"])
        gap_config = environment.scene_config["behavior_tree"]
        self.gap_ray_count = int(gap_config["gap_ray_count"])
        self.gap_min_travel_distance = float(
            gap_config["gap_min_travel_distance"]
        )
        self.gap_safety_margin = float(gap_config["gap_safety_margin"])
        self.gap_open_ratio = float(gap_config["gap_open_ratio"])
        self.gap_entry_ratio = float(gap_config["gap_entry_ratio"])
        self.gap_entry_reached_distance = float(
            gap_config["gap_entry_reached_distance"]
        )
        # 启动时集中验证配置，避免算法运行到一半才出现无意义几何结果。
        if self.gap_ray_count < 3:
            raise ValueError("gap_ray_count must be at least 3")
        if self.gap_min_travel_distance <= 0.0:
            raise ValueError("gap_min_travel_distance must be positive")
        if self.gap_safety_margin < 0.0:
            raise ValueError("gap_safety_margin must be non-negative")
        if not 0.0 < self.gap_open_ratio <= 1.0:
            raise ValueError("gap_open_ratio must be in (0, 1]")
        if not 0.0 < self.gap_entry_ratio < 1.0:
            raise ValueError("gap_entry_ratio must be in (0, 1)")
        if self.gap_entry_reached_distance <= 0.0:
            raise ValueError("gap_entry_reached_distance must be positive")
        # 创建后立即生成首份 snapshot，Behavior 在第一次 tick 前也可安全读取。
        self.snapshot = self.update()

    def update(self) -> PerceptionSnapshot:
        """从当前 Environment 状态生成并保存一份全新的同步快照。"""
        agent = self.environment.agent
        # offset 是世界坐标向量；bearing 再减 heading 后变为 Agent 局部方位。
        offset = self.environment.target - agent.position
        distance = offset.length()
        absolute_bearing = math.atan2(offset.y, offset.x)
        bearing = normalise_angle(absolute_bearing - agent.heading)

        # visible 描述传感器是否看到；available 描述 BT 是否获准使用目标信息。
        target_visible, reason = self._target_visibility(distance, bearing)
        # ground_truth 模式绕过可见性限制；perceived 模式只暴露传感器观测。
        target_available = (
            target_visible or self.target_information_mode == "ground_truth"
        )
        if self.target_information_mode == "ground_truth":
            source = "ground_truth"
        elif target_visible:
            source = "perception"
        else:
            source = None

        # 障碍物与缺口始终来自局部传感器几何，不因目标模式而改变。
        visible_obstacles = self._visible_obstacles()
        traversable_gaps = self._traversable_gaps()
        # 未知目标时优先选择更深且更接近正前方的局部开口。
        best_exploration_gap = (
            max(
                traversable_gaps,
                key=lambda gap: (
                    gap.free_distance,
                    -abs(gap.bearing),
                    -gap.bearing,
                ),
            )
            if traversable_gaps
            else None
        )
        # 只在目标落入当前 FOV 时比较目标射线的自由距离。
        target_path_blocked = bool(
            target_available
            and abs(bearing) <= self.half_fov
            and self._ray_free_distance(bearing) + 1.0
            < min(distance, self.sensor_range)
        )
        # 已知目标且直线路径受阻时，优先选择 bearing 最接近目标的安全缺口。
        best_target_gap = (
            min(
                traversable_gaps,
                key=lambda gap: (
                    abs(normalise_angle(gap.bearing - bearing)),
                    -gap.free_distance,
                    gap.bearing,
                ),
            )
            if target_path_blocked and traversable_gaps
            else None
        )

        # 一次性组装快照，保证同一 BT tick 中不同节点看到完全一致的数据。
        self.snapshot = PerceptionSnapshot(
            target_visible=target_visible,
            target_available=target_available,
            target_source=source,
            target_distance=distance if target_available else None,
            target_bearing=bearing if target_available else None,
            target_unavailable_reason=reason,
            visible_obstacles=visible_obstacles,
            nearest_obstacle=visible_obstacles[0] if visible_obstacles else None,
            traversable_gaps=traversable_gaps,
            best_exploration_gap=best_exploration_gap,
            target_path_blocked=target_path_blocked,
            best_target_gap=best_target_gap,
        )
        return self.snapshot

    def _target_visibility(self, distance: float, bearing: float) -> tuple[bool, str]:
        """依次检查目标距离、视场角和可选 LOS，并返回失败原因。"""
        if distance > self.sensor_range:
            return False, "out of range"
        if abs(bearing) > self.half_fov:
            return False, "outside FOV"
        if self.los_enabled and not self._target_line_of_sight_clear():
            return False, "occluded"
        return True, ""

    def _target_line_of_sight_clear(self) -> bool:
        """检查 Agent 到 Target 的线段是否被任一障碍矩形截断。"""
        start = self.environment.agent.position
        end = self.environment.target
        line = ((start.x, start.y), (end.x, end.y))
        return not any(obstacle.clipline(*line) for obstacle in self.environment.obstacles)

    def _visible_obstacles(self) -> tuple[PerceivedObstacle, ...]:
        """收集 FOV 内障碍物，并按 Agent 圆边缘距离从近到远排序。"""
        agent = self.environment.agent
        visible: list[PerceivedObstacle] = []
        for obstacle in self.environment.obstacles:
            # 把 Agent 圆心坐标分别夹到 Rect 范围，得到矩形上的最近点。
            closest = pygame.Vector2(
                max(obstacle.left, min(agent.position.x, obstacle.right)),
                max(obstacle.top, min(agent.position.y, obstacle.bottom)),
            )
            offset = closest - agent.position
            center_distance = offset.length()
            if center_distance > self.sensor_range:
                continue
            direction = offset
            if not direction.length_squared():
                # 圆心恰在矩形内部/边缘时最近点方向为零，改用矩形中心定 bearing。
                direction = pygame.Vector2(obstacle.center) - agent.position
            bearing = normalise_angle(
                math.atan2(direction.y, direction.x) - agent.heading
            )
            if abs(bearing) > self.half_fov:
                continue
            # center_distance 再减 Agent radius，得到圆形碰撞体边缘净距离。
            visible.append(
                PerceivedObstacle(
                    rect=obstacle,
                    distance=max(0.0, center_distance - agent.radius),
                    bearing=bearing,
                )
            )
        visible.sort(key=lambda item: item.distance)
        return tuple(visible)

    def _traversable_gaps(self) -> tuple[PerceivedGap, ...]:
        """在整个 FOV 均匀采样射线，并把连续开放射线合并为缺口。"""
        # ray_step 是相邻采样线的角度间隔；首尾线正好落在 FOV 两侧边界。
        ray_step = 2.0 * self.half_fov / (self.gap_ray_count - 1)
        samples = [
            (
                -self.half_fov + index * ray_step,
                self._ray_free_distance(-self.half_fov + index * ray_step),
            )
            for index in range(self.gap_ray_count)
        ]
        # 阈值同时要求绝对最小行程和相对当前最深射线的一定比例。
        # 这样短墙前方的“相对最好方向”不会被误当作真正可探索开口。
        opening_threshold = max(
            self.gap_min_travel_distance,
            max(distance for _, distance in samples) * self.gap_open_ratio,
        )

        # 连续超过阈值的射线组成一个候选缺口，而不是逐条射线决策。
        groups: list[list[tuple[float, float]]] = []
        current_group: list[tuple[float, float]] = []
        for sample in samples:
            if sample[1] >= opening_threshold:
                current_group.append(sample)
            elif current_group:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)

        gaps: list[PerceivedGap] = []
        for group in groups:
            # 用组首尾 bearing 的中点代表缺口，再对中线做一次精确射线查询。
            bearing = (group[0][0] + group[-1][0]) / 2.0
            free_distance = self._ray_free_distance(bearing)
            if free_distance < opening_threshold:
                continue
            # 加一个 ray_step 近似每个离散样本覆盖的角宽，并限制不超过 FOV。
            angular_width = min(
                2.0 * self.half_fov,
                group[-1][0] - group[0][0] + ray_step,
            )
            gaps.append(
                PerceivedGap(
                    bearing=bearing,
                    free_distance=free_distance,
                    angular_width=angular_width,
                    entry_position=self._gap_entry_position(
                        bearing, free_distance
                    ),
                )
            )
        return tuple(gaps)

    def _gap_entry_position(
        self, bearing: float, free_distance: float
    ) -> tuple[float, float]:
        """把相对缺口方向转换为固定的世界坐标入口点。"""
        # Action 会锁定此世界坐标，避免 Agent 转向后缺口方向随 FOV 抖动。
        absolute_bearing = self.environment.agent.heading + bearing
        # 不走满 free_distance，保留 entry_ratio 比例的余量，避免入口贴住障碍。
        entry = self.environment.agent.position + pygame.Vector2(
            math.cos(absolute_bearing), math.sin(absolute_bearing)
        ) * (free_distance * self.gap_entry_ratio)
        return float(entry.x), float(entry.y)

    def _ray_free_distance(self, relative_bearing: float) -> float:
        """测量圆形 Agent 沿一条相对射线可安全行进的像素距离。"""
        agent = self.environment.agent
        start = agent.position
        direction = pygame.Vector2(
            math.cos(agent.heading + relative_bearing),
            math.sin(agent.heading + relative_bearing),
        )
        end = start + direction * self.sensor_range
        # 将边界内缩、障碍物外扩同一 clearance，相当于让圆形 Agent 走点射线。
        clearance = math.ceil(agent.radius + self.gap_safety_margin)
        # 安全世界是原世界向内收缩 clearance 后的矩形。
        safe_world = pygame.Rect(
            clearance,
            clearance,
            self.environment.world_size[0] - 2 * clearance,
            self.environment.world_size[1] - 2 * clearance,
        )

        # clipline 返回射线位于安全世界内部的线段；为空表示起点已经不安全。
        clipped = safe_world.clipline((start.x, start.y), (end.x, end.y))
        if not clipped:
            return 0.0

        free_distance = self.sensor_range
        if not safe_world.collidepoint(end.x, end.y):
            # 射线先撞世界边界时，以裁剪线段终点作为当前上限。
            free_distance = min(
                free_distance,
                (pygame.Vector2(clipped[1]) - start).length(),
            )

        ray_end = start + direction * free_distance
        # Rect.inflate 接收总增量，所以宽高各增加 2*clearance，四边各扩一份。
        inflation = 2 * clearance
        for obstacle in self.environment.obstacles:
            hit = obstacle.inflate(inflation, inflation).clipline(
                (start.x, start.y), (ray_end.x, ray_end.y)
            )
            if hit:
                # hit[0] 是沿射线进入膨胀障碍物的第一个点，即安全距离终点。
                free_distance = min(
                    free_distance,
                    (pygame.Vector2(hit[0]) - start).length(),
                )
        return max(0.0, free_distance)
