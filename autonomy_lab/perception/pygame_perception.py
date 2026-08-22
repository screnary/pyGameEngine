"""从 Environment 当前真值同步生成供 Behavior 使用的只读感知快照。

这里的几何计算本身保持确定性。R0.4 research scene 可在 finite-range gate 后的
Hazard measurement 上加入 seed-controlled noise；不模拟延迟或跟踪器。Legacy
目标真值由 ``target_information_mode`` 控制，Research Goal 则始终服从有限距离。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pygame

from .semantic_perception import (
    AgentState,
    BoundaryPerception,
    GoalPerception,
    HazardObservation,
    HazardPerception,
    NavigationGap,
    SectorRange,
    SemanticPerception,
)

if TYPE_CHECKING:
    from ..core.environment import Environment


def normalise_angle(angle: float) -> float:
    """把任意弧度角包裹到 [-π, π)，便于计算最短转向误差。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sector_index_for_bearing(bearing: float, num_bins: int) -> int:
    """把相对方位映射到以 ``-π + i*step`` 为中心的最近 360° 扇区。

    正前方在偶数分箱中位于中间索引（16 bins 时为 8），``π`` 与 ``-π``
    都稳定回绕到索引 0。恰好落在两个中心之间时归入角度递增方向的扇区。
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    step = 2.0 * math.pi / num_bins
    position = (normalise_angle(bearing) + math.pi) / step
    return int(math.floor(position + 0.5)) % num_bins


# Historical import names remain aliases only; the real output model is the
# simulator-neutral SemanticPerception defined in semantic_perception.py.
PerceivedObstacle = HazardObservation
PerceivedGap = NavigationGap
SectorClearance = SectorRange
PerceptionSnapshot = SemanticPerception


class AgentPerception:
    """根据 Agent 位姿和场景几何生成最新 snapshot，不维护长期世界模型。"""

    VALID_TARGET_MODES = {"perceived", "ground_truth"}
    VALID_SENSING_PROFILES = {"legacy", "research"}
    SECTOR_COUNT = 12

    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        # sensor profile 先决定 Goal/Hazard 的 coverage；behavior_tree 中的 gap
        # 参数只负责把局部 free-space 解释为可通行开口。
        self.sensor_config = environment.scene_config["sensor"]
        # profile 缺省必须是 legacy，保证所有 M4/M5 固定场景与 checkpoint 兼容。
        self.sensing_profile = str(self.sensor_config.get("profile", "legacy"))
        if self.sensing_profile not in self.VALID_SENSING_PROFILES:
            raise ValueError("sensor profile must be 'legacy' or 'research'")
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
        self.goal_range = float(
            self.sensor_config.get("goal_range", self.sensor_range)
        )
        self.hazard_range = float(
            self.sensor_config.get("hazard_range", self.sensor_range)
        )
        self.goal_num_bins = int(self.sensor_config.get("goal_num_bins", 16))
        self.hazard_num_bins = int(
            self.sensor_config.get("hazard_num_bins", self.SECTOR_COUNT)
        )
        if self.goal_range <= 0.0 or self.hazard_range <= 0.0:
            raise ValueError("goal_range and hazard_range must be positive")
        if self.goal_num_bins <= 0 or self.hazard_num_bins <= 0:
            raise ValueError("goal_num_bins and hazard_num_bins must be positive")
        if self.sensing_profile == "research":
            # Gap/free-space 的派生射线也不得越过 Research hazard sensing 上限。
            self.sensor_range = self.hazard_range
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

    def observe(self) -> SemanticPerception:
        """实现统一 Provider contract，并保留 ``update()`` 作为 legacy API。"""
        return self.update()

    def update(self) -> SemanticPerception:
        """从当前 Environment 状态生成并保存一份全新的同步快照。"""
        agent = self.environment.agent
        # offset 是世界坐标向量；bearing 再减 heading 后变为 Agent 局部方位。
        offset = self.environment.target - agent.position
        distance = offset.length()
        absolute_bearing = math.atan2(offset.y, offset.x)
        bearing = normalise_angle(absolute_bearing - agent.heading)

        # Research Goal 是长程稳定任务信号：只按 360° finite range sensing，
        # 不读取 FOV、LOS 或 ground_truth 模式。Legacy 分支保持历史语义。
        if self.sensing_profile == "research":
            target_visible = distance <= self.goal_range
            reason = "" if target_visible else "out of range"
            target_available = target_visible
            source = "perception" if target_visible else None
            goal_sector = (
                sector_index_for_bearing(bearing, self.goal_num_bins)
                if target_visible
                else None
            )
        else:
            target_visible, reason = self._target_visibility(distance, bearing)
            target_available = (
                target_visible or self.target_information_mode == "ground_truth"
            )
            if self.target_information_mode == "ground_truth":
                source = "ground_truth"
            elif target_visible:
                source = "perception"
            else:
                source = None
            goal_sector = None

        # 三层职责保持显式：coverage 先筛出可感知对象；object semantics 提供
        # nearest clearance/bearing；sector 再派生 Action 使用的局部 free-space。
        visible_obstacles = self._visible_obstacles()
        sector_clearances = self._sector_clearances()
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

        width, height = self.environment.world_size
        radius = agent.radius
        nearest_hazard = visible_obstacles[0] if visible_obstacles else None
        # 一次性组装唯一语义快照；legacy 属性只映射这些值，不再次计算几何。
        self.snapshot = SemanticPerception(
            agent=AgentState(
                speed=float(agent.speed),
                heading=float(agent.heading),
                radius=float(agent.radius),
            ),
            goal=GoalPerception(
                sensed=target_visible,
                visible=target_visible,
                available=target_available,
                source=source,
                distance=distance if target_available else None,
                bearing=bearing if target_available else None,
                unavailable_reason=reason,
                sector_index=goal_sector,
            ),
            hazard=HazardPerception(
                visible_hazards=visible_obstacles,
                nearest_hazard=nearest_hazard,
                sector_ranges=sector_clearances,
                traversable_gaps=traversable_gaps,
                best_exploration_gap=best_exploration_gap,
                goal_direction_blocked=target_path_blocked,
                best_goal_gap=best_target_gap,
                sector_available=True,
                gaps_available=True,
            ),
            boundary=BoundaryPerception(
                left=float(agent.position.x - radius),
                right=float(width - radius - agent.position.x),
                top=float(agent.position.y - radius),
                bottom=float(height - radius - agent.position.y),
                available=True,
            ),
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

    def _visible_obstacles(self) -> tuple[HazardObservation, ...]:
        """执行 Hazard coverage，并生成按圆边缘距离排序的 object semantics。"""
        agent = self.environment.agent
        visible: list[HazardObservation] = []
        for obstacle in self.environment.obstacles:
            # 把 Agent 圆心坐标分别夹到 Rect 范围，得到矩形上的最近点。
            closest = pygame.Vector2(
                max(obstacle.left, min(agent.position.x, obstacle.right)),
                max(obstacle.top, min(agent.position.y, obstacle.bottom)),
            )
            offset = closest - agent.position
            center_distance = offset.length()
            direction = offset
            if not direction.length_squared():
                # 圆心恰在矩形内部/边缘时最近点方向为零，改用矩形中心定 bearing。
                direction = pygame.Vector2(obstacle.center) - agent.position
            bearing = normalise_angle(
                math.atan2(direction.y, direction.x) - agent.heading
            )
            # center_distance 再减 Agent radius，得到圆形碰撞体边缘真值净距离。
            # R0.4 noise 只扰动交给 Controller 的 Hazard measurement；Rect、
            # collision、gap geometry 和 Goal sensing 均不读取该随机值。Research
            # sector measurement 也在自己的 finite-range gate 后使用同一噪声源。
            true_clearance = max(0.0, center_distance - agent.radius)
            if self.sensing_profile == "research":
                # 必须先用无噪声真值做 finite-range gate；超距 Hazard 之后完全不
                # 进入 measurement 阶段，噪声不能把它“拉回”感知范围。
                if true_clearance > self.hazard_range:
                    continue
            else:
                if center_distance > self.sensor_range:
                    continue
                if abs(bearing) > self.half_fov:
                    continue
            measured_clearance = self._measure_hazard_clearance(
                true_clearance,
                max_distance=(
                    self.hazard_range
                    if self.sensing_profile == "research"
                    else None
                ),
            )
            visible.append(
                HazardObservation(
                    clearance=measured_clearance,
                    bearing=bearing,
                )
            )
        visible.sort(key=lambda item: item.distance)
        return tuple(visible)

    def _sector_clearances(self) -> tuple[SectorRange, ...]:
        """派生局部 footprint-aware free-space（legacy 12，Research 16 bins）。

        射线求交只是当前 Pygame 几何实现，不是公共语义中的“16 根传感器”。
        Sector 使用 Agent 真实半径而不叠加 gap safety margin，使 Action 获得周围
        哪个方向更易安全通过的连续物理净空；Condition 仍只读 nearest Hazard。
        """
        count = (
            self.hazard_num_bins
            if self.sensing_profile == "research"
            else self.SECTOR_COUNT
        )
        step = 2.0 * math.pi / count
        sectors: list[SectorRange] = []
        for index in range(count):
            bearing = -math.pi + index * step
            true_clearance = self._ray_free_distance(
                bearing,
                safety_margin=0.0,
                max_distance=(
                    self.hazard_range
                    if self.sensing_profile == "research"
                    else None
                ),
                include_boundary=self.sensing_profile != "research",
            )
            measured_clearance = true_clearance
            # 等于 max range 表示该射线未检测到 Hazard；只有真实命中才在 range
            # gate 之后加噪，避免超距对象或空扇区被噪声伪造成近距离 Hazard。
            if (
                self.sensing_profile == "research"
                and true_clearance < self.hazard_range
            ):
                measured_clearance = self._measure_hazard_clearance(
                    true_clearance,
                    max_distance=self.hazard_range,
                )
            sectors.append(
                SectorRange(
                    bearing=bearing,
                    clearance=measured_clearance,
                )
            )
        return tuple(sectors)

    def _measure_hazard_clearance(
        self,
        true_clearance: float,
        max_distance: float | None,
    ) -> float:
        """在完成真值 range gate 后加入可复现噪声，并裁剪到物理量程。"""
        measured = true_clearance
        noise_level = self.environment.current_noise_level
        if noise_level > 0.0:
            measured = max(
                0.0,
                true_clearance
                + self.environment.perception_random.gauss(0.0, noise_level),
            )
        if max_distance is not None:
            measured = min(measured, max_distance)
        return measured

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

    def _ray_free_distance(
        self,
        relative_bearing: float,
        safety_margin: float | None = None,
        max_distance: float | None = None,
        include_boundary: bool = True,
    ) -> float:
        """测量圆形 Agent 沿一条相对射线可安全行进的像素距离。"""
        agent = self.environment.agent
        start = agent.position
        direction = pygame.Vector2(
            math.cos(agent.heading + relative_bearing),
            math.sin(agent.heading + relative_bearing),
        )
        ray_range = self.sensor_range if max_distance is None else max_distance
        end = start + direction * ray_range
        # 将边界内缩、障碍物外扩同一 clearance，相当于让圆形 Agent 走点射线。
        margin = self.gap_safety_margin if safety_margin is None else safety_margin
        clearance = math.ceil(agent.radius + margin)
        free_distance = ray_range
        if include_boundary:
            # Legacy safety sector/gap 把 Boundary 作为可通行约束；Research Hazard
            # lidar 则显式跳过这里，由独立 BoundaryPerception 表达边界净空。
            safe_world = pygame.Rect(
                clearance,
                clearance,
                self.environment.world_size[0] - 2 * clearance,
                self.environment.world_size[1] - 2 * clearance,
            )
            clipped = safe_world.clipline((start.x, start.y), (end.x, end.y))
            if not clipped:
                return 0.0
            if not safe_world.collidepoint(end.x, end.y):
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
