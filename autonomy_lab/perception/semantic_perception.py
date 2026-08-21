"""定义与具体模拟器无关的语义感知数据结构。

本模块只负责描述“感知结果是什么”，不负责从 World 中计算这些结果。真正的
range、FOV、LOS、障碍射线和边界几何计算仍由 ``pygame_perception.AgentPerception``
完成，然后一次性组装为本模块中的 ``SemanticPerception``。

所有 dataclass 都使用 ``frozen=True``，并且字段仅包含普通 Python 标量、元组
或其他 frozen dataclass。这里刻意不保存 ``pygame.Rect``、World、Surface 或
Agent 对象，使同一语义接口以后可以接收其他模拟器产生的数据。
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentState:
    """Controller 和 Observation 所需的最小 Agent 自身状态。

    Attributes:
        speed: 当前线速度，单位为像素/秒；允许为负值表示倒车。
        heading: 当前世界朝向，单位为弧度。

    这里只复制必要数值，不把完整 Agent 对象带入感知层。
    """

    speed: float
    heading: float
    # 碰撞半径是跨 simulator 可表达的普通标量，Safety Action 用它解释净空。
    radius: float = 0.0


@dataclass(frozen=True)
class GoalPerception:
    """描述当前传感器或实验模式允许 Controller 知道的任务目标信息。

    Attributes:
        sensed: Goal 是否满足当前传感器的 range、FOV 和光学 LOS 条件。
        visible: 兼容现有视觉可见性语义；当前实现中与 ``sensed`` 相同。
        available: Controller 是否可以使用距离和方位。在 ground-truth 模式下，
            即使 ``sensed`` 为 False，本字段仍可能为 True。
        source: 信息来源，例如 ``perception`` 或 ``ground_truth``；不可用时为 None。
        distance: Agent 圆心到 Goal 的距离，单位为像素；不可用时为 None。
        bearing: Goal 相对当前 heading 的有符号方位，单位为弧度。
        unavailable_reason: 不可感知原因，例如超出 range、FOV 或被遮挡。

    Goal 只回答“目标在哪里、是否被感知”，不回答 Agent 身体能否沿该方向通过。
    footprint-aware 可通行性属于 ``HazardPerception``，两种语义不得混合。
    """

    sensed: bool
    visible: bool
    available: bool
    source: str | None
    distance: float | None
    bearing: float | None
    unavailable_reason: str


@dataclass(frozen=True)
class HazardObservation:
    """一个局部可感知 Hazard 的相对观测。

    ``clearance`` 是 Agent 圆形碰撞体边缘到 Hazard 最近点的净距离，单位为像素；
    ``bearing`` 是相对当前 heading 的有符号弧度。该对象不保存原始矩形或其他
    World geometry。
    """

    clearance: float
    bearing: float

    @property
    def distance(self) -> float:
        """兼容旧代码的只读名称；直接返回已计算的 ``clearance``。"""
        return self.clearance


@dataclass(frozen=True)
class SectorRange:
    """一个 Agent 相对方向上的 footprint-aware 可安全行进距离。

    ``bearing`` 单位为弧度；``clearance`` 单位为像素，并已同时考虑 Agent 半径、
    局部 Hazard 和 World Boundary。它描述 free-space，不参与 Goal 是否可见的判断。
    """

    bearing: float
    clearance: float


@dataclass(frozen=True)
class NavigationGap:
    """局部可通行开口的纯数值描述。

    ``bearing`` 和 ``angular_width`` 使用弧度，``free_distance`` 使用像素；
    ``entry_position`` 是固定的世界坐标元组。Action 可以锁定该入口，而不需要
    持有 pygame Vector2 或每帧追逐不断变化的相对方位。
    """

    bearing: float
    free_distance: float
    angular_width: float
    entry_position: tuple[float, float]


@dataclass(frozen=True)
class HazardPerception:
    """供局部导航安全决策使用的 Hazard 与 free-space 信息。

    ``visible_hazards`` 保存传感器范围内的相对 Hazard 观测；``sector_ranges``
    保存 R0.1 的全方向安全净空；gap 字段保存由连续开放射线得到的局部入口。
    ``goal_direction_blocked`` 只表示 Goal 方向的路径净空不足，不会反过来改变
    ``GoalPerception.sensed`` 或 ``GoalPerception.visible``。
    """

    visible_hazards: tuple[HazardObservation, ...] = ()
    nearest_hazard: HazardObservation | None = None
    sector_ranges: tuple[SectorRange, ...] = ()
    traversable_gaps: tuple[NavigationGap, ...] = ()
    best_exploration_gap: NavigationGap | None = None
    goal_direction_blocked: bool = False
    best_goal_gap: NavigationGap | None = None
    # Sector 属于 research core sensing；gap 是 Pygame navigation 的可选派生量。
    # 未来 Adapter 缺少对应能力时应显式标记 False，而不是伪造全零数据。
    sector_available: bool = False
    gaps_available: bool = False

    @property
    def available(self) -> bool:
        """当前是否至少有一个可感知的局部 Hazard。"""
        return self.nearest_hazard is not None

    @property
    def nearest_clearance(self) -> float | None:
        """返回最近 Hazard 的像素净空；没有 Hazard 时返回 None。"""
        return (
            self.nearest_hazard.clearance
            if self.nearest_hazard is not None
            else None
        )

    @property
    def nearest_bearing(self) -> float | None:
        """返回最近 Hazard 的相对弧度方位；没有 Hazard 时返回 None。"""
        return (
            self.nearest_hazard.bearing
            if self.nearest_hazard is not None
            else None
        )


@dataclass(frozen=True)
class BoundaryPerception:
    """Agent 圆形碰撞体到四条 World Boundary 的净空，单位均为像素。

    这些值已经扣除 Agent 半径，因此 0 表示碰撞体恰好接触边界，负值表示当前
    状态已经越过安全边界。
    """

    left: float
    right: float
    top: float
    bottom: float
    # 某些 simulator 没有矩形 World boundary；此时四个数值只是占位且不得使用。
    available: bool = True


@dataclass(frozen=True)
class SemanticPerception:
    """一个同步 simulation state 对应的唯一语义感知快照。

    ``AgentPerception.update()`` 每次只构造这一份快照。BT、Legacy Observation
    Builder 和未来研究 Adapter 应读取这份相同数据，避免出现两套 perception
    computation 在数值或时序上逐渐分叉。
    """

    agent: AgentState
    goal: GoalPerception
    hazard: HazardPerception
    boundary: BoundaryPerception

    # 以下 properties 是旧 Target/Obstacle 命名与新 Goal/Hazard 语义之间的
    # 集中兼容边界。每个 property 只返回上方 dataclass 中已经计算好的值：
    # 不读取 World、不访问 pygame geometry，也不会再次运行 perception。
    @property
    def target_visible(self) -> bool:
        return self.goal.visible

    @property
    def target_available(self) -> bool:
        return self.goal.available

    @property
    def target_source(self) -> str | None:
        return self.goal.source

    @property
    def target_distance(self) -> float | None:
        return self.goal.distance

    @property
    def target_bearing(self) -> float | None:
        return self.goal.bearing

    @property
    def target_unavailable_reason(self) -> str:
        return self.goal.unavailable_reason

    @property
    def visible_obstacles(self) -> tuple[HazardObservation, ...]:
        return self.hazard.visible_hazards

    @property
    def nearest_obstacle(self) -> HazardObservation | None:
        return self.hazard.nearest_hazard

    @property
    def sector_clearances(self) -> tuple[SectorRange, ...]:
        return self.hazard.sector_ranges

    @property
    def traversable_gaps(self) -> tuple[NavigationGap, ...]:
        return self.hazard.traversable_gaps

    @property
    def best_exploration_gap(self) -> NavigationGap | None:
        return self.hazard.best_exploration_gap

    @property
    def target_path_blocked(self) -> bool:
        return self.hazard.goal_direction_blocked

    @property
    def best_target_gap(self) -> NavigationGap | None:
        return self.hazard.best_goal_gap


@runtime_checkable
class SemanticPerceptionProvider(Protocol):
    """Simulator adapter 向 Research Method Layer 提供的最小稳定合约。

    ``observe()`` 每个 control tick 最多调用一次以刷新同步 observation；叶节点
    随后只读 ``snapshot``，避免噪声或昂贵 sensing 被同一 tick 重复采样。
    """

    snapshot: SemanticPerception

    def observe(self) -> SemanticPerception:
        """采集并返回当前 simulator state 对应的 semantic observation。"""
        ...
