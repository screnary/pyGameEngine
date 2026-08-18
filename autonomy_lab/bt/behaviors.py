"""定义当前 BT 的任务节点，并把感知快照转换为状态或控制命令。

Python ``__init__`` 只在 Loader 构建树时调用一次；py_trees 的
``initialise`` 在节点每次开始运行时调用，``update`` 在每次 tick 时调用，
``terminate`` 在完成、失败或被高优先级分支抢占时调用。
"""

import math

import pygame
import py_trees

from ..perception import PerceivedGap, PerceivedObstacle, normalise_angle
from .context import BehaviorBuildContext


class AgentBehaviour(py_trees.behaviour.Behaviour):
    """项目叶节点的统一构造接口和轻量参数辅助方法。

    Loader 始终调用 ``Behavior(context, name, **params)``。基类保存原始 JSON
    参数并做拼写检查，具体子类决定哪些参数需要场景 fallback。
    """

    visual_type = "behaviour"
    allowed_params: frozenset[str] = frozenset()

    def __init__(
        self,
        context: BehaviorBuildContext,
        name: str,
        **params: object,
    ) -> None:
        # super() 设置 py_trees 的 name、id、status、feedback_message 等运行字段。
        super().__init__(name=name)
        self.context = context
        self.params = dict(params)
        # 每个节点声明自己理解的 JSON 参数，拼写错误在启动时立即暴露。
        unknown = sorted(set(self.params) - self.allowed_params)
        if unknown:
            raise ValueError(
                f"unknown params for {type(self).__name__}: {', '.join(unknown)}"
            )

    def number_param(self, param_name: str, config_name: str) -> float:
        """读取数值型 JSON 覆盖；未提供时读取场景 BT 默认值。

        JSON 和场景配置最终都转换为 float，使 Behavior 后续计算不必关心原始
        JSON 是整数还是小数。
        """
        # JSON 节点参数优先；缺省时才读取当前场景的行为树配置。
        if param_name in self.params:
            value = self.params[param_name]
        else:
            try:
                value = self.context.behavior_config[config_name]
            except KeyError as error:
                raise ValueError(
                    f"missing parameter '{param_name}' for {type(self).__name__}"
                ) from error
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"parameter '{param_name}' for {type(self).__name__} must be numeric"
            ) from error

    def condition_param(
        self,
        param_name: str,
        expected_type: type | tuple[type, ...],
    ) -> py_trees.behaviour.Behaviour:
        """按 JSON 名称解析此前已构建、且类型兼容的 Condition 节点。"""
        condition_name = self.params.get(param_name)
        if not isinstance(condition_name, str) or not condition_name:
            raise ValueError(f"missing {param_name} reference")
        # JSON 按顺序递归构建，因此依赖的 Condition 必须已经进入索引。
        condition = self.context.nodes_by_name.get(condition_name)
        if condition is None:
            raise ValueError(f"unknown condition '{condition_name}'")
        if not isinstance(condition, expected_type):
            raise ValueError(f"node '{condition_name}' is not a compatible condition")
        return condition


class AgentCondition(AgentBehaviour):
    """Condition 基类：读取状态，不应主动产生运动命令。

    Condition 通常返回 SUCCESS 表示“条件成立”，FAILURE 表示“不成立”；这不
    代表程序错误，而是供 Selector/Sequence 选择分支的正常控制信号。
    """

    visual_type = "condition"


class AgentAction(AgentBehaviour):
    """Action 基类：可以写 command，并用 RUNNING 表示动作仍需后续 tick。"""

    visual_type = "action"


class TargetAvailable(AgentCondition):
    """判断 snapshot 是否提供了当前模式允许使用的目标信息。"""

    def update(self) -> py_trees.common.Status:
        snapshot = self.context.perception.snapshot
        # perceived 模式下目标被遮挡/超距/超 FOV 时 available=False。
        if not snapshot.target_available:
            self.feedback_message = snapshot.target_unavailable_reason
            return py_trees.common.Status.FAILURE

        distance = snapshot.target_distance or 0.0
        bearing = math.degrees(snapshot.target_bearing or 0.0)
        # feedback 明确显示信息来自真值还是可见感知，便于实验时核对模式。
        source = (
            "ground truth"
            if snapshot.target_source == "ground_truth"
            else "visible"
        )
        self.feedback_message = f"{source}: {distance:.0f} px, {bearing:+.0f} deg"
        return py_trees.common.Status.SUCCESS


class ObstacleThreat(AgentCondition):
    """从已可见障碍物中选择需要立即避让的最近威胁。

    ``avoidance_distance`` 是圆边缘净距离阈值；``half_angle`` 只检查 Agent
    前方一定范围，避免身后障碍触发强制转向。
    """

    allowed_params = frozenset({"avoidance_distance", "half_angle_degrees"})

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        # 解析发生在构建期；Controller 可在穿越缺口时临时改 avoidance_distance。
        self.avoidance_distance = self.number_param(
            "avoidance_distance", "obstacle_detection_distance"
        )
        self.half_angle = math.radians(
            self.number_param(
                "half_angle_degrees", "obstacle_detection_half_angle_degrees"
            )
        )
        self.threat: PerceivedObstacle | None = None

    def update(self) -> py_trees.common.Status:
        # visible_obstacles 已按距离排序，因此 next() 得到满足条件的最近项。
        self.threat = next(
            (
                obstacle
                for obstacle in self.context.perception.snapshot.visible_obstacles
                if obstacle.distance <= self.avoidance_distance
                and abs(obstacle.bearing) <= self.half_angle
            ),
            None,
        )
        if self.threat is None:
            self.feedback_message = "no nearby threat"
            return py_trees.common.Status.FAILURE

        bearing = math.degrees(self.threat.bearing)
        self.feedback_message = (
            f"threat: {self.threat.distance:.0f} px, {bearing:+.0f} deg"
        )
        return py_trees.common.Status.SUCCESS


class TraversableGap(AgentCondition):
    """读取感知器选出的最佳探索缺口，并把它暴露给后续 Action。"""

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        self.gap: PerceivedGap | None = None

    def update(self) -> py_trees.common.Status:
        # Condition 保存的是当前 snapshot 对象；MoveThroughGap 会在 initialise
        # 时复制世界坐标入口，之后不受下一帧 snapshot 改变影响。
        self.gap = self.context.perception.snapshot.best_exploration_gap
        if self.gap is None:
            self.feedback_message = "no traversable gap"
            return py_trees.common.Status.FAILURE

        self.feedback_message = (
            f"gap: {self.gap.free_distance:.0f} px, "
            f"{math.degrees(self.gap.bearing):+.0f} deg"
        )
        return py_trees.common.Status.SUCCESS


class TargetPathBlocked(AgentCondition):
    """判断朝向已知目标的局部直线路径是否被障碍物阻挡。"""

    def update(self) -> py_trees.common.Status:
        if self.context.perception.snapshot.target_path_blocked:
            self.feedback_message = "target path blocked"
            return py_trees.common.Status.SUCCESS
        self.feedback_message = "target path clear"
        return py_trees.common.Status.FAILURE


class TargetAlignedGap(AgentCondition):
    """选择安全缺口中 bearing 最接近目标方向的一项。"""

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        self.gap: PerceivedGap | None = None

    def update(self) -> py_trees.common.Status:
        self.gap = self.context.perception.snapshot.best_target_gap
        if self.gap is None:
            self.feedback_message = "no target gap"
            return py_trees.common.Status.FAILURE

        self.feedback_message = (
            f"target gap: {self.gap.free_distance:.0f} px, "
            f"{math.degrees(self.gap.bearing):+.0f} deg"
        )
        return py_trees.common.Status.SUCCESS


class AvoidObstacle(AgentAction):
    """在固定时长内输出远离威胁的转向命令。

    该 Action 是有状态的：``remaining`` 跨 tick 递减。Controller 每帧在 tick
    前写入 ``dt``，因为 py_trees 的 update() 本身没有时间参数。
    """

    allowed_params = frozenset({"condition", "duration", "throttle"})

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        # condition 是真实 ObstacleThreat 节点引用，不是复制的感知数据。
        self.condition = self.condition_param("condition", ObstacleThreat)
        self.command = context.command
        self.duration = self.number_param("duration", "avoid_duration")
        self.throttle = self.number_param("throttle", "avoid_throttle")
        self.dt = 0.0
        self.remaining = 0.0
        self.turn_direction = 1.0

    def initialise(self) -> None:
        """每次分支重新进入时重置计时，并锁定本次避让方向。"""
        self.remaining = self.duration
        threat = self.condition.threat
        # 正前方威胁固定向右避让；侧向威胁则转向其反方向。
        if threat is None or abs(threat.bearing) <= math.radians(5.0):
            self.turn_direction = 1.0
        else:
            self.turn_direction = -1.0 if threat.bearing > 0.0 else 1.0

    def update(self) -> py_trees.common.Status:
        """输出本帧命令；计时结束时返回 SUCCESS 让 Sequence 完成。"""
        self.command["turn"] = self.turn_direction
        self.command["throttle"] = self.throttle
        self.remaining -= self.dt
        self.feedback_message = f"avoid: turn={self.turn_direction:+.0f}"
        return (
            py_trees.common.Status.SUCCESS
            if self.remaining <= 0.0
            else py_trees.common.Status.RUNNING
        )

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """被抢占变为 INVALID 时清除未完成计时。"""
        if new_status == py_trees.common.Status.INVALID:
            self.remaining = 0.0


class MoveToTarget(AgentAction):
    """根据 snapshot 中的目标相对距离和方位持续追踪目标。"""

    allowed_params = frozenset({"reached_distance"})

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        self.command = context.command
        self.reached_distance = self.number_param(
            "reached_distance", "target_reached_distance"
        )

    def update(self) -> py_trees.common.Status:
        snapshot = self.context.perception.snapshot
        # Action 不直接读取 Environment.target，避免 perceived 模式泄漏真值。
        if snapshot.target_distance is None or snapshot.target_bearing is None:
            self.command["turn"] = 0.0
            self.command["throttle"] = 0.0
            self.feedback_message = "target unavailable"
            return py_trees.common.Status.FAILURE

        # 达到节点阈值后停车并返回 SUCCESS；main.py 另有 Episode 终止阈值。
        if snapshot.target_distance <= self.reached_distance:
            self.command["turn"] = 0.0
            self.command["throttle"] = 0.0
            self.feedback_message = "target reached"
            return py_trees.common.Status.SUCCESS

        # bearing 本身就是当前 heading 到目标方向的最短有符号转向误差。
        error = snapshot.target_bearing
        self.command["turn"] = max(-1.0, min(1.0, error / math.radians(45.0)))
        # 大角度偏航时降速，避免 Agent 一边高速前进一边原地大转弯。
        self.command["throttle"] = 1.0 if abs(error) < math.radians(60.0) else 0.3
        self.feedback_message = (
            f"pursuit: {snapshot.target_distance:.0f} px, "
            f"{math.degrees(error):+.0f} deg"
        )
        return py_trees.common.Status.RUNNING


class MoveThroughGap(AgentAction):
    """锁定一个世界坐标缺口入口，并持续转向和驶向该入口。

    同一个类可连接 TraversableGap 或 TargetAlignedGap；具体引用由 JSON 的
    ``params.condition`` 指定。
    """

    allowed_params = frozenset({"condition", "throttle", "reached_distance"})

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        self.condition = self.condition_param(
            "condition", (TraversableGap, TargetAlignedGap)
        )
        self.command = context.command
        self.throttle = self.number_param("throttle", "gap_throttle")
        self.reached_distance = self.number_param(
            "reached_distance", "gap_entry_reached_distance"
        )
        self.entry_position: pygame.Vector2 | None = None

    def initialise(self) -> None:
        """分支开始运行时，把 Condition 当前缺口复制为固定入口点。"""
        gap = self.condition.gap
        # 只在 Action 启动时复制一次入口，RUNNING 期间不跟随感知结果漂移。
        self.entry_position = (
            pygame.Vector2(gap.entry_position) if gap is not None else None
        )

    def update(self) -> py_trees.common.Status:
        """驶向锁定入口；入口无效时安全停车并返回 FAILURE。"""
        if self.entry_position is None:
            self.command["turn"] = 0.0
            self.command["throttle"] = 0.0
            self.feedback_message = "gap unavailable"
            return py_trees.common.Status.FAILURE

        # offset 每帧由固定入口减当前位置得到，因此距离会随 Agent 前进而减小。
        offset = (
            self.entry_position - self.context.perception.environment.agent.position
        )
        if offset.length() <= self.reached_distance:
            self.command["turn"] = 0.0
            self.command["throttle"] = 0.0
            self.feedback_message = "gap entry reached"
            return py_trees.common.Status.SUCCESS

        # 世界方向先转换为绝对 heading，再与当前 heading 求最短角误差。
        desired_heading = math.atan2(offset.y, offset.x)
        error = normalise_angle(
            desired_heading
            - self.context.perception.environment.agent.heading
        )
        self.command["turn"] = max(
            -1.0, min(1.0, error / math.radians(45.0))
        )
        self.command["throttle"] = self.throttle
        self.feedback_message = (
            f"gap entry: {offset.length():.0f} px, "
            f"{math.degrees(error):+.0f} deg"
        )
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """分支被抢占时丢弃旧入口，下次进入必须重新选择。"""
        if new_status == py_trees.common.Status.INVALID:
            self.entry_position = None


class SearchTarget(AgentAction):
    """目标未知且无可用缺口时，输出固定低速扫描命令。

    节点始终返回 RUNNING；一旦高优先级 Condition 成立，Selector 会抢占它。
    """

    allowed_params = frozenset({"throttle", "turn"})

    def __init__(
        self, context: BehaviorBuildContext, name: str, **params: object
    ) -> None:
        super().__init__(context, name, **params)
        self.command = context.command
        self.throttle = self.number_param("throttle", "search_throttle")
        self.turn = self.number_param("turn", "search_turn")

    def update(self) -> py_trees.common.Status:
        # 当前默认 throttle=0，因此表现为原地旋转；配置可改为低速弧线搜索。
        self.command["turn"] = self.turn
        self.command["throttle"] = self.throttle
        self.feedback_message = "search scan"
        return py_trees.common.Status.RUNNING
