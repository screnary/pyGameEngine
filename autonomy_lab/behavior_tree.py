"""连接感知、JSON Loader、py_trees tick、命令输出和 BT 可视化。"""

from collections.abc import Iterator

import pygame
import py_trees

from .behavior_context import BehaviorBuildContext
from .behaviors import AvoidObstacle, MoveThroughGap, MoveToTarget, ObstacleThreat
from .bt_loader import load_behavior_tree
from .bt_visualizer import BTVisualizer
from .environment import Environment
from .perception import AgentPerception


PANEL_WIDTH = 480


def _walk_tree(
    root: py_trees.behaviour.Behaviour,
) -> Iterator[py_trees.behaviour.Behaviour]:
    """以根节点优先、JSON 子节点顺序遍历 Runtime Tree。

    Controller 用该顺序建立节点分类缓存，Visualizer 则直接读取同一个 root。
    """
    yield root
    for child in root.children:
        yield from _walk_tree(child)


class BehaviorTreeController:
    """加载并驱动一棵配置化的 py_trees 行为树。

    每帧数据流为：更新 PerceptionSnapshot → 清空旧 command → tick 行为树 →
    返回 turn/throttle。Controller 不直接修改 Agent 的位置或朝向。
    """

    def __init__(self, environment: Environment, bt_config: str = "default") -> None:
        self.environment = environment
        # 所有 Action 写入同一个字典；main.py 在 tick 后读取最终值。
        self.command = {"turn": 0.0, "throttle": 0.0}
        config = environment.scene_config["behavior_tree"]
        # Perception 在 Controller 创建时初始化，此后每个 tick 刷新一次 snapshot。
        self.perception = AgentPerception(environment)
        loaded = load_behavior_tree(
            bt_config,
            BehaviorBuildContext(self.perception, self.command, config),
        )
        # Loader 返回真实 root 以及名称索引，后续代码不再依赖 JSON 原始字典。
        self.bt_config_id = loaded.config_id
        self.bt_definition_name = loaded.name
        self.root = loaded.root
        self.nodes_by_name = loaded.nodes_by_name
        # 按具体类分类一次，避免每帧重复遍历并判断整棵树。
        self._runtime_nodes = list(_walk_tree(self.root))
        self._obstacle_threats = [
            node for node in self._runtime_nodes if isinstance(node, ObstacleThreat)
        ]
        self._avoid_actions = [
            node for node in self._runtime_nodes if isinstance(node, AvoidObstacle)
        ]
        self._gap_actions = [
            node for node in self._runtime_nodes if isinstance(node, MoveThroughGap)
        ]
        self._target_actions = [
            node for node in self._runtime_nodes if isinstance(node, MoveToTarget)
        ]
        # visual_type 来自 AgentAction/AgentCondition 基类，可用于通用状态摘要。
        self._action_nodes = [
            node
            for node in self._runtime_nodes
            if getattr(node, "visual_type", None) == "action"
        ]
        # 保存 JSON/场景解析后的正常阈值，缺口承诺结束后可以恢复。
        self._normal_avoidance_distances = {
            node: node.avoidance_distance for node in self._obstacle_threats
        }
        self.gap_commit_emergency_distance = float(
            config["gap_commit_emergency_distance"]
        )
        if self.gap_commit_emergency_distance < 0.0 or any(
            self.gap_commit_emergency_distance > normal_distance
            for normal_distance in self._normal_avoidance_distances.values()
        ):
            raise ValueError(
                "gap_commit_emergency_distance must be between 0 and each "
                "ObstacleThreat avoidance_distance"
            )

        # BehaviourTree 是唯一 Runtime；SnapshotVisitor 只观察本帧访问路径。
        self.tree = py_trees.trees.BehaviourTree(self.root)
        self.snapshot = py_trees.visitors.SnapshotVisitor()
        self.tree.visitors.append(self.snapshot)
        self.visualizer = BTVisualizer(self.tree.root, self.snapshot)
        self.tick_count = 0

    def tick(self, dt: float) -> tuple[float, float]:
        """推进一次感知和 BT，返回归一化 ``(turn, throttle)``。

        ``dt`` 单位为秒。py_trees 的 tick 不自带时间参数，因此只向需要计时的
        AvoidObstacle Action 显式注入本帧 dt。
        """
        # 穿越缺口期间只允许极近距离威胁打断，避免普通避障把 Agent 推回入口外。
        gap_commitment_active = any(
            action.status == py_trees.common.Status.RUNNING
            and action.entry_position is not None
            for action in self._gap_actions
        )
        # 修改的是 Condition 判定阈值，不是 Environment 的真实碰撞半径。
        for threat in self._obstacle_threats:
            threat.avoidance_distance = (
                self.gap_commit_emergency_distance
                if gap_commitment_active
                else self._normal_avoidance_distances[threat]
            )
        self.perception.update()
        # 每帧先清空命令，防止被抢占 Action 的上一帧输出残留。
        self.command["turn"] = 0.0
        self.command["throttle"] = 0.0
        for action in self._avoid_actions:
            # py_trees 不传入 dt，定时 Action 由 Controller 在 tick 前注入本帧时长。
            action.dt = dt
        # tick 会按 Selector/Sequence 语义调用节点 initialise/update/terminate。
        self.tree.tick()
        self.tick_count += 1
        return self.command["turn"], self.command["throttle"]

    def reset(self) -> None:
        """使全部节点失效，并恢复感知、阈值、访问快照和计数器。

        INVALID 会触发运行中 Action 的 ``terminate()``，从而清理计时或锁定
        的 gap entry，避免新 Episode 继承旧执行状态。
        """
        self.command["turn"] = 0.0
        self.command["throttle"] = 0.0
        self.tree.root.stop(py_trees.common.Status.INVALID)
        # SnapshotVisitor 自己保存前后两帧集合，重置时必须一并清空。
        self.snapshot.visited = {}
        self.snapshot.previously_visited = {}
        self.snapshot.changed = False
        self.perception.update()
        for threat, normal_distance in self._normal_avoidance_distances.items():
            threat.avoidance_distance = normal_distance
        self.tick_count = 0

    @property
    def active_behavior(self) -> str:
        """返回当前 RUNNING Action 名称，供界面和实验转移计数使用。"""
        for action in self._action_nodes:
            if action.status == py_trees.common.Status.RUNNING:
                return action.name
        if any(
            action.status == py_trees.common.Status.SUCCESS
            for action in self._target_actions
        ):
            return "Target Reached"
        return "None"

    @property
    def active_action_label(self) -> str:
        return self.active_behavior

    @property
    def decision_label(self) -> str:
        """从本帧访问节点中选择最有解释力的反馈文本。

        优先显示成功 Condition；若没有，则显示运行中 Action，最后才显示失败
        Condition。这只是界面摘要，不参与 BT 决策。
        """
        visited_conditions = [
            node
            for node in self._runtime_nodes
            if getattr(node, "visual_type", None) == "condition"
            and node.id in self.snapshot.visited
            and node.feedback_message
        ]
        successful = [
            node
            for node in visited_conditions
            if node.status == py_trees.common.Status.SUCCESS
        ]
        if successful:
            return successful[-1].feedback_message
        running_actions = [
            node
            for node in self._action_nodes
            if node.status == py_trees.common.Status.RUNNING
            and node.feedback_message
        ]
        if running_actions:
            return running_actions[0].feedback_message
        if visited_conditions:
            return visited_conditions[-1].feedback_message
        return "Awaiting perception"

    def draw_panel(
        self, surface: pygame.Surface, font: pygame.font.Font, world_width: int
    ) -> None:
        """把 Controller 摘要和真实 Runtime Tree 交给 Visualizer 绘制。"""
        summaries = [
            f"Active Action: {self.active_action_label}",
            f"Decision: {self.decision_label}",
            f"Tick: {self.tick_count}",
        ]
        self.visualizer.draw(surface, font, world_width, summaries)
