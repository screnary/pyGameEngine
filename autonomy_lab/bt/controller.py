"""连接感知、JSON Loader、py_trees tick、命令输出和 BT 可视化。"""

from collections.abc import Iterator

import pygame
import py_trees

from ..environment import Environment
from .behaviors import (
    AvoidObstacle,
    MoveThroughGap,
    MoveToTarget,
    ObstacleThreat,
    PPONavigate,
    SafeBoundaryRecovery,
    SearchTarget,
)
from .context import BehaviorBuildContext
from .loader import load_behavior_tree
from .visualizer import BTVisualizer


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

    def __init__(
        self,
        environment: Environment,
        bt_config: str = "default",
        external_ppo_control: bool = False,
    ) -> None:
        self.environment = environment
        # 所有 Action 写入同一个字典；main.py 在 tick 后读取最终值。
        self.command = {"turn": 0.0, "throttle": 0.0}
        config = environment.scene_config["behavior_tree"]
        # World 持有唯一 Perception；Controller 只刷新并读取同一个实例。
        self.perception = environment.perception
        loaded = load_behavior_tree(
            bt_config,
            BehaviorBuildContext(
                self.perception,
                self.command,
                config,
                external_ppo_control=external_ppo_control,
            ),
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
        self._ppo_actions = [
            node for node in self._runtime_nodes if isinstance(node, PPONavigate)
        ]
        self._boundary_actions = [
            node
            for node in self._runtime_nodes
            if isinstance(node, SafeBoundaryRecovery)
        ]
        self._search_actions = [
            node for node in self._runtime_nodes if isinstance(node, SearchTarget)
        ]
        self.controller_id = "hybrid_bt_ppo" if self._ppo_actions else "bt-v1"
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
        # M5.1 旁路诊断只观察 Runtime 状态，不参与任何节点选择或命令计算。
        self.ppo_active_time = 0.0
        self.boundary_recovery_activation_count = 0
        self.search_activation_count = 0
        self.ppo_preemption_count = 0
        self._previous_ppo_active = False
        self._previous_boundary_active = False
        self._previous_search_active = False

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
        for action in self._ppo_actions:
            # PPO 只用仿真 dt 累计 10 Hz 决策间隔；它不会 sleep 或推进 World。
            action.dt = dt
        # tick 会按 Selector/Sequence 语义调用节点 initialise/update/terminate。
        self.tree.tick()
        self.tick_count += 1
        ppo_active = any(
            action.status == py_trees.common.Status.RUNNING
            for action in self._ppo_actions
        )
        boundary_active = any(
            action.status == py_trees.common.Status.RUNNING
            for action in self._boundary_actions
        )
        search_active = any(
            action.status == py_trees.common.Status.RUNNING
            for action in self._search_actions
        )
        if ppo_active:
            self.ppo_active_time += dt
        if boundary_active and not self._previous_boundary_active:
            self.boundary_recovery_activation_count += 1
        if search_active and not self._previous_search_active:
            self.search_activation_count += 1
        if self._previous_ppo_active and boundary_active:
            self.ppo_preemption_count += 1
        self._previous_ppo_active = ppo_active
        self._previous_boundary_active = boundary_active
        self._previous_search_active = search_active
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
        self.ppo_active_time = 0.0
        self.boundary_recovery_activation_count = 0
        self.search_activation_count = 0
        self.ppo_preemption_count = 0
        self._previous_ppo_active = False
        self._previous_boundary_active = False
        self._previous_search_active = False
        for action in self._ppo_actions:
            action.decision_count = 0

    @property
    def ppo_decision_count(self) -> int:
        """返回当前 Episode 内全部 Frozen PPO Action 的推理次数。"""
        return sum(action.decision_count for action in self._ppo_actions)

    @property
    def ppo_active(self) -> bool:
        """PPO Navigate 是否是当前 RUNNING Action。"""
        return any(
            action.status == py_trees.common.Status.RUNNING
            for action in self._ppo_actions
        )

    @property
    def ppo_action_required(self) -> bool:
        """外部训练 Policy 是否应在当前 World state 产生下一次决策。"""
        return self.ppo_active and any(
            action.action_required for action in self._ppo_actions
        )

    def set_ppo_action(self, action: object) -> None:
        """把 Action 交给当前获得 BT 控制权的唯一 PPONavigate 节点。"""
        active_nodes = [
            node
            for node in self._ppo_actions
            if node.status == py_trees.common.Status.RUNNING
        ]
        if len(active_nodes) != 1 or not active_nodes[0].action_required:
            raise RuntimeError("PPO action is not requested by the Behavior Tree")
        active_nodes[0].provide_external_action(action)

    def clear_ppo_action(self) -> None:
        """结束当前 PPO decision interval，等待下一次外部 Action。"""
        for action in self._ppo_actions:
            action.clear_external_action()

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
