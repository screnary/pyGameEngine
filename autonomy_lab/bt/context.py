"""定义所有 Behavior 在构建和运行时共享的最小上下文。"""

from dataclasses import dataclass, field

import py_trees

from ..core.agent import AgentCommand
from ..perception.semantic_perception import (
    SemanticPerception,
    SemanticPerceptionProvider,
)
from .parameters import ConditionParameters, ParameterStore


@dataclass
class BehaviorBuildContext:
    """把 Behavior 所需公共对象统一打包，避免 Loader 了解节点构造细节。

    Attributes:
        perception: 当前 Agent 的同步感知器。所有节点读取同一份 snapshot。
        command: Action 共享的可变输出字典，包含 ``turn`` 和 ``throttle``。
        behavior_config: 当前场景的 BT 默认参数；JSON params 可覆盖它们。
        nodes_by_name: 已构建节点索引，用于 Action 引用前面的 Condition。

    这个对象不是 Blackboard：它不保存任意业务数据，也不负责跨 Episode
    记忆。它只是当前小型原型的依赖容器和构建期节点索引。
    """

    # 同一个 Context 会传给一棵树中的全部叶节点，因此感知和命令天然共享。
    perception: SemanticPerceptionProvider
    command: AgentCommand
    behavior_config: dict
    # HybridPPOEnv 训练时由外部 SB3 Policy 提供动作；默认 False 保持冻结
    # PPONavigate 自行加载 checkpoint 并推理的 M5.0/M5.1 行为。
    external_ppo_control: bool = False
    # Research Conditions 每次 tick 都读取同一个可变 Store；legacy 节点忽略它。
    condition_parameters: ParameterStore = field(
        default_factory=ConditionParameters
    )
    # Loader 每完成一个节点就写入此字典，后续节点即可按 JSON name 查找。
    nodes_by_name: dict[str, py_trees.behaviour.Behaviour] = field(
        default_factory=dict
    )

    @property
    def semantic(self) -> SemanticPerception:
        """返回 Controller 在本 tick 已采集的只读 semantic observation。"""
        return self.perception.snapshot
