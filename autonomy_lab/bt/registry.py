"""在 BT 子包内把 JSON behavior 名称显式映射到 Python Behavior 类。

Registry 不负责创建实例、解析参数或扫描插件；这些职责分别属于 Loader 和
Behavior 自身。新增符合统一构造接口的节点时，只需在字典中增加一项。
"""

from .behaviors import (
    AgentBehaviour,
    AvoidObstacle,
    MoveThroughGap,
    MoveToTarget,
    ObstacleThreat,
    SearchTarget,
    TargetAlignedGap,
    TargetAvailable,
    TargetPathBlocked,
    TraversableGap,
)


# 所有注册值都必须是 AgentBehaviour 子类，而不是 build_xx 工厂函数。
BehaviorClass = type[AgentBehaviour]


# key 必须与 JSON 的 ``behavior`` 完全一致；value 是尚未实例化的类。
# 注册表只回答“使用哪个类”，构造参数由统一接口直接接收。
BEHAVIOR_REGISTRY: dict[str, BehaviorClass] = {
    "ObstacleThreat": ObstacleThreat,
    "AvoidObstacle": AvoidObstacle,
    "TargetAvailable": TargetAvailable,
    "TargetPathBlocked": TargetPathBlocked,
    "TargetAlignedGap": TargetAlignedGap,
    "MoveThroughGap": MoveThroughGap,
    "MoveToTarget": MoveToTarget,
    "TraversableGap": TraversableGap,
    "SearchTarget": SearchTarget,
}
