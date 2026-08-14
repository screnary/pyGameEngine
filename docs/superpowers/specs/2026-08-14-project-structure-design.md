# 科研原型目录重构设计

## 目标

在不改变 Milestone 0 行为的前提下，将 Python 领域代码从仓库根目录移入
`autonomy_lab` 包。结构需要便于后续逐步加入 Behavior Tree、RL 和混合控制，
但当前不创建这些尚未使用的模块或目录。

## 目录结构

```text
pyGameEngine_2608/
├── main.py
├── autonomy_lab/
│   ├── __init__.py
│   ├── agent.py
│   └── environment.py
├── docs/
│   └── superpowers/specs/
├── README.md
├── PROJECT_GUIDE.md
├── AGENTS.md
├── environment.yml
└── .gitignore
```

用户已有的 `.vscode/` 配置保留原状，不纳入本次重构。

## 模块职责

- `main.py` 作为直接运行入口，只负责 Pygame 初始化、事件循环、刷新和退出。
- `autonomy_lab/agent.py` 负责 Agent 状态、键盘控制输入转换和 Agent 绘制。
- `autonomy_lab/environment.py` 负责场景状态、障碍物、碰撞、重置和场景绘制。
- `autonomy_lab/__init__.py` 仅标记 Python 包，不提前提供公共 API 门面。
- `README.md` 记录当前结构、Conda 环境和启动命令。

导入使用明确的包路径：

```python
# main.py
from autonomy_lab.environment import Environment

# autonomy_lab/environment.py
from .agent import Agent
```

## 数据流

```text
Pygame input
    -> main.py
    -> Environment.update()
    -> Agent
    -> collision resolution
    -> draw
```

本次只移动模块并更新导入，不改变控制、碰撞、重置、显示或运行方式。

## 扩展原则

保持渐进式结构。只有在对应 Milestone 开始后，才根据实际代码增加
`controllers/`、`behaviors/`、`rl/` 或 `experiments/` 等子目录。项目继续作为科研
原型，不采用 `src/` 布局，不引入插件架构、事件总线、依赖注入或复杂配置系统。

## 错误处理

不新增错误处理抽象。导入错误和 Pygame 启动错误保持直接暴露，便于科研原型阶段
快速定位问题。现有窗口退出流程保持不变。

## 验证

重构后进行最小必要验证：

1. 使用 `pygame_lab` 环境编译并导入所有 Python 模块。
2. 检查键盘控制路径、障碍物碰撞回退和重置。
3. 使用无界面视频驱动启动并正常退出 Pygame 主循环。
4. 确认 `conda run -n pygame_lab python main.py` 仍为直接运行命令。

不新增完整测试套件，也不实现 Behavior Tree、RL 或 Gymnasium 接口。
