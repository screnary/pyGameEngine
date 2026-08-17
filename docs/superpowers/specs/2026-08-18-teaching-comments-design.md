# 中文教学型代码注释设计

## 目标

在不修改任何运行逻辑、接口、配置和测试语义的前提下，将当前项目的
注释提升为适合初次阅读自主智能体代码的教学型说明。

## 方案选择

可选密度包括：

1. 仅补充函数 Docstring，改动最小，但不足以解释函数内部数据流；
2. 按逻辑块添加教学注释，解释输入、处理原因、状态变化和输出；
3. 几乎逐行注释，信息最多，但会重复代码并降低可读性。

采用方案 2。注释密度为中高，不逐行翻译显而易见的 Python 或 Pygame
调用。

## 语言与格式

- 模块、类和函数 Docstring 使用中文；
- Python 标识符、库名和数据字段保持英文；
- 关键函数 Docstring 说明职责、主要输入、返回值或副作用；
- 函数内部按逻辑块注释，说明“为什么”和数据下一步流向；
- 数学量注明单位或坐标语义，例如弧度、像素、相对方位；
- 不添加自动生成文档框架或新的开发依赖。

## 覆盖范围

生产代码全部覆盖：

```text
main.py
autonomy_lab/agent.py
autonomy_lab/assets.py
autonomy_lab/environment.py
autonomy_lab/perception.py
autonomy_lab/behavior_context.py
autonomy_lab/behaviors.py
autonomy_lab/behavior_registry.py
autonomy_lab/bt_loader.py
autonomy_lab/behavior_tree.py
autonomy_lab/bt_visualizer.py
autonomy_lab/experiment.py
autonomy_lab/scene_config.py
```

测试模块保留简洁中文模块说明和夹具说明，不为每条断言添加重复注释。
JSON 不支持注释，因此 `bt_configs/default.json` 不修改。

## 重点讲解内容

- `main.py`：初始化顺序、帧循环、`dt`、事件、Episode 生命周期；
- `agent.py`：状态单位、归一化命令、运动积分、图片旋转；
- `environment.py`：状态所有权、分轴碰撞、圆与矩形碰撞、绘制顺序；
- `perception.py`：Ground Truth 与 perceived、FOV/LOS、射线、缺口分组；
- `behavior_context.py`：四个字段的来源、用途和共享方式；
- `behaviors.py`：构造期与 tick 期、参数回退、Condition/Action 生命周期；
- `behavior_registry.py`：名称到类的映射职责；
- `bt_loader.py`：递归校验、构造顺序、Condition 引用；
- `behavior_tree.py`：perception → tick → command、抢占和 reset；
- `bt_visualizer.py`：Runtime topology、布局坐标、状态颜色和文字裁剪；
- `experiment.py`：指标累计、轨迹采样、JSON/CSV 写入和旧表头迁移；
- `scene_config.py`：配置分层以及深拷贝原因。

## 边界

本轮只改注释和 Docstring。不重命名变量、不拆文件、不改变函数签名、
不调整算法、不新增功能，也不顺手重构代码风格。

## 验证

运行现有完整测试、三个场景的无窗口构建/绘制/reset、`compileall` 和
`git diff --check`。注释改动不得改变现有 16 节点 BT 或实验记录。
