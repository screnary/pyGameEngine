"""在 BT 子包内把真实 py_trees Runtime Tree 转换为 Pygame 面板。

Visualizer 不读取 BT JSON，也不参与决策。它只遍历 Controller 已构建的 root，
因此画面结构、节点状态和实际执行树始终来自同一个 Runtime。
"""

from dataclasses import dataclass, field
from uuid import UUID

import pygame
import py_trees


# 状态颜色同时用于节点边框和访问路径连接线。
STATUS_COLORS = {
    py_trees.common.Status.RUNNING: (245, 205, 80),
    py_trees.common.Status.SUCCESS: (80, 220, 120),
    py_trees.common.Status.FAILURE: (235, 90, 90),
    py_trees.common.Status.INVALID: (130, 138, 150),
}


@dataclass
class VisualNode:
    """真实 py_trees 节点对应的轻量绘图适配数据。

    ``node`` 保留 Runtime 对象以读取实时 status/feedback；其余字段只服务于
    布局，不会反向修改行为树。
    """

    node: py_trees.behaviour.Behaviour
    node_id: UUID
    name: str
    node_type: str
    depth: int
    parent_id: UUID | None
    child_ids: list[UUID] = field(default_factory=list)
    x: int = 0
    y: int = 0
    width: int = 140
    height: int = 76


class BTVisualizer:
    """提取、缓存、布局并绘制一棵正在运行的 py_trees 行为树。"""

    def __init__(
        self,
        root: py_trees.behaviour.Behaviour,
        snapshot: py_trees.visitors.SnapshotVisitor,
    ) -> None:
        self.root = root
        self.snapshot = snapshot
        # UUID 是 py_trees 节点稳定 id，适合作为连接与矩形索引键。
        self.visual_nodes: dict[UUID, VisualNode] = {}
        self.connections: list[tuple[UUID, UUID]] = []
        # signature 只描述拓扑；节点 status 改变不会触发昂贵的重新布局。
        self.signature: tuple[tuple[UUID, UUID | None, int], ...] = ()
        self.rebuild_count = 0
        self._layout_rect: pygame.Rect | None = None
        self._small_font: pygame.font.Font | None = None
        self._tiny_font: pygame.font.Font | None = None

    def sync(self, panel_rect: pygame.Rect) -> bool:
        """仅在 Runtime 拓扑或面板尺寸变化时重建布局。

        返回 ``True`` 表示本次执行了 rebuild，主要供测试和诊断使用。
        """
        signature = self._tree_signature()
        layout_rect = pygame.Rect(panel_rect)
        if signature == self.signature and layout_rect == self._layout_rect:
            return False

        self.signature = signature
        self._layout_rect = layout_rect
        self._rebuild(layout_rect)
        self.rebuild_count += 1
        return True

    def runtime_state(
        self, node: py_trees.behaviour.Behaviour
    ) -> tuple[py_trees.common.Status, bool]:
        """同时返回节点状态和“本 tick 是否被访问”。

        status 可能保留上次运行结果，而 visited 只描述当前 tick 的实际路径；
        面板用两者区分“历史状态”和“当前决策证据”。
        """
        return node.status, node.id in self.snapshot.visited

    def feedback_text(
        self, node: py_trees.behaviour.Behaviour, max_chars: int = 18
    ) -> str:
        """返回当前 tick 的短反馈；未访问节点不显示旧反馈。"""
        if node.id not in self.snapshot.visited:
            return ""
        text = str(node.feedback_message).strip()
        return self._truncate(text, max_chars)

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        world_width: int,
        summaries: list[str],
    ) -> None:
        """在世界右侧绘制摘要、连接、节点和状态图例。"""
        # panel 使用 surface 剩余宽度，因此世界尺寸不同也不影响 BT 区域定位。
        panel = pygame.Rect(
            world_width, 0, surface.get_width() - world_width, surface.get_height()
        )
        pygame.draw.rect(surface, (16, 20, 28), panel)
        pygame.draw.line(
            surface, (80, 90, 108), panel.topleft, panel.bottomleft, 2
        )

        # 顶部 summaries 来自 Controller，不属于树节点布局区域。
        title = font.render("BEHAVIOR TREE", True, (230, 235, 245))
        surface.blit(title, (panel.left + 20, 20))
        for index, text in enumerate(summaries):
            color = (150, 205, 255) if index == 0 else (185, 192, 205)
            surface.blit(
                font.render(text, True, color),
                (panel.left + 20, 50 + index * 25),
            )

        divider_y = 132
        pygame.draw.line(
            surface,
            (65, 73, 88),
            (panel.left + 14, divider_y),
            (panel.right - 14, divider_y),
            1,
        )

        # 为标题和底部图例预留空间，树节点只在 layout_rect 内排布。
        layout_rect = pygame.Rect(
            panel.left,
            divider_y,
            panel.width,
            max(panel.height - divider_y - 62, 100),
        )
        self.sync(layout_rect)
        # 先计算所有 Rect，再画连接，保证连线端点与最终节点尺寸一致。
        rectangles = {
            node_id: self._node_rect(visual)
            for node_id, visual in self.visual_nodes.items()
        }

        for parent_id, child_id in self.connections:
            self._draw_connection(
                surface,
                rectangles[parent_id],
                rectangles[child_id],
                self.visual_nodes[child_id].node,
            )
        for node_id, visual in self.visual_nodes.items():
            self._draw_node(surface, rectangles[node_id], visual)

        legend = "RUNNING   SUCCESS   FAILURE   INVALID"
        legend_image = self._get_small_font().render(
            legend, True, (125, 133, 146)
        )
        surface.blit(legend_image, (panel.left + 24, panel.bottom - 42))

    def _tree_signature(self) -> tuple[tuple[UUID, UUID | None, int], ...]:
        """用 ``(节点, 父节点, 子序号)`` 序列描述 Runtime 拓扑。"""
        entries: list[tuple[UUID, UUID | None, int]] = []

        def visit(
            node: py_trees.behaviour.Behaviour,
            parent_id: UUID | None,
            child_order: int,
        ) -> None:
            entries.append((node.id, parent_id, child_order))
            for index, child in enumerate(node.children):
                visit(child, node.id, index)

        visit(self.root, None, 0)
        return tuple(entries)

    def _rebuild(self, panel_rect: pygame.Rect) -> None:
        """从真实 Runtime Tree 递归提取 VisualNode 和父子连接。"""
        self.visual_nodes = {}
        self.connections = []

        def extract(
            node: py_trees.behaviour.Behaviour,
            parent_id: UUID | None,
            depth: int,
        ) -> None:
            # depth 决定横向列，children 顺序保持 JSON/Runtime 原始顺序。
            child_ids = [child.id for child in node.children]
            self.visual_nodes[node.id] = VisualNode(
                node=node,
                node_id=node.id,
                name=node.name,
                node_type=self._node_type(node),
                depth=depth,
                parent_id=parent_id,
                child_ids=child_ids,
            )
            for child in node.children:
                self.connections.append((node.id, child.id))
                extract(child, node.id, depth + 1)

        extract(self.root, None, 0)
        self._calculate_layout(panel_rect)

    def _calculate_layout(self, panel_rect: pygame.Rect) -> None:
        """均匀放置叶节点，并把 Composite 居中到其后代范围。"""
        logical_y: dict[UUID, float] = {}
        next_leaf = 0

        def place(node_id: UUID) -> float:
            nonlocal next_leaf
            visual = self.visual_nodes[node_id]
            if not visual.child_ids:
                y = float(next_leaf)
                next_leaf += 1
            else:
                child_positions = [place(child_id) for child_id in visual.child_ids]
                y = (child_positions[0] + child_positions[-1]) / 2.0
            logical_y[node_id] = y
            return y

        # 先给叶节点连续编号，再用首尾子节点的中点放置父节点。
        place(self.root.id)
        leaf_count = max(next_leaf, 1)
        # 最大 depth 决定横向列间距；叶节点数量决定纵向间距。
        max_depth = max(node.depth for node in self.visual_nodes.values())
        horizontal_margin = min(78, max(24, panel_rect.width // 6))
        vertical_margin = min(26, max(16, panel_rect.height // 10))
        horizontal_span = max(panel_rect.width - 2 * horizontal_margin, 1)
        vertical_span = max(panel_rect.height - 2 * vertical_margin, 1)

        for node_id, visual in self.visual_nodes.items():
            if max_depth == 0:
                x = panel_rect.centerx
            else:
                x = panel_rect.left + horizontal_margin + round(
                    visual.depth * horizontal_span / max_depth
                )
            if leaf_count == 1:
                y = panel_rect.centery
            else:
                y = panel_rect.top + vertical_margin + round(
                    logical_y[node_id] * vertical_span / (leaf_count - 1)
                )
            visual.x = x
            visual.y = y

        # 同一 depth 的节点共享列宽，并根据实际垂直间距收缩高度以避免重叠。
        nodes_by_depth: dict[int, list[VisualNode]] = {}
        for visual in self.visual_nodes.values():
            nodes_by_depth.setdefault(visual.depth, []).append(visual)
        if max_depth == 0:
            level_width = max(24, panel_rect.width - 20)
        else:
            column_spacing = horizontal_span / max_depth
            level_width = max(24, round(column_spacing) - 20)
        for level_nodes in nodes_by_depth.values():
            ordered = sorted(level_nodes, key=lambda item: item.y)
            if len(ordered) > 1:
                min_spacing = min(
                    lower.y - upper.y
                    for upper, lower in zip(ordered, ordered[1:])
                )
                level_height = min(52, max(30, min_spacing - 8))
            else:
                level_height = 52
            for visual in ordered:
                base_width, base_height = self._base_node_size(visual.node_type)
                visual.width = min(base_width, level_width)
                visual.height = min(base_height, level_height)

    @staticmethod
    def _node_type(node: py_trees.behaviour.Behaviour) -> str:
        """把 Runtime 类或叶节点 visual_type 转换为绘图类别。"""
        if isinstance(node, py_trees.composites.Selector):
            return "selector"
        if isinstance(node, py_trees.composites.Sequence):
            return "sequence"
        if isinstance(node, py_trees.composites.Parallel):
            return "parallel"
        if isinstance(node, py_trees.decorators.Decorator):
            return "decorator"
        return str(getattr(node, "visual_type", "behaviour")).lower()

    @staticmethod
    def _node_rect(visual: VisualNode) -> pygame.Rect:
        """由中心坐标和布局尺寸生成 Pygame Rect。"""
        rect = pygame.Rect(0, 0, visual.width, visual.height)
        rect.center = (visual.x, visual.y)
        return rect

    @staticmethod
    def _base_node_size(node_type: str) -> tuple[int, int]:
        """返回各语义类型的理想尺寸，最终仍可能被布局空间压缩。"""
        if node_type == "condition":
            return 138, 92
        if node_type in {"selector", "sequence", "parallel"}:
            return 154, 76
        return 140, 76

    def _draw_connection(
        self,
        surface: pygame.Surface,
        parent_rect: pygame.Rect,
        child_rect: pygame.Rect,
        child: py_trees.behaviour.Behaviour,
    ) -> None:
        """绘制父子折线，并用子节点当前状态决定颜色和粗细。"""
        status, visited = self.runtime_state(child)
        color = STATUS_COLORS[status] if visited else (58, 66, 80)
        # 当前运行路径最粗；未访问连接最细且使用暗色。
        width = 5 if visited and status == py_trees.common.Status.RUNNING else 3
        if not visited:
            width = 2

        start = parent_rect.midright
        end = child_rect.midleft
        # 中间 x 形成正交折线，避免直接斜线穿过相邻节点。
        midpoint_x = (start[0] + end[0]) // 2
        pygame.draw.lines(
            surface,
            color,
            False,
            [start, (midpoint_x, start[1]), (midpoint_x, end[1]), end],
            width,
        )
        pygame.draw.polygon(
            surface,
            color,
            [end, (end[0] - 9, end[1] - 6), (end[0] - 9, end[1] + 6)],
        )

    def _draw_node(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        visual: VisualNode,
    ) -> None:
        """按节点语义绘制形状，并叠加名称、状态和本帧反馈。"""
        status, visited = self.runtime_state(visual.node)
        status_color = STATUS_COLORS[status]
        border_color = status_color if visited else (70, 78, 92)
        fill_color = self._darken(status_color, 0.24) if visited else (25, 30, 40)
        border_width = 5 if visited and status == py_trees.common.Status.RUNNING else 3

        # Selector、Condition 和普通矩形使用不同轮廓，便于不看文字也能区分。
        if visual.node_type == "selector":
            inset = 18
            points = [
                (rect.left + inset, rect.top),
                (rect.right - inset, rect.top),
                (rect.right, rect.centery),
                (rect.right - inset, rect.bottom),
                (rect.left + inset, rect.bottom),
                (rect.left, rect.centery),
            ]
            pygame.draw.polygon(surface, fill_color, points)
            pygame.draw.polygon(surface, border_color, points, border_width)
        elif visual.node_type == "condition":
            points = [rect.midtop, rect.midright, rect.midbottom, rect.midleft]
            pygame.draw.polygon(surface, fill_color, points)
            pygame.draw.polygon(surface, border_color, points, border_width)
        else:
            radius = 12 if visual.node_type == "action" else 3
            pygame.draw.rect(surface, fill_color, rect, border_radius=radius)
            pygame.draw.rect(
                surface, border_color, rect, border_width, border_radius=radius
            )

        text_color = (235, 239, 246) if visited else (135, 143, 157)
        text_font = self._get_tiny_font()
        max_text_width = max(rect.width - 16, 8)
        # 名称最多两行；状态固定一行，空间足够时再显示 feedback。
        name_lines = self._wrap_text(
            visual.name, text_font, max_text_width, max_lines=2
        )
        text_rows = [(line, text_color) for line in name_lines]
        text_rows.append((status.name, status_color))
        max_chars = max(4, max_text_width // 6)
        feedback = self.feedback_text(visual.node, max_chars=max_chars)
        if feedback and rect.height >= 62:
            text_rows.append((feedback, (170, 180, 195)))

        line_height = text_font.get_linesize()
        first_y = rect.centery - ((len(text_rows) - 1) * line_height) // 2
        for index, (text, color) in enumerate(text_rows):
            image = text_font.render(text, True, color)
            surface.blit(
                image,
                image.get_rect(
                    center=(rect.centerx, first_y + index * line_height)
                ),
            )

    def _get_small_font(self) -> pygame.font.Font:
        """延迟创建图例字体，避免未初始化 Pygame font 时提前失败。"""
        if self._small_font is None:
            self._small_font = pygame.font.Font(None, 16)
        return self._small_font

    def _get_tiny_font(self) -> pygame.font.Font:
        """延迟创建节点文字字体。"""
        if self._tiny_font is None:
            self._tiny_font = pygame.font.Font(None, 12)
        return self._tiny_font

    @staticmethod
    def _wrap_text(
        text: str,
        font: pygame.font.Font,
        max_width: int,
        max_lines: int = 2,
    ) -> list[str]:
        """按真实像素宽度换行，只有最后一行允许省略号截断。"""
        remaining = " ".join(text.split())
        if not remaining or max_width <= 0 or max_lines <= 0:
            return []

        def clipped(value: str) -> str:
            # 字体不是等宽字体，必须反复调用 font.size 而不能按字符数切分。
            ellipsis = "..."
            if font.size(value)[0] <= max_width:
                return value
            if font.size(ellipsis)[0] > max_width:
                ellipsis = "."
            prefix = value
            while prefix and font.size(prefix.rstrip() + ellipsis)[0] > max_width:
                prefix = prefix[:-1]
            return prefix.rstrip() + ellipsis

        lines: list[str] = []
        for line_index in range(max_lines):
            if font.size(remaining)[0] <= max_width:
                lines.append(remaining)
                break
            if line_index == max_lines - 1:
                lines.append(clipped(remaining))
                break

            # 优先按单词边界换行；单个超长词再按字符切分。
            words = remaining.split()
            line = ""
            consumed = 0
            for word in words:
                candidate = f"{line} {word}".strip()
                if font.size(candidate)[0] > max_width:
                    break
                line = candidate
                consumed += len(word) + (1 if consumed else 0)

            if not line:
                split_at = 1
                while (
                    split_at < len(remaining)
                    and font.size(remaining[: split_at + 1])[0] <= max_width
                ):
                    split_at += 1
                line = remaining[:split_at]
                consumed = split_at

            lines.append(line)
            remaining = remaining[consumed:].lstrip()

        return lines

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """按字符数生成带省略号的短反馈。"""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return "." * max_chars
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _darken(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
        """缩放 RGB 通道，为状态颜色生成较暗的节点填充色。"""
        return tuple(round(channel * amount) for channel in color)
