"""Definition-driven Pygame view of a live py_trees behavior tree."""

from dataclasses import dataclass, field
from uuid import UUID

import pygame
import py_trees


STATUS_COLORS = {
    py_trees.common.Status.RUNNING: (245, 205, 80),
    py_trees.common.Status.SUCCESS: (80, 220, 120),
    py_trees.common.Status.FAILURE: (235, 90, 90),
    py_trees.common.Status.INVALID: (130, 138, 150),
}


@dataclass
class VisualNode:
    """Lightweight drawing adapter that keeps the real py_trees node."""

    node: py_trees.behaviour.Behaviour
    node_id: UUID
    name: str
    node_type: str
    depth: int
    parent_id: UUID | None
    child_ids: list[UUID] = field(default_factory=list)
    x: int = 0
    y: int = 0


class BTVisualizer:
    """Extract, cache, lay out, and later render a live py_trees topology."""

    def __init__(
        self,
        root: py_trees.behaviour.Behaviour,
        snapshot: py_trees.visitors.SnapshotVisitor,
    ) -> None:
        self.root = root
        self.snapshot = snapshot
        self.visual_nodes: dict[UUID, VisualNode] = {}
        self.connections: list[tuple[UUID, UUID]] = []
        self.signature: tuple[tuple[UUID, UUID | None, int], ...] = ()
        self.rebuild_count = 0
        self._layout_rect: pygame.Rect | None = None
        self._small_font: pygame.font.Font | None = None

    def sync(self, panel_rect: pygame.Rect) -> bool:
        """Rebuild topology and layout only when the real tree changes."""
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
        """Read status and current-tick visitation from the real runtime."""
        return node.status, node.id in self.snapshot.visited

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        world_width: int,
        summaries: list[str],
    ) -> None:
        """Draw the current topology and runtime state in the right panel."""
        panel = pygame.Rect(
            world_width, 0, surface.get_width() - world_width, surface.get_height()
        )
        pygame.draw.rect(surface, (16, 20, 28), panel)
        pygame.draw.line(
            surface, (80, 90, 108), panel.topleft, panel.bottomleft, 2
        )

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

        layout_rect = pygame.Rect(
            panel.left,
            divider_y,
            panel.width,
            max(panel.height - divider_y - 62, 100),
        )
        self.sync(layout_rect)
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
        self.visual_nodes = {}
        self.connections = []

        def extract(
            node: py_trees.behaviour.Behaviour,
            parent_id: UUID | None,
            depth: int,
        ) -> None:
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
        logical_x: dict[UUID, float] = {}
        next_leaf = 0

        def place(node_id: UUID) -> float:
            nonlocal next_leaf
            visual = self.visual_nodes[node_id]
            if not visual.child_ids:
                x = float(next_leaf)
                next_leaf += 1
            else:
                child_positions = [place(child_id) for child_id in visual.child_ids]
                x = (child_positions[0] + child_positions[-1]) / 2.0
            logical_x[node_id] = x
            return x

        place(self.root.id)
        leaf_count = max(next_leaf, 1)
        max_depth = max(node.depth for node in self.visual_nodes.values())
        horizontal_margin = min(75, max(24, panel_rect.width // 6))
        vertical_margin = 55
        horizontal_span = max(panel_rect.width - 2 * horizontal_margin, 1)
        vertical_span = max(panel_rect.height - 2 * vertical_margin, 1)

        for node_id, visual in self.visual_nodes.items():
            if leaf_count == 1:
                x = panel_rect.centerx
            else:
                x = panel_rect.left + horizontal_margin + round(
                    logical_x[node_id] * horizontal_span / (leaf_count - 1)
                )
            if max_depth == 0:
                y = panel_rect.centery
            else:
                y = panel_rect.top + vertical_margin + round(
                    visual.depth * vertical_span / max_depth
                )
            visual.x = x
            visual.y = y

    @staticmethod
    def _node_type(node: py_trees.behaviour.Behaviour) -> str:
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
        if visual.node_type == "condition":
            size = (138, 92)
        elif visual.node_type in {"selector", "sequence", "parallel"}:
            size = (154, 76)
        else:
            size = (140, 76)
        rect = pygame.Rect(0, 0, *size)
        rect.center = (visual.x, visual.y)
        return rect

    def _draw_connection(
        self,
        surface: pygame.Surface,
        parent_rect: pygame.Rect,
        child_rect: pygame.Rect,
        child: py_trees.behaviour.Behaviour,
    ) -> None:
        status, visited = self.runtime_state(child)
        color = STATUS_COLORS[status] if visited else (58, 66, 80)
        width = 5 if visited and status == py_trees.common.Status.RUNNING else 3
        if not visited:
            width = 2

        start = parent_rect.midbottom
        end = child_rect.midtop
        midpoint_y = (start[1] + end[1]) // 2
        pygame.draw.lines(
            surface,
            color,
            False,
            [start, (start[0], midpoint_y), (end[0], midpoint_y), end],
            width,
        )
        pygame.draw.polygon(
            surface,
            color,
            [end, (end[0] - 6, end[1] - 9), (end[0] + 6, end[1] - 9)],
        )

    def _draw_node(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        visual: VisualNode,
    ) -> None:
        status, visited = self.runtime_state(visual.node)
        status_color = STATUS_COLORS[status]
        border_color = status_color if visited else (70, 78, 92)
        fill_color = self._darken(status_color, 0.24) if visited else (25, 30, 40)
        border_width = 5 if visited and status == py_trees.common.Status.RUNNING else 3

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
        small_font = self._get_small_font()
        type_image = small_font.render(visual.node_type.upper(), True, border_color)
        name_image = small_font.render(visual.name, True, text_color)
        status_image = small_font.render(status.name, True, status_color)
        surface.blit(type_image, type_image.get_rect(center=(rect.centerx, rect.top + 17)))
        surface.blit(name_image, name_image.get_rect(center=rect.center))
        surface.blit(
            status_image,
            status_image.get_rect(center=(rect.centerx, rect.bottom - 17)),
        )

    def _get_small_font(self) -> pygame.font.Font:
        if self._small_font is None:
            self._small_font = pygame.font.Font(None, 20)
        return self._small_font

    @staticmethod
    def _darken(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
        return tuple(round(channel * amount) for channel in color)
