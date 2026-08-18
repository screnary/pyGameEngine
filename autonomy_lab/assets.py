"""加载项目内的可选图片素材，并在失败时允许几何图形接管绘制。"""

from pathlib import Path

import pygame


# 路径基于当前源码文件，而不是进程 working directory。
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_optional_image(filename: str) -> pygame.Surface | None:
    """加载带透明通道的图片；文件缺失或无效时返回 ``None``。

    ``convert_alpha`` 将图片转换为适合当前显示格式的透明 Surface。调用方
    只需判断返回值，无须区分“文件不存在”和“图片损坏”。
    """
    try:
        return pygame.image.load(ASSETS_DIR / filename).convert_alpha()
    except (FileNotFoundError, OSError, pygame.error):
        # 素材是可选增强，加载失败时由调用方继续绘制 Pygame 基础图形。
        return None
