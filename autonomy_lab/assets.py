"""Minimal optional image loading for project-local assets."""

from pathlib import Path

import pygame


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_optional_image(filename: str) -> pygame.Surface | None:
    """Load a transparent PNG, returning None when it is absent or invalid."""
    try:
        return pygame.image.load(ASSETS_DIR / filename).convert_alpha()
    except (FileNotFoundError, OSError, pygame.error):
        return None
