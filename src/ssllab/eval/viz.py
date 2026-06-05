"""Lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.utils as vutils


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> Path:
    """Save a grid of images ``(B, 1, H, W)`` in ``[0, 1]`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(images.clamp(0, 1).cpu(), nrow=nrow, padding=2)
    vutils.save_image(grid, str(path))
    return path
