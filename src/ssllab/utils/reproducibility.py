"""Reproducibility helpers: seeding and device selection.

Mirrors the ``genai-lab`` ``utils/reproducibility.py`` API so the sibling
projects share the same surface. CPU is the sensible local default for
correctness; ``get_device("auto")`` upgrades to MPS (Apple) or CUDA when
present.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Base seed applied to every RNG.
    deterministic:
        When ``True``, also force deterministic cuDNN/algorithms. Slower, but
        makes runs bit-reproducible. Off by default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device: str = "auto") -> torch.device:
    """Resolve a torch device.

    ``"auto"`` prefers CUDA, then MPS (Apple Silicon), then CPU. Any explicit
    string (``"cpu"``, ``"cuda"``, ``"mps"``) is passed through unchanged.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
