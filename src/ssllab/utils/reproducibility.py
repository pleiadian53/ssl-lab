"""Reproducibility helpers: seeding and device selection.

Mirrors the ``genai-lab`` ``utils/reproducibility.py`` API so the sibling
projects share the same surface. CPU is the sensible local default for
correctness; ``get_device("auto")`` upgrades to CUDA when present (e.g. a pod).
MPS is deliberately NOT auto-selected — see ``get_device``.
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

    ``"auto"`` prefers **CUDA** (e.g. a GPU pod), else **CPU**. MPS (Apple
    Silicon) is *not* auto-selected: it has recurring operator gaps (e.g.
    ``torch.linalg.svdvals`` is unimplemented; ``smooth_l1_loss`` rejects
    non-contiguous tensors) that silently break training. Request it explicitly
    as a last resort with ``device="mps"``. Any explicit string (``"cpu"``,
    ``"cuda"``, ``"mps"``) is passed through unchanged.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)
