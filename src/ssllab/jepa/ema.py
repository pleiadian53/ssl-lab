"""Exponential-moving-average target encoder.

The teacher (target) encoder is a non-trainable copy of the student whose
weights track the student via EMA. The momentum follows a cosine schedule from
``base`` to ``final`` over training — the standard I-JEPA/BYOL recipe.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, base: float = 0.996, final: float = 1.0) -> None:
        self.base = base
        self.final = final
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def momentum(self, step: int, total: int) -> float:
        """Cosine ramp of the EMA momentum from ``base`` to ``final``."""
        if total <= 1:
            return self.base
        frac = min(step / (total - 1), 1.0)
        return self.final - (self.final - self.base) * (math.cos(math.pi * frac) + 1.0) / 2.0

    @torch.no_grad()
    def update(self, student: nn.Module, step: int, total: int) -> float:
        """Update teacher params toward student. Returns the momentum used."""
        m = self.momentum(step, total)
        for ps, pt in zip(student.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(student.buffers(), self.teacher.buffers()):
            bt.copy_(bs)
        return m

    def to(self, device: torch.device | str) -> "EMA":
        self.teacher.to(device)
        return self
