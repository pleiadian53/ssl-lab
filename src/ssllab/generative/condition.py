"""Condition encoder for the perturbation flow: (baseline, perturbation) -> c.

The conditional flow prior ([flow.py](flow.py)) steers generation by a condition
vector ``c``. For perturbation response that condition is the pair ``(z_b, z_p)``
from the design-space series: ``z_b`` the baseline (a control cell's latent) and
``z_p = e(p)`` a learned embedding of the intervention identity. This module is
the small map that turns that pair into the ``c`` the velocity field consumes —
the two-different-maps design (encoder for the state, learned embedding for the
intervention) made concrete.

Kept separate from the modality-agnostic ``flow.py`` because *this* piece is
perturbation-specific; the velocity field neither knows nor cares what ``c`` means.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Map ``(z_b, perturbation_id)`` to a condition vector ``c`` of width ``cond_dim``.

    ``z_b`` ``(B, latent_dim)`` is the baseline-cell latent; ``pert_id`` ``(B,)`` is
    the integer perturbation label, embedded through a learned table (swap for a
    structure/feature map later for unseen-perturbation generalization — see the
    Q&A on the condition embedding).
    """

    def __init__(
        self,
        latent_dim: int,
        n_perts: int,
        pert_dim: int = 64,
        cond_dim: int = 128,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.pert_emb = nn.Embedding(n_perts, pert_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + pert_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cond_dim),
        )

    def forward(self, z_b: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_b, self.pert_emb(pert_id)], dim=-1)
        return self.net(h)
