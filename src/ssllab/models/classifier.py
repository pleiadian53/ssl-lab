"""A classification head on top of a JEPA encoder: mean pool -> linear.

The same module serves every downstream arm. ``scratch`` builds a fresh encoder;
``finetune`` and ``probe`` take a pretrained one (``probe`` freezes it). Keeping
one head definition is part of keeping the arms comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssllab.models.encoder import JEPAEncoder


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: JEPAEncoder, n_out: int, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, n_out)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, N, token_dim)`` -> logits ``(B, n_out)``."""
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder.embed_pooled(tokens)
        else:
            z = self.encoder.embed_pooled(tokens)
        return self.head(z)

    def train(self, mode: bool = True) -> "EncoderClassifier":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # a frozen encoder never switches dropout on
        return self
