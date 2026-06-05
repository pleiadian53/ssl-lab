"""JEPA predictor: predict target-token embeddings from context embeddings.

Given the context tokens' embeddings (at their positions) and the *positions*
of the target tokens, the predictor inserts a learned mask token at each target
position and runs a shallow transformer over the concatenation. It returns the
embeddings read out at the target positions — the prediction of what the target
encoder would produce there.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssllab.models.vit import TinyViT


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        n_tokens: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.trunk = TinyViT(
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            n_positions=n_tokens,
        )

    def forward(
        self,
        ctx_embed: torch.Tensor,
        ctx_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict embeddings at the target positions.

        Parameters
        ----------
        ctx_embed:
            ``(B, C, D)`` context embeddings from the student encoder.
        ctx_idx:
            ``(B, C)`` positions of the context tokens.
        tgt_idx:
            ``(B, T)`` positions to predict.

        Returns
        -------
        ``(B, T, D)`` predicted target embeddings.
        """
        b, c, d = ctx_embed.shape
        t = tgt_idx.shape[1]

        # Context stream: embeddings already carry content; add positions.
        ctx = ctx_embed + self.trunk.pos_for(ctx_idx)
        # Target stream: learned mask token + target positions.
        mask = self.mask_token.expand(b, t, -1) + self.trunk.pos_for(tgt_idx)

        x = torch.cat([ctx, mask], dim=1)
        x = self.trunk(x)
        # Read out the target slots (the last T positions).
        return x[:, c : c + t, :]
