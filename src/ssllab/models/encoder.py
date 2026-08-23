"""JEPA encoder: patch-embed + positional add + TinyViT trunk.

Used twice in a JEPA: as the *context* (student) encoder that sees a subset of
tokens, and — as an EMA copy — as the *target* (teacher) encoder that sees the
full set. The ``idx`` argument lets the same module embed either the full token
set or an arbitrary subset while preserving absolute positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssllab.models.vit import PatchEmbed, TinyViT


class JEPAEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        n_tokens: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(token_dim, embed_dim)
        self.trunk = TinyViT(
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            n_positions=n_tokens,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Embed a (sub)set of tokens.

        Parameters
        ----------
        tokens:
            ``(B, N, token_dim)`` raw token vectors (full set).
        idx:
            Optional ``(B, M)`` integer indices selecting a subset of the N
            tokens. When ``None`` the full set is embedded with positions
            ``0..N-1``.

        Returns
        -------
        ``(B, M, embed_dim)`` contextualized embeddings (M = N if ``idx`` is None).
        
        
        For the production geometry, imagine:
        tokens.shape == (B, 50, 100)
        idx.shape    == (B, 38)
        where:

        • B: batch size
        • 50: all gene-group tokens in a cell
        • 100: genes/features within each raw token
        • 38: context tokens retained after masking
        """
        b, n, _ = tokens.shape
        if idx is None:
            idx = torch.arange(n, device=tokens.device).unsqueeze(0).expand(b, -1)
            sel = tokens
        else:
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            sel = torch.gather(tokens, 1, gather_idx)
        x = self.patch_embed(sel) + self.trunk.pos_for(idx)
        return self.trunk(x)

    def embed_pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pooled image-level latent ``(B, embed_dim)`` over the full token set.

        This is the ``z`` consumed by the decoder, the flow prior, and the
        linear probe.
        """
        return self.forward(tokens, idx=None).mean(dim=1)
