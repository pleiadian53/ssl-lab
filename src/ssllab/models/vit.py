"""Modality-agnostic transformer backbone (a tiny ViT).

Everything here operates on token tensors ``(B, N, token_dim)`` plus integer
position indices ``(B, N)`` — there is nothing image-specific. The same trunk
serves the context encoder, the EMA target encoder, and (a shallow instance)
the predictor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Linear projection of flattened token vectors into the model dim."""

    def __init__(self, token_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(token_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (MHSA + MLP) with residual connections."""

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """Stack of pre-norm transformer blocks with a learned positional table.

    Parameters
    ----------
    embed_dim, depth, n_heads, mlp_ratio:
        Standard transformer hyper-parameters.
    n_positions:
        Size of the learned positional-embedding table. Positions are gathered
        by index so that context/target *subsets* of tokens still receive the
        correct absolute position embedding.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        n_positions: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def pos_for(self, idx: torch.Tensor) -> torch.Tensor:
        """Gather positional embeddings for integer indices ``(B, N)`` -> ``(B, N, D)``."""
        # pos_embed: (1, P, D) -> expand to batch, then gather along position axis.
        b = idx.shape[0]
        pos = self.pos_embed.expand(b, -1, -1)
        index = idx.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        return torch.gather(pos, 1, index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer over already-embedded, position-added tokens ``(B, N, D)``."""
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
