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

from collections.abc import Sequence

import torch
import torch.nn as nn


def build_pert_gene_matrix(pert_names: Sequence[str]) -> tuple[torch.Tensor, list[str]]:
    """Map a perturbation vocabulary to a multi-hot **target-gene** membership matrix.

    Each perturbation name is a ``+``-joined set of target genes (``"CEBPE+RUNX1T1"``);
    ``"control"`` targets nothing. Returns ``(M, gene_vocab)`` where ``M`` is
    ``(n_perts, n_genes)`` with ``M[p, g] = 1`` iff gene ``g`` is targeted by pert ``p``,
    and ``gene_vocab`` lists the distinct target genes (sorted, index = column).

    This is the bridge that lets an *unseen combo* be embedded: its row references the
    same gene columns as the (seen) singles it is composed of — so ``e(A+B)`` is built
    from the trained ``e(A)``, ``e(B)`` rather than an untrained per-perturbation slot.
    """
    def genes_of(name: str) -> list[str]:
        return [g for g in str(name).split("+") if g and g != "control"]

    vocab = sorted({g for name in pert_names for g in genes_of(name)})
    index = {g: i for i, g in enumerate(vocab)}
    M = torch.zeros(len(pert_names), len(vocab))
    for p, name in enumerate(pert_names):
        for g in genes_of(name):
            M[p, index[g]] = 1.0
    return M, vocab


class GeneSetEmbedding(nn.Module):
    """Compose a perturbation embedding ``z_p`` from its **target genes**.

    Holds a fixed multi-hot ``pert_gene`` buffer ``(n_perts, n_genes)`` (from
    :func:`build_pert_gene_matrix`) and a learned per-gene table. ``forward(pert_id)``
    pools the embeddings of the genes a perturbation targets:

    - ``compose="additive"`` (default): ``z_p = sum_g e(g)`` — so ``z_p(A+B) = e(A)+e(B)``
      *exactly*. Maximal compositional inductive bias; ``control`` (no genes) → ``0``.
    - ``compose="deepsets"``: ``z_p = phi(sum_g psi(e(g)))`` — a permutation-invariant
      DeepSets refinement that can model interactions the pure sum cannot. Ablation.

    Because pooling is over *gene* slots (never a per-perturbation slot), a held-out
    combination is embedded from already-trained parts.
    """

    def __init__(
        self,
        pert_gene: torch.Tensor,
        gene_dim: int = 64,
        out_dim: int | None = None,
        compose: str = "additive",
        hidden: int = 128,
    ) -> None:
        super().__init__()
        if compose not in ("additive", "deepsets"):
            raise ValueError(f"compose must be 'additive' or 'deepsets', got {compose!r}")
        self.compose = compose
        self.gene_dim = gene_dim
        n_perts, n_genes = pert_gene.shape
        self.register_buffer("pert_gene", pert_gene.float())
        self.gene_emb = nn.Embedding(n_genes, gene_dim)
        if compose == "deepsets":
            self.psi = nn.Sequential(nn.Linear(gene_dim, hidden), nn.SiLU(), nn.Linear(hidden, gene_dim))
            self.phi = nn.Sequential(nn.Linear(gene_dim, hidden), nn.SiLU(), nn.Linear(hidden, gene_dim))
        self.out_dim = out_dim or gene_dim
        self.proj = nn.Identity() if self.out_dim == gene_dim else nn.Linear(gene_dim, self.out_dim)

    def forward(self, pert_id: torch.Tensor) -> torch.Tensor:
        mh = self.pert_gene[pert_id]                      # (B, n_genes) multi-hot
        if self.compose == "additive":
            z_p = mh @ self.gene_emb.weight               # (B, gene_dim) = sum of present genes
        else:
            h = self.psi(self.gene_emb.weight)            # (n_genes, gene_dim)
            z_p = self.phi(mh @ h)                        # masked sum -> per-set MLP
        return self.proj(z_p)


class GeneSetConditionEncoder(nn.Module):
    """Drop-in for :class:`ConditionEncoder` whose perturbation embedding **composes**
    from target genes, so it generalizes to unseen combinations.

    Same ``forward(z_b, pert_id) -> c`` signature as :class:`ConditionEncoder`; the only
    change is that ``z_p`` comes from :class:`GeneSetEmbedding` (gene-compositional)
    instead of a per-perturbation table. Build the multi-hot with
    :func:`build_pert_gene_matrix` from the dataset's ``pert_names``.
    """

    def __init__(
        self,
        latent_dim: int,
        pert_gene: torch.Tensor,
        pert_dim: int = 64,
        gene_dim: int = 64,
        cond_dim: int = 128,
        hidden: int = 256,
        compose: str = "additive",
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.pert_emb = GeneSetEmbedding(pert_gene, gene_dim=gene_dim, out_dim=pert_dim, compose=compose)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + pert_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cond_dim),
        )

    def forward(self, z_b: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_b, self.pert_emb(pert_id)], dim=-1)
        return self.net(h)


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
