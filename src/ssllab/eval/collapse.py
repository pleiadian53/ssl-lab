"""Representation-collapse diagnostics.

The failure mode JEPA must avoid is *collapse* — the encoder mapping every
input to (nearly) the same embedding. Two cheap signals:

- **feature_std**: mean per-dimension standard deviation. Near 0 => collapse.
- **effective_rank**: entropy-based soft rank of the embedding covariance. A
  healthy encoder uses many directions (eff-rank >> 1); a collapsed one uses
  ~1.
"""

from __future__ import annotations

import torch


def feature_std(z: torch.Tensor) -> float:
    """Mean per-dimension std over a batch of embeddings ``(M, D)``."""
    return float(z.detach().float().std(dim=0).mean())


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """Effective rank = exp(entropy of normalized singular values) of centered ``z``.

    Ranges in ``[1, D]``; higher means the representation spans more directions.
    """
    z = z.detach().float()
    zc = z - z.mean(dim=0, keepdim=True)
    # Singular values of the centered matrix.
    sv = torch.linalg.svdvals(zc)
    p = sv / (sv.sum() + eps)
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def collapse_report(z: torch.Tensor) -> dict[str, float]:
    """Bundle the diagnostics into a dict for logging."""
    return {
        "feature_std": feature_std(z),
        "effective_rank": effective_rank(z),
        "embed_dim": int(z.shape[-1]),
    }
