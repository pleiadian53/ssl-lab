"""Context/target masking for JEPA.

For each sample we partition the ``n_tokens`` positions into a *target* subset
(what the predictor must predict) and a *context* subset (everything else, what
the student encoder sees). v0 uses a simple random target subset of fixed size,
which is enough to drive learning on the 4x4 MNIST grid. The ``grid`` and
``scale`` arguments are reserved so this can grow into proper I-JEPA contiguous
block masking without changing call sites.

Masks are sampled per-batch but *shared across the batch* (one context/target
split per step), which keeps the variable-length subsets rectangular and the
code simple. This is a common simplification for small-scale JEPA POCs.
"""

from __future__ import annotations

import torch


def sample_block_masks(
    n_tokens: int,
    n_target: int = 4,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one (context, target) index split shared across the batch.

    Parameters
    ----------
    n_tokens:
        Total number of token positions.
    n_target:
        Number of target positions to predict (1 <= n_target < n_tokens).

    Returns
    -------
    ``(ctx_idx, tgt_idx)`` 1-D LongTensors of positions on ``device``.
    ``ctx_idx`` has length ``n_tokens - n_target``; ``tgt_idx`` has ``n_target``.
    """
    if not 1 <= n_target < n_tokens:
        raise ValueError(f"need 1 <= n_target < n_tokens, got {n_target} / {n_tokens}")
    perm = torch.randperm(n_tokens, generator=generator)
    tgt_idx = perm[:n_target].sort().values.to(device)
    ctx_idx = perm[n_target:].sort().values.to(device)
    return ctx_idx, tgt_idx


def batched(idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast a 1-D index vector to ``(batch_size, len(idx))``."""
    return idx.unsqueeze(0).expand(batch_size, -1)
