"""Context/target masking for JEPA.

For each sample we partition the ``n_tokens`` positions into a *target* subset
(what the predictor must predict) and a *context* subset (everything else, what
the student encoder sees). v0 uses a simple random target subset of fixed size,
which is enough to drive learning on the 4x4 MNIST grid. For ordered 1D token
sequences (time series), :func:`sample_span_masks` hides *contiguous* spans
instead: on a quasi-periodic signal such as an ECG, a randomly scattered hidden
token can be interpolated from its neighbours, while a hidden span covering a
whole beat cannot.

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


def sample_span_masks(
    n_tokens: int,
    span_len: int,
    n_spans: int = 1,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n_spans`` non-overlapping contiguous target spans of ``span_len`` tokens.

    Spans may touch (two adjacent spans read as one longer span) but never
    overlap, so the target always has exactly ``n_spans * span_len`` positions.
    Start positions are drawn uniformly over all valid placements (stars and
    bars): choose ``n_spans`` distinct values ``u_1 < ... < u_k`` from
    ``{0, ..., n_tokens - n_spans * span_len + n_spans - 1}`` and set
    ``start_i = u_i + i * (span_len - 1)``.

    Returns ``(ctx_idx, tgt_idx)`` like :func:`sample_block_masks`.
    """
    n_target = span_len * n_spans
    if span_len < 1 or n_spans < 1 or not n_target < n_tokens:
        raise ValueError(f"need span_len, n_spans >= 1 and span_len*n_spans < n_tokens; got {span_len}, {n_spans}, {n_tokens}")
    free = n_tokens - n_target
    u = torch.randperm(free + n_spans, generator=generator)[:n_spans].sort().values
    starts = u + torch.arange(n_spans) * (span_len - 1)
    tgt = (starts.unsqueeze(1) + torch.arange(span_len).unsqueeze(0)).flatten()
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    mask[tgt] = True
    return torch.nonzero(~mask).flatten().to(device), torch.nonzero(mask).flatten().to(device)


def batched(idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast a 1-D index vector to ``(batch_size, len(idx))``."""
    return idx.unsqueeze(0).expand(batch_size, -1)
