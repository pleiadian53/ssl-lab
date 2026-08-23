"""Row-exact frozen latents for Perturb-seq cells.

The dataloader path (``ssllab.extract.extract_latents``) is right when you want *a* batch of latents.
But diagnostics — the ceiling ladder (``14_ceiling_analysis.py``), the Koopman-linearity probe
(``co-adaptation/01_koopman_linearity_probe.py``) — need the latents of an *explicit set of cache rows*,
so that the cells encoded are exactly the cells a metric scores. Indexing the cache by row rather than
draining a loader is what guarantees that. This helper is shared so the two drivers do not duplicate it.
"""

from __future__ import annotations

import numpy as np
import torch

from ssllab.data.perturbseq import tokenize_cells


@torch.no_grad()
def encode_rows(
    jepa,
    feat: np.ndarray,
    partition: torch.Tensor,
    rows: np.ndarray,
    device: torch.device | str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Frozen latents for an explicit set of cache rows, in encoder space (NOT standardized).

    Stage C trains the decoder on ``jepa.embed(tokens)`` directly, and Stage B de-standardizes before
    decoding, so encoder space is what the decoder expects. Callers that want standardized coordinates
    (e.g. for optimizer conditioning) should standardize the returned latents themselves — an affine
    change of coordinates, which leaves any linear-vs-nonlinear comparison unchanged.

    Args:
        jepa: a frozen JEPA (``jepa.embed(tokens) -> (B, D)``).
        feat: ``(N, n_hvg)`` HVG feature matrix (e.g. ``cache.hvg_X``).
        partition: ``(n_tokens, token_dim)`` gene-group partition from ``make_gene_partition``.
        rows: 1-D array of row indices into ``feat`` to encode.
        device: where to run the encoder.
        batch_size: encode in chunks of this many cells.

    Returns:
        ``(len(rows), D)`` latents on CPU, or an empty tensor if ``rows`` is empty.
    """
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(feat[rows[i:i + batch_size]]))
        tokens = tokenize_cells(chunk, partition).to(device)
        out.append(jepa.embed(tokens).cpu())
    return torch.cat(out) if out else torch.empty(0)
