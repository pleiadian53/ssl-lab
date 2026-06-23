"""Extract pooled JEPA latents over a dataloader.

Shared by the probe, decoder-training, and prior-training examples — hence it
lives in ``src/`` rather than being duplicated in each driver.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from ssllab.data.mnist import PATCH_SIZE, patchify


@torch.no_grad()
def extract_latents(
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device | str = "cpu",
    limit: int | None = None,
    patch: int = PATCH_SIZE,
    prepare: Callable[[object, torch.device | str], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``embed_fn`` over the loader, returning ``(Z, Y)``.

    Parameters
    ----------
    embed_fn:
        Maps token tensor ``(B, N, token_dim)`` -> pooled latent ``(B, D)``
        (e.g. ``jepa.embed``).
    limit:
        Optional cap on the number of examples (fast smoke runs).
    prepare:
        Optional ``callable(batch, device) -> (tokens, labels)`` for non-image
        modalities whose loader already yields tokenized batches (e.g. the
        perturbseq dict batch: ``lambda b, d: (b["tokens"].to(d), b["pert_id"])``).
        When ``None`` the loader is assumed to yield ``(images, labels)`` and
        ``patchify`` is applied — the MNIST default (backward-compatible).

    Returns
    -------
    ``Z`` ``(N, D)`` latents and ``Y`` ``(N,)`` integer labels, on CPU.
    """
    zs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        if prepare is None:
            images, labels = batch
            tokens = patchify(images.to(device), patch)
        else:
            tokens, labels = prepare(batch, device)
        z = embed_fn(tokens)
        zs.append(z.cpu())
        ys.append(labels.cpu())
        seen += tokens.shape[0]
        if limit is not None and seen >= limit:
            break
    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0)
    if limit is not None:
        Z, Y = Z[:limit], Y[:limit]
    return Z, Y
