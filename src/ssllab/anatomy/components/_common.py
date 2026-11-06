"""Helpers shared by the walkthroughs: layer descriptions, step-by-step MLPs and blocks, RNG rewind."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ssllab.anatomy.trace import Trace, count_params


def describe(m: nn.Module) -> str:
    """A short plain description of a single layer."""
    if isinstance(m, nn.Linear):
        return f"Linear {m.in_features}→{m.out_features}"
    if isinstance(m, nn.LayerNorm):
        return f"LayerNorm({m.normalized_shape[0]})"
    if isinstance(m, nn.Dropout):
        return f"Dropout(p={m.p:g})" + (", inactive" if m.p == 0 else "")
    if isinstance(m, nn.Embedding):
        return f"Embedding table ({m.num_embeddings} × {m.embedding_dim})"
    if isinstance(m, nn.Identity):
        return "identity"
    return type(m).__name__


def mlp_summary(seq: nn.Sequential) -> str:
    """``Linear 64→256, SiLU, Linear 256→512`` for a small MLP."""
    return ", ".join(describe(layer) for layer in seq if not (isinstance(layer, nn.Dropout) and layer.p == 0))


def run_sequential(tr: Trace, seq: nn.Sequential, x: torch.Tensor, path: str, name: str = "") -> torch.Tensor:
    """Apply ``seq`` one layer at a time, recording each; the last row gets ``name``."""
    for i, layer in enumerate(seq):
        x = layer(x)
        last = i == len(seq) - 1
        tr.record(name if last else "", x, f"{path}[{i}]: {describe(layer)}", module=layer,
                  label=describe(layer))
    return x


def run_residual_mlps(
    tr: Trace, blocks: nn.ModuleList, h: torch.Tensor, path: str, expand: bool, name: str = "h",
) -> torch.Tensor:
    """The velocity field's residual blocks, ``h <- h + block(h)``.

    Collapsed: one row for all blocks. Expanded: block 0 layer by layer, then one row for the rest.
    """
    k = len(blocks)
    if not expand:
        for blk in blocks:
            h = h + blk(h)
        tr.record(name, h, f"{path}[0..{k - 1}]: {k} residual blocks, each h + [{mlp_summary(blocks[0])}](h)",
                  params=sum(count_params(b) for b in blocks), label=f"{k} residual blocks")
        return h
    r = run_sequential(tr, blocks[0], h, f"{path}[0]")
    h = tr.record(name, h + r, "residual: h + block(h)", label="+ residual")
    for blk in blocks[1:]:
        h = h + blk(h)
    if k > 1:
        tr.record(name, h, f"{path}[1..{k - 1}]: {k - 1} more blocks of the same structure",
                  params=sum(count_params(b) for b in blocks[1:]), label=f"{k - 1} more blocks")
    return h


def run_transformer_blocks(
    tr: Trace, blocks: nn.ModuleList, x: torch.Tensor, path: str, expand: bool, name: str = "x",
) -> torch.Tensor:
    """Pre-norm transformer blocks (``ssllab.models.vit.TransformerBlock``).

    Collapsed: one row for all blocks. Expanded: block 0 operation by operation, then one row
    for the rest.
    """
    k = len(blocks)
    b0 = blocks[0]
    heads = b0.attn.num_heads
    d = b0.attn.embed_dim
    hidden = b0.mlp[0].out_features
    if not expand:
        for blk in blocks:
            x = blk(x)
        tr.record(name, x, f"{path}[0..{k - 1}]: {k} pre-norm transformer blocks "
                           f"(self-attention, {heads} heads; MLP {d}→{hidden}→{d})",
                  params=sum(count_params(b) for b in blocks), label=f"{k} transformer blocks")
        return x
    h = tr.record("", b0.norm1(x), f"{path}[0].norm1: LayerNorm({d})", module=b0.norm1, label="LayerNorm")
    a, _ = b0.attn(h, h, h, need_weights=False)
    tr.record("", a, f"{path}[0].attn: self-attention, {heads} heads × {d // heads}; every token attends to every token",
              module=b0.attn, label="self-attention")
    x = tr.record(name, x + a, "residual: x + attention", label="+ residual")
    h = tr.record("", b0.norm2(x), f"{path}[0].norm2: LayerNorm({d})", module=b0.norm2, label="LayerNorm")
    m = run_sequential(tr, b0.mlp, h, f"{path}[0].mlp")
    x = tr.record(name, x + m, "residual: x + MLP", label="+ residual")
    for blk in blocks[1:]:
        x = blk(x)
    if k > 1:
        tr.record(name, x, f"{path}[1..{k - 1}]: {k - 1} more blocks of the same structure",
                  params=sum(count_params(b) for b in blocks[1:]), label=f"{k - 1} more blocks")
    return x


@contextmanager
def rng_checkpoint():
    """Save the global RNG state; ``rewind()`` restores it so a second pass draws the same numbers.

    Used to run the replay and the real forward on identical random draws (masks, noise,
    reparameterization), so their outputs can be compared exactly.
    """
    state = torch.get_rng_state()

    class _Rewinder:
        @staticmethod
        def rewind() -> None:
            torch.set_rng_state(state)

    yield _Rewinder()
