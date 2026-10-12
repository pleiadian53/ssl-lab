"""Contiguous span masking for 1D token sequences, and the JEPAConfig seam."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from ssllab.checkpoint import load_jepa, save_jepa
from ssllab.jepa.masking import sample_span_masks
from ssllab.jepa.model import JEPAConfig, build_jepa


def _runs(idx: torch.Tensor) -> list[int]:
    """Lengths of maximal runs of consecutive integers in a sorted index vector."""
    out, n = [], 1
    for a, b in zip(idx[:-1].tolist(), idx[1:].tolist()):
        if b == a + 1:
            n += 1
        else:
            out.append(n)
            n = 1
    return out + [n]


@pytest.mark.parametrize("n_tokens,span_len,n_spans", [(40, 4, 2), (40, 8, 1), (10, 3, 3), (9, 4, 2)])
def test_span_masks_partition_and_contiguity(n_tokens, span_len, n_spans):
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        ctx, tgt = sample_span_masks(n_tokens, span_len, n_spans, generator=g)
        assert len(tgt) == span_len * n_spans
        assert sorted(ctx.tolist() + tgt.tolist()) == list(range(n_tokens))
        # Target = union of span_len-blocks: every maximal run is a multiple of span_len.
        assert all(r % span_len == 0 for r in _runs(tgt))


def test_span_positions_cover_the_whole_sequence():
    g = torch.Generator().manual_seed(1)
    hits = torch.zeros(40)
    for _ in range(2000):
        _, tgt = sample_span_masks(40, 4, 2, generator=g)
        hits[tgt] += 1
    assert (hits > 0).all()  # ends included, no position unreachable


def test_span_masks_reject_impossible_geometry():
    with pytest.raises(ValueError):
        sample_span_masks(8, 4, 2)  # no context left


def test_config_validates_span_target_count():
    with pytest.raises(ValueError):
        JEPAConfig(token_dim=10, n_tokens=40, n_target=6, mask_kind="span", span_len=4, n_spans=2)
    with pytest.raises(ValueError):
        JEPAConfig(mask_kind="blocks")


def test_span_jepa_trains_and_old_checkpoints_still_load(tmp_path):
    jepa = build_jepa(token_dim=12, n_tokens=40, embed_dim=16, enc_depth=1, pred_depth=1, n_heads=2,
                      n_target=8, mask_kind="span", span_len=4, n_spans=2)
    loss, _ = jepa(torch.randn(3, 40, 12))
    loss.backward()
    assert torch.isfinite(loss)
    # A checkpoint written before the masking fields existed (no mask_kind etc.) still loads.
    old = build_jepa(token_dim=12, n_tokens=16, embed_dim=16, enc_depth=1, pred_depth=1, n_heads=2)
    path = save_jepa(old, tmp_path / "old.pt")
    ck = torch.load(path)
    for k in ("mask_kind", "span_len", "n_spans"):
        ck["cfg"].pop(k)
    torch.save(ck, path)
    re = load_jepa(path)
    assert re.cfg.mask_kind == "random" and dataclasses.asdict(re.cfg)["n_target"] == 4
