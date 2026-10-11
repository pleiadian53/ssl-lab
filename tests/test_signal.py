"""1D signal helpers: physical-time geometry, per-lead z-scoring, and patch layout."""

from __future__ import annotations

import pytest
import torch

from ssllab.data.signal import SignalGeometry, patchify_1d, unpatchify_1d, zscore_per_lead


def test_geometry_at_100hz():
    g = SignalGeometry(fs=100, duration_s=10, patch_s=0.25, span_s=1.0)
    assert (g.n_samples, g.patch_len, g.n_tokens, g.span_tokens) == (1000, 25, 40, 4)
    assert g.token_dim(12) == 300 and g.token_dim(1) == 25


def test_token_count_is_rate_free_but_width_is_not():
    a = SignalGeometry(fs=100, patch_s=0.25)
    b = SignalGeometry(fs=500, patch_s=0.25)
    assert a.n_tokens == b.n_tokens == 40
    assert (a.token_dim(12), b.token_dim(12)) == (300, 1500)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fs=100, patch_s=0.125),             # 12.5 samples per patch
        dict(fs=100, patch_s=0.3),               # 10 s / 0.3 s is not whole
        dict(fs=100, patch_s=0.25, span_s=0.3),  # span not a whole number of patches
        dict(fs=100, patch_s=0.25, span_s=10.0), # span leaves no context
    ],
)
def test_geometry_refuses_inexact_ratios(kwargs):
    with pytest.raises(ValueError):
        SignalGeometry(**kwargs)


def test_describe_records_physical_and_derived():
    d = SignalGeometry(fs=100, patch_s=0.25, span_s=1.0).describe(12)
    assert d["fs"] == 100 and d["patch_s"] == 0.25 and d["span_s"] == 1.0
    assert d["n_tokens"] == 40 and d["token_dim"] == 300


def test_zscore_per_lead_stats_and_invariances():
    torch.manual_seed(0)
    x = torch.randn(4, 12, 1000) * 3 + 2
    z = zscore_per_lead(x)
    assert torch.allclose(z.mean(-1), torch.zeros(4, 12), atol=1e-5)
    assert torch.allclose(z.std(-1, unbiased=False), torch.ones(4, 12), atol=1e-4)
    # Positive gain and DC offset per lead leave the output unchanged.
    gain = torch.rand(1, 12, 1) * 4 + 0.25
    assert torch.allclose(zscore_per_lead(x * gain + 7.0), z, atol=1e-4)


def test_zscore_flat_lead_is_zero_not_nan():
    x = torch.randn(2, 3, 100)
    x[:, 1] = 5.0
    z = zscore_per_lead(x)
    assert torch.isfinite(z).all()
    assert torch.equal(z[:, 1], torch.zeros(2, 100))


def test_patchify_roundtrip_and_layout():
    B, C, L, P = 2, 3, 20, 5
    x = torch.arange(B * C * L, dtype=torch.float32).reshape(B, C, L)
    t = patchify_1d(x, P)
    assert t.shape == (B, L // P, C * P)
    # Token n holds samples [nP, (n+1)P) of each lead, lead-major.
    n = 2
    expect = torch.cat([x[0, c, n * P:(n + 1) * P] for c in range(C)])
    assert torch.equal(t[0, n], expect)
    assert torch.equal(unpatchify_1d(t, C), x)


def test_patchify_rejects_ragged_length():
    with pytest.raises(ValueError):
        patchify_1d(torch.zeros(1, 2, 21), 5)
