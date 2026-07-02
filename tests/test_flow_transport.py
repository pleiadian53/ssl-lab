"""Tests for the transport parameterization — the flow starting from a real source z0."""

from __future__ import annotations

import torch

from ssllab.generative.flow import VelocityMLP, cfm_loss, euler_sample, linear_interpolant, ot_couple


def test_linear_interpolant_endpoints():
    z0, z1 = torch.randn(4, 8), torch.randn(4, 8)
    zt0, u = linear_interpolant(z0, z1, torch.zeros(4))
    zt1, _ = linear_interpolant(z0, z1, torch.ones(4))
    assert torch.allclose(zt0, z0) and torch.allclose(zt1, z1)
    assert torch.allclose(u, z1 - z0)                          # target velocity = displacement


def test_euler_sample_respects_source_z0():
    # A zero velocity field is the identity ODE -> the integrated output must equal the
    # source exactly. This pins that z0 (not noise) is the starting state in transport mode.
    m = VelocityMLP(data_dim=8, hidden=16, n_layers=2, cond_dim=4)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    z0 = torch.randn(5, 8)
    out = euler_sample(m, 5, 8, n_steps=10, c=torch.randn(5, 4), z0=z0)
    assert torch.allclose(out, z0, atol=1e-6)


def test_euler_sample_default_is_noise():
    # z0=None keeps the original noise->data behavior (backward compatible).
    m = VelocityMLP(data_dim=8, hidden=16, n_layers=2, cond_dim=4)
    out = euler_sample(m, 5, 8, n_steps=5, c=torch.randn(5, 4))
    assert out.shape == (5, 8) and torch.isfinite(out).all()


def test_cfm_loss_accepts_source():
    m = VelocityMLP(data_dim=8, hidden=16, n_layers=2, cond_dim=4)
    z1, z0, c = torch.randn(6, 8), torch.randn(6, 8), torch.randn(6, 4)
    loss = cfm_loss(m, z1, c=c, z0=z0)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_ot_couple_finds_the_optimal_pairing():
    # z0 and z1 are the same two points in swapped order; OT must un-swap z0 so each
    # source sits with its nearest target, and the result must be a permutation of z0.
    z0 = torch.tensor([[0.0], [10.0]])
    z1 = torch.tensor([[10.1], [0.1]])
    out = ot_couple(z0, z1)
    assert torch.allclose(out, torch.tensor([[10.0], [0.0]]))
    # cost after OT is <= cost under the identity pairing
    assert (out - z1).pow(2).sum() <= (z0 - z1).pow(2).sum()


def test_ot_couple_is_a_permutation():
    z0, z1 = torch.randn(16, 5), torch.randn(16, 5)
    out = ot_couple(z0, z1)
    # every original row appears exactly once (a reordering, not a resampling)
    assert sorted(map(tuple, out.tolist())) == sorted(map(tuple, z0.tolist()))
