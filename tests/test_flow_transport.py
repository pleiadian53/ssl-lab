"""Tests for the transport parameterization — the flow starting from a real source z0."""

from __future__ import annotations

import torch

from ssllab.generative.flow import VelocityMLP, cfm_loss, euler_sample, linear_interpolant


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
