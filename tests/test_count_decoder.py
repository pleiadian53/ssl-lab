"""Tests for the NB/ZINB count decoder (the G2 piece for single-cell counts)."""

from __future__ import annotations

import numpy as np
import torch

from ssllab.generative.count_decoder import CountDecoder, nb_nll, zinb_nll


def test_nb_nll_matches_scipy():
    # Our NB NLL must equal scipy's negative-binomial log-pmf (n=kappa, p=kappa/(kappa+mu)).
    from scipy.stats import nbinom

    rng = np.random.default_rng(0)
    B, G = 4, 6
    x = rng.integers(0, 20, size=(B, G)).astype(np.float64)
    mu = rng.uniform(0.5, 10.0, size=(B, G))
    kappa = rng.uniform(0.5, 5.0, size=(G,))

    ours = nb_nll(torch.tensor(x), torch.tensor(mu), torch.tensor(kappa)).item()

    p = kappa / (kappa + mu)                       # broadcast (G,) over (B,G)
    ref = -nbinom.logpmf(x, kappa, p).sum(axis=1).mean()
    assert abs(ours - ref) < 1e-6


def test_decoder_forward_and_loss():
    B, D, G = 8, 32, 50
    dec = CountDecoder(latent_dim=D, n_genes=G)
    z = torch.randn(B, D)
    libsize = torch.randint(2000, 8000, (B,)).float()
    out = dec(z, libsize)

    assert out["rho"].shape == (B, G)
    assert torch.allclose(out["rho"].sum(-1), torch.ones(B), atol=1e-5)   # rate sums to 1
    assert torch.allclose(out["mu"], libsize.unsqueeze(-1) * out["rho"])  # mu = ell * rho
    assert (out["kappa"] > 0).all()

    counts = torch.randint(0, 30, (B, G)).float()
    loss = dec.nll(z, counts, libsize)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert dec.log_kappa.grad is not None  # dispersion is learned


def test_zinb_loss_and_sampling():
    B, D, G = 8, 32, 50
    dec = CountDecoder(latent_dim=D, n_genes=G, zinb=True)
    z = torch.randn(B, D)
    libsize = torch.full((B,), 5000.0)
    counts = torch.randint(0, 30, (B, G)).float()
    counts[counts < 12] = 0  # zero-inflate to exercise the dropout case

    loss = dec.nll(z, counts, libsize)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()

    x = dec.sample_counts(z, libsize)
    assert x.shape == (B, G) and (x >= 0).all() and torch.allclose(x, x.round())
    mu = dec.expected_counts(z, libsize)
    assert mu.shape == (B, G) and (mu >= 0).all()


def test_zinb_nll_finite_direct():
    B, G = 3, 5
    x = torch.randint(0, 10, (B, G)).float()
    mu = torch.rand(B, G) * 5 + 0.5
    kappa = torch.rand(G) + 0.5
    pi = torch.randn(B, G)
    assert torch.isfinite(zinb_nll(x, mu, kappa, pi))
