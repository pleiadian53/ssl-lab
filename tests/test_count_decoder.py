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


# --- B1: identity-anchored mean head ----------------------------------------- #
def test_anchored_mean_head_starts_at_baseline():
    # With anchored_mean, rho = softmax(log_rho_base + delta(z)); delta is zero-init,
    # so after set_baseline_profile every cell decodes to the baseline regardless of z.
    B, D, G = 8, 32, 50
    dec = CountDecoder(latent_dim=D, n_genes=G, anchored_mean=True)
    ctrl = torch.rand(G) + 0.1
    ctrl = ctrl / ctrl.sum()
    dec.set_baseline_profile(ctrl)
    z = torch.randn(B, D)
    libsize = torch.full((B,), 5000.0)
    out = dec(z, libsize)
    assert torch.allclose(out["rho"].sum(-1), torch.ones(B), atol=1e-5)
    assert torch.allclose(out["rho"], ctrl.expand(B, G), atol=1e-5)  # starts at baseline
    counts = torch.randint(0, 30, (B, G)).float()
    dec.nll(z, counts, libsize).backward()
    assert dec.log_rho_base.grad is not None                        # baseline is learned


def test_set_baseline_profile_requires_flag():
    dec = CountDecoder(latent_dim=16, n_genes=10)  # anchored_mean=False
    assert dec.log_rho_base is None
    try:
        dec.set_baseline_profile(torch.ones(10))
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# --- B2: state-aware, anchored dispersion ------------------------------------ #
def test_state_dispersion_is_per_cell():
    B, D, G = 8, 32, 50
    dec = CountDecoder(latent_dim=D, n_genes=G, state_dispersion=True)
    z = torch.randn(B, D)
    libsize = torch.full((B,), 5000.0)
    out = dec(z, libsize)
    assert out["kappa"].shape == (B, G)    # per cell, not a single (G,) constant
    assert (out["kappa"] > 0).all()
    assert dec.log_kappa is None
    counts = torch.randint(0, 30, (B, G)).float()
    dec.nll(z, counts, libsize).backward()
    assert dec.kappa_head.weight.grad is not None


def test_moment_dispersion_recovers_kappa():
    # NB variance = mu + mu^2/kappa, so the method of moments inverts to kappa exactly.
    mu = torch.tensor([5.0, 2.0, 10.0])
    kappa = torch.tensor([2.0, 0.5, 4.0])
    var = mu + mu ** 2 / kappa
    assert torch.allclose(CountDecoder.moment_dispersion(mu, var), kappa, atol=1e-3)
    # Non-overdispersed genes (var <= mean) clamp toward Poisson (large kappa).
    assert (CountDecoder.moment_dispersion(torch.tensor([3.0]), torch.tensor([1.0])) > 1e3).all()


def test_dispersion_anchor_adds_penalty_and_backprops():
    B, D, G = 16, 32, 20
    dec = CountDecoder(latent_dim=D, n_genes=G, state_dispersion=True)
    z = torch.randn(B, D)
    libsize = torch.full((B,), 5000.0)
    counts = torch.randint(0, 30, (B, G)).float()
    target = torch.full((G,), 3.0)                 # init kappa ~ 0.69, so the anchor is nonzero
    base = dec.loss(z, counts, libsize)
    anchored = dec.loss(z, counts, libsize, kappa_target=target, anchor_weight=1.0)
    assert torch.isfinite(anchored) and anchored.requires_grad
    assert anchored.item() > base.item()
    anchored.backward()
    assert dec.kappa_head.weight.grad is not None


def test_flag_checkpoint_roundtrip():
    B, D, G = 4, 32, 20
    dec = CountDecoder(latent_dim=D, n_genes=G, anchored_mean=True, state_dispersion=True)
    dec.set_baseline_profile(torch.rand(G) + 0.1)
    state = dec.state_dict()
    # State-dict keys track the config: anchored/state params present, constant absent.
    assert "log_rho_base" in state and "kappa_head.weight" in state and "log_kappa" not in state
    dec2 = CountDecoder(latent_dim=D, n_genes=G, anchored_mean=True, state_dispersion=True)
    dec2.load_state_dict(state)
    z, ell = torch.randn(B, D), torch.full((B,), 3000.0)
    assert torch.allclose(dec(z, ell)["rho"], dec2(z, ell)["rho"], atol=1e-6)
    # The default decoder keeps the constant per-gene dispersion, for backward compat.
    assert "log_kappa" in CountDecoder(latent_dim=D, n_genes=G).state_dict()
