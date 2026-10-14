"""Tests for the conditional NB-VAE baseline and its two encoder modes."""

from __future__ import annotations

import pytest
import torch

from ssllab.generative.condition import build_pert_gene_matrix
from ssllab.generative.cvae import ConditionalNBVAE

PERTS = ["control", "A", "B", "A+B"]
N_GENES, LATENT, COND = 12, 4, 6


def _model(encoder_cond: bool) -> ConditionalNBVAE:
    torch.manual_seed(0)
    pert_gene, _ = build_pert_gene_matrix(PERTS)
    return ConditionalNBVAE(n_genes=N_GENES, pert_gene=pert_gene, latent_dim=LATENT, cond_dim=COND,
                            gene_dim=3, hidden=16, encoder_cond=encoder_cond)


def _batch(b: int = 8):
    g = torch.Generator().manual_seed(1)
    counts = torch.poisson(torch.full((b, N_GENES), 3.0), generator=g)
    libsize = counts.sum(1).clamp_min(1.0)
    x = torch.log1p(1e4 * counts / libsize.unsqueeze(-1))
    pid = torch.randint(len(PERTS), (b,), generator=g)
    return x, counts, libsize, pid


def test_default_encoder_is_condition_blind():
    model = _model(encoder_cond=False)
    assert model.enc[0].in_features == N_GENES                  # reads x only
    x, *_ = _batch()
    mu, logvar = model.encode(x)                                 # no pert_id needed
    assert mu.shape == logvar.shape == (x.shape[0], LATENT)


def test_conditional_encoder_reads_the_condition():
    model = _model(encoder_cond=True)
    assert model.enc[0].in_features == N_GENES + COND           # reads [x, c]
    x, _, _, pid = _batch()
    with pytest.raises(ValueError):
        model.encode(x)                                          # c is required in this mode
    mu_a, _ = model.encode(x, pid)
    mu_b, _ = model.encode(x, (pid + 1) % len(PERTS))
    assert not torch.allclose(mu_a, mu_b)                       # the posterior depends on c


@pytest.mark.parametrize("encoder_cond", [False, True])
def test_loss_is_finite_and_trains(encoder_cond):
    model = _model(encoder_cond)
    x, counts, libsize, pid = _batch()
    loss, parts = model.loss(x, counts, libsize, pid)
    assert torch.isfinite(loss) and set(parts) == {"recon", "kl"}
    loss.backward()
    assert model.enc[0].weight.grad is not None


@pytest.mark.parametrize("encoder_cond", [False, True])
def test_from_checkpoint_round_trip(encoder_cond):
    model = _model(encoder_cond)
    pert_gene, _ = build_pert_gene_matrix(PERTS)
    ck = {"state": model.state_dict(), "n_genes": N_GENES, "latent_dim": LATENT, "cond_dim": COND,
          "gene_dim": 3, "hidden": 16, "compose": "additive", "pert_gene": pert_gene,
          "encoder_cond": encoder_cond}
    loaded = ConditionalNBVAE.from_checkpoint(ck)
    assert loaded.encoder_cond is encoder_cond and not loaded.training
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.allclose(model.eval().generate_expression(1, 5, generator=g1),
                          loaded.generate_expression(1, 5, generator=g2))


def test_old_checkpoint_without_flag_loads_as_condition_blind():
    model = _model(encoder_cond=False)
    pert_gene, _ = build_pert_gene_matrix(PERTS)
    ck = {"state": model.state_dict(), "n_genes": N_GENES, "latent_dim": LATENT, "cond_dim": COND,
          "gene_dim": 3, "hidden": 16, "compose": "additive", "pert_gene": pert_gene}   # no key
    assert ConditionalNBVAE.from_checkpoint(ck).encoder_cond is False
