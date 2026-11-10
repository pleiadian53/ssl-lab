"""Co-adaptation Phase 1: the trainer trains exactly what each mode says, and its outputs load.

All on a tiny synthetic cache (the Perturb-seq loader fixture), CPU, a few seconds.
"""

from __future__ import annotations

import pytest
import torch

from ssllab.checkpoint import load_jepa, save_jepa
from ssllab.coadapt import CoadaptConfig, anchor_weight_by_rule, prepare_jepa, train_coadapt
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.generative.count_decoder import CountDecoder
from ssllab.jepa.model import build_jepa

from test_perturbseq import _toy_cache


def _setup(tmp_path):
    root = _toy_cache(tmp_path, n_hvg=40, n_tokens=8)
    train, _, _ = get_perturbseq_dataloaders(data_dir=root, batch_size=16, split="combo", seed=0)
    meta = train.meta
    torch.manual_seed(0)
    jepa = build_jepa(token_dim=meta["token_dim"], n_tokens=meta["n_tokens"],
                      embed_dim=32, enc_depth=2, n_target=2, reg_coef=0.04)
    path = save_jepa(jepa, tmp_path / "encoder.pt")
    return train, path, meta["n_hvg"]


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_load_jepa_teacher_is_a_copy_of_the_loaded_encoder(tmp_path):
    _, path, _ = _setup(tmp_path)
    jepa = load_jepa(path)
    for ps, pt in zip(jepa.encoder.parameters(), jepa.ema.teacher.parameters()):
        assert torch.equal(ps, pt)
    assert not any(p.requires_grad for p in jepa.parameters())


@pytest.mark.parametrize("mode", ["dec", "joint", "joint_anchor"])
def test_each_mode_trains_what_it_says(tmp_path, mode):
    train, path, n_genes = _setup(tmp_path)
    jepa = prepare_jepa(load_jepa(path), mode, "cpu")
    enc0, pred0 = _snapshot(jepa.encoder), _snapshot(jepa.predictor)
    torch.manual_seed(0)
    decoder = CountDecoder(latent_dim=32, n_genes=n_genes)
    dec0 = _snapshot(decoder)
    cfg = CoadaptConfig(mode=mode, epochs=1)
    if mode == "joint_anchor":
        ref = CountDecoder(latent_dim=32, n_genes=n_genes)
        batches = [b for _, b in zip(range(2), train)]
        rule = anchor_weight_by_rule(jepa, ref, batches, "cpu")
        assert rule["anchor_weight"] == pytest.approx(rule["ref_nb_loss"] / rule["jepa_loss"])
        cfg.anchor_weight = rule["anchor_weight"]
        assert _same(enc0, _snapshot(jepa.encoder))          # measuring the rule changes nothing
    history = train_coadapt(jepa, decoder, train, cfg, "cpu")

    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]["nb_loss"]))
    assert "effective_rank" in history[0] and "feature_std" in history[0]
    assert not _same(dec0, _snapshot(decoder))                # the decoder always trains
    assert _same(enc0, _snapshot(jepa.encoder)) == (mode == "dec")
    assert _same(pred0, _snapshot(jepa.predictor)) == (mode != "joint_anchor")
    if mode == "joint_anchor":
        assert torch.isfinite(torch.tensor(history[0]["jepa_loss"]))
        assert history[0]["grad_anchor"] > 0
    assert ("grad_nb" in history[0]) == (mode != "dec")


def test_anchor_mode_requires_a_weight(tmp_path):
    train, path, n_genes = _setup(tmp_path)
    jepa = prepare_jepa(load_jepa(path), "joint_anchor", "cpu")
    with pytest.raises(ValueError):
        train_coadapt(jepa, CountDecoder(latent_dim=32, n_genes=n_genes), train,
                      CoadaptConfig(mode="joint_anchor", epochs=1), "cpu")
    with pytest.raises(ValueError):
        CoadaptConfig(mode="flow")


def test_trained_encoder_round_trips_through_save_and_load(tmp_path):
    train, path, n_genes = _setup(tmp_path)
    jepa = prepare_jepa(load_jepa(path), "joint", "cpu")
    train_coadapt(jepa, CountDecoder(latent_dim=32, n_genes=n_genes), train,
                  CoadaptConfig(mode="joint", epochs=1), "cpu")
    out = save_jepa(jepa, tmp_path / "coadapted.pt")
    again = load_jepa(out)
    tokens = next(iter(train))["tokens"]
    jepa.eval()
    with torch.no_grad():
        assert torch.allclose(jepa.embed(tokens), again.embed(tokens))


# --------------------------------------------------------------------------- #
# Round 11: the library size in the likelihood, and how sampled counts are normalized.
# --------------------------------------------------------------------------- #
from ssllab.generative.count_decoder import library_size_of, normalize_sampled


def test_library_size_of_reads_the_right_total():
    batch = {"libsize": torch.tensor([30.0, 60.0]), "counts": torch.tensor([[1, 2, 7], [0, 5, 5]])}
    assert torch.equal(library_size_of(batch, "all"), torch.tensor([30.0, 60.0]))
    assert torch.equal(library_size_of(batch, "hvg"), torch.tensor([10.0, 10.0]))
    with pytest.raises(ValueError):
        library_size_of(batch, "genes")


def test_depth_normalization_is_the_published_formula_and_data_normalization_matches_the_cache():
    counts = torch.tensor([[1.0, 3.0, 0.0], [4.0, 4.0, 2.0]])
    ls = torch.tensor([20.0, 20.0])
    assert torch.equal(normalize_sampled(counts, ls, "depth"), torch.log1p(1e4 * counts / ls.unsqueeze(-1)))
    data = normalize_sampled(counts, ls, "data")
    assert torch.allclose(torch.expm1(data).sum(1), torch.full((2,), 1e4))   # CP10K per cell, as the cache


def test_hvg_library_size_trains_and_is_recorded(tmp_path):
    train, path, n_genes = _setup(tmp_path)
    jepa = prepare_jepa(load_jepa(path), "dec", "cpu")
    hist = train_coadapt(jepa, CountDecoder(latent_dim=32, n_genes=n_genes), train,
                         CoadaptConfig(mode="dec", epochs=1, library_size="hvg"), "cpu")
    assert torch.isfinite(torch.tensor(hist[0]["nb_loss"]))
    with pytest.raises(ValueError):
        CoadaptConfig(library_size="genes")


def test_cvae_records_library_size_and_normalizations():
    from ssllab.generative.cvae import ConditionalNBVAE
    pert_gene = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    torch.manual_seed(0)
    m = ConditionalNBVAE(n_genes=6, pert_gene=pert_gene, latent_dim=4, cond_dim=4, gene_dim=4, hidden=8,
                         library_size="hvg")
    ck = {"state": m.state_dict(), "n_genes": 6, "pert_gene": pert_gene, "latent_dim": 4, "cond_dim": 4,
          "gene_dim": 4, "hidden": 8, "compose": "additive", "library_size": "hvg"}
    again = ConditionalNBVAE.from_checkpoint(ck)
    assert again.library_size == "hvg"
    old = {k: v for k, v in ck.items() if k != "library_size"}
    assert ConditionalNBVAE.from_checkpoint(old).library_size == "all"
    ls = torch.full((5,), 100.0)
    # The Gamma-Poisson draws use the global RNG (only z uses the generator), so reseed both.
    torch.manual_seed(7)
    a = m.generate_population(1, 5, ls, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(7)
    b = m.generate_population(1, 5, ls, generator=torch.Generator().manual_seed(3), normalize="data")
    assert torch.equal(a, b)                                    # the default is data-like since round 11
    torch.manual_seed(7)
    c = m.generate_population(1, 5, ls, generator=torch.Generator().manual_seed(3), normalize="depth")
    torch.manual_seed(7)
    z = torch.randn(5, 4, generator=torch.Generator().manual_seed(3))
    out = m.decode(z, torch.full((5,), 1, dtype=torch.long))
    mu = ls.unsqueeze(-1) * out["rho"]
    lam = torch.distributions.Gamma(out["kappa"].expand_as(mu), out["kappa"].expand_as(mu) / (mu + 1e-8)).sample()
    assert torch.allclose(c, torch.log1p(1e4 * torch.poisson(lam) / ls.unsqueeze(-1)))   # the published formula
