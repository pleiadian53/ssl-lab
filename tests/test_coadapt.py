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
