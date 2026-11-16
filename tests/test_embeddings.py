"""The embedding interface: stores, latent sources, projection, and the space check.

All on the tiny synthetic Perturb-seq cache used by the loader tests, CPU, a few seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ssllab.checkpoint import load_jepa, save_jepa
from ssllab.coadapt import CoadaptConfig, prepare_jepa, train_coadapt
from ssllab.data.perturbseq import (
    SPLIT_TRAIN,
    cache_fingerprint,
    get_perturbseq_dataloaders,
    load_cache,
    make_gene_partition,
)
from ssllab.embeddings import (
    JepaSource,
    PrecomputedSource,
    StoreError,
    check_same_space,
    fit_pca,
    jepa_space_key,
    load_store,
    project_store,
    write_store,
)
from ssllab.generative.count_decoder import CountDecoder
from ssllab.generative.perturb import predicted_expression
from ssllab.jepa.model import build_jepa
from ssllab.latents import encode_rows

from test_perturbseq import _toy_cache


def _store(tmp_path, n=12, d=5, rows=None, fp="fp", name="s"):
    rng = np.random.default_rng(0)
    rows = np.arange(0, 2 * n, 2) if rows is None else rows
    return write_store(tmp_path / name, rng.normal(size=(len(rows), d)).astype(np.float32), rows,
                       source="test", dataset_fingerprint=fp, model="toy")


# --------------------------------------------------------------------------- #
# The store contract.
# --------------------------------------------------------------------------- #
def test_store_round_trip_and_manifest(tmp_path):
    s = _store(tmp_path)
    m = json.loads((s.path / "manifest.json").read_text())
    for k in ("source", "width", "n", "dataset_fingerprint", "content_md5", "space_key"):
        assert k in m
    assert m["space_key"] == f"store:{m['content_md5']}" and m["model"] == "toy"
    again = load_store(s.path, expect_fingerprint="fp", verify_content=True)
    assert np.array_equal(np.asarray(again.embeddings), np.asarray(s.embeddings))
    assert again.width == 5 and again.n == 12


def test_lookup_follows_the_requested_order_and_refuses_missing_rows(tmp_path):
    s = _store(tmp_path)
    E = np.asarray(s.embeddings)
    got = s.lookup(np.array([6, 0, 22]))
    assert np.array_equal(got, E[[3, 0, 11]])
    with pytest.raises(StoreError, match="not in the store"):
        s.lookup(np.array([0, 1]))                      # row 1 was never stored
    with pytest.raises(StoreError):
        s.lookup(np.array([100]))


@pytest.mark.parametrize("rows, values, match", [
    (np.array([0, 2, 2]), None, "strictly increasing"),
    (np.array([2, 0, 4]), None, "strictly increasing"),
    (np.array([0, 1, 2]), np.array([[0.0], [np.nan], [1.0]]), "non-finite"),
])
def test_write_store_rejects_bad_inputs(tmp_path, rows, values, match):
    E = np.ones((len(rows), 1), np.float32) if values is None else values.astype(np.float32)
    with pytest.raises(StoreError, match=match):
        write_store(tmp_path / "bad", E, rows, source="test", dataset_fingerprint="fp")


def test_load_store_checks_the_dataset_and_the_contents(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(StoreError, match="different dataset"):
        load_store(s.path, expect_fingerprint="another")
    E = np.load(s.path / "embeddings.npy")
    E[0, 0] += 1.0
    np.save(s.path / "embeddings.npy", E)
    load_store(s.path)                                   # shape and values still valid ...
    with pytest.raises(StoreError, match="content_md5"):
        load_store(s.path, verify_content=True)          # ... but no longer what was written


def test_write_store_owns_the_required_fields(tmp_path):
    with pytest.raises(StoreError, match="set by write_store"):
        write_store(tmp_path / "x", np.ones((2, 2), np.float32), np.arange(2),
                    source="test", dataset_fingerprint="fp", width=3)


# --------------------------------------------------------------------------- #
# The cache side: row keys and the fingerprint.
# --------------------------------------------------------------------------- #
def test_every_batch_carries_its_cache_rows(tmp_path):
    root = _toy_cache(tmp_path, n_hvg=40, n_tokens=8)
    cache = load_cache(root / "norman2019")
    train, val, _ = get_perturbseq_dataloaders(data_dir=root, batch_size=16, split="combo", seed=0)
    assert train.meta["fingerprint"] == cache_fingerprint(cache)
    for loader in (train, val):
        b = next(iter(loader))
        r = b["row"].numpy()
        assert np.array_equal(b["features"].numpy(), cache.hvg_X[r])
        assert np.array_equal(b["pert_id"].numpy(), cache.pert_id[r])


def test_fingerprint_changes_with_the_cell_order(tmp_path):
    cache = load_cache(_toy_cache(tmp_path, n_hvg=40, n_tokens=8) / "norman2019")
    fp = cache_fingerprint(cache)
    cache.split_combo = cache.split_combo.copy()
    cache.split_combo[0] = (cache.split_combo[0] + 1) % 3
    assert cache_fingerprint(cache) != fp


# --------------------------------------------------------------------------- #
# Sources.
# --------------------------------------------------------------------------- #
def _toy_jepa(tmp_path):
    root = _toy_cache(tmp_path, n_hvg=40, n_tokens=8)
    cache = load_cache(root / "norman2019")
    part = make_gene_partition(cache.n_hvg, 8, 0)
    torch.manual_seed(0)
    jepa = build_jepa(token_dim=part.shape[1], n_tokens=8, embed_dim=32, enc_depth=2, n_target=2)
    path = save_jepa(jepa, tmp_path / "encoder.pt")
    return root, cache, part, path


def test_jepa_source_is_encode_rows_and_its_key_survives_a_resave(tmp_path):
    root, cache, part, path = _toy_jepa(tmp_path)
    jepa = load_jepa(path)
    src = JepaSource(jepa, path, cache.hvg_X, part)
    rows = np.array([5, 1, 9])
    assert torch.equal(src.encode(rows), encode_rows(jepa, cache.hvg_X, part, rows, "cpu"))
    resaved = save_jepa(load_jepa(path), tmp_path / "resaved.pt")
    assert jepa_space_key(load_jepa(resaved)) == src.space_key
    assert src.identity()["kind"] == "jepa" and src.dim == 32


def test_dec_mode_on_a_store_of_the_encoders_own_latents_matches_the_live_encoder(tmp_path):
    root, cache, part, path = _toy_jepa(tmp_path)
    all_rows = np.arange(len(cache.pert_id))
    Z = encode_rows(load_jepa(path), cache.hvg_X, part, all_rows, "cpu").numpy()
    store = write_store(tmp_path / "jepa_store", Z, all_rows, source="jepa",
                        dataset_fingerprint=cache_fingerprint(cache),
                        space_key=jepa_space_key(load_jepa(path)))
    src = PrecomputedSource.open(store.path, cache_fingerprint(cache))
    assert src.space_key == JepaSource(load_jepa(path), path, cache.hvg_X, part).space_key

    def run(latents):
        train, _, _ = get_perturbseq_dataloaders(data_dir=root, batch_size=16, split="combo", seed=0)
        jepa = None if latents else prepare_jepa(load_jepa(path), "dec", "cpu")
        torch.manual_seed(0)
        dec = CountDecoder(latent_dim=32, n_genes=cache.n_hvg)
        hist = train_coadapt(jepa, dec, train, CoadaptConfig(mode="dec", epochs=2), "cpu", latents=latents)
        return dec, hist

    live, h_live = run(None)
    stored, h_store = run(src)
    assert h_live[-1]["nb_loss"] == pytest.approx(h_store[-1]["nb_loss"], rel=1e-5)
    for a, b in zip(live.parameters(), stored.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


def test_precomputed_latents_are_only_for_dec_mode(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="only defined for mode 'dec'"):
        train_coadapt(None, CountDecoder(latent_dim=5, n_genes=3), [], CoadaptConfig(mode="joint", epochs=1),
                      latents=PrecomputedSource(s))


# --------------------------------------------------------------------------- #
# Projection.
# --------------------------------------------------------------------------- #
def test_fit_pca_matches_svd_and_is_fit_on_the_given_rows_only(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6)) @ np.diag([5, 3, 2, 1, 0.5, 0.1])
    mean, comps, evals = fit_pca(X, 3)
    _, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    assert np.allclose(np.abs(comps), np.abs(vt[:3]), atol=1e-8)
    assert np.allclose(evals[:6], s ** 2 / 199)

    rows = np.arange(200)
    parent = write_store(tmp_path / "p", X.astype(np.float32), rows, source="test", dataset_fingerprint="fp")
    fit = rows[:120]
    a = project_store(parent, fit, 3, tmp_path / "a", fit_description="first 120 rows")
    X2 = X.copy()
    X2[120:] *= 50                                       # change only rows the fit never sees
    parent2 = write_store(tmp_path / "p2", X2.astype(np.float32), rows, source="test", dataset_fingerprint="fp")
    b = project_store(parent2, fit, 3, tmp_path / "b", fit_description="first 120 rows")
    pa, pb = np.load(a.path / "projection.npz"), np.load(b.path / "projection.npz")
    assert np.allclose(pa["components"], pb["components"]) and np.allclose(pa["mean"], pb["mean"])
    assert np.allclose(np.asarray(a.embeddings)[:120], np.asarray(b.embeddings)[:120], atol=1e-5)
    m = a.manifest
    assert m["source"] == "projection" and m["parent_width"] == 6 and m["width"] == 3
    assert m["n_fit_rows"] == 120 and 0 < m["explained_variance_share"] <= 1
    assert m["parent_space_key"] == parent.space_key and a.space_key != parent.space_key


# --------------------------------------------------------------------------- #
# Pairing artifacts from different spaces is refused.
# --------------------------------------------------------------------------- #
def test_check_same_space():
    a, b = {"space_key": "store:1"}, {"space_key": "jepa:2"}
    check_same_space(a, dict(a))
    check_same_space(None, a)                            # older checkpoints carry no identity
    with pytest.raises(ValueError, match="different latent spaces"):
        check_same_space(a, b)


def test_predicted_expression_refuses_a_flow_and_decoder_from_different_spaces():
    dec = CountDecoder(latent_dim=4, n_genes=3)
    dec.latent_source = {"space_key": "jepa:abc"}
    bundle = {"latent_source": {"space_key": "store:def"}}
    with pytest.raises(ValueError, match="different latent spaces"):
        predicted_expression(bundle, dec, pert_id=0, n=2)


def test_combo_train_rows_cover_the_fit_set(tmp_path):
    """The comparison fits projections on combo-train rows; the toy cache has them."""
    cache = load_cache(_toy_cache(tmp_path, n_hvg=40, n_tokens=8) / "norman2019")
    assert (cache.split_combo == SPLIT_TRAIN).sum() > 0


# --------------------------------------------------------------------------- #
# The flow trainer records its optimizer settings and its training history.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra, lr, clip", [([], 1e-3, None), (["--lr", "3e-4", "--grad-clip", "1.0"], 3e-4, 1.0)])
def test_flow_trainer_records_lr_clip_and_history(tmp_path, extra, lr, clip):
    import subprocess
    import sys
    from pathlib import Path

    root = _toy_cache(tmp_path)
    cache = load_cache(root / "norman2019")
    rows = np.arange(len(cache.pert_id))
    z = np.random.default_rng(0).normal(size=(len(rows), 6)).astype(np.float32)
    store = write_store(tmp_path / "store", z, rows, source="test", dataset_fingerprint=cache_fingerprint(cache))
    script = Path(__file__).resolve().parents[1] / "examples/perturbation_response/04_train_cond_flow.py"
    subprocess.run([sys.executable, str(script), "--experiment", "flow", "--latents", str(store.path),
                    "--output-root", str(tmp_path / "out"), "--data-dir", str(root), "--split", "combo",
                    "--cond-type", "geneset", "--flow-base", "control", "--epochs", "2", "--batch-size", "16",
                    "--hidden", "16", "--device", "cpu", *extra], check=True, capture_output=True)
    ck = torch.load(tmp_path / "out/flow/checkpoints/cond_flow.pt", map_location="cpu", weights_only=False)
    assert ck["lr"] == pytest.approx(lr) and ck["grad_clip"] == clip
    rep = json.loads((tmp_path / "out/flow/reports/stage_b_flow.json").read_text())
    assert rep["lr"] == pytest.approx(lr) and rep["grad_clip"] == clip
    assert len(rep["history"]["cfm_loss"]) == 2 and all(np.isfinite(rep["history"]["grad_norm_max"]))
