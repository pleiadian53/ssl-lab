"""CPU-fast smoke tests for the Perturb-seq data pipeline (Phase 1a).

The loader/tokenizer tests are dependency-free (torch + numpy): they build a tiny
synthetic cache directly and assert the token contract, covariates, splits, and
baseline-sampling behavior. The end-to-end processing test is guarded by
``importorskip`` so it runs only when scanpy/anndata are installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssllab.data.perturbseq import (
    ControlSampler,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    detokenize_cells,
    get_perturbseq_dataloaders,
    make_gene_partition,
    token_dim_for,
    tokenize_cells,
    write_cache,
)
from ssllab.jepa.model import build_jepa


# --------------------------------------------------------------------------- #
# A tiny synthetic cache (no anndata) for the light loader/tokenizer tests.
# --------------------------------------------------------------------------- #
PERT_NAMES = np.array(["control", "A", "B", "C", "D", "A+B", "C+D"], dtype=object)


def _toy_cache(tmp_path: Path, n=160, n_hvg=40, n_tokens=8, seed=0) -> Path:
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(n, n_hvg)).astype(np.int32)
    libsize = counts.sum(1).astype(np.float32)
    hvg_X = np.log1p(counts / (libsize[:, None] + 1e-6) * 1e4).astype(np.float32)

    pert_id = rng.integers(0, len(PERT_NAMES), size=n).astype(np.int64)
    is_control = pert_id == 0
    ctrl_group = np.zeros(n, dtype=np.int64)

    # combo split mirroring the real scheme: combos held out, singles+control train.
    name_split = {"control": SPLIT_TRAIN, "A": SPLIT_TRAIN, "B": SPLIT_TRAIN,
                  "C": SPLIT_TRAIN, "D": SPLIT_TRAIN, "A+B": SPLIT_TEST, "C+D": SPLIT_VAL}
    split_combo = np.array([name_split[PERT_NAMES[i]] for i in pert_id], dtype=np.int8)
    r = rng.random(n)
    split_cells = np.full(n, SPLIT_TRAIN, dtype=np.int8)
    split_cells[r < 0.15] = SPLIT_VAL
    split_cells[r >= 0.85] = SPLIT_TEST

    de_genes = {"method": "wilcoxon", "vs": "control", "top_k": 5, "gene_space": "hvg_index",
                "per_pert": {p: {"top_idx": sorted(rng.choice(n_hvg, 5, replace=False).tolist())}
                             for p in PERT_NAMES if p != "control"}}
    manifest = {"artifact_version": "1", "n_hvg": n_hvg,
                "tokenization": {"n_tokens": n_tokens, "token_dim": token_dim_for(n_hvg, n_tokens),
                                 "partition_seed": seed, "scheme": "random"}}
    cache = tmp_path / "norman2019"
    write_cache(
        cache, hvg_X=hvg_X, counts=counts, libsize=libsize, pert_id=pert_id,
        is_control=is_control, ctrl_group=ctrl_group, split_combo=split_combo,
        split_cells=split_cells, gene_ids=np.array([f"G{i}" for i in range(n_hvg)], dtype=object),
        pert_names=PERT_NAMES, splits={"combo": {}, "cells": {}}, de_genes=de_genes, manifest=manifest,
    )
    return tmp_path


def test_tokenize_roundtrip():
    # Divisible and non-divisible panels (the latter exercises padding).
    for n_hvg, n_tokens in [(40, 8), (37, 8)]:
        part = make_gene_partition(n_hvg, n_tokens, seed=0)
        assert part.shape == (n_tokens, token_dim_for(n_hvg, n_tokens))
        x = torch.randn(5, n_hvg)
        tok = tokenize_cells(x, part)
        assert tok.shape == (5, n_tokens, token_dim_for(n_hvg, n_tokens))
        # Inverse recovers the original at the (unique) partitioned positions.
        rec = detokenize_cells(tok, part, n_hvg)
        assert torch.allclose(rec, x, atol=1e-6)


def test_partition_reproducible_and_seed_guard(tmp_path):
    a = make_gene_partition(40, 8, seed=3)
    b = make_gene_partition(40, 8, seed=3)
    assert torch.equal(a, b)
    assert not torch.equal(a, make_gene_partition(40, 8, seed=4))

    root = _toy_cache(tmp_path, seed=0)
    with pytest.raises(ValueError, match="seed mismatch"):
        get_perturbseq_dataloaders(data_dir=root, seed=99)  # cache built with seed 0


def test_loader_contract_and_encoder(tmp_path):
    root = _toy_cache(tmp_path, n_hvg=40, n_tokens=8)
    train, val, test = get_perturbseq_dataloaders(
        data_dir=root, batch_size=16, split="combo", seed=0
    )
    batch = next(iter(train))
    td = token_dim_for(40, 8)
    assert batch["tokens"].shape == (16, 8, td) and batch["tokens"].dtype == torch.float32
    assert batch["counts"].shape == (16, 40) and not torch.is_floating_point(batch["counts"])
    for k in ("libsize", "pert_id", "is_control", "ctrl_group"):
        assert batch[k].shape == (16,)
    assert batch["is_control"].dtype == torch.bool

    # The token contract feeds the modality-agnostic JEPA encoder unchanged.
    jepa = build_jepa(token_dim=td, n_tokens=8, embed_dim=32)
    z = jepa.embed(batch["tokens"])
    assert z.shape == (16, 32)
    loss = z.pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in jepa.encoder.parameters())


def test_raw_counts_preserved(tmp_path):
    root = _toy_cache(tmp_path)
    train, _, _ = get_perturbseq_dataloaders(data_dir=root, batch_size=32)
    batch = next(iter(train))
    counts = batch["counts"]
    assert (counts >= 0).all()
    # Library size tracks the raw count total (the NB-decoder covariate seam).
    corr = np.corrcoef(counts.sum(1).numpy(), batch["libsize"].numpy())[0, 1]
    assert corr > 0.99


def test_combo_split_disjoint_and_controls_in_train(tmp_path):
    root = _toy_cache(tmp_path)
    train, val, test = get_perturbseq_dataloaders(data_dir=root, batch_size=8, split="combo", seed=0)
    names = train.meta["pert_names"]

    def perts_in(loader):
        seen = set()
        for b in loader:
            seen.update(names[i] for i in b["pert_id"].tolist())
        return seen

    tr, va, te = perts_in(train), perts_in(val), perts_in(test)
    assert tr.isdisjoint(te) and tr.isdisjoint(va) and va.isdisjoint(te)
    assert "A+B" in te and "C+D" in va           # combos held out
    assert {"A", "B", "C", "D"} <= tr             # singles seen in train
    assert "control" in tr


def test_control_sampler(tmp_path):
    root = _toy_cache(tmp_path)
    train, _, _ = get_perturbseq_dataloaders(data_dir=root)
    sampler = train.control_sampler
    assert isinstance(sampler, ControlSampler)
    bundle = train.dataset.b  # the loaded CacheBundle
    idx = sampler.sample(group=0, k=20)
    assert idx.shape == (20,)
    assert bundle.is_control[idx.numpy()].all()
    assert (bundle.ctrl_group[idx.numpy()] == 0).all()


# --------------------------------------------------------------------------- #
# End-to-end processing — needs scanpy/anndata (skipped otherwise).
# --------------------------------------------------------------------------- #
def _load_process_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "perturbation_response" / "00_process_norman.py"
    spec = importlib.util.spec_from_file_location("process_norman", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_process_end_to_end(tmp_path):
    pytest.importorskip("scanpy")
    pytest.importorskip("anndata")
    proc = _load_process_module()

    import types

    adata = proc.synthetic_adata(n_cells=400, n_genes=80, seed=0)
    # Build an args namespace directly (avoid argparse).
    args = types.SimpleNamespace(
        source="h5ad", smoke=True, data_dir=str(tmp_path), artifact="norman2019",
        n_hvg=40, n_tokens=8, de_top_k=5,
        min_genes_per_cell=1, max_pct_mito=100.0, min_cells_per_gene=1, min_cells_per_pert=5,
        val_frac=0.2, test_frac=0.2, seed=0,
    )
    manifest = proc.process(adata, args)
    assert manifest["n_hvg"] == 40 and manifest["n_perts"] >= 5

    cache = tmp_path / "norman2019"
    de = json.loads((cache / "de_genes.json").read_text())
    for entry in de["per_pert"].values():
        assert len(entry["top_idx"]) <= 5
        assert all(0 <= i < 40 for i in entry["top_idx"]) and len(set(entry["top_idx"])) == len(entry["top_idx"])

    splits = json.loads((cache / "splits.json").read_text())["combo"]
    tr, va, te = set(splits["train_perts"]), set(splits["val_perts"]), set(splits["test_perts"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert "control" in tr

    # The processed cache loads and feeds the encoder.
    train, _, _ = get_perturbseq_dataloaders(data_dir=tmp_path, batch_size=8, seed=0)
    batch = next(iter(train))
    jepa = build_jepa(token_dim=batch["tokens"].shape[-1], n_tokens=batch["tokens"].shape[1], embed_dim=32)
    assert jepa.embed(batch["tokens"]).shape[0] == batch["tokens"].shape[0]
