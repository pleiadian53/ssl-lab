"""Input preparation for foundation-encoder embeddings (examples/perturbation_response/foundation-encoder).

Only the parts that do not need Geneformer or CUDA: cell selection, reading raw count rows from an
h5ad layer, the tokenizer's input, the processed→raw index, and re-ordering embeddings by cell.
A misalignment here would silently pair embeddings with the wrong cells, so each step is checked
against values computed independently on a toy file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

_DIR = Path(__file__).resolve().parents[1] / "examples" / "perturbation_response" / "foundation-encoder"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


embed = _load("gf_embed", "02_embed_geneformer.py")
index_mod = _load("gf_index", "01_build_cell_index.py")


def _toy_raw(tmp_path: Path, n_cells: int = 30, n_genes: int = 12, seed: int = 0) -> tuple[Path, ad.AnnData]:
    rng = np.random.default_rng(seed)
    counts = sp.random(n_cells, n_genes, density=0.4, format="csr", random_state=seed,
                       data_rvs=lambda k: rng.integers(1, 20, size=k)).astype(np.float32)
    var = pd.DataFrame({"index": [f"ENSG{j:011d}" for j in range(n_genes)]},
                       index=pd.Index([f"GENE{j}" for j in range(n_genes)], name="gene_symbols"))
    obs = pd.DataFrame({"total_counts": np.asarray(counts.sum(1)).ravel()},
                       index=[f"BC{i:03d}" for i in range(n_cells)])
    a = ad.AnnData(X=np.log1p(counts.toarray()), obs=obs, var=var, layers={"counts": counts})
    path = tmp_path / "raw.h5ad"
    a.write_h5ad(path)
    return path, a


def test_select_cells_sorted_unique_and_all():
    s = embed.select_cells(100, 10, seed=3)
    assert len(s) == 10 and np.all(np.diff(s) > 0) and s.max() < 100
    assert np.array_equal(embed.select_cells(100, 0, seed=3), np.arange(100))
    assert np.array_equal(embed.select_cells(100, 500, seed=3), np.arange(100))
    assert np.array_equal(s, embed.select_cells(100, 10, seed=3))


def test_read_count_rows_matches_layer(tmp_path):
    path, a = _toy_raw(tmp_path)
    rows = np.array([1, 4, 5, 17, 29])
    counts, ens = embed.read_count_rows(path, rows, chunk=2)  # chunk < len(rows) exercises stacking
    assert np.array_equal(counts.toarray(), a.layers["counts"][rows].toarray())
    assert list(ens) == list(a.var["index"])


def test_build_count_adata_fields(tmp_path):
    path, a = _toy_raw(tmp_path)
    rows = np.array([0, 2, 9])
    counts, ens = embed.read_count_rows(path, rows)
    x = embed.build_count_adata(counts, ens, cell_idx=np.array([10, 20, 30]))
    assert np.allclose(x.obs["n_counts"], a.obs["total_counts"].to_numpy()[rows])
    assert list(x.obs["cell_idx"]) == [10, 20, 30]
    assert list(x.var["ensembl_id"]) == list(a.var["index"])
    assert np.array_equal(x.X.toarray(), a.layers["counts"][rows].toarray())  # X is raw counts, not log


def test_align_embeddings_reorders_by_cell_and_rejects_duplicates():
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    df = pd.DataFrame(emb)
    df["cell_idx"] = [7, 2, 9, 4]          # extractor order: sorted by length, not by cell
    out, idx = embed.align_embeddings(df, dim=3)
    assert list(idx) == [2, 4, 7, 9]
    assert np.array_equal(out[0], emb[1]) and np.array_equal(out[3], emb[2])
    df.loc[0, "cell_idx"] = 2
    with pytest.raises(ValueError):
        embed.align_embeddings(df, dim=3)


def test_build_index_maps_processed_to_raw(tmp_path):
    _, raw = _toy_raw(tmp_path)
    keep = np.array([0, 3, 4, 10, 20])
    proc = ad.AnnData(obs=pd.DataFrame({"perturbation": ["control", "A", "A+B", "B", "control"],
                                        "split_combo": [0, 0, 2, 1, 0],
                                        "is_control": [True, False, False, False, True]},
                                       index=raw.obs_names[keep]))
    idx = index_mod.build_index(raw, proc)
    assert np.array_equal(idx["raw_row"], keep)
    assert np.allclose(idx["total_counts"], raw.obs["total_counts"].to_numpy()[keep])
    assert idx["split_combo"].dtype == np.int8
    with pytest.raises(ValueError):          # out of raw order
        index_mod.build_index(raw, proc[[1, 0, 2, 3, 4]].copy())
    missing = ad.AnnData(obs=pd.DataFrame(index=["NOT_A_CELL"]))
    with pytest.raises(ValueError):
        index_mod.build_index(raw, missing)


def test_in_vocab_gene_counts_maps_merges_and_skips():
    # genes: g0 in vocab; g1 retired ID renamed to g0's token (merges with g0); g2 out of vocab;
    # g3 in vocab, lower-case in the data; g4 absent from the mapping dictionary
    ens = np.array(["ENSG0", "ENSG_OLD", "ENSG2", "ensg3", "ENSG4"])
    mapping = {"ENSG0": "ENSG0", "ENSG_OLD": "ENSG0", "ENSG2": "ENSG2", "ENSG3": "ENSG3"}
    vocab = {"ENSG0", "ENSG3", "<cls>", "<eos>"}
    counts = sp.csr_matrix(np.array([
        [1, 1, 5, 2, 9],   # g0 + g1 merge to one token, g3 one token -> 2
        [0, 3, 0, 0, 0],   # only the retired ID expressed -> 1 (via its new ID)
        [0, 0, 4, 0, 7],   # nothing in vocab -> 0
    ], dtype=np.float32))
    assert list(embed.in_vocab_gene_counts(counts, ens, mapping, vocab)) == [2, 1, 0]


def test_describe_reports_rank_and_shape():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 8)).astype(np.float32)
    d = embed.describe(z)
    assert d["dim"] == 8 and d["n"] == 200 and d["n_nonfinite"] == 0
    assert 6 < d["effective_rank"] <= 8


def test_rank_entry_reports_fraction_of_ceiling():
    ref = _load("gf_ref", "03_reference_ranks.py")
    rng = np.random.default_rng(1)
    wide = ref.rank_entry(rng.normal(size=(50, 80)), embed.describe)      # more columns than cells
    assert wide["rank_ceiling"] == 50 and 0 < wide["rank_fraction_of_ceiling"] <= 1
    narrow = ref.rank_entry(rng.normal(size=(300, 10)), embed.describe)
    assert narrow["rank_ceiling"] == 10
    assert np.isclose(narrow["rank_fraction_of_ceiling"], narrow["effective_rank"] / 10)
