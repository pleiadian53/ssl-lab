#!/usr/bin/env python3
"""Embed Norman cells with pretrained Geneformer models, in the processed-cache order. Pod only.

**What this measures.** Route item R2.7 asks whether a single-cell foundation model, pretrained on
tens of millions of cells, gives a better starting representation for perturbation prediction than
our own JEPA encoder, which saw only the ~96,000 training cells. Before any protocol, the first
step is practical: can the models read these cells, how long does embedding take, how much GPU
memory does it need, and what do the embeddings look like. With ``--n-cells 2000`` this script is
that feasibility check; with ``--n-cells 0`` it embeds every cell for the later comparison.

**How Geneformer reads a cell.** Geneformer does not take expression values. It turns each cell
into an ordered list of genes (a "rank value encoding"):

1. the cell's raw counts over all measured genes are divided by the cell's total count and scaled
   to 10,000, as in ordinary library-size normalization;
2. each gene's value is then divided by that gene's median non-zero value across the pretraining
   corpus, so a gene that is always high (a ribosomal protein) does not dominate, and a gene
   that is rarely high moves up when it is;
3. the expressed genes are sorted by that ratio, highest first, and each gene becomes one token.

The first 2,048 genes (V1 models) or 4,096 (V2 models) are kept. V2 also adds a ``<cls>`` token at
the front and an ``<eos>`` token at the end. A cell embedding is either the hidden state of the
``<cls>`` token (``cls``, V2 only) or the mean hidden state over the gene tokens (``cell``). Both
come from the second-to-last layer, Geneformer's recommended default. Only protein-coding and
microRNA genes have tokens. Long non-coding RNAs such as MALAT1 and NEAT1 are invisible to the
model, although our decoder still predicts them from the embedding.

**Alignment.** The input is built from the raw file's count layer for the rows listed in
``foundation/cell_index.npz`` (written by ``01_build_cell_index.py``). Each cell carries its
position in the processed cache as ``cell_idx``. Geneformer sorts cells by length and silently
drops any cell with no in-vocabulary gene, so the output is re-ordered by ``cell_idx``. Dropped
cells are counted and reported, never filled in.

**Outputs** (under ``--out-dir``):

    input/norman.h5ad                 raw counts of the selected cells, Ensembl IDs, n_counts, cell_idx
    tokens_V1.dataset, tokens_V2.dataset    tokenized cells (Hugging Face datasets)
    stores/<model>_<mode>/            one embedding store per model and mode (``ssllab.embeddings``):
                                      embeddings.npy float32 [n_kept, dim] and row_idx.npy int64
                                      [n_kept] (the processed-cache position of each row, sorted),
                                      plus a manifest with the model, mode, layer, Geneformer
                                      revision and the cache's fingerprint
    emb/                              the extractor's own working files
    feasibility.json                  coverage, token lengths, timing, GPU memory, embedding shape

The stores are what the count decoder, the flow and the ceiling analysis read with ``--latents``.
Writing one needs the processed cache's fingerprint, read from ``<data-dir>/norman2019/tokens_meta.npz``.

Requires the Geneformer package and CUDA (its extractor moves batches to ``cuda`` directly).
The run script ``run_geneformer_feasibility_pod.sh`` installs it on the pod.

Usage (on the pod, from the repo root):
    python examples/perturbation_response/foundation-encoder/02_embed_geneformer.py \\
        --gf-root /runpod-volume/ssl-lab/models/geneformer --out-dir /runpod-volume/ssl-lab/output/geneformer_feasibility \\
        --n-cells 2000 --models V1-10M V2-104M
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import cache_fingerprint_from_dir  # noqa: E402
from ssllab.embeddings import provenance, write_store  # noqa: E402

# name -> folder in the Hugging Face repo, model series, embedding modes, embedding width
MODELS: dict[str, dict] = {
    "V1-10M": {"dir": "Geneformer-V1-10M", "version": "V1", "modes": ["cell"], "dim": 256},
    "V2-104M": {"dir": "Geneformer-V2-104M", "version": "V2", "modes": ["cls", "cell"], "dim": 768},
    "V2-104M_CLcancer": {"dir": "Geneformer-V2-104M_CLcancer", "version": "V2", "modes": ["cls", "cell"], "dim": 768},
    "V2-316M": {"dir": "Geneformer-V2-316M", "version": "V2", "modes": ["cls", "cell"], "dim": 1152},
}
INPUT_SIZE = {"V1": 2048, "V2": 4096}
N_PROCESSED_CELLS = 109_737


def select_cells(n_total: int, n_cells: int, seed: int) -> np.ndarray:
    """Positions in the processed cache to embed, sorted. ``n_cells <= 0`` selects every cell."""
    if n_cells <= 0 or n_cells >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_cells, replace=False)).astype(np.int64)


def read_count_rows(raw_h5ad: Path, rows: np.ndarray, chunk: int = 4096):
    """Read the raw count layer for ``rows`` without loading the whole matrix into memory.

    Returns ``(counts csr float32 [len(rows), n_genes], ensembl_ids [n_genes])``.
    """
    import h5py
    import scipy.sparse as sp

    try:
        from anndata.io import sparse_dataset
    except ImportError:  # anndata < 0.11
        from anndata.experimental import sparse_dataset

    with h5py.File(raw_h5ad, "r") as f:
        layer = sparse_dataset(f["layers"]["counts"])
        parts = [sp.csr_matrix(layer[rows[i : i + chunk]]) for i in range(0, len(rows), chunk)]
        var = f["var"]
        idx = var["index"]
        # the Ensembl IDs are a var column called "index"; categorical columns store codes + categories
        if isinstance(idx, h5py.Group):
            ens = idx["categories"][()][idx["codes"][()]]
        else:
            ens = idx[()]
    ens = np.array([e.decode() if isinstance(e, bytes) else str(e) for e in ens])
    return sp.vstack(parts).tocsr().astype(np.float32), ens


def build_count_adata(counts, ensembl_ids: np.ndarray, cell_idx: np.ndarray):
    """AnnData in the form Geneformer's tokenizer expects.

    ``X`` holds raw counts (the tokenizer reads ``X``, not a layer), ``var["ensembl_id"]`` the gene
    IDs, ``obs["n_counts"]`` each cell's total over all measured genes, and ``obs["cell_idx"]`` the
    processed-cache position that later re-orders the embeddings.
    """
    import anndata as ad
    import pandas as pd

    n_counts = np.asarray(counts.sum(axis=1)).ravel()
    obs = pd.DataFrame({"n_counts": n_counts, "cell_idx": cell_idx.astype(np.int64)},
                       index=[f"c{i}" for i in cell_idx])
    var = pd.DataFrame({"ensembl_id": ensembl_ids}, index=ensembl_ids)
    var.index = var.index.astype(str)
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def align_embeddings(df, dim: int, label: str = "cell_idx") -> tuple[np.ndarray, np.ndarray]:
    """Order the extractor's table by ``cell_idx``; return ``(emb float32 [n, dim], cell_idx int64 [n])``."""
    df = df.sort_values(label)
    emb = df.iloc[:, :dim].to_numpy(dtype=np.float32)
    idx = df[label].to_numpy(dtype=np.int64)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("duplicate cell_idx in embedding table")
    return emb, idx


def describe(emb: np.ndarray) -> dict:
    """Shape and spread of an embedding matrix, using the same rank measure as the JEPA diagnostics."""
    import torch

    from ssllab.eval.collapse import collapse_report

    z = torch.from_numpy(emb)
    rep = {k: float(v) for k, v in collapse_report(z).items()}
    rep.update(dim=int(emb.shape[1]), n=int(emb.shape[0]), n_nonfinite=int((~np.isfinite(emb)).sum()),
               mean_norm=float(np.linalg.norm(emb, axis=1).mean()))
    return rep


def in_vocab_gene_counts(counts, ensembl_ids: np.ndarray, mapping: dict, vocab) -> np.ndarray:
    """Expressed in-vocabulary genes per cell, selected as Geneformer's tokenizer selects them.

    The tokenizer upper-cases each Ensembl ID, maps it through its mapping dictionary (which
    also renames retired IDs), keeps only IDs in its token vocabulary (protein-coding and
    microRNA genes), sums genes that map to the same ID, and turns each expressed gene into one
    token. This counts those tokens per cell *before* the input is cut to 2,048 (V1) or 4,096
    (V2) positions, which the tokenized dataset does not record unless asked to.
    """
    import scipy.sparse as sp

    mapped = [mapping.get(str(e).upper()) for e in ensembl_ids]
    cols = [j for j, m in enumerate(mapped) if m is not None and m in vocab]
    tokens = {t: i for i, t in enumerate(sorted({mapped[j] for j in cols}))}
    gene_to_token = sp.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), (cols, [tokens[mapped[j]] for j in cols])),
        shape=(len(ensembl_ids), len(tokens)),
    )
    expressed = sp.csr_matrix(counts > 0, dtype=np.float32)
    return np.asarray(((expressed @ gene_to_token) > 0).sum(axis=1)).ravel().astype(np.int64)


def tokenize(input_dir: Path, out_dir: Path, version: str, nproc: int, counts, ensembl_ids: np.ndarray,
             cell_idx: np.ndarray) -> tuple[Path, dict]:
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer

    t0 = time.perf_counter()
    tk = TranscriptomeTokenizer({"cell_idx": "cell_idx"}, nproc=nproc, model_version=version)
    tk.tokenize_data(input_dir, out_dir, f"tokens_{version}", file_format="h5ad")
    secs = time.perf_counter() - t0
    path = out_dir / f"tokens_{version}.dataset"
    ds = load_from_disk(str(path))

    n_special = 2 if version == "V2" else 0
    cap = INPUT_SIZE[version] - n_special
    genes_all = in_vocab_gene_counts(counts, ensembl_ids, tk.gene_mapping_dict, set(tk.gene_token_dict))
    # the dataset may drop empty cells and is in its own order: match by cell_idx
    pos = {c: i for i, c in enumerate(cell_idx)}
    genes = genes_all[[pos[c] for c in ds["cell_idx"]]]
    kept = np.asarray(ds["length"]) - n_special
    stats = {
        "seconds": secs,
        "n_cells": int(len(ds)),
        "genes_per_cell_median": float(np.median(genes)),
        "genes_per_cell_p05": float(np.percentile(genes, 5)),
        "genes_per_cell_p95": float(np.percentile(genes, 95)),
        "input_size": INPUT_SIZE[version],
        "frac_cells_truncated": float(np.mean(genes > cap)),
        "frac_genes_dropped_by_truncation": float(1 - kept.sum() / genes.sum()),
        # check that the count above replicates the tokenizer: kept tokens = min(count, cap)
        "n_cells_count_mismatch": int(np.sum(kept != np.minimum(genes, cap))),
    }
    # Free the tokenizer and the loaded dataset before the next model series is tokenized: on all
    # 109,737 cells the two series together exceed a 46 GB pod's memory otherwise.
    del ds, tk
    import gc
    gc.collect()
    return path, stats


def embed(model_dir: Path, dataset: Path, version: str, mode: str, batch: int, nproc: int,
          work_dir: Path, dim: int) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from geneformer import EmbExtractor
    from geneformer import perturber_utils as pu

    t0 = time.perf_counter()
    model = pu.load_model("Pretrained", 0, str(model_dir), mode="eval")
    load_secs = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()

    ex = EmbExtractor(model_type="Pretrained", num_classes=0, emb_mode=mode, max_ncells=None,
                      emb_layer=-1, emb_label=["cell_idx"], forward_batch_size=batch, nproc=nproc,
                      model_version=version)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    df = ex.extract_embs(str(model_dir), str(dataset), str(work_dir), f"{model_dir.name}_{mode}")
    torch.cuda.synchronize()
    total_secs = time.perf_counter() - t0
    emb, idx = align_embeddings(df, dim)
    forward_secs = max(total_secs - load_secs, 1e-9)
    timing = {
        "load_seconds": load_secs,
        "total_seconds": total_secs,
        "forward_seconds_est": forward_secs,
        "seconds_per_1k_cells": 1000 * forward_secs / len(idx),
        "projected_hours_all_cells": forward_secs / len(idx) * N_PROCESSED_CELLS / 3600,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "batch_size": batch,
    }
    return emb, idx, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gf-root", required=True, help="snapshot of the Geneformer Hugging Face repo (code + model folders)")
    ap.add_argument("--data-dir", default="data", help="folder holding norman2019/ (raw/ and foundation/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--modes", nargs="+", default=None, choices=["cls", "cell"],
                    help="restrict embedding modes (default: every mode a model supports)")
    ap.add_argument("--n-cells", type=int, default=2000, help="random sample size; 0 embeds every cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--gf-revision", default=None, help="the Hugging Face commit of --gf-root, recorded in each store")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("Geneformer's extractor needs CUDA; run this on the pod.")
    sys.path.insert(0, str(Path(args.gf_root)))  # import geneformer from the downloaded source

    root = Path(args.data_dir) / "norman2019"
    index = np.load(root / "foundation" / "cell_index.npz")
    fingerprint = cache_fingerprint_from_dir(root)          # every store is filed against this cache
    out = Path(args.out_dir)
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "emb").mkdir(parents=True, exist_ok=True)

    report: dict = {"n_cells_requested": args.n_cells, "seed": args.seed,
                    "gpu": torch.cuda.get_device_name(0), "models": {}, "tokenization": {}}

    # 1. raw counts of the selected cells, in processed order
    t0 = time.perf_counter()
    cells = select_cells(len(index["raw_row"]), args.n_cells, args.seed)
    counts, ens = read_count_rows(root / "raw" / "norman_2019.h5ad", index["raw_row"][cells])
    adata = build_count_adata(counts, ens, cells)
    if not np.allclose(adata.obs["n_counts"].to_numpy(), index["total_counts"][cells]):
        sys.exit("n_counts disagrees with the index's total_counts: wrong rows were read")
    adata.write_h5ad(out / "input" / "norman.h5ad")
    report["input"] = {"n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
                       "seconds": time.perf_counter() - t0,
                       "n_control": int(index["is_control"][cells].sum()),
                       "split_counts": {int(s): int((index["split_combo"][cells] == s).sum()) for s in (0, 1, 2)}}
    print(json.dumps(report["input"], indent=2), flush=True)
    del adata

    # 2. tokenize once per model series
    datasets = {}
    for version in sorted({MODELS[m]["version"] for m in args.models}):
        datasets[version], report["tokenization"][version] = tokenize(out / "input", out, version, args.nproc,
                                                                      counts, ens, cells)
        print(version, json.dumps(report["tokenization"][version]), flush=True)
    del counts

    # 3. embed, per model and mode
    for name in args.models:
        spec = MODELS[name]
        modes = [m for m in spec["modes"] if args.modes is None or m in args.modes]
        for mode in modes:
            key = f"{name}_{mode}"
            try:
                emb, idx, timing = embed(Path(args.gf_root) / spec["dir"], datasets[spec["version"]],
                                         spec["version"], mode, args.batch_size, args.nproc, out / "emb", spec["dim"])
            except Exception as e:  # keep going; one failing model should not hide the others
                report["models"][key] = {"error": f"{type(e).__name__}: {e}"}
                print("FAILED", key, e, flush=True)
                continue
            store = write_store(
                out / "stores" / key, emb, idx,
                source="geneformer", dataset_fingerprint=fingerprint,
                model=name, mode=mode, layer="second-to-last (emb_layer=-1)", model_version=spec["version"],
                input_size=INPUT_SIZE[spec["version"]], gf_revision=args.gf_revision,
                n_cells_requested=int(len(cells)), n_dropped=int(len(cells) - len(idx)),
                **provenance(__file__),
            )
            report["models"][key] = {**timing, **describe(emb), "n_dropped": int(len(cells) - len(idx)),
                                     "store": str(store.path), "content_md5": store.manifest["content_md5"]}
            print(key, json.dumps(report["models"][key]), flush=True)
            (out / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")

    (out / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out / 'feasibility.json'}")


if __name__ == "__main__":
    main()
