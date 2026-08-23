"""Process Norman 2019 Perturb-seq into the ssl-lab cache (Phase 1a).

Acquire -> QC -> HVG -> normalize -> differential expression -> cache. This is the
one *heavy* step (needs anndata/scanpy/pertpy); it runs once on CPU and writes the
artifact the light loader (``ssllab.data.perturbseq``) reads at train time. Stage A
(the JEPA cell encoder) then consumes the token contract with no data work.

Dataset: Norman et al. 2019 (K562 CRISPRa, single + combinatorial genetic
perturbations) — the scGen/CPA/GEARS lineage benchmark, graded on effect size.

Requires the single-cell extra (this is the *only* script in the repo that does)::

    pip install -e ".[perturb]"          # anndata, scanpy, scikit-misc, pertpy
    # or, for the whole environment:  mamba env create -f environment.yml

``pertpy`` is needed only for ``--source pertpy``; ``--source h5ad`` skips it.

Usage
-----
    # real run (multi-GB download, CPU, no pod); needs SSLLAB_DATA_ROOT for lake-staging
    python examples/perturbation_response/00_process_norman.py --source pertpy --n-hvg 5000

    # from a pinned local .h5ad instead of pertpy auto-fetch
    python examples/perturbation_response/00_process_norman.py --source h5ad --h5ad path/to/norman.h5ad

    # tiny synthetic smoke run (no network, seconds) — exercises the whole pipeline
    python examples/perturbation_response/00_process_norman.py --smoke

Output
------
    <data_dir>/<artifact>/{processed.h5ad, tokens_meta.npz, splits.json, de_genes.json, manifest.json}
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import (
    ARTIFACT_VERSION,
    DEFAULT_N_HVG,
    DEFAULT_N_TOKENS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    make_gene_partition,
    token_dim_for,
    write_cache,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CONTROL = "control"
_NT_TOKENS = {"control", "ctrl", "nt", "non-targeting", "nontargeting", "neg", "none", ""}


# --------------------------------------------------------------------------- #
# Perturbation label normalization.
# --------------------------------------------------------------------------- #
def normalize_label(raw: str, combo_sep: str | None) -> str:
    """Normalize a raw perturbation string to a canonical label.

    Non-targeting -> ``"control"``; combos -> alphabetized, ``+``-joined gene set
    (e.g. ``"KLF1_CEBPA"`` -> ``"CEBPA+KLF1"``). Auto-detects the combo separator
    when ``combo_sep`` is None.
    """
    s = str(raw).strip()
    if combo_sep is None:
        combo_sep = "+" if "+" in s else ("_" if "_" in s else "|")
    genes = [g.strip() for g in s.split(combo_sep) if g.strip()]
    genes = [g for g in genes if g.lower() not in _NT_TOKENS]
    if not genes:
        return CONTROL
    return "+".join(sorted(set(genes)))


def _detect_pert_col(obs_columns) -> str:
    for cand in ("perturbation", "perturbation_name", "condition", "guide_identity", "gene"):
        if cand in obs_columns:
            return cand
    raise ValueError(
        f"could not auto-detect the perturbation column among {list(obs_columns)}; "
        "pass --pert-col explicitly"
    )


# --------------------------------------------------------------------------- #
# Heavy dependencies.
#
# The single-cell stack is imported lazily, inside the functions that need it, so
# that ``ssllab`` itself, the tests, and every training script stay light. The cost
# of that choice is that a missing dependency surfaces at *run* time rather than at
# install time, so each import carries the command that fixes it.
# --------------------------------------------------------------------------- #
INSTALL_HINT = (
    'pip install -e ".[perturb]"   (or: mamba env create -f environment.yml)'
)


def require(module: str, why: str, pip_name: str | None = None):
    """Import a processing-only dependency, or fail with the command that installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        pip_name = pip_name or module
        raise ImportError(
            f"{module!r} is required {why} but is not installed. It is a "
            f"processing-time-only dependency of ssl-lab ({pip_name}):\n\n"
            f"    {INSTALL_HINT}\n"
        ) from exc


# --------------------------------------------------------------------------- #
# Acquisition.
# --------------------------------------------------------------------------- #
def load_adata(source: str, h5ad: str | None, pert_col: str | None, combo_sep: str | None):
    """Load an AnnData with raw counts in ``.X`` and a canonical ``obs['perturbation']``."""
    ad = require("anndata", "to read and write .h5ad artifacts")

    if source == "pertpy":
        pt = require("pertpy", "for --source pertpy (use --source h5ad to avoid it)")

        logger.info("fetching Norman 2019 via pertpy (first run downloads)...")
        adata = pt.data.norman_2019()
    elif source == "h5ad":
        if not h5ad:
            raise ValueError("--source h5ad requires --h5ad PATH")
        logger.info("reading %s", h5ad)
        adata = ad.read_h5ad(h5ad)
    else:
        raise ValueError(f"unknown source {source!r}")

    col = pert_col or _detect_pert_col(adata.obs.columns)
    logger.info("using perturbation column %r", col)
    adata.obs["perturbation"] = [normalize_label(v, combo_sep) for v in adata.obs[col]]
    return adata


# --------------------------------------------------------------------------- #
# Processing pipeline.
# --------------------------------------------------------------------------- #
def process(adata, args) -> dict:
    """Run QC -> HVG -> normalize -> DE -> splits and write the cache. Returns manifest."""
    sc = require("scanpy", "for QC, HVG selection, and differential expression")
    # seurat_v3 needs the loess fit from scikit-misc, and only fails deep inside
    # highly_variable_genes; check it up front so the error names the real cause.
    require("skmisc", 'for the "seurat_v3" HVG flavor', pip_name="scikit-misc")

    # 0. Ensure raw integer counts live in .X (some sources put them in a layer).
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    # 1. Basic gene/cell filtering + QC metrics.
    sc.pp.filter_genes(adata, min_cells=args.min_cells_per_gene)
    sc.pp.filter_cells(adata, min_genes=args.min_genes_per_cell)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    if adata.var["mt"].any():
        adata = adata[adata.obs["pct_counts_mt"] <= args.max_pct_mito].copy()
    logger.info("after QC: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # 2. Library size over the FULL gene set (a covariate; computed before HVG subset).
    counts_full = adata.X
    libsize = np.asarray(counts_full.sum(axis=1)).ravel().astype(np.float32)

    # 3. Keep raw counts, then select HVGs (seurat_v3 operates on counts).
    adata.layers["counts"] = adata.X.copy()
    n_hvg = min(args.n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts")
    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("HVG panel: %d genes", adata.n_vars)

    # 4. Normalize for token features (log1p CP10K); raw counts stay in the layer.
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 5. Perturbation coding.
    perts = adata.obs["perturbation"].astype(str).to_numpy()
    pert_names = np.array(sorted(set(perts)))
    name_to_id = {n: i for i, n in enumerate(pert_names)}
    pert_id = np.array([name_to_id[p] for p in perts], dtype=np.int64)
    is_control = perts == CONTROL
    if not is_control.any():
        raise ValueError("no control cells found after labeling; check --pert-col/--combo-sep")
    ctrl_group = np.zeros(adata.n_obs, dtype=np.int64)  # Norman: single batch -> one pool

    # 6. Differential expression vs control (effect-size metric seam).
    de_genes = differential_expression(adata, sc, top_k=args.de_top_k, min_cells=args.min_cells_per_pert)
    validate_de_genes(de_genes, adata)          # acceptance gate: refuse to cache silent "DE" genes

    # 7. Splits at the perturbation level.
    split_combo = make_combo_split(perts, pert_names, seed=args.seed,
                                   val_frac=args.val_frac, test_frac=args.test_frac)
    split_cells = make_cells_split(adata.n_obs, seed=args.seed,
                                   val_frac=args.val_frac, test_frac=args.test_frac)

    # 8. Densify HVG matrices for the torch-native cache.
    hvg_X = _dense(adata.X).astype(np.float32)
    counts = _dense(adata.layers["counts"]).astype(np.int32)
    gene_ids = adata.var_names.to_numpy().astype(object)

    cache_dir = Path(args.data_dir) / args.artifact
    n_tokens = args.n_tokens
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source": args.source if not args.smoke else "synthetic",
        "n_cells": int(adata.n_obs),
        "n_hvg": int(adata.n_vars),
        "n_perts": int(len(pert_names)),
        "qc": {
            "min_genes_per_cell": args.min_genes_per_cell,
            "max_pct_mito": args.max_pct_mito,
            "min_cells_per_gene": args.min_cells_per_gene,
        },
        "hvg": {"method": "seurat_v3", "n_top": int(adata.n_vars), "on": "counts"},
        "normalize": "log1p_cpm",
        "de": {"method": "wilcoxon", "vs": CONTROL, "top_k": args.de_top_k},
        "tokenization": {
            "n_tokens": n_tokens,
            "token_dim": token_dim_for(adata.n_vars, n_tokens),
            "partition_seed": args.seed,
            "scheme": "random",
        },
        "split_fracs": {"val": args.val_frac, "test": args.test_frac, "seed": args.seed},
    }

    # Sanity: the declared token partition is well-formed for this panel.
    make_gene_partition(adata.n_vars, n_tokens, args.seed)

    write_cache(
        cache_dir,
        hvg_X=hvg_X, counts=counts, libsize=libsize,
        pert_id=pert_id, is_control=is_control, ctrl_group=ctrl_group,
        split_combo=split_combo, split_cells=split_cells,
        gene_ids=gene_ids, pert_names=pert_names.astype(object),
        splits=split_summary(perts, pert_names, split_combo, args),
        de_genes=de_genes, manifest=manifest,
    )
    # Canonical biology-native artifact alongside the torch cache.
    adata.obs["split_combo"] = split_combo
    adata.obs["split_cells"] = split_cells
    adata.obs["libsize"] = libsize
    adata.obs["is_control"] = is_control
    adata.write_h5ad(cache_dir / "processed.h5ad")
    logger.info("wrote cache -> %s", cache_dir)
    return manifest


def differential_expression(
    adata, sc, top_k: int, min_cells: int,
    min_detect_frac: float = 0.10, max_pval: float = 0.05,
) -> dict:
    """Top-``top_k`` DE HVG indices per perturbation (vs control, Wilcoxon).

    Ranked by ``|score|``, the Wilcoxon z-statistic, which is scanpy's own default ordering.

    **Not by |logFC|.** Scanpy computes the log fold-change with a ``1e-9`` pseudocount, so a
    gene that is silent in *both* control and perturbed can post an enormous |logFC| out of pure
    count noise. Ranking by it selects genes that are not expressed at all. That is not
    hypothetical: it is what this function used to do, and it made every scored gene come back
    with ``pval_adj ~= 1.0`` and *every* held-out cell exactly zero, which silently forced
    interval coverage to 1.00 and left the whole calibration axis unreadable. A rank statistic
    cannot blow up on a silent gene, so ``|score|`` is self-guarding.

    Two guards are applied on top: a gene must be **detected** in at least ``min_detect_frac`` of
    the group's cells and be **significant** at ``max_pval``. Under ``|score|`` ranking these are
    very nearly no-ops, which is exactly the point. They pin the invariant so that a future
    change to the ranking cannot quietly reintroduce silent genes.
    """
    empty = {"method": "wilcoxon", "vs": CONTROL, "top_k": top_k,
             "ranked_by": "abs_wilcoxon_z", "gene_space": "hvg_index",
             "guards": {"min_detect_frac": min_detect_frac, "max_pval": max_pval},
             "per_pert": {}}
    counts_per_pert = adata.obs["perturbation"].value_counts()
    keep = [p for p, n in counts_per_pert.items() if p != CONTROL and n >= min_cells]
    if not keep:
        logger.warning("no perturbation has >= %d cells; DE cache will be empty", min_cells)
        return empty

    sub = adata[adata.obs["perturbation"].isin(keep + [CONTROL])].copy()
    sc.tl.rank_genes_groups(sub, groupby="perturbation", groups=keep,
                            reference=CONTROL, method="wilcoxon")
    res = sub.uns["rank_genes_groups"]
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    counts = sub.layers["counts"]
    group_of = sub.obs["perturbation"].to_numpy()

    per_pert, short = {}, []
    for p in keep:
        names = np.asarray(res["names"][p])
        score = np.asarray(res["scores"][p], dtype=float)
        logfc = np.asarray(res["logfoldchanges"][p], dtype=float)
        pvals = np.asarray(res["pvals_adj"][p], dtype=float)
        gi = np.array([gene_to_idx[g] for g in names])

        cg = counts[group_of == p][:, gi]
        cg = cg.toarray() if hasattr(cg, "toarray") else np.asarray(cg)
        detect = (cg > 0).mean(axis=0)                      # fraction of the group's cells expressing it

        ok = (detect >= min_detect_frac) & (pvals < max_pval)
        rank = np.where(ok, -np.abs(score), np.inf)         # guarded genes only, strongest |z| first
        order = np.argsort(rank)[:top_k]
        order = order[np.isfinite(rank[order])]             # never pad with genes that failed a guard
        if order.size < top_k:
            short.append((p, int(order.size)))
        per_pert[p] = {
            "top_idx": [int(gi[i]) for i in order],
            "score": [float(score[i]) for i in order],
            "logfc": [float(logfc[i]) for i in order],
            "pval_adj": [float(pvals[i]) for i in order],
            "detect_frac": [float(detect[i]) for i in order],
        }
    if short:
        logger.warning("%d/%d perturbations have < %d genes passing the DE guards (weakest: %s)",
                       len(short), len(keep), top_k, sorted(short, key=lambda t: t[1])[:3])
    return {**empty, "per_pert": per_pert}


def validate_de_genes(de: dict, adata) -> None:
    """Fail loudly if the DE selection produced genes the metric cannot meaningfully score.

    This is the acceptance gate for Stage 00. The |logFC| bug shipped silently for months
    because nothing downstream ever asked whether the "top-DE genes" were expressed. Now the
    pipeline refuses to emit a cache that would repeat it.
    """
    per_pert = de["per_pert"]
    if not per_pert:
        return
    detect = np.array([d for v in per_pert.values() for d in v["detect_frac"]])
    pvals = np.array([p for v in per_pert.values() for p in v["pval_adj"]])
    logger.info("DE gene check: median detection %.1f%%, %.0f%% significant, over %d (pert, gene) pairs",
                100 * float(np.median(detect)), 100 * float((pvals < 0.05).mean()), detect.size)
    if float(np.median(detect)) < 0.05:
        raise RuntimeError(
            f"DE selection is degenerate: median detection {np.median(detect):.1%} of cells. "
            "The scored genes are effectively silent, so coverage would be forced to 1.00 and "
            "the effect-size correlation would be noise. Check the ranking criterion."
        )
    if float((pvals < 0.05).mean()) < 0.5:
        raise RuntimeError(
            f"DE selection is degenerate: only {(pvals < 0.05).mean():.1%} of scored genes are "
            "significant vs control. The metric would be scoring genes the perturbation did not move."
        )


# --------------------------------------------------------------------------- #
# Splits.
# --------------------------------------------------------------------------- #
def _genes(pert: str) -> list[str]:
    return pert.split("+")


def make_combo_split(perts, pert_names, seed, val_frac, test_frac) -> np.ndarray:
    """Combinatorial-generalization split: hold out unseen 2-gene combos whose
    *both* constituent singles are seen as singles. Singles + control -> train."""
    rng = np.random.default_rng(seed)
    singles = {p for p in pert_names if p != CONTROL and len(_genes(p)) == 1}
    single_genes = {g for p in singles for g in _genes(p)}
    eligible = [p for p in pert_names
                if len(_genes(p)) >= 2 and all(g in single_genes for g in _genes(p))]
    eligible = sorted(eligible)
    rng.shuffle(eligible)
    n_test = max(1, int(round(len(eligible) * test_frac))) if eligible else 0
    n_val = int(round(len(eligible) * val_frac)) if eligible else 0
    test_perts = set(eligible[:n_test])
    val_perts = set(eligible[n_test:n_test + n_val])

    pert_split = {}
    for p in pert_names:
        pert_split[p] = SPLIT_TEST if p in test_perts else SPLIT_VAL if p in val_perts else SPLIT_TRAIN
    return np.array([pert_split[p] for p in perts], dtype=np.int8)


def make_cells_split(n: int, seed, val_frac, test_frac) -> np.ndarray:
    """Random per-cell sanity split (all perturbations seen in train)."""
    rng = np.random.default_rng(seed + 1)
    r = rng.random(n)
    out = np.full(n, SPLIT_TRAIN, dtype=np.int8)
    out[r < val_frac] = SPLIT_VAL
    out[r >= 1.0 - test_frac] = SPLIT_TEST
    return out


def split_summary(perts, pert_names, split_combo, args) -> dict:
    """Human-readable record of which perturbations landed in each combo split."""
    by_code = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}
    pert_to_code = {}
    for p, c in zip(perts, split_combo):
        pert_to_code[p] = int(c)
    buckets = {"train": [], "val": [], "test": []}
    for p, c in sorted(pert_to_code.items()):
        buckets[by_code[c]].append(p)
    return {
        "combo": {
            "description": "Hold out unseen 2-gene combos whose both singles are seen; control+singles in train.",
            "train_perts": buckets["train"],
            "val_perts": buckets["val"],
            "test_perts": buckets["test"],
        },
        "cells": {
            "description": "Random per-cell sanity split; all perturbations seen in train.",
            "frac": {"val": args.val_frac, "test": args.test_frac}, "seed": args.seed,
        },
    }


# --------------------------------------------------------------------------- #
# Synthetic smoke data (no network) — exercises the full pipeline in seconds.
# --------------------------------------------------------------------------- #
def synthetic_adata(n_cells=400, n_genes=80, seed=0):
    ad = require("anndata", "to build the synthetic smoke artifact")

    rng = np.random.default_rng(seed)
    perts_pool = ["control", "GENE0", "GENE1", "GENE2", "GENE3", "GENE0+GENE1", "GENE2+GENE3"]
    labels = rng.choice(perts_pool, size=n_cells, p=[0.4, 0.12, 0.12, 0.12, 0.12, 0.06, 0.06])
    base = rng.gamma(2.0, 2.0, size=n_genes)
    X = np.zeros((n_cells, n_genes), dtype=np.int64)
    for i, lab in enumerate(labels):
        rate = base.copy()
        for g in lab.split("+"):
            if g != "control":
                k = (hash(g) % n_genes)
                rate[k] *= 4.0  # the perturbed gene is up-regulated -> a real DE signal
        X[i] = rng.poisson(rate)
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = [f"G{j}" for j in range(n_genes)]
    adata.obs["perturbation"] = labels
    return adata


def subsample_adata(adata, keep_perts, max_per_pert, seed):
    """Restrict to ``keep_perts`` and/or cap cells per perturbation (tutorial / fast
    iteration slice). Operates on the canonical ``obs['perturbation']`` labels."""
    rng = np.random.default_rng(seed)
    pert = adata.obs["perturbation"].astype(str).to_numpy()
    if keep_perts:
        adata = adata[np.isin(pert, list(set(keep_perts)))].copy()
        pert = adata.obs["perturbation"].astype(str).to_numpy()
    if max_per_pert:
        rows = []
        for p in np.unique(pert):
            idx = np.flatnonzero(pert == p)
            if len(idx) > max_per_pert:
                idx = rng.choice(idx, max_per_pert, replace=False)
            rows.append(idx)
        adata = adata[np.sort(np.concatenate(rows))].copy()
    return adata


def _dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process Norman 2019 Perturb-seq into the ssl-lab cache.")
    p.add_argument("--source", choices=["pertpy", "h5ad"], default="pertpy")
    p.add_argument("--h5ad", type=str, default=None, help="path to a local .h5ad (--source h5ad)")
    p.add_argument("--pert-col", type=str, default=None, help="obs column with the perturbation label")
    p.add_argument("--combo-sep", type=str, default=None, help="combo separator in raw labels (auto if unset)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--n-hvg", type=int, default=DEFAULT_N_HVG)
    p.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS)
    p.add_argument("--de-top-k", type=int, default=50)
    p.add_argument("--min-genes-per-cell", type=int, default=500)
    p.add_argument("--max-pct-mito", type=float, default=10.0)
    p.add_argument("--min-cells-per-gene", type=int, default=3)
    p.add_argument("--min-cells-per-pert", type=int, default=30)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsample-perts", type=str, default=None,
                   help="comma-separated perturbations to keep (tutorial / iteration slice)")
    p.add_argument("--max-cells-per-pert", type=int, default=None, help="cap cells per perturbation")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run (no network)")
    p.add_argument("--de-only", action="store_true",
                   help="recompute ONLY de_genes.json from the existing processed.h5ad. Leaves "
                        "tokens_meta.npz and splits.json untouched, so trained checkpoints stay "
                        "valid and can be re-evaluated without retraining.")
    return p.parse_args()


def rewrite_de_only(args) -> None:
    """Regenerate ``de_genes.json`` in place, reusing the cached ``processed.h5ad``.

    The DE gene list is the *scoring seam*, not a modeling input: no stage trains on it, and it
    lives in its own file. So a bad DE selection can be corrected without re-deriving the splits
    (which would strand every trained checkpoint) or re-writing the multi-GB token cache.
    """
    ad = require("anndata", "to read the cached processed.h5ad")
    sc = require("scanpy", "for differential expression")

    cache_dir = Path(args.data_dir) / args.artifact
    h5 = cache_dir / "processed.h5ad"
    if not h5.exists():
        raise FileNotFoundError(f"{h5} not found; run 00 without --de-only first")

    logger.info("reading %s", h5)
    adata = ad.read_h5ad(h5)
    de_genes = differential_expression(adata, sc, top_k=args.de_top_k, min_cells=args.min_cells_per_pert)
    validate_de_genes(de_genes, adata)

    out = cache_dir / "de_genes.json"
    backup = cache_dir / "de_genes.prev.json"
    if out.exists() and not backup.exists():
        backup.write_text(out.read_text())
        logger.info("backed up previous DE selection -> %s", backup)
    out.write_text(json.dumps(de_genes, indent=2))
    logger.info("rewrote %s (%d perturbations, ranked_by=%s)",
                out, len(de_genes["per_pert"]), de_genes["ranked_by"])
    logger.info("splits.json and tokens_meta.npz untouched -> existing checkpoints remain valid; "
                "re-run the evals (06 / 09 / 10) to rescore them.")


def main() -> None:
    args = parse_args()
    if args.de_only:
        rewrite_de_only(args)
        return
    if args.smoke:
        logger.info("SMOKE: synthetic AnnData (no network)")
        adata = synthetic_adata(seed=args.seed)
        # Relax thresholds so the tiny synthetic data survives QC.
        args.min_genes_per_cell = 1
        args.min_cells_per_gene = 1
        args.min_cells_per_pert = 5
        args.n_hvg = min(args.n_hvg, adata.n_vars)
        args.n_tokens = min(args.n_tokens, args.n_hvg)
    else:
        adata = load_adata(args.source, args.h5ad, args.pert_col, args.combo_sep)
        if args.subsample_perts or args.max_cells_per_pert:
            keep = ([normalize_label(p, args.combo_sep) for p in args.subsample_perts.split(",")]
                    if args.subsample_perts else None)
            adata = subsample_adata(adata, keep, args.max_cells_per_pert, args.seed)
            logger.info("subsampled to %d cells x %d perts", adata.n_obs,
                        adata.obs["perturbation"].nunique())

    manifest = process(adata, args)
    logger.info("done: %s", json.dumps({k: manifest[k] for k in ("n_cells", "n_hvg", "n_perts")}))


if __name__ == "__main__":
    main()
