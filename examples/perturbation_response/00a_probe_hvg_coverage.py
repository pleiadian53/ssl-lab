"""Diagnose: does the cached HVG panel already contain the genes that respond?

Trains nothing. Answers, in one CPU run, the question that would otherwise be answered
by a multi-day rebuild: **is the gene panel the bottleneck?**

The cache's `de_genes.json` ranks differential expression *within* the panel, so it can
never tell you what the panel left out. This probe re-runs the identical DE procedure over
the **full** gene set of the raw file and asks, per perturbation, what fraction of the true
top-k responding genes made it into the panel.

    high coverage  -> the panel is not the limitation; a wider --n-hvg buys nothing
    low coverage   -> a quantified reason to rebuild, and the missing genes name themselves

Everything that could confound the comparison is read from the cache's own manifest (QC
thresholds, DE top-k, the normalization) rather than re-specified here, so the only
difference between this DE run and the cached one is the gene space it ranks over.

Requires the single-cell extra (see 00_process_norman.py)::

    pip install -e ".[perturb]"

Usage
-----
    # the standard question: what did the 5,000-gene panel miss?
    python examples/perturbation_response/00a_probe_hvg_coverage.py

    # faster, for a first look: cap cells per perturbation
    python examples/perturbation_response/00a_probe_hvg_coverage.py --max-cells-per-pert 200

Output
------
    output/<experiment>/reports/hvg_coverage.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import load_cache  # noqa: E402
from ssllab.experiment import experiment  # noqa: E402

logger = logging.getLogger("hvg_coverage")


def _load_processing_module():
    """Import ``00_process_norman.py`` as a module.

    Its leading digit makes it un-importable by name, but reusing it is the point: the DE
    ranking there carries a hard-won convention (rank by ``|wilcoxon z|``, never ``|logFC|``)
    plus detection and significance guards. Re-implementing that here would let the two
    drift, and a coverage number computed under a different ranking would be meaningless.
    """
    spec = importlib.util.spec_from_file_location("process_norman", _HERE / "00_process_norman.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Is the cached HVG panel missing the responding genes?")
    p.add_argument("--h5ad", type=str, default="data/norman_2019.h5ad",
                   help="the RAW file, full gene set (not the cache's processed.h5ad, which is already subset)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019", help="the cache whose panel is under test")
    p.add_argument("--pert-col", type=str, default=None)
    p.add_argument("--combo-sep", type=str, default=None)
    p.add_argument("--top-k", type=int, default=None,
                   help="DE genes per perturbation (default: whatever the cache used)")
    p.add_argument("--ranks", type=str, default="10,20,50",
                   help="report coverage at these rank depths; the strongest genes matter most")
    p.add_argument("--max-cells-per-pert", type=int, default=None,
                   help="subsample for speed; full-set Wilcoxon over ~19k genes is the slow part")
    p.add_argument("--min-cells-per-pert", type=int, default=30,
                   help="skip perturbations with fewer cells (mirrors 00's default)")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    proc = _load_processing_module()
    sc = proc.require("scanpy", "for differential expression over the full gene set")
    proc.require("anndata", "to read the raw .h5ad")

    # 1. The panel under test, and the settings that produced it.
    cache = load_cache(Path(args.data_dir) / args.artifact)
    panel = set(str(g) for g in cache.gene_ids)
    qc = cache.manifest.get("qc", {})
    top_k = args.top_k or int(cache.manifest.get("de", {}).get("top_k", 50))
    logger.info("panel under test: %s  %d genes  (top_k=%d)", args.artifact, len(panel), top_k)

    # 2. The raw file, put through the SAME QC as the cache so the cell populations match.
    #    A coverage number computed on a different set of cells would not be about the panel.
    adata = proc.load_adata("h5ad", args.h5ad, args.pert_col, args.combo_sep)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.filter_genes(adata, min_cells=int(qc.get("min_cells_per_gene", 3)))
    sc.pp.filter_cells(adata, min_genes=int(qc.get("min_genes_per_cell", 500)))
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    if adata.var["mt"].any():
        adata = adata[adata.obs["pct_counts_mt"] <= float(qc.get("max_pct_mito", 10.0))].copy()
    if args.max_cells_per_pert:
        adata = proc.subsample_adata(adata, None, args.max_cells_per_pert, args.seed)
    n_full = adata.n_vars
    logger.info("full gene space: %d cells x %d genes", adata.n_obs, n_full)

    missing_from_raw = len(panel - set(adata.var_names))
    if missing_from_raw:
        logger.warning("%d panel genes are absent from the raw file after QC; coverage is a "
                       "lower bound (is --h5ad the same source the cache was built from?)",
                       missing_from_raw)

    # 3. Same normalization, same DE, wider gene space.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logger.info("running full-gene-set DE over %d perturbations (the slow step)",
                adata.obs["perturbation"].nunique() - 1)
    de_full = proc.differential_expression(adata, sc, top_k=top_k, min_cells=args.min_cells_per_pert)
    gene_names = np.asarray(adata.var_names)

    # 4. Coverage: of the genes that actually respond, how many did the panel keep?
    ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()} | {top_k})
    per_pert, missing_counter = {}, {}
    for pert, rec in de_full["per_pert"].items():
        hits = [str(gene_names[i]) in panel for i in rec["top_idx"]]
        if not hits:
            continue
        at_rank = {str(r): round(float(np.mean(hits[:r])), 4) for r in ranks if r <= len(hits)}
        missed = [str(gene_names[i]) for i, h in zip(rec["top_idx"], hits) if not h]
        for g in missed:
            missing_counter[g] = missing_counter.get(g, 0) + 1
        per_pert[pert] = {"n_scored": len(hits), "coverage": round(float(np.mean(hits)), 4),
                          "coverage_at_rank": at_rank, "missed": missed}

    if not per_pert:
        raise RuntimeError("no perturbation produced DE genes; check --pert-col and the QC thresholds")

    cov = np.array([v["coverage"] for v in per_pert.values()])
    summary = {
        "mean": round(float(cov.mean()), 4),
        "median": round(float(np.median(cov)), 4),
        "p10": round(float(np.percentile(cov, 10)), 4),
        "min": round(float(cov.min()), 4),
        "max": round(float(cov.max()), 4),
    }
    at_rank = {}
    for r in ranks:
        vals = [v["coverage_at_rank"][str(r)] for v in per_pert.values() if str(r) in v["coverage_at_rank"]]
        if vals:
            at_rank[str(r)] = round(float(np.mean(vals)), 4)

    # 5. The actionable alternative to a bigger panel: how much would a targeted union cost?
    union_extra = len(missing_counter)
    report = {
        "artifact": args.artifact,
        "panel_size": len(panel),
        "n_genes_full": int(n_full),
        "n_cells": int(adata.n_obs),
        "top_k": top_k,
        "n_perturbations": len(per_pert),
        "panel_genes_absent_from_raw": missing_from_raw,
        "max_cells_per_pert": args.max_cells_per_pert,
        "coverage": summary,
        "coverage_at_rank": at_rank,
        "union_panel": {
            "n_missing_unique": union_extra,
            "n_union": len(panel) + union_extra,
            "most_missed": sorted(missing_counter.items(), key=lambda kv: -kv[1])[:25],
        },
        "worst_perturbations": sorted(
            ({"pert": k, "coverage": v["coverage"]} for k, v in per_pert.items()),
            key=lambda d: d["coverage"])[:15],
        "per_pert": per_pert,
    }

    exp = experiment(args.experiment, args.output_root).ensure()
    out = exp.reports / "hvg_coverage.json"
    out.write_text(json.dumps(report, indent=2))

    logger.info("coverage of top-%d DE genes by the %d-gene panel: median %.1f%%  mean %.1f%%  worst %.1f%%",
                top_k, len(panel), 100 * summary["median"], 100 * summary["mean"], 100 * summary["min"])
    for r in ranks:
        if str(r) in at_rank:
            logger.info("  at rank %-3d %.1f%%", r, 100 * at_rank[str(r)])
    logger.info("a targeted union panel (HVG + every missed DE gene) would be %d genes, +%d over the current panel",
                len(panel) + union_extra, union_extra)
    logger.info("wrote %s", out)
    logger.info("Reading it: high coverage means a wider --n-hvg cannot help, because the genes the "
                "metric scores are already in the panel. Low coverage means it might, and the union "
                "figure above is the cheaper targeted alternative to simply doubling the panel.")


if __name__ == "__main__":
    main()
