"""The round's primary endpoint: does the operator's bracket predict genetic interaction?

The operator-algebra thesis is that ``||[M_A, M_B]||`` — how far two single-gene generators fail to
commute — tracks the pair's **epistasis**, the empirical departure of the double perturbation from the
sum of its singles. This script measures that correlation on a *trained* operator-algebra checkpoint.

Two sides, joined per pair:

  * model side   ``bracket = ||[M_A, M_B]||_F``, read straight off the trained generators.
  * data side    ``rel_GI = ||Delta(A+B) - (Delta(A)+Delta(B))|| / ||Delta(A+B)||`` on the pair's
                 top-DE genes, model-free from the cache (the same quantity `15_empirical_epistasis.py`
                 reports), computed here for BOTH training and held-out combinations.

Reported as two correlations, because they answer different questions:

  * in-sample (training combos)   did the mechanism engage at all? These combos shaped the generators
                                  directly, so a null here means the bracket is not learning interaction
                                  even where it was trained on it — a fatal result for the thesis.
  * out-of-sample (held-out)      the real claim: for combinations the operator NEVER trained on, does
                                  the bracket of its two singles' generators predict the held-out pair's
                                  measured epistasis? A positive, permutation-significant correlation is
                                  the round's prize, and it needs no comparison to the NB-VAE.

Usage
-----
    python examples/perturbation_response/17_eval_bracket_epistasis.py \
        --experiment norman_operator_algebra

Output
------
    output/<experiment>/reports/bracket_epistasis.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TEST, SPLIT_TRAIN, load_cache
from ssllab.experiment import experiment
from ssllab.generative.perturb import load_operator_algebra

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate operator brackets with empirical epistasis.")
    p.add_argument("--experiment", type=str, default="norman_operator_algebra")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--operator", type=str, default=None, help="defaults to <exp>/checkpoints/operator_algebra.pt")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--n-perm", type=int, default=10000, help="permutation-null resamples for significance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on ranks; nan if a side is constant."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def _perm_p(x: np.ndarray, y: np.ndarray, rho: float, n_perm: int, rng) -> float:
    """One-sided permutation p-value for a positive rank correlation."""
    if not np.isfinite(rho) or len(x) < 3:
        return float("nan")
    count = 0
    for _ in range(n_perm):
        if _spearman(x, rng.permutation(y)) >= rho:
            count += 1
    return (count + 1) / (n_perm + 1)


def empirical_rel_gi(cache, combo: str, top_k: int, min_cells: int):
    """rel_GI for one combo, model-free. None if any of {combo, singleA, singleB} is too small."""
    names = np.asarray(cache.pert_names)
    name_to_id = {n: int(i) for i, n in enumerate(names)}
    control_mean = cache.hvg_X[cache.is_control.astype(bool)].mean(0)

    def delta(name: str):
        pid = name_to_id.get(name)
        if pid is None:
            return None
        mask = cache.pert_id == pid
        if int(mask.sum()) < min_cells:
            return None
        return cache.hvg_X[mask].mean(0) - control_mean

    a, b = combo.split("+")
    dAB, dA, dB = delta(combo), delta(a), delta(b)
    if dAB is None or dA is None or dB is None:
        return None
    idx = np.asarray([i for i in cache.de_genes["per_pert"][combo]["top_idx"][:top_k] if i < cache.hvg_X.shape[1]])
    resid = (dAB - (dA + dB))[idx]
    effect = float(np.linalg.norm(dAB[idx]))
    return float(np.linalg.norm(resid)) / effect if effect > 0 else None


def collect(cache, bundle, split_code: int, top_k: int, min_cells: int):
    """(bracket, rel_gi, name) for every evaluable combo in a split."""
    names = np.asarray(cache.pert_names)
    split_col = cache.split_combo
    ids = np.unique(cache.pert_id[split_col == split_code])
    combos = sorted(names[p] for p in ids if "+" in str(names[p]) and names[p] in cache.de_genes["per_pert"])
    rows = []
    for combo in combos:
        rel = empirical_rel_gi(cache, combo, top_k, min_cells)
        if rel is None:
            continue
        pid = int(np.where(names == combo)[0][0])
        with torch.no_grad():
            br = float(bundle["model"].bracket_norm(torch.tensor([pid])))
        rows.append({"pair": combo, "bracket": br, "rel_gi": rel})
    return rows


def report_block(name: str, rows: list[dict], n_perm: int, rng) -> dict:
    if len(rows) < 3:
        logger.info("%-14s  too few pairs (%d) to correlate", name, len(rows))
        return {"n": len(rows), "spearman": float("nan"), "perm_p": float("nan"), "pairs": rows}
    br = np.array([r["bracket"] for r in rows])
    gi = np.array([r["rel_gi"] for r in rows])
    rho = _spearman(br, gi)
    p = _perm_p(br, gi, rho, n_perm, rng)
    logger.info("%-14s  n=%2d  Spearman(||[M_A,M_B]||, rel_GI) = %+.3f   perm-p = %.4f",
                name, len(rows), rho, p)
    return {"n": len(rows), "spearman": rho, "perm_p": p,
            "pairs": sorted(rows, key=lambda r: r["bracket"], reverse=True)}


def main() -> None:
    args = parse_args()
    exp = experiment(args.experiment, args.output_root).ensure()
    op_path = Path(args.operator) if args.operator else (exp.checkpoints / "operator_algebra.pt")
    if not op_path.exists():
        raise FileNotFoundError(f"no trained operator-algebra checkpoint at {op_path}; run 16 first")

    bundle = load_operator_algebra(op_path, args.device)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    rng = np.random.default_rng(args.seed)

    in_rows = collect(cache, bundle, SPLIT_TRAIN, args.top_k, args.min_cells)
    out_rows = collect(cache, bundle, SPLIT_TEST, args.top_k, args.min_cells)

    logger.info("bracket-epistasis correlation (top-%d DE genes)", args.top_k)
    in_block = report_block("in-sample", in_rows, args.n_perm, rng)
    out_block = report_block("held-out", out_rows, args.n_perm, rng)

    # The held-out correlation is the pre-committed primary endpoint of this round.
    verdict = "n/a"
    if np.isfinite(out_block["spearman"]):
        rho, p = out_block["spearman"], out_block["perm_p"]
        verdict = (
            "CONFIRMED: the bracket predicts held-out epistasis (positive, permutation-significant)."
            if rho > 0 and p < 0.05 else
            "DIRECTIONAL: positive but not significant on 20 held-out pairs (underpowered, as warned)."
            if rho > 0 else
            "REFUTED: the bracket does not track held-out epistasis. The thesis does not hold here."
        )
    logger.info("PRIMARY ENDPOINT (held-out): %s", verdict)

    exp.write_report("bracket_epistasis", {
        "top_k": args.top_k, "min_cells": args.min_cells, "n_perm": args.n_perm,
        "operator": str(op_path), "de_genes_ranked_by": cache.de_genes.get("ranked_by"),
        "in_sample": in_block, "held_out": out_block, "verdict": verdict,
    })
    logger.info("wrote report -> %s", exp.reports / "bracket_epistasis.json")


if __name__ == "__main__":
    main()
