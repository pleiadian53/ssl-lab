r"""Empirical epistasis — the data-side target for the operator-algebra round.

The operator-algebra design (`dev/planning/action_operator/03-...`) rests on one claim: in a model
where each single gene has its own generator M_g, the *non-commutativity* of two generators,
||[M_A, M_B]||, tracks the *genetic interaction* of the pair. That claim needs a data-side quantity
to correlate against, and it needs that quantity to actually VARY across pairs. If every held-out
Norman combination were near-additive, the bracket would have nothing to predict and the round would
be dead before a line of the model is written.

This script computes the model-free target and checks for variation. The **empirical epistasis** of a
pair is how far the double perturbation departs from the sum of its singles, which is the standard
operational definition of a genetic interaction:

    Delta(X) = mean_expression(cells of X) - control_mean          the effect of perturbation X
    additive = Delta(A) + Delta(B)                                 what a no-interaction model predicts
    residual = Delta(A+B) - additive                               the epistasis
    GI(A,B)  = ||residual||  on the combo's top-DE genes           its magnitude

Everything comes from the cache: the combo's cells (held out) give Delta(A+B); the two single-gene
perturbations (in the training split, both verified present) give Delta(A) and Delta(B). No model.

Reported per pair, on the same top-DE support the benchmark scores:

    GI              ||residual||                      absolute epistasis (UNSIGNED: a norm)
    effect          ||Delta(A+B)||                    how big the combo's effect is at all
    rel_GI          GI / effect                        epistasis as a fraction of the effect
    add_r           corr(Delta(A+B), additive)         additivity: 1.0 = perfectly additive, low = epistatic
    add_cos         cosine(Delta(A+B), additive)       direction agreement with the additive model

``GI`` is a norm, so on its own it can say the additive model is wrong but never whether the pair did
MORE than the sum of its singles or LESS. That distinction is the difference between synergy and
buffering, so it needs a signed quantity. Fit the least-squares scale ``lam`` that best stretches the
additive prediction onto the truth, and the epistasis splits orthogonally:

    true = lam * add + perp,   perp _|_ add
    GI^2 = (lam-1)^2 ||add||^2  +  ||perp||^2
            \___ magnitude ___/     \_ direction _/

    scale (lam)     >1 super-additive (synergy), <1 sub-additive (buffering/redundancy)
    GI_mag          |lam-1| * ||add||                 the pair is the right shape, the wrong size
    GI_dir          ||perp||                          the pair moved genes the additive model did not
    dir%            GI_dir^2 / GI^2                    how much of the interaction is directional

The split matters for the operator thesis, because the algebra treats the two flavors differently: a
magnitude error is what a symmetric quadratic object (an anticommutator) would predict, while a
direction error is the commutator's natural target. Reporting only ``GI`` conflates them.

The verdict this script exists to deliver: does rel_GI (or 1 - add_r) span a real range across the 20
pairs? A spread from near-additive to strongly-interacting is what gives the bracket correlation
something to fit. A tight cluster near zero would kill the thesis cheaply.

Usage
-----
    python examples/perturbation_response/15_empirical_epistasis.py

Output
------
    output/<experiment>/reports/epistasis.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TEST, load_cache
from ssllab.experiment import experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Empirical epistasis of held-out Norman combinations.")
    p.add_argument("--experiment", type=str, default="norman_epistasis")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--top-k", type=int, default=20, help="top-DE genes, matching the benchmark support")
    p.add_argument("--min-cells", type=int, default=20, help="skip a pert with fewer cells than this")
    p.add_argument("--scale-tol", type=float, default=0.10,
                   help="how far the additive scale lam may sit from 1.0 before a pair is called "
                        "super- or sub-additive")
    return p.parse_args()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, guarded against a zero-variance vector."""
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else float("nan")


def main() -> None:
    args = parse_args()
    exp = experiment(args.experiment, args.output_root).ensure()
    cache = load_cache(Path(args.data_dir) / args.artifact)

    hvg = cache.hvg_X                                   # (N, G) normalized log1p-CP10K
    names = np.asarray(cache.pert_names)
    name_to_id = {n: int(i) for i, n in enumerate(names)}
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)
    per_pert_de = cache.de_genes["per_pert"]

    def delta(pert_name: str) -> np.ndarray | None:
        """Effect of a perturbation: its cell-mean minus the control mean. None if too few cells."""
        pid = name_to_id.get(pert_name)
        if pid is None:
            return None
        mask = cache.pert_id == pid
        if int(mask.sum()) < args.min_cells:
            return None
        return hvg[mask].mean(0) - control_mean

    # The held-out combinations: the benchmark's test set.
    test_ids = np.unique(cache.pert_id[cache.split_combo == SPLIT_TEST])
    combos = sorted(names[p] for p in test_ids if names[p] != "control" and names[p] in per_pert_de)

    rows: list[dict] = []
    skipped: list[str] = []
    for combo in combos:
        if "+" not in combo:
            continue
        a, b = combo.split("+")
        dAB, dA, dB = delta(combo), delta(a), delta(b)
        if dAB is None or dA is None or dB is None:
            skipped.append(combo)
            continue
        top = [i for i in per_pert_de[combo]["top_idx"][: args.top_k] if i < hvg.shape[1]]
        idx = np.asarray(top)
        add = (dA + dB)[idx]                       # the no-interaction prediction
        true = dAB[idx]                            # what the double actually did
        gi = float(np.linalg.norm(true - add))
        effect = float(np.linalg.norm(true))

        # Split the epistasis into a MAGNITUDE part and a DIRECTION part, orthogonally.
        # ``lam`` is the least-squares scale that best fits the additive prediction to the truth:
        #
        #     true = lam * add + perp,        with  perp _|_ add
        #
        # so the residual is  (true - add) = (lam - 1) * add + perp  and, because the two pieces are
        # orthogonal, the total epistasis splits exactly:
        #
        #     gi^2 = (lam - 1)^2 ||add||^2  +  ||perp||^2
        #             \__ magnitude __/         \_ direction _/
        #
        # This is what ``gi`` alone cannot tell you. ``gi`` is a norm, so it is UNSIGNED: it says the
        # additive model is off, never whether it under- or over-shot. ``lam`` carries that sign, and
        # the two components separate the two flavors of interaction the algebra treats differently.
        # A magnitude error (lam far from 1, direction fine) is what a symmetric, quadratic object like
        # an anticommutator would predict; a direction error (large perp) is the commutator's target.
        aa = float(add @ add)
        lam = float(true @ add / aa) if aa > 0 else float("nan")
        perp = true - lam * add if aa > 0 else true
        gi_par = abs(lam - 1.0) * float(np.linalg.norm(add)) if aa > 0 else float("nan")
        gi_perp = float(np.linalg.norm(perp))

        if not np.isfinite(lam):
            mode = "undefined"
        elif lam > 1.0 + args.scale_tol:
            mode = "super-additive"          # the pair does MORE than the sum of its singles
        elif lam < 1.0 - args.scale_tol:
            mode = "sub-additive"            # the pair does LESS: buffering / redundancy
        else:
            mode = "magnitude-ok"            # the additive model has the size right

        rows.append({
            "pair": combo,
            "gi": gi,
            "effect": effect,
            "rel_gi": gi / effect if effect > 0 else float("nan"),
            "scale": lam,
            "gi_par": gi_par,
            "gi_perp": gi_perp,
            "perp_share": (gi_perp ** 2 / gi ** 2) if gi > 0 else float("nan"),
            "mode": mode,
            "add_r": _corr(true, add),
            "add_cos": _cos(true, add),
            "n_de": len(idx),
        })

    if not rows:
        raise RuntimeError("no evaluable held-out pair had all of {combo, singleA, singleB} present")

    rel = np.array([r["rel_gi"] for r in rows])
    addr = np.array([r["add_r"] for r in rows])
    rows_sorted = sorted(rows, key=lambda r: r["rel_gi"])

    logger.info("empirical epistasis on %d held-out pairs (top-%d DE genes)%s",
                len(rows), args.top_k,
                f"; skipped {len(skipped)} for <{args.min_cells} cells" if skipped else "")
    logger.info("%-22s %8s %8s %7s %8s %8s %7s  %s",
                "pair", "rel_GI", "add_r", "scale", "GI_mag", "GI_dir", "dir%", "mode")
    for r in rows_sorted:
        logger.info("%-22s %8.3f %8.3f %7.3f %8.3f %8.3f %6.0f%%  %s",
                    r["pair"], r["rel_gi"], r["add_r"], r["scale"],
                    r["gi_par"], r["gi_perp"], 100 * r["perp_share"], r["mode"])

    scale = np.array([r["scale"] for r in rows])
    dirshare = np.array([r["perp_share"] for r in rows])
    modes = {m: sum(1 for r in rows if r["mode"] == m) for m in
             ("super-additive", "sub-additive", "magnitude-ok", "undefined") }
    modes = {m: n for m, n in modes.items() if n}

    summary = {
        "n_pairs": len(rows),
        "rel_gi_min": float(rel.min()), "rel_gi_median": float(np.median(rel)), "rel_gi_max": float(rel.max()),
        "add_r_min": float(np.nanmin(addr)), "add_r_median": float(np.nanmedian(addr)), "add_r_max": float(np.nanmax(addr)),
        "scale_min": float(np.nanmin(scale)), "scale_median": float(np.nanmedian(scale)), "scale_max": float(np.nanmax(scale)),
        "dir_share_median": float(np.nanmedian(dirshare)),
        "modes": modes,
        "most_additive": rows_sorted[0]["pair"], "most_epistatic": rows_sorted[-1]["pair"],
    }
    logger.info("")
    logger.info("rel_GI  range [%.3f, %.3f]  median %.3f", summary["rel_gi_min"], summary["rel_gi_max"], summary["rel_gi_median"])
    logger.info("add_r   range [%.3f, %.3f]  median %.3f", summary["add_r_min"], summary["add_r_max"], summary["add_r_median"])
    logger.info("scale   range [%.3f, %.3f]  median %.3f  (>1 super-additive, <1 sub-additive)",
                summary["scale_min"], summary["scale_max"], summary["scale_median"])
    logger.info("flavor  %s; median %.0f%% of the interaction is DIRECTIONAL",
                ", ".join(f"{n} {m}" for m, n in modes.items()), 100 * summary["dir_share_median"])
    logger.info("most additive: %s   most epistatic: %s", summary["most_additive"], summary["most_epistatic"])

    # The verdict this script exists to render.
    span = summary["rel_gi_max"] - summary["rel_gi_min"]
    verdict = (
        "STRONG spread: the bracket claim has clear signal to fit." if span >= 0.30 else
        "MODERATE spread: fittable but noisy; expect a weak correlation." if span >= 0.15 else
        "TIGHT cluster: little epistasis variation; the bracket claim may have nothing to predict."
    )
    logger.info("VERDICT (rel_GI span %.3f): %s", span, verdict)

    exp.write_report("epistasis", {
        "top_k": args.top_k, "min_cells": args.min_cells,
        "de_genes_ranked_by": cache.de_genes.get("ranked_by"),
        "skipped": skipped, "verdict": verdict, "rel_gi_span": span,
        **summary, "per_pair": rows_sorted,
    })
    logger.info("wrote report -> %s", exp.reports / "epistasis.json")


if __name__ == "__main__":
    main()
