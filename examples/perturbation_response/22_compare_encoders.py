"""The regularizer swap: does SIGReg (LeJEPA) give a better Stage A encoder than VICReg?

Stage A trains the cell encoder by self-supervision (JEPA: predict the embeddings of masked
gene-group tokens from the visible ones), with an anti-collapse regularizer so the encoder
cannot map every cell to the same vector. Every encoder in this project so far used VICReg,
which constrains only second moments: each embedding dimension's spread, and the pairwise
covariances. SIGReg (Balestriero and LeCun, "LeJEPA", 2025) asks the whole embedding
distribution to look like an isotropic standard Gaussian, by testing many random
one-dimensional projections against N(0, 1). This comparison swaps the regularizer and
changes nothing else: the predictor, the EMA teacher and the stop-gradient stay, so any
difference is the regularizer's. (Dropping the teacher, full LeJEPA, is a separate later step.)

It reads the encoder alone. A new encoder would also need a new decoder and flow before the
full pipeline could be scored; that retraining happens only if the encoder passes here.

PROTOCOL (fixed in writing before any SIGReg encoder was trained)
=================================================================

Arms, three training seeds each (0, 1, 2), Stage A on the combo split's training cells, every
setting at 01_pretrain_stage_a.py's defaults (50 epochs, batch 256, learning rate 1e-3,
embedding 256, depth 6, predictor depth 2, 4 heads, 12 masked tokens) except:

    vicreg        VICReg on the context-token embeddings, weight 0.04 (pred + 0.04·VICReg),
                  exactly the existing encoder's recipe          norman_stage_a_vicreg_s{0,1,2}
    sigreg_tok    SIGReg on the context-token embeddings          norman_stage_a_sigreg_tok_s{0,1,2}
    sigreg_pool   SIGReg on the cell vector, the mean of the context-token embeddings
                                                                  norman_stage_a_sigreg_pool_s{0,1,2}

SIGReg settings, the paper's defaults, untuned: weight 0.05 in LeJEPA's convex form,
0.95·pred + 0.05·SIGReg; 1,024 random unit directions redrawn every step; Epps–Pulley
statistic with 17 trapezoid knots on [−5, 5] and Gaussian weight; mean over directions.
The statistic is multiplied by its number of samples, so on the token target it is applied
to each context-token position's batch of B cell embeddings as one view and averaged over
positions, as LeJEPA averages over views; pooled over all B·C token rows at once it would be
about 38 times stronger than on the cell vector at the same weight. Both SIGReg arms
therefore test B samples per group and differ only in what they regularize.
The VICReg arm is retrained at the same three seeds because only one VICReg encoder exists
(seed 42, ``norman_stage_a``); that encoder is reported beside the arms, outside the contrasts,
under the names ``norman_ceiling_vicreg_s42`` and ``norman_stage_a_vicreg_s42`` (probe only).

Primary endpoint. The ceiling analysis's linear readout (14_ceiling_analysis.py, arm
``linear``): a closed-form ridge (λ = 1) from the encoder's cell vector to expression, fit on
training cells only, applied to the real held-out cells of each held-out pair; scored by
Δ-correlation on each pair's top-20 DE genes, for the 20 held-out combinations. It is the
part of the perturbation effect a linear map can read from the frozen cell vector, with no
decoder involved. Today's value is 0.852.

Aggregation. Per pair and seed; seed mean per pair (arithmetic); mean over pairs.

Contrasts. One family: sigreg_tok − vicreg and sigreg_pool − vicreg, per-pair differences of
seed-averaged values. Joint paired bootstrap over pairs, 10,000 resamples, seed 0, max-t
simultaneous 95% intervals (ssllab.eval.bootstrap).

Decision rule (the route's criteria, agreed 2026-11-08). A SIGReg arm passes if its
contrast is at least +0.02 and its simultaneous interval lies wholly above 0. It is worse
than VICReg if the interval lies wholly below 0; otherwise it does not pass. Separately,
the decision to go on to retraining the decoder and flow (not a claim of improvement) is
also triggered if a SIGReg arm's seed-mean probe accuracy or effective rank lies above the
VICReg arm's three-seed maximum. If neither happens for either arm, the regularizer swap
is retired and full LeJEPA is not run on this basis.

Secondaries, percentile intervals on the shared resample, no verdicts.
    1. The size-aware error of the linear readout: e_p = ‖Δ̂_p − Δ_p‖ / ‖Δ_p‖ on the same
       genes (Round 8's endpoint; ssllab.eval.interaction), per arm, against VICReg, and
       beside the additive reference's value.
    2. Interaction recovery of the linear readout (correlation between predicted and
       measured departure from additivity, measured singles).
    3. The linear probe for perturbation identity (02_probe_cell_encoder.py, cells split,
       its defaults): test accuracy, seed mean and range per arm.
    4. Effective rank and mean feature standard deviation of the cell vectors (02's report).

Gates, checked before any arm is read. (a) Each ceiling run's identity arm scores 1.000
(14 enforces it). (b) Code drift: the existing VICReg encoder (``norman_stage_a``, md5
38f51067), re-scored with the same command, reproduces its stored linear readout 0.85214
to within 1e-4; a tolerance, not exact equality, because the stored value may have been
computed on other hardware. If a gate fails, nothing is read.

Missing values and collapse. A pair enters the family only if every arm and seed has a
finite value; excluded pairs are listed; a verdict needs at least 18 of 20 pairs. A run is
flagged collapsed if its effective rank is below 10 or its feature standard deviation below
0.01; collapsed runs stay in their arm (collapse is the regularizer's failure, not missing data).

Usage
-----
    # on the pod, after run_sigreg_pod.sh has trained and scored every run
    python examples/perturbation_response/22_compare_encoders.py --out output/sigreg_compare.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TEST, SPLIT_TRAIN, load_cache
from ssllab.eval.bootstrap import maxt_family, percentile_interval, resample_indices
from ssllab.eval.interaction import measured_single_deltas, pair_metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARMS = ["vicreg", "sigreg_tok", "sigreg_pool"]
REFERENCE = "vicreg"
CONTRASTS = [("sigreg_tok", REFERENCE), ("sigreg_pool", REFERENCE)]
PASS_MARGIN = 0.02
MIN_PAIRS = 18
EXISTING = "norman_stage_a"
EXISTING_LINEAR = 0.85214
DRIFT_TOL = 1e-4
TOP_K = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage A regularizer swap: VICReg vs SIGReg.")
    p.add_argument("--output-root", default="output")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifact", default="norman2019")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def encoder_name(arm: str, seed: int | str) -> str:
    return f"norman_stage_a_{arm}_s{seed}"


def ceiling_name(arm: str, seed: int | str) -> str:
    return f"norman_ceiling_{arm}_s{seed}"


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    seeds = [int(x) for x in args.seeds.split(",")]

    # ---- gate (b): code drift on the existing encoder --------------------------------------- #
    drift = json.loads((root / ceiling_name("vicreg", 42) / "reports" / "ceiling.json").read_text())
    got = drift["arms"]["linear"]["mean_delta_r"]
    if drift["encoder_md5"][:8] != "38f51067" or abs(got - EXISTING_LINEAR) > DRIFT_TOL:
        raise SystemExit(f"GATE FAILED: existing encoder re-scores {got:.5f} (md5 {drift['encoder_md5'][:8]}), "
                         f"stored {EXISTING_LINEAR}; nothing is read")
    logger.info("gate passed: existing VICReg encoder re-scores %.5f (stored %.5f)", got, EXISTING_LINEAR)

    # ---- the truth, for the size-aware secondaries ------------------------------------------- #
    cache = load_cache(Path(args.data_dir) / args.artifact)
    hvg, split = cache.hvg_X, cache.split_combo
    is_test, is_train = split == SPLIT_TEST, split == SPLIT_TRAIN
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)
    names = np.asarray(cache.pert_names)

    def truth(pair: str) -> tuple[np.ndarray, list[int]]:
        pid = int(np.where(names == pair)[0][0])
        top = [i for i in cache.de_genes["per_pert"][pair]["top_idx"][:TOP_K] if i < hvg.shape[1]]
        return hvg[(cache.pert_id == pid) & is_test].mean(0), top

    # ---- load every run ------------------------------------------------------------------------ #
    per: dict[str, dict[str, dict[str, list[float]]]] = {a: {} for a in ARMS + ["existing"]}
    probe: dict[str, list[dict]] = {a: [] for a in ARMS + ["existing"]}
    measured = None
    runs = [(a, s) for a in ARMS for s in seeds] + [("existing", 42)]
    for arm, s in runs:
        cname = ceiling_name("vicreg", 42) if arm == "existing" else ceiling_name(arm, s)
        ename = encoder_name("vicreg", 42) if arm == "existing" else encoder_name(arm, s)
        rep = json.loads((root / cname / "reports" / "ceiling.json").read_text())
        if rep["arms"]["identity"]["mean_delta_r"] <= 0.999:
            raise SystemExit(f"GATE FAILED: {cname} identity arm {rep['arms']['identity']['mean_delta_r']:.4f}")
        prof = np.load(root / cname / "reports" / "ceiling_profiles.npz")
        pairs = [str(p) for p in prof["linear__names"]]
        if measured is None:
            genes = sorted({g for p in pairs for g in p.split("+")})
            measured = measured_single_deltas(genes, hvg_X=hvg, pert_names=cache.pert_names,
                                              pert_id=cache.pert_id, is_train=is_train, control_mean=control_mean)
        for p, x in zip(pairs, prof["linear__profiles"]):
            a, b = p.split("+")
            true_pair, top = truth(p)
            m = pair_metrics(pred_pair=x, true_pair=true_pair, control_mean=control_mean,
                             delta_a=measured[a], delta_b=measured[b], top_idx=top)
            stored = rep["arms"]["linear"]["per_pert"][p]
            if abs(m["delta_r"] - stored) > 1e-5:
                raise SystemExit(f"{cname} {p}: recomputed Δr {m['delta_r']:.6f} != report {stored:.6f}")
            for k in ("delta_r", "rel_err", "rel_gi", "interaction_r"):
                per[arm].setdefault(k, {}).setdefault(p, []).append(m[k])
        pr = json.loads((root / ename / "reports" / "stage_a_probe.json").read_text())
        probe[arm].append({"seed": s, "test_acc": pr["probe"]["test_acc"], "effective_rank": pr["effective_rank"],
                           "feature_std": pr["feature_std"],
                           "collapsed": pr["effective_rank"] < 10 or pr["feature_std"] < 0.01})

    def seed_mean(arm: str, k: str) -> dict[str, float]:
        n_runs = 1 if arm == "existing" else len(seeds)
        return {p: float(np.mean(v)) for p, v in per[arm][k].items() if len(v) == n_runs and np.all(np.isfinite(v))}

    out: dict = {"protocol": "regularizer swap, frozen before training (see the script's docstring)",
                 "seeds": seeds, "n_boot": args.n_boot, "alpha": args.alpha, "drift_gate": got}
    D = {a: seed_mean(a, "delta_r") for a in ARMS}
    support = sorted(set.intersection(*[set(D[a]) for a in ARMS]))
    n = len(support)
    evaluable = n >= MIN_PAIRS
    excluded = sorted(set().union(*[set(D[a]) for a in ARMS]) - set(support))
    logger.info("primary support: %d pairs; excluded: %s", n, excluded or "none")
    out.update({"pairs": support, "excluded_pairs": excluded, "primary_evaluable": evaluable})
    idx = resample_indices(n, args.n_boot, args.seed)
    V = {a: np.array([D[a][p] for p in support]) for a in ARMS}

    # ---- primary ------------------------------------------------------------------------------- #
    logger.info("\n%s\nPRIMARY: linear-readout Δ-correlation (higher better)\n%s", "=" * 90, "=" * 90)
    ex = seed_mean("existing", "delta_r")
    logger.info("%-12s mean %.4f   (existing encoder, seed 42, outside the family)", "existing",
                float(np.mean([ex[p] for p in support if p in ex])))
    for a in ARMS:
        pi = percentile_interval(V[a], idx, args.alpha)
        logger.info("%-12s mean %.4f  [%.4f, %.4f]", a, pi["mean"], *pi["ci"])
    fam = maxt_family({f"{a}-{b}": V[a] - V[b] for a, b in CONTRASTS}, idx, args.alpha)
    logger.info("max-t critical value: %.3f", fam["critical_value"])
    out["primary"] = {"arm_means": {a: float(V[a].mean()) for a in ARMS},
                      "critical_value": fam["critical_value"], "contrasts": {}}
    passed = {}
    for a, b in CONTRASTS:
        k = f"{a}-{b}"
        c = fam["contrasts"][k]
        lo, hi = c["ci_simultaneous"]
        if not evaluable:
            v = "not evaluable (fewer than 18 pairs)"
        elif c["diff"] >= PASS_MARGIN and lo > 0:
            v = f"PASSES (at least +{PASS_MARGIN}, simultaneous interval wholly above 0)"
        elif hi < 0:
            v = "WORSE than VICReg (simultaneous interval wholly below 0)"
        else:
            v = "does not pass"
        passed[a] = v.startswith("PASSES")
        logger.info("%-22s %+.4f  simultaneous [%+.4f, %+.4f]  pairs higher %d/%d\n    -> %s",
                    k, c["diff"], lo, hi, int(((V[a] - V[b]) > 0).sum()), n, v)
        out["primary"]["contrasts"][k] = {**c, "verdict": v}

    # ---- probe, effective rank, and the go/no-go for downstream retraining ------------------- #
    logger.info("\nPROBE AND COLLAPSE (per seed)")
    out["probe"] = probe
    for a in ARMS + ["existing"]:
        for r in probe[a]:
            logger.info("  %-12s seed %-3s probe acc %.4f  eff rank %6.1f  feat std %.3f%s", a, r["seed"],
                        r["test_acc"], r["effective_rank"], r["feature_std"], "  COLLAPSED" if r["collapsed"] else "")
    vmax = {k: max(r[k] for r in probe[REFERENCE]) for k in ("test_acc", "effective_rank")}
    go = {}
    for a, _ in CONTRASTS:
        lifts = {k: float(np.mean([r[k] for r in probe[a]])) > vmax[k] for k in vmax}
        go[a] = passed[a] or any(lifts.values())
        logger.info("  %-12s above VICReg's seed max: probe %s, eff rank %s  -> retrain decoder and flow: %s",
                    a, lifts["test_acc"], lifts["effective_rank"], "YES" if go[a] else "no")
    out["continue_downstream"] = go

    # ---- secondaries --------------------------------------------------------------------------- #
    logger.info("\nSECONDARY (percentile intervals, no verdicts)")
    out["secondary"] = {}
    for k, label, better in (("rel_err", "size-aware error e_p of the linear readout", "lower"),
                             ("interaction_r", "interaction recovery of the linear readout", "higher")):
        S = {a: seed_mean(a, k) for a in ARMS}
        sup = sorted(set.intersection(*[set(S[a]) for a in ARMS]))
        ix = resample_indices(len(sup), args.n_boot, args.seed)
        W = {a: np.array([S[a][p] for p in sup]) for a in ARMS}
        logger.info("\n%s (%s better; %d pairs)", label, better, len(sup))
        res = {"n_pairs": len(sup), "arms": {}, "contrasts": {}}
        for a in ARMS:
            res["arms"][a] = percentile_interval(W[a], ix, args.alpha)
            logger.info("  %-12s %.4f  [%.4f, %.4f]", a, res["arms"][a]["mean"], *res["arms"][a]["ci"])
        for a, b in CONTRASTS:
            res["contrasts"][f"{a}-{b}"] = percentile_interval(W[a] - W[b], ix, args.alpha)
            logger.info("  %-22s %+.4f  [%+.4f, %+.4f]", f"{a}-{b}", res["contrasts"][f"{a}-{b}"]["mean"],
                        *res["contrasts"][f"{a}-{b}"]["ci"])
        if k == "rel_err":
            add = seed_mean("vicreg", "rel_gi")
            res["additive_reference"] = float(np.mean([add[p] for p in sup]))
            logger.info("  additive reference e_p (rel_GI): %.4f", res["additive_reference"])
        out["secondary"][k] = res

    logger.info("\nNOTE: intervals reflect the finite set of %d pairs; seeds are averaged first.", n)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
