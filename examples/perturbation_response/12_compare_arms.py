"""Compare experiment arms with a JOINT paired bootstrap and simultaneous intervals.

Why this exists, and why it replaces per-contrast bootstrapping done one call at a time.

The unit of analysis is the **perturbation** (the held-out combo), not the cell, not the gene,
not the seed. Every metric here is defined per perturbation, so an arm yields a vector of ~20
numbers rather than a scalar, and that vector is the only sample we have. Two consequences
drive the design:

1. **Contrasts are dependent.** ``transport - gaussian`` and ``transport - vae`` share an arm,
   and every contrast is computed on the *same* 20 perturbations. Bootstrapping each contrast
   in a separate call throws that dependence away and makes a Bonferroni correction (which
   assumes independence) needlessly conservative. So we resample the perturbation indices
   **once per bootstrap iteration** and evaluate *every* contrast on that one resample. The
   dependence structure is preserved by construction.

2. **Multiplicity is real.** With several contrasts times several metrics, testing each at
   alpha=0.05 means expecting a false positive under the global null. So:

   - ONE metric is **primary** (``--primary``, default the effect-size delta-r). Its contrasts
     get a **max-t simultaneous interval**: bootstrap the studentized statistic for every
     primary contrast on the shared resample, take the max |t| across contrasts per iteration,
     and use that distribution's 95th percentile as a single critical value. The resulting
     intervals hold *simultaneously* at 95% over the whole family.
   - Every other metric is **secondary / exploratory**. It is reported with a plain CI and NO
     significance verdict, because a claim discovered on a secondary endpoint after the fact
     is a hypothesis, not a result. Confirm it on fresh data or not at all.

Seeds are averaged **per perturbation before** the bootstrap, so the intervals below describe
uncertainty from the finite 20-perturbation test set, NOT from training randomness. With three
seeds we cannot estimate seed variance well enough to propagate it, so the intervals are
conditional on the seed-averaged model and understate total uncertainty. Stated, not hidden.

Usage
-----
    python examples/perturbation_response/12_compare_arms.py \
        --arm "gaussian=norman_sweep_gaussian_s{s}:flow:0,1,2" \
        --arm "transport=norman_sweep_control_s{s}:flow:0,1,2" \
        --arm "OT=norman_sweep_control_ot_s{s}:flow:0,1,2" \
        --arm "NB-VAE=norman_vae_s{s}:vae:0,1,2" \
        --contrast transport-gaussian --contrast OT-transport --contrast transport-NB-VAE
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Reports an arm writes, by model kind. Effect size stores per_pert as {name: float};
# calibration stores per_pert as {name: {metric: float}}.
REPORTS = {
    "flow": {"effect": "effect_size.json", "calib": "calibration_flow.json"},
    "vae": {"effect": "effect_size_cvae.json", "calib": "calibration_vae.json"},
}
CALIB_METRICS = ["spread_r", "coverage", "wasserstein", "energy"]
LOWER_IS_BETTER = {"wasserstein", "energy"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint paired bootstrap across arms, with simultaneous intervals.")
    p.add_argument("--arm", action="append", required=True,
                   help="label=experiment_pattern:kind:seeds  e.g. transport=norman_sweep_control_s{s}:flow:0,1,2")
    p.add_argument("--contrast", action="append", required=True, help="A-B (uses the arm labels)")
    p.add_argument("--primary", default="delta_r", help="the ONE pre-committed primary metric")
    p.add_argument("--output-root", default="output")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional path to write the JSON summary")
    return p.parse_args()


def load_arm(pattern: str, kind: str, seeds: list[int], root: str) -> dict[str, dict[str, float]]:
    """Seed-average every metric per perturbation. Returns {metric: {pert: value}}."""
    acc: dict[str, dict[str, list[float]]] = {}

    def add(metric: str, pert: str, val: float) -> None:
        acc.setdefault(metric, {}).setdefault(pert, []).append(float(val))

    for s in seeds:
        exp = pattern.format(s=s)
        base = Path(root) / exp / "reports"
        eff = base / REPORTS[kind]["effect"]
        if eff.exists():
            for pert, v in json.loads(eff.read_text())["per_pert"].items():
                if v == v:                                        # drop NaN
                    add("delta_r", pert, v)
        cal = base / REPORTS[kind]["calib"]
        if cal.exists():
            for pert, d in json.loads(cal.read_text())["per_pert"].items():
                for m in CALIB_METRICS:
                    if m in d and d[m] == d[m]:
                        add(m, pert, d[m])
    # A perturbation counts only if every requested seed produced it.
    return {m: {p: float(np.mean(v)) for p, v in d.items() if len(v) == len(seeds)}
            for m, d in acc.items()}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    arms: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.arm:
        label, rest = spec.split("=", 1)
        pattern, kind, seeds = rest.split(":")
        arms[label] = load_arm(pattern, kind, [int(x) for x in seeds.split(",")], args.output_root)
        n_seeds = len(seeds.split(","))
        logger.info("arm %-12s kind=%-4s seeds=%d  metrics=%s", label, kind, n_seeds,
                    ",".join(sorted(arms[label])))

    contrasts = [tuple(c.split("-", 1)) if c.count("-") == 1 else
                 (c.rsplit("-", 1)[0], c.rsplit("-", 1)[1]) for c in args.contrast]
    for a, b in contrasts:
        for x in (a, b):
            if x not in arms:
                raise SystemExit(f"unknown arm {x!r}; known: {list(arms)}")

    metrics = [args.primary] + [m for m in CALIB_METRICS if all(m in arms[a] for a in arms)]

    # The common support: perturbations every arm scored, on every metric we will report.
    keys = None
    for m in metrics:
        for a in arms:
            ks = set(arms[a].get(m, {}))
            keys = ks if keys is None else (keys & ks)
    keys = sorted(keys)
    n = len(keys)
    if n < 5:
        raise SystemExit(f"only {n} shared perturbations; nothing to test")
    logger.info("\nunit of analysis = perturbation.  n = %d shared held-out perturbations.", n)

    V = {a: {m: np.array([arms[a][m][k] for k in keys]) for m in metrics if m in arms[a]} for a in arms}

    # ---- THE joint resample: one set of indices, reused for every contrast and metric. ----
    idx = rng.integers(0, n, size=(args.n_boot, n))
    lo_pct, hi_pct = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)

    def stats(a: str, b: str, m: str):
        d = V[a][m] - V[b][m]
        obs = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n))
        bs = d[idx]                                                    # (B, n)
        bmean = bs.mean(1)
        bse = bs.std(axis=1, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(bse > 1e-12, (bmean - obs) / bse, 0.0)        # studentized, centered
        return obs, se, bmean, t

    # ---- PRIMARY: max-t simultaneous intervals over the whole contrast family. ----
    prim = {c: stats(c[0], c[1], args.primary) for c in contrasts}
    T = np.vstack([np.abs(prim[c][3]) for c in contrasts])             # (n_contrasts, B)
    crit = float(np.percentile(T.max(axis=0), 100 * (1 - args.alpha)))

    out: dict = {"n_perturbations": n, "n_boot": args.n_boot, "alpha": args.alpha,
                 "primary": args.primary, "maxt_critical_value": crit,
                 "perturbations": keys, "arms": {}, "primary_contrasts": {}, "secondary_contrasts": {}}
    for a in arms:
        out["arms"][a] = {m: float(V[a][m].mean()) for m in V[a]}

    logger.info("\n%s", "=" * 92)
    logger.info("PRIMARY ENDPOINT: %s   (pre-committed; the ONLY family we make claims on)", args.primary)
    logger.info("%s", "=" * 92)
    logger.info("arm means:  " + "   ".join(f"{a}={V[a][args.primary].mean():.3f}" for a in arms))
    logger.info("\nmax-t critical value over %d contrasts: %.3f  (vs %.3f for a single unadjusted test)",
                len(contrasts), crit, 1.96)
    logger.info("\n%-26s %8s  %-22s %-24s", "contrast", "diff", "unadjusted 95% CI", "SIMULTANEOUS 95% CI")
    logger.info("%s", "-" * 92)
    for c in contrasts:
        a, b = c
        obs, se, bmean, _ = prim[c]
        ulo, uhi = np.percentile(bmean, [lo_pct, hi_pct])
        slo, shi = obs - crit * se, obs + crit * se
        sig = "**" if (slo > 0 or shi < 0) else "  "
        logger.info("%-26s %+8.3f  [%+.3f, %+.3f]%s [%+.3f, %+.3f] %s",
                    f"{a} - {b}", obs, ulo, uhi, " ", slo, shi, sig)
        out["primary_contrasts"][f"{a}-{b}"] = {
            "diff": obs, "se": se,
            "ci_unadjusted": [float(ulo), float(uhi)],
            "ci_simultaneous": [float(slo), float(shi)],
            "significant_simultaneous": bool(slo > 0 or shi < 0),
        }
    logger.info("\n** = simultaneous interval excludes 0. This is the only significance claim made.")

    # ---- SECONDARY: CIs only. No verdicts. ----
    sec = [m for m in metrics if m != args.primary]
    if sec:
        logger.info("\n%s", "=" * 92)
        logger.info("SECONDARY / EXPLORATORY  (CIs only, NO significance verdicts)")
        logger.info("A difference found here after the fact is a HYPOTHESIS, not a result.")
        logger.info("It needs confirmation on a fresh held-out set, or it does not get claimed.")
        logger.info("%s", "=" * 92)
        for m in sec:
            arrow = "(lower better)" if m in LOWER_IS_BETTER else "(higher better)"
            logger.info("\n%s %s   arm means:  %s", m, arrow,
                        "   ".join(f"{a}={V[a][m].mean():.3f}" for a in arms))
            for c in contrasts:
                a, b = c
                obs, _, bmean, _ = stats(a, b, m)
                ulo, uhi = np.percentile(bmean, [lo_pct, hi_pct])
                logger.info("    %-26s %+8.3f  [%+.3f, %+.3f]", f"{a} - {b}", obs, ulo, uhi)
                out["secondary_contrasts"].setdefault(m, {})[f"{a}-{b}"] = {
                    "diff": obs, "ci_unadjusted": [float(ulo), float(uhi)]}

    logger.info("\nNOTE: seeds are averaged per perturbation BEFORE the bootstrap, so these intervals")
    logger.info("      capture test-set uncertainty only, not training-seed uncertainty.")
    logger.info("      The %d perturbations also share genes, so they are not strictly i.i.d.;", n)
    logger.info("      the bootstrap treats them as such, which if anything makes these CIs too narrow.")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        logger.info("\nwrote %s", args.out)


if __name__ == "__main__":
    main()
