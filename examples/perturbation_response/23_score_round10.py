"""Round 10: the control reference. Is there a decoder gap once each readout is read against its own control?

Every learned model's predicted change has been scored as its predicted pair profile minus the
*measured* control mean, while the true change is the measured pair mean minus the same measured
control mean. A readout through the negative-binomial count decoder reports the log of an
expected rate, ``log1p(1e4·ρ)``; the truth is the mean of ``log1p`` of noisy normalized counts.
The two differ systematically, mostly at low counts, so a decoded profile can carry an offset
against measured expression that is the same for every perturbation. Subtracting the measured
control mean leaves that offset inside every predicted change. Taking the change against the
readout's *own* prediction for the control cells removes the part of it that is shared between
control and perturbed cells. This round rescores existing checkpoints under both conventions.
Nothing is trained.

Two conventions, for a readout with predicted pair profile ``P_p``, its own prediction for the
control condition ``C``, and the measured control mean ``m`` (all profiles over the 5,000 HVGs,
log1p-CP10K):

    measured   Δ̂_p = P_p − m                       (the published convention, rounds 1–9)
    own        Δ̂_p = P_p − C                       (computed as the shifted profile P_p − C + m)

The truth is always Δ_p = mean of the pair's held-out cells − m. The own-control prediction
uses only control cells, which are training data, so it reads nothing held out.

PROTOCOL (fixed in writing before any round-10 number was scored)
================================================================

What was already seen, disclosed. A magnitude audit on 2026-11-08, run before this protocol as
the preflight the Phase 1 plan required, computed the two conventions descriptively with a scratch
script for: the readout ceiling on the frozen encoder (measured Δr 0.679, e_p 1.274; own Δr 0.858,
e_p 0.649, own control from 3,000 random control cells), the transport flow (measured 0.648 /
1.261; own 0.826 / 0.682) and the conditioned NB-VAE (measured 0.858 / 1.053; own 0.930 / 0.497),
their own controls generated with generation seed 0. The primary below is therefore not blind.
What the pre-registration protects is the endpoint, the rule and the arm list, written before
this round's runs. Not seen before this protocol: the linear readout under the own convention,
the condition-blind NB-VAE, the Gaussian flow, both action operators and every round-9 encoder.
This round's control predictions differ in detail from the audit's (all control cells for the
ceiling; generation seed + 2 for the models), so its numbers can differ slightly from the audit's.

Readouts and their control predictions.
    Readout ceiling, frozen encoder ``norman_stage_a`` (md5 38f51067) and the shared count decoder
    (md5 52a44337): ``14_ceiling_analysis.py --arms identity roundtrip latent_mean linear
    --save-profiles``, experiment ``norman_ceiling_r10``. ``roundtrip`` decodes each held-out
    pair's real cells and averages; its own control decodes every control cell (the cells ``m``
    averages) the same way. ``linear`` is the closed-form ridge (λ = 1) from the cell vector to
    expression, fit on training cells; its own control applies the ridge to every control cell.
    ``latent_mean`` decodes the mean latent of the same cells (exploratory). ``identity`` is the
    truth itself, and its own control is ``m``.
    Round-9 encoders: the same script with ``--arms identity linear --save-profiles`` for each of
    ``norman_stage_a_{vicreg,sigreg_tok,sigreg_pool}_s{0,1,2}``, experiments
    ``norman_ceiling_r10_{arm}_s{seed}``.
    Learned models: ``19_score_interaction.py`` regenerates each run's pair predictions exactly as
    its own evaluation did (200 cells, generation seed 0; flows 100 Euler steps, guidance 1; flows
    and operators through the shared decoder) and then the model's prediction for the control
    condition with the same settings and a separate generator seeded 2. Runs, seeds 0, 1, 2:
        vae_cond        norman_vae_cond_s{s}          vae
        flow_transport  norman_sweep_control_s{s}     flow
        vae_blind       norman_vae_blind_s{s}         vae        (exploratory)
        flow_gaussian   norman_sweep_gaussian_s{s}    flow       (exploratory)
        operator        norman_operator_s{s}          operator   (exploratory)
        operator_stoch  norman_operator_stoch_s{s}    operator   (exploratory)
    The additive reference (``norman_additive``) uses the measured control by construction, so its
    two conventions coincide.

Pairs and genes. The 20 held-out combinations of the combo split with at least 20 held-out cells,
each pair's top 20 genes by Wilcoxon |z| (``de_genes.json`` md5 9844c2c2), log1p-CP10K.

PRIMARY ENDPOINT. The readout gap under the own convention, on the frozen encoder:
    gap_p = Δr_p(linear, own) − Δr_p(roundtrip, own),
averaged over the 20 pairs (one run; no training seeds are involved). Interval: joint paired
bootstrap over pairs, 10,000 resamples, seed 0, max-t simultaneous 95% interval for the family of
this one contrast (``ssllab.eval.bootstrap``).

Decision rule. The gap is *established* if its estimate is at least +0.02 and its simultaneous
interval lies wholly above 0. That establishes a positive gap under this rule; it does not
establish that the true gap exceeds 0.02, which would need the interval's lower bound above 0.02,
and the result is worded accordingly. If the interval lies wholly below 0, the decoder readout is
better than the linear readout under this convention. Otherwise no gap is established. Consequence,
agreed before scoring: co-adaptation Phase 1 is re-planned against whichever gap this round
finds, on the convention this round supports.

Secondaries (plain percentile intervals on the shared resample, no verdicts).
    1. Round 8 under the own convention: mean e_p for vae_cond − additive and flow_transport −
       additive, and the same contrasts on Δr.
    2. Every readout and model under both conventions: mean Δr, mean e_p, mean interaction
       correlation (against the measured singles), with the convention effect own − measured on
       Δr and e_p.
    3. The readout gap under the measured convention (it reproduces 0.852 − 0.679).
    4. Round 9 under the own convention: sigreg_tok − vicreg and sigreg_pool − vicreg on the
       linear readout's Δr and e_p (round 9's verdict stands under its own protocol).
    5. The size of the offset: for each readout, the mean of C − m over each pair's top-20 genes,
       and the slope of C against m across all genes.

Population quality and calibration (coverage, spread, W1, energy) are not rescored: they are
computed from sampled populations against held-out cells, not from a predicted change, and are
unaffected by the control reference.

Aggregation. Per pair and seed; the seed mean per pair (arithmetic); the mean over pairs.

Gates, checked before any arm is read. (a) Every ceiling run's identity arm scores 1.000 (14
enforces it), and its own control equals ``m``, so the two conventions coincide for the truth. (b)
``norman_ceiling_r10`` reproduces the stored ceiling (``norman_ceiling``): roundtrip 0.67916 and
linear 0.85214 in mean Δr, to within 1e-4. (c) Each round-9 rerun reproduces the linear Δr stored
by round 9 (``norman_ceiling_{arm}_s{seed}``) to within 1e-4. (d) For every learned run, the
measured-convention Δr recomputed here from the saved profiles matches the run's stored
effect-size report per pair to 1e-9, and a control profile is present. The tolerance in (b) and
(c) is not exact equality because the ceilings are recomputed on a GPU. If a gate fails, nothing
is read.

Missing values. A pair enters a metric only if every seed of every arm involved gives a finite
value; excluded pairs are listed; the primary verdict needs at least 18 of the 20 pairs.

Usage
-----
    python examples/perturbation_response/23_score_round10.py --out output/round10_compare.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
PRIMARY_CEILING = "norman_ceiling_r10"
STORED_CEILING = "norman_ceiling"
ROUND9_ARMS = ("vicreg", "sigreg_tok", "sigreg_pool")
LEARNED = {   # name: (experiment pattern, kind, role)
    "vae_cond": ("norman_vae_cond_s{}", "vae", "primary"),
    "flow_transport": ("norman_sweep_control_s{}", "flow", "primary"),
    "vae_blind": ("norman_vae_blind_s{}", "vae", "exploratory"),
    "flow_gaussian": ("norman_sweep_gaussian_s{}", "flow", "exploratory"),
    "operator": ("norman_operator_s{}", "operator", "exploratory"),
    "operator_stoch": ("norman_operator_stoch_s{}", "operator", "exploratory"),
}
EFFECT_REPORT = {"flow": "effect_size.json", "operator": "effect_size.json", "vae": "effect_size_cvae.json",
                 "additive": "effect_size_additive.json"}
KEYS = ("delta_r", "rel_err", "interaction_r")
CONVENTIONS = ("measured", "own")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round 10: the control reference.")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=str, default="output/round10_compare.json")
    return p.parse_args()


class Scorer:
    """Per-pair metrics of a profile set under both conventions."""

    def __init__(self, cache, top_k: int = 20, min_test_cells: int = 20) -> None:
        self.hvg = cache.hvg_X
        self.names = np.asarray(cache.pert_names)
        self.m = self.hvg[cache.is_control.astype(bool)].mean(0).astype(np.float64)
        is_test = cache.split_combo == SPLIT_TEST
        is_train = cache.split_combo == SPLIT_TRAIN
        de = cache.de_genes["per_pert"]
        self.pairs, self.truth, self.top = [], {}, {}
        for k in self.names:
            if "+" not in k or k not in de:
                continue
            rows = (cache.pert_id == int(np.where(self.names == k)[0][0])) & is_test
            if rows.sum() < min_test_cells:
                continue
            self.pairs.append(str(k))
            self.truth[k] = self.hvg[rows].mean(0)
            self.top[k] = [i for i in de[k]["top_idx"][:top_k] if i < self.hvg.shape[1]]
        genes = sorted({g for k in self.pairs for g in k.split("+")})
        self.singles = measured_single_deltas(genes, hvg_X=self.hvg, pert_names=cache.pert_names,
                                              pert_id=cache.pert_id, is_train=is_train, control_mean=self.m)

    def score(self, profiles: dict[str, np.ndarray], control: np.ndarray) -> dict[str, dict[str, dict]]:
        """``{convention: {pair: {delta_r, rel_err, interaction_r}}}``."""
        out = {c: {} for c in CONVENTIONS}
        shift = self.m - np.asarray(control, dtype=np.float64)
        for k in self.pairs:
            a, b = k.split("+")
            for conv, pred in (("measured", profiles[k]), ("own", profiles[k] + shift)):
                mm = pair_metrics(pred_pair=np.asarray(pred, dtype=np.float64), true_pair=self.truth[k],
                                  control_mean=self.m, delta_a=self.singles[a], delta_b=self.singles[b],
                                  top_idx=self.top[k])
                out[conv][k] = {key: float(mm[key]) for key in KEYS}
        return out

    def offset(self, control: np.ndarray) -> dict[str, float]:
        c = np.asarray(control, dtype=np.float64)
        top = sorted({i for k in self.pairs for i in self.top[k]})
        return {"mean_offset_top_genes": float((c - self.m)[top].mean()),
                "slope_vs_measured": float(np.polyfit(self.m, c, 1)[0])}


def _fail(msg: str) -> None:
    raise SystemExit(f"GATE FAILED: {msg}")


def load_ceiling(root: Path, exp: str, arm: str) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    rep = json.loads((root / exp / "reports/ceiling.json").read_text())
    if rep["arms"]["identity"]["mean_delta_r"] <= 0.999:
        _fail(f"{exp}: identity arm scores {rep['arms']['identity']['mean_delta_r']:.4f}")
    z = np.load(root / exp / "reports/ceiling_profiles.npz", allow_pickle=True)
    prof = {str(n): p for n, p in zip(z[f"{arm}__names"], z[f"{arm}__profiles"])}
    return prof, z[f"{arm}__control"], rep["arms"][arm]["mean_delta_r"]


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    sc = Scorer(cache)
    n = len(sc.pairs)
    idx = resample_indices(n, args.n_boot, 0)
    logger.info("%d pairs", n)

    # ---- gates and per-arm scores -------------------------------------------------------- #
    scored: dict[str, list[dict]] = {}     # arm -> list over seeds of {conv: {pair: metrics}}
    offsets: dict[str, list[dict]] = {}

    z = np.load(root / PRIMARY_CEILING / "reports/ceiling_profiles.npz", allow_pickle=True)
    if not np.allclose(z["identity__control"], sc.m, atol=1e-6):
        _fail("identity arm's own control is not the measured control mean")
    stored = json.loads((root / STORED_CEILING / "reports/ceiling.json").read_text())["arms"]
    for arm in ("roundtrip", "linear", "latent_mean"):
        prof, ctrl, got = load_ceiling(root, PRIMARY_CEILING, arm)
        if arm in ("roundtrip", "linear") and abs(got - stored[arm]["mean_delta_r"]) > 1e-4:
            _fail(f"{PRIMARY_CEILING} {arm} {got:.5f} != stored {stored[arm]['mean_delta_r']:.5f}")
        scored[f"ceiling_{arm}"] = [sc.score(prof, ctrl)]
        offsets[f"ceiling_{arm}"] = [sc.offset(ctrl)]
    logger.info("gate passed: %s reproduces the stored roundtrip and linear ceiling", PRIMARY_CEILING)

    for arm in ROUND9_ARMS:
        scored[f"r9_{arm}"], offsets[f"r9_{arm}"] = [], []
        for s in SEEDS:
            exp = f"norman_ceiling_r10_{arm}_s{s}"
            prof, ctrl, got = load_ceiling(root, exp, "linear")
            want = json.loads((root / f"norman_ceiling_{arm}_s{s}/reports/ceiling.json").read_text())["arms"]["linear"]["mean_delta_r"]
            if abs(got - want) > 1e-4:
                _fail(f"{exp} linear {got:.5f} != round 9's {want:.5f}")
            scored[f"r9_{arm}"].append(sc.score(prof, ctrl))
            offsets[f"r9_{arm}"].append(sc.offset(ctrl))
    logger.info("gate passed: every round-9 rerun reproduces its stored linear readout")

    def load_learned(exp: str, kind: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        z = np.load(root / exp / f"reports/profiles_{kind}.npz", allow_pickle=True)
        prof = {str(k): p for k, p in zip(z["pair_names"], z["pair_profiles"])}
        ctrl = z["control_profile"]
        if ctrl.size != sc.hvg.shape[1]:
            _fail(f"{exp}: no control profile (rerun 19_score_interaction.py)")
        ref = json.loads((root / exp / "reports" / EFFECT_REPORT[kind]).read_text())["per_pert"]
        got = sc.score(prof, ctrl)["measured"]
        worst = max(abs(got[k]["delta_r"] - ref[k]) for k in sc.pairs)
        if worst > 1e-9:
            _fail(f"{exp}: measured Δr does not reproduce {EFFECT_REPORT[kind]} (max |diff| {worst:.2g})")
        return prof, ctrl

    for name, (pat, kind, _) in LEARNED.items():
        scored[name], offsets[name] = [], []
        for s in SEEDS:
            prof, ctrl = load_learned(pat.format(s), kind)
            scored[name].append(sc.score(prof, ctrl))
            offsets[name].append(sc.offset(ctrl))
    prof, ctrl = load_learned("norman_additive", "additive")
    if not np.allclose(ctrl, sc.m, atol=1e-6):
        _fail("the additive reference's control is not the measured control mean")
    scored["additive"], offsets["additive"] = [sc.score(prof, ctrl)], [sc.offset(ctrl)]
    logger.info("gate passed: every learned run reproduces its stored Δr and has a control profile")

    # ---- seed means per pair ------------------------------------------------------------- #
    def per_pair(arm: str, conv: str, key: str) -> np.ndarray:
        return np.array([np.mean([run[conv][k][key] for run in scored[arm]]) for k in sc.pairs])

    finite = np.all([np.isfinite(per_pair(a, c, k)) for a in scored for c in CONVENTIONS for k in KEYS], axis=0)
    excluded = [k for k, ok in zip(sc.pairs, finite) if not ok]
    if finite.sum() < 18:
        _fail(f"only {int(finite.sum())} evaluable pairs")
    keep = np.flatnonzero(finite)
    idx = resample_indices(len(keep), args.n_boot, 0)

    def vec(arm, conv, key):
        return per_pair(arm, conv, key)[keep]

    # ---- primary ------------------------------------------------------------------------- #
    gap = vec("ceiling_linear", "own", "delta_r") - vec("ceiling_roundtrip", "own", "delta_r")
    fam = maxt_family({"linear-roundtrip (own)": gap}, idx, args.alpha)
    c = fam["contrasts"]["linear-roundtrip (own)"]
    lo, hi = c["ci_simultaneous"]
    verdict = ("gap established (positive under the rule; not a claim that it exceeds 0.02 unless the lower bound does)"
               if c["diff"] >= 0.02 and lo > 0 else
               "decoder readout better than the linear readout" if hi < 0 else "no gap established")
    c["verdict"] = verdict

    # ---- secondaries --------------------------------------------------------------------- #
    def pi(v):
        return percentile_interval(v, idx, args.alpha)

    table = {a: {conv: {k: pi(vec(a, conv, k)) for k in KEYS} for conv in CONVENTIONS} for a in scored}
    effect = {a: {k: pi(vec(a, "own", k) - vec(a, "measured", k)) for k in ("delta_r", "rel_err")} for a in scored}
    r8_own = {f"{a}-additive": {k: pi(vec(a, "own", k) - vec("additive", "own", k)) for k in ("rel_err", "delta_r")}
              for a in ("vae_cond", "flow_transport")}
    gap_measured = pi(vec("ceiling_linear", "measured", "delta_r") - vec("ceiling_roundtrip", "measured", "delta_r"))
    r9_own = {f"{a}-vicreg": {k: pi(vec(f"r9_{a}", "own", k) - vec("r9_vicreg", "own", k)) for k in ("delta_r", "rel_err")}
              for a in ("sigreg_tok", "sigreg_pool")}
    off = {a: {k: float(np.mean([o[k] for o in v])) for k in v[0]} for a, v in offsets.items()}

    # ---- report -------------------------------------------------------------------------- #
    bar = "=" * 92
    print(bar + "\nPRIMARY: readout gap under the own-control convention (linear − decoder roundtrip, Δr)\n" + bar)
    for a in ("ceiling_linear", "ceiling_roundtrip"):
        t = table[a]
        print(f"  {a:20s} own {t['own']['delta_r']['mean']:.4f}   measured {t['measured']['delta_r']['mean']:.4f}")
    print(f"  gap (own)        {c['diff']:+.4f}  simultaneous [{lo:+.4f}, {hi:+.4f}]  (critical value {fam['critical_value']:.3f})")
    print(f"    -> {verdict}")
    print(f"  gap (measured)   {gap_measured['mean']:+.4f}  [{gap_measured['ci'][0]:+.4f}, {gap_measured['ci'][1]:+.4f}]  (secondary)")
    print("\nSECONDARY: every readout under both conventions (mean over pairs; percentile intervals)")
    print(f"  {'arm':20s} {'Δr measured':>12s} {'Δr own':>10s} {'e_p measured':>13s} {'e_p own':>10s} {'offset':>8s} {'slope':>6s}")
    for a in scored:
        t = table[a]
        print(f"  {a:20s} {t['measured']['delta_r']['mean']:12.3f} {t['own']['delta_r']['mean']:10.3f} "
              f"{t['measured']['rel_err']['mean']:13.3f} {t['own']['rel_err']['mean']:10.3f} "
              f"{off[a]['mean_offset_top_genes']:+8.3f} {off[a]['slope_vs_measured']:6.3f}")
    print("\nSECONDARY: round 8 under the own convention (e_p, Δr; learned − additive)")
    for k, v in r8_own.items():
        print(f"  {k:24s} e_p {v['rel_err']['mean']:+.3f} [{v['rel_err']['ci'][0]:+.3f}, {v['rel_err']['ci'][1]:+.3f}]   "
              f"Δr {v['delta_r']['mean']:+.3f} [{v['delta_r']['ci'][0]:+.3f}, {v['delta_r']['ci'][1]:+.3f}]")
    print("\nSECONDARY: round 9 under the own convention (linear readout; SIGReg − VICReg)")
    for k, v in r9_own.items():
        print(f"  {k:24s} Δr {v['delta_r']['mean']:+.3f} [{v['delta_r']['ci'][0]:+.3f}, {v['delta_r']['ci'][1]:+.3f}]   "
              f"e_p {v['rel_err']['mean']:+.3f} [{v['rel_err']['ci'][0]:+.3f}, {v['rel_err']['ci'][1]:+.3f}]")
    print("\nNOTE: intervals reflect the finite set of pairs; seeds are averaged first.")

    out = {"protocol": "round 10, the control reference, frozen before scoring (see the script's docstring)",
           "pairs": [sc.pairs[i] for i in keep], "excluded_pairs": excluded, "n_boot": args.n_boot,
           "alpha": args.alpha, "bootstrap_seed": 0,
           "primary": {"critical_value": fam["critical_value"], "contrast": c},
           "secondary": {"table": table, "convention_effect": effect, "round8_own": r8_own,
                         "gap_measured": gap_measured, "round9_own": r9_own, "offset": off},
           "per_pair": {a: {conv: {k: per_pair(a, conv, k).tolist() for k in KEYS} for conv in CONVENTIONS}
                        for a in scored}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
