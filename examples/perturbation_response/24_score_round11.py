"""Round 11: decoder calibration. Does counting the library size over the modeled genes fix the decoder?

Round 10 found that the negative-binomial count decoder's readout is shifted and compressed against
measured expression, the same way for every condition, and that this offset sits inside every
published predicted change. A diagnosis on control cells (training data, never scored) found that
the decoder's rate profile ρ is too flat: for highly expressed genes it is about 2.5 times too
small. The leading hypothesis, not yet a proven cause, is the library size in the likelihood. Every
decoder so far was trained with μ = L·ρ where L is the total over *all* measured genes, about 3.2
times the total over the 5,000 genes the decoder models; a softmax ρ sums to 1, so the likelihood is
fit with means about 3.2 times the observed counts, and the surplus probability mass is pushed to
genes where over-prediction costs least. This round changes only that total.

PROTOCOL (fixed in writing before any round-11 run)
==================================================

Arms, three training seeds each (0, 1, 2), the combo split's training cells:

    dec_all   decoder only on the frozen encoder norman_stage_a (md5 38f51067), library size over
              all measured genes (the published convention), co-adaptation/02_coadapt_decoder.py
              --mode dec --library-size all: a fresh decoder, 30 epochs, AdamW 1e-3, gradient
              clipping 5.0                                                  norman_r11_dec_all_s{s}
    dec_hvg   the same with --library-size hvg (the total over the 5,000 modeled genes)
                                                                            norman_r11_dec_hvg_s{s}
    vae_all   the conditioned NB-VAE of round 6 (08_train_cvae_baseline.py --encoder-cond, 60
              epochs), library size all; the existing runs                  norman_vae_cond_s{s}
    vae_hvg   the same recipe with --library-size hvg                       norman_r11_vae_hvg_s{s}

Flows are not retrained: the encoder does not change, so the transport flows norman_sweep_control_s{s}
are read through each decoder arm's seed-s decoder (experiments norman_r11_flow_{all,hvg}_s{s}).

PRIMARY ENDPOINT. Δ-correlation under the measured control (the published convention), on the 20
held-out pairs, each pair's top 20 genes. One max-t family of two contrasts:
    roundtrip     the readout ceiling (real held-out cells → frozen encoder → decoder, mean of
                  log1p(1e4·ρ)), 14_ceiling_analysis.py: dec_hvg − dec_all
    cvae          the cVAE's benchmark prediction (09_eval_cvae_baseline.py, 200 cells from the
                  prior, generation seed 0): vae_hvg − vae_all
Seed mean per pair, then the mean over pairs; joint paired bootstrap over pairs, 10,000 resamples,
seed 0, max-t simultaneous 95% intervals (ssllab.eval.bootstrap). Rule per contrast: it passes if
its estimate is at least +0.02 and its simultaneous interval lies wholly above 0, which establishes
a positive gain under the rule, not a true gain above 0.02; it is worse if the interval lies wholly
below 0; otherwise it does not pass.

WHETHER THE OFFSET IS FIXED: magnitude and calibration diagnostics, judged per model family against
its own all-gene baseline (dec_hvg against dec_all, vae_hvg against vae_all), on 3,000 control
cells drawn with random seed 0 (training data, never scored), on the 500 genes with the highest
measured mean in those cells:
    (i)   rate readout: mean over cells of log1p(1e4·ρ) against the measured mean; offset (mean
          difference on the top-500 genes) and slope (least-squares, all genes). DESCRIPTIVE ONLY,
          shown beside the data's population-level Jensen gap (log1p of the mean proportion against
          the mean of log1p, +0.198 and slope 1.071), which is context, not an exact target: the
          readout averages per-cell log rates, so the two need not coincide even for a perfect decoder.
    (ii)  sampled readout: counts sampled from the decoder's negative binomial at each control
          cell's own total over the modeled genes, normalized like the data (each sampled cell by
          its own total), averaged; offset and slope as in (i). A like-for-like decoder brings
          these near 0 and 1. Judgement for each family's hvg arm, seed-averaged:
              fixed      |offset| ≤ 0.1 and slope in [0.9, 1.1];
              reduced    |offset| at most half its baseline's, and slope closer to 1 than its baseline's;
              not fixed  otherwise.
    (iii) the coefficient of variation of the sampled per-cell totals in (ii), against the measured
          one; descriptive.
For the decoders, the control cells' latents come from the frozen encoder; for the cVAE, from its
prior with the control condition (its generative prediction for control cells). Sampling uses the
global random generator seeded 0 before each arm. The result's wording follows this judgement: a
primary pass without a fixed or reduced offset is not reported as the cause being found.

Secondaries (percentile intervals on the shared resample, no verdicts).
    1. The primary contrasts under the own-control convention (round 10), and e_p under both.
    2. The readout gap (linear − roundtrip) per decoder arm under both conventions; the linear
       readout is round 10's (norman_ceiling_r10, the same frozen encoder).
    3. Full pipeline: the transport flow read through each decoder arm (Δr and e_p, both
       conventions, dec_hvg − dec_all); each cVAE arm against the additive reference.
    4. Population quality and calibration, every arm sampled at the same two depths, 4,832 (the
       median real total over the modeled genes) and 13,921 (the published depth, the median
       all-gene library size), with sampled counts normalized like the data; arms compared only
       within a depth. The published normalization at the published depth is a continuity column
       for the flow arms. Definitions (10_eval_calibration.py, ssllab.eval.calibration): 200
       generated cells per pair, generation seed 0; on each pair's top 20 genes,
           coverage  the fraction of (true cell, gene) entries inside the gene's central 80%
                     interval of the generated cells (0.1 and 0.9 quantiles), pooled over the 20
                     genes; the target is the nominal 0.80, not the maximum, and the signed gap
                     to 0.80 is reported;
           spread    Pearson correlation across the 20 genes of generated and true per-gene SDs;
           W1        one-dimensional Wasserstein-1 per gene, averaged over the 20 genes (marginals);
           energy    energy distance computed jointly on the 20-gene space, Euclidean, V-statistic.

Gates, checked before any arm is read. (a) On the pod before training, the existing decoder's
readout ceiling reproduces the stored roundtrip 0.67916 to within 1e-4 (norman_ceiling_r11_gate).
(b) Every ceiling run's identity arm scores 1.000. (c) vae_all's Δr recomputed here from its saved
profiles matches its stored report per pair to within 1e-6 (the cross-path tolerance of round 10's
amended gate), and each new run's 19 job reproduced its own fresh report on the harness path. (d)
Every new checkpoint records its library size, and records the one its arm names. If a gate
fails, nothing is read.

Aggregation and missing values as round 10: per pair and seed, seed mean per pair, mean over
pairs; per metric, a pair enters only if every seed of every arm involved is finite; the primary
verdict needs at least 18 of the 20 pairs.

Usage
-----
    python examples/perturbation_response/24_score_round11.py --out output/round11_compare.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import load_cache, make_gene_partition
from ssllab.eval.bootstrap import maxt_family, percentile_interval, resample_indices
from ssllab.generative.count_decoder import normalize_sampled
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.generative.perturb import load_count_decoder
from ssllab.latents import encode_rows

_spec = importlib.util.spec_from_file_location("round10", Path(__file__).with_name("23_score_round10.py"))
round10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(round10)
Scorer = round10.Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
DEPTHS = (4832, 13921)
DEC = {"dec_all": "all", "dec_hvg": "hvg"}
VAE = {"vae_all": ("norman_vae_cond_s{}", "all"), "vae_hvg": ("norman_r11_vae_hvg_s{}", "hvg")}
JENSEN_REF = {"offset": 0.198, "slope": 1.071}
POP = ("coverage", "spread_r", "wasserstein", "energy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round 11: library size in the NB likelihood.")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-control", type=int, default=3000)
    p.add_argument("--out", type=str, default="output/round11_compare.json")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="smoke only; the protocol uses 0 1 2")
    return p.parse_args()


def _fail(msg: str) -> None:
    raise SystemExit(f"GATE FAILED: {msg}")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def offset_slope(pred_mean: np.ndarray, meas_mean: np.ndarray, top: np.ndarray) -> dict[str, float]:
    return {"offset": float((pred_mean - meas_mean)[top].mean()),
            "slope": float(np.polyfit(meas_mean, pred_mean, 1)[0])}


def main() -> None:
    args = parse_args()
    global SEEDS
    SEEDS = tuple(args.seeds)
    root = Path(args.output_root)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    sc = Scorer(cache)
    hvg = cache.hvg_X

    # ---- gates --------------------------------------------------------------------------- #
    gate = _json(root / "norman_ceiling_r11_gate/reports/ceiling.json")["arms"]
    if gate["identity"]["mean_delta_r"] <= 0.999 or abs(gate["roundtrip"]["mean_delta_r"] - 0.67916) > 1e-4:
        _fail("norman_ceiling_r11_gate does not reproduce the stored readout ceiling")
    for arm, lib in DEC.items():
        for s in SEEDS:
            rep = _json(root / f"norman_ceiling_r11_{lib}_s{s}/reports/ceiling.json")
            if rep["arms"]["identity"]["mean_delta_r"] <= 0.999:
                _fail(f"identity arm of norman_ceiling_r11_{lib}_s{s}")
            ck = torch.load(root / f"norman_r11_dec_{lib}_s{s}/checkpoints/count_decoder.pt", map_location="cpu")
            if ck.get("library_size") != lib:
                _fail(f"norman_r11_dec_{lib}_s{s} records library size {ck.get('library_size')!r}")
    for s in SEEDS:
        ck = torch.load(root / f"norman_r11_vae_hvg_s{s}/checkpoints/cvae_baseline.pt", map_location="cpu")
        if ck.get("library_size") != "hvg":
            _fail(f"norman_r11_vae_hvg_s{s} records library size {ck.get('library_size')!r}")
    logger.info("gates (a), (b), (d) passed")

    # ---- per-pair scores under both conventions -------------------------------------------- #
    scored: dict[str, list[dict]] = {}
    for arm, lib in DEC.items():
        scored[f"roundtrip_{arm}"] = []
        for s in SEEDS:
            z = np.load(root / f"norman_ceiling_r11_{lib}_s{s}/reports/ceiling_profiles.npz", allow_pickle=True)
            prof = {str(n): p for n, p in zip(z["roundtrip__names"], z["roundtrip__profiles"])}
            scored[f"roundtrip_{arm}"].append(sc.score(prof, z["roundtrip__control"]))
    z = np.load(root / "norman_ceiling_r10/reports/ceiling_profiles.npz", allow_pickle=True)
    scored["linear"] = [sc.score({str(n): p for n, p in zip(z["linear__names"], z["linear__profiles"])},
                                 z["linear__control"])]

    def learned(exp: str, kind: str, report: str, check: bool) -> dict:
        zz = np.load(root / exp / f"reports/profiles_{kind}.npz", allow_pickle=True)
        prof = {str(k): p for k, p in zip(zz["pair_names"], zz["pair_profiles"])}
        out = sc.score(prof, zz["control_profile"])
        if check:
            ref = _json(root / exp / "reports" / report)["per_pert"]
            worst = max(abs(out["measured"][k]["delta_r"] - ref[k]) for k in sc.pairs)
            if worst > 1e-6:
                _fail(f"{exp}: Δr does not reproduce {report} (max |diff| {worst:.2g})")
        return out

    for arm, (pat, _) in VAE.items():
        scored[f"cvae_{arm}"] = [learned(pat.format(s), "vae", "effect_size_cvae.json", True) for s in SEEDS]
    for arm, lib in DEC.items():
        scored[f"flow_{arm}"] = [learned(f"norman_r11_flow_{lib}_s{s}", "flow", "effect_size.json", True) for s in SEEDS]
    scored["additive"] = [learned("norman_additive", "additive", "effect_size_additive.json", True)]
    logger.info("gate (c) passed: every learned run's Δr reproduces its stored report")

    def per_pair(arm: str, conv: str, key: str) -> np.ndarray:
        return np.array([np.mean([run[conv][k][key] for run in scored[arm]]) for k in sc.pairs])

    def masked(*vecs):
        ok = np.all([np.isfinite(v) for v in vecs], axis=0)
        return ok, [v[ok] for v in vecs], resample_indices(int(ok.sum()), args.n_boot, 0)

    # ---- primary ------------------------------------------------------------------------- #
    pr = {"roundtrip": ("roundtrip_dec_hvg", "roundtrip_dec_all"), "cvae": ("cvae_vae_hvg", "cvae_vae_all")}
    vecs = [per_pair(a, "measured", "delta_r") for pair in pr.values() for a in pair]
    ok, sub, idx = masked(*vecs)
    if ok.sum() < 18:
        _fail(f"only {int(ok.sum())} evaluable pairs for the primary")
    diffs = {"roundtrip: dec_hvg − dec_all": sub[0] - sub[1], "cvae: vae_hvg − vae_all": sub[2] - sub[3]}
    fam = maxt_family(diffs, idx, args.alpha)
    for c in fam["contrasts"].values():
        lo, hi = c["ci_simultaneous"]
        c["verdict"] = ("passes (a positive gain under the rule)" if c["diff"] >= 0.02 and lo > 0 else
                        "worse" if hi < 0 else "does not pass")
    excluded = [k for k, g in zip(sc.pairs, ok) if not g]

    # ---- whether the offset is fixed: control-cell diagnostics ------------------------------ #
    rng = np.random.default_rng(0)
    ctrl = rng.choice(np.flatnonzero(cache.is_control.astype(bool)), args.n_control, replace=False)
    meas = hvg[ctrl].mean(0).astype(np.float64)
    top = np.argsort(-meas)[:500]
    cnt = np.asarray(cache.counts[ctrl]).astype(np.float64)
    hvg_tot = torch.as_tensor(cnt.sum(1), dtype=torch.float32)
    cv_meas = float(cnt.sum(1).std() / cnt.sum(1).mean())
    tok = cache.manifest.get("tokenization", {})
    part = make_gene_partition(cache.n_hvg, int(tok.get("n_tokens", 50)), int(tok.get("partition_seed", 0)))
    jepa = load_jepa(root / "norman_stage_a/checkpoints/encoder.pt", "cpu")
    z_ctrl = encode_rows(jepa, hvg, part, ctrl, "cpu", 256)

    def diag_from(rho: torch.Tensor, x: torch.Tensor) -> dict:
        rate = torch.log1p(1e4 * rho).mean(0).numpy().astype(np.float64)
        samp = normalize_sampled(x, hvg_tot, "data").mean(0).numpy().astype(np.float64)
        tot = x.sum(1).numpy()
        return {"rate": offset_slope(rate, meas, top), "sampled": offset_slope(samp, meas, top),
                "sampled_total_cv": float(tot.std() / max(tot.mean(), 1e-12))}

    diag: dict[str, list[dict]] = {}
    with torch.no_grad():
        for arm, lib in DEC.items():
            diag[arm] = []
            for s in SEEDS:
                d = load_count_decoder(root / f"norman_r11_dec_{lib}_s{s}/checkpoints/count_decoder.pt", "cpu")
                torch.manual_seed(0)
                rho = d(z_ctrl, library_size=torch.ones(len(z_ctrl)))["rho"]
                diag[arm].append(diag_from(rho, d.sample_counts(z_ctrl, hvg_tot)))
        cid = int(np.where(np.asarray(cache.pert_names) == "control")[0][0])
        for arm, (pat, _) in VAE.items():
            diag[arm] = []
            for s in SEEDS:
                m = ConditionalNBVAE.from_checkpoint(
                    torch.load(root / pat.format(s) / "checkpoints/cvae_baseline.pt", map_location="cpu"), "cpu")
                torch.manual_seed(0)
                zz = torch.randn(len(ctrl), m.latent_dim)
                out = m.decode(zz, torch.full((len(ctrl),), cid, dtype=torch.long))
                mu = hvg_tot.unsqueeze(-1) * out["rho"]
                kappa = out["kappa"].expand_as(mu)
                x = torch.poisson(torch.distributions.Gamma(kappa, kappa / (mu + 1e-8)).sample())
                diag[arm].append(diag_from(out["rho"], x))

    def dmean(arm, part_, key):
        return float(np.mean([r[part_][key] for r in diag[arm]]))

    judgement = {}
    for fam_name, (new, base) in {"decoder": ("dec_hvg", "dec_all"), "cvae": ("vae_hvg", "vae_all")}.items():
        o_new, s_new = dmean(new, "sampled", "offset"), dmean(new, "sampled", "slope")
        o_base, s_base = dmean(base, "sampled", "offset"), dmean(base, "sampled", "slope")
        if abs(o_new) <= 0.1 and 0.9 <= s_new <= 1.1:
            j = "fixed"
        elif abs(o_new) <= 0.5 * abs(o_base) and abs(s_new - 1) < abs(s_base - 1):
            j = "reduced"
        else:
            j = "not fixed"
        judgement[fam_name] = {"judgement": j, "hvg_offset": o_new, "hvg_slope": s_new,
                               "baseline_offset": o_base, "baseline_slope": s_base}

    # ---- secondaries --------------------------------------------------------------------- #
    def pi(*v, combine=lambda *a: a[0]):
        ok_, sub_, ix = masked(*v)
        r = percentile_interval(combine(*sub_), ix, args.alpha)
        r["n_pairs"] = int(ok_.sum())
        return r

    diff = lambda a, b: a - b
    arms = list(scored)
    table = {a: {conv: {k: pi(per_pair(a, conv, k)) for k in ("delta_r", "rel_err")} for conv in ("measured", "own")}
             for a in arms}
    contrasts = {}
    for name, (a, b) in {**pr, "flow": ("flow_dec_hvg", "flow_dec_all")}.items():
        contrasts[name] = {conv: {k: pi(per_pair(a, conv, k), per_pair(b, conv, k), combine=diff)
                                  for k in ("delta_r", "rel_err")} for conv in ("measured", "own")}
    gap = {a: {conv: pi(per_pair("linear", conv, "delta_r"), per_pair(f"roundtrip_{a}", conv, "delta_r"), combine=diff)
               for conv in ("measured", "own")} for a in DEC}
    vs_add = {a: {conv: {k: pi(per_pair(f"cvae_{a}", conv, k), per_pair("additive", conv, k), combine=diff)
                         for k in ("delta_r", "rel_err")} for conv in ("measured", "own")} for a in VAE}

    def pop_vec(exp_pat: str, model: str, report: str, key: str) -> np.ndarray:
        rows = []
        for s in SEEDS:
            pp = _json(root / exp_pat.format(s) / "reports" / f"{report}.json")["per_pert"]
            rows.append([pp[k][key] for k in sc.pairs])
        v = np.mean(np.array(rows, dtype=float), axis=0)
        return v - 0.8 if key == "coverage_gap" else v

    pop_arms = {"flow_dec_all": ("norman_r11_flow_all_s{}", "flow"), "flow_dec_hvg": ("norman_r11_flow_hvg_s{}", "flow"),
                "vae_all": ("norman_vae_cond_s{}", "vae"), "vae_hvg": ("norman_r11_vae_hvg_s{}", "vae")}
    population: dict = {}
    for d in DEPTHS:
        population[f"d{d}_data"] = {}
        for a, (pat, model) in pop_arms.items():
            population[f"d{d}_data"][a] = {k: pi(pop_vec(pat, model, f"calibration_{model}_d{d}_data", k)) for k in POP}
            population[f"d{d}_data"][a]["coverage_gap_to_0.80"] = {
                "mean": population[f"d{d}_data"][a]["coverage"]["mean"] - 0.8}
        population[f"d{d}_data"]["contrasts"] = {
            fam_name: {k: pi(pop_vec(*pop_arms[new], f"calibration_{pop_arms[new][1]}_d{d}_data", k),
                             pop_vec(*pop_arms[base], f"calibration_{pop_arms[base][1]}_d{d}_data", k), combine=diff)
                       for k in POP}
            for fam_name, (new, base) in {"flow": ("flow_dec_hvg", "flow_dec_all"), "cvae": ("vae_hvg", "vae_all")}.items()}
    population["published_continuity"] = {
        a: {k: pi(pop_vec(pat, "flow", "calibration_flow", k)) for k in POP}
        for a, (pat, _) in pop_arms.items() if a.startswith("flow")}

    # ---- report -------------------------------------------------------------------------- #
    bar = "=" * 96
    print(bar + "\nPRIMARY: Δr under the measured control (the published convention)\n" + bar)
    for a in ("roundtrip_dec_all", "roundtrip_dec_hvg", "cvae_vae_all", "cvae_vae_hvg"):
        print(f"  {a:22s} measured {table[a]['measured']['delta_r']['mean']:.4f}   own {table[a]['own']['delta_r']['mean']:.4f}")
    print(f"  max-t critical value {fam['critical_value']:.3f}")
    for k, c in fam["contrasts"].items():
        print(f"  {k:30s} {c['diff']:+.4f}  simultaneous [{c['ci_simultaneous'][0]:+.4f}, {c['ci_simultaneous'][1]:+.4f}]  -> {c['verdict']}")
    print("\nWHETHER THE OFFSET IS FIXED (control cells; top-500 offset, slope; seed means)")
    for a in diag:
        print(f"  {a:10s} rate offset {dmean(a, 'rate', 'offset'):+.3f} slope {dmean(a, 'rate', 'slope'):.3f}   "
              f"sampled offset {dmean(a, 'sampled', 'offset'):+.3f} slope {dmean(a, 'sampled', 'slope'):.3f}   "
              f"sampled-total cv {np.mean([r['sampled_total_cv'] for r in diag[a]]):.2f} (measured {cv_meas:.2f})")
    print(f"  (rate-readout context only: the data's Jensen gap {JENSEN_REF['offset']:+.3f}, slope {JENSEN_REF['slope']:.3f})")
    for k, v in judgement.items():
        print(f"  {k:8s} -> {v['judgement']}")
    print("\nSECONDARY: contrasts under both conventions (hvg − all; Δr, e_p)")
    for name, v in contrasts.items():
        for conv in ("measured", "own"):
            print(f"  {name:10s} {conv:9s} Δr {v[conv]['delta_r']['mean']:+.3f} [{v[conv]['delta_r']['ci'][0]:+.3f}, {v[conv]['delta_r']['ci'][1]:+.3f}]"
                  f"   e_p {v[conv]['rel_err']['mean']:+.3f} [{v[conv]['rel_err']['ci'][0]:+.3f}, {v[conv]['rel_err']['ci'][1]:+.3f}]")
    print("\nSECONDARY: every arm (Δr / e_p, measured | own)")
    for a in arms:
        t = table[a]
        print(f"  {a:22s} {t['measured']['delta_r']['mean']:.3f} / {t['measured']['rel_err']['mean']:.3f}   |   "
              f"{t['own']['delta_r']['mean']:.3f} / {t['own']['rel_err']['mean']:.3f}")
    print("\nSECONDARY: readout gap, linear − roundtrip (Δr)")
    for a, v in gap.items():
        print(f"  {a:10s} measured {v['measured']['mean']:+.3f} [{v['measured']['ci'][0]:+.3f}, {v['measured']['ci'][1]:+.3f}]"
              f"   own {v['own']['mean']:+.3f} [{v['own']['ci'][0]:+.3f}, {v['own']['ci'][1]:+.3f}]")
    print("\nSECONDARY: population quality and calibration (data-like normalization; coverage target 0.80)")
    for dk in [f"d{d}_data" for d in DEPTHS]:
        print(f"  depth {dk}:")
        for a in pop_arms:
            v = population[dk][a]
            print(f"    {a:14s} coverage {v['coverage']['mean']:.3f} (gap {v['coverage_gap_to_0.80']['mean']:+.3f})  "
                  f"spread {v['spread_r']['mean']:.3f}  W1 {v['wasserstein']['mean']:.3f}  energy {v['energy']['mean']:.3f}")
    print("  published normalization at the published depth (continuity):")
    for a, v in population["published_continuity"].items():
        print(f"    {a:14s} coverage {v['coverage']['mean']:.3f}  spread {v['spread_r']['mean']:.3f}  "
              f"W1 {v['wasserstein']['mean']:.3f}  energy {v['energy']['mean']:.3f}")
    print("\nNOTE: intervals reflect the finite set of pairs; seeds are averaged first.")

    out = {"protocol": "round 11, library size in the NB likelihood, frozen before training (see docstring)",
           "pairs": sc.pairs, "excluded_pairs_primary": excluded, "n_boot": args.n_boot, "alpha": args.alpha,
           "bootstrap_seed": 0, "primary": fam,
           "offset_diagnostics": {"per_arm": diag, "judgement": judgement, "jensen_reference": JENSEN_REF,
                                  "measured_total_cv": cv_meas, "n_control_cells": int(len(ctrl))},
           "secondary": {"table": table, "contrasts": contrasts, "readout_gap": gap, "cvae_vs_additive": vs_add,
                         "population": population}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
