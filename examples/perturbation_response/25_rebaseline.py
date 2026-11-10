"""The re-baseline after round 11: every published model, read under the corrected conventions.

Round 11 found that the count decoder's offset came from the library size in its likelihood (the
total over all measured genes instead of the 5,000 modeled ones), and that the benchmark's sampled
populations were normalized by the sampling depth instead of, as the data was, by each cell's own
total. Both corrections are now the defaults. This script tabulates every model on the published
scoreboard under them, so later rounds and the method chapters can cite current numbers.

It tests no hypothesis and states no verdict. Every interval is a plain percentile interval from a
paired bootstrap over the held-out pairs (10,000 resamples, seed 0), seeds averaged per pair first.

What is read (``run_rebaseline_pod.sh`` produces it):
    flow family     norman_rb_{gaussian,transport,ot,operator,operator_stoch}_s{0,1,2}: the round-1
                    flows and round-3 operators, each seed read through round 11's corrected decoder
                    of the same seed; Δr (06), e_p and the own-control readout (19), population
                    quality and calibration (10), variance decomposition on seed 0 (11)
    cVAE            conditioned: norman_r11_vae_hvg_s{s} (round 11); condition-blind:
                    norman_rb_vae_blind_s{s}, retrained with the corrected likelihood
    additive        norman_additive (unchanged by either correction)
    ladder          norman_rb_ceiling_s{s}: readout ceiling and mean-latent rung through each
                    corrected decoder; the linear rung is round 10's (norman_ceiling_r10)

Population quality and calibration (``ssllab.eval.calibration``): 200 generated cells per pair,
generation seed 0, sampled at the median real total over the modeled genes and normalized like the
data; on each pair's top 20 genes, coverage is the fraction of (true cell, gene) entries inside the
gene's central 80% interval (target: the nominal 0.80, not the maximum), spread is the correlation
of per-gene standard deviations across the 20 genes, W1 is per gene and averaged (marginals only),
and energy distance is joint over the 20 genes (Euclidean, V-statistic).

Usage
-----
    python examples/perturbation_response/25_rebaseline.py --out output/rebaseline_compare.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import load_cache
from ssllab.eval.bootstrap import percentile_interval, resample_indices

_spec = importlib.util.spec_from_file_location("round10", Path(__file__).with_name("23_score_round10.py"))
round10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(round10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
FLOWS = {"gaussian": "flow", "transport": "flow", "ot": "flow", "operator": "operator", "operator_stoch": "operator"}
VAES = {"vae_cond": "norman_r11_vae_hvg_s{}", "vae_blind": "norman_rb_vae_blind_s{}"}
POP = ("coverage", "spread_r", "wasserstein", "energy")


def main() -> None:
    p = argparse.ArgumentParser(description="Re-baseline after round 11.")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", type=str, default="output/rebaseline_compare.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="smoke only; the re-baseline uses 0 1 2")
    args = p.parse_args()
    seeds = tuple(args.seeds)
    root = Path(args.output_root)
    sc = round10.Scorer(load_cache(Path(args.data_dir) / args.artifact))
    pairs = sc.pairs

    def js(path: Path) -> dict:
        return json.loads(path.read_text())

    def pi(v: np.ndarray) -> dict:
        ok = np.isfinite(v)
        r = percentile_interval(v[ok], resample_indices(int(ok.sum()), args.n_boot, 0))
        r["n_pairs"] = int(ok.sum())
        return r

    def scored_run(exp: str, kind: str, report: str) -> dict:
        z = np.load(root / exp / f"reports/profiles_{kind}.npz", allow_pickle=True)
        out = sc.score({str(k): q for k, q in zip(z["pair_names"], z["pair_profiles"])}, z["control_profile"])
        ref = js(root / exp / "reports" / report)["per_pert"]
        worst = max(abs(out["measured"][k]["delta_r"] - ref[k]) for k in pairs)
        if worst > 1e-6:
            raise SystemExit(f"{exp}: Δr does not reproduce {report} (max |diff| {worst:.2g})")
        return out

    runs = {f"flow_{a}": [scored_run(f"norman_rb_{a}_s{s}", k, "effect_size.json") for s in seeds] for a, k in FLOWS.items()}
    runs.update({a: [scored_run(pat.format(s), "vae", "effect_size_cvae.json") for s in seeds] for a, pat in VAES.items()})
    runs["additive"] = [scored_run("norman_additive", "additive", "effect_size_additive.json")]

    def per_pair(arm, conv, key):
        return np.array([np.mean([r[conv][k][key] for r in runs[arm]]) for k in pairs])

    effect = {a: {conv: {k: pi(per_pair(a, conv, k)) for k in ("delta_r", "rel_err", "interaction_r")}
                  for conv in ("measured", "own")} for a in runs}
    vs_additive = {a: {k: pi(per_pair(a, "measured", k) - per_pair("additive", "measured", k)) for k in ("delta_r", "rel_err")}
                   for a in runs if a != "additive"}

    pop_src = {**{f"flow_{a}": (f"norman_rb_{a}_s{{}}", "flow") for a in FLOWS}, **{a: (pat, "vae") for a, pat in VAES.items()}}
    population, variance = {}, {}
    for a, (pat, model) in pop_src.items():
        reps = [js(root / pat.format(s) / f"reports/calibration_{model}.json") for s in seeds]
        if any(r.get("normalize") != "data" for r in reps):
            raise SystemExit(f"{a}: a population report is not normalized like the data")
        population[a] = {k: pi(np.mean([[r["per_pert"][q][k] for q in pairs] for r in reps], axis=0)) for k in POP}
        population[a]["depth"] = reps[0]["depth"]
        v = js(root / pat.format(0) / f"reports/variance_decomposition_{model}.json")
        variance[a] = {k: v[k] for k in ("sigma2_dec", "sigma2_bio", "bio_fraction", "total_pred", "var_real", "spread_ratio")}
        variance[a]["normalize"], variance[a]["depth"] = v.get("normalize"), v.get("depth")

    ladder = {}
    for arm in ("roundtrip", "latent_mean"):
        vals = [js(root / f"norman_rb_ceiling_s{s}/reports/ceiling.json")["arms"][arm] for s in seeds]
        ladder[arm] = {"mean_delta_r": float(np.mean([v["mean_delta_r"] for v in vals])),
                       "per_seed": [v["mean_delta_r"] for v in vals]}
    ladder["linear"] = {"mean_delta_r": js(root / "norman_ceiling_r10/reports/ceiling.json")["arms"]["linear"]["mean_delta_r"]}
    ladder["transport_flow"] = {"mean_delta_r": effect["flow_transport"]["measured"]["delta_r"]["mean"]}

    print("=" * 100 + "\nRE-BASELINE (corrected likelihood; populations normalized like the data at the real depth)\n" + "=" * 100)
    print(f"  {'model':22s} {'Δr':>7s} {'e_p':>7s} {'Δr own':>7s} {'e_p own':>8s} {'cov':>6s} {'spread':>7s} {'W1':>6s} {'energy':>7s}")
    for a in runs:
        e = effect[a]
        line = (f"  {a:22s} {e['measured']['delta_r']['mean']:7.3f} {e['measured']['rel_err']['mean']:7.3f} "
                f"{e['own']['delta_r']['mean']:7.3f} {e['own']['rel_err']['mean']:8.3f}")
        if a in population:
            q = population[a]
            line += (f" {q['coverage']['mean']:6.3f} {q['spread_r']['mean']:7.3f} {q['wasserstein']['mean']:6.3f}"
                     f" {q['energy']['mean']:7.3f}")
        print(line)
    print("\n  against additivity (measured; Δr, e_p):")
    for a, v in vs_additive.items():
        print(f"    {a:22s} Δr {v['delta_r']['mean']:+.3f} [{v['delta_r']['ci'][0]:+.3f}, {v['delta_r']['ci'][1]:+.3f}]"
              f"   e_p {v['rel_err']['mean']:+.3f} [{v['rel_err']['ci'][0]:+.3f}, {v['rel_err']['ci'][1]:+.3f}]")
    print("\n  ladder:", {k: round(v["mean_delta_r"], 3) for k, v in ladder.items()})
    print("  variance decomposition (seed 0):")
    for a, v in variance.items():
        print(f"    {a:22s} σ²_dec {v['sigma2_dec']:.3f}  σ²_bio {v['sigma2_bio']:.3f}  latent share {v['bio_fraction']:.3f}"
              f"  predicted/real {v['spread_ratio']:.3f}")
    out = {"description": "re-baseline after round 11 (see docstring); descriptive, no verdicts",
           "pairs": pairs, "n_boot": args.n_boot, "effect": effect, "vs_additive": vs_additive,
           "population": population, "variance_decomposition_seed0": variance, "ladder": ladder}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
