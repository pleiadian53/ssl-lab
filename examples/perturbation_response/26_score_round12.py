"""Round 12: co-adaptation Phase 1. Does training the encoder and the decoder together raise the stack's readout ceiling?

With the corrected likelihood (round 11), the stack's readout ceiling, real held-out cells read
through the frozen JEPA encoder and the count decoder, is 0.865 in Δ-correlation, below the
conditioned cVAE's generative 0.939. The transition (the flow) has about 0.03 of headroom under
that ceiling; about 0.135 is lost by the frozen encoder and the decoder reading real cells. Phase 1
trains the encoder and a fresh decoder together by reconstruction, with the flow not co-trained,
and asks whether that raises the ceiling and whether a flow retrained on the co-adapted latents
carries the gain into the full pipeline.

PROTOCOL (fixed in writing before any round-12 run)
==================================================

Arms, three seeds each (0, 1, 2), the combo split's 96,322 training cells, the corrected likelihood
(library size over the 5,000 modeled genes), co-adaptation/02_coadapt_decoder.py (ssllab.coadapt):
a fresh count decoder of the default architecture, 30 epochs, batch 256, AdamW, decoder learning
rate 1e-3, encoder (and predictor) learning rate 1e-4, gradient clipping 5.0, encoder initialized
from norman_stage_a (md5 38f51067):

    dec            --mode dec: the encoder frozen. The runs are round 11's norman_r11_dec_hvg_s{s},
                   trained by this trainer with this recipe and these flags; re-read here
    joint          --mode joint: every encoder parameter trains with the decoder; the predictor is
                   unused                                                   norman_r12_joint_s{s}
    joint_anchor   --mode joint_anchor: joint, while maintaining the JEPA predictive objective (the
                   original recipe, smooth-L1 prediction + 0.04 VICReg) with weight w fixed by rule
                   before any update: w = mean NB loss of the reference decoder norman_r11_dec_hvg_s0
                   / mean JEPA loss of the loaded JEPA, on the first 20 training batches, in eval mode
                   without gradients; a mean JEPA loss below 1e-6 stops the run. The predictor
                   trains at the encoder's rate; the EMA teacher is rebuilt from the loaded encoder
                   (a new fine-tuning start, not Stage A's teacher, which was never saved) and
                   follows the student, so the anchor keeps an objective satisfied and does not
                   penalize distance from the original encoder        norman_r12_joint_anchor_s{s}
No run resumes: an interrupted run restarts from scratch.

Readout ceiling: 14_ceiling_analysis.py --arms identity roundtrip linear --save-profiles on each
arm's own encoder and decoder (experiments norman_ceiling_r12_{arm}_s{s}). Full pipeline: the round-1
transport flow recipe (04_train_cond_flow.py --split combo --cond-type geneset --flow-base control
--coupling independent --epochs 60 --seed s) retrained on each moved encoder's latents (norman_r12_
flow_{joint,joint_anchor}_s{s}); the dec arm reuses norman_sweep_control_s{s}, trained on its
unchanged encoder; every pipeline reads out through its arm's decoder. Predictions as every round
(200 cells, generation seed 0, 100 Euler steps, guidance 1).

PRIMARY ENDPOINT. Readout-ceiling Δ-correlation (roundtrip) under the measured control, on the 20
held-out pairs, top 20 genes. One max-t family of two contrasts, joint − dec and joint_anchor − dec,
seed mean per pair then mean over pairs, joint paired bootstrap over pairs (10,000 resamples, seed
0). Rule per contrast: it passes if the estimate is at least +0.02 and the simultaneous interval
lies wholly above 0, which establishes a positive gain under the rule (a gain above 0.02 needs the
lower bound above 0.02, and the result is worded accordingly); it is worse if the interval lies
wholly below 0; otherwise it does not pass.

Reconstruction discourages complete information loss, but rank, reconstruction quality and
task-relevant information are monitored, and reported as secondaries.

Secondaries (percentile intervals on the shared resample, no verdicts).
    1. The readout ceiling's e_p, and both under the own-control convention.
    2. The linear readout of each arm's encoder (Δr, e_p): did the representation itself change?
    3. Full pipeline, transport flow → arm decoder: Δr and e_p under both conventions, each joint
       arm − dec; and every arm against the conditioned cVAE with the corrected likelihood
       (norman_r11_vae_hvg_s{s}) and against additivity.
    4. Transition headroom per arm: readout ceiling − full pipeline.
    5. Population quality and calibration of the full pipelines, at the defaults (sampled cells
       normalized like the data, at the median real depth over the modeled genes): coverage (the
       fraction of true (cell, gene) entries inside each gene's central 80% interval, pooled over
       the top 20 genes; target the nominal 0.80, signed gap reported), spread (correlation of
       per-gene SDs across the 20 genes), W1 (per gene, averaged: marginals), energy distance
       (joint over the 20 genes, Euclidean, V-statistic); joint arms − dec.
    6. Representation: effective rank and feature spread of the cell vectors; the perturbation
       probe's train and test accuracy (02_probe_cell_encoder.py); linear CKA between each
       encoder's cell vectors and the original encoder's, on 2,000 training cells drawn with
       random seed 0 (how far the representation actually moved).
    7. Training diagnostics from each run's report: final NB loss, anchor weight, and the
       encoder-gradient norm from each loss term on the first batch of each epoch.

Gates, checked before any arm is read. (a) Each dec-arm ceiling reproduces round 11's roundtrip for
the same seed (norman_ceiling_r11_hvg_s{s}) to within 1e-4. (b) Every identity arm scores 1.000.
(c) Every pipeline's Δr recomputed here from its saved profiles matches its stored report per pair
to within 1e-6 (each 19 job also reproduced its own report on the harness path to 1e-9). (d) Every
joint-arm report records its mode and the corrected library size, and its decoder checkpoint
records the corrected library size. If a gate fails, nothing is read.

Aggregation and missing values as rounds 10 and 11.

Usage
-----
    python examples/perturbation_response/26_score_round12.py --out output/round12_compare.json
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
from ssllab.data.perturbseq import SPLIT_TRAIN, load_cache, make_gene_partition
from ssllab.eval.bootstrap import maxt_family, percentile_interval, resample_indices
from ssllab.latents import encode_rows

_spec = importlib.util.spec_from_file_location("round10", Path(__file__).with_name("23_score_round10.py"))
round10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(round10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
ARMS = ("dec", "joint", "joint_anchor")
POP = ("coverage", "spread_r", "wasserstein", "energy")


def _fail(msg: str) -> None:
    raise SystemExit(f"GATE FAILED: {msg}")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    num = np.linalg.norm(y.T @ x) ** 2
    return float(num / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def main() -> None:
    p = argparse.ArgumentParser(description="Round 12: co-adaptation Phase 1.")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="smoke only; the protocol uses 0 1 2")
    p.add_argument("--out", type=str, default="output/round12_compare.json")
    args = p.parse_args()
    seeds = tuple(args.seeds)
    root = Path(args.output_root)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    sc = round10.Scorer(cache)
    pairs = sc.pairs

    def js(path: Path) -> dict:
        return json.loads(path.read_text())

    def enc_path(arm, s):
        return root / "norman_stage_a/checkpoints/encoder.pt" if arm == "dec" else root / f"norman_r12_{arm}_s{s}/checkpoints/encoder.pt"

    def dec_path(arm, s):
        return (root / f"norman_r11_dec_hvg_s{s}/checkpoints/count_decoder.pt" if arm == "dec"
                else root / f"norman_r12_{arm}_s{s}/checkpoints/count_decoder.pt")

    # ---- gates --------------------------------------------------------------------------- #
    for s in seeds:
        got = js(root / f"norman_ceiling_r12_dec_s{s}/reports/ceiling.json")["arms"]["roundtrip"]["mean_delta_r"]
        want = js(root / f"norman_ceiling_r11_hvg_s{s}/reports/ceiling.json")["arms"]["roundtrip"]["mean_delta_r"]
        if abs(got - want) > 1e-4:
            _fail(f"dec s{s} roundtrip {got:.5f} != round 11's {want:.5f}")
    for arm in ARMS:
        for s in seeds:
            if js(root / f"norman_ceiling_r12_{arm}_s{s}/reports/ceiling.json")["arms"]["identity"]["mean_delta_r"] <= 0.999:
                _fail(f"identity arm, {arm} s{s}")
            if torch.load(dec_path(arm, s), map_location="cpu").get("library_size") != "hvg":
                _fail(f"{arm} s{s}: decoder does not record the corrected library size")
            if arm != "dec":
                rep = js(root / f"norman_r12_{arm}_s{s}/reports/coadapt.json")
                if rep.get("mode") != arm or rep.get("library_size") != "hvg":
                    _fail(f"{arm} s{s}: report records mode {rep.get('mode')!r}, library size {rep.get('library_size')!r}")
    logger.info("gates (a), (b), (d) passed")

    # ---- scores -------------------------------------------------------------------------- #
    scored: dict[str, list[dict]] = {}
    for arm in ARMS:
        for rung in ("roundtrip", "linear"):
            scored[f"{rung}_{arm}"] = []
            for s in seeds:
                z = np.load(root / f"norman_ceiling_r12_{arm}_s{s}/reports/ceiling_profiles.npz", allow_pickle=True)
                prof = {str(n): q for n, q in zip(z[f"{rung}__names"], z[f"{rung}__profiles"])}
                scored[f"{rung}_{arm}"].append(sc.score(prof, z[f"{rung}__control"]))

    def learned(exp, kind, report):
        z = np.load(root / exp / f"reports/profiles_{kind}.npz", allow_pickle=True)
        out = sc.score({str(k): q for k, q in zip(z["pair_names"], z["pair_profiles"])}, z["control_profile"])
        ref = js(root / exp / "reports" / report)["per_pert"]
        worst = max(abs(out["measured"][k]["delta_r"] - ref[k]) for k in pairs)
        if worst > 1e-6:
            _fail(f"{exp}: Δr does not reproduce {report} (max |diff| {worst:.2g})")
        return out

    for arm in ARMS:
        scored[f"pipeline_{arm}"] = [learned(f"norman_r12_flow_{arm}_s{s}", "flow", "effect_size.json") for s in seeds]
    scored["cvae_hvg"] = [learned(f"norman_r11_vae_hvg_s{s}", "vae", "effect_size_cvae.json") for s in seeds]
    scored["additive"] = [learned("norman_additive", "additive", "effect_size_additive.json")]
    logger.info("gate (c) passed")

    def per_pair(a, conv, key):
        return np.array([np.mean([r[conv][k][key] for r in scored[a]]) for k in pairs])

    def masked(*v):
        ok = np.all([np.isfinite(x) for x in v], axis=0)
        return ok, [x[ok] for x in v], resample_indices(int(ok.sum()), args.n_boot, 0)

    # ---- primary ------------------------------------------------------------------------- #
    v = [per_pair(f"roundtrip_{a}", "measured", "delta_r") for a in ARMS]
    ok, sub, idx = masked(*v)
    if ok.sum() < 18:
        _fail(f"only {int(ok.sum())} evaluable pairs")
    fam = maxt_family({"joint − dec": sub[1] - sub[0], "joint_anchor − dec": sub[2] - sub[0]}, idx, args.alpha)
    for c in fam["contrasts"].values():
        lo, hi = c["ci_simultaneous"]
        c["verdict"] = ("passes (a positive gain under the rule)" if c["diff"] >= 0.02 and lo > 0 else
                        "worse" if hi < 0 else "does not pass")

    # ---- secondaries --------------------------------------------------------------------- #
    def pi(*vecs, combine=lambda *a: a[0]):
        ok_, sub_, ix = masked(*vecs)
        r = percentile_interval(combine(*sub_), ix, args.alpha)
        r["n_pairs"] = int(ok_.sum())
        return r

    diff = lambda a, b: a - b
    table = {a: {conv: {k: pi(per_pair(a, conv, k)) for k in ("delta_r", "rel_err")} for conv in ("measured", "own")}
             for a in scored}
    vs_dec = {f"{rung}_{a}": {conv: {k: pi(per_pair(f"{rung}_{a}", conv, k), per_pair(f"{rung}_dec", conv, k), combine=diff)
                                      for k in ("delta_r", "rel_err")} for conv in ("measured", "own")}
              for rung in ("roundtrip", "linear", "pipeline") for a in ("joint", "joint_anchor")}
    vs_ref = {f"pipeline_{a}": {ref: {k: pi(per_pair(f"pipeline_{a}", "measured", k), per_pair(ref, "measured", k), combine=diff)
                                      for k in ("delta_r", "rel_err")} for ref in ("cvae_hvg", "additive")} for a in ARMS}
    headroom = {a: pi(per_pair(f"roundtrip_{a}", "measured", "delta_r"), per_pair(f"pipeline_{a}", "measured", "delta_r"),
                      combine=diff) for a in ARMS}

    population = {}
    for a in ARMS:
        reps = [js(root / f"norman_r12_flow_{a}_s{s}/reports/calibration_flow.json") for s in seeds]
        if any(r.get("normalize") != "data" for r in reps):
            _fail(f"{a}: population report not normalized like the data")
        population[a] = {k: np.mean([[r["per_pert"][q][k] for q in pairs] for r in reps], axis=0) for k in POP}
    pop_table = {a: {k: pi(population[a][k]) for k in POP} for a in ARMS}
    pop_vs_dec = {a: {k: pi(population[a][k], population["dec"][k], combine=diff) for k in POP} for a in ("joint", "joint_anchor")}

    rep_diag = {}
    rng = np.random.default_rng(0)
    rows = rng.choice(np.flatnonzero(cache.split_combo == SPLIT_TRAIN), 2000, replace=False)
    tok = cache.manifest.get("tokenization", {})
    part = make_gene_partition(cache.n_hvg, int(tok.get("n_tokens", 50)), int(tok.get("partition_seed", 0)))
    z0 = encode_rows(load_jepa(enc_path("dec", 0), "cpu"), cache.hvg_X, part, rows, "cpu", 256).numpy()
    for a in ARMS:
        rep_diag[a] = []
        for s in seeds:
            pr = js(root / f"norman_ceiling_r12_{a}_s{s}/reports/stage_a_probe.json")
            z = z0 if a == "dec" else encode_rows(load_jepa(enc_path(a, s), "cpu"), cache.hvg_X, part, rows, "cpu", 256).numpy()
            d = {"probe_train_acc": pr["probe"]["train_acc"], "probe_test_acc": pr["probe"]["test_acc"],
                 "effective_rank": pr["effective_rank"], "feature_std": pr["feature_std"], "cka_to_original": linear_cka(z, z0)}
            if a != "dec":
                h = js(root / f"norman_r12_{a}_s{s}/reports/coadapt.json")
                d.update({"final_nb_loss": h["final_nb_loss"], "anchor_weight": (h.get("anchor_rule") or {}).get("anchor_weight"),
                          "grad_nb_last": h["history"][-1].get("grad_nb"), "grad_anchor_last": h["history"][-1].get("grad_anchor")})
            rep_diag[a].append(d)

    # ---- report -------------------------------------------------------------------------- #
    bar = "=" * 96
    print(bar + "\nPRIMARY: readout-ceiling Δr (roundtrip, measured control)\n" + bar)
    for a in ARMS:
        print(f"  {a:14s} {table[f'roundtrip_{a}']['measured']['delta_r']['mean']:.4f}")
    print(f"  max-t critical value {fam['critical_value']:.3f}")
    for k, c in fam["contrasts"].items():
        print(f"  {k:20s} {c['diff']:+.4f}  simultaneous [{c['ci_simultaneous'][0]:+.4f}, {c['ci_simultaneous'][1]:+.4f}]  -> {c['verdict']}")
    print("\nSECONDARY: every readout (Δr / e_p; measured | own)")
    for a in scored:
        t = table[a]
        print(f"  {a:24s} {t['measured']['delta_r']['mean']:.3f} / {t['measured']['rel_err']['mean']:.3f}   |   "
              f"{t['own']['delta_r']['mean']:.3f} / {t['own']['rel_err']['mean']:.3f}")
    print("\nSECONDARY: joint arms − dec (measured; Δr, e_p)")
    for k, v in vs_dec.items():
        m = v["measured"]
        print(f"  {k:24s} Δr {m['delta_r']['mean']:+.3f} [{m['delta_r']['ci'][0]:+.3f}, {m['delta_r']['ci'][1]:+.3f}]"
              f"   e_p {m['rel_err']['mean']:+.3f} [{m['rel_err']['ci'][0]:+.3f}, {m['rel_err']['ci'][1]:+.3f}]")
    print("\nSECONDARY: full pipeline against the corrected cVAE and additivity (measured)")
    for k, v in vs_ref.items():
        for ref, m in v.items():
            print(f"  {k:20s} − {ref:9s} Δr {m['delta_r']['mean']:+.3f} [{m['delta_r']['ci'][0]:+.3f}, {m['delta_r']['ci'][1]:+.3f}]"
                  f"   e_p {m['rel_err']['mean']:+.3f} [{m['rel_err']['ci'][0]:+.3f}, {m['rel_err']['ci'][1]:+.3f}]")
    print("\nSECONDARY: transition headroom (ceiling − pipeline, Δr)")
    for a, v in headroom.items():
        print(f"  {a:14s} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    print("\nSECONDARY: population quality and calibration of the pipelines (coverage target 0.80)")
    for a, v in pop_table.items():
        print(f"  {a:14s} coverage {v['coverage']['mean']:.3f} (gap {v['coverage']['mean'] - 0.8:+.3f})  spread {v['spread_r']['mean']:.3f}"
              f"  W1 {v['wasserstein']['mean']:.3f}  energy {v['energy']['mean']:.3f}")
    print("\nSECONDARY: representation and training (seed means)")
    for a, rs in rep_diag.items():
        keys = sorted({k for r in rs for k in r if r[k] is not None})
        print(f"  {a:14s} " + "  ".join(f"{k} {np.mean([r[k] for r in rs if r.get(k) is not None]):.4g}" for k in keys))
    print("\nNOTE: intervals reflect the finite set of pairs; seeds are averaged first.")

    out = {"protocol": "round 12, co-adaptation Phase 1, frozen before training (see docstring)", "pairs": pairs,
           "n_boot": args.n_boot, "alpha": args.alpha, "bootstrap_seed": 0, "primary": fam,
           "secondary": {"table": table, "vs_dec": vs_dec, "vs_reference": vs_ref, "transition_headroom": headroom,
                         "population": pop_table, "population_vs_dec": pop_vs_dec, "representation": rep_diag}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
