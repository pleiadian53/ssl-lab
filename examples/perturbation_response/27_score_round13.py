"""Round 13: does the JEPA pretraining contribute to round 12's gain? Joint training from a random start.

Round 12 trained the pretrained JEPA encoder and a fresh count decoder together by reconstruction,
and the stack's readout ceiling rose from 0.865 (frozen encoder) to 0.936. The encoder moved far
from its pretrained form (linear CKA about 0.5), so the round could not say whether the pretraining
contributed or whether joint reconstruction training alone would get there. This round trains the
same architecture with the same recipe from freshly initialized weights.

PROTOCOL (fixed in writing before any round-13 run)
==================================================

Arms, three seeds each (0, 1, 2), the combo split's 96,322 training cells, the corrected likelihood
(library size over the 5,000 modeled genes), co-adaptation/02_coadapt_decoder.py --mode joint: a
fresh count decoder of the default architecture, 30 epochs, batch 256, AdamW, decoder learning rate
1e-3, gradient clipping 5.0, every encoder parameter trained, the JEPA predictor unused:

    joint         --init pretrained, encoder learning rate 1e-4: the encoder starts from
                  norman_stage_a (md5 38f51067). Round 12's runs norman_r12_joint_s{s}, with their
                  readout ceilings and pipelines, re-read here and not retrained
    random_lr4    --init random, encoder learning rate 1e-4: round 12's recipe exactly, with the
                  encoder's starting weights freshly drawn by the JEPA constructor, the
                  initialization Stage A pretraining itself started from     norman_r13_random_lr4_s{s}
    random_lr3    --init random, encoder learning rate 1e-3, the decoder's: the same, with a rate
                  suited to training from scratch, since 1e-4 was chosen for fine-tuning a
                  pretrained encoder and may under-train a random one     norman_r13_random_lr3_s{s}

The random start reads only the checkpoint's config. It draws from the random-number generator
exactly as loading the checkpoint does, so at the same seed the decoder's initialization and the
batch order match the joint arm's, and in random_lr4 only the encoder's starting weights differ.
No run resumes: an interrupted run restarts from scratch.

Readout ceiling: 14_ceiling_analysis.py --arms identity roundtrip linear --save-profiles on each
arm's own encoder and decoder (norman_ceiling_r13_{arm}_s{s}). Full pipeline: the round-1 transport
flow recipe (04_train_cond_flow.py --split combo --cond-type geneset --flow-base control --coupling
independent --epochs 60 --seed s) trained on each random encoder's latents (norman_r13_flow_{arm}_s{s}),
read out through its arm's decoder. Predictions as every round (200 cells, generation seed 0, 100
Euler steps, guidance 1); populations at the defaults (sampled cells normalized like the data, at
the median real total over the modeled genes).

PRIMARY ENDPOINT. Readout-ceiling Δ-correlation (roundtrip) under the measured control, on the 20
held-out pairs, top 20 genes. One max-t family of two contrasts, joint − random_lr4 and joint −
random_lr3, seed mean per pair then mean over pairs, joint paired bootstrap over pairs (10,000
resamples, seed 0). Per contrast: it passes if the estimate is at least +0.02 and the simultaneous
interval lies wholly above 0; it is worse if the interval lies wholly below 0; otherwise it does
not pass. The round's verdict:
    - pretraining's contribution is established only if BOTH contrasts pass, since the pretrained
      start must beat every from-scratch configuration tried. That establishes a positive
      contribution under the rule; one above 0.02 needs both lower bounds above 0.02, and the
      result is worded accordingly;
    - if either contrast is worse, that from-scratch arm reads held-out cells better than the
      pretrained start;
    - otherwise no contribution of pretraining is established. That is not evidence that it
      contributes nothing: the intervals say how large a contribution the data can rule out.

Reconstruction discourages complete information loss, but rank, reconstruction quality and
task-relevant information are monitored, and reported as secondaries.

Secondaries (percentile intervals on the shared resample, no verdicts).
    1. The readout ceiling's e_p, and both under the own-control convention.
    2. The linear readout of each arm's encoder (Δr, e_p), joint − each random arm.
    3. Full pipeline, transport flow → arm decoder: Δr and e_p under both conventions, joint − each
       random arm; every arm against the conditioned cVAE with the corrected likelihood
       (norman_r11_vae_hvg_s{s}) and against additivity.
    4. Each random arm against round 12's frozen-encoder arm dec (norman_ceiling_r12_dec_s{s},
       norman_r12_flow_dec_s{s}): does joint training from scratch beat the frozen pretrained
       encoder? Readout ceiling and pipeline Δr and e_p.
    5. Transition headroom per arm: readout ceiling − full pipeline.
    6. Population quality and calibration of the full pipelines: coverage (the fraction of true
       (cell, gene) entries inside each gene's central 80% interval, pooled over the top 20 genes;
       target the nominal 0.80, signed gap reported), spread (correlation of per-gene SDs across the
       20 genes), W1 (per gene, averaged: marginals), energy distance (joint over the 20 genes,
       Euclidean, V-statistic); joint − each random arm.
    7. Representation: effective rank and feature spread of the cell vectors; the perturbation
       probe's train and test accuracy (02_probe_cell_encoder.py); linear CKA, on 2,000 training
       cells drawn with random seed 0, of each encoder's cell vectors to the original pretrained
       encoder's and to the same seed's joint encoder's.
    8. Training: final NB loss, and the mean NB loss over the last five epochs against the five
       before them (whether a run was still improving when it stopped).

Gates, checked before any arm is read. (a) Every readout ceiling's roundtrip Δr recomputed here from
its saved profiles matches its stored report to within 1e-6, and every joint-arm encoder's md5
matches the md5 its round-12 report recorded. (b) Every identity arm scores 1.000. (c) Every
pipeline's Δr recomputed from its saved profiles matches its stored report per pair to within 1e-6.
(d) Every random-arm report records mode joint, init random, the corrected library size and its
arm's encoder learning rate, with every other training setting equal to the joint arm's; the joint
arm's reports record a pretrained start from md5 38f51067; every decoder checkpoint records the
corrected library size. If a gate fails, nothing is read.

Aggregation and missing values as rounds 10 to 12.

Usage
-----
    python examples/perturbation_response/27_score_round13.py --out output/round13_compare.json
"""

from __future__ import annotations

import argparse
import hashlib
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
ARMS = ("joint", "random_lr4", "random_lr3")
RANDOM = {"random_lr4": 1e-4, "random_lr3": 1e-3}
POP = ("coverage", "spread_r", "wasserstein", "energy")
STAGE_A_MD5 = "38f51067"


def _fail(msg: str) -> None:
    raise SystemExit(f"GATE FAILED: {msg}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    num = np.linalg.norm(y.T @ x) ** 2
    return float(num / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def main() -> None:
    p = argparse.ArgumentParser(description="Round 13: joint training from a random start.")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="smoke only; the protocol uses 0 1 2")
    p.add_argument("--out", type=str, default="output/round13_compare.json")
    p.add_argument("--smoke", action="store_true",
                   help="smoke only: gate (d) ignores epochs and the cell limit, which a smoke run shrinks")
    args = p.parse_args()
    seeds = tuple(args.seeds)
    root = Path(args.output_root)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    sc = round10.Scorer(cache)
    pairs = sc.pairs

    def js(path: Path) -> dict:
        return json.loads(path.read_text())

    run = {"joint": "norman_r12_joint_s{}", "random_lr4": "norman_r13_random_lr4_s{}", "random_lr3": "norman_r13_random_lr3_s{}"}
    ceil = {"joint": "norman_ceiling_r12_joint_s{}", "random_lr4": "norman_ceiling_r13_random_lr4_s{}",
            "random_lr3": "norman_ceiling_r13_random_lr3_s{}", "dec": "norman_ceiling_r12_dec_s{}"}
    flow = {"joint": "norman_r12_flow_joint_s{}", "random_lr4": "norman_r13_flow_random_lr4_s{}",
            "random_lr3": "norman_r13_flow_random_lr3_s{}", "dec": "norman_r12_flow_dec_s{}"}
    readouts = ARMS + ("dec",)

    def enc_path(arm, s):
        return root / run[arm].format(s) / "checkpoints/encoder.pt"

    # ---- gates (a), (b), (d) --------------------------------------------------------------- #
    scored: dict[str, list[dict]] = {}
    for arm in readouts:
        for rung in ("roundtrip", "linear"):
            scored[f"{rung}_{arm}"] = []
        for s in seeds:
            c = js(root / ceil[arm].format(s) / "reports/ceiling.json")["arms"]
            if c["identity"]["mean_delta_r"] <= 0.999:
                _fail(f"identity arm, {arm} s{s}")
            z = np.load(root / ceil[arm].format(s) / "reports/ceiling_profiles.npz", allow_pickle=True)
            for rung in ("roundtrip", "linear"):
                out = sc.score({str(n): q for n, q in zip(z[f"{rung}__names"], z[f"{rung}__profiles"])}, z[f"{rung}__control"])
                if rung == "roundtrip":
                    got = float(np.nanmean([out["measured"][k]["delta_r"] for k in pairs]))
                    if abs(got - c["roundtrip"]["mean_delta_r"]) > 1e-6:
                        _fail(f"{arm} s{s}: roundtrip {got:.6f} does not reproduce {c['roundtrip']['mean_delta_r']:.6f}")
                scored[f"{rung}_{arm}"].append(out)

    reports = {a: [js(root / run[a].format(s) / "reports/coadapt.json") for s in seeds] for a in ARMS}
    for s, rep in zip(seeds, reports["joint"]):
        if rep.get("mode") != "joint" or rep.get("init", "pretrained") != "pretrained" or \
                not str(rep.get("init_encoder_md5", "")).startswith(STAGE_A_MD5):
            _fail(f"joint s{s}: not a pretrained start from {STAGE_A_MD5}")
        if _md5(enc_path("joint", s)) != rep.get("encoder_md5"):
            _fail(f"joint s{s}: encoder on disk differs from the one round 12 recorded")
    for a, lr in RANDOM.items():
        for s, rep, ref in zip(seeds, reports[a], reports["joint"]):
            if rep.get("mode") != "joint" or rep.get("init") != "random" or rep.get("library_size") != "hvg":
                _fail(f"{a} s{s}: report records mode {rep.get('mode')!r}, init {rep.get('init')!r}, "
                      f"library size {rep.get('library_size')!r}")
            cfg, cfg_ref = dict(rep["config"]), dict(ref["config"])
            if abs(cfg.pop("encoder_lr") - lr) > 1e-12:
                _fail(f"{a} s{s}: encoder learning rate is not {lr:g}")
            cfg_ref.pop("encoder_lr")
            if args.smoke:
                cfg.pop("epochs"), cfg_ref.pop("epochs")
            same_limit = args.smoke or rep.get("limit") == ref.get("limit")
            if cfg != cfg_ref or rep.get("split") != ref.get("split") or not same_limit:
                _fail(f"{a} s{s}: a training setting other than the encoder learning rate differs from the joint arm")
    for a in ARMS:
        for s in seeds:
            if torch.load(root / run[a].format(s) / "checkpoints/count_decoder.pt", map_location="cpu").get("library_size") != "hvg":
                _fail(f"{a} s{s}: decoder does not record the corrected library size")
    logger.info("gates (a), (b), (d) passed")

    # ---- gate (c) and the pipelines -------------------------------------------------------- #
    def learned(exp, kind, report):
        z = np.load(root / exp / f"reports/profiles_{kind}.npz", allow_pickle=True)
        out = sc.score({str(k): q for k, q in zip(z["pair_names"], z["pair_profiles"])}, z["control_profile"])
        ref = js(root / exp / "reports" / report)["per_pert"]
        worst = max(abs(out["measured"][k]["delta_r"] - ref[k]) for k in pairs)
        if worst > 1e-6:
            _fail(f"{exp}: Δr does not reproduce {report} (max |diff| {worst:.2g})")
        return out

    for a in readouts:
        scored[f"pipeline_{a}"] = [learned(flow[a].format(s), "flow", "effect_size.json") for s in seeds]
    scored["cvae_hvg"] = [learned(f"norman_r11_vae_hvg_s{s}", "vae", "effect_size_cvae.json") for s in seeds]
    scored["additive"] = [learned("norman_additive", "additive", "effect_size_additive.json")]
    logger.info("gate (c) passed")

    def per_pair(a, conv, key):
        return np.array([np.mean([r[conv][k][key] for r in scored[a]]) for k in pairs])

    def masked(*v):
        ok = np.all([np.isfinite(x) for x in v], axis=0)
        return ok, [x[ok] for x in v], resample_indices(int(ok.sum()), args.n_boot, 0)

    # ---- primary --------------------------------------------------------------------------- #
    v = [per_pair(f"roundtrip_{a}", "measured", "delta_r") for a in ARMS]
    ok, sub, idx = masked(*v)
    if ok.sum() < 18:
        _fail(f"only {int(ok.sum())} evaluable pairs")
    fam = maxt_family({"joint − random_lr4": sub[0] - sub[1], "joint − random_lr3": sub[0] - sub[2]}, idx, args.alpha)
    for c in fam["contrasts"].values():
        lo, hi = c["ci_simultaneous"]
        c["verdict"] = ("passes" if c["diff"] >= 0.02 and lo > 0 else "worse" if hi < 0 else "does not pass")
    verdicts = [c["verdict"] for c in fam["contrasts"].values()]
    lows = [c["ci_simultaneous"][0] for c in fam["contrasts"].values()]
    if all(x == "passes" for x in verdicts):
        fam["round_verdict"] = ("pretraining's contribution is established"
                                + (", above 0.02" if min(lows) > 0.02 else " (a positive contribution under the rule)"))
    elif "worse" in verdicts:
        fam["round_verdict"] = "a from-scratch arm reads held-out cells better than the pretrained start"
    else:
        fam["round_verdict"] = "no contribution of pretraining is established"

    # ---- secondaries ----------------------------------------------------------------------- #
    def pi(*vecs, combine=lambda *a: a[0]):
        ok_, sub_, ix = masked(*vecs)
        r = percentile_interval(combine(*sub_), ix, args.alpha)
        r["n_pairs"] = int(ok_.sum())
        return r

    diff = lambda a, b: a - b
    keys = ("delta_r", "rel_err")
    table = {a: {conv: {k: pi(per_pair(a, conv, k)) for k in keys} for conv in ("measured", "own")} for a in scored}
    joint_vs_random = {f"{rung}: joint − {r}": {conv: {k: pi(per_pair(f"{rung}_joint", conv, k), per_pair(f"{rung}_{r}", conv, k),
                                                             combine=diff) for k in keys} for conv in ("measured", "own")}
                       for rung in ("roundtrip", "linear", "pipeline") for r in RANDOM}
    random_vs_dec = {f"{rung}: {r} − dec": {conv: {k: pi(per_pair(f"{rung}_{r}", conv, k), per_pair(f"{rung}_dec", conv, k),
                                                     combine=diff) for k in keys} for conv in ("measured", "own")}
                     for rung in ("roundtrip", "pipeline") for r in RANDOM}
    vs_ref = {f"pipeline_{a}": {ref: {k: pi(per_pair(f"pipeline_{a}", "measured", k), per_pair(ref, "measured", k), combine=diff)
                                      for k in keys} for ref in ("cvae_hvg", "additive")} for a in ARMS}
    headroom = {a: pi(per_pair(f"roundtrip_{a}", "measured", "delta_r"), per_pair(f"pipeline_{a}", "measured", "delta_r"),
                      combine=diff) for a in ARMS}

    population = {}
    for a in ARMS:
        reps = [js(root / flow[a].format(s) / "reports/calibration_flow.json") for s in seeds]
        if any(r.get("normalize") != "data" for r in reps):
            _fail(f"{a}: population report not normalized like the data")
        population[a] = {k: np.mean([[r["per_pert"][q][k] for q in pairs] for r in reps], axis=0) for k in POP}
    pop_table = {a: {k: pi(population[a][k]) for k in POP} for a in ARMS}
    pop_joint_vs_random = {r: {k: pi(population["joint"][k], population[r][k], combine=diff) for k in POP} for r in RANDOM}

    rep_diag = {}
    rng = np.random.default_rng(0)
    rows = rng.choice(np.flatnonzero(cache.split_combo == SPLIT_TRAIN), 2000, replace=False)
    tok = cache.manifest.get("tokenization", {})
    part = make_gene_partition(cache.n_hvg, int(tok.get("n_tokens", 50)), int(tok.get("partition_seed", 0)))

    def encode(path):
        return encode_rows(load_jepa(path, "cpu"), cache.hvg_X, part, rows, "cpu", 256).numpy()

    z0 = encode(root / "norman_stage_a/checkpoints/encoder.pt")
    zj = {s: encode(enc_path("joint", s)) for s in seeds}
    for a in ARMS:
        rep_diag[a] = []
        for s, h in zip(seeds, reports[a]):
            pr = js(root / ceil[a].format(s) / "reports/stage_a_probe.json")
            z = zj[s] if a == "joint" else encode(enc_path(a, s))
            losses = [e["nb_loss"] for e in h["history"]]
            rep_diag[a].append({
                "probe_train_acc": pr["probe"]["train_acc"], "probe_test_acc": pr["probe"]["test_acc"],
                "effective_rank": pr["effective_rank"], "feature_std": pr["feature_std"],
                "cka_to_pretrained": linear_cka(z, z0), "cka_to_joint_same_seed": linear_cka(z, zj[s]),
                "final_nb_loss": h["final_nb_loss"],
                "nb_loss_last5_minus_prev5": float(np.mean(losses[-5:]) - np.mean(losses[-10:-5])) if len(losses) >= 10 else None,
                "grad_nb_last": h["history"][-1].get("grad_nb"),
            })

    # ---- report ---------------------------------------------------------------------------- #
    bar = "=" * 96
    print(bar + "\nPRIMARY: readout-ceiling Δr (roundtrip, measured control)\n" + bar)
    for a in readouts:
        print(f"  {a:14s} {table[f'roundtrip_{a}']['measured']['delta_r']['mean']:.4f}")
    print(f"  max-t critical value {fam['critical_value']:.3f}")
    for k, c in fam["contrasts"].items():
        print(f"  {k:22s} {c['diff']:+.4f}  simultaneous [{c['ci_simultaneous'][0]:+.4f}, {c['ci_simultaneous'][1]:+.4f}]  -> {c['verdict']}")
    print(f"  ROUND: {fam['round_verdict']}")
    print("\nSECONDARY: every readout (Δr / e_p; measured | own)")
    for a in scored:
        t = table[a]
        print(f"  {a:24s} {t['measured']['delta_r']['mean']:.3f} / {t['measured']['rel_err']['mean']:.3f}   |   "
              f"{t['own']['delta_r']['mean']:.3f} / {t['own']['rel_err']['mean']:.3f}")
    for title, block in (("joint − random arms", joint_vs_random), ("random arms − dec (frozen pretrained)", random_vs_dec)):
        print(f"\nSECONDARY: {title} (measured; Δr, e_p)")
        for k, v_ in block.items():
            m = v_["measured"]
            print(f"  {k:30s} Δr {m['delta_r']['mean']:+.3f} [{m['delta_r']['ci'][0]:+.3f}, {m['delta_r']['ci'][1]:+.3f}]"
                  f"   e_p {m['rel_err']['mean']:+.3f} [{m['rel_err']['ci'][0]:+.3f}, {m['rel_err']['ci'][1]:+.3f}]")
    print("\nSECONDARY: full pipeline against the corrected cVAE and additivity (measured)")
    for k, v_ in vs_ref.items():
        for ref, m in v_.items():
            print(f"  {k:22s} − {ref:9s} Δr {m['delta_r']['mean']:+.3f} [{m['delta_r']['ci'][0]:+.3f}, {m['delta_r']['ci'][1]:+.3f}]"
                  f"   e_p {m['rel_err']['mean']:+.3f} [{m['rel_err']['ci'][0]:+.3f}, {m['rel_err']['ci'][1]:+.3f}]")
    print("\nSECONDARY: transition headroom (ceiling − pipeline, Δr)")
    for a, v_ in headroom.items():
        print(f"  {a:14s} {v_['mean']:+.3f} [{v_['ci'][0]:+.3f}, {v_['ci'][1]:+.3f}]")
    print("\nSECONDARY: population quality and calibration of the pipelines (coverage target 0.80)")
    for a, v_ in pop_table.items():
        print(f"  {a:14s} coverage {v_['coverage']['mean']:.3f} (gap {v_['coverage']['mean'] - 0.8:+.3f})  spread {v_['spread_r']['mean']:.3f}"
              f"  W1 {v_['wasserstein']['mean']:.3f}  energy {v_['energy']['mean']:.3f}")
    print("\nSECONDARY: representation and training (seed means)")
    for a, rs in rep_diag.items():
        ks = sorted({k for r in rs for k in r if r[k] is not None})
        print(f"  {a:14s} " + "  ".join(f"{k} {np.mean([r[k] for r in rs if r.get(k) is not None]):.4g}" for k in ks))
    print("\nNOTE: intervals reflect the finite set of pairs; seeds are averaged first.")

    out = {"protocol": "round 13, joint training from a random start, frozen before training (see docstring)",
           "smoke": bool(args.smoke),
           "pairs": pairs, "n_boot": args.n_boot, "alpha": args.alpha, "bootstrap_seed": 0, "primary": fam,
           "secondary": {"table": table, "joint_vs_random": joint_vs_random, "random_vs_dec": random_vs_dec,
                         "vs_reference": vs_ref, "transition_headroom": headroom, "population": pop_table,
                         "population_joint_vs_random": pop_joint_vs_random, "representation": rep_diag}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
