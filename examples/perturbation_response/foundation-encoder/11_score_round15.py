#!/usr/bin/env python3
"""Score round 15: which property of Geneformer V2 carries its gain over our JEPA encoder?

This docstring is the pre-registered protocol. It was committed before any round-15 arm was
trained, and the code below implements it; a change to either after the first full run would be
reported as a deviation.

Question
--------
Round 14 found that Geneformer V2-104M's gene-mean embedding, frozen, reads held-out Norman cells
better than our frozen JEPA encoder: a readout ceiling of 0.908 against 0.863, +0.045 with a
simultaneous interval of [+0.009, +0.081]. Geneformer V1-10M did not (0.862). V2-104M differs from
our encoder in several ways at once, and round 15 changes one of them at a time to see which change
costs the gain:

- **pretraining**: V2-104M's architecture with random weights, never trained;
- **input breadth**: V2-104M reading only the 5,000 genes of our panel rather than the whole
  transcriptome. The library-size total stays each cell's full total, so every gene's normalized
  value is unchanged and only the set of genes that enter the ranking shrinks; 4,316 panel genes
  have V2 tokens;
- **input length**: V2-104M with every gene list cut after its first 2,048 genes, the limit V1
  reads. This also separates V1's shorter input from its smaller size;
- **model size**: V2-316M (18 layers, 1,152 wide) in place of V2-104M (12 layers, 768 wide).

Arms
----
Every embedding is Geneformer's gene mean at the second-to-last layer, for all 109,737 processed
cells, written as an embedding store (``10_embed_variants.py``), at Hugging Face revision
``1f7fbae4``. Every arm uses round 11's corrected decoder recipe (``co-adaptation/02_coadapt_decoder.py
--mode dec --library-size hvg``, 30 epochs, combo split) and is read by ``14_ceiling_analysis.py``,
seeds 0, 1 and 2.

==============  ================================================  =====  ================================
arm             store                                             width  runs
==============  ================================================  =====  ================================
v2_cell         V2-104M, round 14's store                         768    round 14's, re-read, not retrained
v2rand_cell     V2-104M architecture, random weights (seed 0)     768    new
v2panel_cell    V2-104M on the 5,000-gene panel                   768    new
v2cut_cell      V2-104M, lists cut at 2,048 genes                 768    new
v2big_cell      V2-316M                                           1,152  new
jepa            our frozen JEPA encoder's store                   256    round 14's, re-read (context)
<arm>_p         each V2 store above projected onto its 256        256    v2_cell_p re-read; four new
                leading principal components, fit on combo
                training cells only (``05_project_store.py``)
==============  ================================================  =====  ================================

Primary endpoint and decision rule
----------------------------------
**Readout-ceiling Δr**, as in round 14: real held-out cells looked up in the arm's store, decoded by
the arm's decoder, scored by the Pearson correlation of predicted and measured change over each
held-out pair's top 20 genes, under the measured control; the three seeds averaged per pair.

One max-t family of four contrasts, each written so that a positive value means the property helps:

- ``pretraining``: ``v2_cell − v2rand_cell``
- ``input breadth``: ``v2_cell − v2panel_cell``
- ``input length``: ``v2_cell − v2cut_cell``
- ``model size``: ``v2big_cell − v2_cell``

Joint paired bootstrap over the 20 pairs, 10,000 resamples, seed 0; simultaneous intervals from the
95th percentile of the maximum studentized deviation across the family
(``ssllab.eval.bootstrap.maxt_family``). At least 18 pairs must be finite in every arm.

Per contrast:

- **the property carries a measurable share of the gain** if the estimate is at least +0.02 and the
  simultaneous interval lies wholly above 0;
- **the property costs readout** if the simultaneous interval lies wholly below 0;
- **not shown to matter** otherwise; the interval then says how large a share can be ruled out.

Round-level reading: the properties that pass are named as carrying the gain. If none passes, "no
single property is shown to carry the V2 gain", which leaves open that several small shares add up.

Secondaries (percentile intervals on the same resamples, no verdicts)
----------------------------------------------------------------------
1. The four contrasts under the own-control convention, and on the size-aware error e_p under both
   conventions.
2. Every arm against ``jepa`` (ceiling, both conventions, Δr and e_p). In particular
   ``v2panel_cell − jepa``: both read the same genes, so it compares pretraining on many cells plus a
   rank encoding with our encoder at equal input breadth.
3. Visible genes only: the four contrasts rescored on the scored genes V2's vocabulary covers (the
   scored genes are all panel genes, so every V2 arm sees the same ones).
4. The linear readout (ridge λ = 1), descriptive.
5. The cost of projecting: each ``_p`` ceiling minus its full-width ceiling, with the variance share.
6. **Full pipelines on the projections.** The round-1 transport flow (``04_train_cond_flow.py
   --split combo --cond-type geneset --flow-base control --coupling independent --epochs 60``,
   hidden width 256, learning rate 1e-3, no clipping) on each ``_p`` store, read through the same
   seed's ``_p`` decoder: Δr and e_p under both conventions; each new ``_p`` pipeline against
   ``v2_cell_p`` and against ``jepa``; headroom (ceiling minus pipeline); population quality and
   calibration (``10_eval_calibration.py``, sampled counts normalized like the data at the median
   real total over the modeled genes: coverage against the nominal 0.80, spread correlation,
   per-gene W1, joint energy distance). A flow whose epoch-mean loss ever exceeds twice its first
   epoch's is reported as diverged, and its arm's pipeline as not trainable under this recipe.
7. The effective rank of every store on 2,000 training cells (seed 0).

Gates (any failure stops the scoring)
-------------------------------------
(a) Every ceiling's identity arm scores above 0.999, and its roundtrip mean recomputed from the saved
    profiles reproduces ``ceiling.json`` within 1e-6.
(b) Every new decoder records mode ``dec``, library size ``hvg``, the combo split, no cell limit and
    30 epochs, and reads its arm's store (content hash).
(c) Every pipeline reproduces its ``effect_size.json`` within 1e-6; every flow is 256 wide with
    hidden width 256, records learning rate 1e-3 (the trainer's default when not recorded) and no
    clipping, reads its arm's store and the same latent space as its decoder.
(d) Every store covers all 109,737 processed cells and was computed on this cache.
(e) The re-read round-14 arms reproduce round 14: the ``v2_cell`` and ``jepa`` ceilings computed here
    equal those in round 14's ``gf_compare.json`` within 1e-9.

Usage (on the pod or locally after the fetch, from the repo root):
    python examples/perturbation_response/foundation-encoder/11_score_round15.py \\
        --output-root output --out output/gf15_compare.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TRAIN, cache_fingerprint, load_cache  # noqa: E402
from ssllab.embeddings import load_store  # noqa: E402
from ssllab.eval.bootstrap import maxt_family, percentile_interval, resample_indices  # noqa: E402
from ssllab.eval.collapse import collapse_report  # noqa: E402

_spec = importlib.util.spec_from_file_location("gf07", _HERE / "07_score_comparison.py")
gf07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gf07)          # round 14's scorer: its Scorer, constants and experiment names

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEEDS, N_CELLS, POP, KEYS, CONVS = gf07.SEEDS, gf07.N_CELLS, gf07.POP, gf07.KEYS, gf07.CONVS
dec_exp, ceil_exp, flow_exp, _fail = gf07.dec_exp, gf07.ceil_exp, gf07.flow_exp, gf07._fail

# arm -> (store path under the output root, width)
ARMS: dict[str, tuple[str, int]] = {
    "v2_cell": ("geneformer_full/stores/V2-104M_cell", 768),
    "v2rand_cell": ("geneformer_r15/stores/v2rand_cell", 768),
    "v2panel_cell": ("geneformer_r15/stores/v2panel_cell", 768),
    "v2cut_cell": ("geneformer_r15/stores/v2cut_cell", 768),
    "v2big_cell": ("geneformer_r15/stores/v2big_cell", 1152),
    "jepa": ("embeddings/norman_jepa_stage_a", 256),
    "v2_cell_p": ("geneformer_full/stores_p256/V2-104M_cell", 256),
    "v2rand_cell_p": ("geneformer_r15/stores_p256/v2rand_cell", 256),
    "v2panel_cell_p": ("geneformer_r15/stores_p256/v2panel_cell", 256),
    "v2cut_cell_p": ("geneformer_r15/stores_p256/v2cut_cell", 256),
    "v2big_cell_p": ("geneformer_r15/stores_p256/v2big_cell", 256),
}
REREAD = ("v2_cell", "jepa", "v2_cell_p")                  # round 14's runs
PIPELINE_ARMS = ("jepa", "v2_cell_p", "v2rand_cell_p", "v2panel_cell_p", "v2cut_cell_p", "v2big_cell_p")
PRIMARY = {"pretraining": ("v2_cell", "v2rand_cell"), "input breadth": ("v2_cell", "v2panel_cell"),
           "input length": ("v2_cell", "v2cut_cell"), "model size": ("v2big_cell", "v2_cell")}
NEW_FULL = ("v2rand_cell", "v2panel_cell", "v2cut_cell", "v2big_cell")
FLOW_LR, FLOW_HIDDEN = 1e-3, 256


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default="output")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifact", default="norman2019")
    p.add_argument("--visible", default=None, help="default: <data-dir>/<artifact>/foundation/visible_hvg.json")
    p.add_argument("--round14", default=None, help="round 14's gf_compare.json (default: <output-root>/gf_compare.json)")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="smoke only; the protocol uses 0 1 2")
    p.add_argument("--smoke", action="store_true",
                   help="smoke only: gate (b) ignores epochs and the cell limit, gates (d) and (e) are skipped")
    p.add_argument("--out", default="output/gf15_compare.json")
    args = p.parse_args()
    seeds = tuple(args.seeds)
    root = Path(args.output_root)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    fingerprint = cache_fingerprint(cache)
    sc = gf07.round10.Scorer(cache)
    pairs = sc.pairs
    visible = json.loads(Path(args.visible or Path(args.data_dir) / args.artifact / "foundation/visible_hvg.json").read_text())

    def js(path: Path) -> dict:
        return json.loads(path.read_text())

    # ---- gate (d): stores ------------------------------------------------------------------- #
    stores = {}
    for arm, (rel, width) in ARMS.items():
        st = load_store(root / rel, expect_fingerprint=fingerprint)
        if st.width != width:
            _fail(f"{arm}: store is {st.width} wide, the protocol says {width}")
        if not args.smoke and (st.n != N_CELLS or not np.array_equal(st.row_idx, np.arange(N_CELLS))):
            _fail(f"{arm}: store covers {st.n} rows, not all {N_CELLS} processed cells")
        stores[arm] = st
    logger.info("gate (d) passed")

    # ---- gates (a), (b): decoders and ceilings ----------------------------------------------- #
    scored: dict[str, list[dict]] = {}
    profiles: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for arm in ARMS:
        for rung in ("roundtrip", "linear"):
            scored[f"{rung}_{arm}"] = []
        profiles[arm] = []
        for s in seeds:
            rep = js(root / dec_exp(arm, s) / "reports/coadapt.json")
            if rep.get("mode") != "dec" or rep.get("library_size") != "hvg" or rep.get("split") != "combo":
                _fail(f"{arm} s{s}: decoder report records mode {rep.get('mode')!r}, library size "
                      f"{rep.get('library_size')!r}, split {rep.get('split')!r}")
            if not args.smoke and (rep.get("limit") is not None or rep["config"]["epochs"] != 30):
                _fail(f"{arm} s{s}: decoder trained with limit {rep.get('limit')} / {rep['config']['epochs']} epochs")
            if (rep.get("latent_source") or {}).get("content_md5") != stores[arm].manifest["content_md5"]:
                _fail(f"{arm} s{s}: decoder did not read the arm's store")
            c = js(root / ceil_exp(arm, s) / "reports/ceiling.json")
            if c["arms"]["identity"]["mean_delta_r"] <= 0.999:
                _fail(f"identity arm, {arm} s{s}")
            if (c.get("latent_source") or {}).get("content_md5") != stores[arm].manifest["content_md5"]:
                _fail(f"{arm} s{s}: ceiling did not read the arm's store")
            z = np.load(root / ceil_exp(arm, s) / "reports/ceiling_profiles.npz", allow_pickle=True)
            for rung in ("roundtrip", "linear"):
                prof = {str(n): q for n, q in zip(z[f"{rung}__names"], z[f"{rung}__profiles"])}
                out = sc.score(prof, z[f"{rung}__control"])
                if rung == "roundtrip":
                    got = float(np.nanmean([out["measured"][k]["delta_r"] for k in pairs]))
                    if abs(got - c["arms"]["roundtrip"]["mean_delta_r"]) > 1e-6:
                        _fail(f"{arm} s{s}: roundtrip {got:.6f} does not reproduce {c['arms']['roundtrip']['mean_delta_r']:.6f}")
                    profiles[arm].append((prof, z[f"{rung}__control"]))
                scored[f"{rung}_{arm}"].append(out)
    logger.info("gates (a), (b) passed")

    def per_pair(key_arm: str, conv: str, key: str, table=None) -> np.ndarray:
        src = scored if table is None else table
        return np.array([np.mean([r[conv][k][key] for r in src[key_arm]]) for k in pairs])

    # ---- gate (e): the re-read arms reproduce round 14 ---------------------------------------- #
    r14_path = Path(args.round14) if args.round14 else root / "gf_compare.json"
    reproduce = {}
    if not args.smoke:
        r14 = js(r14_path)["primary"]
        for arm, ref in (("v2_cell", r14["arms"]["v2_cell"]), ("jepa", r14["jepa"])):
            here = float(np.mean(per_pair(f"roundtrip_{arm}", "measured", "delta_r")))
            reproduce[arm] = {"here": here, "round14": ref}
            if abs(here - ref) > 1e-9:
                _fail(f"re-read {arm}: ceiling {here:.10f} does not reproduce round 14's {ref:.10f}")
        logger.info("gate (e) passed")

    # ---- gate (c): pipelines on the projections ---------------------------------------------- #
    def learned(exp: Path, kind: str, report: str) -> dict:
        z = np.load(exp / f"reports/profiles_{kind}.npz", allow_pickle=True)
        out = sc.score({str(k): q for k, q in zip(z["pair_names"], z["pair_profiles"])}, z["control_profile"])
        ref = js(exp / "reports" / report)["per_pert"]
        worst = max(abs(out["measured"][k]["delta_r"] - ref[k]) for k in pairs)
        if worst > 1e-6:
            _fail(f"{exp.name}: Δr does not reproduce {report} (max |diff| {worst:.2g})")
        return out

    def training_record(exp: Path, s: int) -> dict:
        h = js(exp / "reports/stage_b_flow.json").get("history")
        if not h:
            return {"seed": s, "history": None}
        loss = np.array(h["cfm_loss"], float)
        ratio = float(np.max(loss) / loss[0])
        return {"seed": s, "final_loss": float(loss[-1]), "max_over_first": ratio, "diverged": ratio > 2.0}

    population, flow_training, not_trainable = {}, {}, []
    for arm in PIPELINE_ARMS:
        exps = []
        for s in seeds:
            exp = root / flow_exp(arm, s)
            fl = torch.load(exp / "checkpoints/cond_flow.pt", map_location="cpu", weights_only=False)
            dec = torch.load(exp / "checkpoints/count_decoder.pt", map_location="cpu", weights_only=False)
            if fl["dim"] != 256 or fl["hidden"] != FLOW_HIDDEN:
                _fail(f"{arm} s{s}: flow is {fl['dim']} wide with hidden {fl['hidden']}, protocol 256/{FLOW_HIDDEN}")
            if abs(fl.get("lr", gf07.LR_DEFAULT_BEFORE_RECORDING) - FLOW_LR) > 1e-12 or fl.get("grad_clip") is not None:
                _fail(f"{arm} s{s}: flow trained with lr {fl.get('lr')} and grad_clip {fl.get('grad_clip')}")
            if (fl.get("latent_source") or {}).get("space_key") != (dec.get("latent_source") or {}).get("space_key"):
                _fail(f"{arm} s{s}: flow and decoder read different latent spaces")
            if (fl.get("latent_source") or {}).get("content_md5") != stores[arm].manifest["content_md5"]:
                _fail(f"{arm} s{s}: flow did not read the arm's store")
            flow_training.setdefault(arm, []).append(training_record(exp, s))
            exps.append(exp)
        if any(r.get("diverged") for r in flow_training[arm]):
            not_trainable.append(arm)
            continue
        scored[f"pipeline_{arm}"] = [learned(exp, "flow", "effect_size.json") for exp in exps]
        pops = [js(exp / "reports/calibration_flow.json") for exp in exps]
        if any(pp.get("normalize") != "data" for pp in pops):
            _fail(f"{arm}: population report not normalized like the data")
        population[arm] = {k: np.mean([[r["per_pert"][q][k] for q in pairs] for r in pops], axis=0) for k in POP}
    logger.info("gate (c) passed")

    def masked(*v):
        ok = np.all([np.isfinite(x) for x in v], axis=0)
        return ok, [x[ok] for x in v], resample_indices(int(ok.sum()), args.n_boot, 0)

    def pi(*vecs, combine=lambda *a: a[0]):
        ok_, sub_, ix = masked(*vecs)
        r = percentile_interval(combine(*sub_), ix, args.alpha)
        r["n_pairs"] = int(ok_.sum())
        return r

    diff = lambda a, b: a - b  # noqa: E731

    # ---- primary ------------------------------------------------------------------------------ #
    vec = {a: per_pair(f"roundtrip_{a}", "measured", "delta_r") for a in ARMS}
    ok, _, idx = masked(*[vec[a] for a in {x for pr in PRIMARY.values() for x in pr}])
    if ok.sum() < 18:
        _fail(f"only {int(ok.sum())} evaluable pairs")
    fam = maxt_family({name: vec[a][ok] - vec[b][ok] for name, (a, b) in PRIMARY.items()}, idx, args.alpha)
    for c in fam["contrasts"].values():
        lo, hi = c["ci_simultaneous"]
        c["verdict"] = ("carries a measurable share of the gain" if c["diff"] >= 0.02 and lo > 0
                        else "costs readout" if hi < 0 else "not shown to matter")
    carriers = [k for k, c in fam["contrasts"].items() if c["verdict"].startswith("carries")]
    fam["round_verdict"] = ("carried by: " + ", ".join(carriers)) if carriers else \
        "no single property is shown to carry the V2 gain"
    means = {a: float(np.mean(vec[a][ok])) for a in ARMS}

    # ---- secondaries -------------------------------------------------------------------------- #
    table = {a: {conv: {k: pi(per_pair(a, conv, k)) for k in KEYS} for conv in CONVS} for a in scored}
    contrasts_all = {f"{rung}: {name}": {conv: {k: pi(per_pair(f"{rung}_{a}", conv, k), per_pair(f"{rung}_{b}", conv, k),
                                                       combine=diff) for k in KEYS} for conv in CONVS}
                     for rung in ("roundtrip", "linear") for name, (a, b) in PRIMARY.items()}
    vs_jepa = {f"roundtrip: {a} − jepa": {conv: {k: pi(per_pair(f"roundtrip_{a}", conv, k),
                                                       per_pair("roundtrip_jepa", conv, k), combine=diff)
                                                 for k in KEYS} for conv in CONVS} for a in ARMS if a != "jepa"}
    vis = set(visible["series"]["V2"]["visible_hvg_idx"])
    sv = copy.copy(sc)
    sv.top = {k: [i for i in sc.top[k] if i in vis] for k in pairs}
    vt = {a: [sv.score(prof, ctrl) for prof, ctrl in profiles[a]] for a in ARMS}
    visible_contrasts = {name: {conv: {k: pi(per_pair(a, conv, k, vt), per_pair(b, conv, k, vt), combine=diff)
                                       for k in KEYS} for conv in CONVS} for name, (a, b) in PRIMARY.items()}
    projection_cost = {f"{a}_p − {a}": {k: pi(per_pair(f"roundtrip_{a}_p", "measured", k),
                                              per_pair(f"roundtrip_{a}", "measured", k), combine=diff) for k in KEYS}
                       for a in ("v2_cell",) + NEW_FULL}
    projection_variance = {a: stores[a].manifest.get("explained_variance_share") for a in ARMS if a.endswith("_p")}
    piped = [a for a in PIPELINE_ARMS if a not in not_trainable]
    pipe_vs = {f"pipeline: {a} − {ref}": {conv: {k: pi(per_pair(f"pipeline_{a}", conv, k), per_pair(f"pipeline_{ref}", conv, k),
                                                       combine=diff) for k in KEYS} for conv in CONVS}
               for a in piped for ref in ("v2_cell_p", "jepa") if a != ref and ref in piped}
    headroom = {a: pi(per_pair(f"roundtrip_{a}", "measured", "delta_r"), per_pair(f"pipeline_{a}", "measured", "delta_r"),
                      combine=diff) for a in piped}
    pop_table = {a: {k: pi(population[a][k]) for k in POP} for a in piped}
    pop_vs = {f"{a} − {ref}": {k: pi(population[a][k], population[ref][k], combine=diff) for k in POP}
              for a in piped for ref in ("v2_cell_p", "jepa") if a != ref and ref in piped}
    rng = np.random.default_rng(0)
    rank_rows = np.sort(rng.choice(np.flatnonzero(cache.split_combo == SPLIT_TRAIN), 2000, replace=False))
    eff_rank = {a: collapse_report(torch.from_numpy(st.lookup(rank_rows[np.isin(rank_rows, st.row_idx)])))
                for a, st in stores.items()}

    # ---- report ------------------------------------------------------------------------------- #
    bar = "=" * 96
    print(bar + "\nPRIMARY: readout-ceiling Δr (roundtrip, measured control), seeds averaged per pair\n" + bar)
    for a in ARMS:
        print(f"  {a:16s} {means[a]:.4f}")
    for name, c in fam["contrasts"].items():
        lo, hi = c["ci_simultaneous"]
        print(f"  {name:14s} {PRIMARY[name][0]} − {PRIMARY[name][1]}: {c['diff']:+.4f}  "
              f"simultaneous [{lo:+.4f}, {hi:+.4f}]  {c['verdict']}")
    print(f"  max-t critical value {fam['critical_value']:.3f}\n  ROUND: {fam['round_verdict']}")
    print(bar + "\nSECONDARY\n" + bar)
    for a in piped:
        pl = table[f"pipeline_{a}"]["measured"]
        print(f"  pipeline {a:16s} Δr {pl['delta_r']['mean']:.3f} e_p {pl['rel_err']['mean']:.3f} "
              f"coverage {pop_table[a]['coverage']['mean']:.3f} energy {pop_table[a]['energy']['mean']:.3f}")
    print(f"  pipelines not trainable: {', '.join(not_trainable) if not_trainable else 'none'}")

    result = {
        "protocol": "foundation-encoder/11_score_round15.py docstring", "seeds": list(seeds), "smoke": args.smoke,
        "pairs": pairs, "n_boot": args.n_boot, "alpha": args.alpha,
        "stores": {a: {k: st.manifest.get(k) for k in ("source", "model", "mode", "variant", "base_model", "width", "n",
                                                       "content_md5", "space_key", "explained_variance_share")}
                   for a, st in stores.items()},
        "reread_round14": reproduce,
        "primary": {"endpoint": "readout-ceiling Δr, roundtrip, measured control", "arms": means,
                    "contrast_arms": PRIMARY, **fam},
        "secondary": {
            "table": table, "contrasts": contrasts_all, "vs_jepa": vs_jepa, "visible_genes": visible_contrasts,
            "projection_cost": projection_cost, "projection_variance_share": projection_variance,
            "pipeline_contrasts": pipe_vs, "headroom": headroom, "population": pop_table, "population_contrasts": pop_vs,
            "flow_training": flow_training, "pipelines_not_trainable": not_trainable, "effective_rank": eff_rank,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
