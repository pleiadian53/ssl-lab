#!/usr/bin/env python3
"""Score the comparison: Geneformer cell embeddings against our JEPA encoder, frozen.

This docstring is the pre-registered protocol. It was committed before any arm was trained, and the
code below implements it; a change to either after the first full run would be reported as a
deviation.

Question
--------
Round 13 found that our own JEPA pretraining contributes nothing measurable once the encoder is
trained jointly with the count decoder. But that encoder was pretrained on the ~96,000 Norman
training cells only. Geneformer was pretrained on about 30 million (V1) or 104 million (V2) cells
from many tissues and experiments. With the encoder **frozen**, does a Geneformer cell embedding
give the corrected count decoder and the conditional flow a better starting point than our JEPA
encoder's, on the 20 held-out two-gene combinations? Published benchmarks (Kedzierska et al.,
Genome Biology 2025; Ahlmann-Eltze et al., Nature Methods 2025) found foundation-model embeddings
did not beat simple baselines on perturbation tasks, so a tie is the expected outcome.

Arms
----
Every arm reads its cell vectors from an embedding store (``ssllab.embeddings``) through the same
code path, including our own encoder, whose vectors are exported to a store first. All arms use
the round-11 corrected decoder recipe (``co-adaptation/02_coadapt_decoder.py --mode dec
--library-size hvg``, 30 epochs, combo split) and the round-1 transport flow
(``04_train_cond_flow.py --split combo --cond-type geneset --flow-base control --coupling
independent --epochs 60``), seeds 0, 1 and 2. Each seed's flow is read out through the same seed's
decoder.

=============  ================================================  =====  ===========
arm            store                                             width  flow hidden
=============  ================================================  =====  ===========
jepa           ``norman_stage_a`` exported (the within-round      256    256
               frozen-JEPA control)
v1_cell        Geneformer V1-10M, mean over genes                 256    256
v2_cls         Geneformer V2-104M, ``<cls>`` token                768    768
v2_cell        Geneformer V2-104M, mean over genes                768    768
v2cl_cls       Geneformer V2-104M_CLcancer, ``<cls>`` token       768    768
v2cl_cell      Geneformer V2-104M_CLcancer, mean over genes       768    768
v2_cls_p, v2_cell_p, v2cl_cls_p, v2cl_cell_p
               the four V2 stores projected onto their 256        256    256
               leading principal components, fit on combo
               training cells only
=============  ================================================  =====  ===========

Embeddings come from Geneformer's second-to-last layer (its default), pinned at Hugging Face
revision ``1f7fbae4``. The flow's hidden width equals the embedding's width (decision D19): with a
256-wide hidden layer, a 768-wide flow could only move cells inside one fixed 256-dimensional
subspace, a limit the 256-wide arms do not have. The projected arms are the equal-width,
equal-capacity comparison.

Primary endpoint and decision rule
----------------------------------
**Readout-ceiling Δr**: the real held-out cells of each pair are encoded (looked up), decoded by the
arm's decoder, and scored as in every earlier round (``14_ceiling_analysis.py``, ``roundtrip``
arm, the measured-control convention, Pearson correlation of the predicted and measured change over
each pair's top 20 differentially expressed genes). Per pair, the three seeds are averaged.

One max-t family of five contrasts, each full-width Geneformer arm minus the ``jepa`` arm:
``v1_cell − jepa``, ``v2_cls − jepa``, ``v2_cell − jepa``, ``v2cl_cls − jepa``, ``v2cl_cell − jepa``.
Joint paired bootstrap over the 20 pairs, 10,000 resamples, seed 0; simultaneous intervals from the
95th percentile of the maximum studentized deviation across the family
(``ssllab.eval.bootstrap.maxt_family``). At least 18 pairs must be finite in every arm.

Per contrast:

- **passes** if the estimate is at least +0.02 and the simultaneous interval lies wholly above 0;
- **worse** if the simultaneous interval lies wholly below 0;
- **does not pass** otherwise.

A pass establishes a positive gain under the rule, not a gain above 0.02.

Round-level reading: if any contrast passes, "a Geneformer embedding reads held-out cells better than
the JEPA encoder" (naming which). If none passes and at least one is worse, "the JEPA encoder reads
held-out cells better than <the worse arms>". Otherwise "no advantage of a Geneformer embedding is
established".

**Trigger for a joint-training arm** (decision D17): if the best full-width Geneformer arm's
ceiling estimate is at least the ``jepa`` arm's minus 0.02, the frozen arm "comes close", and a
separate protocol trains that Geneformer model jointly with the decoder against a random-start twin
of the same architecture, as round 13 did. The trigger is reported, not acted on here.

Secondaries (percentile intervals on the same resamples, no verdicts)
----------------------------------------------------------------------
1. The ceiling under the own-control convention, and the size-aware error e_p (round 8) under both
   conventions, for every arm and every Geneformer − JEPA contrast.
2. **Visible genes only.** Each contrast is rescored on only the scored genes the Geneformer arm's
   vocabulary covers (V1 or V2, from ``06_visible_genes.py``), *both* arms on the same 17 to 20
   genes per pair. A gap that persists is about the representation; one that closes is about the
   vocabulary.
3. **The cost of projecting**: each ``_p`` ceiling minus its full-width ceiling, with the share of
   the training cells' variance the 256 components keep.
4. **The full pipeline** (flow + decoder): Δr and e_p under both conventions for every arm; each
   Geneformer arm minus ``jepa``, at full width and projected; every arm against the stored
   corrected cVAE (round 11, ``norman_r11_vae_hvg``) and the additive reference.
5. **Headroom**: ceiling minus pipeline, per arm.
6. **Population quality and calibration** (``10_eval_calibration.py``, sampled counts normalized
   like the data, at the default depth, the median total over the modeled genes): interval
   coverage against the nominal 0.80, spread correlation, per-gene W1 (marginals only), joint energy
   distance, per arm and against ``jepa``.
7. **The ranking hypothesis**, descriptive: a rank encoding keeps the direction of a change better
   than its size, so Geneformer − JEPA may come out relatively better on Δr than on e_p. Both
   contrasts are reported side by side for every arm.
8. The ``linear`` readout (ridge λ = 1, fit on training cells) for every arm, descriptive only: λ is
   not scale-invariant across embeddings of different width and scale.
9. The effective rank of every store on 2,000 training cells (seed 0).

Gates (any failure stops the scoring)
-------------------------------------
(a) Every ceiling's ``identity`` arm scores above 0.999, and the ``roundtrip`` mean recomputed here
    from the saved profiles reproduces ``ceiling.json`` within 1e-6.
(b) Every decoder's report records mode ``dec``, library size ``hvg``, the combo split, no cell limit
    and 30 epochs, and reads the arm's own store (its content hash matches the store's manifest).
(c) Every pipeline's saved profiles reproduce its ``effect_size.json`` within 1e-6; every flow
    records the same latent space as its decoder and the hidden width the table above gives.
(d) Every store covers all 109,737 processed cells and was computed on this cache.
(e) The interface gate written by the embedding job holds: reading the JEPA store reproduces the
    published round-11 ceilings (0.86695, 0.86338, 0.86502) within 1e-4.
(f) The ``jepa`` arm's own ceiling is reported against 0.865; a gap above 0.01 is flagged in the
    report (a sanity check, not a stop).

Amendment 1 (2026-11-15): gradient clipping for the 768-wide flows
------------------------------------------------------------------
**What happened.** In the first full run, all twelve 768-wide flows (``v2_cls``, ``v2_cell``,
``v2cl_cls``, ``v2cl_cell``, three seeds each, hidden width 768, learning rate 1e-3, no gradient
clipping) became unstable during training: the epoch-mean flow-matching loss, about 1.5 after the
first epoch, spiked to between 10^5 and 4 x 10^9 before falling back, and three of these flows then
generated non-finite cells, so their population reports could not be computed. All eighteen
256-wide flows, including the four projected V2 arms, trained smoothly. A flow that diverged is a
failed fit, not a measurement, so the full-width pipeline numbers of that run are not reported.

**What changes.** The twelve 768-wide flows are retrained with the gradient's total norm clipped at
1.0 before each optimizer step (``04_train_cond_flow.py --grad-clip 1.0``); learning rate, hidden
width, epochs, seeds and everything else stay as above. In a short test run on our encoder's store,
a healthy 256-wide flow's gradient norm was about 0.1 to 0.2, so the threshold should leave ordinary
steps alone and cap only spikes. The 256-wide
flows are kept as trained, without clipping. Their decoders and ceilings, and every 768-wide arm's
decoder and ceiling, are kept unchanged: none of them involves a flow.

**What it cannot affect.** The primary endpoint and secondaries 1, 2, 3, 8 and 9 read only decoders
and ceilings, which the first run completed and gates (a), (b), (d) and (e) passed. Their values
were computed from that run, with this script's functions, before this amendment was written; the
amendment concerns the flows only.

**What counts as still unstable.** Each flow now reports its per-epoch loss and gradient norms. A
768-wide flow whose epoch-mean loss ever exceeds twice its first epoch's is reported as diverged,
and its arm's pipeline results are then reported as not trainable under this recipe rather than as
a measurement; the full-width pipeline comparison would then rest on the projected arms.

Gate (c) gains one condition: every 768-wide flow records ``grad_clip`` 1.0 and every 256-wide flow
records none. Secondary 10, descriptive: per flow, the final loss, the largest epoch-mean loss over
the first, the largest gradient norm and the share of clipped steps.

**Outcome.** Clipping did not stabilize them: all twelve exceeded the criterion (largest epoch loss
4.6 x 10^3 to 8.3 x 10^7 times the first; gradient norm before clipping up to 10^8 to 4 x 10^12), so
under this amendment the four full-width pipelines were reported as not trainable.

Amendment 2 (2026-11-16): a lower learning rate for the 768-wide flows
---------------------------------------------------------------------
**Diagnosis.** Two checks were run after amendment 1's outcome, before this amendment was written.
``09_coordinate_stats.py`` found that the flow's per-coordinate standardization makes no coordinate
of any V2 store extreme (the largest standardized value is about 10 in every store, as in the
projected stores that train smoothly), so the input's scale is not the cause.
``run_flow_stability_probe_pod.sh`` then trained five flows on seed 0 for 60 epochs, reading only
their loss histories:

=================  =========================  ======  =======  =================================
probe              store                      hidden  lr       outcome
=================  =========================  ======  =======  =================================
lr3e-4             V2-104M gene mean (768)    768     3e-4     stable, final loss 0.868
lr1e-4             V2-104M gene mean (768)    768     1e-4     stable, final loss 0.881
hid256             V2-104M gene mean (768)    256     1e-3     stable, final loss 0.958
jepa_hid768        our JEPA encoder (256)     768     1e-3     stable
p256_hid768        V2 projected to 256        768     1e-3     diverged from epoch 13
=================  =========================  ======  =======  =================================

A 768-wide hidden layer at learning rate 1e-3 diverges on Geneformer's embeddings, whatever their
width, and does not on our encoder's; at 3e-4 it trains stably.

**What changes.** The twelve 768-wide flows are retrained at learning rate 3e-4
(``04_train_cond_flow.py --lr 3e-4``) without gradient clipping; hidden width, epochs, seeds and
everything else stay as above. The 256-wide flows are kept as trained, at 1e-3. Amendment 1's
divergence rule still applies. Like amendment 1, this one cannot affect the primary endpoint or any
ceiling-based secondary.

Gate (c) now checks each flow's recorded learning rate and clipping: 3e-4 and none for the 768-wide
flows, 1e-3 and none for the 256-wide ones. Flows trained before the learning rate was recorded in
the checkpoint are read at the trainer's default of the time, 1e-3.

Usage (on the pod or locally after the fetch, from the repo root):
    python examples/perturbation_response/foundation-encoder/07_score_comparison.py \\
        --output-root output --out output/gf_compare.json
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

_EP = Path(__file__).resolve().parents[1]
_SRC = _EP.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TRAIN, cache_fingerprint, load_cache
from ssllab.embeddings import load_store
from ssllab.eval.bootstrap import maxt_family, percentile_interval, resample_indices
from ssllab.eval.collapse import collapse_report

_spec = importlib.util.spec_from_file_location("round10", _EP / "23_score_round10.py")
round10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(round10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
N_CELLS = 109_737
GATE2_REF = {0: 0.86695, 1: 0.86338, 2: 0.86502}
JEPA_CEILING_REF = 0.865
# amendment 2: (learning rate, gradient clip) each flow must record, by its hidden width
FLOW_RECIPE = {768: (3e-4, None), 256: (1e-3, None)}
LR_DEFAULT_BEFORE_RECORDING = 1e-3  # 04's default when flows did not yet store their learning rate
POP = ("coverage", "spread_r", "wasserstein", "energy")
KEYS = ("delta_r", "rel_err")
CONVS = ("measured", "own")

# arm -> (store path under the output root, width, flow hidden, vocabulary series or None)
ARMS: dict[str, tuple[str, int, int, str | None]] = {
    "jepa": ("embeddings/norman_jepa_stage_a", 256, 256, None),
    "v1_cell": ("geneformer_full/stores/V1-10M_cell", 256, 256, "V1"),
    "v2_cls": ("geneformer_full/stores/V2-104M_cls", 768, 768, "V2"),
    "v2_cell": ("geneformer_full/stores/V2-104M_cell", 768, 768, "V2"),
    "v2cl_cls": ("geneformer_full/stores/V2-104M_CLcancer_cls", 768, 768, "V2"),
    "v2cl_cell": ("geneformer_full/stores/V2-104M_CLcancer_cell", 768, 768, "V2"),
    "v2_cls_p": ("geneformer_full/stores_p256/V2-104M_cls", 256, 256, "V2"),
    "v2_cell_p": ("geneformer_full/stores_p256/V2-104M_cell", 256, 256, "V2"),
    "v2cl_cls_p": ("geneformer_full/stores_p256/V2-104M_CLcancer_cls", 256, 256, "V2"),
    "v2cl_cell_p": ("geneformer_full/stores_p256/V2-104M_CLcancer_cell", 256, 256, "V2"),
}
PRIMARY = ("v1_cell", "v2_cls", "v2_cell", "v2cl_cls", "v2cl_cell")
PROJECTED = {"v2_cls_p": "v2_cls", "v2_cell_p": "v2_cell", "v2cl_cls_p": "v2cl_cls", "v2cl_cell_p": "v2cl_cell"}


def dec_exp(arm: str, s: int) -> str:
    return f"norman_gf_dec_{arm}_s{s}"


def ceil_exp(arm: str, s: int) -> str:
    return f"norman_gf_ceiling_{arm}_s{s}"


def flow_exp(arm: str, s: int) -> str:
    return f"norman_gf_flow_{arm}_s{s}"


def _fail(msg: str) -> None:
    raise SystemExit(f"GATE FAILED: {msg}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default="output")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifact", default="norman2019")
    p.add_argument("--visible", default=None, help="default: <data-dir>/<artifact>/foundation/visible_hvg.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="smoke only; the protocol uses 0 1 2")
    p.add_argument("--smoke", action="store_true",
                   help="smoke only: gate (b) ignores epochs and the cell limit, gate (d) store coverage, "
                        "and gate (e)'s reference values, which a smoke run cannot reproduce")
    p.add_argument("--out", default="output/gf_compare.json")
    args = p.parse_args()
    seeds = tuple(args.seeds)
    root = Path(args.output_root)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    fingerprint = cache_fingerprint(cache)
    sc = round10.Scorer(cache)
    pairs = sc.pairs
    visible = json.loads(Path(args.visible or Path(args.data_dir) / args.artifact / "foundation/visible_hvg.json").read_text())

    def js(path: Path) -> dict:
        return json.loads(path.read_text())

    # ---- gates (d), (e): the stores and the interface ---------------------------------------- #
    stores = {}
    for arm, (rel, width, _, _) in ARMS.items():
        st = load_store(root / rel, expect_fingerprint=fingerprint)
        if st.width != width:
            _fail(f"{arm}: store is {st.width} wide, the protocol says {width}")
        if not args.smoke and (st.n != N_CELLS or not np.array_equal(st.row_idx, np.arange(N_CELLS))):
            _fail(f"{arm}: store covers {st.n} rows, not all {N_CELLS} processed cells")
        stores[arm] = st
    gate2 = js(root / "norman_gf_gate2/gate2.json")
    for s in seeds:
        got = gate2["roundtrip"][str(s)]
        if not args.smoke and abs(got - GATE2_REF[s]) > 1e-4:
            _fail(f"interface gate: JEPA store reads seed {s} at {got:.5f}, published {GATE2_REF[s]:.5f}")
    logger.info("gates (d), (e) passed")

    # ---- gates (a), (b): decoders and ceilings ------------------------------------------------ #
    scored: dict[str, list[dict]] = {}
    profiles: dict[str, list[tuple[dict, np.ndarray]]] = {}
    for arm in ARMS:
        for rung in ("roundtrip", "linear"):
            scored[f"{rung}_{arm}"] = []
        profiles[arm] = []
        for s in seeds:
            rep = js(root / dec_exp(arm, s) / "reports/coadapt.json")
            src = rep.get("latent_source") or {}
            if rep.get("mode") != "dec" or rep.get("library_size") != "hvg" or rep.get("split") != "combo":
                _fail(f"{arm} s{s}: decoder report records mode {rep.get('mode')!r}, library size "
                      f"{rep.get('library_size')!r}, split {rep.get('split')!r}")
            if not args.smoke and (rep.get("limit") is not None or rep["config"]["epochs"] != 30):
                _fail(f"{arm} s{s}: decoder trained with limit {rep.get('limit')} / {rep['config']['epochs']} epochs")
            if src.get("content_md5") != stores[arm].manifest["content_md5"]:
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

    # ---- gate (c): pipelines ------------------------------------------------------------------ #
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
        if not h:  # flows trained before amendment 1 did not record their history
            return {"seed": s, "history": None}
        loss = np.array(h["cfm_loss"], float)
        ratio = float(np.max(loss) / loss[0])
        return {"seed": s, "final_loss": float(loss[-1]), "max_over_first": ratio, "diverged": ratio > 2.0,
                "grad_norm_max": float(np.max(h["grad_norm_max"])),
                "clipped_fraction": float(np.mean(h["clipped_fraction"]))}

    population, flow_training, not_trainable = {}, {}, []
    for arm, (_, width, hidden, _) in ARMS.items():
        exps = []
        for s in seeds:
            exp = root / flow_exp(arm, s)
            fl = torch.load(exp / "checkpoints/cond_flow.pt", map_location="cpu", weights_only=False)
            dec = torch.load(exp / "checkpoints/count_decoder.pt", map_location="cpu", weights_only=False)
            if fl["hidden"] != hidden or fl["dim"] != width:
                _fail(f"{arm} s{s}: flow is {fl['dim']} wide with hidden {fl['hidden']}, protocol {width}/{hidden}")
            if (fl.get("latent_source") or {}).get("space_key") != (dec.get("latent_source") or {}).get("space_key"):
                _fail(f"{arm} s{s}: flow and decoder read different latent spaces")
            if (fl.get("latent_source") or {}).get("content_md5") != stores[arm].manifest["content_md5"]:
                _fail(f"{arm} s{s}: flow did not read the arm's store")
            want_lr, want_clip = FLOW_RECIPE[hidden]
            got_lr = fl.get("lr", LR_DEFAULT_BEFORE_RECORDING)
            if fl.get("grad_clip") != want_clip or abs(got_lr - want_lr) > 1e-12:
                _fail(f"{arm} s{s}: flow trained with lr {got_lr} and grad_clip {fl.get('grad_clip')}, "
                      f"amendment 2 says {want_lr} and {want_clip}")
            flow_training.setdefault(arm, []).append(training_record(exp, s))
            exps.append(exp)
        # Amendment 1: an arm with a diverged flow has no pipeline measurement; it is reported as not trainable.
        if any(r.get("diverged") for r in flow_training[arm]):
            not_trainable.append(arm)
            continue
        scored[f"pipeline_{arm}"] = []
        pops = []
        for exp in exps:
            scored[f"pipeline_{arm}"].append(learned(exp, "flow", "effect_size.json"))
            pop = js(exp / "reports/calibration_flow.json")
            if pop.get("normalize") != "data":
                _fail(f"{exp.name}: population report not normalized like the data")
            pops.append(pop)
        population[arm] = {k: np.mean([[r["per_pert"][q][k] for q in pairs] for r in pops], axis=0) for k in POP}
    scored["cvae_hvg"] = [learned(root / f"norman_r11_vae_hvg_s{s}", "vae", "effect_size_cvae.json") for s in seeds]
    scored["additive"] = [learned(root / "norman_additive", "additive", "effect_size_additive.json")]
    logger.info("gate (c) passed")

    # ---- helpers ------------------------------------------------------------------------------ #
    def per_pair(key_arm: str, conv: str, key: str, table=None) -> np.ndarray:
        src = scored if table is None else table
        return np.array([np.mean([r[conv][k][key] for r in src[key_arm]]) for k in pairs])

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
    base = per_pair("roundtrip_jepa", "measured", "delta_r")
    vecs = {a: per_pair(f"roundtrip_{a}", "measured", "delta_r") for a in PRIMARY}
    ok, sub, idx = masked(base, *vecs.values())
    if ok.sum() < 18:
        _fail(f"only {int(ok.sum())} evaluable pairs")
    fam = maxt_family({f"{a} − jepa": sub[i + 1] - sub[0] for i, a in enumerate(PRIMARY)}, idx, args.alpha)
    for c in fam["contrasts"].values():
        lo, hi = c["ci_simultaneous"]
        c["verdict"] = "passes" if c["diff"] >= 0.02 and lo > 0 else "worse" if hi < 0 else "does not pass"
    passes = [k for k, c in fam["contrasts"].items() if c["verdict"] == "passes"]
    worse = [k for k, c in fam["contrasts"].items() if c["verdict"] == "worse"]
    if passes:
        fam["round_verdict"] = "a Geneformer embedding reads held-out cells better than the JEPA encoder: " + ", ".join(passes)
    elif worse:
        fam["round_verdict"] = "the JEPA encoder reads held-out cells better than: " + ", ".join(worse)
    else:
        fam["round_verdict"] = "no advantage of a Geneformer embedding is established"
    means = {a: float(np.mean(v[ok])) for a, v in vecs.items()}
    jepa_mean = float(np.mean(base[ok]))
    best = max(means, key=means.get)
    trigger = {"best_arm": best, "best_ceiling": means[best], "jepa_ceiling": jepa_mean,
               "threshold": jepa_mean - 0.02, "fires": means[best] >= jepa_mean - 0.02}
    sanity = {"jepa_ceiling": jepa_mean, "reference": JEPA_CEILING_REF,
              "flag": abs(jepa_mean - JEPA_CEILING_REF) > 0.01}

    # ---- secondaries -------------------------------------------------------------------------- #
    table = {a: {conv: {k: pi(per_pair(a, conv, k)) for k in KEYS} for conv in CONVS} for a in scored}
    gf_arms = [a for a in ARMS if a != "jepa"]
    piped = [a for a in ARMS if a not in not_trainable]          # arms with a pipeline measurement
    if "jepa" not in piped:
        _fail("the jepa arm's flow diverged; no pipeline contrast is possible")
    gf_piped = [a for a in gf_arms if a in piped]
    rung_arms = {"roundtrip": gf_arms, "pipeline": gf_piped, "linear": gf_arms}
    vs_jepa = {f"{rung}: {a} − jepa": {conv: {k: pi(per_pair(f"{rung}_{a}", conv, k), per_pair(f"{rung}_jepa", conv, k),
                                                     combine=diff) for k in KEYS} for conv in CONVS}
               for rung in ("roundtrip", "pipeline", "linear") for a in rung_arms[rung]}

    # Visible genes only: both arms rescored on the genes the Geneformer arm's vocabulary covers.
    visible_tables = {}
    for series in ("V1", "V2"):
        vis = set(visible["series"][series]["visible_hvg_idx"])
        sv = copy.copy(sc)
        sv.top = {k: [i for i in sc.top[k] if i in vis] for k in pairs}
        visible_tables[series] = {a: [sv.score(prof, ctrl) for prof, ctrl in profiles[a]]
                                  for a in ARMS if a == "jepa" or ARMS[a][3] == series}
    visible_vs_jepa = {}
    for a in gf_arms:
        t = visible_tables[ARMS[a][3]]
        visible_vs_jepa[f"roundtrip: {a} − jepa"] = {
            conv: {k: pi(per_pair(a, conv, k, t), per_pair("jepa", conv, k, t), combine=diff) for k in KEYS} for conv in CONVS}
    visible_table = {f"{series}: {a}": {conv: {k: pi(per_pair(a, conv, k, t)) for k in KEYS} for conv in CONVS}
                     for series, t in visible_tables.items() for a in t}

    projection_cost = {f"{p_} − {f_}": {k: pi(per_pair(f"roundtrip_{p_}", "measured", k), per_pair(f"roundtrip_{f_}", "measured", k),
                                            combine=diff) for k in KEYS} for p_, f_ in PROJECTED.items()}
    projection_variance = {p_: stores[p_].manifest.get("explained_variance_share") for p_ in PROJECTED}
    vs_ref = {f"pipeline_{a}": {ref: {k: pi(per_pair(f"pipeline_{a}", "measured", k), per_pair(ref, "measured", k), combine=diff)
                                      for k in KEYS} for ref in ("cvae_hvg", "additive")} for a in piped}
    headroom = {a: pi(per_pair(f"roundtrip_{a}", "measured", "delta_r"), per_pair(f"pipeline_{a}", "measured", "delta_r"),
                      combine=diff) for a in piped}
    pop_table = {a: {k: pi(population[a][k]) for k in POP} for a in piped}
    pop_vs_jepa = {a: {k: pi(population[a][k], population["jepa"][k], combine=diff) for k in POP} for a in gf_piped}
    # Ranking hypothesis: Δr gain and e_p gain side by side (e_p is lower-better, so jepa − arm).
    ranking = {f"{rung}: {a}": {"delta_r_gain": vs_jepa[f"{rung}: {a} − jepa"]["measured"]["delta_r"],
                                "rel_err_gain": pi(per_pair(f"{rung}_jepa", "measured", "rel_err"),
                                                   per_pair(f"{rung}_{a}", "measured", "rel_err"), combine=diff)}
               for rung in ("roundtrip", "pipeline") for a in rung_arms[rung]}
    rng = np.random.default_rng(0)
    rank_rows = np.sort(rng.choice(np.flatnonzero(cache.split_combo == SPLIT_TRAIN), 2000, replace=False))
    eff_rank = {}
    for a, st in stores.items():
        rows = rank_rows[np.isin(rank_rows, st.row_idx)]
        eff_rank[a] = collapse_report(torch.from_numpy(st.lookup(rows)))

    diverged = [f"{a} s{r['seed']}" for a, recs in flow_training.items() for r in recs if r.get("diverged")]

    # ---- report ------------------------------------------------------------------------------- #
    def short(x):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} if isinstance(x, dict) else x

    bar = "=" * 96
    print(bar + "\nPRIMARY: readout-ceiling Δr (roundtrip, measured control), seeds averaged per pair\n" + bar)
    print(f"  {'jepa':12s} {jepa_mean:.4f}")
    for a in PRIMARY:
        c = fam["contrasts"][f"{a} − jepa"]
        lo, hi = c["ci_simultaneous"]
        print(f"  {a:12s} {means[a]:.4f}   diff {c['diff']:+.4f}  simultaneous [{lo:+.4f}, {hi:+.4f}]  {c['verdict']}")
    print(f"  max-t critical value {fam['critical_value']:.3f}")
    print(f"  ROUND: {fam['round_verdict']}")
    print(f"  joint-arm trigger: best {best} {means[best]:.4f} vs threshold {trigger['threshold']:.4f} -> "
          f"{'FIRES' if trigger['fires'] else 'does not fire'}")
    print(f"  sanity: jepa ceiling {jepa_mean:.4f} vs {JEPA_CEILING_REF}{'  FLAG' if sanity['flag'] else ''}")
    print(bar + "\nSECONDARY (percentile intervals)\n" + bar)
    for a in ARMS:
        r = table[f"roundtrip_{a}"]["measured"]
        line = f"  {a:12s} ceiling Δr {r['delta_r']['mean']:.3f} e_p {r['rel_err']['mean']:.3f} | "
        if a in piped:
            pl = table[f"pipeline_{a}"]["measured"]
            line += (f"pipeline Δr {pl['delta_r']['mean']:.3f} e_p {pl['rel_err']['mean']:.3f} | "
                     f"coverage {pop_table[a]['coverage']['mean']:.3f}")
        else:
            line += "pipeline not trainable under this recipe (amendment 1)"
        print(line)
    for k, v in visible_vs_jepa.items():
        print(f"  visible genes only, {k}: Δr {short(v['measured']['delta_r'])}")
    for k, v in projection_cost.items():
        print(f"  projection cost {k}: Δr {short(v['delta_r'])}")
    print(f"  diverged flows (amendments 1 and 2): {', '.join(diverged) if diverged else 'none'}")

    result = {
        "protocol": "foundation-encoder/07_score_comparison.py docstring", "seeds": list(seeds), "smoke": args.smoke,
        "pairs": pairs, "n_boot": args.n_boot, "alpha": args.alpha,
        "stores": {a: {k: st.manifest.get(k) for k in ("source", "model", "mode", "width", "n", "content_md5",
                                                       "space_key", "explained_variance_share")}
                   for a, st in stores.items()},
        "gate2": gate2,
        "primary": {"endpoint": "readout-ceiling Δr, roundtrip, measured control", "jepa": jepa_mean,
                    "arms": means, **fam},
        "joint_arm_trigger": trigger,
        "sanity_jepa_ceiling": sanity,
        "secondary": {
            "table": table, "vs_jepa": vs_jepa, "visible_genes_vs_jepa": visible_vs_jepa,
            "visible_genes_table": visible_table,
            "visible_genes_counts": {s: {k: visible["series"][s][k] for k in ("n_panel_visible", "scored_entries_visible",
                                                                              "pairs_by_n_visible", "unseen_scored_genes")}
                                     for s in ("V1", "V2")},
            "projection_cost": projection_cost, "projection_variance_share": projection_variance,
            "pipeline_vs_references": vs_ref, "headroom": headroom,
            "population": pop_table, "population_vs_jepa": pop_vs_jepa,
            "ranking_hypothesis": ranking, "effective_rank": eff_rank,
            "flow_training": flow_training,
            "diverged_flows": diverged,
            "pipelines_not_trainable": not_trainable,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
