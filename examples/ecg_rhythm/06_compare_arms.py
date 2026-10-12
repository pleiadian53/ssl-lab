"""Compare arms of study v1 with paired patient-cluster bootstrap (D22, D23).

For each budget and each pair of arms (``A - B``; primary pair ``finetune - scratch``):

* per run: core macro AUROC at recording level on the split's eligible rhythm
  cohort, plus per-label AUROC/AUPRC;
* per repeat: the difference with a paired patient-cluster bootstrap interval
  (patients resampled with all their recordings, both arms scored on the same
  resample);
* across repeats: the mean difference, with a bootstrap interval in which every
  repeat is scored on the same patient resample (test-patient uncertainty of the
  repeat mean, **not** training variability), and separately the repeat-to-repeat
  spread of the per-repeat differences.

With ``--split test`` it also reports the AF operating point: on the primary AF
cohort, the threshold is chosen on that run's validation predictions by the
pre-declared rule (highest threshold with validation sensitivity >= target) and
applied unchanged to test. With ``--split val`` (the pilot), only machinery is
exercised: numbers on validation are not results.

Usage
-----
    python examples/ecg_rhythm/06_compare_arms.py --tag pilot --split val
    python examples/ecg_rhythm/06_compare_arms.py --tag main --split test

Output
------
    output/<experiment>/reports/compare_<tag>_<split>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

import _study as S
from ssllab.data.ptbxl import af_pilot_cohort, load_cache
from ssllab.eval.classification import (
    macro_auroc,
    macro_auroc_metric,
    operating_point,
    paired_cluster_bootstrap,
    per_label_metrics,
    threshold_at_sensitivity,
)

logger = logging.getLogger("06_compare_arms")
PAIRS = (("finetune", "scratch"), ("probe", "scratch"), ("finetune", "probe"))
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _pair_task(task):
    """One (pair, budget): per-repeat paired bootstraps, the repeat-mean bootstrap, the repeat spread."""
    key, Y, clusters, A, B, common, core_idx, n_boot = task
    metric = macro_auroc_metric(core_idx)
    per_rep = {}
    for i, r in enumerate(common):
        bs = paired_cluster_bootstrap(metric, Y, A[i], B[i], clusters, n_boot=n_boot, seed=1000 + r)
        per_rep[str(r)] = {"diff": bs.estimate, "ci95": [bs.ci_low, bs.ci_high], "n_valid": bs.n_valid}
    diffs = np.array([v["diff"] for v in per_rep.values()])
    res = {"repeats": common, "per_repeat": per_rep,
           "repeat_spread": {"mean": float(diffs.mean()), "sd": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
                             "min": float(diffs.min()), "max": float(diffs.max())}}
    if len(common) > 1:
        bs = paired_cluster_bootstrap(metric, Y, A, B, clusters, n_boot=n_boot, seed=999, repeats=True)
        res["repeat_mean"] = {"diff": bs.estimate, "ci95_patients": [bs.ci_low, bs.ci_high], "n_valid": bs.n_valid}
    return key, res


def score_summary(y: np.ndarray, s: np.ndarray) -> dict:
    """Quantiles of the AF score among positives and among negatives, plus prevalence."""
    y = np.asarray(y).astype(bool)
    q = lambda v: {f"q{int(x * 100):02d}": float(np.quantile(v, x)) for x in QUANTILES} if v.size else {}  # noqa: E731
    return {"n_pos": int(y.sum()), "n_neg": int((~y).sum()), "prevalence": float(y.mean()),
            "pos": q(s[y]), "neg": q(s[~y]), "mean_score": float(s.mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default="main")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--study", default=None)
    p.add_argument("--n-boot", type=int, default=None)
    p.add_argument("--selection-tag", default=None, help="frozen lr selection (default: the protocol's selection tag)")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes for the bootstrap (results identical)")
    p.add_argument("--allow-smoke", action="store_true", help="plumbing checks only: accept smoke runs (recorded in the output)")
    p.add_argument("--ignore-unselected", action="store_true",
                   help="skip runs at non-selected rates (recorded) instead of stopping; for mixed tags such as a search tag")
    args = p.parse_args()

    study = S.load_study(args.study)
    n_boot = args.n_boot or study["evaluation"]["n_boot"]
    cache = load_cache(args.data_dir / study["data"]["artifact"])
    rl = cache.rhythm()
    names, core = list(rl.codes), study["labels"]["core"]
    core_idx = [names.index(c) for c in core]
    pid = cache.patient_id
    af_inc, af_y = af_pilot_cohort(rl, "primary")
    target = study["af_operating_point"]["target_sensitivity"]
    metric = macro_auroc_metric(core_idx)

    root = S.study_root(args.output_root, args.experiment)
    # Only each arm's frozen learning rate enters the comparison (D19, amendment 1).
    subsets = S.load_subsets(study, args.output_root, args.experiment)
    sel_tag = args.selection_tag or S.selection_tag(study)
    sel_path = S.selection_path(args.output_root, args.experiment, sel_tag)
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} missing: run 05_select_lr.py first")
    selection = json.loads(sel_path.read_text())
    sel_reason = S.check_compatible(selection, study, "run", str(sel_path))
    if selection["grid"] != study["downstream"]["lr_grid"]:
        raise SystemExit(f"{sel_path} used grid {selection['grid']}, protocol grid is {study['downstream']['lr_grid']}")
    chosen = {a: v["lr"] for a, v in selection["arms"].items()}
    runs = defaultdict(dict)  # (arm, budget) -> {repeat: (report, preds)}
    excluded = []             # every file not combined, with the reason
    for f in sorted((root / "runs" / args.tag).glob("*/report.json")):
        rep = json.loads(f.read_text())
        if rep["smoke"] and not args.allow_smoke:
            raise SystemExit(f"{f.parent}: smoke run in tag {args.tag}; refusing to combine")
        if rep["shuffle_labels"]:
            excluded.append({"run": f.parent.name, "reason": "shuffled-label sanity check, never compared"})
            continue
        if rep["arm"] not in chosen:
            raise SystemExit(f"{f.parent}: arm {rep['arm']} has no frozen rate in {sel_path.name}")
        if chosen[rep["arm"]] != rep["lr"]:
            if not args.ignore_unselected:
                raise SystemExit(f"{f.parent}: lr {rep['lr']:g} is not the frozen rate {chosen[rep['arm']]:g} for {rep['arm']}"
                                 " (pass --ignore-unselected for a search tag)")
            excluded.append({"run": f.parent.name, "reason": f"lr {rep['lr']:g} is not the frozen rate {chosen[rep['arm']]:g}"})
            continue
        rep["_compat"] = S.check_compatible(rep, study, "run", str(f.parent))
        if (rep["arm"], rep["budget"]) in runs and rep["repeat"] in runs[(rep["arm"], rep["budget"])]:
            raise SystemExit(f"duplicate run for {rep['arm']} budget {rep['budget']} repeat {rep['repeat']}")
        pr = np.load(f.parent / "preds.npz")
        if f"{args.split}_scores" not in pr:
            raise SystemExit(f"{f.parent}: no {args.split} predictions (rerun with --eval-test)")
        runs[(rep["arm"], rep["budget"])][rep["repeat"]] = (rep, pr)

    out = {**S.provenance(study), "tag": args.tag, "split": args.split, "n_boot": n_boot, "lr": chosen,
           "selection": {"tag": sel_tag, "path": sel_path.name, "compatibility": sel_reason},
           "subsets_reuse_reason": subsets["_reuse_reason"], "excluded": excluded,
           "allow_smoke": args.allow_smoke,
           "core": core, "per_run": {}, "pairs": {}, "af_operating_point": {}}
    if args.split == "val":
        out["note"] = "validation split: machinery check only; these numbers are not study results"

    for (arm, budget), reps in sorted(runs.items()):
        for r, (rep, pr) in sorted(reps.items()):
            rows, Sx = pr[f"{args.split}_rows"], pr[f"{args.split}_scores"]
            Y = rl.y[rows]
            m, complete = macro_auroc(Y, Sx, core_idx)
            key = f"{arm}|{budget:g}|{r}"
            out["per_run"][key] = {"compatibility": rep["_compat"], "reused": rep.get("reused", {}), "core_macro_auroc": m, "complete": complete, "unsupported_labels": rep["unsupported_labels"],
                                   "per_label": per_label_metrics(Y, Sx, names)}
            # AF operating point: threshold on validation (pre-declared rule), applied unchanged.
            # A low threshold alone says nothing about calibration; we report the operating
            # point and the score distributions, and leave calibration to its own analysis.
            afi = names.index("AFIB")
            vr, vs = pr["val_rows"], pr["val_scores"][:, afi]
            keep = af_inc[vr]
            thr = threshold_at_sensitivity(af_y[vr][keep], vs[keep], target)
            entry = {"threshold": thr, "rule": f"highest threshold with validation sensitivity >= {target}",
                     "val": {**operating_point(af_y[vr][keep], vs[keep], thr),
                             "scores": score_summary(af_y[vr][keep], vs[keep])}}
            if args.split == "test":
                tk = af_inc[rows]
                entry["test"] = {**operating_point(af_y[rows][tk], Sx[:, afi][tk], thr),
                                 "scores": score_summary(af_y[rows][tk], Sx[:, afi][tk])}
            out["af_operating_point"][key] = entry

    # Assemble every (pair, budget) task first; the bootstraps are independent and seeded
    # per repeat (1000 + r) and for the repeat mean (999), so results do not depend on --jobs.
    tasks = []
    for a, b in PAIRS:
        for budget in sorted({bud for (_, bud) in runs}):
            ra, rb = runs.get((a, budget), {}), runs.get((b, budget), {})
            common = sorted(set(ra) & set(rb))
            if not common:
                continue
            rows = ra[common[0]][1][f"{args.split}_rows"]
            for r in common:
                for src in (ra, rb):
                    if not np.array_equal(src[r][1][f"{args.split}_rows"], rows):
                        raise SystemExit(f"row mismatch between runs at budget {budget}, repeat {r}")
            A = np.stack([ra[r][1][f"{args.split}_scores"] for r in common])
            B = np.stack([rb[r][1][f"{args.split}_scores"] for r in common])
            tasks.append((f"{a}-{b}|{budget:g}", rl.y[rows], pid[rows], A, B, common, core_idx, n_boot))
    logger.info("%d bootstrap tasks (%d resamples each), jobs=%d", len(tasks), n_boot, args.jobs)
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_pair_task, tasks))
    else:
        results = [_pair_task(t) for t in tasks]
    for key, res in results:
        out["pairs"][key] = res

    rp = root / "reports" / f"compare_{args.tag}_{args.split}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=2))

    L = [f"# Arm comparison: tag `{args.tag}`, split `{args.split}`\n"]
    if args.split == "val":
        L.append("> Validation split: a machinery check, not study results.\n")
    L.append("## Core macro AUROC per run\n\n| arm | budget | repeat | core macro AUROC | complete | unsupported labels |\n|---|---|---|---|---|---|")
    for k, v in out["per_run"].items():
        arm, bud, r = k.split("|")
        L.append(f"| {arm} | {bud} | {r} | {v['core_macro_auroc']:.4f} | {v['complete']} | {', '.join(v['unsupported_labels']) or 'none'} |")
    L.append(f"\n## Paired differences (A − B), core macro AUROC\n\nIntervals: paired patient-cluster bootstrap over **{args.split}** patients "
             "(patient-sampling uncertainty only; repeat-to-repeat spread is reported separately).\n\n"
             f"| pair | budget | repeat | diff | 95% CI ({args.split} patients) |\n|---|---|---|---|---|")
    for k, v in out["pairs"].items():
        pair, bud = k.split("|")
        for r, d in v["per_repeat"].items():
            L.append(f"| {pair} | {bud} | {r} | {d['diff']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
        if "repeat_mean" in v:
            m = v["repeat_mean"]
            sp = v["repeat_spread"]
            L.append(f"| {pair} | {bud} | mean of {len(v['repeats'])} | {m['diff']:+.4f} | [{m['ci95_patients'][0]:+.4f}, "
                     f"{m['ci95_patients'][1]:+.4f}]; repeat SD {sp['sd']:.4f} |")
    L.append(f"\n## AF operating point (primary AF cohort; threshold rule: {next(iter(out['af_operating_point'].values()))['rule'] if out['af_operating_point'] else ''})\n")
    L.append("Scores are the AFIB output. Quantiles of the score among AF-positive and AF-negative recordings are shown; "
             "a low threshold by itself is not evidence of miscalibration.\n")
    L.append("| arm | budget | repeat | split | threshold | sens | spec | PPV | NPV | pos q05 / q50 / q95 | neg q50 / q95 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in out["af_operating_point"].items():
        arm, bud, r = k.split("|")
        for sp_name in ("val", "test"):
            if sp_name not in v:
                continue
            o, sc = v[sp_name], v[sp_name]["scores"]
            L.append(f"| {arm} | {bud} | {r} | {sp_name} | {o['threshold']:.4f} | {o['sensitivity']:.3f} | {o['specificity']:.3f} | "
                     f"{o['ppv']:.3f} | {o['npv']:.3f} | {sc['pos']['q05']:.3f} / {sc['pos']['q50']:.3f} / {sc['pos']['q95']:.3f} | "
                     f"{sc['neg']['q50']:.4f} / {sc['neg']['q95']:.3f} |")
    (rp.with_suffix(".md")).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    logger.info("wrote %s", rp)


if __name__ == "__main__":
    main()
