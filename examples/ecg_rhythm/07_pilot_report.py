"""Summarise the pilot and estimate the full grid's runtime and cost (D25).

Reads the repeat-0 pretraining report and the pilot downstream runs (validation
only), then prints:

* pretraining: loss, collapse diagnostics, wall time
* each pilot run: arm, budget, learning rate, best validation core macro AUROC,
  best step, whether the best step sat at the step cap, train time
* the shuffled-label sanity check (should sit near 0.5 AUROC)
* a runtime and cost estimate for the full grid: independent pretraining per
  repeat, plus every (arm, budget, repeat) downstream run with test predictions.
  Unmeasured budgets borrow the slower measured neighbour; a worst case assumes
  every run hits the step cap.

Usage
-----
    python examples/ecg_rhythm/07_pilot_report.py --tag pilot --usd-per-hr 0.63

Output
------
    output/<experiment>/reports/pilot_report.{json,md}
"""

from __future__ import annotations

import argparse
import json
import logging

import _study as S

logger = logging.getLogger("07_pilot_report")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default="pilot")
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--study", default=None)
    p.add_argument("--usd-per-hr", type=float, default=0.63, help="pod price (A40 default from ops/gpu_runner)")
    p.add_argument("--overhead-min", type=float, default=15.0, help="pod setup + staging + fetch, minutes")
    p.add_argument("--selection-tag", default=None, help="frozen lr selection (default: the protocol's selection tag)")
    p.add_argument("--timing-tags", default=None, help="comma-separated run tags to time from (default: tag + selection tag)")
    p.add_argument("--bootstrap-unit-s", type=float, default=20.0,
                   help="seconds per single-repeat paired bootstrap at n_boot (measured locally: ~20 s for 2000)")
    args = p.parse_args()

    study = S.load_study(args.study)
    root = S.study_root(args.output_root, args.experiment)
    pre = json.loads((root / "reports" / "pretrain_r0.json").read_text())
    sel_tag = args.selection_tag or S.selection_tag(study)
    sel_path = S.selection_path(args.output_root, args.experiment, sel_tag)
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {"arms": {}}
    tags = args.timing_tags.split(",") if args.timing_tags else sorted({args.tag, sel_tag})
    runs = [json.loads(f.read_text()) for t in tags for f in sorted((root / "runs" / t).glob("*/report.json"))]
    runs = [r for r in runs if not r.get("smoke")]

    L = [f"# Pilot report: {study['study']} (protocol {study['_sha256']}, base {study['_base_sha256']})\n",
         "Repeat 0 only, scored on **validation**. These numbers check the pipeline; they are not study results.\n",
         "## Pretraining (repeat 0)\n",
         f"{pre['n_recordings']:,} training recordings, {pre['epochs']} epochs, {pre['steps']:,} steps on {pre['gpu'] or pre['device']}: "
         f"**{pre['wall_seconds'] / 60:.1f} min** ({pre['seconds_per_epoch']:.1f} s/epoch).\n",
         "| epoch | loss | effective rank (of 128) | feature std |\n|---|---|---|---|"]
    for h in pre["history"]:
        if "collapse" in h:
            L.append(f"| {h['epoch']} | {h['loss']:.4f} | {h['collapse']['effective_rank']:.1f} | {h['collapse']['feature_std']:.3f} |")
    L.append(f"\n## Learning-rate selection `{sel_tag}` (budget 10%, repeat 0, validation; grid {sel.get('grid')})\n")
    for arm, v in sel["arms"].items():
        flags = [f for f in ((f"at {v['grid_edge']} grid edge" if v.get("grid_edge") else ""),
                             ("best step at the step cap" if v.get("best_at_cap") else "")) if f]
        L.append(f"- **{arm}**: chose {v['lr']:g} (best step {v.get('best_step')}; {', '.join(flags) or 'interior, not at cap'}); "
                 "by rate: " + ", ".join(f"{k}: {x:.4f}" for k, x in v["val_core_macro_auroc"].items()))
    L.append("\n## Pilot runs\n\n| arm | budget | lr | val core macro AUROC | best step | stopped | best at cap | train min | unsupported |\n|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["shuffle_labels"], r["arm"], r["budget"], r["lr"])):
        arm = r["arm"] + (" (shuffled labels)" if r["shuffle_labels"] else "")
        L.append(f"| {arm} | {r['budget'] * 100:g}% | {r['lr']:g} | {r['val_core_macro_auroc']:.4f} | {r['best_step']} | "
                 f"{r['stopped_step']} | {'YES' if r['best_at_cap'] else 'no'} | {r['train_seconds'] / 60:.1f} | "
                 f"{', '.join(r['unsupported_labels']) or 'none'} |")

    # Runtime estimate per (arm, budget): the selected rate's measurement if present, else the slowest
    # measurement at that budget at any rate, else the slowest measured budget for that arm.
    meas, any_lr = {}, {}
    for r in runs:
        if r["shuffle_labels"]:
            continue
        k = (r["arm"], r["budget"])
        any_lr[k] = max(any_lr.get(k, 0.0), r["train_seconds"])
        if sel["arms"].get(r["arm"], {}).get("lr") == r["lr"]:
            meas[k] = r["train_seconds"]
    budgets, arms, n_rep = study["budgets"], list(study["downstream"]["arms"]), study["repeats"]["n"]
    est, worst, basis = {}, {}, {}
    for arm in arms:
        sps = [r["steps_per_second"] for r in runs if r["arm"] == arm and not r["shuffle_labels"]]
        cap_s = study["downstream"]["max_steps"] / min(sps) if sps else None
        for b in budgets:
            if (arm, b) in meas:
                est[(arm, b)], basis[(arm, b)] = max(meas[(arm, b)], any_lr[(arm, b)]), "measured"
            elif (arm, b) in any_lr:
                est[(arm, b)], basis[(arm, b)] = any_lr[(arm, b)], "same budget, other rate"
            else:
                known = [v for (a, _), v in any_lr.items() if a == arm]
                est[(arm, b)], basis[(arm, b)] = (max(known) if known else None), "slowest measured budget"
            worst[(arm, b)] = cap_s
    pre_h = (n_rep - 1) * pre["wall_seconds"] / 3600   # repeat 0 is reused (amendment reuse record)
    down_h = n_rep * sum(v for v in est.values() if v) / 3600 * 1.05   # +5% for test prediction
    worst_h = n_rep * sum(v for v in worst.values() if v) / 3600 * 1.05
    over_h = args.overhead_min / 60
    total, total_worst = pre_h + down_h + over_h, pre_h + worst_h + over_h
    n_pairs = 3
    boot_h = len(budgets) * n_pairs * (n_rep + n_rep) * args.bootstrap_unit_s / 3600   # per-repeat + repeat-mean (R units)
    L.append("\n## Full-grid estimate\n")
    L.append(f"Grid: {n_rep} repeats; independent pretraining per repeat; arms {', '.join(arms)} × budgets "
             f"{', '.join(f'{b * 100:g}%' for b in budgets)} = {len(arms) * len(budgets) * n_rep} downstream runs, all with test predictions.\n")
    L.append("| component | hours (expected) | hours (worst case: every run hits the step cap) |\n|---|---|---|")
    L.append(f"| pretraining ({n_rep - 1} new runs; repeat 0 reused) | {pre_h:.2f} | {pre_h:.2f} |")
    L.append(f"| downstream ({len(arms) * len(budgets) * n_rep} runs) | {down_h:.2f} | {worst_h:.2f} |")
    L.append(f"| pod overhead | {over_h:.2f} | {over_h:.2f} |")
    L.append(f"| **total** | **{total:.2f}** | **{total_worst:.2f}** |")
    L.append(f"| **pod cost at ${args.usd_per_hr:.2f}/h** | **${total * args.usd_per_hr:.2f}** | **${total_worst * args.usd_per_hr:.2f}** |")
    L.append(f"\nComparison/bootstrap (06, test split): {len(budgets)} budgets × {n_pairs} pairs × ({n_rep} per-repeat + 1 repeat-mean "
             f"costing {n_rep}) × {args.bootstrap_unit_s:.0f} s ≈ **{boot_h:.2f} h of CPU**, run locally after fetching predictions "
             f"(no pod cost; ${boot_h * args.usd_per_hr:.2f} if run on the pod instead).")
    md = "\n".join(L) + "\n"
    rp = root / "reports" / "pilot_report"
    rp.with_suffix(".md").write_text(md)
    rp.with_suffix(".json").write_text(json.dumps({
        "pretrain_wall_seconds": pre["wall_seconds"], "measured_train_seconds": {f"{a}|{b:g}": v for (a, b), v in meas.items()},
        "estimate_hours": {"pretrain": pre_h, "downstream": down_h, "downstream_worst": worst_h, "overhead": over_h,
                           "total": total, "total_worst": total_worst, "bootstrap_cpu": boot_h},
        "usd_per_hr": args.usd_per_hr}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
