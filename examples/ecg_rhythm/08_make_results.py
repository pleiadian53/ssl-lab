"""Build the committed results for setting 01 (study v1: JEPA label efficiency) from the frozen run outputs.

Reads the test-split comparison (``06``), the frozen learning-rate selection, the
pretraining reports and the saved subsets, and writes everything the results
page shows, so no number on the page is transcribed by hand:

    examples/ecg_rhythm/results/01-jepa-label-efficiency/
      figures/label_efficiency.png     core macro AUROC vs labelled-patient budget, per arm
      figures/paired_differences.png   finetune - scratch and probe - scratch, with intervals
      figures/per_label_auroc.png      the five core labels, small multiples
      figures/af_operating_point.png   AF specificity and PPV at the validation-set threshold
      metrics/summary.json             curated numbers + provenance
      metrics/{per_run,pairs,per_label,af_operating_point}.csv
      metrics/tables.md                the markdown tables the README embeds

Usage
-----
    python examples/ecg_rhythm/08_make_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _study as S

logger = logging.getLogger("08_make_results")

ARMS = ("scratch", "finetune", "probe")
ARM_LABEL = {"scratch": "Scratch", "finetune": "JEPA + fine-tune", "probe": "JEPA + frozen probe"}
# Reference categorical palette, first three slots (validated all-pairs, light surface).
# Aqua sits below 3:1 contrast on the surface, so every series also gets a marker shape
# and a direct label, and the README carries the full tables.
COLOR = {"scratch": "#2a78d6", "finetune": "#eb6834", "probe": "#1baf7a"}
MARKER = {"scratch": "o", "finetune": "s", "probe": "^"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BUDGETS = ("0.05", "0.1", "0.25", "1")
XPOS = {b: float(b) * 100 for b in BUDGETS}


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xticks([XPOS[b] for b in BUDGETS])
    ax.set_xticklabels([f"{XPOS[b]:g}%" for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlabel("labelled training patients", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def _fig(w: float, h: float, n: int = 1):
    fig, axes = plt.subplots(1, n, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "01-jepa-label-efficiency",
                   help="topic folder for this setting (default: results/01-jepa-label-efficiency)")
    args = p.parse_args()

    study = S.load_study()
    root = S.study_root(args.output_root, args.experiment)
    comp = json.loads((root / "reports" / "compare_main_test.json").read_text())
    S.check_compatible(comp, study, "run", "compare_main_test.json")
    sel = json.loads(S.selection_path(args.output_root, args.experiment, S.selection_tag(study)).read_text())
    subsets = S.load_subsets(study, args.output_root, args.experiment)
    pre = [json.loads((root / "reports" / f"pretrain_r{r}.json").read_text()) for r in range(study["repeats"]["n"])]
    runs = {}
    for f in (root / "runs" / "main").glob("*/report.json"):
        r = json.loads(f.read_text())
        runs[(r["arm"], f"{r['budget']:g}", r["repeat"])] = r
    core, n_rep = comp["core"], study["repeats"]["n"]
    figs, mets = args.out / "figures", args.out / "metrics"
    figs.mkdir(parents=True, exist_ok=True)
    mets.mkdir(parents=True, exist_ok=True)

    pr = comp["per_run"]
    val = lambda arm, b, r: pr[f"{arm}|{b}|{r}"]["core_macro_auroc"]  # noqa: E731
    arm_stats = {arm: {b: {"mean": float(np.mean([val(arm, b, r) for r in range(n_rep)])),
                           "sd": float(np.std([val(arm, b, r) for r in range(n_rep)], ddof=1)),
                           "repeats": [val(arm, b, r) for r in range(n_rep)]} for b in BUDGETS} for arm in ARMS}

    # ---- Figure 1: label efficiency ------------------------------------------------------
    fig, (ax,) = _fig(7.2, 4.4)
    for arm in ARMS:
        xs = np.array([XPOS[b] for b in BUDGETS])
        jit = {"scratch": 0.94, "finetune": 1.0, "probe": 1.06}[arm]
        for b in BUDGETS:
            ax.scatter([XPOS[b] * jit] * n_rep, arm_stats[arm][b]["repeats"], s=14, color=COLOR[arm], alpha=0.35,
                       linewidths=0, zorder=2)
        m = [arm_stats[arm][b]["mean"] for b in BUDGETS]
        ax.plot(xs * jit, m, color=COLOR[arm], linewidth=2, marker=MARKER[arm], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=ARM_LABEL[arm])
        ax.annotate(ARM_LABEL[arm], (xs[-1] * jit, m[-1]), xytext=(12, {"scratch": -9, "finetune": 9, "probe": 0}[arm]),
                    textcoords="offset points", color=INK, fontsize=9, va="center")
    _style(ax, "core macro AUROC (test)")
    ax.set_xlim(4, 380)
    ax.set_title("Pretraining helps most when few patients are labelled", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, "Lines: mean of 5 repeats. Dots: individual repeats. Test fold 10, recording level.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(figs / "label_efficiency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 2: paired differences ----------------------------------------------------
    pairs = comp["pairs"]
    fig, (ax,) = _fig(7.2, 4.2)
    ax.axhline(0, color=INK2, linewidth=1, zorder=1)
    for pair, arm, jit in (("finetune-scratch", "finetune", 0.97), ("probe-scratch", "probe", 1.03)):
        xs, ms, lo, hi = [], [], [], []
        for b in BUDGETS:
            v = pairs[f"{pair}|{b}"]
            xs.append(XPOS[b] * jit)
            ms.append(v["repeat_mean"]["diff"])
            lo.append(v["repeat_mean"]["ci95_patients"][0])
            hi.append(v["repeat_mean"]["ci95_patients"][1])
            ax.scatter([XPOS[b] * jit] * len(v["per_repeat"]), [d["diff"] for d in v["per_repeat"].values()], s=14,
                       color=COLOR[arm], alpha=0.35, linewidths=0, zorder=2)
        ms, lo, hi = np.array(ms), np.array(lo), np.array(hi)
        ax.errorbar(xs, ms, yerr=[ms - lo, hi - ms], color=COLOR[arm], linewidth=2, marker=MARKER[arm], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3, zorder=3,
                    label=f"{ARM_LABEL[arm]} − scratch")
    _style(ax, "difference in core macro AUROC")
    ax.set_xlim(4, 140)
    ax.set_title("Paired differences against training from scratch", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.01, "Bars: 95% interval over test patients for the 5-repeat mean (paired patient-cluster bootstrap). "
             "Dots: individual repeats.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(figs / "paired_differences.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 3: per-label small multiples ---------------------------------------------
    lab = {arm: {c: {b: float(np.mean([pr[f"{arm}|{b}|{r}"]["per_label"][c]["auroc"] for r in range(n_rep)]))
                     for b in BUDGETS} for c in core} for arm in ARMS}
    desc = study["labels"].get("descriptions") or {}
    names = {"SR": "SR · sinus rhythm", "AFIB": "AFIB · atrial fibrillation", "STACH": "STACH · sinus tachycardia",
             "SARRH": "SARRH · sinus arrhythmia", "SBRAD": "SBRAD · sinus bradycardia"}
    fig, axes = _fig(14, 3.4, len(core))
    for ax, c in zip(axes, core):
        for arm in ARMS:
            ax.plot([XPOS[b] for b in BUDGETS], [lab[arm][c][b] for b in BUDGETS], color=COLOR[arm], linewidth=2,
                    marker=MARKER[arm], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=ARM_LABEL[arm])
        _style(ax, "test AUROC" if c == core[0] else "")
        ax.set_ylim(0.5, 1.0)
        ax.set_title(names.get(c, desc.get(c, c)), loc="left", color=INK, fontsize=10)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK)
    fig.text(0.005, 0.01, "Mean of 5 repeats per point; no intervals shown per label.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "per_label_auroc.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 4: AF operating point ------------------------------------------------------
    af = comp["af_operating_point"]
    afs = {arm: {b: {k: float(np.mean([af[f"{arm}|{b}|{r}"]["test"][k] for r in range(n_rep)]))
                     for k in ("sensitivity", "specificity", "ppv", "npv")} for b in BUDGETS} for arm in ARMS}
    fig, axes = _fig(11, 3.8, 3)
    for ax, k, title in zip(axes, ("sensitivity", "specificity", "ppv"),
                            ("Sensitivity (validation target 0.90)", "Specificity", "Positive predictive value")):
        if k == "sensitivity":
            ax.axhline(0.9, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for arm in ARMS:
            ax.plot([XPOS[b] for b in BUDGETS], [afs[arm][b][k] for b in BUDGETS], color=COLOR[arm], linewidth=2,
                    marker=MARKER[arm], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2, label=ARM_LABEL[arm])
        _style(ax, "")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK)
    prev = af["scratch|1|0"]["test"]["scores"]
    fig.text(0.005, 0.01, f"AF vs non-AF on the primary AF cohort of the test fold ({prev['n_pos']} AF, {prev['n_neg']} non-AF "
             f"recordings; prevalence {prev['prevalence']:.1%}). Threshold fixed on validation. Mean of 5 repeats.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / "af_operating_point.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # ---- Metrics files --------------------------------------------------------------------
    with (mets / "per_run.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "budget", "repeat", "lr", "core_macro_auroc_test", "val_core_macro_auroc", "best_step",
                    "best_at_cap", "n_train_records", "n_train_patients", "unsupported_labels"])
        for (arm, b, r), rep in sorted(runs.items()):
            w.writerow([arm, b, r, rep["lr"], round(val(arm, b, r), 6), round(rep["val_core_macro_auroc"], 6),
                        rep["best_step"], rep["best_at_cap"], rep["n_train_records"], rep["n_train_patients"],
                        ";".join(rep["unsupported_labels"])])
    with (mets / "pairs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "budget", "repeat", "diff", "ci95_low", "ci95_high", "interval"])
        for key, v in sorted(pairs.items()):
            pair, b = key.split("|")
            for r, d in v["per_repeat"].items():
                w.writerow([pair, b, r, round(d["diff"], 6), round(d["ci95"][0], 6), round(d["ci95"][1], 6), "test patients"])
            m = v["repeat_mean"]
            w.writerow([pair, b, "mean", round(m["diff"], 6), round(m["ci95_patients"][0], 6),
                        round(m["ci95_patients"][1], 6), f"test patients; repeat SD {v['repeat_spread']['sd']:.6f}"])
    with (mets / "per_label.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "arm", "budget", "auroc_mean", "auprc_mean", "prevalence"])
        for c in comp["per_run"]["scratch|1|0"]["per_label"]:
            for arm in ARMS:
                for b in BUDGETS:
                    rs = [pr[f"{arm}|{b}|{r}"]["per_label"][c] for r in range(n_rep)]
                    w.writerow([c, arm, b, round(float(np.nanmean([x["auroc"] for x in rs])), 6),
                                round(float(np.nanmean([x["auprc"] for x in rs])), 6), round(rs[0]["prevalence"], 6)])
    with (mets / "af_operating_point.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "budget", "repeat", "threshold", "sensitivity", "specificity", "ppv", "npv", "f1"])
        for key, v in sorted(af.items()):
            arm, b, r = key.split("|")
            t = v["test"]
            w.writerow([arm, b, r, round(v["threshold"], 6), round(t["sensitivity"], 6), round(t["specificity"], 6),
                        round(t["ppv"], 6), round(t["npv"], 6), round(t["f1"], 6)])

    summary = {
        **S.provenance(study),
        "protocol_files": {"base": "configs/study_v1.json", "in_force": f"configs/{S.STUDY_PATH.name}",
                           "amendments": ["configs/study_v1_amend1.json", "configs/study_v1_amend2.json"]},
        "core_labels": core, "budgets": [float(b) for b in BUDGETS], "repeats": n_rep,
        "selected_lr": {a: v["lr"] for a, v in sel["arms"].items()},
        "selection_grid": sel["grid"],
        "core_macro_auroc_test": arm_stats,
        "paired_differences": {k: {"repeat_mean": v["repeat_mean"], "repeat_sd": v["repeat_spread"]["sd"],
                                   "per_repeat": v["per_repeat"]} for k, v in pairs.items()},
        "per_label_auroc_test_mean": lab,
        "af_operating_point_test_mean": afs,
        "af_test_cohort": {k: prev[k] for k in ("n_pos", "n_neg", "prevalence")},
        "pretraining": [{"repeat": r, "minutes": round(p["wall_seconds"] / 60, 1), "final_loss": p["history"][-1]["loss"],
                         "effective_rank": p["final_collapse"]["effective_rank"]} for r, p in enumerate(pre)],
        "subsets_sha256_pinned": (study["_amendment"] or {}).get("reuse", {}).get("subsets", {}).get("sha256"),
        "n_train_patients": subsets["n_train_patients"],
    }
    (mets / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Markdown tables -------------------------------------------------------------------
    T = []
    T.append("| | " + " | ".join(f"{XPOS[b]:g}%" for b in BUDGETS) + " |\n|---|" + "---|" * len(BUDGETS))
    for arm in ARMS:
        T.append(f"| {ARM_LABEL[arm]} | " + " | ".join(f"{arm_stats[arm][b]['mean']:.3f} ± {arm_stats[arm][b]['sd']:.3f}"
                                                         for b in BUDGETS) + " |")
    T.append("")
    for pair, label in (("finetune-scratch", "JEPA + fine-tune − scratch"), ("probe-scratch", "JEPA + frozen probe − scratch"),
                        ("finetune-probe", "JEPA + fine-tune − frozen probe")):
        T.append(f"**{label}**\n\n| budget | mean difference | 95% interval (test patients) | repeat SD | repeats above 0 |\n|---|---|---|---|---|")
        for b in BUDGETS:
            v = pairs[f"{pair}|{b}"]
            m = v["repeat_mean"]
            pos = sum(d["diff"] > 0 for d in v["per_repeat"].values())
            T.append(f"| {XPOS[b]:g}% | {m['diff']:+.3f} | [{m['ci95_patients'][0]:+.3f}, {m['ci95_patients'][1]:+.3f}] | "
                     f"{v['repeat_spread']['sd']:.3f} | {pos} / {len(v['per_repeat'])} |")
        T.append("")
    T.append("| label | " + " | ".join(f"{XPOS[b]:g}%" for b in BUDGETS) + " |\n|---|" + "---|" * len(BUDGETS))
    for c in core:
        T.append(f"| {c} | " + " | ".join(" / ".join(f"{lab[a][c][b]:.3f}" for a in ARMS) for b in BUDGETS) + " |")
    T.append("")
    T.append("| arm | budget | sensitivity | specificity | PPV | NPV |\n|---|---|---|---|---|---|")
    for arm in ARMS:
        for b in BUDGETS:
            s = afs[arm][b]
            T.append(f"| {ARM_LABEL[arm]} | {XPOS[b]:g}% | {s['sensitivity']:.3f} | {s['specificity']:.3f} | {s['ppv']:.3f} | {s['npv']:.3f} |")
    (mets / "tables.md").write_text("\n".join(T) + "\n")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
