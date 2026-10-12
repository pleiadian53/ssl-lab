"""Draw, check and save the labelled-patient subsets for study v1 (D17, D24).

For each repeat ``r`` the subsets come from one seeded permutation of all training
patients (``budget_seed = r``), so within a repeat the budgets are nested
(5% within 10% within 25% within 100%), and every method later reads the same
saved patient ids. At 100% every repeat holds the full training set.

For every (repeat, budget) the file records:

* patients selected (out of all training patients) and their ids
* selected patients with eligible recordings (at least one rhythm statement)
* eligible recordings
* positive patients and positive recordings for every rhythm label, and which
  labels are unsupported (zero positive patients) under the masking policy

Subsets are **never redrawn**. If any core label has zero positive patients in
any subset, the file is still written, the failure is printed, and the script
exits with status 2 so a pipeline stops before training.

Usage
-----
    python examples/ecg_rhythm/02_make_subsets.py
    python examples/ecg_rhythm/02_make_subsets.py --data-dir /runpod-volume/ssl-lab/data

Output
------
    output/<experiment>/subsets/{subsets.json, support.md}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

import _study as S
from ssllab.data.ptbxl import SPLIT_TRAIN, budget_patients, load_cache, patient_support

logger = logging.getLogger("02_make_subsets")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--study", default=None, help="protocol (default: configs/study_v1_amend1.json)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing subsets file (never for a pinned one)")
    args = p.parse_args()

    study = S.load_study(args.study)
    path = S.subsets_path(args.output_root, args.experiment)
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; saved subsets are never silently replaced (pass --overwrite to rebuild an unpinned file)")
    if path.exists() and (study["_amendment"] or {}).get("reuse", {}).get("subsets"):
        raise SystemExit(f"{path} is pinned by {study['_amendment']['amendment']}; it must not be rebuilt")
    cache = load_cache(args.data_dir / study["data"]["artifact"])
    rl = cache.rhythm()
    eligible = rl.has_rhythm
    split, pid = cache.split, cache.patient_id
    train_pats = np.unique(pid[split == SPLIT_TRAIN])
    core = study["labels"]["core"]
    budgets = study["budgets"]
    logger.info("training patients: %d; core labels: %s; budgets: %s", len(train_pats), core, budgets)

    out = {
        **S.provenance(study),
        "artifact": study["data"]["artifact"], "n_records": cache.n,
        "n_train_patients": int(len(train_pats)), "labels": list(rl.codes), "core": core,
        "budgets": [S.budget_key(b) for b in budgets], "repeats": {},
    }
    failures = []
    for r in range(study["repeats"]["n"]):
        sd = S.seeds(r)
        rep, prev = {"seeds": sd}, None
        for frac in budgets:
            chosen = budget_patients(train_pats, frac, sd["budget_seed"])
            if prev is not None and not set(prev) <= set(chosen.tolist()):
                raise AssertionError(f"repeat {r}: budget {frac} does not contain the previous budget")
            prev = chosen.tolist()
            rows = S.cohort_rows(cache, "train", eligible, chosen)
            support = {}
            for j, code in enumerate(rl.codes):
                pos_p, _ = patient_support(rl.y[rows, j], pid[rows])
                support[code] = {"pos_patients": pos_p, "pos_records": int(rl.y[rows, j].sum()), "supported": pos_p > 0}
            unsupported = [c for c in rl.codes if not support[c]["supported"]]
            core_fail = [c for c in core if not support[c]["supported"]]
            failures += [(r, frac, c) for c in core_fail]
            rep[S.budget_key(frac)] = {
                "n_patients_selected": int(len(chosen)),
                "n_patients_eligible": int(len(np.unique(pid[rows]))),
                "n_records_eligible": int(len(rows)),
                "support": support,
                "unsupported_labels": unsupported,
                "core_supported": not core_fail,
                "patients": [int(x) for x in chosen],
            }
        out["repeats"][str(r)] = rep

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))

    # Human-readable support table (core labels; min over repeats per budget).
    lines = [f"# Saved subsets: {study['study']} (base protocol sha {study['_base_sha256']})\n",
             f"Budgets are fractions of all {len(train_pats):,} training patients. Each cell: positive patients, "
             "min–max over repeats.\n",
             "| budget | patients selected | eligible patients | eligible recordings | " + " | ".join(core) + " | unsupported (any label) |",
             "|---|---|---|---|" + "---|" * len(core) + "---|"]
    for frac in budgets:
        k = S.budget_key(frac)
        reps = [out["repeats"][str(r)][k] for r in range(study["repeats"]["n"])]
        rng = lambda xs: f"{min(xs):,}" if min(xs) == max(xs) else f"{min(xs):,}–{max(xs):,}"  # noqa: E731
        cells = [rng([x["support"][c]["pos_patients"] for x in reps]) for c in core]
        uns = sorted({c for x in reps for c in x["unsupported_labels"]})
        lines.append(f"| {float(frac) * 100:g}% | {rng([x['n_patients_selected'] for x in reps])} | "
                     f"{rng([x['n_patients_eligible'] for x in reps])} | {rng([x['n_records_eligible'] for x in reps])} | "
                     + " | ".join(cells) + f" | {', '.join(uns) or 'none'} |")
    (path.parent / "support.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    logger.info("wrote %s", path)

    if failures:
        for r, frac, c in failures:
            logger.error("CORE SUPPORT FAILURE: repeat %d, budget %g, label %s has zero positive patients", r, frac, c)
        logger.error("subsets were NOT redrawn; resolve before training")
        sys.exit(2)


if __name__ == "__main__":
    main()
