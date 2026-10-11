"""PTB-XL data card: cohort sizes, label semantics, and patient-level class support.

Reads the cache written by ``00_process_ptbxl.py``. Trains nothing. The card
exists to fix the prediction task *before* any model output is seen:

* records and patients per split, records per patient
* label semantics: likelihood values per rhythm code (zero likelihood included),
  records with no rhythm statement, rhythm co-occurrence, AFIB/AFLT overlap
* patient-level positive and negative counts per rhythm label in train
  (folds 1-8) and validation (fold 9); test-fold prevalence, descriptive only
* the same counts inside the nested label budgets (1/5/10/25% of training
  patients) over several seeded draws: minimum, median, maximum, and how many
  draws leave a label with no positive patient
* AF pilot cohorts under the primary and sensitivity rules
* signal-quality flags and demographics per split

Usage
-----
    python examples/ecg_rhythm/01_data_card.py                     # data/ptbxl100
    python examples/ecg_rhythm/01_data_card.py --n-draws 20

Output
------
    output/<experiment>/reports/{data_card.json, data_card.md}
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.ptbxl import (
    QUALITY_FIELDS,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    af_pilot_cohort,
    budget_patients,
    load_cache,
    patient_support,
)
from ssllab.experiment import experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BUDGETS = (0.01, 0.05, 0.10, 0.25)


def _split_overview(split, pid):
    out = {}
    for s, name in SPLIT_NAMES.items():
        m = split == s
        per_patient = Counter(pid[m].tolist())
        out[name] = {
            "records": int(m.sum()),
            "patients": len(per_patient),
            "patients_with_multiple_records": sum(1 for v in per_patient.values() if v > 1),
            "max_records_per_patient": max(per_patient.values()) if per_patient else 0,
        }
    return out


def _likelihoods(rl):
    out = {}
    for j, code in enumerate(rl.codes):
        vals = rl.likelihood[:, j][rl.y[:, j] == 1]
        out[code] = {f"{v:g}": int(c) for v, c in sorted(Counter(vals.tolist()).items())}
    return out


def _cooccurrence(rl):
    y = rl.y.astype(np.int64)
    co = y.T @ y
    n_codes = y.sum(1)
    pairs = {}
    for a in range(len(rl.codes)):
        for b in range(a + 1, len(rl.codes)):
            if co[a, b]:
                pairs[f"{rl.codes[a]}+{rl.codes[b]}"] = int(co[a, b])
    return {
        "records_by_number_of_rhythm_codes": {str(k): int(v) for k, v in sorted(Counter(n_codes.tolist()).items())},
        "pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
    }


def _support(rl, split, pid):
    """Per code: patient-level (pos, neg) in train and val over the rhythm-task cohort."""
    cohort = rl.has_rhythm
    out = {}
    for j, code in enumerate(rl.codes):
        row = {}
        for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            m = cohort & (split == s)
            pos, neg = patient_support(rl.y[m, j], pid[m])
            row[SPLIT_NAMES[s]] = {"pos_patients": pos, "neg_patients": neg, "pos_records": int(rl.y[m, j].sum())}
        out[code] = row
    return out


def _summ(xs):
    xs = np.asarray(xs)
    return {"min": int(xs.min()), "median": float(np.median(xs)), "max": int(xs.max()), "draws_with_zero": int((xs == 0).sum())}


def _budget_support(rl, split, pid, af_inc, af_y, n_draws):
    train = split == SPLIT_TRAIN
    train_pats = pid[train]
    out = {}
    for frac in BUDGETS:
        per_code = {c: [] for c in rl.codes}
        af_pos, af_neg, n_pat, n_pat_rhythm, n_rec_rhythm = [], [], [], [], []
        for seed in range(n_draws):
            chosen = budget_patients(train_pats, frac, seed)
            n_pat.append(len(chosen))
            m = train & np.isin(pid, chosen) & rl.has_rhythm
            n_pat_rhythm.append(len(np.unique(pid[m])))
            n_rec_rhythm.append(int(m.sum()))
            for j, c in enumerate(rl.codes):
                per_code[c].append(patient_support(rl.y[m, j], pid[m])[0])
            ma = train & np.isin(pid, chosen) & af_inc
            p, n = patient_support(af_y[ma], pid[ma])
            af_pos.append(p)
            af_neg.append(n)
        out[f"{frac:g}"] = {
            "budget_patients": int(np.median(n_pat)),
            "patients_with_rhythm_records": _summ(n_pat_rhythm),
            "rhythm_records": _summ(n_rec_rhythm),
            "pos_patients": {c: _summ(v) for c, v in per_code.items()},
            "af_primary": {"pos_patients": _summ(af_pos), "neg_patients": _summ(af_neg)},
        }
    return out


def _af_cohorts(rl, split, pid):
    aflt = rl.y[:, rl.codes.index("AFLT")].astype(bool)
    afib = rl.y[:, rl.codes.index("AFIB")].astype(bool)
    out = {
        "excluded_no_rhythm_statement": {SPLIT_NAMES[s]: int((~rl.has_rhythm & (split == s)).sum()) for s in SPLIT_NAMES},
        "afib_and_aflt_records": int((afib & aflt).sum()),
        "aflt_only_records": int((aflt & ~afib).sum()),
    }
    for rule in ("primary", "sensitivity"):
        inc, y = af_pilot_cohort(rl, rule)
        r = {}
        for s, name in SPLIT_NAMES.items():
            m = inc & (split == s)
            pos, neg = patient_support(y[m], pid[m])
            r[name] = {"records": int(m.sum()), "pos_records": int(y[m].sum()), "pos_patients": pos, "neg_patients": neg}
        out[rule] = r
    return out


def _quality_and_demographics(records, split):
    out = {}
    age = np.array([r["age"] if r["age"] is not None else np.nan for r in records], dtype=float)
    sex = np.array([r["sex"] if r["sex"] is not None else np.nan for r in records], dtype=float)
    for s, name in SPLIT_NAMES.items():
        m = split == s
        a = age[m]
        real = a[(a < 300) & ~np.isnan(a)]
        out[name] = {
            "quality_flag_records": {q: int(sum(1 for r, k in zip(records, m) if k and r[q])) for q in QUALITY_FIELDS},
            "age_median": float(np.median(real)) if real.size else None,
            "age_iqr": [float(np.percentile(real, 25)), float(np.percentile(real, 75))] if real.size else None,
            "age_coded_300": int((a == 300).sum()),
            "sex_counts": {f"{k:g}": int(v) for k, v in sorted(Counter(sex[m][~np.isnan(sex[m])].tolist()).items())},
        }
    return out


# --------------------------------------------------------------------------- #
# Markdown rendering.
# --------------------------------------------------------------------------- #
def _fmt_budget(d):
    z = f", 0 in {d['draws_with_zero']}" if d["draws_with_zero"] else ""
    return f"{d['median']:g} [{d['min']}–{d['max']}{z}]"


def render_md(card) -> str:
    L = []
    m = card["manifest"]
    L.append(f"# PTB-XL data card ({m['source']['dataset']} {m['source']['version']}, {m['fs']:g} Hz)\n")
    L.append(f"{m['n_records']:,} records, {m['n_patients']:,} patients, leads {', '.join(m['leads'])}. "
             f"Checksums verified: {m['checksums_verified']}. Budget draws: {card['n_draws']}.\n")
    L.append("## Splits\n\n| split | records | patients | patients with >1 record | max records/patient |\n|---|---|---|---|---|")
    for k, v in card["splits"].items():
        L.append(f"| {k} | {v['records']:,} | {v['patients']:,} | {v['patients_with_multiple_records']:,} | {v['max_records_per_patient']} |")
    L.append("\n## Rhythm labels: patient-level support (rhythm-task cohort)\n")
    L.append("A patient is positive when any of their records carries the code. Test columns are descriptive prevalence only.\n")
    L.append("| code | description | train pos pts | train neg pts | train pos recs | val pos pts | val neg pts | test pos pts |\n|---|---|---|---|---|---|---|---|")
    desc = m["rhythm_descriptions"]
    for c, v in card["support"].items():
        L.append(f"| {c} | {desc[c]} | {v['train']['pos_patients']:,} | {v['train']['neg_patients']:,} | "
                 f"{v['train']['pos_records']:,} | {v['val']['pos_patients']:,} | {v['val']['neg_patients']:,} | {v['test']['pos_patients']:,} |")
    L.append("\n## Positive training patients inside label budgets\n")
    L.append("Each cell: median [min–max] over draws; \"0 in k\" counts draws with no positive patient.\n")
    hdr = " | ".join(f"{float(f) * 100:g}% ({card['budgets'][f]['budget_patients']:,} pts)" for f in card["budgets"])
    L.append(f"| code | {hdr} |\n|---|" + "---|" * len(card["budgets"]))
    for c in card["support"]:
        L.append(f"| {c} | " + " | ".join(_fmt_budget(card["budgets"][f]["pos_patients"][c]) for f in card["budgets"]) + " |")
    L.append("| AF pilot (primary) pos | " + " | ".join(_fmt_budget(card["budgets"][f]["af_primary"]["pos_patients"]) for f in card["budgets"]) + " |")
    L.append("| AF pilot (primary) neg | " + " | ".join(_fmt_budget(card["budgets"][f]["af_primary"]["neg_patients"]) for f in card["budgets"]) + " |")
    L.append("\n## AF pilot cohorts\n")
    a = card["af_cohorts"]
    L.append(f"Excluded for no rhythm statement: {a['excluded_no_rhythm_statement']}. AFIB+AFLT records: {a['afib_and_aflt_records']}. "
             f"Flutter-only records: {a['aflt_only_records']}.\n")
    L.append("| rule | split | records | pos records | pos patients | neg patients |\n|---|---|---|---|---|---|")
    for rule in ("primary", "sensitivity"):
        for s, v in a[rule].items():
            L.append(f"| {rule} | {s} | {v['records']:,} | {v['pos_records']:,} | {v['pos_patients']:,} | {v['neg_patients']:,} |")
    L.append("\n## Label semantics\n")
    L.append("Likelihood values among present rhythm codes:\n")
    L.append("| code | likelihood: records |\n|---|---|")
    for c, v in card["likelihoods"].items():
        L.append(f"| {c} | " + ", ".join(f"{k}: {n:,}" for k, n in v.items()) + " |")
    co = card["cooccurrence"]
    L.append(f"\nRecords by number of rhythm codes: {co['records_by_number_of_rhythm_codes']}.\n")
    L.append("Co-occurring rhythm pairs (records): " + (", ".join(f"{k}: {v}" for k, v in co["pairs"].items()) or "none") + ".\n")
    L.append("## Quality flags and demographics\n")
    L.append("| split | " + " | ".join(QUALITY_FIELDS) + " | age median [IQR] | age coded 300 | sex counts |\n|---|" + "---|" * (len(QUALITY_FIELDS) + 3))
    for s, v in card["quality_demographics"].items():
        q = " | ".join(f"{v['quality_flag_records'][k]:,}" for k in QUALITY_FIELDS)
        iqr = v["age_iqr"] or [None, None]
        L.append(f"| {s} | {q} | {v['age_median']:g} [{iqr[0]:g}–{iqr[1]:g}] | {v['age_coded_300']} | {v['sex_counts']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--artifact", default="ptbxl100")
    p.add_argument("--experiment", default="ecg_rhythm")
    p.add_argument("--output-root", default="output")
    p.add_argument("--n-draws", type=int, default=10, help="seeded budget draws per fraction")
    args = p.parse_args()

    cache = load_cache(args.data_dir / args.artifact)
    rl, split, pid = cache.rhythm(), cache.split, cache.patient_id
    af_inc, af_y = af_pilot_cohort(rl, "primary")
    logger.info("loaded %d records; computing card over %d budget draws", cache.n, args.n_draws)

    card = {
        "manifest": cache.manifest,
        "n_draws": args.n_draws,
        "splits": _split_overview(split, pid),
        "likelihoods": _likelihoods(rl),
        "no_rhythm_statement_records": int((~rl.has_rhythm).sum()),
        "cooccurrence": _cooccurrence(rl),
        "support": _support(rl, split, pid),
        "budgets": _budget_support(rl, split, pid, af_inc, af_y, args.n_draws),
        "af_cohorts": _af_cohorts(rl, split, pid),
        "quality_demographics": _quality_and_demographics(cache.records, split),
    }
    dirs = experiment(args.experiment, args.output_root).ensure()
    path = dirs.write_report("data_card", card)
    md = dirs.reports / "data_card.md"
    md.write_text(render_md(card))
    logger.info("wrote %s and %s", path, md)


if __name__ == "__main__":
    main()
