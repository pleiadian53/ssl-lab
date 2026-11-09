"""What is in the holdout, how unequal the 20 contributions are, and what that costs.

Every headline number in the [results](../docs/results/index.md) is an unweighted mean over
20 held-out two-gene combinations. This script opens that mean up. The holdout is **7,375
cells**, but they are not evenly spread: the largest combination has 1,198 and the smallest
81, a 14x range, and the summary gives each of them the same weight.

That matters in two concrete ways, and both are measured here rather than asserted.

**Precision is unequal.** A pair's score is a comparison against a mean over its own held-out
cells, so a pair with 81 cells has a noisier target than one with 1,198. ``split_half_floor``
puts a number on it: the error an oracle predictor would still show.

**One perturbation can dominate an interval.** The project grades a family of contrasts with a
max-t simultaneous bootstrap (``ssllab.eval.bootstrap.maxt_family``), whose critical value is
the 95th percentile of the largest studentized statistic across the family. A single pair with
unusual leverage inflates that critical value for *every* contrast in the family, which widens
every interval. Round 9 is the worked case: CEBPE+RUNX1T1 pushed the critical value to 4.63,
and the SIGReg intervals included zero partly for that reason rather than because the arms
were close.

The arms used below are all **training-free**, so the demonstration needs no checkpoint:

    no change     Δ̂ = 0                 scores e_p = 1 for every pair, by construction
    best single   Δ̂ = Δ_A  or  Δ_B      whichever single alone is closer
    additive      Δ̂ = Δ_A + Δ_B         the Round 7 reference

Sections
--------
    1. the holdout, by perturbation
    2. precision versus cell count
    3. the max-t critical value, and the leverage of one perturbation

Usage
-----
    python examples/perturbation_response/data-science/03_holdout_power.py
    python .../03_holdout_power.py --split cells --n-boot 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
for _p in (str(_HERE), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _data import control_mean, load_narrow, pert_index, top_genes  # noqa: E402
from ssllab.data.perturbseq import (  # noqa: E402
    SPLIT_TEST,
    SPLIT_TRAIN,
    perturbation_support,
    target_genes,
)
from ssllab.eval.bootstrap import maxt_family, resample_indices  # noqa: E402
from ssllab.eval.interaction import measured_single_deltas, split_half_floor  # noqa: E402


def section_1_holdout(data, split: str) -> None:
    print("=" * 78)
    print(f"1. The holdout, by perturbation (split scheme: {split})")
    print("=" * 78)
    codes = data.split_combo if split == "combo" else data.split_cells
    support = perturbation_support(data.pert_id, data.pert_names, codes)
    held = {n: r for n, r in support.items() if r["split"] == "test"}
    if not held:
        print("\n  No perturbation is wholly in test under this split scheme.")
        print("  Under 'cells' the split is per cell, so every perturbation is 'mixed' and")
        print("  there is no held-out perturbation at all. That is the point of the scheme:")
        print("  it is an in-distribution sanity check, not a generalization test.")
        return
    cells = np.array([r["total"] for r in held.values()])
    kinds = sorted({r["kind"] for r in held.values()})
    print(f"\n  {len(held)} perturbations, {cells.sum():,} cells, kinds: {', '.join(kinds)}")
    print(f"  cells per perturbation: min {cells.min():,}  median {int(np.median(cells)):,}"
          f"  mean {cells.mean():,.1f}  max {cells.max():,}   ({cells.max() / cells.min():.0f}x range)")
    share = np.sort(cells)[::-1] / cells.sum()
    print(f"  the largest 3 hold {100 * share[:3].sum():.0f}% of the holdout's cells;"
          f" the smallest 10 hold {100 * share[-10:].sum():.0f}%")
    print(f"\n  Each contributes 1/{len(held)} = {1 / len(held):.3f} of every reported mean,")
    print(f"  regardless of its size. An unweighted mean is a deliberate choice: the estimand")
    print(f"  is 'how well does the method do on a perturbation', not 'on a cell'. The cost is")
    print(f"  that the noisiest pairs carry as much weight as the best-measured ones.")


def _arms(data, top_k: int, seed: int) -> dict[str, dict[str, float]]:
    """Per-pair e_p for three training-free arms, plus the oracle floor and cell counts."""
    ctrl = control_mean(data)
    is_train = data.split_combo == SPLIT_TRAIN
    is_test = data.split_combo == SPLIT_TEST
    names = [str(n) for n in data.pert_names]
    out: dict[str, dict[str, float]] = {}
    for pid, name in enumerate(names):
        genes = target_genes(name)
        if len(genes) < 2 or name not in data.de_genes:
            continue
        mask = (data.pert_id == pid) & is_test
        if not mask.any():
            continue
        top = top_genes(data, name, top_k)
        if top.size == 0:
            continue
        deltas = measured_single_deltas(
            genes, hvg_X=data.hvg_X, pert_names=data.pert_names, pert_id=data.pert_id,
            is_train=is_train, control_mean=ctrl,
        )
        pair_X = data.hvg_X[mask]
        d = pair_X[:, top].mean(0).astype(np.float64) - ctrl[top]
        da, db = deltas[genes[0]][top], deltas[genes[1]][top]
        nd = float(np.linalg.norm(d))
        e = lambda pred: float(np.linalg.norm(pred - d) / nd)  # noqa: E731
        floor = split_half_floor(
            pair_X, control_mean=ctrl, delta_a=deltas[genes[0]], delta_b=deltas[genes[1]],
            top_idx=top, seed=seed,
        )
        out[name] = {
            "n_cells": int(mask.sum()),
            "e_nochange": e(np.zeros_like(d)),
            "e_single": min(e(da), e(db)),
            "e_additive": e(da + db),
            "floor_e": floor["floor_e"],
            "delta_norm": nd,
        }
    return out


def section_2_precision(rows: dict) -> None:
    print("\n" + "=" * 78)
    print("2. Precision versus cell count")
    print("=" * 78)
    n = np.array([r["n_cells"] for r in rows.values()], dtype=float)
    fl = np.array([r["floor_e"] for r in rows.values()], dtype=float)
    ok = np.isfinite(fl)
    print("\n  floor_e is the e_p an oracle would score, caused only by estimating the target")
    print("  from finitely many cells. Theory says it should fall like 1/sqrt(cells).\n")
    print(f"  {'bucket':<18}{'pairs':>7}{'median cells':>14}{'mean floor_e':>14}")
    order = np.argsort(n)
    keys = np.array(list(rows))[order]
    for lo, hi, label in [(0, 150, "under 150"), (150, 400, "150 to 400"), (400, 10**9, "over 400")]:
        sel = [k for k in keys if lo <= rows[k]["n_cells"] < hi]
        if not sel:
            continue
        c = np.array([rows[k]["n_cells"] for k in sel], dtype=float)
        f = np.array([rows[k]["floor_e"] for k in sel], dtype=float)
        f = f[np.isfinite(f)]
        print(f"  {label:<18}{len(sel):>7}{int(np.median(c)):>14,}{f.mean():>14.3f}")
    if ok.sum() > 3:
        r = np.corrcoef(np.log(n[ok]), np.log(fl[ok]))[0, 1]
        slope = np.polyfit(np.log(n[ok]), np.log(fl[ok]), 1)[0]
        print(f"\n  log-log fit over {int(ok.sum())} pairs: slope {slope:+.2f}, correlation {r:+.2f}")
        print(f"  (−0.50 would be exactly 1/sqrt(cells); the scatter is real biology, since")
        print(f"  ‖Δ‖ differs per pair and the floor is a ratio to it)")
    print("\n  Practical reading: the three smallest pairs carry floors of 0.09-0.24 on a scale")
    print("  where the additive reference scores 0.289 overall. For those pairs a large part")
    print("  of the measured error is the measurement, not the model.")


def section_3_maxt(rows: dict, n_boot: int, seed: int) -> None:
    print("\n" + "=" * 78)
    print("3. The max-t critical value, and one perturbation's leverage")
    print("=" * 78)
    names = list(rows)
    add = np.array([rows[k]["e_additive"] for k in names])
    sing = np.array([rows[k]["e_single"] for k in names])
    noch = np.array([rows[k]["e_nochange"] for k in names])
    print(f"\n  Three training-free arms, mean e_p over {len(names)} pairs:")
    print(f"    no change    {noch.mean():.4f}   (exactly 1 per pair, by construction)")
    print(f"    best single  {sing.mean():.4f}")
    print(f"    additive     {add.mean():.4f}")

    diffs = {"additive − no change": add - noch, "additive − best single": add - sing}
    idx = resample_indices(len(names), n_boot=n_boot, seed=seed)
    fam = maxt_family(diffs, idx)
    print(f"\n  A max-t family of {len(diffs)} contrasts on one shared bootstrap"
          f" ({n_boot:,} resamples):")
    print(f"    critical value {fam['critical_value']:.3f}   (1.96 would be the normal"
          f" quantile for a single contrast)")
    for k, v in fam["contrasts"].items():
        lo, hi = v["ci_simultaneous"]
        verdict = "wholly below 0" if hi < 0 else ("wholly above 0" if lo > 0 else "spans 0")
        print(f"    {k:<24} {v['diff']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    print("\n  Leave-one-out: the critical value recomputed with each pair dropped,")
    print("  lowest first (a negative change means that pair was inflating it).\n")
    print(f"  {'pair dropped':<20}{'crit':>8}{'change':>9}{'cells':>8}")
    base = fam["critical_value"]
    deltas = []
    for i, k in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        sub = {name: d[keep] for name, d in diffs.items()}
        c = maxt_family(sub, resample_indices(len(keep), n_boot=n_boot, seed=seed))["critical_value"]
        deltas.append((k, c, c - base, rows[k]["n_cells"]))
    deltas.sort(key=lambda t: t[2])
    for k, c, dd, nc in deltas[:3] + [("...", float("nan"), float("nan"), 0)] + deltas[-2:]:
        if k == "...":
            print(f"  {'...':<20}")
            continue
        print(f"  {k:<20}{c:>8.3f}{dd:>+9.3f}{nc:>8,}")
    # Isolate the family-size effect from the leverage effect.
    singles = {
        k: maxt_family({k: d}, idx)["critical_value"] for k, d in diffs.items()
    }
    print(f"\n  Family size alone accounts for part of it. Each contrast on its own:")
    for k, c in singles.items():
        print(f"    {k:<24} critical value {c:.3f}")
    print(f"    both together            critical value {base:.3f}")

    cebpe = next((t for t in deltas if t[0] == "CEBPE+RUNX1T1"), None)
    print("\n  What this does and does not show. The critical value is a property of the")
    print("  FAMILY and of the pair-to-pair spread of the differences, not of any one")
    print("  perturbation, and these numbers make that concrete in a way worth keeping:")
    if cebpe is not None:
        print(f"\n  * CEBPE+RUNX1T1, the pair that drove Round 9's critical value to 4.63, has")
        print(f"    almost NO leverage here: dropping it moves the value by {cebpe[2]:+.3f}.")
    print("  * The pairs with real leverage are different ones, and for most pairs dropping")
    print("    them RAISES the critical value, meaning their presence was stabilizing.")
    print("  * So leverage belongs to the (arm, contrast) combination, not to the pair. Round")
    print("    9's 4.63 came from how the SIGReg arms happened to differ on that pair, which")
    print("    these training-free references do not reproduce. Do not read 'CEBPE+RUNX1T1 is")
    print("    a high-leverage perturbation' as a standing fact about the holdout.")
    print("\n  The transferable lesson is the diagnostic, not the number: when an interval")
    print("  spans zero, check whether the critical value is unusually large before concluding")
    print("  the arms are close, because those are different findings. Rerun this leave-one-out")
    print("  against the arms you actually care about. And it is not a licence to drop a pair:")
    print("  the family and the 20 pairs are fixed in the protocol before scoring, precisely so")
    print("  that leverage cannot be discovered and then removed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=4000, help="bootstrap resamples for section 3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sections", type=str, default="1,2,3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    want = {s.strip() for s in args.sections.split(",")}
    need_X = bool(want & {"2", "3"})
    data = load_narrow(Path(args.data_dir) / args.artifact, *(("hvg_X",) if need_X else ()))
    if "1" in want:
        section_1_holdout(data, args.split)
    if need_X:
        rows = _arms(data, args.top_k, args.seed)
        if "2" in want:
            section_2_precision(rows)
        if "3" in want:
            section_3_maxt(rows, args.n_boot, args.seed)
    print()


if __name__ == "__main__":
    main()
