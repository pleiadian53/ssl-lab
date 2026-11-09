"""How Δr and e_p turn two populations into one number, and what they each refuse to see.

The project's benchmark has two endpoints, and they disagree about how good the models are.
This script computes both on real held-out data, using only training-free references, so every
figure below is reproducible from the cache with no checkpoint.

**Δr (``delta_correlation``)** is a Pearson correlation between the predicted and the true
change, over a pair's top-20 DE genes:

    Δ̂ = predicted pair mean − control mean        Δ = true pair mean − control mean
    Δr = pearson(Δ̂, Δ)

Being a correlation, it is invariant to scale. A model that predicts the right *pattern* of
change at 10% of the right *magnitude* scores the same Δr as one that gets both right. That is
the blind spot Round 8 was built to close.

**e_p (``pair_metrics``'s ``rel_err``)** is a relative error, which is not scale-invariant:

    e_p = ‖Δ̂ − Δ‖ / ‖Δ‖

0 is perfect, and 1 is the score of predicting no change at all, because then ‖Δ̂ − Δ‖ = ‖Δ‖.
So e_p > 1 means a prediction is further from the truth than silence would have been.

Two further terms from the same function explain *why* a model's e_p is large. Projecting both
the prediction and the truth onto the additive direction s = Δ_A + Δ_B gives

    s_p = ⟨Δ̂, s⟩ / ⟨s, s⟩     the size of the prediction along s
    λ_p = ⟨Δ, s⟩ / ⟨s, s⟩      the size of the truth along s

so s_p / λ_p is the fraction of the real change a model actually commits to. Round 8's finding
was that the learned models sit near 0.6: they get the direction and under-shoot the distance.

Where it lives. ``ssllab.eval.effect_size.delta_correlation``, ``ssllab.eval.interaction``
(``pair_metrics``, ``measured_single_deltas``, ``split_half_floor``), and
``ssllab.eval.baselines.additive_predictor``. All imported, none reimplemented.

Sections
--------
    1. Δr and e_p on a hand-sized example, and what each one misses
    2. the additive reference on the 20 held-out combinations: both endpoints, per pair
    3. the scale terms: how much of the change does a prediction commit to?
    4. the floor: how well can a perfect predictor score, given finite held-out cells?

Usage
-----
    python examples/perturbation_response/data-science/02_metric_effect_size_and_ep.py
    python .../02_metric_effect_size_and_ep.py --sections 2,3 --top-k 20
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
from ssllab.data.perturbseq import SPLIT_TEST, SPLIT_TRAIN, target_genes  # noqa: E402
from ssllab.eval.baselines import additive_predictor  # noqa: E402
from ssllab.eval.effect_size import delta_correlation  # noqa: E402
from ssllab.eval.interaction import measured_single_deltas, pair_metrics, split_half_floor  # noqa: E402


def section_1_by_hand() -> None:
    print("=" * 78)
    print("1. Both endpoints on a hand-sized example")
    print("=" * 78)
    ctrl = np.zeros(5)
    true = np.array([2.0, 1.0, -1.0, 0.5, -2.0])  # the true change on 5 genes
    arms = {
        "perfect":                 true.copy(),
        "right pattern, 60% size": 0.6 * true,
        "right pattern, 10% size": 0.1 * true,
        "no change at all":        np.zeros(5),
        "sign flipped":            -true,
    }
    idx = np.arange(5)
    print(f"\n  true change Δ on 5 genes: {true}")
    print(f"\n  {'prediction':<26}{'Δr':>8}{'e_p':>8}   comment")
    for label, pred in arms.items():
        dr = delta_correlation(pred, true, ctrl, idx)
        ep = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
        note = {
            "perfect": "both endpoints agree",
            "right pattern, 60% size": "Δr cannot tell this from perfect",
            "right pattern, 10% size": "nor this",
            "no change at all": "e_p = 1 exactly, by construction",
            "sign flipped": "e_p = 2; Δr = −1",
        }[label]
        dr_s = "nan" if dr != dr else f"{dr:.3f}"
        print(f"  {label:<26}{dr_s:>8}{ep:>8.3f}   {note}")
    print("\n  The three rows that share Δr = 1.000 are the whole argument for Round 8:")
    print("  a scale-invariant endpoint grades a 10%-magnitude prediction as perfect. Note")
    print("  also that 'no change' has no Δr at all (a constant vector has no correlation),")
    print("  which is why the no-change reference only shows up on the e_p scale.")


def _pairs(data, top_k: int) -> list[str]:
    """Held-out combinations with a DE record and both singles measurable in training."""
    names = [str(n) for n in data.pert_names]
    is_test = data.split_combo == SPLIT_TEST
    out = []
    for pid, name in enumerate(names):
        genes = target_genes(name)
        if len(genes) < 2 or not (data.pert_id == pid)[is_test].any():
            continue
        if name not in data.de_genes or len(top_genes(data, name, top_k)) == 0:
            continue
        out.append(name)
    return out


def _compute(data, top_k: int, seed: int):
    """Per-pair metrics for the additive reference, plus the split-half floor."""
    ctrl = control_mean(data)
    is_train = data.split_combo == SPLIT_TRAIN
    is_test = data.split_combo == SPLIT_TEST
    predict = additive_predictor(
        hvg_X=data.hvg_X, pert_names=data.pert_names, pert_id=data.pert_id,
        is_train=is_train, control_mean=ctrl,
    )
    rows = {}
    for name in _pairs(data, top_k):
        pid, top = pert_index(data, name), top_genes(data, name, top_k)
        genes = target_genes(name)
        deltas = measured_single_deltas(
            genes, hvg_X=data.hvg_X, pert_names=data.pert_names, pert_id=data.pert_id,
            is_train=is_train, control_mean=ctrl,
        )
        pair_X = data.hvg_X[(data.pert_id == pid) & is_test]
        m = pair_metrics(
            pred_pair=predict(pid, name), true_pair=pair_X.mean(0).astype(np.float64),
            control_mean=ctrl, delta_a=deltas[genes[0]], delta_b=deltas[genes[1]], top_idx=top,
        )
        f = split_half_floor(
            pair_X, control_mean=ctrl, delta_a=deltas[genes[0]], delta_b=deltas[genes[1]],
            top_idx=top, seed=seed,
        )
        rows[name] = {**m, **f}
    return rows


def section_2_additive(rows: dict, top_k: int) -> None:
    print("\n" + "=" * 78)
    print(f"2. The additive reference on the held-out combinations (top-{top_k} genes)")
    print("=" * 78)
    print("\n  The prediction is Δ̂ = Δ_A + Δ_B, each single's measured effect from its TRAINING")
    print("  cells. Nothing is fitted, so this is reproducible from the cache alone.\n")
    print(f"  {'pair':<18}{'cells':>7}{'Δr':>8}{'e_p':>8}{'‖Δ‖':>8}{'rel_GI':>8}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["rel_err"]):
        print(f"  {name:<18}{r['n_cells']:>7}{r['delta_r']:>8.3f}{r['rel_err']:>8.3f}"
              f"{r['delta_norm']:>8.3f}{r['rel_gi']:>8.3f}")
    dr = np.array([r["delta_r"] for r in rows.values()])
    ep = np.array([r["rel_err"] for r in rows.values()])
    gi = np.array([r["rel_gi"] for r in rows.values()])
    print(f"\n  mean over {len(rows)} pairs:   Δr {dr.mean():.4f}    e_p {ep.mean():.4f}"
          f"    rel_GI {gi.mean():.4f}")
    gap = float(np.max(np.abs(ep - gi)))
    assert gap < 1e-9, f"e_p and rel_GI should be identical for the additive arm (gap {gap:.2e})"
    print("\n  The e_p and rel_GI columns are not merely close, they are IDENTICAL to"
          f" {gap:.1e}.")
    print("  That is an identity, not a coincidence. rel_GI = ‖I_p‖/‖Δ‖ measures how much of")
    print("  the change is interaction rather than the sum of the singles, and I_p is defined")
    print("  as Δ − s. The additive prediction is exactly Δ̂ = s, so its error Δ̂ − Δ = −I_p and")
    print("  the two ratios coincide. Read it as: the additive reference's error IS the")
    print("  interaction, so rel_GI is simultaneously its score and the floor it cannot beat.")
    print("\n  Which also calibrates the learned models. Round 8 put them at e_p 1.05 (cVAE)")
    print("  and 1.26 (transport flow) against this 0.289, i.e. 3.6-4.4x the error of a")
    print("  reference whose only knowledge is the two singles, and above the 1.0 that")
    print("  predicting no change scores.")


def section_3_scale(rows: dict) -> None:
    print("\n" + "=" * 78)
    print("3. The scale terms: how much of the change does a prediction commit to?")
    print("=" * 78)
    print("\n  Projecting onto the additive direction s = Δ_A + Δ_B:")
    print("    s_p = ⟨Δ̂, s⟩/⟨s, s⟩   the prediction's size along s")
    print("    λ_p = ⟨Δ,  s⟩/⟨s, s⟩   the truth's size along s\n")
    print(f"  {'pair':<18}{'s_p':>8}{'λ_p':>8}{'s_p/λ_p':>9}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["lam"]):
        ratio = r["scale"] / r["lam"] if r["lam"] not in (0.0,) and r["lam"] == r["lam"] else float("nan")
        print(f"  {name:<18}{r['scale']:>8.3f}{r['lam']:>8.3f}{ratio:>9.3f}")
    lam = np.array([r["lam"] for r in rows.values()])
    print(f"\n  s_p is exactly 1.000 for every pair, which is not a finding: the additive")
    print(f"  prediction IS s, so its projection onto s is 1 by construction. It is the")
    print(f"  control line for reading a learned model's s_p.")
    print(f"\n  λ_p has mean {lam.mean():.3f} over the pairs: the truth itself sits a little")
    print(f"  below the sum of the singles, which is sub-additivity and is what the")
    print(f"  interaction term I_p records. Round 8 measured the learned models at s_p ≈ 0.6,")
    print(f"  i.e. they under-shoot a target that is already slightly under the sum.")


def section_4_floor(rows: dict) -> None:
    print("\n" + "=" * 78)
    print("4. The floor: what a perfect predictor would score")
    print("=" * 78)
    print("\n  The truth is itself a mean over finitely many held-out cells. Splitting those")
    print("  cells in two and measuring the gap between the halves estimates the error an")
    print("  oracle would still show (``split_half_floor``). On the e_p scale:\n")
    print(f"  {'pair':<18}{'cells':>7}{'floor_e':>9}{'e_p (additive)':>16}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["n_cells"]):
        print(f"  {name:<18}{r['n_cells']:>7}{r['floor_e']:>9.3f}{r['rel_err']:>16.3f}")
    fl = np.array([r["floor_e"] for r in rows.values()])
    ep = np.array([r["rel_err"] for r in rows.values()])
    print(f"\n  mean floor_e {fl.mean():.3f}   mean e_p {ep.mean():.3f}")
    print(f"\n  The floor rises as cells fall, which is the link to 03_holdout_power.py: a")
    print(f"  pair with 81 cells cannot be scored as precisely as one with 1,198, yet the")
    print(f"  summary weights all 20 equally.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sections", type=str, default="1,2,3,4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    want = {s.strip() for s in args.sections.split(",")}
    if "1" in want:
        section_1_by_hand()
    if want & {"2", "3", "4"}:
        data = load_narrow(Path(args.data_dir) / args.artifact, "hvg_X")
        rows = _compute(data, args.top_k, args.seed)
        if "2" in want:
            section_2_additive(rows, args.top_k)
        if "3" in want:
            section_3_scale(rows)
        if "4" in want:
            section_4_floor(rows)
    print()


if __name__ == "__main__":
    main()
