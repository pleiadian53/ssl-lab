"""How energy distance is computed, and why this repository has two estimators of it.

Energy distance grades a *population* against a *population*. It is the only metric in the
project that compares a generated cloud of cells to the real held-out cloud without needing
any correspondence between individual cells, which is exactly the situation: generated cell
``i`` is not real cell ``i``, and the two clouds need not even be the same size.

    E(P, Q) = 2·avg‖p − q‖  −  avg‖p − p'‖  −  avg‖q − q'‖
              └ between ──┘    └── within P ──┘  └── within Q ──┘

Twice the average distance *between* the clouds, minus the average spread *within* each.
Zero exactly when the two clouds share a distribution, positive otherwise.

Where it lives. The scoring version is ``ssllab.eval.calibration.energy_distance``, one of
the population-quality reads alongside per-gene 1-Wasserstein and interval coverage. The
training version is ``ssllab.generative.operator_perturb.energy_distance``, which is the
operator's loss. Both are the **V-statistic**: ``cdist(x, x).mean()`` averages over all n²
within-cloud pairs, including the n zero self-distances on the diagonal.

That diagonal is the subject of section 2. Including it understates each cloud's internal
spread, and the understatement grows as the cloud shrinks, so two clouds of different sizes
are not on the same scale. The Koopman probe needs exactly that comparison (its finite-sample
floor is computed on half-size clouds) and so carries a **U-statistic** in
``co-adaptation/01_koopman_linearity_probe.py:energy_u``, which divides by n(n−1) instead.
This script asserts the exact identity relating the two, rather than asking you to take the
word "biased" on trust.

Sections
--------
    1. a 1-D example small enough to check by hand
    2. the diagonal: V versus U, the exact identity, and the bias as a function of n
    3. what the joint view buys over per-gene marginals
    4. the real scale: held-out cells of four perturbations, against control and against a floor
    5. the estimator at the sizes the calibration evaluation actually uses

Usage
-----
    python examples/perturbation_response/data-science/01_metric_energy_distance.py
    python .../01_metric_energy_distance.py --perturbation TBX2+TBX3 --perturbation PLK4+STIL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
for _p in (str(_HERE), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _data import load_narrow, pert_index, top_genes  # noqa: E402  local, file plumbing only
from ssllab.data.perturbseq import SPLIT_TEST  # noqa: E402
from ssllab.eval.calibration import energy_distance as energy_v  # noqa: E402  the production estimator

DEFAULT_PERTS = ["CEBPE+RUNX1T1", "TBX2+TBX3", "PLK4+STIL", "DLX2+ZBTB10"]


def energy_u(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased (U-statistic) energy distance: the same formula with the diagonal excluded.

    This mirrors ``energy_u`` in ``co-adaptation/01_koopman_linearity_probe.py``. It is the one
    place this script implements something itself rather than importing it, and section 2
    asserts its exact relationship to the production V-statistic so the two cannot drift apart
    silently.
    """
    X = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    if X.ndim == 1:
        X, Y = X[:, None], Y[:, None]
    n, m = len(X), len(Y)
    return float(
        2 * torch.cdist(X, Y).mean()
        - torch.cdist(X, X).sum() / (n * (n - 1))
        - torch.cdist(Y, Y).sum() / (m * (m - 1))
    )


def _v(x: np.ndarray, y: np.ndarray) -> float:
    """The production V-statistic, called on all columns (it takes an explicit gene index)."""
    x, y = np.atleast_2d(np.asarray(x, dtype=np.float64)), np.atleast_2d(np.asarray(y, dtype=np.float64))
    if x.shape[0] == 1 and x.shape[1] != 1:
        x, y = x.T, y.T
    return energy_v(x, y, np.arange(x.shape[1]))


def _within_v(x: np.ndarray) -> float:
    """``avg‖p − p'‖`` over all n² pairs, the V-statistic's within-cloud term."""
    X = torch.as_tensor(np.atleast_2d(np.asarray(x, dtype=np.float64)), dtype=torch.float32)
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    return float(torch.cdist(X, X).mean())


# --------------------------------------------------------------------------- #
def section_1_by_hand() -> None:
    print("=" * 78)
    print("1. A 1-D example small enough to check by hand")
    print("=" * 78)
    P, Q = np.array([0.0, 2.0]), np.array([4.0, 6.0])
    cross = [abs(p - q) for p in P for q in Q]
    within = [abs(a - b) for a in P for b in P]
    print(f"\n  P = {list(P)}    Q = {list(Q)}")
    print(f"\n  between, all 4 pairs : {cross}")
    print(f"    avg‖p − q‖         = {sum(cross)}/4 = {np.mean(cross)}")
    print(f"\n  within P, all 4 pairs: {within}        <- two of them are the diagonal, both 0")
    print(f"    avg over n² = 4    = {sum(within)}/4 = {np.mean(within)}   (V-statistic)")
    print(f"    avg over n(n−1) = 2= {sum(within)}/2 = {sum(within) / 2}   (U-statistic)")
    print("\n  within Q is the same by symmetry.")
    print(f"\n  E_V = 2({np.mean(cross)}) − {np.mean(within)} − {np.mean(within)} = {_v(P, Q):.1f}")
    print(f"  E_U = 2({np.mean(cross)}) − {sum(within) / 2} − {sum(within) / 2} = {energy_u(P, Q):.1f}")
    print("\n  Move Q closer, to {1, 3}, and the cross term falls from 4.0 to 1.5:")
    Q2 = np.array([1.0, 3.0])
    print(f"    E_V = {_v(P, Q2):.1f}   E_U = {energy_u(P, Q2):.1f}")
    print("\n  The clouds are 4 apart with internal spread 1, so most of the cross term")
    print("  survives. E_U going negative here is not a bug: an unbiased estimator of a")
    print("  quantity that is 0 must be able to land on both sides of it, and with n = 2 its")
    print("  variance is enormous. Section 2 shows that variance shrinking with n.")


def section_2_diagonal(seed: int, reps: int) -> None:
    print("\n" + "=" * 78)
    print("2. The diagonal: V versus U, and what the difference costs")
    print("=" * 78)
    print("\n  The two estimators differ only in the within-cloud denominator, so their")
    print("  difference is exactly the diagonal's share of it:")
    print("\n      E_V − E_U  =  within_V(P)/(n−1)  +  within_V(Q)/(m−1)")
    print("\n  Asserting that identity on random clouds of unequal size:")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n, m, d in [(7, 11, 1), (40, 40, 3), (200, 1198, 20), (83, 500, 20)]:
        X, Y = rng.normal(size=(n, d)), rng.normal(size=(m, d)) + 0.4
        lhs = _v(X, Y) - energy_u(X, Y)
        rhs = _within_v(X) / (n - 1) + _within_v(Y) / (m - 1)
        worst = max(worst, abs(lhs - rhs))
        print(f"    n={n:5d} m={m:5d} d={d:3d}   E_V−E_U = {lhs:.6f}   identity = {rhs:.6f}")
    assert worst < 1e-4, f"the V/U identity does not hold (worst {worst:.2e})"
    print(f"\n  Largest discrepancy {worst:.2e} (float32 cdist) -> the identity holds.")

    print("\n  So the V-statistic is inflated by roughly within/(n−1): a bias that shrinks")
    print("  like 1/n. Measured on two samples drawn from the SAME distribution, where the")
    print("  true energy distance is exactly 0:\n")
    print(f"    {'n':>6}  {'E[E_V]':>12}  {'E[E_U]':>12}")
    for n in [10, 50, 200, 500]:
        vs, us = [], []
        for _ in range(reps):
            a, b = rng.normal(size=n), rng.normal(size=n)
            vs.append(_v(a, b))
            us.append(energy_u(a, b))
        print(f"    {n:>6}  {np.mean(vs):>+12.5f}  {np.mean(us):>+12.5f}")
    print("\n  E_U sits at zero at every size and is free to go slightly negative, which is")
    print("  what an unbiased estimator does. E_V is systematically positive.")
    print("\n  Why it mattered. The Koopman probe computes a finite-sample floor by splitting")
    print("  the perturbed cloud in half, so the floor is estimated at HALF the sample size of")
    print("  the quantity it is a floor for. Under the V-statistic the half-size floor came out")
    print("  LARGER than the signal, and the reduction ratio (E_id − E_x)/(E_id − E_floor)")
    print("  exploded to 216. The project keeps the V-statistic as the training loss, where a")
    print("  constant offset does not move the argmin, and scores with the U-statistic.")


def section_3_joint(seed: int) -> None:
    print("\n" + "=" * 78)
    print("3. What the joint view buys over per-gene marginals")
    print("=" * 78)
    from scipy.stats import wasserstein_distance

    rng = np.random.default_rng(seed)
    n, rho = 4000, 0.95
    A = rng.normal(size=(n, 2))
    B = rng.normal(size=(n, 2))
    B[:, 1] = rho * B[:, 0] + np.sqrt(1 - rho**2) * B[:, 1]
    print(f"\n  Two 2-gene clouds with IDENTICAL marginals, differing only in correlation")
    print(f"  (0 versus {rho}). Every gene is standard normal in both.\n")
    print(f"    per-gene std   A {A.std(0).round(3)}   B {B.std(0).round(3)}")
    w1 = np.mean([wasserstein_distance(A[:, g], B[:, g]) for g in range(2)])
    e = energy_u(A, B)
    print(f"\n    mean per-gene 1-Wasserstein (the marginal read) : {w1:.4f}")
    print(f"    energy distance             (the joint read)    : {e:.4f}   ({e / w1:.1f}x larger)")
    print("\n  The marginal metric sees almost nothing: gene by gene the two ARE the same")
    print("  distribution. ‖p − q‖ runs over all genes at once, so energy distance responds")
    print("  to gene-gene correlation and to multimodality. That is the structure a latent")
    print("  flow could represent and a per-gene summary cannot, which is why this is the")
    print("  metric that decides whether the generative machinery earns its keep once the")
    print("  means tie. ssllab.eval.calibration reports both, deliberately.")


def section_4_real(data, perts: list[str], top_k: int, seed: int) -> None:
    print("\n" + "=" * 78)
    print(f"4. The real scale: held-out cells on their top-{top_k} DE genes")
    print("=" * 78)
    is_test = data.split_combo == SPLIT_TEST
    ctrl_all = data.hvg_X[data.is_control.astype(bool)]
    rng = np.random.default_rng(seed)

    print("\n  E_id    control cells vs held-out perturbed cells: how far the perturbation")
    print("          actually moves the cloud. The scale any model is judged against.")
    print("  floor   the perturbed cells split in two at random: what a perfect model would")
    print("          score, given that the target itself is estimated from finite cells.")
    print(f"\n  {'perturbation':<18}{'n_test':>8}{'E_id':>10}{'floor':>10}{'signal':>10}")
    for name in perts:
        if name not in data.de_genes:
            print(f"  {name:<18}{'-':>8}  (no DE record in this cache)")
            continue
        pid, top = pert_index(data, name), top_genes(data, name, top_k)
        pert = data.hvg_X[(data.pert_id == pid) & is_test][:, top]
        if len(pert) < 4:
            print(f"  {name:<18}{len(pert):>8}  (too few cells)")
            continue
        k = min(len(pert), len(ctrl_all))
        ctrl = ctrl_all[rng.choice(len(ctrl_all), size=k, replace=False)][:, top]
        perm = rng.permutation(len(pert))
        h = len(pert) // 2
        e_id = energy_u(ctrl, pert)
        floor = energy_u(pert[perm[:h]], pert[perm[h : 2 * h]])
        print(f"  {name:<18}{len(pert):>8}{e_id:>10.4f}{floor:>10.4f}{e_id - floor:>10.4f}")
    print("\n  The floor is essentially zero even at 81 cells, so the measurement is not")
    print("  floor-limited: the dynamic range between a perfect model and no model at all is")
    print("  the whole of E_id. Note the floor may be slightly negative, which is the")
    print("  unbiased estimator behaving correctly rather than a defect.")


def section_5_eval_sizes(data, name: str, top_k: int, seed: int) -> None:
    print("\n" + "=" * 78)
    print("5. The estimator at the sizes the calibration evaluation uses")
    print("=" * 78)
    if name not in data.de_genes:
        print(f"\n  {name} has no DE record; skipping.")
        return
    pid, top = pert_index(data, name), top_genes(data, name, top_k)
    pert = data.hvg_X[(data.pert_id == pid) & (data.split_combo == SPLIT_TEST)][:, top]
    ctrl_all = data.hvg_X[data.is_control.astype(bool)]
    rng = np.random.default_rng(seed)
    print(f"\n  10_eval_calibration generates 200 cells per perturbation by default and")
    print(f"  compares them to every held-out cell. Standing in control cells for a model's")
    print(f"  output, against {name}'s {len(pert)} held-out cells:\n")
    print(f"    {'n_pred':>8}{'E_V':>10}{'E_U':>10}{'V−U':>9}")
    for n in [200, 500, len(pert)]:
        cs = ctrl_all[rng.choice(len(ctrl_all), size=n, replace=False)][:, top]
        v, u = _v(cs, pert), energy_u(cs, pert)
        print(f"    {n:>8}{v:>10.4f}{u:>10.4f}{v - u:>+9.4f}")
    print("\n  About +0.03 at n=200. The reported transport-flow mean energy is 3.62 and the")
    print("  conditioned VAE's 4.09, so the bias is under 1% of either and does not touch the")
    print("  comparison. It would matter for any reported energy of order 0.01-0.05.")
    print("\n  Worth knowing while reading old reports: the four norman_dec_* calibration")
    print("  reports (mean_energy ~0.039, coverage 0.9997) predate the 2026-07-13 DE gene")
    print("  rewrite and were scored on silent genes. They are not comparable to the above.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--perturbation", action="append", default=[],
                   help="held-out perturbation for section 4; repeatable (default: four of them)")
    p.add_argument("--top-k", type=int, default=20, help="DE genes, as the evaluators use")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=120,
                   help="repetitions for section 2's bias sweep (higher = smoother, slower)")
    p.add_argument("--sections", type=str, default="1,2,3,4,5")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    want = {s.strip() for s in args.sections.split(",")}
    perts = args.perturbation or DEFAULT_PERTS

    if "1" in want:
        section_1_by_hand()
    if "2" in want:
        section_2_diagonal(args.seed, args.reps)
    if "3" in want:
        section_3_joint(args.seed)
    if want & {"4", "5"}:
        data = load_narrow(Path(args.data_dir) / args.artifact, "hvg_X")
        if "4" in want:
            section_4_real(data, perts, args.top_k, args.seed)
        if "5" in want:
            section_5_eval_sizes(data, perts[0], args.top_k, args.seed)
    print()


if __name__ == "__main__":
    main()
