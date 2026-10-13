"""Measure where the bracket description of an interaction stops being valid.

The interaction of two composed interventions is the generator of the composite minus the sum of the
single generators. The Baker-Campbell-Hausdorff (BCH) series writes that difference as nested brackets,
but BCH is a series with a radius of convergence. This script draws random generator pairs at a fixed
norm and, for two composites, compares the true interaction against its second-order bracket prediction:

    Strang         log(e^{A/2} e^B e^{A/2})            = A + B - (1/24)[A,[A,B]] + (1/12)[B,[B,A]] + ...
    even average   (log(e^A e^B) + log(e^B e^A)) / 2   = A + B + (1/12)([A,[A,B]] + [B,[B,A]]) + ...

The two composites are different objects with different second-order coefficients. For each norm the
table reports, as fractions of the composite's norm, the error of assuming additivity ("add"), the error
after the second-order bracket correction ("+2nd ord"), and their ratio ("gain"). A gain below 1 means
the bracket correction is worse than assuming the interventions do not interact at all. Chapter 5 of the
action-operator series reports the Strang columns.

    python examples/perturbation_response/docs/action-operator/checks/verify_bch_radius.py
"""

import numpy as np
from scipy.linalg import expm, logm

rng = np.random.default_rng(0)
D = 6


def comm(X, Y):
    return X @ Y - Y @ X


def errs(s, n=12):
    out = {"add_s": [], "bch_s": [], "add_e": [], "bch_e": []}
    for _ in range(n):
        A = rng.standard_normal((D, D))
        A *= s / np.linalg.norm(A)
        B = rng.standard_normal((D, D))
        B *= s / np.linalg.norm(B)
        AAB, BBA = comm(A, comm(A, B)), comm(B, comm(B, A))
        # Strang composite
        L = np.real(logm(expm(A / 2) @ expm(B) @ expm(A / 2)))
        if np.isfinite(L).all():
            n_ = np.linalg.norm(L)
            out["add_s"].append(np.linalg.norm(L - (A + B)) / n_)
            out["bch_s"].append(np.linalg.norm(L - (A + B - AAB / 24 + BBA / 12)) / n_)
        # Even average of the two orderings
        L1 = np.real(logm(expm(A) @ expm(B)))
        L2 = np.real(logm(expm(B) @ expm(A)))
        if np.isfinite(L1).all() and np.isfinite(L2).all():
            E = 0.5 * (L1 + L2)
            n_ = np.linalg.norm(E)
            out["add_e"].append(np.linalg.norm(E - (A + B)) / n_)
            out["bch_e"].append(np.linalg.norm(E - (A + B + (AAB + BBA) / 12)) / n_)
    return {k: (np.mean(v) if v else float("nan")) for k, v in out.items()}


print(f"{'||A||=||B||':>11} | {'STRANG add':>10} {'+2nd ord':>10} {'gain':>6} | {'EVEN add':>9} {'+2nd ord':>10} {'gain':>6}")
print("-" * 74)
for s in [0.1, 0.25, 0.5, 0.693, 1.0, 1.1, 2.0, 5.0, 12.4]:
    e = errs(s)
    gs = e["add_s"] / e["bch_s"] if e["bch_s"] else float("nan")
    ge = e["add_e"] / e["bch_e"] if e["bch_e"] else float("nan")
    print(f"{s:11.3f} | {e['add_s']:10.5f} {e['bch_s']:10.5f} {gs:5.1f}x | {e['add_e']:9.5f} {e['bch_e']:10.5f} {ge:5.1f}x")
