"""Numerical checks of the composition identities the action-operator series relies on.

The series composes two interventions with generators ``A`` and ``B`` in the operator group and reads
their interaction off the Lie bracket ``[A, B] = AB - BA``. Four facts carry that reading, and this
script checks each one on small random generators, where the Baker-Campbell-Hausdorff (BCH) series
converges quickly:

1. Commuting generators compose additively: if ``[A, B] = 0`` then ``exp(A) exp(B) = exp(A + B)``.
2. Averaging the two orderings cancels the first-order bracket: the even part
   ``(log(e^A e^B) + log(e^B e^A)) / 2`` equals ``A + B`` plus double commutators only, while the odd
   part ``(log(e^A e^B) - log(e^B e^A)) / 2`` is ``[A, B] / 2`` to leading order.
3. The symmetric (Strang) product ``exp(A/2) exp(B) exp(A/2)`` reduces to ``exp(A + B)`` when the
   generators commute, and otherwise departs from it only at second order, far less than the naive
   product ``exp(A) exp(B)``.
4. Across random pairs, the bracket norm tracks how far the Strang composite departs from additivity.

Run it directly; each line prints the measured quantity next to what the identity predicts.

    python examples/perturbation_response/docs/action-operator/checks/verify_bch.py
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
D = 6
# Small generators, so BCH converges and orders beyond the second are negligible.
A = 0.15 * rng.standard_normal((D, D))
B = 0.15 * rng.standard_normal((D, D))


def comm(X, Y):
    return X @ Y - Y @ X


# 1. Commuting generators compose additively. A commutes with itself, so use the pair (A, A).
err_commute = np.linalg.norm(expm(A) @ expm(A) - expm(2 * A))
print(f"1. commuting: ||exp(A)exp(A) - exp(2A)||        = {err_commute:.2e}  (expect ~0)")

# 2. The even average of the two orderings cancels the 1/2 [A, B] term; the odd part is that term.
logAB = logm(expm(A) @ expm(B))
logBA = logm(expm(B) @ expm(A))
even = 0.5 * (logAB + logBA)
pred_even = A + B + (1 / 12) * (comm(A, comm(A, B)) + comm(B, comm(B, A)))
print(f"2. even avg vs A+B+double commutators          = {np.linalg.norm(even - pred_even):.2e}  (expect small, higher order)")
print(f"   even avg minus A+B                           = {np.linalg.norm(even - (A + B)):.2e} vs ||1/2[A,B]|| = {0.5 * np.linalg.norm(comm(A, B)):.2e}")
odd = 0.5 * (logAB - logBA)
print(f"   odd part vs 1/2[A,B]                         = {np.linalg.norm(odd - 0.5 * comm(A, B)):.2e}  (expect small)")

# 3. Strang reduces to exp(A + B) for commuting generators and is second-order accurate otherwise.
strang_commute = expm(A / 2) @ expm(A) @ expm(A / 2)
print(f"3. Strang, commuting, vs exp(2A)                = {np.linalg.norm(strang_commute - expm(2 * A)):.2e}  (expect ~0)")
dev_strang = logm(expm(A / 2) @ expm(B) @ expm(A / 2)) - (A + B)
dev_naive = logm(expm(A) @ expm(B)) - (A + B)
print(f"   Strang deviation from A+B                    = {np.linalg.norm(dev_strang):.2e}")
print(f"   naive deviation from A+B                     = {np.linalg.norm(dev_naive):.2e}  (expect larger: it keeps 1/2[A,B])")
print(f"   ratio naive / Strang                         = {np.linalg.norm(dev_naive) / np.linalg.norm(dev_strang):.1f}x")
print(f"   ||1/2[A,B]||, the term Strang removes        = {0.5 * np.linalg.norm(comm(A, B)):.2e}")
print(f"   Strang deviation vs double-commutator scale  = {np.linalg.norm(dev_strang):.2e} vs {np.linalg.norm(comm(A, comm(A, B))):.2e}")

# 4. The bracket norm tracks the Strang composite's departure from additivity across random pairs.
brs, devs = [], []
for _ in range(200):
    X = 0.2 * rng.standard_normal((D, D))
    Y = 0.2 * rng.standard_normal((D, D))
    brs.append(np.linalg.norm(comm(X, Y)))
    devs.append(np.linalg.norm(logm(expm(X / 2) @ expm(Y) @ expm(X / 2)) - (X + Y)))
r = spearmanr(np.array(brs), np.array(devs)).statistic
print(f"4. Spearman(||[A,B]||, Strang non-additivity)    = {r:.3f}  (expect strongly positive)")
