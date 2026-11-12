"""Numerical checks for every worked example in ``03a-lie-groups-from-the-ground-up.md``.

The tutorial builds Lie groups, Lie algebras and generators from small concrete matrices. Each
section below re-computes one claim from that chapter and prints the measured value next to what
the chapter says it should be. Nothing here is random except section 9, which is seeded.

    python examples/perturbation_response/docs/action-operator/checks/verify_lie_examples.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, logm

np.set_printoptions(precision=4, suppress=True)


def comm(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Lie bracket of two square matrices: [X, Y] = XY - YX."""
    return X @ Y - Y @ X


def E(i: int, j: int, n: int) -> np.ndarray:
    """Matrix unit: a 1 in row i, column j (1-indexed), zeros elsewhere."""
    out = np.zeros((n, n))
    out[i - 1, j - 1] = 1.0
    return out


def rot(theta: float) -> np.ndarray:
    """2-D rotation by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


I2 = np.eye(2)
J = np.array([[0.0, -1.0], [1.0, 0.0]])  # generator of 2-D rotation

# §1 SO(2) is a group but not a vector space.
a, b = 0.7, 1.9
print("§1  R(a)R(b) - R(a+b)            :", np.linalg.norm(rot(a) @ rot(b) - rot(a + b)), "(expect 0)")
print("§1  R(0) + R(pi)                 :\n", rot(0) + rot(np.pi), "(expect zero matrix: the sum left the group)")

# §2 derivative of R(theta) at 0 is J.
h = 1e-6
dR = (rot(h) - rot(-h)) / (2 * h)
print("§2  dR/dtheta at 0               :\n", dR, "(expect J)")
print("§2  J @ J                        :\n", J @ J, "(expect -I)")

# §3 exp(theta J) = R(theta).
theta = 1.234
print("§3  exp(theta J) - R(theta)      :", np.linalg.norm(expm(theta * J) - rot(theta)), "(expect 0)")
# The series, truncated, converging.
partial, term = np.zeros((2, 2)), np.eye(2)
for n in range(1, 21):
    partial = partial + term
    term = term @ (theta * J) / n
print("§3  20-term series - R(theta)    :", np.linalg.norm(partial - rot(theta)), "(expect ~1e-16)")

# §4 a gallery of 2x2 generators.
shear = np.array([[0.0, 1.0], [0.0, 0.0]])
print("§4  shear: exp(tN) - (I + tN)    :", np.linalg.norm(expm(2.5 * shear) - (I2 + 2.5 * shear)), "(expect 0)")
stretch = np.diag([0.5, -0.3])
print("§4  stretch: exp(diag)           :", np.diag(expm(stretch)), "vs", np.exp([0.5, -0.3]))
aa, bb = -0.2, 0.9
spiral = aa * I2 + bb * J
print("§4  spiral: exp(aI+bJ) - e^a R(b):", np.linalg.norm(expm(spiral) - np.exp(aa) * rot(bb)), "(expect 0)")
print("§4  eig(spiral) =", np.linalg.eigvals(spiral), "  |eig(exp)| =", np.abs(np.linalg.eigvals(expm(spiral))),
      " e^a =", np.exp(aa))

# §5 so(3): non-commuting rotations.
Lx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
Ly = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
Lz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
print("§5  [Lx,Ly] - Lz                 :", np.linalg.norm(comm(Lx, Ly) - Lz), "(expect 0)")
print("§5  [Ly,Lz] - Lx                 :", np.linalg.norm(comm(Ly, Lz) - Lx), "(expect 0)")
print("§5  [Lz,Lx] - Ly                 :", np.linalg.norm(comm(Lz, Lx) - Ly), "(expect 0)")
q = np.pi / 2
xy = expm(q * Ly) @ expm(q * Lx)  # x first, then y
yx = expm(q * Lx) @ expm(q * Ly)  # y first, then x
print("§5  90deg book: x-then-y vs y-then-x differ by", np.linalg.norm(xy - yx), "(expect > 0)")

# §5 group commutator ~ I + eps^2 [X, Y].
for eps in (1e-1, 1e-2, 1e-3):
    g = expm(eps * Lx) @ expm(eps * Ly) @ expm(-eps * Lx) @ expm(-eps * Ly)
    print(f"§5  eps={eps:g}: ||(g - I)/eps^2 - [Lx,Ly]|| = {np.linalg.norm((g - np.eye(3)) / eps**2 - Lz):.2e}"
          " (expect -> 0 linearly in eps)")


# §5 hat map: R^3 with cross product.
def hat(v: np.ndarray) -> np.ndarray:
    return v[0] * Lx + v[1] * Ly + v[2] * Lz


v, w = np.array([1.0, 2.0, -0.5]), np.array([0.3, -1.0, 2.0])
print("§5  hat(v) w - v x w             :", np.linalg.norm(hat(v) @ w - np.cross(v, w)), "(expect 0)")
print("§5  [hat v, hat w] - hat(v x w)  :", np.linalg.norm(comm(hat(v), hat(w)) - hat(np.cross(v, w))), "(expect 0)")

# §6 axioms: antisymmetry and Jacobi on so(3) and on random matrices.
rng = np.random.default_rng(0)
X, Y, Z = (rng.standard_normal((4, 4)) for _ in range(3))
jac = comm(X, comm(Y, Z)) + comm(Y, comm(Z, X)) + comm(Z, comm(X, Y))
print("§6  Jacobi residual (random 4x4) :", np.linalg.norm(jac), "(expect ~1e-14)")
print("§6  [X,Y] + [Y,X]                :", np.linalg.norm(comm(X, Y) + comm(Y, X)), "(expect 0)")

# §6 closure: skew stays skew; symmetric does not stay symmetric.
K1, K2 = X - X.T, Y - Y.T
C = comm(K1, K2)
print("§6  [skew,skew] is skew          :", np.linalg.norm(C + C.T), "(expect 0)")
S1 = np.array([[1.0, 0.0], [0.0, -1.0]])
S2 = np.array([[0.0, 1.0], [1.0, 0.0]])
print("§6  [S1,S2] (two symmetric)      :\n", comm(S1, S2), "(expect [[0,2],[-2,0]]: skew, a rotation)")
P = expm(S1) @ expm(S2)
print("§6  exp(S1)exp(S2) symmetric?    : asym part norm =", np.linalg.norm(P - P.T), "(expect > 0)")

# §7 Heisenberg: BCH terminates exactly.
Xh, Yh = E(1, 2, 3), E(2, 3, 3)
Zh = comm(Xh, Yh)
print("§7  [X,Y] = E13?                 :", np.allclose(Zh, E(1, 3, 3)))
print("§7  [X,Z], [Y,Z] zero?           :", np.allclose(comm(Xh, Zh), 0), np.allclose(comm(Yh, Zh), 0))
lhs = expm(Xh) @ expm(Yh)
rhs = expm(Xh + Yh + 0.5 * Zh)
print("§7  e^X e^Y - e^(X+Y+[X,Y]/2)    :", np.linalg.norm(lhs - rhs), "(expect 0 exactly)")
print("§7  e^X e^Y - e^(X+Y)            :", np.linalg.norm(lhs - expm(Xh + Yh)), "(expect 0.5)")

# §7 sl(2).
H, Ep, F = np.diag([1.0, -1.0]), E(1, 2, 2), E(2, 1, 2)
print("§7  [H,E]-2E, [H,F]+2F, [E,F]-H  :",
      np.linalg.norm(comm(H, Ep) - 2 * Ep), np.linalg.norm(comm(H, F) + 2 * F), np.linalg.norm(comm(Ep, F) - H))
M = rng.standard_normal((3, 3))
print("§7  det(e^M) - e^(tr M)          :", abs(np.linalg.det(expm(M)) - np.exp(np.trace(M))), "(expect ~0)")

# §7 se(2): parallel parking.
Rg = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)  # turn in place
Tg = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)   # drive forward along x
print("§7  [R,T]                        :\n", comm(Rg, Tg), "(expect E23 = slide along y)")
eps = 1e-2
g = expm(eps * Rg) @ expm(eps * Tg) @ expm(-eps * Rg) @ expm(-eps * Tg)
print("§7  parking manoeuvre displacement (x, y)/eps^2 :", g[:2, 2] / eps**2, "(expect ~ (0, 1))")

# §8 BCH to second order, and the symmetric product.
X, Y = 0.05 * rng.standard_normal((4, 4)), 0.05 * rng.standard_normal((4, 4))
Zab = logm(expm(X) @ expm(Y)).real
print("§8  ||log(e^X e^Y) - (X+Y)||           :", f"{np.linalg.norm(Zab - (X + Y)):.2e}", "(second order)")
print("§8  ||log(e^X e^Y) - (X+Y+[X,Y]/2)||   :", f"{np.linalg.norm(Zab - (X + Y + 0.5 * comm(X, Y))):.2e}",
      "(third order: much smaller)")
Zs = logm(expm(X / 2) @ expm(Y) @ expm(X / 2)).real
print("§8  ||log(e^(X/2) e^Y e^(X/2)) - (X+Y)||:", f"{np.linalg.norm(Zs - (X + Y)):.2e}",
      "(third order: the 1/2[X,Y] term cancelled)")
pred3 = -comm(X, comm(X, Y)) / 24 - comm(Y, comm(X, Y)) / 12
print("§8  ... minus (-[X,[X,Y]]/24 - [Y,[X,Y]]/12):", f"{np.linalg.norm(Zs - (X + Y) - pred3):.2e}",
      "(fifth order: tiny)")

# §9 the exponential is neither injective nor surjective.
print("§9  exp(2 pi J) - I              :", np.linalg.norm(expm(2 * np.pi * J) - I2), "(expect 0: two generators, one operator)")
Jb = np.array([[-1.0, 1.0], [0.0, -1.0]])
print("§9  det of [[-1,1],[0,-1]]       :", np.linalg.det(Jb), "(positive, yet no real logarithm exists)")
Lb = logm(Jb)
print("§9  logm returns a real matrix?  :", np.allclose(Lb.imag, 0), "(expect False)")
