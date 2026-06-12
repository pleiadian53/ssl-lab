"""
context_operator.py
===================

Context-dependent latent operators for action-operator JEPA / GRL.

Implements the conditioned latent dynamics

        f_{theta(c_t)}(z) = exp(M_theta) @ z + b_theta ,
        M_theta = sum_i alpha_i B_i ,        alpha = theta ~ pi_psi(z, c_t)

where {B_i} is a *generator basis* and alpha is the coefficient vector emitted
by a context policy. The basis is the main design decision: it sets what kind
of operator exp(M_theta) is allowed to be.

Object map (to the running notation):
    z        latent state, in R^D
    c        context / intervention covariates at time t
    B_i      generator basis matrices, each (D, D)        <- GeneratorBasis
    alpha    = theta, coefficients over the basis          <- ContextPolicy
    M_theta  = sum_i alpha_i B_i, the flow generator (Lie-algebra element)
    A_theta  = exp(M_theta), the operator (group element)  <- ContextConditionedOperator
    f_theta  z -> A_theta z + b_theta

Four bases, spanning the expressiveness <-> structure dial:
    FreeBasis    : learnable dense matrices            -> GL(D)  (invertible only)
    NamedBasis   : one learnable matrix per labeled    -> GL(D)  (invertible + interpretable)
                   intervention; alpha can *be* c
    SkewBasis    : skew-symmetric generators           -> SO(D)  (orthogonal / norm-preserving)
    SE3Basis     : 3 rotation + 3 translation gens     -> SE(3)  (rigid motion, 4x4 homogeneous)

Design choices worth noting:
  * Near-identity init: policy head is zero-initialized, so alpha = 0 -> M = 0
    -> A = I at start. The operator begins as "do nothing" and training pushes it
    away from identity only as far as the data demands.
  * Stop-gradient on the target lives in the loss, not here (see conditioned_jepa_loss).
  * Invertibility/orthogonality/rigidity are *structural*: they follow from the
    basis + exp(.), never from a penalty.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# --------------------------------------------------------------------------- #
# Generator bases: each returns {B_i} as a tensor of shape (m, D, D).          #
# --------------------------------------------------------------------------- #
class GeneratorBasis(nn.Module):
    """Base class. Subclasses expose `dim` (D), `num_generators` (m), and
    return the basis tensor of shape (m, D, D) from `forward()`."""

    dim: int
    num_generators: int

    def forward(self) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_generators={self.num_generators}"


class FreeBasis(GeneratorBasis):
    """Unconstrained learnable generators. exp(sum alpha_i B_i) lands in GL(D):
    invertible, but otherwise free. The model discovers its own latent modes.

    This is the phenotyping *default* when you do not yet know the intervention
    semantics and want the policy to learn the dynamics modes end-to-end.
    """

    def __init__(self, dim: int, num_generators: int = 16, init_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_generators = num_generators
        # Basis is O(1); near-identity init comes from alpha=0 at the policy head,
        # not from shrinking the basis. init_scale just keeps early M well-conditioned.
        self.B = nn.Parameter(torch.randn(num_generators, dim, dim) * init_scale)

    def forward(self) -> Tensor:
        return self.B


class NamedBasis(GeneratorBasis):
    """One learnable generator per *labeled* intervention. The coefficient
    alpha_i then reads as "how much of intervention i happened", and

        M = sum_i (intervention_amount_i) * B_i .

    Set `alpha = c` directly (DirectInterventionPolicy) to make the quantified
    intervention log *be* theta. The B_i are learned (what each intervention
    does to the latent); inspecting eig(B_i) tells you the dynamics each induces.
    """

    def __init__(self, dim: int, names: list[str], init_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.names = list(names)
        self.num_generators = len(names)
        self.B = nn.Parameter(torch.randn(self.num_generators, dim, dim) * init_scale)

    def forward(self) -> Tensor:
        return self.B

    def extra_repr(self) -> str:
        return f"dim={self.dim}, names={self.names}"


class SkewBasis(GeneratorBasis):
    """Skew-symmetric generators (B_i^T = -B_i). Any combination is skew, so
    exp(M) is orthogonal -> the operator *rotates* the latent without changing
    its norm. Natural for cyclic / circadian latent structure; cannot blow up.

    If `num_generators` is None, uses the full canonical so(D) basis
    {E_ij - E_ji : i < j} (m = D(D-1)/2) as a fixed buffer. Otherwise uses a
    learnable skew-constrained subset of size `num_generators`.
    """

    def __init__(self, dim: int, num_generators: Optional[int] = None,
                 init_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        if num_generators is None:
            basis = self._canonical_so(dim)             # (D(D-1)/2, D, D)
            self.num_generators = basis.shape[0]
            self.register_buffer("B_fixed", basis)
            self._learnable = False
        else:
            self.num_generators = num_generators
            self.raw = nn.Parameter(torch.randn(num_generators, dim, dim) * init_scale)
            self._learnable = True

    @staticmethod
    def _canonical_so(dim: int) -> Tensor:
        gens = []
        for i in range(dim):
            for j in range(i + 1, dim):
                B = torch.zeros(dim, dim)
                B[i, j] = 1.0
                B[j, i] = -1.0
                gens.append(B)
        return torch.stack(gens, dim=0)

    def forward(self) -> Tensor:
        if self._learnable:
            # Skew-symmetrize each raw matrix: B = R - R^T is always skew.
            return self.raw - self.raw.transpose(-1, -2)
        return self.B_fixed


class SE3Basis(GeneratorBasis):
    """The 6 generators of se(3), acting on homogeneous 3D points (x, y, z, 1).
    Coefficients are (omega_x, omega_y, omega_z, t_x, t_y, t_z): rotation
    axis-angle + translation. exp(sum alpha_i G_i) is a 4x4 rigid transform
    [[R, t], [0, 1]] with R in SO(3). This is the protein-frame instance.

    Fixed (non-learnable) basis: the generators are the *structure*, not learned.
    """

    def __init__(self):
        super().__init__()
        self.dim = 4
        self.num_generators = 6
        G = torch.zeros(6, 4, 4)
        # Rotation generators (top-left 3x3 skew).
        G[0, 1, 2], G[0, 2, 1] = -1.0, 1.0   # about x
        G[1, 0, 2], G[1, 2, 0] = 1.0, -1.0   # about y
        G[2, 0, 1], G[2, 1, 0] = -1.0, 1.0   # about z
        # Translation generators (last column, top 3).
        G[3, 0, 3] = 1.0
        G[4, 1, 3] = 1.0
        G[5, 2, 3] = 1.0
        self.register_buffer("G", G)

    def forward(self) -> Tensor:
        return self.G


# --------------------------------------------------------------------------- #
# Context policies: emit theta = alpha (one distribution over Theta per input).#
# --------------------------------------------------------------------------- #
class MLPPolicy(nn.Module):
    """pi_psi(z, c) -> distribution over alpha (= theta).

    Deterministic mode emits a point (Dirac in Delta(Theta)); stochastic mode
    emits a diagonal Gaussian and reparameterized-samples, so gradients flow to
    psi (the "differentiable one-step optimization suffices" regime). The full
    RL machinery attaches later, on top of `log_prob`/`entropy` of this dist.

    The final layer is zero-initialized -> alpha = 0 at start -> near-identity op.
    """

    def __init__(self, z_dim: int, c_dim: int, num_generators: int,
                 hidden: int = 128, stochastic: bool = False):
        super().__init__()
        self.stochastic = stochastic
        self.num_generators = num_generators
        self.trunk = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, num_generators)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        if stochastic:
            self.log_std = nn.Parameter(torch.full((num_generators,), -2.0))

    def distribution(self, z: Tensor, c: Tensor) -> torch.distributions.Distribution:
        h = self.trunk(torch.cat([z, c], dim=-1))
        mu = self.mu_head(h)
        if self.stochastic:
            std = self.log_std.exp().expand_as(mu)
            return torch.distributions.Normal(mu, std)
        # Dirac: represent as a zero-variance Normal so .rsample() == mu.
        return torch.distributions.Normal(mu, torch.zeros_like(mu) + 1e-12)

    def forward(self, z: Tensor, c: Tensor) -> tuple[Tensor, torch.distributions.Distribution]:
        dist = self.distribution(z, c)
        alpha = dist.rsample() if self.stochastic else dist.mean
        return alpha, dist


class DirectInterventionPolicy(nn.Module):
    """theta = alpha = c directly (optionally through a learnable per-generator
    gain). Use with NamedBasis so the quantified intervention vector *is* the
    coefficient vector: M = sum_i c_i * B_i. Requires c_dim == num_generators.
    """

    def __init__(self, num_generators: int, learn_gain: bool = True):
        super().__init__()
        self.num_generators = num_generators
        self.gain = nn.Parameter(torch.ones(num_generators)) if learn_gain \
            else None

    def forward(self, z: Tensor, c: Tensor) -> tuple[Tensor, None]:
        assert c.shape[-1] == self.num_generators, \
            "DirectInterventionPolicy needs c_dim == num_generators"
        alpha = c if self.gain is None else c * self.gain
        return alpha, None


# --------------------------------------------------------------------------- #
# The context-conditioned operator: ties basis + policy into f_{theta(c)}.     #
# --------------------------------------------------------------------------- #
class ContextConditionedOperator(nn.Module):
    """f_{theta(c)}(z) = exp(sum_i alpha_i B_i) @ z + b(c).

    `policy` emits alpha from (z, c); `basis` supplies {B_i}; exp(.) lifts the
    flat generator M into the (curved) operator A. Bias b is an optional affine
    term (off by default; SE(3) carries translation inside A and needs no bias).
    """

    def __init__(self, basis: GeneratorBasis, policy: nn.Module,
                 use_bias: bool = False, c_dim: Optional[int] = None):
        super().__init__()
        self.basis = basis
        self.policy = policy
        self.dim = basis.dim
        if use_bias:
            assert c_dim is not None, "c_dim required when use_bias=True"
            self.bias_head = nn.Linear(c_dim, basis.dim)
            nn.init.zeros_(self.bias_head.weight)
            nn.init.zeros_(self.bias_head.bias)
        else:
            self.bias_head = None

    # -- core pieces, exposed individually so they can be inspected ---------- #
    def coefficients(self, z: Tensor, c: Tensor):
        """Return (alpha, dist). alpha is theta; dist is the Delta(Theta) point."""
        out = self.policy(z, c)
        return out if isinstance(out, tuple) else (out, None)

    def generator(self, alpha: Tensor) -> Tensor:
        """M_theta = sum_i alpha_i B_i, shape (..., D, D)."""
        B = self.basis()                                  # (m, D, D)
        return torch.einsum("...m,mij->...ij", alpha, B)

    def operator_matrix(self, z: Tensor, c: Tensor) -> Tensor:
        """A_theta = exp(M_theta), shape (..., D, D)."""
        alpha, _ = self.coefficients(z, c)
        return torch.matrix_exp(self.generator(alpha))

    def eigenvalues(self, z: Tensor, c: Tensor) -> Tensor:
        """Eigenvalues of the *generator* M (complex). Real part > 0 in any mode
        flags locally growing dynamics -> the "decompensation" read."""
        alpha, _ = self.coefficients(z, c)
        return torch.linalg.eigvals(self.generator(alpha))

    def energy(self, z: Tensor, c: Tensor) -> Tensor:
        """Least-action penalty: squared Frobenius norm of M (how far the
        operator departs from identity). Mean over batch."""
        alpha, _ = self.coefficients(z, c)
        M = self.generator(alpha)
        return (M ** 2).flatten(start_dim=-2).sum(-1).mean()

    # -- forward ------------------------------------------------------------- #
    def forward(self, z: Tensor, c: Tensor, return_aux: bool = False):
        """z' = A_theta z + b(c)."""
        alpha, dist = self.coefficients(z, c)
        A = torch.matrix_exp(self.generator(alpha))        # (..., D, D)
        z_next = torch.matmul(A, z.unsqueeze(-1)).squeeze(-1)
        if self.bias_head is not None:
            z_next = z_next + self.bias_head(c)
        if return_aux:
            return z_next, {"alpha": alpha, "A": A, "dist": dist}
        return z_next

    # -- composition (flow property) ---------------------------------------- #
    @staticmethod
    def compose(A_first: Tensor, A_second: Tensor) -> Tensor:
        """Apply A_first then A_second: A_second @ A_first. Order matters --
        these operators do not commute in general (BCH / [M1, M2])."""
        return torch.matmul(A_second, A_first)


# --------------------------------------------------------------------------- #
# Conditioned-JEPA loss: the single edit g_phi(z, q) -> f_{theta(c)}(z).       #
# --------------------------------------------------------------------------- #
def conditioned_jepa_loss(operator: ContextConditionedOperator,
                          z_t: Tensor, c_t: Tensor, z_target: Tensor) -> Tensor:
    """L = || f_{theta(c_t)}(z_t) - sg(z_target) ||^2, averaged over batch.

    z_target should be E_{ema}(x_{t+1}); the stop-gradient is applied here so no
    gradient reaches the target encoder through this term (anti-collapse).
    """
    z_pred = operator(z_t, c_t)
    return ((z_pred - z_target.detach()) ** 2).sum(-1).mean()


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float = 0.999) -> None:
    """Exponential moving average: bar_xi <- tau*bar_xi + (1-tau)*xi."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(tau).add_(p_o, alpha=1.0 - tau)


# --------------------------------------------------------------------------- #
# Smoke tests: verify the structural guarantees actually hold numerically.     #
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    torch.manual_seed(0)
    B, D = 4, 8

    # (1) Free basis + deterministic policy. Check invertibility and that
    #     zero-init policy => A = I (near-identity start).
    free = ContextConditionedOperator(
        FreeBasis(D, num_generators=12),
        MLPPolicy(z_dim=D, c_dim=3, num_generators=12, stochastic=False),
    )
    z = torch.randn(B, D)
    c = torch.randn(B, 3)
    A = free.operator_matrix(z, c)
    A_inv = torch.matrix_exp(-free.generator(free.coefficients(z, c)[0]))
    I = torch.eye(D).expand(B, D, D)
    assert torch.allclose(A @ A_inv, I, atol=1e-4), "free op not invertible via exp(-M)"
    assert torch.allclose(free(z, c), z, atol=1e-5), "zero-init should give A=I"
    print(f"[free ]  invertible: yes   near-identity init: yes   A.shape={tuple(A.shape)}")

    # (2) Skew basis (learnable subset). Check A is orthogonal (A^T A = I).
    skew = ContextConditionedOperator(
        SkewBasis(D, num_generators=6),
        MLPPolicy(z_dim=D, c_dim=3, num_generators=6, stochastic=True),
    )
    # force nonzero alpha so we're not just testing the identity
    with torch.no_grad():
        skew.policy.mu_head.bias.add_(torch.randn(6) * 0.5)
    A = skew.operator_matrix(z, c)
    orth_err = (A.transpose(-1, -2) @ A - I).abs().max().item()
    norm_pres = (torch.linalg.vector_norm(skew(z, c), dim=-1)
                 - torch.linalg.vector_norm(z, dim=-1)).abs().max().item()
    print(f"[skew ]  orthogonality err={orth_err:.2e}   norm-preservation err={norm_pres:.2e}")
    assert orth_err < 1e-4 and norm_pres < 1e-4

    # (3) Named-intervention basis + DirectInterventionPolicy: alpha = c.
    names = ["sleep", "stress", "meds"]
    named = ContextConditionedOperator(
        NamedBasis(D, names),
        DirectInterventionPolicy(num_generators=len(names)),
    )
    interventions = torch.tensor([[7.0, 0.0, 1.0]]).expand(B, 3)  # hrs sleep, stress, meds
    alpha, _ = named.coefficients(z, interventions)
    assert torch.allclose(alpha, interventions), "DirectIntervention should pass c through (gain=1 init)"
    # Composition: one week of the same daily op = exp(7 M) = (exp M)^7.
    A_day = named.operator_matrix(z, torch.tensor([[1., 0., 1.]]).expand(B, 3))
    A_week = named.operator_matrix(z, torch.tensor([[7., 0., 7.]]).expand(B, 3))
    A_pow = torch.linalg.matrix_power(A_day, 7)
    comp_err = (A_week - A_pow).abs().max().item()
    print(f"[named]  exp(7M) == (exp M)^7 err={comp_err:.2e}   (flow/composition holds)")
    assert torch.allclose(A_week, A_pow, atol=1e-3)

    # (4) SE(3) basis: 4x4 rigid transform. Check R in SO(3), bottom row [0,0,0,1].
    se3 = SE3Basis()
    twist = torch.tensor([[0.3, -0.1, 0.2, 1.0, 2.0, -0.5]])   # (wx,wy,wz, tx,ty,tz)
    M = torch.einsum("...m,mij->...ij", twist, se3())
    T = torch.matrix_exp(M)[0]
    R = T[:3, :3]
    rot_err = (R.T @ R - torch.eye(3)).abs().max().item()
    det_err = abs(torch.det(R).item() - 1.0)
    bottom_ok = torch.allclose(T[3], torch.tensor([0., 0., 0., 1.]), atol=1e-5)
    print(f"[se3  ]  R^T R = I err={rot_err:.2e}   det(R)-1={det_err:.2e}   "
          f"bottom-row ok: {bottom_ok}   t={T[:3,3].tolist()}")
    assert rot_err < 1e-5 and det_err < 1e-5 and bottom_ok

    # (5) End-to-end conditioned-JEPA step: loss is finite and backprops to psi.
    z_target = torch.randn(B, D)
    loss = conditioned_jepa_loss(free, z, c, z_target) + 1e-3 * free.energy(z, c)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum() for p in free.parameters() if p.grad is not None)
    print(f"[jepa ]  loss={loss.item():.4f}   grad flows to policy/basis: {grad_norm.item() > 0}")
    assert math.isfinite(loss.item()) and grad_norm.item() > 0

    # (6) Eigenvalue / decompensation read.
    eig = free.eigenvalues(z, c)
    print(f"[eig  ]  max Re(eig of M) across batch = {eig.real.max().item():+.3f} "
          f"(>0 would flag locally growing dynamics)")

    print("\nall checks passed.")


if __name__ == "__main__":
    _smoke()
