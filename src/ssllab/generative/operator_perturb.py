"""Stage B, the action-operator variant: model the *transition*, not the destination.

The flow of ``flow.py`` learns a free velocity field and samples an outcome latent. This
module learns the **operator** that carries a control cell to its perturbed counterpart, and
the perturbation's effect falls out as the operator's departure from doing nothing
(see the conditional-flow-jepa chapter 7).

The construction, in the order the symbols appear:

    c        = e(p)                        the perturbation embedding (gene-set, so unseen
                                           combinations compose from their single-gene parts)
    alpha    = pi(c)          in R^m       coefficients, ONE vector per perturbation
    M        = sum_i alpha_i B_i           the flow generator, a D x D matrix
    A        = exp(M)                      the operator; exp is the matrix exponential
    z'       = A z_b                       the pushforward of a control latent z_b

Three properties are baked into the parameterization rather than hoped for.

**Near-identity start.** ``pi`` is zero-initialized, so ``alpha = 0``, ``M = 0``, and
``A = exp(0) = I``. The operator begins at "this perturbation does nothing" and has to earn
every departure from identity. That is the transition-side twin of the decoder's
identity-anchored mean head, and it encodes the same fact about the data: effects are small
shifts on a large, intervention-independent baseline.

**Alpha depends on the CONDITION, not on the cell.** ``pi`` reads ``c`` alone, so every cell
receiving perturbation ``p`` is transported by the *same* operator ``A_p``. This is the right
model (an operator is a property of the intervention, not of the cell it acts on) and it is
also what makes it cheap: one ``matrix_exp`` on a D x D matrix per perturbation, not per cell.
Epistasis still has somewhere to live, because ``e(A+B)`` need not equal ``e(A) + e(B)``, so
``alpha(e(p))`` is free to be non-additive even though the basis expansion is linear.

**Cells come unpaired, so we match distributions rather than pairs.** Sequencing destroys the
cell, so there is no "same cell before and after" and the per-pair equivariance loss cannot be
computed at all. Instead we push the whole control cloud through ``A_p`` and match its
*marginal* against the real perturbed cloud with an **energy distance**, which needs no
correspondence between the two sets. The paired case is recovered automatically the moment
pairs exist, because per-pair MSE is the special case of distribution matching where the
coupling is given.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ssllab.generative.condition import GeneSetEmbedding


def energy_distance(x: Tensor, y: Tensor) -> Tensor:
    """Sample energy distance between two point clouds. No pairing required.

    ``E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||``, zero (in the population limit) exactly
    when the two clouds share a distribution and positive otherwise. In words: twice the
    average distance *between* the clouds, minus the average spread *within* each of them.

    Args:
        x: ``(n, D)`` the PREDICTED cloud. Gradients flow here, so minimizing this trains the
            operator.
        y: ``(m, D)`` the OBSERVED cloud. A fixed target; ``n`` need not equal ``m``.
    """
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).mean()
    d_yy = torch.cdist(y, y).mean()
    return 2.0 * d_xy - d_xx - d_yy


class PerturbationOperator(nn.Module):
    """A_p = exp(sum_i alpha_i(e(p)) B_i), acting on frozen cell latents.

    Args:
        pert_gene: ``(n_perts, n_genes)`` multi-hot matrix mapping a perturbation to its target
            genes. This is what lets a held-out combination compose from single-gene parts.
        dim: the latent dimension D.
        num_generators: the size ``m`` of the learned basis ``{B_i}``. Small on purpose: the
            operator is a low-rank object, and the Frobenius penalty below is the least-action
            prior that keeps it near identity.
        gene_dim, cond_dim: widths of the gene-set condition encoder.
        stochastic: if True, ``alpha`` is Gaussian rather than a point, so the perturbation
            induces a *mixture* of operators. A deterministic operator is invertible, hence a
            diffeomorphism, hence it preserves the number of modes of the cloud it transports.
            It can rotate, scale and shear the control cloud but it cannot split it, and it
            barely widens it. A distribution over ``alpha`` is the cheapest way to let the
            pushforward carry more spread than the control cloud it started from, which is what
            the calibration axis needs.
        residual_scale: if > 0, add a learned per-condition residual displacement with this
            initial scale. This is the "drift plus residual" of chapter 7 in its simplest form:
            the operator carries the structured part of the effect and the residual supplies
            what a linear map cannot.
    """

    def __init__(
        self,
        pert_gene: Tensor,
        dim: int = 256,
        num_generators: int = 16,
        gene_dim: int = 64,
        cond_dim: int = 128,
        compose: str = "additive",
        stochastic: bool = False,
        residual_scale: float = 0.0,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_generators = num_generators
        self.stochastic = stochastic

        self.cond = GeneSetEmbedding(pert_gene, gene_dim=gene_dim, out_dim=cond_dim, compose=compose)

        # The learned generator basis {B_i}: m matrices of shape (D, D).
        self.basis = nn.Parameter(torch.randn(num_generators, dim, dim) * (1.0 / dim ** 0.5))

        # The policy pi: c -> alpha. Zero-initialized on the last layer, so alpha = 0 and A = I
        # at the start of training. The operator must earn every departure from doing nothing.
        self.policy = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, num_generators),
        )
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)

        # Per-condition log-scale of the alpha distribution (only used when stochastic).
        self.log_sigma = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, num_generators),
        ) if stochastic else None
        if stochastic:
            nn.init.zeros_(self.log_sigma[-1].weight)
            nn.init.constant_(self.log_sigma[-1].bias, -3.0)     # start near-deterministic

        self.residual = None
        if residual_scale > 0:
            self.residual = nn.Sequential(
                nn.Linear(cond_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, dim),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        self.residual_scale = residual_scale

    # ---- the operator, per perturbation ---------------------------------------------- #
    def coefficients(self, pert_id: Tensor, n: int | None = None) -> Tensor:
        """alpha for one perturbation. ``(m,)`` deterministic, or ``(n, m)`` if stochastic."""
        c = self.cond(pert_id.view(1))                                   # (1, cond_dim)
        mu = self.policy(c).squeeze(0)                                   # (m,)
        if not self.stochastic or n is None:
            return mu
        sigma = self.log_sigma(c).squeeze(0).exp()                       # (m,)
        return mu + sigma * torch.randn(n, self.num_generators, device=mu.device)

    def generator(self, alpha: Tensor) -> Tensor:
        """M = sum_i alpha_i B_i. ``(D, D)`` or ``(n, D, D)``."""
        return torch.einsum("...m,mij->...ij", alpha, self.basis)

    def pushforward(self, z_ctrl: Tensor, pert_id: Tensor) -> Tensor:
        """Transport a cloud of control latents through this perturbation's operator.

        ``z_ctrl`` ``(n, D)``. Returns ``(n, D)``, the predicted perturbed cloud.
        """
        n = z_ctrl.shape[0]
        if self.stochastic:
            # One operator per cell, drawn from the condition's distribution over alpha:
            # a mixture of operators, which can broaden the cloud.
            alpha = self.coefficients(pert_id, n=n)                      # (n, m)
            M = self.generator(alpha)                                    # (n, D, D)
            A = torch.matrix_exp(M)                                      # (n, D, D)
            z_next = torch.bmm(A, z_ctrl.unsqueeze(-1)).squeeze(-1)
        else:
            # ONE operator shared by every cell of this perturbation: a single matrix_exp.
            alpha = self.coefficients(pert_id)                           # (m,)
            A = torch.matrix_exp(self.generator(alpha))                  # (D, D)
            z_next = z_ctrl @ A.transpose(-1, -2)
        if self.residual is not None:
            c = self.cond(pert_id.view(1))
            z_next = z_next + self.residual_scale * self.residual(c)     # broadcast (1, D)
        return z_next

    def action_energy(self, pert_id: Tensor) -> Tensor:
        """Least-action penalty: squared Frobenius norm of M, i.e. how far A departs from I.

        This is the prior that says a perturbation is a *small* structured change to the cell,
        not an arbitrary remapping of latent space. It is what keeps baseline dominance in the
        parameterization rather than in a hope.
        """
        M = self.generator(self.coefficients(pert_id))
        return (M ** 2).sum()

    @torch.no_grad()
    def operator_matrix(self, pert_id: Tensor) -> Tensor:
        """``A_p`` itself, for inspection (eigenvalues, departure from identity)."""
        return torch.matrix_exp(self.generator(self.coefficients(pert_id)))
