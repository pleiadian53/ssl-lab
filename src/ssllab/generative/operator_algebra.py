"""Stage B, the operator-*algebra* variant: compose in the group, so the bracket is epistasis.

Round 3's action operator (:mod:`ssllab.generative.operator_perturb`) tied the flow it was meant to
replace, and the reason was structural: it applied ``exp`` exactly once, to a single generator, and it
composed two genes in the *additive gene-set embedding* (the same object the NB-VAE uses), never in the
operator algebra. The matrix exponential was a reparameterization, not a capability. See
``dev/planning/action_operator/03-the-operator-algebra-composition-and-epistasis.md``.

This module gives the operator its algebra back. Each single gene ``g`` owns a generator ``M_g``, a
``D x D`` matrix, and its operator is ``A_g = exp(M_g)``. A combination composes *in the group*:

    A_{A+B} = exp(M_A / 2) exp(M_B) exp(M_A / 2)          (symmetric / Strang product)

read against a canonical gene ordering so it is deterministic. Two facts about this product carry the
whole idea, and both are proved in ``dev/planning/action_operator/verify_bch.py``.

**Commuting generators compose additively.** If ``[M_A, M_B] = M_A M_B - M_B M_A = 0`` then the product
collapses to ``exp(M_A + M_B)``: the double is exactly the sum of the singles, i.e. *no interaction*. So
non-commutativity is precisely the departure from additivity, and departure from additivity is what a
geneticist calls **epistasis**. A pair is epistatic iff its single-gene generators do not commute, and
the strength is governed by ``||[M_A, M_B]||``. The NB-VAE has no generators, hence no bracket, hence no
version of this statement to make.

**A simultaneous experiment sees the bracket's magnitude, not its sign.** The perturbation is delivered
with both guides at once, so the observable has no order. The first-order term that *would* distinguish
orderings, ``(1/2)[M_A, M_B]``, is odd under swapping A and B; the symmetric product cancels it, leaving
an even double-commutator whose size scales with ``||[M_A, M_B]||``. Ordered composition (the odd part,
= path-dependence of a plan) becomes observable only in a temporal world model, not here.

The near-identity start and the least-action prior are unchanged from round 3: generators are
zero-initialized so every ``A_g = exp(0) = I`` (the perturbation begins by doing nothing), and a
Frobenius penalty on the generators keeps effects small departures from the baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# The unpaired marginal-matching loss is identical to round 3; reuse it rather than fork it.
from ssllab.generative.operator_perturb import energy_distance

__all__ = ["NamedGeneratorOperator", "energy_distance"]


class NamedGeneratorOperator(nn.Module):
    """One dense generator ``M_g`` per single gene; combinations compose in the group.

    Args:
        pert_gene: ``(n_perts, n_genes)`` multi-hot target-gene matrix from
            :func:`ssllab.generative.condition.build_pert_gene_matrix`. Row ``p`` marks the genes that
            perturbation ``p`` targets (one for a single, two for a Norman pair, none for control). The
            columns are the generator index, and because the gene vocabulary is *sorted*, the ascending
            column order is the canonical gene ordering the symmetric product reads against.
        dim: the latent dimension ``D``.

    The generators are the only parameters. There is no policy network and no basis: gene ``g``'s
    generator is learned directly, and a combination's operator is a fixed algebraic function of its
    genes' generators. That is the point. A held-out combination composes from single-gene generators
    the model actually trained, with the interaction supplied by their non-commutativity rather than by
    a black-box map on an embedding.
    """

    def __init__(self, pert_gene: Tensor, dim: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("pert_gene", (pert_gene > 0).float())
        n_genes = self.pert_gene.shape[1]
        # Dense per-gene generators, zero-initialized so A_g = exp(0) = I at the start of training.
        self.generators = nn.Parameter(torch.zeros(n_genes, dim, dim))

    # ---- gene bookkeeping ------------------------------------------------------------- #
    def target_genes(self, pert_id: int | Tensor) -> list[int]:
        """The generator indices a perturbation targets, in canonical (ascending) order.

        Ascending index is the sorted-vocabulary order, which is the canonical ordering the symmetric
        product uses, so composition is deterministic regardless of how a pair name was written.
        """
        pid = int(pert_id)
        return torch.nonzero(self.pert_gene[pid], as_tuple=False).flatten().sort().values.tolist()

    # ---- the operator, per perturbation ----------------------------------------------- #
    def operator(self, pert_id: int | Tensor) -> Tensor:
        """``A_p`` for one perturbation. ``(D, D)``.

        Control (no genes) is the identity; a single gene is ``exp(M_g)``; a pair is the symmetric
        Strang product ``exp(M_a/2) exp(M_b) exp(M_a/2)`` over the two genes in canonical order, which
        reduces to ``exp(M_a + M_b)`` exactly when the generators commute.
        """
        genes = self.target_genes(pert_id)
        if len(genes) == 0:
            return torch.eye(self.dim, device=self.generators.device, dtype=self.generators.dtype)
        if len(genes) == 1:
            return torch.matrix_exp(self.generators[genes[0]])
        if len(genes) == 2:
            a, b = genes
            half = torch.matrix_exp(0.5 * self.generators[a])
            return half @ torch.matrix_exp(self.generators[b]) @ half
        # Norman is singles and pairs only; a >2-gene perturbation would need the symmetric product
        # generalized (recursive Strang). Refuse rather than silently mis-compose.
        raise NotImplementedError(
            f"perturbation targets {len(genes)} genes; symmetric composition is implemented for <=2"
        )

    def pushforward(self, z_ctrl: Tensor, pert_id: int | Tensor) -> Tensor:
        """Transport a cloud of control latents through this perturbation's operator.

        ``z_ctrl`` ``(n, D)``. One operator, shared by every cell of the perturbation (the operator is a
        property of the intervention, not the cell). Returns ``(n, D)``.
        """
        A = self.operator(pert_id)
        return z_ctrl @ A.transpose(-1, -2)

    # ---- the algebra, for training penalty and for the epistasis claim ---------------- #
    def bracket(self, gene_a: int, gene_b: int) -> Tensor:
        """The Lie bracket ``[M_A, M_B] = M_A M_B - M_B M_A``. ``(D, D)``."""
        ma, mb = self.generators[gene_a], self.generators[gene_b]
        return ma @ mb - mb @ ma

    def bracket_norm(self, pert_id: int | Tensor) -> Tensor:
        """``||[M_A, M_B]||_F`` for a two-gene perturbation; ``0`` for a single or control.

        This is the model-side quantity the round's primary endpoint correlates against the pair's
        empirical genetic interaction (``15_empirical_epistasis.py``).
        """
        genes = self.target_genes(pert_id)
        if len(genes) != 2:
            return self.generators.new_zeros(())
        return torch.linalg.norm(self.bracket(genes[0], genes[1]))

    def action_energy(self, pert_id: int | Tensor) -> Tensor:
        """Least-action penalty: summed squared Frobenius norm of the involved generators.

        The prior that a perturbation is a *small* structured change to the cell. Applied to the genes a
        perturbation targets, so composing two genes pays for both, which keeps each single-gene
        generator near identity unless its own and its partners' data demand a departure.
        """
        genes = self.target_genes(pert_id)
        if not genes:
            return self.generators.new_zeros(())
        return sum((self.generators[g] ** 2).sum() for g in genes)

    @torch.no_grad()
    def operator_matrix(self, pert_id: int | Tensor) -> Tensor:
        """``A_p`` itself, for inspection (eigenvalues, departure from identity)."""
        return self.operator(pert_id)
