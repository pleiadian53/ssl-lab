"""Tests for the operator-algebra Stage B: the bracket is epistasis, composition is in the group.

The load-bearing properties, each pinned so a refactor cannot silently break the thesis:
  - zero-init => every operator (single, pair, control) is the identity;
  - a single gene's operator is exp(M_g);
  - COMMUTING generators compose additively (A_{AB} = exp(M_A + M_B)) — no epistasis;
  - NON-COMMUTING generators do NOT, and the bracket is what differs;
  - a held-out pair composes from the SAME single-gene generators (generalization);
  - the symmetric product is canonical (deterministic in gene order).
"""

from __future__ import annotations

import torch

from ssllab.generative.condition import build_pert_gene_matrix
from ssllab.generative.operator_algebra import NamedGeneratorOperator

# 'A+C' is held out (unseen): its generators M_A, M_C are trained via the singles / other combos.
PERTS = ["control", "A", "B", "C", "A+B", "B+C", "A+C"]
DIM = 8


def _op(perts=PERTS, dim=DIM):
    M, _ = build_pert_gene_matrix(perts)
    return NamedGeneratorOperator(M, dim=dim), perts


def test_zero_init_is_identity_everywhere():
    op, perts = _op()
    eye = torch.eye(DIM)
    for name in perts:
        A = op.operator(perts.index(name))
        assert torch.allclose(A, eye, atol=1e-6), f"{name} not identity at init"


def test_single_gene_is_matrix_exp():
    op, perts = _op()
    with torch.no_grad():
        op.generators[0].copy_(0.1 * torch.randn(DIM, DIM))     # gene 'A' (vocab index 0)
    A = op.operator(perts.index("A"))
    assert torch.allclose(A, torch.matrix_exp(op.generators[0]), atol=1e-6)


def test_commuting_generators_compose_additively():
    # If M_A and M_B commute, the group product must collapse to exp(M_A + M_B): no epistasis.
    op, perts = _op()
    with torch.no_grad():
        base = 0.1 * torch.randn(DIM, DIM)
        op.generators[0].copy_(base)                 # M_A
        op.generators[1].copy_(2.3 * base)           # M_B = scalar * M_A  => [M_A, M_B] = 0
    a, b = 0, 1
    assert torch.linalg.norm(op.bracket(a, b)) < 1e-6
    A_ab = op.operator(perts.index("A+B"))
    additive = torch.matrix_exp(op.generators[a] + op.generators[b])
    assert torch.allclose(A_ab, additive, atol=1e-5)


def test_noncommuting_generators_are_not_additive():
    # A generic pair does NOT commute; the composed operator departs from the additive one, and the
    # size of the departure is governed by the bracket. This is epistasis living in the algebra.
    op, perts = _op()
    with torch.no_grad():
        op.generators[0].copy_(0.2 * torch.randn(DIM, DIM))
        op.generators[1].copy_(0.2 * torch.randn(DIM, DIM))
    a, b = 0, 1
    assert torch.linalg.norm(op.bracket(a, b)) > 1e-3       # they don't commute
    A_ab = op.operator(perts.index("A+B"))
    additive = torch.matrix_exp(op.generators[a] + op.generators[b])
    assert not torch.allclose(A_ab, additive, atol=1e-3)     # so composition is non-additive


def test_bracket_norm_matches_and_is_antisymmetric():
    op, _ = _op()
    with torch.no_grad():
        op.generators[0].copy_(0.2 * torch.randn(DIM, DIM))
        op.generators[1].copy_(0.2 * torch.randn(DIM, DIM))
    assert torch.allclose(op.bracket(0, 1), -op.bracket(1, 0), atol=1e-6)      # [A,B] = -[B,A]
    assert torch.allclose(op.bracket_norm(PERTS.index("A+B")),
                          torch.linalg.norm(op.bracket(0, 1)), atol=1e-6)
    assert op.bracket_norm(PERTS.index("A")) == 0            # a single gene has no bracket


def test_heldout_pair_composes_from_trained_singles():
    # The generalization property: 'A+C' is unseen, but its operator is a fixed function of M_A and
    # M_C, which are trained. Setting the singles determines the held-out combo with no combo-specific
    # parameter anywhere.
    op, perts = _op()
    with torch.no_grad():
        op.generators[0].copy_(0.15 * torch.randn(DIM, DIM))   # M_A
        op.generators[2].copy_(0.15 * torch.randn(DIM, DIM))   # M_C (vocab index 2)
    a, c = 0, 2
    half = torch.matrix_exp(0.5 * op.generators[a])
    expected = half @ torch.matrix_exp(op.generators[c]) @ half
    assert torch.allclose(op.operator(perts.index("A+C")), expected, atol=1e-6)


def test_composition_reduces_to_additive_in_the_commuting_limit_via_pushforward():
    # End-to-end through pushforward: commuting => pushed cloud equals the additive-operator cloud.
    op, perts = _op()
    with torch.no_grad():
        base = 0.1 * torch.randn(DIM, DIM)
        op.generators[0].copy_(base)
        op.generators[1].copy_(-0.7 * base)          # commutes with M_A
    z = torch.randn(16, DIM)
    got = op.pushforward(z, perts.index("A+B"))
    additive = z @ torch.matrix_exp(op.generators[0] + op.generators[1]).T
    assert got.shape == (16, DIM)
    assert torch.allclose(got, additive, atol=1e-5)


def test_control_and_action_energy():
    op, perts = _op()
    with torch.no_grad():
        op.generators[0].copy_(torch.ones(DIM, DIM))     # M_A: energy = D*D
        op.generators[1].copy_(2 * torch.ones(DIM, DIM)) # M_B: energy = 4*D*D
    assert torch.allclose(op.operator(perts.index("control")), torch.eye(DIM))
    assert op.action_energy(perts.index("control")) == 0
    assert torch.allclose(op.action_energy(perts.index("A")), torch.tensor(float(DIM * DIM)))
    # a pair pays for BOTH its genes
    assert torch.allclose(op.action_energy(perts.index("A+B")),
                          torch.tensor(float(DIM * DIM + 4 * DIM * DIM)))
