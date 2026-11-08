"""Magnitude and interaction metrics for two-gene combinations.

The effect-size benchmark (:mod:`ssllab.eval.effect_size`) scores a prediction by the
Pearson correlation between predicted and measured change on a perturbation's top-DE
genes. A correlation ignores scale: a prediction with the right shape and the wrong size
scores perfectly. For combinations this hides the very thing a model of interactions
should get right, because the additive reference (the sum of the two singles' measured
effects) already has nearly the right shape. This module scores the size of the change
as well, and reads it against additivity.

Notation, all on one pair ``p = A+B`` and restricted to its top-DE genes ``T``, in the
per-gene normalized space the cache stores (log1p-CP10K):

- ``Δ_p``  = mean over the pair's held-out cells − mean over control cells (the truth)
- ``Δ̂_p``  = the model's predicted mean − the control mean (the prediction)
- ``Δ_A``, ``Δ_B`` = each single's measured effect, from its *training* cells
- ``I_p  = Δ_p − Δ_A − Δ_B``   the measured interaction (departure from additivity)
- ``Î_p  = Δ̂_p − Δ_A − Δ_B``   the predicted interaction, against the measured singles

Because ``Î_p`` and ``I_p`` subtract the same measured singles, ``Î_p − I_p = Δ̂_p − Δ_p``:
once size counts, the error on the interaction *is* the error on the pair. What changes is
the yardstick. The additive reference predicts ``Î_p = 0``, so its error is exactly
``‖I_p‖``, and the natural score is the ratio

    r_p = ‖Δ̂_p − Δ_p‖ / ‖I_p‖ = e_p(model) / e_p(additive),

where ``e_p = ‖Δ̂_p − Δ_p‖ / ‖Δ_p‖`` is the relative error proposed for Round 8 and
``e_p(additive) = ‖I_p‖ / ‖Δ_p‖`` is Round 4's rel_GI. ``r_p < 1`` means the model is
closer to the truth than additivity on that pair; ``r_p = 1`` means no closer. It is
summarized on the log scale: the denominator cancels in any contrast between two models,
so a pair whose interaction is tiny (and whose ratio is therefore large for everyone)
cannot dominate a comparison, and a difference of mean logs reads as a relative change
in error.

Round 4's additive scale is read the same way for a prediction. ``λ_p = ⟨Δ_p, a_p⟩ / ‖a_p‖²``
with ``a_p = Δ_A + Δ_B`` is the least-squares coefficient of the truth on the additive
direction (above 1 super-additive, below 1 sub-additive); ``s_p = ⟨Δ̂_p, a_p⟩ / ‖a_p‖²`` is the
same projection of the prediction, so ``s_p`` against ``λ_p`` asks whether a model knows
which pairs do more, or less, than their singles combined. The additive reference has
``s_p = 1`` by construction.

Two quantities need a model's *own* single predictions ``Δ̂_A``, ``Δ̂_B``. The
model-internal interaction ``Δ̂_p − Δ̂_A − Δ̂_B`` is the only one where the subtraction
does not cancel, so it is the only measure that isolates what the model's composition
of two genes adds beyond its own singles.

Fixed numerical rules (the composition-ablation protocol freezes these; change them
only with a new protocol):

- ``NUMERATOR_FLOOR``: an error norm below it is raised to it before the log, so a
  perfect prediction gives a large negative but finite log ratio instead of ``-inf``.
- ``DENOMINATOR_REL_FLOOR``: if ``‖I_p‖ ≤ DENOMINATOR_REL_FLOOR · ‖Δ_p‖`` the measured
  interaction is numerically zero, ratios against it are undefined, and ``r_p``,
  ``log r_p`` and the internal log ratio are NaN (the pair then leaves those metrics).
- ``‖Δ_p‖`` below ``NUMERATOR_FLOOR`` makes ``e_p`` NaN (no measured change to scale by).
- Correlations use :func:`ssllab.eval.effect_size.pearson`, which is NaN when either
  vector is constant (standard deviation below 1e-12), e.g. the additive reference's
  predicted interaction, which is exactly zero.
"""

from __future__ import annotations

import math

import numpy as np

from ssllab.eval.effect_size import pearson

NUMERATOR_FLOOR = 1e-12
DENOMINATOR_REL_FLOOR = 1e-6


def _vec(x) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float64)


def _log_ratio(num: float, den: float) -> float:
    return math.log(max(num, NUMERATOR_FLOOR)) - math.log(den)


def measured_single_deltas(
    genes, *, hvg_X: np.ndarray, pert_names, pert_id: np.ndarray, is_train: np.ndarray,
    control_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    """Each single gene's measured effect ``Δ_g`` (G,), from its training cells.

    The same estimate the additive reference uses (:func:`ssllab.eval.baselines.additive_predictor`):
    mean over the single-``g`` cells on the training side of the split, minus the control
    mean. Raises if a gene has no single perturbation or no training cells, because the
    metrics would otherwise silently describe a different reference.
    """
    names = np.asarray(pert_names)
    index = {str(n): i for i, n in enumerate(names)}
    is_train = np.asarray(is_train, dtype=bool)
    ctrl = np.asarray(control_mean, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for g in genes:
        if g not in index:
            raise KeyError(f"no single perturbation named {g!r}")
        mask = (pert_id == index[g]) & is_train
        if not mask.any():
            raise ValueError(f"single {g!r} has no training cells")
        out[g] = hvg_X[mask].mean(0).astype(np.float64) - ctrl
    return out


def pair_metrics(
    *, pred_pair, true_pair, control_mean, delta_a, delta_b, top_idx,
    pred_a=None, pred_b=None,
) -> dict[str, float]:
    """All per-pair metrics for one combination.

    ``pred_pair``, ``true_pair`` and ``control_mean`` are mean expression profiles (G,);
    ``delta_a``, ``delta_b`` are the measured single effects (G,) from
    :func:`measured_single_deltas`; ``top_idx`` are the pair's top-DE gene indices.
    ``pred_a`` and ``pred_b`` are the model's own predicted single profiles (G,), optional;
    without them the model-internal metrics are NaN.

    Returns ``log_r``, ``r``, ``rel_err`` (Round 8's ``e_p``), ``rel_gi`` (``‖I_p‖/‖Δ_p‖``,
    the additive reference's ``e_p``), ``interaction_r``, ``delta_r`` (the standing
    benchmark, recomputed), ``internal_r`` and ``internal_log_r``, the additive-direction
    scales ``scale`` (``s_p``, the prediction's) and ``lam`` (``λ_p``, the truth's), plus the
    norms they are built from.
    """
    idx = np.asarray(top_idx, dtype=int)
    ctrl = _vec(control_mean)[idx]
    d_hat = _vec(pred_pair)[idx] - ctrl
    d = _vec(true_pair)[idx] - ctrl
    s = _vec(delta_a)[idx] + _vec(delta_b)[idx]
    inter, inter_hat = d - s, d_hat - s

    err = float(np.linalg.norm(d_hat - d))
    n_delta = float(np.linalg.norm(d))
    n_inter = float(np.linalg.norm(inter))
    ratio_ok = n_inter > DENOMINATOR_REL_FLOOR * n_delta and n_inter > 0.0
    nan = float("nan")
    ss = float(s @ s)

    out = {
        "err_norm": err,
        "delta_norm": n_delta,
        "interaction_norm": n_inter,
        "rel_err": err / n_delta if n_delta >= NUMERATOR_FLOOR else nan,
        "rel_gi": n_inter / n_delta if n_delta >= NUMERATOR_FLOOR else nan,
        "r": err / n_inter if ratio_ok else nan,
        "log_r": _log_ratio(err, n_inter) if ratio_ok else nan,
        "interaction_r": pearson(inter_hat, inter),
        "delta_r": pearson(d_hat, d),
        "scale": float(d_hat @ s) / ss if ss > NUMERATOR_FLOOR else nan,
        "lam": float(d @ s) / ss if ss > NUMERATOR_FLOOR else nan,
        "internal_r": nan,
        "internal_log_r": nan,
    }
    if pred_a is not None and pred_b is not None:
        inter_int = d_hat - (_vec(pred_a)[idx] - ctrl) - (_vec(pred_b)[idx] - ctrl)
        out["internal_r"] = pearson(inter_int, inter)
        if ratio_ok:
            out["internal_log_r"] = _log_ratio(float(np.linalg.norm(inter_int - inter)), n_inter)
    return out


def split_half_floor(
    pair_X: np.ndarray, *, control_mean, delta_a, delta_b, top_idx, seed: int = 0,
) -> dict[str, float]:
    """How small ``r_p`` can get when the truth is itself estimated from finite cells.

    Splits the pair's held-out cells ``pair_X`` (n, G) at random into two halves and
    measures the distance between the halves' mean changes on the top-DE genes. Each
    half's mean is noisier than the full mean by a factor of about 2 in variance, and the
    difference of two halves carries both, so ``‖mean₁ − mean₂‖ / 2`` approximates the
    error a perfect predictor would show against the full held-out mean. Returned as a
    ratio to ``‖I_p‖`` (``floor_r``) and its log, on the same scale as ``r_p``, and as a
    ratio to ``‖Δ_p‖`` (``floor_e``), on the scale of ``e_p``.
    """
    rng = np.random.default_rng(seed)
    n = pair_X.shape[0]
    perm = rng.permutation(n)
    h1, h2 = perm[: n // 2], perm[n // 2:]
    idx = np.asarray(top_idx, dtype=int)
    diff = float(np.linalg.norm(pair_X[h1][:, idx].mean(0) - pair_X[h2][:, idx].mean(0))) / 2.0
    ctrl = _vec(control_mean)[idx]
    d = pair_X[:, idx].mean(0).astype(np.float64) - ctrl
    inter = d - _vec(delta_a)[idx] - _vec(delta_b)[idx]
    n_inter, n_delta = float(np.linalg.norm(inter)), float(np.linalg.norm(d))
    ok = n_inter > DENOMINATOR_REL_FLOOR * n_delta and n_inter > 0.0
    return {
        "floor_r": diff / n_inter if ok else float("nan"),
        "floor_log_r": _log_ratio(diff, n_inter) if ok else float("nan"),
        "floor_e": diff / n_delta if n_delta >= NUMERATOR_FLOOR else float("nan"),
        "n_cells": int(n),
    }
