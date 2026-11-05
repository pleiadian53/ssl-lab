"""Training-free reference predictors, graded by the same effect-size harness as the models.

A learned model's score means little until it is set beside the obvious answer a biologist
would write down without any model. For a two-gene combination that answer is the
**additive model** of genetic interaction: assume the two interventions do not interact,
so the combination's effect is the sum of the singles' effects,

    Δ(A+B) = Δ(A) + Δ(B),     Δ(g) = mean(cells perturbed by g) − mean(control).

Each Δ(g) is estimated from the single-g cells on the **training** side of the split, so
the predictor never sees a held-out cell. Under the ``combo`` split every single is on the
training side and only combinations are held out, which is exactly the situation this
reference is meant for. Published work finds it hard to beat on Norman-style combination
benchmarks (Ahlmann-Eltze, Huber & Anders, *Nature Methods*, 2025).

The sum is taken in the same per-gene normalized space the metric scores (log1p-CP10K),
so the prediction is a mean profile ``control_mean + Σ_g Δ(g)`` that drops straight into
:func:`ssllab.eval.effect_size.run_effect_size_eval`. It is a mean, not a population, so
it has no calibration score. Additivity in log space corresponds to effects that multiply
on the count scale, the usual convention when the additive model is used as a baseline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

CONTROL = "control"


def additive_predictor(
    *,
    hvg_X: np.ndarray,
    pert_names: np.ndarray,
    pert_id: np.ndarray,
    is_train: np.ndarray,
    control_mean: np.ndarray,
    sep: str = "+",
) -> Callable[[int, str], np.ndarray]:
    """Build ``predict_fn(pid, name) -> pred_mean`` for the additive model.

    ``hvg_X`` ``(n_cells, G)`` is normalized expression, ``pert_id`` ``(n_cells,)`` indexes
    ``pert_names``, and ``is_train`` marks the cells the predictor may read. A
    perturbation's name lists its target genes joined by ``sep`` (``"AHR+KLF1"``), and each
    gene must also exist as a single perturbation with at least one training cell.
    Otherwise the call raises, because the additive model has no estimate of that gene's
    effect and silently dropping it would score a different model.
    """
    names = np.asarray(pert_names)
    index = {str(n): i for i, n in enumerate(names)}
    is_train = np.asarray(is_train, dtype=bool)
    control_mean = np.asarray(control_mean, dtype=np.float64)
    cache: dict[str, np.ndarray] = {}

    def single_delta(gene: str) -> np.ndarray:
        if gene not in cache:
            if gene not in index:
                raise KeyError(f"no single perturbation named {gene!r}; the additive model needs one")
            mask = (pert_id == index[gene]) & is_train
            if not mask.any():
                raise ValueError(f"single {gene!r} has no training cells; the additive model needs them")
            cache[gene] = hvg_X[mask].mean(0).astype(np.float64) - control_mean
        return cache[gene]

    def predict(pid: int, name: str) -> np.ndarray:
        genes = [g for g in str(name).split(sep) if g and g != CONTROL]
        return control_mean + sum(single_delta(g) for g in genes)

    return predict
