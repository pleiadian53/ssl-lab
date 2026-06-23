"""Effect-size metric — the perturbation-biology benchmark.

The discipline from the design-space series: score the **change** an intervention
causes, not the absolute state. For a perturbation, the *effect* is its
differential expression Δ = mean(perturbed) − mean(control). A model is graded on
how well its predicted Δ correlates with the true Δ on the **top differentially-
expressed genes** — the genes the perturbation actually moved (Pearson r). This is
what scGen / CPA / scPPDM report, and the metric this project must reproduce to
test whether the conditional-flow + count-decoder method recovers effect size.
"""

from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def pearson(a, b) -> float:
    """Pearson correlation between two 1-D vectors."""
    a, b = _np(a).ravel(), _np(b).ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def delta_correlation(
    pred_pert_mean, true_pert_mean, control_mean, top_idx,
) -> float:
    """Effect-size correlation on the top-DE genes.

    All means are per-gene normalized expression (log1p-CP10K). ``top_idx`` are the
    perturbation's top-DE gene indices. Returns Pearson(predicted Δ, true Δ) over
    those genes, where Δ is expression relative to control.
    """
    pred, true, ctrl = _np(pred_pert_mean), _np(true_pert_mean), _np(control_mean)
    idx = np.asarray(top_idx, dtype=int)
    d_pred = pred[idx] - ctrl[idx]
    d_true = true[idx] - ctrl[idx]
    return pearson(d_pred, d_true)


def summarize(per_pert: dict[str, float]) -> dict[str, float]:
    """Aggregate per-perturbation correlations (ignoring NaNs)."""
    vals = np.array([v for v in per_pert.values() if v == v], dtype=float)  # drop NaN
    return {
        "mean_delta_r": float(vals.mean()) if vals.size else float("nan"),
        "median_delta_r": float(np.median(vals)) if vals.size else float("nan"),
        "n_perturbations": int(vals.size),
    }
