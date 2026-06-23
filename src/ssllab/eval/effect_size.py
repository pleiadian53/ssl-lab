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

from collections.abc import Callable

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


def run_effect_size_eval(
    predict_fn: Callable[[int, str], "np.ndarray"],
    *,
    hvg_X: np.ndarray,
    pert_names: np.ndarray,
    pert_id: np.ndarray,
    is_test: np.ndarray,
    de_genes: dict,
    control_mean: np.ndarray,
    top_k: int = 20,
    min_test_cells: int = 20,
    limit_perts: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score effect-size recovery for *any* predictor, single-sourced across methods.

    ``predict_fn(pid, name) -> pred_mean`` returns one perturbation's predicted per-gene
    normalized expression ``(G,)``. The held-out truth is the mean over that pert's
    **test** cells (``is_test``); the score is :func:`delta_correlation` on its top-DE
    genes. Returns ``(per_pert, summary)``. Used by the flow eval (06), the NB-VAE
    baseline (09), and any future predictor — so every method is graded identically.
    """
    names = np.asarray(pert_names)
    n_genes = hvg_X.shape[1]
    evaluable = [p for p in names if p != "control" and p in de_genes]
    if limit_perts:
        evaluable = evaluable[:limit_perts]

    per_pert: dict[str, float] = {}
    for name in evaluable:
        pid = int(np.where(names == name)[0][0])
        test_mask = (pert_id == pid) & is_test
        if int(test_mask.sum()) < min_test_cells:
            continue
        true_mean = hvg_X[test_mask].mean(0)
        pred_mean = _np(predict_fn(pid, name))
        top_idx = [i for i in de_genes[name]["top_idx"][:top_k] if i < n_genes]
        per_pert[name] = delta_correlation(pred_mean, true_mean, control_mean, top_idx)

    summary = summarize(per_pert)
    if log:
        ranked = sorted(((v, k) for k, v in per_pert.items() if v == v), reverse=True)
        if ranked:
            log("best: " + ", ".join(f"{k}={v:.2f}" for v, k in ranked[:5]))
            log("worst: " + ", ".join(f"{k}={v:.2f}" for v, k in ranked[-5:]))
    return per_pert, summary
