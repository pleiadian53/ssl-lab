"""Calibration metrics — does the model capture the *spread* of the response, not just its mean?

The effect-size metric ([effect_size.py](effect_size.py)) grades the mean shift Delta. It is
blind to a model's predictive *distribution*: two models with the same mean can disagree
completely on how much a perturbed population varies, and which genes vary. This is exactly
where a flow (a full distribution over latents) should beat a cruder generator, so it is the
axis that decides whether the generative machinery earns its keep once the means tie.

Three complementary reads, all on a perturbation's top-DE genes, comparing a generated
population to the held-out real cells:

- **spread correlation** — Pearson between predicted and true per-gene standard deviation.
  Does the model know *which* genes vary in the response?
- **interval coverage** — fraction of true cells inside the model's central predicted interval
  per gene. Below nominal means the model is over-confident (under-dispersed); above means
  under-confident. The signed gap says which way.
- **mean 1-Wasserstein** — average earth-mover distance between predicted and true per-gene
  distributions. One holistic number capturing mean, spread, and shape together (lower better).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ssllab.eval.effect_size import _np, pearson


def spread_correlation(pred_pop, true_pop, top_idx) -> float:
    """Pearson corr between predicted and true per-gene std on the top-DE genes."""
    p, t = _np(pred_pop), _np(true_pop)
    idx = np.asarray(top_idx, dtype=int)
    return pearson(p[:, idx].std(0), t[:, idx].std(0))


def interval_coverage(pred_pop, true_pop, top_idx, lo: float = 0.1, hi: float = 0.9) -> float:
    """Mean fraction of true cells within the predicted central ``[lo, hi]`` interval per gene.

    The ideal value is ``hi - lo`` (e.g. 0.8). Less than that = the predicted population is too
    tight (over-confident); more = too diffuse.
    """
    p, t = _np(pred_pop), _np(true_pop)
    idx = np.asarray(top_idx, dtype=int)
    ql = np.quantile(p[:, idx], lo, axis=0)
    qh = np.quantile(p[:, idx], hi, axis=0)
    inside = (t[:, idx] >= ql) & (t[:, idx] <= qh)
    return float(inside.mean())


def mean_wasserstein(pred_pop, true_pop, top_idx) -> float:
    """Mean 1-Wasserstein distance between predicted and true per-gene distributions (top-DE)."""
    from scipy.stats import wasserstein_distance

    p, t = _np(pred_pop), _np(true_pop)
    idx = np.asarray(top_idx, dtype=int)
    return float(np.mean([wasserstein_distance(p[:, g], t[:, g]) for g in idx]))


def energy_distance(pred_pop, true_pop, top_idx) -> float:
    """Multivariate 2-sample energy distance on the joint top-DE gene space.

    ``E = 2·E‖P−Q‖ − E‖P−P'‖ − E‖Q−Q'‖`` (0 iff the two populations are identically
    distributed). Unlike the per-gene reads, this sees the *joint* distribution, so it is
    sensitive to gene-gene correlations and multimodality — the structure a rich latent
    flow can capture but marginals cannot. Lower is better.
    """
    import torch

    idx = np.asarray(top_idx, dtype=int)
    P = torch.as_tensor(_np(pred_pop)[:, idx], dtype=torch.float32)
    Q = torch.as_tensor(_np(true_pop)[:, idx], dtype=torch.float32)
    dpq = torch.cdist(P, Q).mean()
    dpp = torch.cdist(P, P).mean()
    dqq = torch.cdist(Q, Q).mean()
    return float(2 * dpq - dpp - dqq)


def run_calibration_eval(
    predict_population_fn: Callable[[int, str], "np.ndarray"],
    *,
    hvg_X: np.ndarray,
    pert_names: np.ndarray,
    pert_id: np.ndarray,
    is_test: np.ndarray,
    de_genes: dict,
    top_k: int = 20,
    min_test_cells: int = 20,
    nominal: tuple[float, float] = (0.1, 0.9),
    limit_perts: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict], dict[str, float]]:
    """Score distributional calibration for any population generator, single-sourced.

    ``predict_population_fn(pid, name) -> (n_gen, G)`` returns a generated population's per-cell
    normalized expression. Compared against the held-out real cells (``is_test``) of that
    perturbation on its top-DE genes. Returns ``(per_pert, summary)`` mirroring
    :func:`ssllab.eval.effect_size.run_effect_size_eval`.
    """
    names = np.asarray(pert_names)
    n_genes = hvg_X.shape[1]
    lo, hi = nominal
    evaluable = [p for p in names if p != "control" and p in de_genes]
    if limit_perts:
        evaluable = evaluable[:limit_perts]

    per_pert: dict[str, dict] = {}
    for name in evaluable:
        pid = int(np.where(names == name)[0][0])
        test_mask = (pert_id == pid) & is_test
        if int(test_mask.sum()) < min_test_cells:
            continue
        true_pop = hvg_X[test_mask]
        pred_pop = _np(predict_population_fn(pid, name))
        top_idx = [i for i in de_genes[name]["top_idx"][:top_k] if i < n_genes]
        per_pert[name] = {
            "spread_r": spread_correlation(pred_pop, true_pop, top_idx),
            "coverage": interval_coverage(pred_pop, true_pop, top_idx, lo, hi),
            "wasserstein": mean_wasserstein(pred_pop, true_pop, top_idx),
            "energy": energy_distance(pred_pop, true_pop, top_idx),
        }

    def agg(key):
        vals = np.array([d[key] for d in per_pert.values() if d[key] == d[key]], dtype=float)
        return float(vals.mean()) if vals.size else float("nan")

    summary = {
        "mean_spread_r": agg("spread_r"),
        "mean_coverage": agg("coverage"),
        "nominal_coverage": hi - lo,
        "mean_wasserstein": agg("wasserstein"),
        "mean_energy": agg("energy"),
        "n_perturbations": len(per_pert),
    }
    if log:
        log(f"spread_r {summary['mean_spread_r']:.3f}  coverage {summary['mean_coverage']:.3f} "
            f"(nominal {hi - lo:.2f})  wasserstein {summary['mean_wasserstein']:.4f}  "
            f"energy {summary['mean_energy']:.4f}  over {summary['n_perturbations']} perts")
    return per_pert, summary
