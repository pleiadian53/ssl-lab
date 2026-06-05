"""Intrinsic evaluation metrics for generated samples (modality-agnostic).

Every function here operates on **feature vectors** ``(N, D)`` or class
**probabilities** ``(N, C)`` — never on raw images — so the same code evaluates
MNIST samples (features from a small CNN oracle) or DNA/protein samples (features
from a domain encoder). See
``examples/generative_jepa/docs/evaluating-generated-samples.md`` for the why.

Metric families implemented:
- Distributional distance: :func:`fid`, :func:`kid`.
- Fidelity vs diversity: :func:`precision_recall` (Kynkäänniemi 2019),
  :func:`density_coverage` (Naeem 2020).
- Novelty / memorization: :func:`nn_distance_stats`.
- Classifier-oracle: :func:`classifier_metrics`.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _as_np(x) -> np.ndarray:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --------------------------------------------------------------------------
# Distributional distance
# --------------------------------------------------------------------------


def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    """Fréchet (2-Wasserstein) distance between two Gaussians."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def fid(real_feats, gen_feats, eps: float = 1e-6) -> float:
    """Fréchet Inception-style Distance in an arbitrary feature space.

    ``real_feats``/``gen_feats``: ``(N, D)`` feature matrices. Lower is better
    (0 = identical Gaussian moments). ``eps`` regularizes the covariances so the
    matrix square root is stable when N is small or features are collinear.
    Use enough samples (ideally N > feature dim, thousands is better) for FID to
    be reliable.
    """
    r, g = _as_np(real_feats), _as_np(gen_feats)
    d = r.shape[1]
    mu_r, mu_g = r.mean(0), g.mean(0)
    cov_r = np.cov(r, rowvar=False) + eps * np.eye(d)
    cov_g = np.cov(g, rowvar=False) + eps * np.eye(d)
    return frechet_distance(mu_r, cov_r, mu_g, cov_g)


def kid(real_feats, gen_feats, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> float:
    """Kernel Inception Distance = unbiased squared MMD with a polynomial kernel.

    More reliable than FID at small sample sizes. Lower is better.
    """
    r, g = _as_np(real_feats), _as_np(gen_feats)
    d = r.shape[1]
    g_ = 1.0 / d if gamma is None else gamma

    def k(a, b):
        return (g_ * (a @ b.T) + coef0) ** degree

    m, n = len(r), len(g)
    krr, kgg, krg = k(r, r), k(g, g), k(r, g)
    # unbiased: drop diagonals on the within-set terms
    sum_rr = (krr.sum() - np.trace(krr)) / (m * (m - 1))
    sum_gg = (kgg.sum() - np.trace(kgg)) / (n * (n - 1))
    sum_rg = krg.mean()
    return float(sum_rr + sum_gg - 2.0 * sum_rg)


# --------------------------------------------------------------------------
# Fidelity vs diversity (manifold metrics)
# --------------------------------------------------------------------------


def _knn_radius(X: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point in X to its k-th nearest neighbor *within X*."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1: self is neighbor 0
    dists, _ = nn.kneighbors(X)
    return dists[:, k]


def precision_recall(real_feats, gen_feats, k: int = 3) -> dict[str, float]:
    """Improved Precision & Recall (Kynkäänniemi et al. 2019).

    precision = fraction of generated points inside the real manifold (fidelity);
    recall    = fraction of real points inside the generated manifold (diversity).
    """
    r, g = _as_np(real_feats), _as_np(gen_feats)
    r_radii = _knn_radius(r, k)
    g_radii = _knn_radius(g, k)

    # cross distances
    d_gr = NearestNeighbors(n_neighbors=1).fit(r)  # for each g, is it within some real ball?
    dist_g_to_r, idx_g = d_gr.kneighbors(g)
    precision = float(np.mean(dist_g_to_r[:, 0] <= r_radii[idx_g[:, 0]]))

    d_rg = NearestNeighbors(n_neighbors=1).fit(g)
    dist_r_to_g, idx_r = d_rg.kneighbors(r)
    recall = float(np.mean(dist_r_to_g[:, 0] <= g_radii[idx_r[:, 0]]))
    return {"precision": precision, "recall": recall}


def density_coverage(real_feats, gen_feats, k: int = 5) -> dict[str, float]:
    """Density & Coverage (Naeem et al. 2020) — outlier-robust fidelity/diversity.

    density  ~ how deeply generated points sit inside real neighborhoods (>=1 ok);
    coverage = fraction of real points with a generated point in their k-NN ball.
    """
    r, g = _as_np(real_feats), _as_np(gen_feats)
    r_radii = _knn_radius(r, k)
    nn_r = NearestNeighbors(n_neighbors=1).fit(r)
    # all real points within radius of each gen point: use radius_neighbors
    # density: average count of real-balls containing each gen point / k
    full = NearestNeighbors(n_neighbors=len(r)).fit(r)
    dists, idx = full.kneighbors(g)
    in_ball = dists <= r_radii[idx]
    density = float(in_ball.sum(axis=1).mean() / k)
    # coverage: fraction of real points whose nearest gen point is within its radius
    nn_g = NearestNeighbors(n_neighbors=1).fit(g)
    dist_r_to_g, _ = nn_g.kneighbors(r)
    coverage = float(np.mean(dist_r_to_g[:, 0] <= r_radii))
    return {"density": density, "coverage": coverage}


# --------------------------------------------------------------------------
# Novelty / memorization
# --------------------------------------------------------------------------


def nn_distance_stats(query_feats, ref_feats, k: int = 1) -> dict[str, float]:
    """Nearest-neighbor distance from each query point to a reference set.

    Use ``query=generated, ref=train`` to detect memorization (tiny distances ⇒
    the model copied training data) vs off-distribution drift (huge distances).
    Compare against a ``query=real_test, ref=train`` baseline.
    """
    q, ref = _as_np(query_feats), _as_np(ref_feats)
    nn = NearestNeighbors(n_neighbors=k).fit(ref)
    dists, _ = nn.kneighbors(q)
    d = dists[:, k - 1]
    return {
        "nn_mean": float(d.mean()),
        "nn_median": float(np.median(d)),
        "nn_p05": float(np.percentile(d, 5)),
        "nn_min": float(d.min()),
    }


# --------------------------------------------------------------------------
# Classifier-oracle metrics
# --------------------------------------------------------------------------


def classifier_metrics(probs, n_classes: int | None = None) -> dict[str, float]:
    """Summarize an oracle classifier's predictions on generated samples.

    ``probs``: ``(N, C)`` softmax outputs. Returns:
    - ``confidence``: mean top-1 probability (fidelity proxy — realistic samples
      get confident predictions);
    - ``coverage_entropy``: entropy of the mean predicted-class distribution,
      normalized to [0,1] (1 = all classes equally produced; low ⇒ mode collapse);
    - ``classes_covered``: # distinct argmax classes present.
    """
    p = _as_np(probs)
    c = n_classes or p.shape[1]
    confidence = float(p.max(axis=1).mean())
    preds = p.argmax(axis=1)
    counts = np.bincount(preds, minlength=c).astype(float)
    dist = counts / counts.sum()
    nz = dist[dist > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "confidence": confidence,
        "coverage_entropy": entropy / np.log(c),  # normalized
        "classes_covered": int((counts > 0).sum()),
        "n_classes": int(c),
    }
