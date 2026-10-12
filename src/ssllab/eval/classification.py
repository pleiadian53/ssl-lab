"""Multi-label classification metrics, operating points, and paired patient bootstrap.

Conventions used throughout:

* Scores are probabilities or logits; only their order matters for AUROC/AUPRC.
* A label whose evaluation set has no positive (or no negative) recording has an
  undefined AUROC. It is returned as NaN with ``defined=False``, never dropped
  silently, so a macro average over a fixed label set can be flagged incomplete.
* Uncertainty for a **difference between methods** comes from a *paired
  patient-cluster bootstrap*: resample patients with replacement, keep all of
  each patient's recordings, score both methods on the same resample. The
  interval reflects test-patient sampling only, not training randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Per-label metrics.
# --------------------------------------------------------------------------- #
def _defined(y: np.ndarray) -> bool:
    return 0 < int(y.sum()) < len(y)


def per_label_metrics(Y: np.ndarray, S: np.ndarray, names: Sequence[str]) -> dict[str, dict]:
    """AUROC, AUPRC and prevalence per label. ``Y`` binary ``(n, K)``, ``S`` scores ``(n, K)``."""
    Y, S = np.asarray(Y), np.asarray(S)
    out = {}
    for j, name in enumerate(names):
        y = Y[:, j].astype(int)
        ok = _defined(y)
        out[name] = {
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else float("nan"),
            "defined": bool(ok),
            "auroc": float(roc_auc_score(y, S[:, j])) if ok else float("nan"),
            "auprc": float(average_precision_score(y, S[:, j])) if ok else float("nan"),
        }
    return out


def macro_auroc(Y: np.ndarray, S: np.ndarray, label_idx: Sequence[int]) -> tuple[float, bool]:
    """Mean AUROC over a fixed label set -> ``(value, complete)``.

    If any label in the set is undefined, ``complete`` is False and the value is
    NaN: the set is never quietly shrunk.
    """
    vals = []
    for j in label_idx:
        y = np.asarray(Y)[:, j].astype(int)
        if not _defined(y):
            return float("nan"), False
        vals.append(roc_auc_score(y, np.asarray(S)[:, j]))
    return float(np.mean(vals)), True


# --------------------------------------------------------------------------- #
# Binary operating points.
# --------------------------------------------------------------------------- #
def threshold_at_sensitivity(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """The highest threshold whose sensitivity (``s >= t``) is at least ``target``.

    That is the most specific threshold meeting the sensitivity target. Fitted on
    validation, applied unchanged to test.
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    if not y.any():
        raise ValueError("no positives: sensitivity is undefined")
    pos = np.sort(s[y])[::-1]                   # positive scores, high to low
    k = int(np.ceil(target * len(pos)))         # positives that must score >= t
    return float(pos[max(k, 1) - 1])


def operating_point(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts and rates at ``s >= threshold``."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(s, dtype=float) >= threshold
    tp, fn = int((p & y).sum()), int((~p & y).sum())
    fp, tn = int((p & ~y).sum()), int((~p & ~y).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    sens, ppv = div(tp, tp + fn), div(tp, tp + fp)
    return {
        "threshold": float(threshold), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens, "specificity": div(tn, tn + fp), "ppv": ppv, "npv": div(tn, tn + fn),
        "f1": div(2 * sens * ppv, sens + ppv) if sens == sens and ppv == ppv else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Paired patient-cluster bootstrap.
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapResult:
    estimate: float              # metric difference on the original sample (A - B)
    ci_low: float
    ci_high: float
    n_boot: int
    n_valid: int                 # resamples where the metric was defined for both methods
    samples: np.ndarray


def _cluster_index(clusters: np.ndarray) -> list[np.ndarray]:
    clusters = np.asarray(clusters)
    order = np.argsort(clusters, kind="stable")
    _, starts = np.unique(clusters[order], return_index=True)
    return np.split(order, starts[1:])


def paired_cluster_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    repeats: bool = False,
) -> BootstrapResult:
    """Bootstrap the difference ``metric(y, a) - metric(y, b)`` over patient clusters.

    ``scores_a`` / ``scores_b`` are ``(n, ...)`` for one repeat, or
    ``(R, n, ...)`` with ``repeats=True`` for ``R`` repeats. With repeats, the statistic is the **mean
    over repeats** of the per-repeat difference, and every repeat is scored on
    the *same* patient resample, so the interval reflects test-patient sampling
    of the repeat-averaged difference, not training variability.

    ``metric(y_subset, scores_subset)`` must return a float (NaN if undefined);
    resamples where it is undefined are dropped and counted in ``n_valid``.
    """
    y = np.asarray(y)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if a.shape != b.shape:
        raise ValueError(f"score shapes differ: {a.shape} vs {b.shape}")
    if not repeats:
        a, b = a[None], b[None]
    if a.shape[1] != len(y):
        raise ValueError("scores and labels have different numbers of recordings")

    def stat(idx: np.ndarray) -> float:
        d = [metric(y[idx], a[r][idx]) - metric(y[idx], b[r][idx]) for r in range(a.shape[0])]
        return float(np.mean(d))

    groups = _cluster_index(clusters)
    est = stat(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        samples[i] = stat(np.concatenate([groups[g] for g in pick]))
    ok = samples[np.isfinite(samples)]
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if ok.size else (np.nan, np.nan))
    return BootstrapResult(float(est), float(lo), float(hi), n_boot, int(ok.size), samples)


def macro_auroc_metric(label_idx: Sequence[int]) -> Callable[[np.ndarray, np.ndarray], float]:
    """A ``metric(y, s)`` closure for :func:`paired_cluster_bootstrap` (NaN if incomplete)."""
    return lambda Y, S: macro_auroc(Y, S, label_idx)[0]
