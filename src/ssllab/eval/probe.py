"""Linear probe: is the frozen JEPA latent semantically useful?

Fit a linear classifier (logistic regression) on frozen embeddings and report
test accuracy. Well-above-chance accuracy means the encoder learned structure
without ever seeing labels — the headline SSL claim.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def linear_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict[str, float]:
    """Train a logistic-regression probe on frozen embeddings.

    Parameters
    ----------
    z_*, y_*:
        Embeddings ``(N, D)`` and integer labels ``(N,)`` (torch or numpy).

    Returns
    -------
    ``{"train_acc": ..., "test_acc": ...}``.
    """
    def _np(a: torch.Tensor | np.ndarray) -> np.ndarray:
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

    Xtr, ytr, Xte, yte = _np(z_train), _np(y_train), _np(z_test), _np(y_test)
    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(Xtr, ytr)
    return {
        "train_acc": float(clf.score(Xtr, ytr)),
        "test_acc": float(clf.score(Xte, yte)),
    }
