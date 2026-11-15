"""Principal-component projection of an embedding store to fewer dimensions.

A wide embedding, such as Geneformer V2's 768 numbers per cell, is a harder space for the
conditional flow to learn than our encoder's 256, with the same amount of training data. One way
to make the comparison at equal width is to project the wide vectors onto their leading principal
components. The projection is fit on **training rows only**, so nothing about the held-out cells
shapes the coordinates they are later scored in, and it is written as a store of its own, so every
consumer reads it exactly like any other store and never needs to know a projection happened.

The projection is ``y = (x - mean) @ components.T``, with ``mean`` and ``components`` (the top
``k`` eigenvectors of the training rows' covariance, sign-fixed so each component's largest-magnitude
loading is positive) saved beside the vectors in ``projection.npz``. The manifest records the parent
store, the number of fit rows and the share of the training variance the kept components explain,
which is itself a number worth reporting: it says how much of the wide embedding's spread survives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssllab.embeddings.store import EmbeddingStore, StoreError, write_store


def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean (d,), components (k, d), explained_variance (d,))`` from the rows of ``X``.

    Uses the eigendecomposition of the ``d x d`` covariance in float64, which is exact and cheap
    when there are many more rows than dimensions, as here (about 96,000 rows, 768 dimensions).
    ``explained_variance`` holds every eigenvalue, largest first, so callers can report the share
    the kept ``k`` explain.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if not 0 < k <= d:
        raise ValueError(f"k must be in 1..{d}, got {k}")
    if n <= 1:
        raise ValueError("need at least two rows to fit a projection")
    mean = X.mean(0)
    Xc = X - mean
    cov = Xc.T @ Xc / (n - 1)
    evals, evecs = np.linalg.eigh(cov)                  # ascending
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    comps = evecs[:, :k].T                              # (k, d)
    signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(1)])
    comps = comps * np.where(signs == 0, 1.0, signs)[:, None]
    return mean, comps, np.clip(evals, 0.0, None)


def project_store(parent: EmbeddingStore, fit_rows: np.ndarray, k: int, out: str | Path,
                  fit_description: str, **fields) -> EmbeddingStore:
    """Fit a ``k``-dimensional PCA on ``parent``'s ``fit_rows`` and write every row's projection.

    ``fit_rows`` are dataset rows (for the comparison, the combo split's training cells), all of
    which must be in the parent store. ``fit_description`` says in words which rows they are and
    is recorded in the manifest.
    """
    fit_rows = np.unique(np.asarray(fit_rows, dtype=np.int64))
    if len(fit_rows) == 0:
        raise StoreError("project_store needs at least one fit row")
    X_fit = parent.lookup(fit_rows)
    mean, comps, evals = fit_pca(X_fit, k)
    Y = np.empty((parent.n, k), dtype=np.float32)
    step = 8192
    for i in range(0, parent.n, step):
        Y[i:i + step] = ((np.asarray(parent.embeddings[i:i + step], dtype=np.float64) - mean) @ comps.T)
    out = Path(out)
    store = write_store(
        out, Y, parent.row_idx,
        source="projection",
        dataset_fingerprint=parent.manifest["dataset_fingerprint"],
        method="pca",
        parent_store=str(parent.path),
        parent_content_md5=parent.manifest["content_md5"],
        parent_space_key=parent.space_key,
        parent_width=parent.width,
        model=parent.manifest.get("model"),
        mode=parent.manifest.get("mode"),
        n_fit_rows=int(len(fit_rows)),
        fit_rows=fit_description,
        explained_variance_share=float(evals[:k].sum() / evals.sum()) if evals.sum() > 0 else float("nan"),
        **fields,
    )
    np.savez(out / "projection.npz", mean=mean, components=comps, eigenvalues=evals)
    return store
