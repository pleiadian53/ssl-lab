"""The embedding store: one matrix of vectors per dataset, filed by row, with a description.

Every generative stage in this project reads its inputs as vectors: the count decoder turns a
cell's vector into gene counts, and the conditional flow learns how those vectors move under a
perturbation. Until now the vectors were always computed on the fly by our own JEPA encoder. An
*embedding store* is the same vectors written to disk once, by any encoder, in a form every stage
can read without knowing what produced them. That is what lets a pretrained foundation model
such as Geneformer, or any other encoder, take the JEPA encoder's place without touching the
decoder or the flow.

A store is a directory with three files::

    embeddings.npy   float32 [n, d]   one vector per row
    row_idx.npy      int64   [n]      which row of the dataset each vector belongs to,
                                      strictly increasing (sorted, no duplicates)
    manifest.json                     where the vectors came from (see below)

The rows are positions in a named dataset, for Perturb-seq the processed cache's cell order, so
the store itself assumes nothing about cells: the same format holds gene vectors, protein vectors
or anything else that can be filed by an index. A store may cover only some rows of its dataset
(a feasibility sample, say). Readers that need every row check coverage themselves; a missing row
is never filled in.

The manifest always holds:

``source``               what produced the vectors (``jepa``, ``geneformer``, ``projection``, ...)
``width``, ``n``         the matrix shape
``dataset_fingerprint``  a hash of the dataset's row order (for Perturb-seq,
                         :func:`ssllab.data.perturbseq.cache_fingerprint`); checked on load, so
                         vectors computed on one version of a dataset are never looked up by the
                         row numbers of another
``content_md5``          a hash of the vectors and row indices, computed when the store is written
``space_key``            which vector space the store lives in. Two artifacts trained on the same
                         space can be combined (a flow and a decoder, say); two trained on
                         different spaces cannot, even when their widths agree. It defaults to
                         ``store:<content_md5>``. A store exported from a JEPA encoder uses
                         ``jepa:<hash of the encoder's weights>``, the same key the live encoder
                         reports (:func:`ssllab.embeddings.source.jepa_space_key`), because the two
                         give the same vectors.

and any further descriptive fields the producer adds: the model, the embedding mode and layer,
the producing script and commit, and for a projection its parent store and the rows it was fit on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDINGS_NPY = "embeddings.npy"
ROW_IDX_NPY = "row_idx.npy"
MANIFEST_JSON = "manifest.json"
REQUIRED_FIELDS = ("source", "width", "n", "dataset_fingerprint", "content_md5", "space_key")


class StoreError(ValueError):
    """A store is malformed, or does not belong to the dataset it is being read against."""


@dataclass
class EmbeddingStore:
    """A loaded store. ``embeddings`` may be a read-only memory map."""

    path: Path
    embeddings: np.ndarray       # (n, d) float32
    row_idx: np.ndarray          # (n,) int64, strictly increasing
    manifest: dict[str, Any]

    @property
    def width(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def space_key(self) -> str:
        return str(self.manifest["space_key"])

    def positions(self, rows: np.ndarray) -> np.ndarray:
        """Positions in the matrix of the given dataset rows; raises if any row is not stored."""
        rows = np.asarray(rows, dtype=np.int64).ravel()
        pos = np.searchsorted(self.row_idx, rows)
        pos_c = np.minimum(pos, self.n - 1)
        missing = (pos >= self.n) | (self.row_idx[pos_c] != rows)
        if missing.any():
            bad = rows[missing]
            raise StoreError(f"{int(missing.sum())} requested rows are not in the store {self.path} "
                             f"(first: {bad[:5].tolist()}); a store never fills in missing rows")
        return pos

    def lookup(self, rows: np.ndarray) -> np.ndarray:
        """The vectors of the given dataset rows, in the order requested, as a float32 copy."""
        return np.ascontiguousarray(self.embeddings[self.positions(rows)], dtype=np.float32)


def content_md5(embeddings: np.ndarray, row_idx: np.ndarray) -> str:
    """Hash of the vectors and the row indices, in chunks so a memory map is never read at once."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(row_idx, dtype=np.int64).tobytes())
    step = max(1, (1 << 26) // max(1, embeddings.shape[1] * 4))    # about 64 MB per chunk
    for i in range(0, len(embeddings), step):
        h.update(np.ascontiguousarray(embeddings[i:i + step], dtype=np.float32).tobytes())
    return h.hexdigest()


def _validate_arrays(embeddings: np.ndarray, row_idx: np.ndarray, where: str) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise StoreError(f"{where}: embeddings must be a non-empty 2-D matrix, got shape {embeddings.shape}")
    if row_idx.ndim != 1 or len(row_idx) != len(embeddings):
        raise StoreError(f"{where}: row_idx must be 1-D with one entry per vector "
                         f"({len(embeddings)}), got shape {row_idx.shape}")
    if (row_idx < 0).any():
        raise StoreError(f"{where}: row_idx has negative entries")
    if len(row_idx) > 1 and not (np.diff(row_idx) > 0).all():
        raise StoreError(f"{where}: row_idx must be strictly increasing (sorted, no duplicate rows)")
    step = max(1, (1 << 26) // (embeddings.shape[1] * 4))
    for i in range(0, len(embeddings), step):
        if not np.isfinite(embeddings[i:i + step]).all():
            raise StoreError(f"{where}: embeddings contain non-finite values (first chunk at row {i})")


def provenance(script: str | Path) -> dict[str, Any]:
    """The producing script's name and, when the code sits in a git checkout, the commit.

    Pods install the package from a synced copy of the tree that may have no ``.git``; the commit
    is then ``None`` rather than an error.
    """
    import subprocess

    script = Path(script).resolve()
    try:
        commit = subprocess.run(["git", "-C", str(script.parent), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"producer": script.name, "git_commit": commit}


def write_store(
    path: str | Path,
    embeddings: np.ndarray,
    row_idx: np.ndarray,
    *,
    source: str,
    dataset_fingerprint: str,
    space_key: str | None = None,
    **fields: Any,
) -> EmbeddingStore:
    """Validate and write a store, returning it as loaded from disk.

    ``row_idx`` must already be strictly increasing; a producer whose rows come out in another
    order sorts the rows and the vectors together first. ``fields`` are extra descriptive
    manifest entries (model, mode, layer, producing script, ...).
    """
    path = Path(path)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    row_idx = np.ascontiguousarray(row_idx, dtype=np.int64)
    _validate_arrays(embeddings, row_idx, str(path))
    clash = set(fields) & set(REQUIRED_FIELDS)
    if clash:
        raise StoreError(f"manifest fields {sorted(clash)} are set by write_store, not by the caller")
    md5 = content_md5(embeddings, row_idx)
    manifest = {
        "source": source,
        "width": int(embeddings.shape[1]),
        "n": int(embeddings.shape[0]),
        "dataset_fingerprint": dataset_fingerprint,
        "content_md5": md5,
        "space_key": space_key or f"store:{md5}",
        **fields,
    }
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / EMBEDDINGS_NPY, embeddings)
    np.save(path / ROW_IDX_NPY, row_idx)
    (path / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2, default=str))
    return load_store(path, expect_fingerprint=dataset_fingerprint)


def load_store(
    path: str | Path,
    expect_fingerprint: str | None = None,
    mmap: bool = True,
    verify_content: bool = False,
) -> EmbeddingStore:
    """Load and validate a store.

    Checks the manifest's required fields, the shapes, that ``row_idx`` is strictly increasing,
    that every value is finite, and, when ``expect_fingerprint`` is given, that the store was
    computed on that dataset. ``verify_content`` also re-hashes the files against the manifest's
    ``content_md5`` (a full read; pod preflights use it). Raises :class:`StoreError` on any failure.
    """
    path = Path(path)
    for name in (EMBEDDINGS_NPY, ROW_IDX_NPY, MANIFEST_JSON):
        if not (path / name).exists():
            raise StoreError(f"{path} is not an embedding store: {name} is missing")
    manifest = json.loads((path / MANIFEST_JSON).read_text())
    missing = [k for k in REQUIRED_FIELDS if k not in manifest]
    if missing:
        raise StoreError(f"{path}: manifest is missing {missing}")
    embeddings = np.load(path / EMBEDDINGS_NPY, mmap_mode="r" if mmap else None)
    row_idx = np.load(path / ROW_IDX_NPY)
    _validate_arrays(embeddings, row_idx, str(path))
    if embeddings.dtype != np.float32:
        raise StoreError(f"{path}: embeddings must be float32, got {embeddings.dtype}")
    if (int(manifest["width"]), int(manifest["n"])) != (embeddings.shape[1], embeddings.shape[0]):
        raise StoreError(f"{path}: manifest says {manifest['n']} x {manifest['width']}, "
                         f"files hold {embeddings.shape[0]} x {embeddings.shape[1]}")
    if expect_fingerprint is not None and manifest["dataset_fingerprint"] != expect_fingerprint:
        raise StoreError(f"{path} was computed on a different dataset (fingerprint "
                         f"{manifest['dataset_fingerprint'][:8]}, expected {expect_fingerprint[:8]}); "
                         f"its row numbers do not refer to these cells")
    if verify_content and content_md5(embeddings, row_idx) != manifest["content_md5"]:
        raise StoreError(f"{path}: contents do not match the manifest's content_md5")
    return EmbeddingStore(path=path, embeddings=embeddings, row_idx=row_idx, manifest=manifest)
