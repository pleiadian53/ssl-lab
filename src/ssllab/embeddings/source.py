"""Latent sources: one interface for "the vector of these rows", however it is produced.

The decoder, the flow and the ceiling analysis need the latent vectors of explicit dataset rows.
A :class:`LatentSource` answers that one question, ``encode(rows) -> (len(rows), dim)`` tensor,
and describes itself with :meth:`LatentSource.identity`, a small dict that scripts write into
their checkpoints and reports so every number can say which vectors it was computed on.

Two sources exist:

:class:`JepaSource`         runs our frozen JEPA encoder on the rows, exactly as
                            :func:`ssllab.latents.encode_rows` always has.
:class:`PrecomputedSource`  looks the rows up in an embedding store
                            (:mod:`ssllab.embeddings.store`), whatever encoder wrote it.

Both report a ``space_key``, which names the vector space. A flow and a decoder can be combined
only if they were trained on the same space; :func:`check_same_space` enforces that wherever the
two are paired, which matters because two different encoders can share a width (our JEPA and
Geneformer V1 are both 256-dimensional) and a mismatch would otherwise run silently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ssllab.embeddings.store import EmbeddingStore, load_store
from ssllab.latents import encode_rows


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jepa_space_key(jepa) -> str:
    """The space key of a JEPA's latents: a hash of the weights that compute them.

    It hashes the online encoder's parameters and buffers (``jepa.embed`` reads only those), not
    the checkpoint file, so a checkpoint that is loaded and saved again, as the decoder-only
    co-adaptation run does, keeps its key even though the file's bytes change. The live encoder and
    a store exported from it report the same key.
    """
    h = hashlib.md5()
    for name, tensor in sorted(jepa.encoder.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return f"jepa:{h.hexdigest()}"


class LatentSource:
    """Base interface. Subclasses set ``dim`` and implement :meth:`encode` and :meth:`identity`."""

    dim: int

    def encode(self, rows: np.ndarray | torch.Tensor) -> torch.Tensor:
        """``(len(rows), dim)`` float32 latents on CPU, in the source's own (unstandardized) space."""
        raise NotImplementedError

    def identity(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def space_key(self) -> str:
        return str(self.identity()["space_key"])


class JepaSource(LatentSource):
    """Latents from a frozen JEPA encoder, computed on demand by :func:`encode_rows`."""

    def __init__(self, jepa, encoder_path: str | Path, feat: np.ndarray, partition: torch.Tensor,
                 device: torch.device | str = "cpu", batch_size: int = 256) -> None:
        self.jepa = jepa
        self.encoder_path = Path(encoder_path)
        self.encoder_md5 = file_md5(self.encoder_path)
        self.feat, self.partition, self.device, self.batch_size = feat, partition, device, batch_size
        self.dim = int(jepa.cfg.embed_dim)
        self._space_key = jepa_space_key(jepa)

    def encode(self, rows) -> torch.Tensor:
        rows = np.asarray(rows.cpu() if isinstance(rows, torch.Tensor) else rows, dtype=np.int64)
        if len(rows) == 0:
            return torch.empty(0, self.dim)
        return encode_rows(self.jepa, self.feat, self.partition, rows, self.device, self.batch_size)

    def identity(self) -> dict[str, Any]:
        return {"kind": "jepa", "encoder": str(self.encoder_path), "encoder_md5": self.encoder_md5,
                "width": self.dim, "space_key": self._space_key}


class PrecomputedSource(LatentSource):
    """Latents looked up by row in an embedding store. A row the store lacks raises."""

    def __init__(self, store: EmbeddingStore) -> None:
        self.store = store
        self.dim = store.width

    @classmethod
    def open(cls, path: str | Path, expect_fingerprint: str | None) -> "PrecomputedSource":
        """Load a store and check it was computed on the dataset with this fingerprint.

        ``expect_fingerprint=None`` skips the check; scripts always pass the cache's fingerprint.
        """
        return cls(load_store(path, expect_fingerprint=expect_fingerprint))

    def encode(self, rows) -> torch.Tensor:
        rows = np.asarray(rows.cpu() if isinstance(rows, torch.Tensor) else rows, dtype=np.int64)
        if len(rows) == 0:
            return torch.empty(0, self.dim)
        return torch.from_numpy(self.store.lookup(rows))

    def identity(self) -> dict[str, Any]:
        m = self.store.manifest
        keep = ("source", "model", "mode", "layer", "parent_store", "parent_space_key")
        return {"kind": "store", "store": str(self.store.path), "content_md5": m["content_md5"],
                "width": self.dim, "space_key": m["space_key"], **{k: m[k] for k in keep if k in m}}


def check_same_space(a: dict[str, Any] | None, b: dict[str, Any] | None, what: str = "flow and decoder") -> None:
    """Refuse to pair two artifacts trained on different latent spaces.

    Each argument is a recorded :meth:`LatentSource.identity` (or ``None`` for a checkpoint written
    before identities were recorded, which is accepted, since every such checkpoint came from the
    one JEPA pipeline).
    """
    if a is None or b is None:
        return
    if a.get("space_key") != b.get("space_key"):
        raise ValueError(f"the {what} were trained on different latent spaces "
                         f"({a.get('space_key')} vs {b.get('space_key')}); pair artifacts from the same encoder")
