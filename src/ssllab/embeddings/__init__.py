"""Precomputed embeddings: let the decoder and the flow read vectors from any encoder.

See :mod:`ssllab.embeddings.store` for the on-disk contract, :mod:`ssllab.embeddings.source` for
the one interface every consumer reads latents through, and :mod:`ssllab.embeddings.project` for
fitting an equal-width projection of a wide embedding on training rows only.
"""

from ssllab.embeddings.project import fit_pca, project_store
from ssllab.embeddings.source import (
    JepaSource,
    LatentSource,
    PrecomputedSource,
    check_same_space,
    file_md5,
    jepa_space_key,
)
from ssllab.embeddings.store import (
    EmbeddingStore,
    StoreError,
    content_md5,
    load_store,
    provenance,
    write_store,
)

__all__ = [
    "EmbeddingStore",
    "JepaSource",
    "LatentSource",
    "PrecomputedSource",
    "StoreError",
    "check_same_space",
    "content_md5",
    "file_md5",
    "fit_pca",
    "jepa_space_key",
    "load_store",
    "project_store",
    "provenance",
    "write_store",
]
