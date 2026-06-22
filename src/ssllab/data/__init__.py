"""Data adapters. Each adapter emits modality-agnostic token tensors
``(B, n_tokens, token_dim)`` that the models consume."""

from ssllab.data.mnist import (
    GRID,
    IMG_SIZE,
    N_TOKENS,
    PATCH_SIZE,
    TOKEN_DIM,
    get_mnist_dataloaders,
    patchify,
    unpatchify,
)
from ssllab.data.perturbseq import (
    ARTIFACT_VERSION,
    DEFAULT_N_HVG,
    DEFAULT_N_TOKENS,
    ControlSampler,
    detokenize_cells,
    get_perturbseq_dataloaders,
    load_cache,
    make_gene_partition,
    token_dim_for,
    tokenize_cells,
    write_cache,
)

__all__ = [
    # MNIST (image proxy)
    "get_mnist_dataloaders",
    "patchify",
    "unpatchify",
    "IMG_SIZE",
    "PATCH_SIZE",
    "GRID",
    "N_TOKENS",
    "TOKEN_DIM",
    # Perturb-seq (single-cell)
    "get_perturbseq_dataloaders",
    "tokenize_cells",
    "detokenize_cells",
    "make_gene_partition",
    "token_dim_for",
    "ControlSampler",
    "load_cache",
    "write_cache",
    "DEFAULT_N_HVG",
    "DEFAULT_N_TOKENS",
    "ARTIFACT_VERSION",
]
