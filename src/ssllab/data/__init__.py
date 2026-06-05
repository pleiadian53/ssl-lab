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

__all__ = [
    "get_mnist_dataloaders",
    "patchify",
    "unpatchify",
    "IMG_SIZE",
    "PATCH_SIZE",
    "GRID",
    "N_TOKENS",
    "TOKEN_DIM",
]
