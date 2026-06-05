"""MNIST data adapter.

This is the *only* modality-specific module in the starter. Its job is to turn
images into the modality-agnostic token tensor the rest of ``ssllab`` consumes:

    images ``(B, 1, H, W)``  ->  tokens ``(B, n_tokens, token_dim)``

A future modality (gene-count vectors, DNA windows) provides a sibling adapter
that emits the same ``(B, n_tokens, token_dim)`` shape; nothing downstream
changes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# MNIST geometry, fixed for the POC.
IMG_SIZE = 28
PATCH_SIZE = 7
GRID = IMG_SIZE // PATCH_SIZE          # 4  -> a 4x4 grid of patches
N_TOKENS = GRID * GRID                 # 16 patches
TOKEN_DIM = PATCH_SIZE * PATCH_SIZE    # 49 pixels per patch


def patchify(images: torch.Tensor, patch: int = PATCH_SIZE) -> torch.Tensor:
    """Split images into flattened, row-major patches.

    Parameters
    ----------
    images:
        ``(B, 1, H, W)`` float tensor.
    patch:
        Square patch side length; ``H`` and ``W`` must be divisible by it.

    Returns
    -------
    ``(B, n_tokens, patch*patch)`` tensor of flattened patches, ordered
    row-major over the patch grid.
    """
    b, c, h, w = images.shape
    if c != 1:
        raise ValueError(f"patchify expects single-channel images, got {c}")
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} not divisible by patch {patch}")
    gh, gw = h // patch, w // patch
    # (B, 1, gh, patch, gw, patch) -> (B, gh, gw, patch, patch)
    x = images.reshape(b, gh, patch, gw, patch).permute(0, 1, 3, 2, 4)
    return x.reshape(b, gh * gw, patch * patch)


def unpatchify(tokens: torch.Tensor, patch: int = PATCH_SIZE, img: int = IMG_SIZE) -> torch.Tensor:
    """Inverse of :func:`patchify`: ``(B, n_tokens, patch*patch)`` -> ``(B, 1, H, W)``."""
    b, n, d = tokens.shape
    grid = img // patch
    if n != grid * grid or d != patch * patch:
        raise ValueError(f"shape mismatch for img={img}, patch={patch}: got {tokens.shape}")
    x = tokens.reshape(b, grid, grid, patch, patch).permute(0, 1, 3, 2, 4)
    return x.reshape(b, 1, img, img)


def get_mnist_dataloaders(
    batch_size: int = 256,
    data_dir: str | Path = "data",
    limit: int | None = None,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)`` of MNIST in ``[0, 1]``.

    Parameters
    ----------
    limit:
        If set, truncate the *train* split to this many examples (fast smoke
        runs). The test split is left intact.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()  # -> float in [0, 1], shape (1, 28, 28)

    train = datasets.MNIST(str(data_dir), train=True, download=download, transform=tfm)
    test = datasets.MNIST(str(data_dir), train=False, download=download, transform=tfm)

    if limit is not None:
        train = Subset(train, range(min(limit, len(train))))

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
