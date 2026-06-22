"""Perturb-seq data adapter (single-cell genetic perturbation).

The bio sibling of :mod:`ssllab.data.mnist`. Its job is the same: turn a modality
into the modality-agnostic token tensor the rest of ``ssllab`` consumes,

    a cell's HVG vector ``(n_hvg,)``  ->  gene-group tokens ``(n_tokens, token_dim)``

so the *same* JEPA encoder (``build_jepa(token_dim=..., n_tokens=...)``) trains on
cells with no architectural change. Where MNIST splits an image into spatial
patches, here we split a cell's highly-variable-gene (HVG) vector into a fixed
random partition of **gene groups** — the gene-space analogue of ``patchify``.

This module is deliberately *light* (torch + numpy + json only). All the heavy
single-cell processing — acquiring Norman 2019, QC, HVG selection, normalization,
differential expression — lives in the one-time processing script
``examples/perturbation_response/00_process_norman.py`` (which needs
anndata/scanpy/pertpy) and writes the cache this module reads. Keeping the split
means ``pytest`` and training never import the heavy stack.

Cache layout (written by the processing step, read here)::

    <data_dir>/<artifact>/
      tokens_meta.npz   torch-native arrays (the train-time fast path)
      splits.json       perturbation-level train/val/test assignments
      de_genes.json     top-DE HVG indices per perturbation (effect-size metric)
      manifest.json     provenance + every knob (incl. the token partition seed)
      processed.h5ad    canonical biology-native artifact (NOT read at train time)

The batch is a **dict**, not a bare ``(x, y)`` — perturbation modeling needs the
covariates (raw counts for an NB decoder, library size, the perturbation label,
the control grouping for baseline pairing), not just an image and a class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Default geometry. Token geometry is a *modeling* knob revisited at Stage A
# (random gene groups are a sensible default, not a biological commitment).
DEFAULT_N_HVG = 5000
DEFAULT_N_TOKENS = 50                       # -> token_dim = ceil(5000 / 50) = 100
ARTIFACT_VERSION = "1"

# Split codes stored per cell.
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
_SPLIT_CODE = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}

# Cache filenames.
TOKENS_NPZ = "tokens_meta.npz"
SPLITS_JSON = "splits.json"
DE_JSON = "de_genes.json"
MANIFEST_JSON = "manifest.json"


# --------------------------------------------------------------------------- #
# Tokenization — the gene-space analogue of patchify.
# --------------------------------------------------------------------------- #
def token_dim_for(n_hvg: int, n_tokens: int) -> int:
    """Group size needed to cover ``n_hvg`` genes in ``n_tokens`` groups (ceil)."""
    return (n_hvg + n_tokens - 1) // n_tokens


def make_gene_partition(n_hvg: int, n_tokens: int, seed: int) -> torch.Tensor:
    """Deterministic random partition of HVG indices into ``n_tokens`` groups.

    Returns a ``(n_tokens, group_size)`` long tensor of gene indices into the HVG
    panel, row-major over a seeded permutation. When ``n_hvg`` is not divisible by
    ``n_tokens`` the final positions are padded with ``-1`` (handled by
    :func:`tokenize_cells`). Deterministic in ``seed`` so processing and training
    agree on the partition.
    """
    group_size = token_dim_for(n_hvg, n_tokens)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_hvg, generator=g)
    pad = n_tokens * group_size - n_hvg
    if pad:
        perm = torch.cat([perm, torch.full((pad,), -1, dtype=perm.dtype)])
    return perm.view(n_tokens, group_size)


def tokenize_cells(
    hvg_features: torch.Tensor, partition: torch.Tensor, pad_value: float = 0.0
) -> torch.Tensor:
    """Gather HVG features into gene-group tokens.

    ``hvg_features`` ``(B, n_hvg)`` -> ``(B, n_tokens, token_dim)``. Padding
    positions (``partition == -1``) are filled with ``pad_value``.
    """
    if hvg_features.dim() != 2:
        raise ValueError(f"expected (B, n_hvg), got {tuple(hvg_features.shape)}")
    valid = partition >= 0
    safe = partition.clamp(min=0)
    gathered = hvg_features[:, safe]  # (B, n_tokens, token_dim)
    if not bool(valid.all()):
        gathered = torch.where(valid.unsqueeze(0), gathered, gathered.new_full((), pad_value))
    return gathered


def detokenize_cells(tokens: torch.Tensor, partition: torch.Tensor, n_hvg: int) -> torch.Tensor:
    """Inverse of :func:`tokenize_cells`: ``(B, n_tokens, token_dim)`` -> ``(B, n_hvg)``.

    Assumes ``partition`` is a permutation of ``range(n_hvg)`` (the default), so
    valid gene indices are unique.
    """
    b = tokens.shape[0]
    valid = partition >= 0
    idx = partition[valid]              # (n_hvg,) unique gene indices
    vals = tokens[:, valid]             # (B, n_hvg)
    out = tokens.new_zeros(b, n_hvg)
    out[:, idx] = vals
    return out


# --------------------------------------------------------------------------- #
# Cache I/O — shared by the processing script and the tests (DRY).
# --------------------------------------------------------------------------- #
@dataclass
class CacheBundle:
    """In-memory view of a processed Perturb-seq cache."""

    hvg_X: np.ndarray            # (n, n_hvg) f32, normalized features
    counts: np.ndarray           # (n, n_hvg) i32, raw integer counts (NB decoder seam)
    libsize: np.ndarray          # (n,) f32, total UMI per cell
    pert_id: np.ndarray          # (n,) i64, perturbation label index
    is_control: np.ndarray       # (n,) bool, non-targeting-guide cells
    ctrl_group: np.ndarray       # (n,) i64, control-pool key (batch/covariate)
    split_combo: np.ndarray      # (n,) i8, {0,1,2} combinatorial-generalization split
    split_cells: np.ndarray      # (n,) i8, {0,1,2} random per-cell sanity split
    gene_ids: np.ndarray         # (n_hvg,) str, HVG panel identity
    pert_names: np.ndarray       # (n_perts,) str, index = pert_id
    splits: dict[str, Any]
    de_genes: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def n_hvg(self) -> int:
        return self.hvg_X.shape[1]


def write_cache(
    cache_dir: str | Path,
    *,
    hvg_X: np.ndarray,
    counts: np.ndarray,
    libsize: np.ndarray,
    pert_id: np.ndarray,
    is_control: np.ndarray,
    ctrl_group: np.ndarray,
    split_combo: np.ndarray,
    split_cells: np.ndarray,
    gene_ids: np.ndarray,
    pert_names: np.ndarray,
    splits: dict[str, Any],
    de_genes: dict[str, Any],
    manifest: dict[str, Any],
) -> Path:
    """Write the torch-native cache (npz + json). The ``processed.h5ad`` is written
    separately by the processing script. Returns the cache directory."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / TOKENS_NPZ,
        hvg_X=hvg_X.astype(np.float32),
        counts=counts.astype(np.int32),
        libsize=libsize.astype(np.float32),
        pert_id=pert_id.astype(np.int64),
        is_control=is_control.astype(bool),
        ctrl_group=ctrl_group.astype(np.int64),
        split_combo=split_combo.astype(np.int8),
        split_cells=split_cells.astype(np.int8),
        gene_ids=np.asarray(gene_ids, dtype=object),
        pert_names=np.asarray(pert_names, dtype=object),
    )
    (cache_dir / SPLITS_JSON).write_text(json.dumps(splits, indent=2))
    (cache_dir / DE_JSON).write_text(json.dumps(de_genes, indent=2))
    (cache_dir / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2))
    return cache_dir


def load_cache(cache_dir: str | Path) -> CacheBundle:
    """Load a processed Perturb-seq cache into memory."""
    cache_dir = Path(cache_dir)
    npz = np.load(cache_dir / TOKENS_NPZ, allow_pickle=True)
    return CacheBundle(
        hvg_X=npz["hvg_X"],
        counts=npz["counts"],
        libsize=npz["libsize"],
        pert_id=npz["pert_id"],
        is_control=npz["is_control"],
        ctrl_group=npz["ctrl_group"],
        split_combo=npz["split_combo"],
        split_cells=npz["split_cells"],
        gene_ids=npz["gene_ids"],
        pert_names=npz["pert_names"],
        splits=json.loads((cache_dir / SPLITS_JSON).read_text()),
        de_genes=json.loads((cache_dir / DE_JSON).read_text()),
        manifest=json.loads((cache_dir / MANIFEST_JSON).read_text()),
    )


# --------------------------------------------------------------------------- #
# Baseline pairing — population-level (cells are unpaired/destroyed on measure).
# --------------------------------------------------------------------------- #
class ControlSampler:
    """Draw baseline ``z_b`` cells from the matched control *population*.

    Perturb-seq is destructive: a cell is measured once, so there is no paired
    "before" cell. The baseline is therefore the control population for the same
    covariate (``ctrl_group``). This maps each group to its control-cell row
    indices and samples (with replacement) on request.
    """

    def __init__(self, is_control: np.ndarray, ctrl_group: np.ndarray) -> None:
        ctrl_rows = np.flatnonzero(is_control)
        self._pools: dict[int, np.ndarray] = {}
        for g in np.unique(ctrl_group[ctrl_rows]):
            self._pools[int(g)] = ctrl_rows[ctrl_group[ctrl_rows] == g]
        if not self._pools:
            raise ValueError("no control cells found (is_control all False)")

    @property
    def groups(self) -> list[int]:
        return sorted(self._pools)

    def sample(self, group: int, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return ``k`` control-cell row indices for ``group`` (with replacement)."""
        pool = self._pools.get(int(group))
        if pool is None:
            raise KeyError(f"no control pool for ctrl_group={group}")
        idx = torch.randint(len(pool), (k,), generator=generator)
        return torch.as_tensor(pool[idx.numpy()], dtype=torch.long)


# --------------------------------------------------------------------------- #
# Dataset + loaders.
# --------------------------------------------------------------------------- #
class PerturbDataset(Dataset):
    """Per-cell rows for one split; tokenization is done in the collate (batched)."""

    def __init__(self, bundle: CacheBundle, rows: np.ndarray, normalize: str) -> None:
        self.b = bundle
        self.rows = rows
        feat = bundle.hvg_X if normalize == "log1p_cpm" else bundle.counts.astype(np.float32)
        self._feat = torch.from_numpy(np.ascontiguousarray(feat))
        self._counts = torch.from_numpy(np.ascontiguousarray(bundle.counts))
        self._libsize = torch.from_numpy(bundle.libsize)
        self._pert_id = torch.from_numpy(bundle.pert_id)
        self._is_control = torch.from_numpy(bundle.is_control)
        self._ctrl_group = torch.from_numpy(bundle.ctrl_group)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.rows[i])
        return {
            "feat": self._feat[r],
            "counts": self._counts[r],
            "libsize": self._libsize[r],
            "pert_id": self._pert_id[r],
            "is_control": self._is_control[r],
            "ctrl_group": self._ctrl_group[r],
        }


def _make_collate(partition: torch.Tensor, n_hvg: int):
    def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        feat = torch.stack([it["feat"] for it in items])
        return {
            "tokens": tokenize_cells(feat, partition),       # (B, n_tokens, token_dim)
            "counts": torch.stack([it["counts"] for it in items]),
            "libsize": torch.stack([it["libsize"] for it in items]),
            "pert_id": torch.stack([it["pert_id"] for it in items]),
            "is_control": torch.stack([it["is_control"] for it in items]),
            "ctrl_group": torch.stack([it["ctrl_group"] for it in items]),
        }

    return collate


def get_perturbseq_dataloaders(
    batch_size: int = 128,
    data_dir: str | Path = "data",
    artifact: str = "norman2019",
    split: str = "combo",
    limit: int | None = None,
    num_workers: int = 0,
    normalize: str = "log1p_cpm",
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train, val, test)`` loaders of tokenized Perturb-seq cells.

    Parameters
    ----------
    split:
        ``"combo"`` (hold out unseen 2-gene combinations — the Norman headline) or
        ``"cells"`` (random per-cell sanity split).
    normalize:
        ``"log1p_cpm"`` (use the cache's normalized features for tokens) or
        ``"none"`` (tokenize raw counts).
    seed:
        Gene-partition seed; must equal the cache's partition seed (asserted).
    limit:
        Cap the *train* split (fast smoke runs); val/test left intact.

    Each returned loader carries two extra attributes: ``.control_sampler`` (a
    :class:`ControlSampler` over the full cell pool, for baseline ``z_b`` pairing)
    and ``.meta`` (geometry + perturbation vocabulary).
    """
    if split not in ("combo", "cells"):
        raise ValueError(f"split must be 'combo' or 'cells', got {split!r}")
    if normalize not in ("log1p_cpm", "none"):
        raise ValueError(f"normalize must be 'log1p_cpm' or 'none', got {normalize!r}")

    cache_dir = Path(data_dir) / artifact
    bundle = load_cache(cache_dir)
    n_hvg = bundle.n_hvg
    tok = bundle.manifest.get("tokenization", {})
    n_tokens = int(tok.get("n_tokens", DEFAULT_N_TOKENS))
    cache_seed = int(tok.get("partition_seed", seed))
    if seed != cache_seed:
        raise ValueError(
            f"partition seed mismatch: loader seed={seed} but cache was built with "
            f"seed={cache_seed}; pass seed={cache_seed} or rebuild the cache"
        )
    partition = make_gene_partition(n_hvg, n_tokens, seed)

    split_col = bundle.split_combo if split == "combo" else bundle.split_cells
    sampler = ControlSampler(bundle.is_control, bundle.ctrl_group)

    def rows_for(code: int) -> np.ndarray:
        return np.flatnonzero(split_col == code)

    train_rows, val_rows, test_rows = rows_for(SPLIT_TRAIN), rows_for(SPLIT_VAL), rows_for(SPLIT_TEST)
    if limit is not None:
        train_rows = train_rows[:limit]

    collate = _make_collate(partition, n_hvg)
    meta = {
        "n_hvg": n_hvg,
        "n_tokens": n_tokens,
        "token_dim": partition.shape[1],
        "partition_seed": seed,
        "pert_names": list(bundle.pert_names),
        "split": split,
    }

    def loader(rows: np.ndarray, shuffle: bool, drop_last: bool) -> DataLoader:
        dl = DataLoader(
            PerturbDataset(bundle, rows, normalize),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate,
        )
        dl.control_sampler = sampler  # type: ignore[attr-defined]
        dl.meta = meta                # type: ignore[attr-defined]
        return dl

    return (
        loader(train_rows, shuffle=True, drop_last=True),
        loader(val_rows, shuffle=False, drop_last=False),
        loader(test_rows, shuffle=False, drop_last=False),
    )
