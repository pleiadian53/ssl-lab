"""Narrow cache reads, shared by the scripts in this directory.

``ssllab.data.perturbseq.load_cache`` is the right loader for training: it returns everything,
including ``counts`` (2.2 GB) next to ``hvg_X`` (2.2 GB). The explainers here never touch raw
counts, and paying 4.4 GB of reads and resident memory to print a table is the difference
between a script you run casually and one you do not.

An ``.npz`` is a zip of ``.npy`` members, and ``np.load`` reads a member only when it is
indexed, so asking for two small label arrays costs milliseconds. (``mmap_mode`` does not help
here: numpy memory-maps a bare ``.npy``, not members inside an archive.) This module is the
same trick ``00b_data_card.py`` uses in its ``load_labels``, generalized so three scripts can
share it.

Nothing here reimplements a metric or a model. It is file plumbing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LABELS = ("pert_id", "pert_names", "is_control", "split_combo", "split_cells", "libsize")


def load_narrow(cache_dir: str | Path, *members: str, labels: bool = True) -> SimpleNamespace:
    """Read only the cache members asked for, plus the small JSON sidecars.

    Arguments
        cache_dir   ``<data-dir>/<artifact>``, the directory ``00_process_norman.py`` wrote.
        members     extra npz members to read, e.g. ``"hvg_X"``. Each costs its real size.
        labels      also read the per-cell label arrays (cheap, and almost always wanted).

    Returns a namespace whose attribute names match :class:`~ssllab.data.perturbseq.CacheBundle`,
    so code written against one reads the same against the other, plus ``de_genes`` already
    unwrapped to its ``per_pert`` mapping (the form every evaluator indexes by name).
    """
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / "tokens_meta.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"{npz_path} not found. Build the cache first:\n"
            f"    python examples/perturbation_response/00_process_norman.py --artifact {cache_dir.name}"
        )
    npz = np.load(npz_path, allow_pickle=True)
    want = (*(LABELS if labels else ()), *members)
    out = {k: npz[k] for k in dict.fromkeys(want)}

    de_raw = json.loads((cache_dir / "de_genes.json").read_text())
    out["de_genes"] = de_raw.get("per_pert", de_raw)
    out["de_meta"] = {k: v for k, v in de_raw.items() if k != "per_pert"}
    out["manifest"] = json.loads((cache_dir / "manifest.json").read_text())
    out["splits"] = json.loads((cache_dir / "splits.json").read_text())
    out["cache_dir"] = cache_dir
    return SimpleNamespace(**out)


def top_genes(data: SimpleNamespace, name: str, top_k: int) -> np.ndarray:
    """A perturbation's top-``top_k`` DE gene indices, as the evaluators select them.

    Mirrors ``ssllab.eval.effect_size.run_effect_size_eval``: take the first ``top_k`` of the
    ranked list and drop any index outside the current panel, so a stale DE file cannot index
    past the end of a narrower cache.
    """
    n_genes = int(data.manifest["n_hvg"])
    if name not in data.de_genes:
        raise KeyError(f"{name!r} has no DE record in {data.cache_dir.name}")
    return np.asarray([i for i in data.de_genes[name]["top_idx"][:top_k] if i < n_genes], dtype=int)


def pert_index(data: SimpleNamespace, name: str) -> int:
    """The integer id of a perturbation label, or a readable error listing near misses."""
    names = np.asarray([str(n) for n in data.pert_names])
    hit = np.where(names == name)[0]
    if hit.size:
        return int(hit[0])
    import difflib

    close = difflib.get_close_matches(name, list(names), n=5)
    hint = f"; close matches: {', '.join(close)}" if close else ""
    raise SystemExit(f"{name!r} is not a perturbation in {data.cache_dir.name}{hint}")


def control_mean(data: SimpleNamespace) -> np.ndarray:
    """Mean control profile over the training-side control cells, in float64.

    The same reference the additive predictor and every Δ in the project is measured against
    (``ssllab.eval.baselines.additive_predictor``). Train-side only, because a reference built
    from held-out cells would leak.
    """
    from ssllab.data.perturbseq import SPLIT_TRAIN

    mask = data.is_control.astype(bool) & (data.split_combo == SPLIT_TRAIN)
    return data.hvg_X[mask].mean(0).astype(np.float64)
