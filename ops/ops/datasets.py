"""Storage-dedup helpers: link large shared datasets into the local data dir.

ssl-lab keeps its own small data (SSL-specific e.g. MNIST) as real files under
``data/``. Large datasets shared across projects should live in ONE
project-neutral place — a "data lake" — and be referenced by symlink, not
duplicated. The lake's location is the env var ``SSLLAB_DATA_ROOT`` so it never
gets baked into config or coupled to another project's name.

The lake follows a ``<modality>/<sub-topic>/<dataset>`` layout (e.g.
``scrna/perturb_seq/norman2019``), so datasets stay organized as the lake grows.
A dataset's lake path is its identity; the local symlink defaults to a *flat*
alias (the basename) so project code refers to it by a short name.

Typical use::

    export SSLLAB_DATA_ROOT=~/work/data          # or an existing shared store
    # flat dataset:
    python -c "from ops.datasets import link_dataset; link_dataset('GRCh38')"
    # creates  data/GRCh38 -> $SSLLAB_DATA_ROOT/GRCh38
    # nested (modality) dataset -> flat local alias:
    python -c "from ops.datasets import link_dataset; link_dataset('scrna/perturb_seq/norman2019')"
    # creates  data/norman2019 -> $SSLLAB_DATA_ROOT/scrna/perturb_seq/norman2019

Local-only data (e.g. MNIST) needs none of this — it just lives in ``data/``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "SSLLAB_DATA_ROOT"


def data_root() -> Path | None:
    """The shared data-lake root from ``$SSLLAB_DATA_ROOT``, or ``None`` if unset."""
    val = os.environ.get(DATA_ROOT_ENV)
    return Path(val).expanduser() if val else None


def link_dataset(
    name: str,
    data_dir: str | Path = "data",
    root: str | Path | None = None,
    overwrite: bool = True,
    link_name: str | None = None,
) -> Path:
    """Symlink one shared dataset from the lake into the local data dir.

    ``name`` is the dataset's path *within the lake* — flat (``"GRCh38"``) or
    nested by modality (``"scrna/perturb_seq/norman2019"``). The local symlink is
    created at ``<data_dir>/<link_name>``; ``link_name`` defaults to the basename
    of ``name``, so a nested lake path produces a *flat* local alias
    (``data/norman2019 -> <root>/scrna/perturb_seq/norman2019``). Pass
    ``link_name`` to mirror the nesting locally (e.g. another project's
    convention) — intermediate dirs are created.

    ``root`` defaults to ``$SSLLAB_DATA_ROOT``. Idempotent: re-linking refreshes
    the symlink. Only the named dataset is linked — never the whole tree — so the
    dependency on the shared lake stays explicit and minimal.

    Returns the path of the created symlink.
    """
    lake = Path(root).expanduser() if root is not None else data_root()
    if lake is None:
        raise RuntimeError(
            f"No data lake configured. Set ${DATA_ROOT_ENV}=/path/to/shared/data "
            "or pass root=..."
        )
    src = lake / name
    if not src.exists():
        raise FileNotFoundError(f"dataset {name!r} not found in lake: {src}")

    data_dir = Path(data_dir)
    dst = data_dir / (link_name if link_name is not None else Path(name).name)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() or dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists (overwrite=False)")
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            raise IsADirectoryError(
                f"{dst} is a real directory, not a symlink — refusing to replace. "
                "Move or remove it first."
            )
    dst.symlink_to(src, target_is_directory=src.is_dir())
    return dst
