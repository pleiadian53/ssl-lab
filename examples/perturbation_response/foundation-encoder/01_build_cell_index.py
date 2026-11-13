#!/usr/bin/env python3
"""Map every cell of the processed Norman cache to its row in the raw file, for foundation encoders.

A pretrained single-cell foundation model such as Geneformer does not read our 5,000-gene panel.
It reads each cell's whole transcriptome as raw counts, keyed by Ensembl gene IDs, which only the
raw file ``raw/norman_2019.h5ad`` carries (19,018 genes, 111,255 cells before quality control).
Everything downstream in ssl-lab (decoder, flows, scoring) works on the processed cache, whose
109,737 cells are an ordered subset of the raw rows. The embeddings a foundation model produces
therefore have to be computed from the raw rows and then put back in the processed order, so that
embedding ``i`` belongs to the same cell as row ``i`` of every other array in the project.

This script writes that correspondence once, as a small file next to the cache:

    <data-dir>/norman2019/foundation/cell_index.npz
        raw_row        int64 [n_cells]   row of each processed cell in the raw file
        obs_name       str   [n_cells]   cell barcode, identical in both files
        perturbation   str   [n_cells]   condition label, as in the processed cache
        split_combo    int8  [n_cells]   combination split, 0 train, 1 val, 2 test (ssllab.data.perturbseq)
        is_control     bool  [n_cells]
        total_counts   float [n_cells]   UMI total over all raw genes (Geneformer's ``n_counts``)

It opens both files in backed mode and reads only cell names and labels, never the expression
matrix, so it is cheap and runs locally. It refuses to write if any processed cell is missing
from the raw file or if the order differs, because a silent misalignment would corrupt every
later comparison.

Usage:
    python examples/perturbation_response/foundation-encoder/01_build_cell_index.py
    python examples/perturbation_response/foundation-encoder/01_build_cell_index.py --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from ssllab.data.perturbseq import _SPLIT_CODE


def build_index(raw: ad.AnnData, processed: ad.AnnData) -> dict[str, np.ndarray]:
    """Return the processed→raw row mapping and the labels later steps need.

    Raises ``ValueError`` if a processed cell is absent from the raw file or the order differs.
    """
    if not raw.obs_names.is_unique:
        raise ValueError("raw cell names are not unique; cannot map cells by name")
    pos = raw.obs_names.get_indexer(processed.obs_names)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} processed cells are not in the raw file")
    if not np.all(np.diff(pos) > 0):
        raise ValueError("processed cells are not in raw-file order")
    obs = processed.obs
    return {
        "raw_row": pos.astype(np.int64),
        "obs_name": np.asarray(processed.obs_names, dtype=str),
        "perturbation": obs["perturbation"].astype(str).to_numpy(),
        "split_combo": obs["split_combo"].to_numpy().astype(np.int8),
        "is_control": obs["is_control"].to_numpy(dtype=bool),
        "total_counts": raw.obs["total_counts"].to_numpy(dtype=np.float64)[pos],
    }


def _md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder holding norman2019/ (default: data)")
    args = ap.parse_args()

    root = Path(args.data_dir) / "norman2019"
    raw_path, proc_path = root / "raw" / "norman_2019.h5ad", root / "processed.h5ad"
    raw = ad.read_h5ad(raw_path, backed="r")
    processed = ad.read_h5ad(proc_path, backed="r")
    index = build_index(raw, processed)

    out = root / "foundation"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cell_index.npz", **index)
    meta = {
        "n_processed_cells": int(len(index["raw_row"])),
        "n_raw_cells": int(raw.n_obs),
        "n_raw_genes": int(raw.n_vars),
        "raw_md5": _md5(raw_path),
        "split_counts": {name: int((index["split_combo"] == code).sum()) for name, code in _SPLIT_CODE.items()},
    }
    (out / "cell_index.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out / 'cell_index.npz'}")


if __name__ == "__main__":
    main()
