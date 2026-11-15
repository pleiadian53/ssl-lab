#!/usr/bin/env python3
"""Project wide embedding stores down to 256 dimensions, fit on the training cells only.

**Why.** Geneformer V2's cell embeddings have 768 numbers per cell, three times our JEPA encoder's
256. The conditional flow then has to learn how cells move in a space three times as wide from the
same training cells, so a worse full-pipeline score could come from the width alone rather than
from what the embedding contains. The comparison therefore runs every V2 embedding twice: at full
width, and projected onto its 256 leading principal components, the same width as our encoder.

**Leakage.** The principal components are fit on the combo split's training cells only
(``split_combo == 0``), the cells the decoder and the flow train on. The held-out cells, whose
scores the comparison reports, never shape the coordinates they are later scored in. Every cell,
held-out ones included, is then projected with the fitted components.

**Output.** One new store per parent, written by ``ssllab.embeddings.project_store``. Its manifest
records the parent store, the number of fit cells and the share of the training cells' variance the
256 components keep; that share is reported with the comparison, since it says how much of the
wide embedding's spread survives the projection. The fitted mean and components are saved beside
it in ``projection.npz``.

CPU, a minute or two per store. Runs wherever the stores are, on the pod for the comparison.

Usage (from the repo root):
    python examples/perturbation_response/foundation-encoder/05_project_store.py \\
        --parents output/geneformer_full/stores/V2-104M_cls output/geneformer_full/stores/V2-104M_cell \\
        --out-root output/geneformer_full/stores_p256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TRAIN, TOKENS_NPZ, cache_fingerprint_from_dir
from ssllab.embeddings import load_store, project_store, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parents", nargs="+", required=True, help="store directories to project")
    ap.add_argument("--out-root", required=True, help="each projection is written to <out-root>/<parent name>")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifact", default="norman2019")
    args = ap.parse_args()

    cache_dir = Path(args.data_dir) / args.artifact
    fingerprint = cache_fingerprint_from_dir(cache_dir)
    split = np.load(cache_dir / TOKENS_NPZ, allow_pickle=True)["split_combo"]
    fit_rows = np.flatnonzero(split == SPLIT_TRAIN)

    summary = {}
    for parent_dir in args.parents:
        parent = load_store(parent_dir, expect_fingerprint=fingerprint)
        if parent.width <= args.k:
            sys.exit(f"{parent_dir} is {parent.width} wide; projecting to {args.k} would not reduce it")
        out = Path(args.out_root) / Path(parent_dir).name
        store = project_store(parent, fit_rows, args.k, out,
                              fit_description="combo split, training cells (split_combo == 0)",
                              **provenance(__file__))
        summary[Path(parent_dir).name] = {k: store.manifest[k] for k in
                                          ("width", "n", "n_fit_rows", "explained_variance_share", "content_md5")}
        print(Path(parent_dir).name, json.dumps(summary[Path(parent_dir).name]), flush=True)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "projection_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
