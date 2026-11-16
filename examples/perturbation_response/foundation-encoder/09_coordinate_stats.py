#!/usr/bin/env python3
"""How extreme does each embedding coordinate become once the flow standardizes it?

Why this check exists
---------------------
In round 14, every flow trained on a 768-wide Geneformer V2 embedding blew up: the epoch-mean
flow-matching loss rose to thousands or millions of times its first value, with or without the
gradient clipped at 1.0, while every flow on a 256-wide embedding, including the same V2
embeddings projected onto their 256 leading principal components, trained smoothly. One candidate
cause is the flow's input standardization. ``04_train_cond_flow.py`` rescales every coordinate
separately,

    z_std[i, j] = (z[i, j] - mean_j) / (std_j + 1e-6),

where ``z[i, j]`` is coordinate ``j`` of cell ``i`` and ``mean_j`` and ``std_j`` are that coordinate's
mean and standard deviation over the training cells. A coordinate that barely varies across most
cells but takes a very different value in a few of them becomes, after this rescaling, a
coordinate whose typical value is about 1 and whose rare values are tens or hundreds. A principal
component projection never produces such a coordinate among its leading components, because each
component is chosen to carry as much variance as possible, so the check also explains why the
projected arms would be immune if it finds such coordinates.

What it reports
---------------
For every store named, over the combo split's training cells (the cells the flow trains on), in
float64:

- the spread of the per-coordinate standard deviations: smallest, median, largest and their ratio;
- per coordinate, the largest absolute standardized value and the excess kurtosis (the fourth
  standardized moment minus 3; 0 for a Gaussian, large when a few cells dominate);
- how many coordinates reach a largest absolute standardized value above 10, 30 and 100, and how
  many cells hold at least one value above 10, 30 and 100;
- the twenty most extreme coordinates with their standard deviation, mean, kurtosis and the number
  of cells beyond 10;
- the squared length of the standardized vector per cell, against its expectation, the width ``d``:
  a cell far above ``d`` is far outside the cloud the flow is trained on.

Nothing here trains anything. It reads the stores and the cache's split labels only.

Usage (on the pod, where the stores are; from the repo root):
    python examples/perturbation_response/foundation-encoder/09_coordinate_stats.py \\
        --output-root /runpod-volume/ssl-lab/output --data-dir /runpod-volume/ssl-lab/data \\
        --out /runpod-volume/ssl-lab/output/geneformer_full/coordinate_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ssllab.data.perturbseq import SPLIT_TRAIN, cache_fingerprint, load_cache  # noqa: E402
from ssllab.embeddings import load_store  # noqa: E402

DEFAULT_STORES = [
    "embeddings/norman_jepa_stage_a",
    "geneformer_full/stores/V1-10M_cell",
    "geneformer_full/stores/V2-104M_cls",
    "geneformer_full/stores/V2-104M_cell",
    "geneformer_full/stores/V2-104M_CLcancer_cls",
    "geneformer_full/stores/V2-104M_CLcancer_cell",
    "geneformer_full/stores_p256/V2-104M_cls",
    "geneformer_full/stores_p256/V2-104M_cell",
]
EPS = 1e-6  # the flow trainer's guard
THRESHOLDS = (10, 30, 100)


def coordinate_report(z: np.ndarray, top: int = 20) -> dict:
    """Statistics of every coordinate after the flow's standardization; ``z`` is [cells, d], float64."""
    mean = z.mean(0)
    std = z.std(0)
    zs = (z - mean) / (std + EPS)
    absmax = np.abs(zs).max(0)
    m2 = (zs ** 2).mean(0)
    kurt = (zs ** 4).mean(0) / np.maximum(m2 ** 2, 1e-30) - 3.0
    beyond10 = (np.abs(zs) > 10).sum(0)
    cell_max = np.abs(zs).max(1)
    sq_len = (zs ** 2).sum(1)
    d = z.shape[1]
    order = np.argsort(-absmax)[:top]
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
    return {
        "n_cells": int(z.shape[0]), "width": int(d),
        "std": {"min": float(std.min()), "median": float(np.median(std)), "max": float(std.max()),
                "max_over_min": float(std.max() / max(std.min(), 1e-30)),
                "n_below_1e-3": int((std < 1e-3).sum()), "n_below_1e-4": int((std < 1e-4).sum())},
        "abs_standardized_max": {"median_over_coords": float(np.median(absmax)), "max": float(absmax.max()),
                                 **{f"n_coords_above_{t}": int((absmax > t).sum()) for t in THRESHOLDS}},
        "excess_kurtosis": {"median_over_coords": float(np.median(kurt)), "max": float(kurt.max()),
                            "n_coords_above_10": int((kurt > 10).sum()), "n_coords_above_100": int((kurt > 100).sum())},
        "cells": {**{f"n_with_a_value_above_{t}": int((cell_max > t).sum()) for t in THRESHOLDS},
                  "sq_length_over_width": {"median": q(sq_len, 50) / d, "p99": q(sq_len, 99) / d,
                                           "p99.9": q(sq_len, 99.9) / d, "max": float(sq_len.max()) / d}},
        "most_extreme_coords": [
            {"coord": int(j), "abs_standardized_max": float(absmax[j]), "std": float(std[j]), "mean": float(mean[j]),
             "excess_kurtosis": float(kurt[j]), "n_cells_beyond_10": int(beyond10[j])} for j in order],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifact", default="norman2019")
    ap.add_argument("--stores", nargs="+", default=DEFAULT_STORES, help="store paths under the output root")
    ap.add_argument("--out", default="output/geneformer_full/coordinate_stats.json")
    args = ap.parse_args()

    cache = load_cache(Path(args.data_dir) / args.artifact)
    fp = cache_fingerprint(cache)
    train_rows = np.flatnonzero(cache.split_combo == SPLIT_TRAIN)
    report = {"standardization": "(z - mean) / (std + 1e-6), per coordinate, over combo training cells",
              "n_train_cells": int(len(train_rows)), "stores": {}}
    for rel in args.stores:
        st = load_store(Path(args.output_root) / rel, expect_fingerprint=fp)
        z = np.asarray(st.lookup(train_rows), dtype=np.float64)
        r = coordinate_report(z)
        report["stores"][rel] = r
        a, c = r["abs_standardized_max"], r["cells"]
        print(f"{rel:48s} d {r['width']:4d}  std {r['std']['min']:.2e}..{r['std']['max']:.2e}  "
              f"max|z| {a['max']:9.1f}  coords >10/30/100: {a['n_coords_above_10']}/{a['n_coords_above_30']}/"
              f"{a['n_coords_above_100']}  cells >10/30/100: {c['n_with_a_value_above_10']}/"
              f"{c['n_with_a_value_above_30']}/{c['n_with_a_value_above_100']}  "
              f"kurtosis max {r['excess_kurtosis']['max']:.0f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
