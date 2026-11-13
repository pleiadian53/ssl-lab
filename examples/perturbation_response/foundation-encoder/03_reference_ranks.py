#!/usr/bin/env python3
"""Effective rank of our own representations on the cells the Geneformer feasibility run embedded.

**Why.** The feasibility run (``02_embed_geneformer.py --n-cells 2000``) reports the effective rank
of every Geneformer embedding on a random sample of 2,000 processed cells. A number like "452" means
little on its own. This script computes the same measure, with the same function, on the same 2,000
cells, for three representations we already have, so the Geneformer column can be read against them:

- ``jepa_pretrained``: the frozen JEPA encoder of the published pipeline (``norman_stage_a``), the
  256-dimensional latent that the count decoder and the flow read.
- ``jepa_random``: the same architecture with freshly initialized weights, never trained. A random
  network also spreads its outputs over many directions, so this shows how much of a high rank
  comes from the architecture alone rather than from anything learned.
- ``input_hvg``: the encoder's input itself, the 5,000 log-normalized expression values per cell.

**The measure.** Effective rank is the exponential of the entropy of the normalized singular values
of the centered embedding matrix (``ssllab.eval.collapse.effective_rank``, the function the JEPA
training diagnostics and ``02``'s ``describe`` both use). Its ceiling is the smaller of the width
and the number of cells, so a 256-wide JEPA can reach at most 256 while a 768-wide Geneformer model
can reach 768. The report therefore also gives the rank as a fraction of that ceiling. Neither
number says whether the directions carry the perturbation response rather than technical
variation; that is what the comparison itself has to measure.

**The cells.** ``select_cells(109_737, n, seed)`` from ``02``, the same call the feasibility run made,
so with the default ``--n-cells 2000 --seed 0`` these are exactly its cells. If a feasibility report
is given, its seed and cell count are checked, and every model must have dropped no cell.

Light enough for a laptop: it reads one 2.2 GB array from the cache and encodes 2,000 cells on the
CPU. Nothing here touches Geneformer.

Usage (from the repo root):
    python examples/perturbation_response/foundation-encoder/03_reference_ranks.py \\
        --feasibility output/geneformer_feasibility/feasibility.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ssllab.checkpoint import load_jepa, random_jepa_like
from ssllab.data.perturbseq import DEFAULT_N_TOKENS, TOKENS_NPZ, make_gene_partition
from ssllab.latents import encode_rows
from ssllab.utils import set_seed

_HERE = Path(__file__).resolve().parent


def _embed_module():
    spec = importlib.util.spec_from_file_location("gf_embed", _HERE / "02_embed_geneformer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rank_entry(z: np.ndarray, describe) -> dict:
    """``describe`` from ``02`` plus the rank as a fraction of its ceiling, min(width, cells)."""
    d = describe(np.ascontiguousarray(z, dtype=np.float32))
    d["rank_ceiling"] = int(min(d["dim"], d["n"]))
    d["rank_fraction_of_ceiling"] = d["effective_rank"] / d["rank_ceiling"]
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifact", default="norman2019")
    ap.add_argument("--encoder", default="output/norman_stage_a/checkpoints/encoder.pt")
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feasibility", default=None, help="feasibility.json to check the cell selection against")
    ap.add_argument("--out", default="output/geneformer_feasibility/reference_ranks.json")
    args = ap.parse_args()

    emb = _embed_module()
    cache_dir = Path(args.data_dir) / args.artifact
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if int(manifest["n_cells"]) != emb.N_PROCESSED_CELLS:
        raise ValueError(f"cache has {manifest['n_cells']} cells, Geneformer run assumed {emb.N_PROCESSED_CELLS}")
    rows = emb.select_cells(emb.N_PROCESSED_CELLS, args.n_cells, args.seed)

    feas = None
    if args.feasibility:
        feas = json.loads(Path(args.feasibility).read_text())
        if feas["seed"] != args.seed or feas["n_cells_requested"] != args.n_cells:
            raise ValueError(f"feasibility run used seed {feas['seed']}, {feas['n_cells_requested']} cells")
        dropped = {k: v.get("n_dropped") for k, v in feas["models"].items() if v.get("n_dropped")}
        if dropped:
            raise ValueError(f"feasibility run dropped cells, row sets differ: {dropped}")

    # Only the HVG matrix is needed (stored uncompressed in the npz, so this reads 2.2 GB, not 4.4).
    hvg = np.load(cache_dir / TOKENS_NPZ, allow_pickle=True)["hvg_X"]
    feat = np.ascontiguousarray(hvg[rows])
    del hvg
    tok = manifest.get("tokenization", {})
    partition = make_gene_partition(
        feat.shape[1], int(tok.get("n_tokens", DEFAULT_N_TOKENS)), int(tok.get("partition_seed", 0))
    )
    all_rows = np.arange(len(rows))

    set_seed(args.seed)
    z_pre = encode_rows(load_jepa(args.encoder, "cpu"), feat, partition, all_rows, "cpu").numpy()
    set_seed(args.seed)
    z_rand = encode_rows(random_jepa_like(args.encoder, "cpu"), feat, partition, all_rows, "cpu").numpy()

    report = {
        "n_cells": int(len(rows)),
        "seed": args.seed,
        "encoder": args.encoder,
        "rows_first_last": [int(rows[0]), int(rows[-1])],
        "representations": {
            "jepa_pretrained": rank_entry(z_pre, emb.describe),
            "jepa_random": rank_entry(z_rand, emb.describe),
            "input_hvg": rank_entry(feat, emb.describe),
        },
    }
    if feas is not None:
        report["geneformer"] = {
            k: {"dim": v["dim"], "effective_rank": v["effective_rank"],
                "rank_fraction_of_ceiling": v["effective_rank"] / min(v["dim"], v["n"])}
            for k, v in feas["models"].items()
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for name, r in {**report["representations"], **report.get("geneformer", {})}.items():
        print(f"{name:24s} dim {r['dim']:5d}  effective rank {r['effective_rank']:7.1f}  "
              f"fraction of ceiling {r['rank_fraction_of_ceiling']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
