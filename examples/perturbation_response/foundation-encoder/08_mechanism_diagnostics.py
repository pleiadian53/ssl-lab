#!/usr/bin/env python3
"""Two descriptive checks of how Geneformer uses the deep end of a cell's gene list. Pod only.

**The question.** The V1 models read at most 2,048 genes per cell and the V2 models up to 4,094, so
V2 looks as if it sees a third more of each Norman cell. But a slot deep in the list is filled only
by cells that express that many genes, so during pretraining the position vectors for slots past
about 2,000 were trained on fewer examples, and the genes sitting there are the barely expressed
ones whose order is mostly decided by ties. V2 may therefore make little use of its extra input.
Neither check below is an endpoint of the comparison; they help explain a V1-versus-V2 difference
if one appears.

**Check 1, cutting the list.** The same cells are embedded twice with a V2 model: once with their
full gene lists, and once with every list cut after its first ``--cut`` genes (2,048 by default,
V1's limit), keeping the ``<cls>`` and ``<eos>`` tokens. For each cell the script reports how far the
embedding moved, as the cosine similarity between the two versions and as the distance between
them divided by the median distance between two different cells' full embeddings, so a move can be
read against how far apart cells normally are. Cells whose list was not long enough to be cut are
reported separately; for them nothing changed in the input, so they also check the procedure.

**Check 2, the position table.** The learned position table is read straight from the model's
weights (``bert.embeddings.position_embeddings``), one row per slot. For every slot the script
records the row's length and its cosine similarity to the next row. Well-trained neighboring slots
tend to have similar rows; rows that were rarely trained stay closer to their random starting
values, which are unrelated to their neighbors.

**Input.** The tokenized V2 dataset of the feasibility run (``tokens_V2.dataset`` under its output
directory, 2,000 cells), so nothing is re-tokenized.

Usage (on the pod, from the repo root):
    python examples/perturbation_response/foundation-encoder/08_mechanism_diagnostics.py \\
        --gf-root /runpod-volume/ssl-lab/models/geneformer \\
        --tokens /runpod-volume/ssl-lab/output/geneformer_feasibility/tokens_V2.dataset \\
        --out-dir /runpod-volume/ssl-lab/output/geneformer_mechanism
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _embed_module():
    spec = importlib.util.spec_from_file_location("gf_embed", _HERE / "02_embed_geneformer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cut_lists(ds, cut: int):
    """Keep ``<cls>``, the first ``cut`` genes and ``<eos>`` of every V2 list."""
    def f(ex):
        ids = list(ex["input_ids"])
        if len(ids) - 2 > cut:
            ids = ids[: cut + 1] + [ids[-1]]
        return {"input_ids": ids, "length": len(ids)}
    return ds.map(f)


def movement(full: np.ndarray, cut: np.ndarray, affected: np.ndarray, seed: int = 0) -> dict:
    cos = (full * cut).sum(1) / (np.linalg.norm(full, axis=1) * np.linalg.norm(cut, axis=1))
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(full), 20000), rng.integers(0, len(full), 20000)
    keep = a != b
    typical = float(np.median(np.linalg.norm(full[a[keep]] - full[b[keep]], axis=1)))
    shift = np.linalg.norm(full - cut, axis=1) / typical

    def summ(x):
        return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None,
                "p05": float(np.percentile(x, 5)) if len(x) else None,
                "p95": float(np.percentile(x, 95)) if len(x) else None}

    return {"median_between_cell_distance": typical,
            "cut_cells": {"cosine": summ(cos[affected]), "shift_over_typical_distance": summ(shift[affected])},
            "uncut_cells": {"cosine": summ(cos[~affected]), "shift_over_typical_distance": summ(shift[~affected])}}


def position_table(model_dir: Path) -> dict:
    from transformers import BertForMaskedLM

    model = BertForMaskedLM.from_pretrained(str(model_dir))
    W = model.bert.embeddings.position_embeddings.weight.detach().cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(W, axis=1)
    nxt = (W[:-1] * W[1:]).sum(1) / (norms[:-1] * norms[1:])
    bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2048), (2048, 2500), (2500, 3000), (3000, 3500), (3500, len(W))]
    by_bin = {f"{lo}-{min(hi, len(W))}": {"norm_median": float(np.median(norms[lo:hi])),
                                          "neighbor_cosine_median": float(np.median(nxt[lo:min(hi, len(nxt))]))}
              for lo, hi in bins if lo < len(W)}
    return {"n_rows": int(len(W)), "width": int(W.shape[1]), "by_slot_range": by_bin,
            "row_norm": norms.round(5).tolist(), "neighbor_cosine": nxt.round(5).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gf-root", required=True)
    ap.add_argument("--tokens", required=True, help="tokens_V2.dataset of the feasibility run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="V2-104M")
    ap.add_argument("--cut", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk

    if not torch.cuda.is_available():
        sys.exit("Geneformer's extractor needs CUDA; run this on the pod.")
    sys.path.insert(0, str(Path(args.gf_root)))
    emb = _embed_module()
    spec = emb.MODELS[args.model]
    if spec["version"] != "V2":
        sys.exit("the cut check compares against V2's longer input; pick a V2 model")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "work").mkdir(exist_ok=True)  # Geneformer writes its CSV here and does not create it
    model_dir = Path(args.gf_root) / spec["dir"]

    ds = load_from_disk(args.tokens)
    cut_path = out / f"tokens_V2_cut{args.cut}.dataset"
    cut_lists(ds, args.cut).save_to_disk(str(cut_path))
    n_genes = {int(c): int(n) - 2 for c, n in zip(ds["cell_idx"], ds["length"])}

    report = {"model": args.model, "cut": args.cut, "n_cells": len(ds), "modes": {}}
    for mode in spec["modes"]:
        full, idx_f, _ = emb.embed(model_dir, Path(args.tokens), "V2", mode, args.batch_size, args.nproc,
                                   out / "work", spec["dim"])
        cut, idx_c, _ = emb.embed(model_dir, cut_path, "V2", mode, args.batch_size, args.nproc,
                                  out / "work", spec["dim"])
        if not np.array_equal(idx_f, idx_c):
            sys.exit("the full and cut runs embedded different cells")
        affected = np.array([n_genes[int(c)] > args.cut for c in idx_f])
        report["modes"][mode] = {"n_cells_cut": int(affected.sum()), **movement(full, cut, affected)}
        print(mode, json.dumps(report["modes"][mode]), flush=True)
        (out / "mechanism.json").write_text(json.dumps(report, indent=2))

    report["position_table"] = {args.model: position_table(model_dir),
                                "V1-10M": position_table(Path(args.gf_root) / emb.MODELS["V1-10M"]["dir"])}
    for name, t in report["position_table"].items():
        print(name, json.dumps(t["by_slot_range"]), flush=True)
    (out / "mechanism.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "mechanism.json")


if __name__ == "__main__":
    main()
