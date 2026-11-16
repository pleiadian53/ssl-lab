#!/usr/bin/env python3
"""What would it cost to train Geneformer V2-104M jointly with the count decoder? A benchmark. Pod only.

**Why.** Round 14's protocol fixed a trigger for a follow-up round: if the best frozen Geneformer
arm came within 0.02 of our JEPA encoder at the readout ceiling, train it jointly with the decoder,
against a random-weight twin, as round 13 did for our encoder. The trigger fired (0.908 against
0.863). Round 13's recipe trains the encoder and the decoder together for 30 epochs over the 96,322
training cells. Our JEPA encoder is small, so that took about 30 minutes; V2-104M has 104 million
parameters and reads up to 4,096 genes per cell, so the same recipe could take hundreds of
GPU-hours. Nothing is trained for that round before its recipe and cost are approved, and this
script measures what the cost would be.

**What it measures.**

1. *Equivalence.* For 256 training cells, the cell embedding computed here in plain PyTorch, the
   mean of the second-to-last layer's hidden states over each cell's gene tokens (``<cls>``,
   ``<eos>`` and padding excluded), is compared with round 14's frozen ``V2-104M_cell`` store, which
   Geneformer's own extractor wrote. A jointly trained encoder must be read exactly as the frozen
   one was, or the comparison with the frozen arm would mix two definitions of the embedding. It
   passes when every cell's relative difference is below 1e-2 and its cosine similarity above
   0.9999, a tolerance that admits the extractor computing in reduced precision.
2. *Training speed.* Full optimizer steps (forward, NB loss of a fresh count decoder, backward,
   AdamW on every encoder and decoder parameter) for each combination of precision (fp32, or bf16
   autocast), list length (as tokenized, up to 4,096, or cut at 2,048) and batch size (8, 16, 32
   cells), on randomly drawn training cells. It records seconds per step and peak GPU memory, and
   projects hours per epoch over the 96,322 training cells. A configuration that runs out of
   memory is recorded as such.

Usage (on the pod, from the repo root):
    python examples/perturbation_response/foundation-encoder/12_joint_benchmark.py \\
        --gf-root /runpod-volume/ssl-lab/models/geneformer --data-dir /runpod-volume/ssl-lab/data \\
        --tokens /runpod-volume/ssl-lab/output/geneformer_full/tokens_V2.dataset \\
        --store /runpod-volume/ssl-lab/output/geneformer_full/stores/V2-104M_cell \\
        --out /runpod-volume/ssl-lab/output/geneformer_joint_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TRAIN, load_cache  # noqa: E402
from ssllab.embeddings import load_store  # noqa: E402

N_TRAIN = 96_322


def special_ids(gf_root: Path) -> dict:
    """``<pad>``, ``<cls>``, ``<eos>`` ids from the V2 token dictionary shipped with Geneformer."""
    sys.path.insert(0, str(gf_root))
    import geneformer

    path = getattr(geneformer, "TOKEN_DICTIONARY_FILE", None) or next(Path(geneformer.__file__).parent.rglob("token_dictionary_gc104M.pkl"))
    with open(path, "rb") as f:
        tok = pickle.load(f)
    return {k: tok[k] for k in ("<pad>", "<cls>", "<eos>")}


def batch_of(lists: list[list[int]], pad: int, cut: int | None, device):
    """Pad token lists into ``(input_ids, attention_mask)``; optionally keep ``<cls>``, ``cut`` genes, ``<eos>``."""
    import torch

    if cut is not None:
        lists = [ids[: cut + 1] + [ids[-1]] if len(ids) - 2 > cut else ids for ids in lists]
    n = max(len(x) for x in lists)
    ids = torch.full((len(lists), n), pad, dtype=torch.long)
    mask = torch.zeros((len(lists), n), dtype=torch.long)
    for i, x in enumerate(lists):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return ids.to(device), mask.to(device)


def cell_embedding(model, ids, mask, sp: dict):
    """Mean of the second-to-last layer over gene tokens: what the extractor's ``cell`` mode writes."""
    out = model.bert(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    h = out.hidden_states[-2]                     # hidden_states[0] is the embedding layer; [-1] the last layer
    gene = mask.bool() & (ids != sp["<cls>"]) & (ids != sp["<eos>"])
    w = gene.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp_min(1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gf-root", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokens", required=True, help="round 14's full tokens_V2.dataset")
    ap.add_argument("--store", required=True, help="round 14's V2-104M_cell store")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8, help="timed steps per configuration, after 2 warm-up steps")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    args = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import BertForMaskedLM

    from ssllab.generative.count_decoder import CountDecoder

    if not torch.cuda.is_available():
        sys.exit("run this on the pod")
    dev = torch.device("cuda")
    gf_root = Path(args.gf_root)
    sp = special_ids(gf_root)
    cache = load_cache(Path(args.data_dir) / "norman2019")
    ds = load_from_disk(args.tokens)
    pos = {int(c): i for i, c in enumerate(ds["cell_idx"])}
    train = np.array([r for r in np.flatnonzero(cache.split_combo == SPLIT_TRAIN) if r in pos])
    rng = np.random.default_rng(0)
    lists_of = lambda rows: [list(ds[pos[int(r)]]["input_ids"]) for r in rows]  # noqa: E731
    report: dict = {"gpu": torch.cuda.get_device_name(0), "n_train_cells": N_TRAIN, "special_ids": sp}

    # 1. equivalence with the frozen store
    model = BertForMaskedLM.from_pretrained(str(gf_root / "Geneformer-V2-104M")).to(dev).eval()
    store = load_store(args.store)
    rows = np.sort(rng.choice(train, args.n_check, replace=False))
    embs = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            ids, mask = batch_of(lists_of(rows[i : i + 16]), sp["<pad>"], None, dev)
            embs.append(cell_embedding(model, ids, mask, sp).float().cpu().numpy())
    here, ref = np.concatenate(embs), store.lookup(rows)
    cos = (here * ref).sum(1) / (np.linalg.norm(here, axis=1) * np.linalg.norm(ref, axis=1))
    rel = np.linalg.norm(here - ref, axis=1) / np.linalg.norm(ref, axis=1)
    report["equivalence"] = {"n_cells": int(len(rows)), "max_abs_diff": float(np.abs(here - ref).max()),
                             "max_relative_diff": float(rel.max()), "min_cosine": float(cos.min()),
                             # the extractor may run in reduced precision, so both measures are required
                             "passed": bool(rel.max() < 1e-2 and cos.min() > 0.9999)}
    print("equivalence", json.dumps(report["equivalence"]), flush=True)
    lengths = np.asarray(ds.select(range(0, len(ds), 50))["length"])
    report["list_length"] = {"median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)),
                             "max": int(lengths.max())}

    # 2. training speed
    report["speed"] = []
    for precision in ("fp32", "bf16"):
        for cut in (None, 2048):
            for bsz in args.batches:
                cfg = {"precision": precision, "cut": cut, "batch": bsz}
                model = BertForMaskedLM.from_pretrained(str(gf_root / "Geneformer-V2-104M")).to(dev).train()
                dec = CountDecoder(latent_dim=model.config.hidden_size, n_genes=cache.n_hvg).to(dev)
                opt = torch.optim.AdamW([{"params": model.bert.parameters(), "lr": 2e-5},
                                         {"params": dec.parameters(), "lr": 1e-3}])
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    times = []
                    for step in range(args.steps + 2):
                        rows = rng.choice(train, bsz, replace=False)
                        ids, mask = batch_of(lists_of(rows), sp["<pad>"], cut, dev)
                        counts = torch.from_numpy(cache.counts[rows]).float().to(dev)
                        lib = counts.sum(-1).clamp_min(1.0)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                            z = cell_embedding(model, ids, mask, sp)
                        loss = dec.loss(z.float(), counts, lib)
                        loss = loss[0] if isinstance(loss, tuple) else loss
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(dec.parameters()), 5.0)
                        opt.step()
                        torch.cuda.synchronize()
                        if step >= 2:
                            times.append(time.perf_counter() - t0)
                    s = float(np.median(times))
                    cfg.update(seconds_per_step=s, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                               hours_per_epoch=s * N_TRAIN / bsz / 3600, loss_finite=bool(np.isfinite(float(loss))))
                except torch.cuda.OutOfMemoryError:
                    cfg.update(out_of_memory=True)
                del model, dec, opt
                torch.cuda.empty_cache()
                report["speed"].append(cfg)
                print("speed", json.dumps(cfg), flush=True)
                Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
