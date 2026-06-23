"""Stage A — pretrain an intra-cell JEPA encoder on Norman 2019 cells.

Self-supervised, no labels: mask a subset of a cell's gene-group tokens and
predict their embeddings from the visible ones (the same I-JEPA objective the
MNIST encoder uses, since the masking and model are modality-agnostic). The
result is one pooled latent ``z`` per cell — the substrate the downstream count
decoder and conditional flow prior act on.

This is the bio sibling of ``examples/jepa_basics/01_train_jepa_mnist.py``: the
*only* change is the data seam — cells arrive already tokenized as
``(B, n_tokens, token_dim)`` from the perturbseq loader, so there is no patchify.

Usage
-----
    # full run (a pod): all combo-train cells
    python examples/perturbation_response/01_pretrain_stage_a.py --epochs 50 --reg-coef 0.04
    # local CPU smoke
    python examples/perturbation_response/01_pretrain_stage_a.py --limit 3000 --epochs 3 --embed-dim 128

Output
------
    output/<experiment>/checkpoints/encoder.pt   JEPA config + weights (consumed by 02 + Stage B/C)
    output/<experiment>/reports/stage_a_train.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import save_jepa
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.eval.collapse import collapse_report
from ssllab.experiment import experiment
from ssllab.jepa.model import build_jepa
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain an intra-cell JEPA encoder on Perturb-seq cells.")
    p.add_argument("--experiment", type=str, default="norman_stage_a", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019", help="processed cache under <data-dir>/")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"],
                   help="pretrain on the train side of this split (combo = no test-combo leakage)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=6)
    p.add_argument("--pred-depth", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-target", type=int, default=12, help="# gene-group tokens to mask/predict (< n_tokens)")
    p.add_argument("--reg-coef", type=float, default=0.04, help="VICReg weight (collapse guard; 0 disables)")
    p.add_argument("--limit", type=int, default=None, help="cap train cells (smoke runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _tokens(batch: dict, device) -> torch.Tensor:
    return batch["tokens"].to(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    train_loader, val_loader, _ = get_perturbseq_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir, artifact=args.artifact,
        split=args.split, limit=args.limit, seed=0,
    )
    meta = train_loader.meta
    logger.info("device=%s exp=%s  geometry %d tokens x %d dim  train cells=%d",
                device, exp.root, meta["n_tokens"], meta["token_dim"], len(train_loader.dataset))

    jepa = build_jepa(
        token_dim=meta["token_dim"],
        n_tokens=meta["n_tokens"],
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        n_heads=args.n_heads,
        n_target=args.n_target,
        reg_coef=args.reg_coef,
    ).to(device)
    jepa.ema.to(device)
    opt = torch.optim.AdamW(jepa.parameters(), lr=args.lr)

    total_steps = args.epochs * max(len(train_loader), 1)
    step = 0
    rep = {"effective_rank": float("nan"), "feature_std": float("nan"), "embed_dim": args.embed_dim}
    avg = float("nan")
    for epoch in range(args.epochs):
        jepa.train()
        running = 0.0
        for batch in train_loader:
            loss, comp = jepa(_tokens(batch, device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            jepa.update_target(step, total_steps)
            running += comp["loss"]
            step += 1
        avg = running / max(len(train_loader), 1)

        # Collapse diagnostics on a held-out (val) batch.
        jepa.eval()
        with torch.no_grad():
            diag_loader = val_loader if len(val_loader.dataset) else train_loader
            z = jepa.embed(_tokens(next(iter(diag_loader)), device))
        rep = collapse_report(z)
        logger.info(
            "epoch %d/%d  loss=%.4f  eff_rank=%.2f/%d  feat_std=%.3f",
            epoch + 1, args.epochs, avg, rep["effective_rank"], rep["embed_dim"], rep["feature_std"],
        )

    path = save_jepa(jepa, exp.checkpoints / "encoder.pt")
    report = exp.write_report("stage_a_train", {
        "experiment": args.experiment,
        "artifact": args.artifact,
        "split": args.split,
        "epochs": args.epochs,
        "n_target": args.n_target,
        "reg_coef": args.reg_coef,
        "geometry": {"n_tokens": meta["n_tokens"], "token_dim": meta["token_dim"]},
        "train_cells": len(train_loader.dataset),
        "final_loss": avg,
        **rep,
    })
    logger.info("saved encoder -> %s ; report -> %s", path, report)


if __name__ == "__main__":
    main()
