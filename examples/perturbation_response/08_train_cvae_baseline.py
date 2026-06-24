"""Baseline — train the from-scratch conditional NB-VAE (no JEPA, no flow).

The control the full method must beat. Trains :class:`ConditionalNBVAE` end-to-end on
counts, conditioned on the SAME gene-compositional perturbation embedding the flow
uses, so the comparison isolates the generative machinery (JEPA latent + flow prior)
rather than the perturbation encoding. Default split is **combo** — the held-out-combo
generalization test where the compositional embedding matters.

Usage
-----
    python examples/perturbation_response/08_train_cvae_baseline.py --epochs 60
    python examples/perturbation_response/08_train_cvae_baseline.py --limit 3000 --epochs 3   # smoke

Output
------
    output/<experiment>/checkpoints/cvae_baseline.pt
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

from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.experiment import experiment
from ssllab.generative.condition import build_pert_gene_matrix
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the conditional NB-VAE baseline.")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"],
                   help="combo = held-out-combo generalization (default); cells = in-distribution")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--gene-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--compose", type=str, default="additive", choices=["additive", "deepsets"])
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE)")
    p.add_argument("--grad-clip", type=float, default=5.0, help="max grad norm (VAE stability)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    train_loader, _, _ = get_perturbseq_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir, artifact=args.artifact,
        split=args.split, limit=args.limit, seed=0,
    )
    n_genes = train_loader.meta["n_hvg"]
    pert_gene, gene_vocab = build_pert_gene_matrix(train_loader.meta["pert_names"])
    model = ConditionalNBVAE(
        n_genes=n_genes, pert_gene=pert_gene, latent_dim=args.latent_dim,
        cond_dim=args.cond_dim, gene_dim=args.gene_dim, hidden=args.hidden, compose=args.compose,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger.info("cvae baseline: genes=%d latent=%d cond=%d genes_vocab=%d compose=%s split=%s cells=%d",
                n_genes, args.latent_dim, args.cond_dim, len(gene_vocab), args.compose, args.split,
                len(train_loader.dataset))

    avg = float("nan")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["features"].float().to(device)
            counts = batch["counts"].float().to(device)
            libsize = batch["libsize"].to(device)
            pid = batch["pert_id"].to(device)
            loss, _ = model.loss(x, counts, libsize, pid, beta=args.beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # VAE stability
            opt.step()
            running += loss.item()
        avg = running / max(len(train_loader), 1)
        logger.info("epoch %d/%d  loss=%.3f", epoch + 1, args.epochs, avg)

    out = exp.checkpoints / "cvae_baseline.pt"
    torch.save({
        "state": model.state_dict(), "n_genes": n_genes, "latent_dim": args.latent_dim,
        "cond_dim": args.cond_dim, "gene_dim": args.gene_dim, "hidden": args.hidden,
        "compose": args.compose, "pert_gene": pert_gene,
    }, out)
    exp.write_report("baseline_cvae", {"final_loss": avg, "split": args.split, "beta": args.beta,
                                       "compose": args.compose, "train_cells": len(train_loader.dataset)})
    logger.info("saved cvae baseline -> %s", out)


if __name__ == "__main__":
    main()
