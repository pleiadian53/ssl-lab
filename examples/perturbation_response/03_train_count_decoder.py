"""Stage C — train the NB/ZINB count decoder on frozen cell latents.

The G2 piece: learn ``z -> gene counts`` so the model emits *data* (where effect
size lives), not just a latent. The encoder stays frozen; we fit a count decoder
by the negative-binomial likelihood of the real counts. Bio sibling of
``examples/generative_jepa/03_train_decoder.py`` — the only changes are the data
seam (perturbseq dict batches) and the likelihood (NB/ZINB instead of pixel BCE).

Usage
-----
    python examples/perturbation_response/03_train_count_decoder.py --epochs 30
    python examples/perturbation_response/03_train_count_decoder.py --limit 3000 --epochs 3   # smoke

Output
------
    output/<experiment>/checkpoints/count_decoder.pt
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

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.experiment import experiment
from ssllab.generative.count_decoder import CountDecoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an NB/ZINB count decoder on frozen cell latents.")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--encoder", type=str, default=None, help="override encoder path")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="cells", choices=["combo", "cells"],
                   help="cells = in-distribution effect-size test (default); see 06")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--zinb", action="store_true", help="zero-inflated NB (adds a dropout gate)")
    p.add_argument("--anchored-mean", action="store_true",
                   help="B1: rho = softmax(log rho_base + delta(z)); baseline = control profile")
    p.add_argument("--state-dispersion", action="store_true",
                   help="B2: per-cell kappa head instead of one constant per gene")
    p.add_argument("--anchor-weight", type=float, default=0.1,
                   help="weight on the moment-of-moments dispersion anchor (state-dispersion only)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()
    encoder_path = args.encoder or (exp.checkpoints / "encoder.pt")

    jepa = load_jepa(encoder_path, device)
    train_loader, _, _ = get_perturbseq_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir, artifact=args.artifact,
        split=args.split, limit=args.limit, seed=0,
    )
    n_genes = train_loader.meta["n_hvg"]
    decoder = CountDecoder(
        latent_dim=jepa.cfg.embed_dim, n_genes=n_genes, zinb=args.zinb,
        anchored_mean=args.anchored_mean, state_dispersion=args.state_dispersion,
    ).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
    logger.info("encoder dim=%d  genes=%d  zinb=%s  anchored_mean=%s  state_dispersion=%s  train cells=%d",
                jepa.cfg.embed_dim, n_genes, args.zinb, args.anchored_mean, args.state_dispersion,
                len(train_loader.dataset))

    # Control pre-pass: baseline rate profile (B1) and the moment-of-moments dispersion
    # target (B2), both estimated from the control population only.
    kappa_target = None
    if args.anchored_mean or (args.state_dispersion and args.anchor_weight > 0):
        rate_sum = torch.zeros(n_genes)
        cnt_sum = torch.zeros(n_genes)
        cnt_sumsq = torch.zeros(n_genes)
        n_ctrl = 0
        for batch in train_loader:
            m = batch["is_control"].bool()
            if not m.any():
                continue
            c = batch["counts"][m].float()
            ell = batch["libsize"][m].float().clamp_min(1.0)
            rate_sum += (c / ell.unsqueeze(-1)).sum(0)
            cnt_sum += c.sum(0)
            cnt_sumsq += (c ** 2).sum(0)
            n_ctrl += int(m.sum())
        if n_ctrl == 0:
            logger.warning("no control cells found; skipping baseline profile and dispersion anchor")
        else:
            if args.anchored_mean:
                decoder.set_baseline_profile((rate_sum / n_ctrl).to(device))
            if args.state_dispersion and args.anchor_weight > 0:
                mean = cnt_sum / n_ctrl
                var = (cnt_sumsq / n_ctrl - mean ** 2).clamp_min(0.0)
                kappa_target = CountDecoder.moment_dispersion(mean, var).to(device)
            logger.info("control pre-pass: n_ctrl=%d  baseline=%s  anchor=%s",
                        n_ctrl, args.anchored_mean, "yes" if kappa_target is not None else "no")

    avg = float("nan")
    for epoch in range(args.epochs):
        decoder.train()
        running = 0.0
        for batch in train_loader:
            with torch.no_grad():
                z = jepa.embed(batch["tokens"].to(device))
            counts = batch["counts"].float().to(device)
            libsize = batch["libsize"].to(device)
            loss = decoder.loss(z, counts, libsize, kappa_target=kappa_target,
                                 anchor_weight=args.anchor_weight if args.state_dispersion else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / max(len(train_loader), 1)
        logger.info("epoch %d/%d  nb_nll=%.3f", epoch + 1, args.epochs, avg)

    out = exp.checkpoints / "count_decoder.pt"
    torch.save({"state": decoder.state_dict(), "latent_dim": jepa.cfg.embed_dim,
                "n_genes": n_genes, "zinb": args.zinb,
                "anchored_mean": args.anchored_mean, "state_dispersion": args.state_dispersion}, out)
    exp.write_report("stage_c_decoder", {"final_nb_nll": avg, "n_genes": n_genes, "zinb": args.zinb,
                                         "anchored_mean": args.anchored_mean,
                                         "state_dispersion": args.state_dispersion,
                                         "train_cells": len(train_loader.dataset)})
    logger.info("saved count decoder -> %s", out)


if __name__ == "__main__":
    main()
