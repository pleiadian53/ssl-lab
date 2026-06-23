"""Stage B — train the conditional flow prior over cell latents.

The G1 piece: model the *distribution* of a cell's outcome latent given a
condition ``c = (z_b, z_p)`` — baseline control cell + perturbation. Reuses the
modality-agnostic conditional velocity field ``v_eta(z, t, c)`` from
``ssllab.generative.flow`` (the same object built and de-risked on MNIST), now on
frozen cell latents, with the perturbation-specific ``ConditionEncoder`` building
``c``. Sampling from this flow + decoding (Stage C) generates perturbed cells.

Pipeline: freeze encoder -> precompute latents once -> standardize -> train the
conditional flow (+ condition encoder) by flow-matching, with the baseline z_b
drawn from the control-cell population each step.

Usage
-----
    python examples/perturbation_response/04_train_cond_flow.py --epochs 60
    python examples/perturbation_response/04_train_cond_flow.py --limit 3000 --epochs 5   # smoke

Output
------
    output/<experiment>/checkpoints/cond_flow.pt   velocity field + condition encoder + stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.experiment import experiment
from ssllab.generative.condition import ConditionEncoder
from ssllab.generative.flow import VelocityMLP, cfm_loss
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a conditional flow prior over cell latents.")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="cells", choices=["combo", "cells"],
                   help="cells = in-distribution effect-size test (default); see 06")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--pert-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--p-drop", type=float, default=0.1, help="condition-dropout for classifier-free guidance")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def precompute_latents(jepa, loader, device):
    """Encode every cell once (frozen encoder) -> aligned (Z, pert_id, is_control)."""
    Z, P, C = [], [], []
    for b in loader:
        Z.append(jepa.embed(b["tokens"].to(device)).cpu())
        P.append(b["pert_id"])
        C.append(b["is_control"])
    return torch.cat(Z), torch.cat(P), torch.cat(C)


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
    dim = jepa.cfg.embed_dim
    n_perts = len(train_loader.meta["pert_names"])

    # 1. Precompute frozen latents; standardize for the flow.
    Z, pert_id, is_control = precompute_latents(jepa, train_loader, device)
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mean) / std
    ctrl_pool = Zn[is_control.bool()]
    if len(ctrl_pool) == 0:
        raise RuntimeError("no control cells in the precomputed set — increase --limit")
    logger.info("precomputed %d latents (dim %d); %d controls; %d perturbations",
                len(Zn), dim, len(ctrl_pool), n_perts)

    # 2. Conditional velocity field + condition encoder.
    flow = VelocityMLP(data_dim=dim, hidden=args.hidden, n_layers=args.n_layers, cond_dim=args.cond_dim).to(device)
    cond = ConditionEncoder(latent_dim=dim, n_perts=n_perts, pert_dim=args.pert_dim, cond_dim=args.cond_dim).to(device)
    opt = torch.optim.AdamW(list(flow.parameters()) + list(cond.parameters()), lr=args.lr)

    loader = DataLoader(TensorDataset(Zn, pert_id), batch_size=args.batch_size, shuffle=True, drop_last=True)
    ctrl_pool = ctrl_pool.to(device)
    g = torch.Generator().manual_seed(args.seed)

    avg = float("nan")
    for epoch in range(args.epochs):
        flow.train(); cond.train()
        running = 0.0
        for z1, pid in loader:
            z1, pid = z1.to(device), pid.to(device)
            # baseline z_b: a control-cell latent sampled per example (population baseline)
            zb = ctrl_pool[torch.randint(len(ctrl_pool), (z1.shape[0],), generator=g)]
            c = cond(zb, pid)
            loss = cfm_loss(flow, z1, c=c, p_drop=args.p_drop)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / max(len(loader), 1)
        logger.info("epoch %d/%d  cfm_loss=%.4f", epoch + 1, args.epochs, avg)

    out = exp.checkpoints / "cond_flow.pt"
    torch.save({
        "flow": flow.state_dict(), "cond": cond.state_dict(),
        "dim": dim, "cond_dim": args.cond_dim, "pert_dim": args.pert_dim,
        "hidden": args.hidden, "n_layers": args.n_layers, "n_perts": n_perts,
        "mean": mean, "std": std,
        "ctrl_pool": ctrl_pool.cpu(),  # standardized control latents, for sampling z_b
    }, out)
    exp.write_report("stage_b_flow", {"final_cfm_loss": avg, "n_perts": n_perts,
                                      "train_cells": len(Zn), "n_controls": int(is_control.sum())})
    logger.info("saved conditional flow -> %s", out)


if __name__ == "__main__":
    main()
