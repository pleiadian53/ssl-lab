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
import torch.nn as nn

from ssllab.generative.condition import (
    ConditionEncoder,
    GeneSetConditionEncoder,
    GeneSetEmbedding,
    build_pert_gene_matrix,
)
from ssllab.generative.flow import VelocityMLP, cfm_loss, ot_couple
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
    p.add_argument("--flow-base", type=str, default="gaussian", choices=["gaussian", "control"],
                   help="gaussian = noise->outcome prior, condition fuses (z_b, z_p) [current default]; "
                        "control = transport a control latent -> outcome (source z0=z_b, condition = z_p only) "
                        "so the field models the displacement (the effect) directly")
    p.add_argument("--coupling", type=str, default="independent", choices=["independent", "ot"],
                   help="control base only: independent = random control per target; "
                        "ot = minibatch optimal-transport pairing (straighter paths, lower-variance target)")
    p.add_argument("--cond-type", type=str, default="table", choices=["table", "geneset"],
                   help="table = learned per-pert embedding (in-distribution); "
                        "geneset = gene-compositional (generalizes to held-out combos)")
    p.add_argument("--compose", type=str, default="additive", choices=["additive", "deepsets"],
                   help="geneset composition: additive sum e(A+B)=e(A)+e(B), or a DeepSets refinement")
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--pert-dim", type=int, default=64)
    p.add_argument("--gene-dim", type=int, default=64, help="per-target-gene embedding width (geneset)")
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
    #    - gaussian base: condition fuses (z_b, z_p) -> the field maps noise -> outcome.
    #    - control base:  source is a control latent and the condition is z_p ALONE -> the field
    #      transports baseline -> outcome, so z_b anchors the sample and z_p need only encode the shift.
    flow = VelocityMLP(data_dim=dim, hidden=args.hidden, n_layers=args.n_layers, cond_dim=args.cond_dim).to(device)
    pert_gene = None
    if args.cond_type == "geneset":
        pert_gene, gene_vocab = build_pert_gene_matrix(train_loader.meta["pert_names"])
        logger.info("geneset condition: %d target genes, compose=%s", len(gene_vocab), args.compose)

    if args.flow_base == "control":
        if args.cond_type == "geneset":
            cond = GeneSetEmbedding(pert_gene, gene_dim=args.gene_dim, out_dim=args.cond_dim, compose=args.compose).to(device)
        else:
            cond = nn.Embedding(n_perts, args.cond_dim).to(device)          # pert_id -> z_p
    else:
        if args.cond_type == "geneset":
            cond = GeneSetConditionEncoder(latent_dim=dim, pert_gene=pert_gene, pert_dim=args.pert_dim,
                                           gene_dim=args.gene_dim, cond_dim=args.cond_dim, compose=args.compose).to(device)
        else:
            cond = ConditionEncoder(latent_dim=dim, n_perts=n_perts, pert_dim=args.pert_dim, cond_dim=args.cond_dim).to(device)
    logger.info("flow-base=%s  cond-type=%s", args.flow_base, args.cond_type)
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
            if args.flow_base == "control":
                if args.coupling == "ot":
                    zb = ot_couple(zb, z1)                                          # OT-pair source to target
                loss = cfm_loss(flow, z1, c=cond(pid), p_drop=args.p_drop, z0=zb)   # transport z_b -> z1
            else:
                loss = cfm_loss(flow, z1, c=cond(zb, pid), p_drop=args.p_drop)      # noise -> z1, c=(z_b,z_p)
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
        "cond_type": args.cond_type, "compose": args.compose, "gene_dim": args.gene_dim,
        "flow_base": args.flow_base,
        "pert_gene": pert_gene,  # multi-hot (geneset) or None (table); lets the loader rebuild cond
        "mean": mean, "std": std,
        "ctrl_pool": ctrl_pool.cpu(),  # standardized control latents, for sampling z_b (source or baseline)
    }, out)
    exp.write_report("stage_b_flow", {"final_cfm_loss": avg, "n_perts": n_perts, "cond_type": args.cond_type,
                                      "flow_base": args.flow_base,
                                      "coupling": args.coupling if args.flow_base == "control" else None,
                                      "compose": args.compose if args.cond_type == "geneset" else None,
                                      "train_cells": len(Zn), "n_controls": int(is_control.sum())})
    logger.info("saved conditional flow -> %s", out)


if __name__ == "__main__":
    main()
