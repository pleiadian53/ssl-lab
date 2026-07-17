"""Stage B (operator variant) — learn the action operator that carries control to perturbed.

The flow of `04_train_cond_flow.py` learns a free velocity field and samples where a perturbed
cell *lands*. This script learns the **transition** instead: the operator ``A_p = exp(M(e(p)))``
that transports a control cell to its perturbed counterpart, so the perturbation's effect is
the operator's departure from the identity. See chapter 7 of the conditional-flow-jepa series.

Because single-cell sequencing destroys the cell, there is no "same cell before and after" and
the per-pair equivariance loss cannot be computed. So each step pushes a cloud of control
latents through the perturbation's operator and matches the resulting cloud's **marginal**
against the real perturbed cloud with an energy distance, which needs no pairing.

The batch is a **perturbation**, not a cell. Each step samples a few perturbations; for each it
draws a control cloud and that perturbation's real cloud, and averages the energy distance.

Usage
-----
    # phase 1: the pure operator, deterministic. The effect-size play.
    python examples/perturbation_response/13_train_operator.py --experiment norman_operator \
        --split combo --epochs 60

    # phase 2: let the perturbation induce a DISTRIBUTION over operators, so the pushforward
    # can carry more spread than the control cloud it started from (the calibration play).
    python examples/perturbation_response/13_train_operator.py --experiment norman_operator_stoch \
        --split combo --epochs 60 --stochastic --residual-scale 1.0

Output
------
    output/<experiment>/checkpoints/operator.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.experiment import experiment
from ssllab.generative.condition import build_pert_gene_matrix
from ssllab.generative.operator_perturb import PerturbationOperator, energy_distance
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the action operator over frozen cell latents.")
    p.add_argument("--experiment", type=str, default="norman_operator")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--encoder", type=str, default=None, help="defaults to <experiment>/checkpoints/encoder.pt")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--perts-per-step", type=int, default=8, help="perturbations sampled per optimizer step")
    p.add_argument("--cloud-size", type=int, default=128, help="cells per cloud (control and real)")
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-generators", type=int, default=16, help="size of the learned basis {B_i}")
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--gene-dim", type=int, default=64)
    p.add_argument("--compose", type=str, default="additive", choices=["additive", "deepsets"])
    p.add_argument("--action-weight", type=float, default=1e-4,
                   help="Frobenius (least-action) penalty on M: the near-identity prior")
    p.add_argument("--stochastic", action="store_true",
                   help="alpha ~ Gaussian, so the perturbation induces a MIXTURE of operators. "
                        "A deterministic operator is invertible and cannot widen the cloud much; "
                        "this is what lets the pushforward carry extra spread (calibration).")
    p.add_argument("--residual-scale", type=float, default=0.0,
                   help="add a learned per-condition residual displacement (drift + residual)")
    p.add_argument("--min-cells", type=int, default=20, help="skip perturbations with fewer train cells")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def precompute_latents(jepa, loader, device):
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
        batch_size=256, data_dir=args.data_dir, artifact=args.artifact,
        split=args.split, limit=args.limit, seed=0,
    )
    dim = jepa.cfg.embed_dim
    pert_names = train_loader.meta["pert_names"]
    n_perts = len(pert_names)

    # 1. Frozen latents, standardized. Identical convention to Stage B so the two are comparable.
    Z, pert_id, is_control = precompute_latents(jepa, train_loader, device)
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mean) / std
    ctrl_pool = Zn[is_control.bool()].to(device)
    if len(ctrl_pool) == 0:
        raise RuntimeError("no control cells in the precomputed set; increase --limit")

    # 2. Index the training latents by perturbation. The unit of a training step is a
    #    perturbation with enough cells to estimate a cloud, so weak ones are skipped.
    by_pert: dict[int, torch.Tensor] = {}
    pid_np = pert_id.numpy()
    for p in np.unique(pid_np):
        if pert_names[p] == "control":
            continue
        idx = np.where((pid_np == p) & (~is_control.numpy().astype(bool)))[0]
        if len(idx) >= args.min_cells:
            by_pert[int(p)] = Zn[idx].to(device)
    trainable = sorted(by_pert)
    if not trainable:
        raise RuntimeError("no perturbation has enough cells to form a cloud")
    logger.info("precomputed %d latents (dim %d); %d controls; %d trainable perturbations",
                len(Zn), dim, len(ctrl_pool), len(trainable))

    # 3. The operator. Gene-set condition, so a held-out combination composes from its parts.
    pert_gene, gene_vocab = build_pert_gene_matrix(pert_names)
    model = PerturbationOperator(
        pert_gene=pert_gene, dim=dim, num_generators=args.num_generators,
        gene_dim=args.gene_dim, cond_dim=args.cond_dim, compose=args.compose,
        stochastic=args.stochastic, residual_scale=args.residual_scale,
    ).to(device)
    logger.info("operator: %d generators, %d target genes, stochastic=%s, residual=%.2f",
                args.num_generators, len(gene_vocab), args.stochastic, args.residual_scale)
    logger.info("A = exp(M) starts at the IDENTITY (policy is zero-initialized): the operator "
                "must earn every departure from 'this perturbation does nothing'.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    avg = float("nan")
    for epoch in range(args.epochs):
        model.train()
        running, n_seen = 0.0, 0
        for _ in range(args.steps_per_epoch):
            perts = rng.choice(trainable, size=min(args.perts_per_step, len(trainable)), replace=False)
            loss = torch.zeros((), device=device)
            for p in perts:
                real = by_pert[int(p)]
                m = min(args.cloud_size, len(real))
                real_cloud = real[torch.randint(len(real), (m,), device=device)]
                ctrl_cloud = ctrl_pool[torch.randint(len(ctrl_pool), (m,), device=device)]
                pid = torch.tensor([int(p)], device=device)
                pred_cloud = model.pushforward(ctrl_cloud, pid)
                loss = loss + energy_distance(pred_cloud, real_cloud)
                if args.action_weight > 0:
                    loss = loss + args.action_weight * model.action_energy(pid)
            loss = loss / len(perts)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss)
            n_seen += 1
        avg = running / max(n_seen, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                pid = torch.tensor([trainable[0]], device=device)
                A = model.operator_matrix(pid)
                dev = float((A - torch.eye(dim, device=device)).norm())
            logger.info("epoch %d/%d  energy=%.4f  ||A - I||_F=%.3f", epoch + 1, args.epochs, avg, dev)
        else:
            logger.info("epoch %d/%d  energy=%.4f", epoch + 1, args.epochs, avg)

    out = exp.checkpoints / "operator.pt"
    torch.save({
        "model": model.state_dict(),
        "dim": dim, "num_generators": args.num_generators, "cond_dim": args.cond_dim,
        "gene_dim": args.gene_dim, "compose": args.compose, "n_perts": n_perts,
        "stochastic": args.stochastic, "residual_scale": args.residual_scale,
        "pert_gene": pert_gene,
        "mean": mean, "std": std, "ctrl_pool": ctrl_pool.cpu(),
    }, out)
    exp.write_report("stage_b_operator", {
        "final_energy": avg, "n_perts": n_perts, "n_trainable_perts": len(trainable),
        "num_generators": args.num_generators, "stochastic": args.stochastic,
        "residual_scale": args.residual_scale, "action_weight": args.action_weight,
        "train_cells": len(Zn), "n_controls": int(is_control.sum()),
    })
    logger.info("saved operator -> %s", out)


if __name__ == "__main__":
    main()
