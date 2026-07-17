"""Stage B (operator-algebra variant) — one generator per gene, combinations compose in the group.

Round 3's operator (`13_train_operator.py`) applied ``exp`` once to a single generator and composed
genes in the additive embedding, so it tied the flow. This variant gives each single gene its own
generator ``M_g`` and composes a combination *in the group*, ``A_{A+B} = exp(M_A/2) exp(M_B) exp(M_A/2)``,
so the departure from additive composition is the Lie bracket ``[M_A, M_B]`` and the bracket is
epistasis. See `src/ssllab/generative/operator_algebra.py` and the design spec
`dev/planning/action_operator/03-the-operator-algebra-composition-and-epistasis.md`.

Training is identical in spirit to round 3: cells are unpaired, so each step pushes a control cloud
through a perturbation's operator and matches the marginal with an energy distance. The one thing that
changes downstream is that an observed *combination* now flows through the composed operator, so its
gradient reaches BOTH single-gene generators. That is how a single's generator is pinned not only by its
own cells but by every combination it participates in, which is what shapes the bracket from real pair
data. Held-out combinations are never trained; they are composed at eval time from the trained singles.

Usage
-----
    python examples/perturbation_response/16_train_operator_algebra.py \
        --experiment norman_operator_algebra --encoder output/norman_stage_a/checkpoints/encoder.pt \
        --split combo --epochs 60

Output
------
    output/<experiment>/checkpoints/operator_algebra.pt
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
from ssllab.generative.operator_algebra import NamedGeneratorOperator, energy_distance
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the operator-algebra Stage B (per-gene generators).")
    p.add_argument("--experiment", type=str, default="norman_operator_algebra")
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
    p.add_argument("--action-weight", type=float, default=1e-4,
                   help="Frobenius (least-action) penalty on the generators: the near-identity prior")
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

    # 1. Frozen latents, standardized. Identical convention to round 3 so the two are comparable.
    Z, pert_id, is_control = precompute_latents(jepa, train_loader, device)
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mean) / std
    ctrl_pool = Zn[is_control.bool()].to(device)
    if len(ctrl_pool) == 0:
        raise RuntimeError("no control cells in the precomputed set; increase --limit")

    # 2. Index training latents by perturbation. A step's unit is a perturbation with enough cells.
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
    n_combo = sum(1 for p in trainable if "+" in str(pert_names[p]))
    logger.info("precomputed %d latents (dim %d); %d controls; %d trainable perturbations (%d combos)",
                len(Zn), dim, len(ctrl_pool), len(trainable), n_combo)

    # 3. The operator: one dense generator per target gene, zero-initialized (A = I at the start).
    pert_gene, gene_vocab = build_pert_gene_matrix(pert_names)
    model = NamedGeneratorOperator(pert_gene=pert_gene, dim=dim).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    logger.info("operator-algebra: %d per-gene generators (%.1fM params); A_g = exp(M_g) starts at I; "
                "combinations compose in the group so the bracket [M_A,M_B] carries epistasis.",
                len(gene_vocab), n_params / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    def mean_bracket_over_combos() -> float:
        """Diagnostic: average ||[M_A,M_B]||_F over trainable combos — did the algebra engage?"""
        vals = [float(model.bracket_norm(torch.tensor([p]))) for p in trainable if "+" in str(pert_names[p])]
        return float(np.mean(vals)) if vals else 0.0

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
            running += float(loss.detach())
            n_seen += 1
        avg = running / max(n_seen, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("epoch %d/%d  energy=%.4f  mean||[M_A,M_B]||=%.3f",
                        epoch + 1, args.epochs, avg, mean_bracket_over_combos())
        else:
            logger.info("epoch %d/%d  energy=%.4f", epoch + 1, args.epochs, avg)

    out = exp.checkpoints / "operator_algebra.pt"
    torch.save({
        "model": model.state_dict(),
        "dim": dim, "n_perts": n_perts, "pert_gene": pert_gene, "gene_vocab": gene_vocab,
        "mean": mean, "std": std, "ctrl_pool": ctrl_pool.cpu(),
    }, out)
    exp.write_report("stage_b_operator_algebra", {
        "final_energy": avg, "mean_bracket": mean_bracket_over_combos(),
        "n_perts": n_perts, "n_trainable_perts": len(trainable), "n_trainable_combos": n_combo,
        "n_generators": len(gene_vocab), "action_weight": args.action_weight,
        "train_cells": len(Zn), "n_controls": int(is_control.sum()), "seed": args.seed,
    })
    logger.info("saved operator-algebra -> %s", out)


if __name__ == "__main__":
    main()
