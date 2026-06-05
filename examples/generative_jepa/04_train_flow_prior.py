"""Train a flow-matching prior over the frozen JEPA latent.

Milestone: the "sample latents" half of making JEPA sampleable. We extract the
pooled latents of the training set (frozen encoder) and fit a rectified-flow
velocity field ``v(z_t, t)`` so that integrating the ODE from Gaussian noise
lands on the data-latent distribution.

Usage
-----
    python examples/generative_jepa/04_train_flow_prior.py --epochs 20
    python examples/generative_jepa/04_train_flow_prior.py --epochs 2 --limit 4000   # quick smoke

Output
------
    output/<experiment>/checkpoints/prior.pt   flow-prior weights (consumed by example 05)
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
from ssllab.data.mnist import get_mnist_dataloaders
from ssllab.experiment import experiment
from ssllab.extract import extract_latents
from ssllab.generative.flow import VelocityMLP, cfm_loss
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a flow-matching prior over the JEPA latent.")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--encoder", type=str, default=None, help="override encoder path (default: from experiment)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="cap train latents (smoke runs)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    encoder_path = args.encoder or (exp.checkpoints / "encoder.pt")
    jepa = load_jepa(encoder_path, device)
    dim = jepa.cfg.embed_dim

    # 1. Extract the data-latent distribution once (encoder is frozen).
    src_loader, _ = get_mnist_dataloaders(args.batch_size, args.data_dir)
    Z, _ = extract_latents(jepa.embed, src_loader, device, limit=args.limit)
    logger.info("extracted %d latents of dim %d", Z.shape[0], dim)

    # Standardize latents -> easier flow target; store stats for sampling.
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mean) / std
    latent_loader = DataLoader(TensorDataset(Zn), batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Fit the rectified-flow velocity field.
    prior = VelocityMLP(data_dim=dim, hidden=args.hidden, n_layers=args.n_layers).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        prior.train()
        running = 0.0
        for (z1,) in latent_loader:
            loss = cfm_loss(prior, z1.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        logger.info("epoch %d/%d  cfm_loss=%.4f", epoch + 1, args.epochs, running / max(len(latent_loader), 1))

    out = exp.checkpoints / "prior.pt"
    torch.save(
        {"state": prior.state_dict(), "dim": dim, "hidden": args.hidden,
         "n_layers": args.n_layers, "mean": mean, "std": std},
        out,
    )
    logger.info("saved flow prior -> %s", out)


if __name__ == "__main__":
    main()
