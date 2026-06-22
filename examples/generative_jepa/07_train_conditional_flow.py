"""Train a *conditional* flow-matching prior over the frozen JEPA latent.

The Part 9 step, on the MNIST proxy: take example 04's unconditional flow prior
and give its velocity field one more input — a condition ``c``. Here ``c`` is a
learned embedding of the digit class, so integrating the ODE from noise samples
``p(z | class)`` instead of the marginal ``p(z)``. The same machinery carries the
``(z_b, z_p)`` perturbation condition in the bio phase; only the condition encoder
changes (a class ``nn.Embedding`` here, a baseline+intervention map there).

This is a *method de-risking* run — it proves the ``v(z, t, c)`` code path works
before any bio data pipeline exists. Quality is graded by conditional fidelity
(example 08), not photorealism.

Usage
-----
    python examples/generative_jepa/07_train_conditional_flow.py --epochs 30
    python examples/generative_jepa/07_train_conditional_flow.py --epochs 2 --limit 4000   # smoke

Output
------
    output/<experiment>/checkpoints/prior_cond.pt   conditional flow prior + class embedding
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
    p = argparse.ArgumentParser(description="Train a class-conditional flow prior over the JEPA latent.")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--encoder", type=str, default=None, help="override encoder path (default: from experiment)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--cond-dim", type=int, default=64, help="width of the condition embedding")
    p.add_argument("--n-classes", type=int, default=10, help="number of conditioning classes (MNIST: 10)")
    p.add_argument("--p-drop", type=float, default=0.1, help="condition-dropout prob for classifier-free guidance")
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

    # 1. Extract the data-latent distribution *with labels* (encoder is frozen).
    src_loader, _ = get_mnist_dataloaders(args.batch_size, args.data_dir)
    Z, Y = extract_latents(jepa.embed, src_loader, device, limit=args.limit)
    logger.info("extracted %d latents of dim %d (with class labels)", Z.shape[0], dim)

    # Standardize latents -> easier flow target; store stats for sampling.
    mean, std = Z.mean(0), Z.std(0) + 1e-6
    Zn = (Z - mean) / std
    latent_loader = DataLoader(
        TensorDataset(Zn, Y), batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 2. The condition encoder: class label -> cond vector. (Swapped for a
    #    (z_b, z_p) map in the bio phase; the velocity field is unchanged.)
    class_embed = nn.Embedding(args.n_classes, args.cond_dim).to(device)

    # 3. The conditional rectified-flow velocity field.
    prior = VelocityMLP(
        data_dim=dim, hidden=args.hidden, n_layers=args.n_layers, cond_dim=args.cond_dim
    ).to(device)

    params = list(prior.parameters()) + list(class_embed.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    for epoch in range(args.epochs):
        prior.train()
        class_embed.train()
        running = 0.0
        for z1, y in latent_loader:
            z1, y = z1.to(device), y.to(device)
            c = class_embed(y)
            loss = cfm_loss(prior, z1, c=c, p_drop=args.p_drop)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        logger.info("epoch %d/%d  cfm_loss=%.4f", epoch + 1, args.epochs, running / max(len(latent_loader), 1))

    out = exp.checkpoints / "prior_cond.pt"
    torch.save(
        {
            "state": prior.state_dict(),
            "class_embed": class_embed.state_dict(),
            "dim": dim,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "cond_dim": args.cond_dim,
            "n_classes": args.n_classes,
            "mean": mean,
            "std": std,
        },
        out,
    )
    logger.info("saved conditional flow prior -> %s", out)


if __name__ == "__main__":
    main()
