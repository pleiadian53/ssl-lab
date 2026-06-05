"""Train a JEPA encoder on MNIST patches (self-supervised, no labels).

Milestone: the JEPA representation learner. Predicts target-patch *embeddings*
from context-patch embeddings in latent space; an EMA target encoder provides
the prediction targets. Prints collapse diagnostics each epoch (effective rank
should stay well above 1) and saves the encoder for the downstream generative
slice.

Usage
-----
    python examples/jepa_basics/01_train_jepa_mnist.py
    python examples/jepa_basics/01_train_jepa_mnist.py --epochs 5 --reg-coef 0.04
    python examples/jepa_basics/01_train_jepa_mnist.py --epochs 1 --limit 2000   # quick smoke

Output
------
    output/<experiment>/checkpoints/encoder.pt   JEPA config + weights (consumed by 02-05)
    output/<experiment>/reports/jepa_train.json  final loss + collapse diagnostics
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
from ssllab.data.mnist import N_TOKENS, TOKEN_DIM, get_mnist_dataloaders, patchify
from ssllab.eval.collapse import collapse_report
from ssllab.experiment import experiment
from ssllab.jepa.model import build_jepa
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a JEPA encoder on MNIST patches.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--enc-depth", type=int, default=4)
    p.add_argument("--pred-depth", type=int, default=2)
    p.add_argument("--n-target", type=int, default=4, help="# target patches to predict")
    p.add_argument("--reg-coef", type=float, default=0.0, help="VICReg weight (0 disables)")
    p.add_argument("--limit", type=int, default=None, help="cap train examples (smoke runs)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()
    logger.info("device=%s epochs=%d reg_coef=%.3g n_target=%d exp=%s",
                device, args.epochs, args.reg_coef, args.n_target, exp.root)

    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir, limit=args.limit)

    jepa = build_jepa(
        token_dim=TOKEN_DIM,
        n_tokens=N_TOKENS,
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        n_target=args.n_target,
        reg_coef=args.reg_coef,
    ).to(device)
    jepa.ema.to(device)
    opt = torch.optim.AdamW(jepa.parameters(), lr=args.lr)

    total_steps = args.epochs * len(train_loader)
    step = 0
    for epoch in range(args.epochs):
        jepa.train()
        running = 0.0
        for images, _ in train_loader:
            tokens = patchify(images.to(device))
            loss, comp = jepa(tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            jepa.update_target(step, total_steps)
            running += comp["loss"]
            step += 1
        avg = running / max(len(train_loader), 1)

        # Collapse diagnostics on a held-out batch.
        jepa.eval()
        with torch.no_grad():
            imgs, _ = next(iter(test_loader))
            z = jepa.embed(patchify(imgs.to(device)))
        rep = collapse_report(z)
        logger.info(
            "epoch %d/%d  loss=%.4f  eff_rank=%.2f/%d  feat_std=%.3f",
            epoch + 1, args.epochs, avg, rep["effective_rank"], rep["embed_dim"], rep["feature_std"],
        )

    path = save_jepa(jepa, exp.checkpoints / "encoder.pt")
    report = exp.write_report("jepa_train", {
        "experiment": args.experiment,
        "epochs": args.epochs,
        "final_loss": avg,
        "reg_coef": args.reg_coef,
        "n_target": args.n_target,
        **rep,
    })
    logger.info("saved encoder -> %s ; report -> %s", path, report)


if __name__ == "__main__":
    main()
