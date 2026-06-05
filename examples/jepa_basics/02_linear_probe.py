"""Linear-probe the frozen JEPA latent on MNIST digit classification.

Milestone: confirm the self-supervised representation is semantically useful.
We freeze the encoder, extract pooled latents, and fit a logistic-regression
probe. Well-above-chance test accuracy (chance = 10%) means JEPA learned
structure without ever seeing a label.

Usage
-----
    python examples/jepa_basics/02_linear_probe.py
    python examples/jepa_basics/02_linear_probe.py --train-limit 5000 --test-limit 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.mnist import get_mnist_dataloaders
from ssllab.eval.probe import linear_probe
from ssllab.experiment import experiment
from ssllab.extract import extract_latents
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe of the frozen JEPA latent.")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--encoder", type=str, default=None, help="override encoder path (default: from experiment)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-limit", type=int, default=10000)
    p.add_argument("--test-limit", type=int, default=5000)
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
    logger.info("loaded frozen encoder from %s", encoder_path)

    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)
    Ztr, Ytr = extract_latents(jepa.embed, train_loader, device, limit=args.train_limit)
    Zte, Yte = extract_latents(jepa.embed, test_loader, device, limit=args.test_limit)
    logger.info("latents: train=%s test=%s", tuple(Ztr.shape), tuple(Zte.shape))

    scores = linear_probe(Ztr, Ytr, Zte, Yte)
    logger.info("linear probe  train_acc=%.4f  test_acc=%.4f  (chance=0.10)", scores["train_acc"], scores["test_acc"])
    report = exp.write_report("probe", {
        "experiment": args.experiment,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        **scores,
    })
    logger.info("report -> %s", report)


if __name__ == "__main__":
    main()
