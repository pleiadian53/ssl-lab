"""Sample the conditional flow prior *given a class*, and measure whether it worked.

The payoff of the Part 9 step on the MNIST proxy. For each digit class we fix the
condition, integrate the flow's ODE from noise to a latent, and decode to an
image. Then — crucially — we grade it with an *independent* frozen oracle: does
the oracle agree the generated digit is the class we asked for?

That number, **conditional fidelity**, is the whole point. It separates "the
conditioning actually steers generation" from "the samples merely look like
plausible digits." Chance is 1/n_classes (0.1 for MNIST); anything well above
that means ``v(z, t, c)`` is conditioning correctly.

Usage
-----
    python examples/generative_jepa/08_sample_conditional.py
    python examples/generative_jepa/08_sample_conditional.py --per-class 16 --guidance 2.0

Output
------
    output/<experiment>/samples/samples_conditional.png   per-class grid (one row per class)
    output/<experiment>/reports/conditional_eval.json     fidelity (overall + per class)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.eval.oracle_mnist import load_or_train_oracle
from ssllab.eval.viz import save_image_grid
from ssllab.experiment import experiment
from ssllab.generative.flow import VelocityMLP, euler_sample
from ssllab.models.decoder import LatentDecoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample p(z | class) and measure conditional fidelity.")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--prior", type=str, default=None, help="override conditional prior path")
    p.add_argument("--decoder", type=str, default=None, help="override decoder path")
    p.add_argument("--oracle-path", type=str, default="output/oracles/mnist_cnn.pt")
    p.add_argument("--oracle-epochs", type=int, default=2)
    p.add_argument("--per-class", type=int, default=16, help="samples generated per class")
    p.add_argument("--steps", type=int, default=100, help="ODE integration steps")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight (1 = off)")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    prior_path = args.prior or (exp.checkpoints / "prior_cond.pt")
    decoder_path = args.decoder or (exp.checkpoints / "decoder.pt")

    # Load the conditional flow prior (+ class embedding + standardization stats).
    pck = torch.load(prior_path, map_location=device)
    cond_dim, n_classes = pck["cond_dim"], pck["n_classes"]
    prior = VelocityMLP(
        data_dim=pck["dim"], hidden=pck["hidden"], n_layers=pck["n_layers"], cond_dim=cond_dim
    ).to(device)
    prior.load_state_dict(pck["state"])
    class_embed = nn.Embedding(n_classes, cond_dim).to(device)
    class_embed.load_state_dict(pck["class_embed"])
    class_embed.eval()
    mean, std = pck["mean"].to(device), pck["std"].to(device)

    # Load the decoder and the independent oracle.
    dck = torch.load(decoder_path, map_location=device)
    decoder = LatentDecoder(latent_dim=dck["latent_dim"]).to(device)
    decoder.load_state_dict(dck["state"])
    decoder.eval()
    oracle = load_or_train_oracle(args.oracle_path, device=device, epochs=args.oracle_epochs, data_dir=args.data_dir)

    # For each class: fix the condition, sample latents, decode, ask the oracle.
    per = args.per_class
    grid_rows: list[torch.Tensor] = []
    per_class_fidelity: dict[str, float] = {}
    correct_total = 0
    with torch.no_grad():
        for k in range(n_classes):
            labels = torch.full((per,), k, dtype=torch.long, device=device)
            c = class_embed(labels)
            z_norm = euler_sample(
                prior, per, pck["dim"], n_steps=args.steps, device=device, c=c, guidance=args.guidance
            )
            z = z_norm * std + mean
            images = decoder.decode_images(z)  # (per, 1, 28, 28) in [0, 1]
            grid_rows.append(images)
            preds = oracle(images).argmax(1)
            hits = int((preds == k).sum().item())
            correct_total += hits
            per_class_fidelity[str(k)] = hits / per
            logger.info("class %d: conditional fidelity %d/%d = %.3f", k, hits, per, hits / per)

    overall = correct_total / (n_classes * per)
    chance = 1.0 / n_classes
    logger.info("overall conditional fidelity = %.3f  (chance = %.3f, guidance = %.1f)", overall, chance, args.guidance)

    grid = torch.cat(grid_rows, dim=0)  # class-major: row k = class k
    grid_path = exp.samples / "samples_conditional.png"
    save_image_grid(grid, grid_path, nrow=per)
    logger.info("saved per-class sample grid -> %s", grid_path)

    report = exp.write_report(
        "conditional_eval",
        {
            "overall_fidelity": overall,
            "chance": chance,
            "per_class_fidelity": per_class_fidelity,
            "per_class": per,
            "n_classes": n_classes,
            "guidance": args.guidance,
            "ode_steps": args.steps,
        },
    )
    logger.info("wrote report -> %s", report)


if __name__ == "__main__":
    main()
