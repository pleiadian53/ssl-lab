"""Sample from the flow prior and decode to images — the generative payoff.

Milestone: closes the loop. Draw ``z ~ p(z)`` by integrating the flow prior's
ODE from noise, undo the standardization, and decode ``z -> image``. This is
JEPA turned into a sampleable generative model.

Usage
-----
    python examples/generative_jepa/05_sample_and_decode.py
    python examples/generative_jepa/05_sample_and_decode.py --n 64 --steps 100

Output
------
    output/<experiment>/samples/samples.png   grid of generated digits
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

from ssllab.eval.viz import save_image_grid
from ssllab.experiment import experiment
from ssllab.generative.flow import VelocityMLP, euler_sample
from ssllab.models.decoder import LatentDecoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample z ~ flow prior and decode to images.")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--prior", type=str, default=None, help="override prior path (default: from experiment)")
    p.add_argument("--decoder", type=str, default=None, help="override decoder path (default: from experiment)")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=100, help="ODE integration steps")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    prior_path = args.prior or (exp.checkpoints / "prior.pt")
    decoder_path = args.decoder or (exp.checkpoints / "decoder.pt")

    # Load the flow prior (+ its latent standardization stats).
    pck = torch.load(prior_path, map_location=device)
    prior = VelocityMLP(data_dim=pck["dim"], hidden=pck["hidden"], n_layers=pck["n_layers"]).to(device)
    prior.load_state_dict(pck["state"])
    mean, std = pck["mean"].to(device), pck["std"].to(device)

    # Load the decoder.
    dck = torch.load(decoder_path, map_location=device)
    decoder = LatentDecoder(latent_dim=dck["latent_dim"]).to(device)
    decoder.load_state_dict(dck["state"])
    decoder.eval()

    # Sample z ~ prior, undo standardization, decode.
    with torch.no_grad():
        z_norm = euler_sample(prior, args.n, pck["dim"], n_steps=args.steps, device=device)
        z = z_norm * std + mean
        images = decoder.decode_images(z)

    out = exp.samples / "samples.png"
    save_image_grid(images, out, nrow=8)
    logger.info("sampled %d digits via %d ODE steps -> %s", args.n, args.steps, out)


if __name__ == "__main__":
    main()
