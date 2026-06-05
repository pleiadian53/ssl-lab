"""Train a decoder that maps the frozen JEPA latent back to pixels.

Milestone: the "map latents -> data" half of making JEPA sampleable. The
encoder stays frozen; we learn ``g: z -> image`` with a Bernoulli (BCE)
reconstruction loss. Because JEPA latents are not trained to be decodable,
reconstructions may be soft/averaged — an expected, instructive property of
this two-stage route.

Usage
-----
    python examples/generative_jepa/03_train_decoder.py --epochs 5
    python examples/generative_jepa/03_train_decoder.py --epochs 1 --limit 2000   # quick smoke

Output
------
    output/<experiment>/checkpoints/decoder.pt   decoder weights (consumed by example 05)
    output/<experiment>/samples/recon.png        reconstructions of a held-out batch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.mnist import get_mnist_dataloaders, patchify
from ssllab.eval.viz import save_image_grid
from ssllab.experiment import experiment
from ssllab.models.decoder import LatentDecoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a z->image decoder on the frozen JEPA latent.")
    p.add_argument("--experiment", type=str, default="jepa_mnist", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output", help="root dir for experiment outputs")
    p.add_argument("--encoder", type=str, default=None, help="override encoder path (default: from experiment)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit", type=int, default=None)
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
    decoder = LatentDecoder(latent_dim=jepa.cfg.embed_dim).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir, limit=args.limit)

    for epoch in range(args.epochs):
        decoder.train()
        running = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            with torch.no_grad():
                z = jepa.embed(patchify(images))
            target = images.reshape(images.shape[0], -1)
            logits = decoder(z)["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        logger.info("epoch %d/%d  recon_bce=%.4f", epoch + 1, args.epochs, running / max(len(train_loader), 1))

    # Save reconstructions of a held-out batch.
    decoder.eval()
    with torch.no_grad():
        imgs, _ = next(iter(test_loader))
        imgs = imgs[:32].to(device)
        recon = decoder.decode_images(jepa.embed(patchify(imgs)))
        grid = torch.cat([imgs.cpu(), recon.cpu()], dim=0)  # originals (top) vs recon (bottom)
    recon_out = exp.samples / "recon.png"
    out = exp.checkpoints / "decoder.pt"
    save_image_grid(grid, recon_out, nrow=16)
    torch.save({"state": decoder.state_dict(), "latent_dim": jepa.cfg.embed_dim}, out)
    logger.info("saved decoder -> %s ; reconstructions -> %s", out, recon_out)


if __name__ == "__main__":
    main()
