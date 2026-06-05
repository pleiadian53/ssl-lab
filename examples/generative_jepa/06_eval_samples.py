"""Intrinsic evaluation of generated samples — beyond eyeballing.

Generates samples from the trained flow prior + decoder, embeds them (and real
test images) with an *independent* MNIST CNN oracle, and computes a battery of
modality-agnostic metrics: classifier confidence + class coverage (mode-collapse
check), FID/KID (distributional distance), precision/recall + density/coverage
(fidelity vs diversity), and nearest-neighbor novelty (memorization check).

See ../docs/evaluating-generated-samples.md for the methodology.

Usage
-----
    python examples/generative_jepa/06_eval_samples.py
    python examples/generative_jepa/06_eval_samples.py --n 5000 --oracle-epochs 3

Output
------
    output/<experiment>/reports/sample_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.mnist import get_mnist_dataloaders
from ssllab.eval.generative import (
    classifier_metrics,
    density_coverage,
    fid,
    kid,
    nn_distance_stats,
    precision_recall,
)
from ssllab.eval.oracle_mnist import load_or_train_oracle
from ssllab.experiment import experiment
from ssllab.generative.flow import VelocityMLP, euler_sample
from ssllab.models.decoder import LatentDecoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intrinsic evaluation of generated samples.")
    p.add_argument("--experiment", type=str, default="jepa_mnist")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--n", type=int, default=5000, help="# samples (and # real refs) to evaluate")
    p.add_argument("--steps", type=int, default=100, help="ODE steps for sampling")
    p.add_argument("--oracle-epochs", type=int, default=2)
    p.add_argument("--oracle-path", type=str, default="output/oracles/mnist_cnn.pt")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def _oracle_features_probs(model, images, device, batch=512):
    feats, probs = [], []
    for i in range(0, len(images), batch):
        x = images[i : i + batch].to(device)
        feats.append(model.features(x).cpu())
        probs.append(model.proba(x).cpu())
    return torch.cat(feats), torch.cat(probs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    # 1. Independent oracle (train + cache if needed).
    oracle = load_or_train_oracle(args.oracle_path, device=device,
                                  epochs=args.oracle_epochs, data_dir=args.data_dir)

    # 2. Generate samples: z ~ prior -> decode.
    pck = torch.load(exp.checkpoints / "prior.pt", map_location=device)
    prior = VelocityMLP(data_dim=pck["dim"], hidden=pck["hidden"], n_layers=pck["n_layers"]).to(device)
    prior.load_state_dict(pck["state"])
    mean, std = pck["mean"].to(device), pck["std"].to(device)
    dck = torch.load(exp.checkpoints / "decoder.pt", map_location=device)
    decoder = LatentDecoder(latent_dim=dck["latent_dim"]).to(device)
    decoder.load_state_dict(dck["state"])
    decoder.eval()
    with torch.no_grad():
        z = euler_sample(prior, args.n, pck["dim"], n_steps=args.steps, device=device) * std + mean
        gen_images = decoder.decode_images(z).cpu()
    logger.info("generated %d samples", args.n)

    # 3. Real reference sets (test for FID/PR, train for novelty baseline).
    _, test_loader = get_mnist_dataloaders(256, args.data_dir)
    train_loader, _ = get_mnist_dataloaders(256, args.data_dir)
    real = torch.cat([x for x, _ in test_loader])[: args.n]
    train_real = torch.cat([x for x, _ in train_loader])[: args.n]

    # 4. Oracle embeddings + probabilities. Standardize features by the REAL set's
    #    per-dim statistics so the distance-based metrics (FID/PR/coverage/novelty)
    #    are scale-invariant across this arbitrary (non-Inception) feature space.
    gen_f, gen_p = _oracle_features_probs(oracle, gen_images, device)
    real_f, _ = _oracle_features_probs(oracle, real, device)
    train_f, _ = _oracle_features_probs(oracle, train_real, device)
    mu, sd = real_f.mean(0, keepdim=True), real_f.std(0, keepdim=True) + 1e-6
    real_f = (real_f - mu) / sd
    gen_f = (gen_f - mu) / sd
    train_f = (train_f - mu) / sd

    # 5. The battery.
    report = {
        "experiment": args.experiment,
        "n_samples": args.n,
        "classifier": classifier_metrics(gen_p),
        "fid": fid(real_f, gen_f),
        "kid": kid(real_f, gen_f),
        "precision_recall": precision_recall(real_f, gen_f, k=3),
        "density_coverage": density_coverage(real_f, gen_f, k=5),
        "novelty_gen_vs_train": nn_distance_stats(gen_f, train_f, k=1),
        "novelty_baseline_test_vs_train": nn_distance_stats(real_f, train_f, k=1),
    }

    out = exp.write_report("sample_eval", report)
    logger.info("=== sample evaluation (%s) ===", args.experiment)
    logger.info("classifier   : confidence=%.3f  coverage_entropy=%.3f  classes=%d/%d",
                report["classifier"]["confidence"], report["classifier"]["coverage_entropy"],
                report["classifier"]["classes_covered"], report["classifier"]["n_classes"])
    logger.info("FID=%.3f  KID=%.4f", report["fid"], report["kid"])
    logger.info("precision=%.3f recall=%.3f | density=%.3f coverage=%.3f",
                report["precision_recall"]["precision"], report["precision_recall"]["recall"],
                report["density_coverage"]["density"], report["density_coverage"]["coverage"])
    logger.info("novelty NN(gen->train) median=%.3f  vs baseline(test->train)=%.3f",
                report["novelty_gen_vs_train"]["nn_median"],
                report["novelty_baseline_test_vs_train"]["nn_median"])
    logger.info("report -> %s", out)


if __name__ == "__main__":
    main()
