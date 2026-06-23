"""Generate perturbed-cell count profiles for one perturbation.

The generative payoff: pick a perturbation, draw a population of outcome latents
from the conditional flow (baseline from the control pool), and decode each to a
gene-count profile via the count decoder. A thousand draws simulate the responding-
cell population. Saves the generated counts and prints the top up-regulated genes.

Usage
-----
    python examples/perturbation_response/05_sample_perturbed.py --pert KLF1 --n 500
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

from ssllab.data.perturbseq import load_cache
from ssllab.experiment import experiment
from ssllab.generative.perturb import load_cond_flow, load_count_decoder, sample_perturbed_latents
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate perturbed-cell count profiles for one perturbation.")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--pert", type=str, required=True, help="perturbation name (e.g. KLF1 or CEBPE+RUNX1T1)")
    p.add_argument("--n", type=int, default=500, help="cells to generate")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--libsize", type=float, default=None, help="library size for sampled counts (default: data median)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    bundle = load_cond_flow(exp.checkpoints / "cond_flow.pt", device)
    decoder = load_count_decoder(exp.checkpoints / "count_decoder.pt", device)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    names = list(cache.pert_names)
    if args.pert not in names:
        raise SystemExit(f"unknown perturbation {args.pert!r}. e.g. {names[1:6]}")
    pid = names.index(args.pert)
    libsize = args.libsize or float(np.median(cache.libsize))
    gene_ids = np.asarray(cache.gene_ids)

    gen = torch.Generator().manual_seed(args.seed)
    z = sample_perturbed_latents(bundle, pid, args.n, args.guidance, args.steps, device, gen)
    lib = torch.full((args.n,), libsize, device=device)
    counts = decoder.sample_counts(z, lib).cpu().numpy()

    out = exp.samples / f"generated_{args.pert.replace('+', '_')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, counts=counts.astype(np.int32), gene_ids=gene_ids, pert=args.pert, libsize=libsize)

    mean_counts = counts.mean(0)
    top = np.argsort(-mean_counts)[:10]
    logger.info("generated %d cells for '%s' (libsize %.0f) -> %s", args.n, args.pert, libsize, out)
    logger.info("top predicted genes: %s", ", ".join(f"{gene_ids[i]}({mean_counts[i]:.1f})" for i in top))


if __name__ == "__main__":
    main()
