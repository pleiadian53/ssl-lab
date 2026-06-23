"""Eval the conditional NB-VAE baseline with the SAME effect-size metric as the flow.

Loads the trained baseline and scores Pearson Δ-correlation on top-DE genes via the
shared :func:`run_effect_size_eval`, so its number sits directly beside the flow's
(06). Default split **combo** = held-out-combo generalization.

Usage
-----
    python examples/perturbation_response/09_eval_cvae_baseline.py --split combo --n 200

Output
------
    output/<experiment>/reports/effect_size_cvae.json
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

from ssllab.data.perturbseq import SPLIT_TEST, load_cache
from ssllab.eval.effect_size import run_effect_size_eval
from ssllab.experiment import experiment
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the NB-VAE baseline's effect-size recovery.")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="combo", choices=["cells", "combo"])
    p.add_argument("--n", type=int, default=200, help="generated cells per perturbation")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-test-cells", type=int, default=20)
    p.add_argument("--limit-perts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    ck = torch.load(exp.checkpoints / "cvae_baseline.pt", map_location=device)
    model = ConditionalNBVAE(
        n_genes=ck["n_genes"], pert_gene=ck["pert_gene"], latent_dim=ck["latent_dim"],
        cond_dim=ck["cond_dim"], gene_dim=ck["gene_dim"], hidden=ck["hidden"], compose=ck["compose"],
    ).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    cache = load_cache(Path(args.data_dir) / args.artifact)
    hvg = cache.hvg_X
    split_col = cache.split_cells if args.split == "cells" else cache.split_combo
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)
    gen = torch.Generator().manual_seed(args.seed)

    def predict(pid: int, name: str):
        return model.generate_expression(pid, args.n, device=device, generator=gen)

    per_pert, summary = run_effect_size_eval(
        predict,
        hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
        is_test=(split_col == SPLIT_TEST), de_genes=cache.de_genes["per_pert"],
        control_mean=control_mean, top_k=args.top_k, min_test_cells=args.min_test_cells,
        limit_perts=args.limit_perts, log=logger.info,
    )
    logger.info("NB-VAE baseline Δ-correlation: mean %.3f  median %.3f  over %d perturbations (split=%s)",
                summary["mean_delta_r"], summary["median_delta_r"], summary["n_perturbations"], args.split)

    exp.write_report("effect_size_cvae", {
        "model": "conditional_nb_vae", "split": args.split, "n_generated": args.n, "top_k": args.top_k,
        "compose": ck["compose"], **summary, "per_pert": per_pert,
    })
    logger.info("wrote report -> %s", exp.reports / "effect_size_cvae.json")


if __name__ == "__main__":
    main()
