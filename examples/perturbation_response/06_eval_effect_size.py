"""Eval — does the method recover effect size?

The benchmark that decides everything (Part 4 of the tutorial / Part 5 of the
design-space series): for each perturbation, generate a predicted response, and
correlate its differential expression Δ = mean(perturbed) − mean(control) against
the true Δ on the perturbation's top-DE genes (Pearson r). A model with a good
latent but poor effect size — the Cell-JEPA failure — scores low here; the whole
point of the conditional-flow + count-decoder method is to score high.

Default split is **cells** (held-out cells of *seen* perturbations) — the
in-distribution test of effect-size recovery. (Held-out *combos* need a
compositional perturbation embedding; that is a follow-up, since a learned
per-perturbation table cannot embed an unseen combination.)

Usage
-----
    python examples/perturbation_response/06_eval_effect_size.py --n 200
    python examples/perturbation_response/06_eval_effect_size.py --limit-perts 20 --n 100   # smoke

Output
------
    output/<experiment>/reports/effect_size.json
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

from ssllab.data.perturbseq import SPLIT_TEST, load_cache
from ssllab.eval.effect_size import run_effect_size_eval
from ssllab.experiment import experiment
from ssllab.generative.perturb import load_cond_flow, load_count_decoder, predicted_expression
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate effect-size recovery (Pearson Δ on top-DE genes).")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="cells", choices=["cells", "combo"])
    p.add_argument("--n", type=int, default=200, help="generated cells per perturbation")
    p.add_argument("--steps", type=int, default=100, help="ODE integration steps")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight")
    p.add_argument("--top-k", type=int, default=20, help="top-DE genes for the correlation")
    p.add_argument("--min-test-cells", type=int, default=20, help="skip perts with fewer held-out cells")
    p.add_argument("--limit-perts", type=int, default=None, help="cap #perturbations (smoke)")
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

    hvg = cache.hvg_X                                  # (N, G) normalized expression
    split_col = cache.split_cells if args.split == "cells" else cache.split_combo
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)
    gen = torch.Generator().manual_seed(args.seed)

    def predict(pid: int, name: str):
        return predicted_expression(bundle, decoder, pid, args.n,
                                    guidance=args.guidance, steps=args.steps, device=device, generator=gen)

    per_pert, summary = run_effect_size_eval(
        predict,
        hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
        is_test=(split_col == SPLIT_TEST), de_genes=cache.de_genes["per_pert"],
        control_mean=control_mean, top_k=args.top_k, min_test_cells=args.min_test_cells,
        limit_perts=args.limit_perts, log=logger.info,
    )
    logger.info("effect-size Δ-correlation: mean %.3f  median %.3f  over %d perturbations (split=%s, n=%d)",
                summary["mean_delta_r"], summary["median_delta_r"], summary["n_perturbations"], args.split, args.n)

    exp.write_report("effect_size", {
        "split": args.split, "n_generated": args.n, "guidance": args.guidance, "top_k": args.top_k,
        "cond_type": bundle.get("cond_type", "table"), **summary, "per_pert": per_pert,
    })
    logger.info("wrote report -> %s", exp.reports / "effect_size.json")


if __name__ == "__main__":
    main()
