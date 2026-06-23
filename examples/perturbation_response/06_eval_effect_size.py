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
from ssllab.eval.effect_size import delta_correlation, summarize
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
    names = np.asarray(cache.pert_names)
    split_col = cache.split_cells if args.split == "cells" else cache.split_combo
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)
    de = cache.de_genes["per_pert"]
    gen = torch.Generator().manual_seed(args.seed)

    # Perturbations evaluable on this split: have DE genes + enough held-out cells.
    per_pert: dict[str, float] = {}
    n_genes = hvg.shape[1]
    evaluable = [p for p in names if p != "control" and p in de]
    if args.limit_perts:
        evaluable = evaluable[: args.limit_perts]

    for name in evaluable:
        pid = int(np.where(names == name)[0][0])
        test_mask = (cache.pert_id == pid) & (split_col == SPLIT_TEST)
        if int(test_mask.sum()) < args.min_test_cells:
            continue
        true_mean = hvg[test_mask].mean(0)
        pred_mean = predicted_expression(bundle, decoder, pid, args.n,
                                         guidance=args.guidance, steps=args.steps, device=device, generator=gen)
        top_idx = [i for i in de[name]["top_idx"][: args.top_k] if i < n_genes]
        per_pert[name] = delta_correlation(pred_mean, true_mean, control_mean, top_idx)

    summary = summarize(per_pert)
    logger.info("effect-size Δ-correlation: mean %.3f  median %.3f  over %d perturbations (split=%s, n=%d)",
                summary["mean_delta_r"], summary["median_delta_r"], summary["n_perturbations"], args.split, args.n)
    # Show a few of the best/worst for intuition.
    ranked = sorted(((v, k) for k, v in per_pert.items() if v == v), reverse=True)
    if ranked:
        logger.info("best: %s", ", ".join(f"{k}={v:.2f}" for v, k in ranked[:5]))
        logger.info("worst: %s", ", ".join(f"{k}={v:.2f}" for v, k in ranked[-5:]))

    exp.write_report("effect_size", {
        "split": args.split, "n_generated": args.n, "guidance": args.guidance, "top_k": args.top_k,
        **summary, "per_pert": per_pert,
    })
    logger.info("wrote report -> %s", exp.reports / "effect_size.json")


if __name__ == "__main__":
    main()
