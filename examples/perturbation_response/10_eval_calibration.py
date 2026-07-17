"""Eval — calibration axis: does the model capture the response *distribution*, not just its mean?

Grades a generated population against the held-out real cells on spread correlation, central-
interval coverage, and mean 1-Wasserstein (see [calibration.py](../../src/ssllab/eval/calibration.py)).
Works for the flow (``--model flow``, reads ``cond_flow.pt`` + ``count_decoder.pt``) or the NB-VAE
baseline (``--model vae``, reads ``cvae_baseline.pt``), so the two are graded identically.

Usage
-----
    python examples/perturbation_response/10_eval_calibration.py --experiment norman_flow_control --model flow
    python examples/perturbation_response/10_eval_calibration.py --experiment norman_combo --model vae

Output
------
    output/<experiment>/reports/calibration_<model>.json
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

import numpy as np

from ssllab.data.perturbseq import SPLIT_TEST, load_cache
from ssllab.eval.calibration import run_calibration_eval
from ssllab.experiment import experiment
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.generative.perturb import (
    load_cond_flow,
    load_count_decoder,
    load_operator,
    load_operator_algebra,
    predicted_population,
)
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate distributional calibration of the response.")
    p.add_argument("--experiment", type=str, default="norman_flow_control")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--model", type=str, default="flow", choices=["flow", "vae"])
    p.add_argument("--split", type=str, default="combo", choices=["cells", "combo"])
    p.add_argument("--n", type=int, default=200, help="generated cells per perturbation")
    p.add_argument("--steps", type=int, default=100, help="ODE steps (flow only)")
    p.add_argument("--guidance", type=float, default=1.0, help="CFG weight (flow only)")
    p.add_argument("--stage-b", type=str, default="flow", choices=["flow", "operator", "operator_algebra"],
                   help="which Stage-B model to grade: the conditional flow (cond_flow.pt), round 3's "
                        "action operator (operator.pt), or the operator-algebra variant "
                        "(operator_algebra.pt). All produce a cloud of outcome latents, so the harness "
                        "below is identical for any of them.")
    p.add_argument("--decoder", type=str, default=None,
                   help="path to a count_decoder.pt to REUSE. A Stage-B lever (flow or "
                        "operator) does not change the readout, so the decoder is reused "
                        "rather than retrained. Defaults to this experiment\'s own.")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-test-cells", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()
    gen = torch.Generator().manual_seed(args.seed)

    if args.model == "flow":
        if args.stage_b == "operator":
            bundle = load_operator(exp.checkpoints / "operator.pt", device)
        elif args.stage_b == "operator_algebra":
            bundle = load_operator_algebra(exp.checkpoints / "operator_algebra.pt", device)
        else:
            bundle = load_cond_flow(exp.checkpoints / "cond_flow.pt", device)
        decoder = load_count_decoder(args.decoder or (exp.checkpoints / "count_decoder.pt"), device)

        def predict_pop(pid: int, name: str):
            return predicted_population(bundle, decoder, pid, args.n, libsize,
                                        guidance=args.guidance, steps=args.steps, device=device, generator=gen)
    else:
        ck = torch.load(exp.checkpoints / "cvae_baseline.pt", map_location=device)
        model = ConditionalNBVAE(n_genes=ck["n_genes"], pert_gene=ck["pert_gene"], latent_dim=ck["latent_dim"],
                                 cond_dim=ck["cond_dim"], gene_dim=ck["gene_dim"], hidden=ck["hidden"],
                                 compose=ck["compose"]).to(device)
        model.load_state_dict(ck["state"]); model.eval()

        def predict_pop(pid: int, name: str):
            return model.generate_population(pid, args.n, libsize, device=device, generator=gen)

    cache = load_cache(Path(args.data_dir) / args.artifact)
    split_col = cache.split_cells if args.split == "cells" else cache.split_combo
    # Representative sequencing depth for the sampled counts (sets the technical-noise level).
    med_libsize = float(np.median(np.asarray(cache.libsize)))
    libsize = torch.full((args.n,), med_libsize)
    logger.info("sampling counts at median library size %.0f", med_libsize)
    per_pert, summary = run_calibration_eval(
        predict_pop,
        hvg_X=cache.hvg_X, pert_names=cache.pert_names, pert_id=cache.pert_id,
        is_test=(split_col == SPLIT_TEST), de_genes=cache.de_genes["per_pert"],
        top_k=args.top_k, min_test_cells=args.min_test_cells, log=logger.info,
    )
    logger.info("[%s/%s] calibration: spread_r %.3f  coverage %.3f (nominal %.2f)  wasserstein %.3f  over %d perts",
                args.model, args.split, summary["mean_spread_r"], summary["mean_coverage"],
                summary["nominal_coverage"], summary["mean_wasserstein"], summary["n_perturbations"])

    exp.write_report(f"calibration_{args.model}", {
        "model": args.model, "split": args.split, "n_generated": args.n, "top_k": args.top_k,
        **summary, "per_pert": per_pert,
    })
    logger.info("wrote report -> %s", exp.reports / f"calibration_{args.model}.json")


if __name__ == "__main__":
    main()
