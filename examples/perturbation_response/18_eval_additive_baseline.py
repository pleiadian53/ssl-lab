"""Score the additive model, Δ(A+B) = Δ(A) + Δ(B), with the SAME effect-size metric as the models.

The additive model is the reference a learned model has to beat before its score means
anything: assume the two genes do not interact and add the single-gene effects. It needs
no training. Each single's effect is the mean of its cells on the training side of the
split minus the control mean, so no held-out cell is read. Grading goes through the shared
:func:`run_effect_size_eval`, so the number sits directly beside the flow's (06) and the
NB-VAE's (09), and ``12_compare_arms.py`` can contrast it with them as kind ``additive``.

``15_empirical_epistasis.py`` also builds an additive prediction, but from the *true*
singles and only to measure how far each pair departs from additivity. This script treats
the additive model as a competitor and scores it.

The default split is **combo**, the only split where the question makes sense: every single
is on the training side and the twenty held-out combinations are what gets scored. It is
deterministic, so there are no seeds.

The report also carries an ``exploratory`` block of cheaper references, added after the
additive result and reported without verdicts: each single alone, one average training
effect predicted for every pair, and two randomly chosen singles added together. They show
how much of the score depends on knowing the pair at all.

Usage
-----
    python examples/perturbation_response/18_eval_additive_baseline.py

Output
------
    output/<experiment>/reports/effect_size_additive.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TEST, SPLIT_TRAIN, load_cache
from ssllab.eval.baselines import additive_predictor
from ssllab.eval.effect_size import run_effect_size_eval
from ssllab.experiment import experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the additive model's effect-size recovery.")
    p.add_argument("--experiment", type=str, default="norman_additive")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="combo", choices=["combo"],
                   help="combo only: singles are all training, combinations are held out")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-test-cells", type=int, default=20)
    p.add_argument("--random-draws", type=int, default=20,
                   help="draws of the exploratory wrong-genes reference (two random singles)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp = experiment(args.experiment, args.output_root).ensure()

    cache = load_cache(Path(args.data_dir) / args.artifact)
    hvg = cache.hvg_X
    split_col = cache.split_combo
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)

    predict = additive_predictor(
        hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
        is_train=(split_col == SPLIT_TRAIN), control_mean=control_mean,
    )
    per_pert, summary = run_effect_size_eval(
        predict,
        hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
        is_test=(split_col == SPLIT_TEST), de_genes=cache.de_genes["per_pert"],
        control_mean=control_mean, top_k=args.top_k, min_test_cells=args.min_test_cells,
        log=logger.info,
    )
    logger.info("additive model Δ-correlation: mean %.3f  median %.3f  over %d perturbations (split=%s)",
                summary["mean_delta_r"], summary["median_delta_r"], summary["n_perturbations"], args.split)

    # Exploratory references, added after the pre-registered result and reported without
    # verdicts. They say how much of the score needs the pair at all: one single alone, a
    # prediction that ignores which pair it is, and the right-shaped answer for the wrong genes.
    def score(fn) -> dict[str, float]:
        _, s = run_effect_size_eval(
            fn, hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
            is_test=(split_col == SPLIT_TEST), de_genes=cache.de_genes["per_pert"],
            control_mean=control_mean, top_k=args.top_k, min_test_cells=args.min_test_cells,
        )
        return {"mean_delta_r": s["mean_delta_r"], "median_delta_r": s["median_delta_r"]}

    names = np.asarray(cache.pert_names)
    is_train = split_col == SPLIT_TRAIN
    train_perts = [i for i in np.unique(cache.pert_id[is_train]) if names[i] != "control"]
    mean_effect = np.mean([hvg[is_train & (cache.pert_id == i)].mean(0) for i in train_perts], axis=0)
    singles = [str(n) for n in names if n != "control" and "+" not in str(n)]
    rng = np.random.default_rng(0)
    random_pairs = [
        score(lambda pid, name: predict(pid, "+".join(rng.choice(singles, 2, replace=False))))["mean_delta_r"]
        for _ in range(args.random_draws)
    ]
    exploratory = {
        "first_single_alone": score(lambda pid, name: predict(pid, name.split("+")[0])),
        "second_single_alone": score(lambda pid, name: predict(pid, name.split("+")[1])),
        "mean_training_effect": score(lambda pid, name: mean_effect),
        "random_singles_pair": {"mean_delta_r": float(np.mean(random_pairs)),
                                "min": float(np.min(random_pairs)), "max": float(np.max(random_pairs)),
                                "draws": args.random_draws},
    }
    for k, v in exploratory.items():
        logger.info("exploratory  %-22s mean %.3f", k, v["mean_delta_r"])

    exp.write_report("effect_size_additive", {
        "model": "additive", "split": args.split, "top_k": args.top_k,
        "space": "log1p-CP10K", "singles_from": "train split",
        **summary, "per_pert": per_pert,
        "exploratory": exploratory,
    })
    logger.info("wrote report -> %s", exp.reports / "effect_size_additive.json")


if __name__ == "__main__":
    main()
