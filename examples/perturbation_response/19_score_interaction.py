"""Score one run on magnitude and interaction metrics for the held-out combinations.

The standing benchmark (06, 09, 18) reports Δ-correlation, which ignores the size of a
change. This script scores the same predictions on size and on the departure from
additivity, using :mod:`ssllab.eval.interaction`:

- ``log_r``: log of the pair's error relative to the additive reference's error,
  ``r_p = ‖Δ̂_p − Δ_p‖ / ‖I_p‖`` on the pair's top-DE genes, where ``I_p`` is the measured
  interaction. Below 0 means closer to the truth than additivity.
- ``rel_err``: the relative error proposed for Round 8, ``e_p = ‖Δ̂_p − Δ_p‖ / ‖Δ_p‖``.
- ``interaction_r``: the correlation between predicted and measured interaction.
- ``internal_r``, ``internal_log_r``: the same with the model's *own* predicted singles,
  the only measures that isolate what the model adds to a pair beyond its singles.
- ``delta_r``: Δ-correlation, recomputed from the same predictions.
- ``floor_r``, ``floor_log_r``: a split-half estimate of how small ``r_p`` can get given
  the finite number of held-out cells (independent of the model).

**Same predictions as the standing benchmark.** The pair profiles are generated through
:func:`ssllab.eval.effect_size.run_effect_size_eval`, in the same order, with the same
generator seed and settings as 06 (flows) and 09 (NB-VAE), so they are the predictions
behind those scripts' Δ-correlation. If the run already has a Δ-correlation report, the
recomputed per-pair values must match it to 1e-9, or the script stops: a mismatch means
the predictions are not the ones the benchmark scored.

**Single-gene predictions** (for the internal metrics) are generated *after* the pairs,
with a separate generator seeded ``seed + 1``, so they cannot change the pair profiles.
The additive reference has no predicted singles of its own; its internal metrics are NaN.

Output
------
    output/<experiment>/reports/interaction_<kind>.json     per-pair metrics
    output/<experiment>/reports/profiles_<kind>.npz         the predicted mean profiles

Usage
-----
    python examples/perturbation_response/19_score_interaction.py \
        --experiment norman_sweep_control_s0 --kind flow \
        --decoder output/norman_flow_control/checkpoints/count_decoder.pt
    python examples/perturbation_response/19_score_interaction.py --experiment norman_vae_cond_s0 --kind vae
    python examples/perturbation_response/19_score_interaction.py --experiment norman_additive --kind additive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TEST, SPLIT_TRAIN, load_cache
from ssllab.eval.baselines import additive_predictor
from ssllab.eval.effect_size import run_effect_size_eval
from ssllab.eval.interaction import measured_single_deltas, pair_metrics, split_half_floor
from ssllab.experiment import experiment
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.generative.perturb import load_cond_flow, load_count_decoder, predicted_expression
from ssllab.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Where each kind's standing Δ-correlation report lives (written by 06, 09, 18).
EFFECT_REPORT = {"flow": "effect_size.json", "vae": "effect_size_cvae.json", "additive": "effect_size_additive.json"}
METRICS = ("log_r", "r", "rel_err", "rel_gi", "interaction_r", "delta_r", "internal_r", "internal_log_r")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score one run on magnitude and interaction metrics.")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--kind", type=str, required=True, choices=["flow", "vae", "additive"])
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--decoder", type=str, default=None, help="flow only: count decoder (default: the run's own)")
    p.add_argument("--n", type=int, default=200, help="generated cells per perturbation (as 06/09)")
    p.add_argument("--steps", type=int, default=100, help="flow ODE steps (as 06)")
    p.add_argument("--guidance", type=float, default=1.0, help="flow guidance (as 06)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-test-cells", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="generation seed (as 06/09)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-internal", action="store_true", help="skip the model's own single predictions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    # 1. The predictor, built exactly as 06 / 09 / 18 build it.
    if args.kind == "flow":
        bundle = load_cond_flow(exp.checkpoints / "cond_flow.pt", device)
        decoder = load_count_decoder(args.decoder or (exp.checkpoints / "count_decoder.pt"), device)
    elif args.kind == "vae":
        ck = torch.load(exp.checkpoints / "cvae_baseline.pt", map_location=device)
        model = ConditionalNBVAE.from_checkpoint(ck, device)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    hvg = cache.hvg_X
    split_col = cache.split_combo
    is_test, is_train = split_col == SPLIT_TEST, split_col == SPLIT_TRAIN
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)
    gen = torch.Generator().manual_seed(args.seed)

    if args.kind == "flow":
        def predict(pid: int, name: str, g: torch.Generator = gen):
            return predicted_expression(bundle, decoder, pid, args.n, guidance=args.guidance,
                                        steps=args.steps, device=device, generator=g)
    elif args.kind == "vae":
        def predict(pid: int, name: str, g: torch.Generator = gen):
            return model.generate_expression(pid, args.n, device=device, generator=g)
    else:
        additive = additive_predictor(hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
                                      is_train=is_train, control_mean=control_mean)

        def predict(pid: int, name: str, g=None):
            return additive(pid, name)

    # 2. Pair profiles through the shared harness: the predictions the benchmark scores.
    profiles: dict[str, np.ndarray] = {}
    per_pert_r, _ = run_effect_size_eval(
        predict, hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id, is_test=is_test,
        de_genes=cache.de_genes["per_pert"], control_mean=control_mean, top_k=args.top_k,
        min_test_cells=args.min_test_cells, profiles=profiles,
    )
    pairs = [k for k in profiles if "+" in k]
    if len(pairs) != len(profiles):
        raise SystemExit(f"expected only combinations on the combo split, got {sorted(set(profiles) - set(pairs))}")

    stored = exp.reports / EFFECT_REPORT[args.kind]
    if stored.exists():
        ref = json.loads(stored.read_text())["per_pert"]
        worst = max(abs(per_pert_r[k] - ref[k]) for k in per_pert_r)
        if set(ref) != set(per_pert_r) or worst > 1e-9:
            raise SystemExit(f"Δ-correlation does not reproduce {stored} (max |diff| {worst:.3g}); "
                             "these are not the predictions the benchmark scored")
        logger.info("Δ-correlation reproduces %s exactly (max |diff| %.1e)", stored.name, worst)
    else:
        logger.info("no stored %s; nothing to reproduce", stored.name)

    # 3. Measured singles (the common reference) and, optionally, the model's own singles.
    genes = sorted({g for k in pairs for g in k.split("+")})
    measured = measured_single_deltas(genes, hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
                                      is_train=is_train, control_mean=control_mean)
    names = np.asarray(cache.pert_names)
    pred_single: dict[str, np.ndarray] = {}
    if args.kind != "additive" and not args.no_internal:
        gen_s = torch.Generator().manual_seed(args.seed + 1)
        for g in genes:
            pid = int(np.where(names == g)[0][0])
            out = predict(pid, g, gen_s)
            pred_single[g] = np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=np.float64)

    # 4. Per-pair metrics.
    per_pert: dict[str, dict] = {}
    for k in pairs:
        a, b = k.split("+")
        pid = int(np.where(names == k)[0][0])
        mask = (cache.pert_id == pid) & is_test
        top_idx = [i for i in cache.de_genes["per_pert"][k]["top_idx"][:args.top_k] if i < hvg.shape[1]]
        m = pair_metrics(pred_pair=profiles[k], true_pair=hvg[mask].mean(0), control_mean=control_mean,
                         delta_a=measured[a], delta_b=measured[b], top_idx=top_idx,
                         pred_a=pred_single.get(a), pred_b=pred_single.get(b))
        if abs(m["delta_r"] - per_pert_r[k]) > 1e-5:
            raise SystemExit(f"{k}: recomputed Δ-correlation {m['delta_r']:.6f} != harness {per_pert_r[k]:.6f}")
        m.update(split_half_floor(hvg[mask], control_mean=control_mean, delta_a=measured[a],
                                  delta_b=measured[b], top_idx=top_idx, seed=0))
        per_pert[k] = m

    def mean_of(key: str) -> float:
        v = np.array([d[key] for d in per_pert.values()], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    summary = {f"mean_{key}": mean_of(key) for key in METRICS}
    summary["geometric_mean_r"] = float(np.exp(summary["mean_log_r"]))
    for key in ("log_r", "rel_err", "interaction_r", "internal_r", "delta_r"):
        logger.info("%-16s mean %+.4f", key, summary[f"mean_{key}"])
    logger.info("geometric-mean error ratio to additivity: %.3f over %d pairs", summary["geometric_mean_r"], len(per_pert))

    exp.write_report(f"interaction_{args.kind}", {
        "experiment": args.experiment, "kind": args.kind, "split": "combo", "top_k": args.top_k,
        "n_generated": args.n, "steps": args.steps if args.kind == "flow" else None,
        "guidance": args.guidance if args.kind == "flow" else None, "seed": args.seed,
        "singles_seed": args.seed + 1 if pred_single else None,
        "decoder": str(args.decoder) if args.decoder else None,
        "space": "log1p-CP10K", "singles_from": "train split", "n_pairs": len(per_pert),
        **summary, "per_pert": per_pert,
    })
    np.savez_compressed(
        exp.reports / f"profiles_{args.kind}.npz",
        pair_names=np.array(pairs), pair_profiles=np.stack([profiles[k] for k in pairs]),
        single_names=np.array(sorted(pred_single)),
        single_profiles=(np.stack([pred_single[g] for g in sorted(pred_single)]) if pred_single
                         else np.zeros((0, hvg.shape[1]))),
    )
    logger.info("wrote %s and profiles_%s.npz", exp.reports / f"interaction_{args.kind}.json", args.kind)


if __name__ == "__main__":
    main()
