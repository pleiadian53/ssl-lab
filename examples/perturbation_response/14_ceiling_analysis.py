"""Ceiling analysis — how well could Stage B *possibly* do?

Three rounds of this project have improved Stage B (the thing that turns a perturbation into a
cloud of outcome latents): a Gaussian flow, a transport flow, an action operator. They land at
0.612, 0.648, 0.645. The NB-VAE sits at 0.766. Every round asked "is this lever better than the
last one" and none asked the prior question:

    **Given this frozen encoder and this trained decoder, what is the best score ANY Stage B
    could achieve?**

That is a ceiling, and it is measurable without training anything. Stage B's entire job is to
produce the perturbed latents. So hand the pipeline the *real* ones -- encode the actual held-out
perturbed cells -- and score the result through the same decoder and the same metric. That is a
Stage B which is perfect by construction, and whatever it scores is the ceiling. Any shortfall
below 1.0 is information the **encoder and decoder** destroyed, and no Stage B can put it back.

The arms form a ladder, each one removing a suspect:

    identity     pred = the true test mean.                       Must score 1.000.
                 An acceptance gate on the harness itself: if this is not 1.0, the metric is
                 broken and every other number on this page is meaningless. (Chapter 3e of the
                 method series exists because exactly that happened once.)

    roundtrip    z = E(real held-out perturbed cells) -> decoder. THE CEILING.
                 Stage B is perfect: it produced the real latents, cell for cell. What survives
                 is what the encoder kept and the decoder could read back.

    latent_mean  the *mean* real test latent, zero spread -> decoder.
                 Asks whether latent spread matters for the effect-size metric at all. If this
                 ties `roundtrip`, a deterministic Stage B is sufficient for Delta-r, and the
                 whole stochastic-operator question is irrelevant to the primary endpoint.

    linear       ridge z -> expression, fit on TRAIN cells only, applied to real test latents.
                 Bypasses the NB decoder entirely, so it *attributes* the loss. If `linear` beats
                 `roundtrip`, the effect is present in the latent and the decoder is losing it. If
                 `linear` is no better, the **encoder** already threw it away, and no decoder lever
                 can help.

Read against the standing scoreboard (transport flow 0.648, operator 0.645, NB-VAE 0.766), the
`roundtrip` number decides the project's next move:

  * roundtrip near 0.65  -> Stage B is SATURATED. The flow/operator tie is a wall, not a failure,
                            and further Stage-B levers cannot pay. If it also sits *below* 0.766,
                            then no Stage-B lever can ever reach the VAE and the frozen
                            representation is provably the bottleneck.
  * roundtrip near 0.95  -> Stage B has real headroom and genuinely underperforms.

Nothing is trained here (the ridge is a closed-form solve). CPU, minutes.

Usage
-----
    python examples/perturbation_response/14_ceiling_analysis.py \
        --encoder output/norman_stage_a/checkpoints/encoder.pt \
        --decoder output/norman_flow_control/checkpoints/count_decoder.pt \
        --split combo

Output
------
    output/<experiment>/reports/ceiling.json
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import (
    DEFAULT_N_TOKENS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    load_cache,
    make_gene_partition,
    tokenize_cells,
)
from ssllab.eval.effect_size import run_effect_size_eval
from ssllab.experiment import experiment
from ssllab.generative.perturb import load_count_decoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ARMS = ("identity", "roundtrip", "latent_mean", "linear")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle ceiling for Stage B (no training).")
    p.add_argument("--experiment", type=str, default="norman_ceiling")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--encoder", type=str, default="output/norman_stage_a/checkpoints/encoder.pt")
    p.add_argument("--decoder", type=str, default="output/norman_flow_control/checkpoints/count_decoder.pt")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"])
    p.add_argument("--arms", type=str, nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--ridge-lambda", type=float, default=1.0, help="L2 for the linear readout arm")
    p.add_argument("--batch-size", type=int, default=256)
    # These three MUST match the harness the published numbers were graded on.
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-test-cells", type=int, default=20)
    p.add_argument("--limit-perts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _md5(path: Path) -> str:
    """Hash of an input artifact, recorded in the report so it can prove what it was computed on.

    The Round-3 reports do not say which decoder they used, and establishing it after the fact
    meant hash-comparing checkpoints. Reports should not need forensics.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def encode_rows(jepa, feat: np.ndarray, partition: torch.Tensor, rows: np.ndarray,
                device, batch_size: int) -> torch.Tensor:
    """Frozen latents for an explicit set of cache rows, in encoder space (NOT standardized).

    Stage C trains the decoder on `jepa.embed(tokens)` directly, and Stage B de-standardizes before
    decoding, so encoder space is what the decoder expects. Indexing the cache by row (rather than
    draining a dataloader) is what guarantees we encode *exactly* the cells the metric scores.
    """
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = torch.from_numpy(np.ascontiguousarray(feat[rows[i:i + batch_size]]))
        tokens = tokenize_cells(chunk, partition).to(device)
        out.append(jepa.embed(tokens).cpu())
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def decode_mean(decoder, z: torch.Tensor, device) -> torch.Tensor:
    """Mean predicted log1p-CP10K over a cloud of latents -- the identical readout path as `06`.

    `predicted_expression()` in perturb.py does exactly this to the flow's sampled latents; the
    only thing this script changes is *where the latents came from*.
    """
    z = z.to(device)
    rho = decoder(z, library_size=torch.ones(len(z), device=device))["rho"]
    return torch.log1p(1e4 * rho).mean(0).cpu()


def fit_ridge(Z: torch.Tensor, Y: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge from latents to expression, fit on TRAIN cells only.

    Solves min_W ||Z W - Y||^2 + lam ||W||^2 on centered data. Returns (W, intercept). This is a
    *readout probe*, not a model: it asks how much of the effect is linearly present in the frozen
    latent, with the NB decoder taken out of the picture entirely.
    """
    z_mu, y_mu = Z.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Zc, Yc = (Z - z_mu).double(), (Y - y_mu).double()
    d = Zc.shape[1]
    A = Zc.T @ Zc + lam * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.solve(A, Zc.T @ Yc)
    return W.float(), (y_mu - z_mu @ W.float()).squeeze(0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    enc_path, dec_path = Path(args.encoder), Path(args.decoder)
    for p in (enc_path, dec_path):
        if not p.exists():
            raise FileNotFoundError(f"missing required checkpoint: {p}")

    jepa = load_jepa(enc_path, device)
    decoder = load_count_decoder(dec_path, device)
    cache = load_cache(Path(args.data_dir) / args.artifact)

    hvg = cache.hvg_X                                              # (N, G) normalized log1p-CP10K
    split_col = cache.split_combo if args.split == "combo" else cache.split_cells
    is_test = split_col == SPLIT_TEST
    control_mean = hvg[cache.is_control.astype(bool)].mean(0)

    tok = cache.manifest.get("tokenization", {})
    partition = make_gene_partition(
        cache.n_hvg, int(tok.get("n_tokens", DEFAULT_N_TOKENS)), int(tok.get("partition_seed", 0))
    )

    logger.info("encoder %s (md5 %s)", enc_path, _md5(enc_path)[:8])
    logger.info("decoder %s (md5 %s)", dec_path, _md5(dec_path)[:8])
    logger.info("split=%s  %d test cells  %d genes", args.split, int(is_test.sum()), cache.n_hvg)

    # The linear arm's readout is fit on TRAIN cells only -- never on anything the metric scores.
    ridge = None
    if "linear" in args.arms:
        train_rows = np.flatnonzero(split_col == SPLIT_TRAIN)
        Z_tr = encode_rows(jepa, hvg, partition, train_rows, device, args.batch_size)
        Y_tr = torch.from_numpy(np.ascontiguousarray(hvg[train_rows]))
        ridge = fit_ridge(Z_tr, Y_tr, args.ridge_lambda)
        logger.info("ridge readout fit on %d TRAIN cells (lambda=%.3g)", len(train_rows), args.ridge_lambda)

    # Cache the per-perturbation test latents once; every arm reads the same cells the metric does.
    def rows_of(pid: int) -> np.ndarray:
        return np.flatnonzero((cache.pert_id == pid) & is_test)

    latents: dict[int, torch.Tensor] = {}

    def z_of(pid: int) -> torch.Tensor:
        if pid not in latents:
            latents[pid] = encode_rows(jepa, hvg, partition, rows_of(pid), device, args.batch_size)
        return latents[pid]

    def predictor(arm: str):
        def predict(pid: int, name: str) -> torch.Tensor:
            if arm == "identity":
                # The truth itself. Scores 1.000 or the harness is broken.
                return torch.from_numpy(hvg[rows_of(pid)].mean(0))
            z = z_of(pid)
            if arm == "roundtrip":
                return decode_mean(decoder, z, device)
            if arm == "latent_mean":
                return decode_mean(decoder, z.mean(0, keepdim=True), device)
            if arm == "linear":
                W, b = ridge
                return (z @ W + b).mean(0)
            raise ValueError(arm)
        return predict

    results: dict[str, dict] = {}
    for arm in args.arms:
        per_pert, summary = run_effect_size_eval(
            predictor(arm),
            hvg_X=hvg, pert_names=cache.pert_names, pert_id=cache.pert_id,
            is_test=is_test, de_genes=cache.de_genes["per_pert"],
            control_mean=control_mean, top_k=args.top_k, min_test_cells=args.min_test_cells,
            limit_perts=args.limit_perts,
        )
        results[arm] = {**summary, "per_pert": per_pert}
        logger.info("%-12s  Delta-r mean %.3f  median %.3f  over %d perturbations",
                    arm, summary["mean_delta_r"], summary["median_delta_r"], summary["n_perturbations"])

    # Acceptance gate. A metric that cannot score the truth as 1.0 cannot score anything.
    if "identity" in results:
        got = results["identity"]["mean_delta_r"]
        if not (got > 0.999):
            raise RuntimeError(
                f"ACCEPTANCE GATE FAILED: the `identity` arm feeds the metric its own ground truth "
                f"and must score 1.000, but scored {got:.4f}. The harness is misaligned (most likely "
                f"the predicted rows are not the rows the metric averages). Every other number in "
                f"this report is meaningless until this passes."
            )
        logger.info("acceptance gate PASSED: identity arm scores %.4f", got)

    exp.write_report("ceiling", {
        "split": args.split,
        "top_k": args.top_k,
        "min_test_cells": args.min_test_cells,
        "ridge_lambda": args.ridge_lambda,
        "encoder": str(enc_path), "encoder_md5": _md5(enc_path),
        "decoder": str(dec_path), "decoder_md5": _md5(dec_path),
        "de_genes_ranked_by": cache.de_genes.get("ranked_by"),
        "arms": results,
    })
    logger.info("wrote report -> %s", exp.reports / "ceiling.json")

    if "roundtrip" in results:
        ceil = results["roundtrip"]["mean_delta_r"]
        logger.info("")
        logger.info("CEILING (oracle Stage B) = %.3f   [flow 0.648 | operator 0.645 | NB-VAE 0.766]", ceil)


if __name__ == "__main__":
    main()
