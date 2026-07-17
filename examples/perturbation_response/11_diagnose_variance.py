"""Diagnostic — split the predicted population's spread into its two owners.

The law of total variance says a generated population's per-gene variance is exactly

    Var[y]  =  E_z[ Var(y | z) ]  +  Var_z[ E(y | z) ]
               \____ sigma2_dec ___/    \___ sigma2_bio __/

where ``z`` is the latent drawn from the model (flow or VAE) and ``y`` is the readout the
calibration metric actually scores: log1p-CP10K expression of *sampled counts*.

  sigma2_bio  the spread of the decoded MEAN across the latent cloud. This is the part the
              latent distribution owns, and the ONLY part the flow and the VAE do differently.
  sigma2_dec  the count noise the shared decoder adds around each cell's own mean. Shared.

This script measures both, per perturbation, on the top-DE genes, and compares their sum
against the variance of the real held-out cells. It answers three questions that decide
whether the decoder lever is worth a training run at all:

  1. Is the decoder over-dispersed, and by what factor?   (sigma2_dec vs the real variance)
  2. Does the latent cloud contribute ANY spread?         (sigma2_bio / total)
  3. What should the dispersion anchor actually target?   (the RESIDUAL: var_real - sigma2_bio,
     not var_real, because anchoring to the observed variance double-counts sigma2_bio)

It trains nothing and needs no GPU. Read-only over existing checkpoints.

Usage
-----
    python examples/perturbation_response/11_diagnose_variance.py --experiment norman_flow_control
    python examples/perturbation_response/11_diagnose_variance.py --experiment norman_dec_statedisp
    python examples/perturbation_response/11_diagnose_variance.py --experiment norman_combo --model vae

Output
------
    output/<experiment>/reports/variance_decomposition_<model>.json
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
from ssllab.experiment import experiment
from ssllab.generative.cvae import ConditionalNBVAE
from ssllab.generative.perturb import load_cond_flow, load_count_decoder, load_operator, sample_perturbed_latents
from ssllab.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decompose predicted spread into latent (bio) and decoder (technical).")
    p.add_argument("--experiment", type=str, default="norman_flow_control")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--model", type=str, default="flow", choices=["flow", "vae"])
    p.add_argument("--stage-b", type=str, default="flow", choices=["flow", "operator"])
    p.add_argument("--decoder", type=str, default=None, help="count_decoder.pt to REUSE")
    p.add_argument("--split", type=str, default="combo", choices=["cells", "combo"])
    p.add_argument("--n-latents", type=int, default=200, help="latents drawn per perturbation (the cloud)")
    p.add_argument("--n-reps", type=int, default=64, help="count draws PER latent (estimates Var(y|z))")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-test-cells", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def decompose(
    rho: torch.Tensor,          # (L, G) decoded rate profile, one row per latent
    kappa: torch.Tensor,        # (G,) or (L, G) NB dispersion
    top_idx: np.ndarray,        # (K,) the scored genes
    libsize: float,
    n_reps: int,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo estimate of (sigma2_dec, sigma2_bio) per scored gene.

    For each latent i we draw ``n_reps`` count vectors, map them into the metric's space
    ``y = log1p(1e4 * counts / libsize)``, and form the within-latent mean and variance.
    Then sigma2_bio = Var_i(mean_j y) and sigma2_dec = E_i(var_j y). This is the law of
    total variance estimated directly in the space the metric scores, so it needs no
    delta-method approximation through the log1p.
    """
    idx = torch.as_tensor(top_idx, dtype=torch.long)
    rho_k = rho[:, idx]                                            # (L, K)
    kap_k = kappa[:, idx] if kappa.dim() == 2 else kappa[idx]      # (L, K) or (K,)
    mu_k = libsize * rho_k                                         # (L, K), constant library size

    L, K = mu_k.shape
    mu_e = mu_k.unsqueeze(1).expand(L, n_reps, K)                  # (L, R, K)
    kap_e = (kap_k.unsqueeze(1).expand(L, n_reps, K) if kap_k.dim() == 2
             else kap_k.view(1, 1, K).expand(L, n_reps, K))

    # NB as a Gamma-Poisson mixture, matching CountDecoder.sample_counts exactly.
    lam = torch._standard_gamma(kap_e.clamp_min(1e-8), generator=generator) * (mu_e / kap_e.clamp_min(1e-8))
    counts = torch.poisson(lam, generator=generator)                # (L, R, K)
    y = torch.log1p(1e4 * counts / libsize)                         # (L, R, K) the metric's space

    m_i = y.mean(dim=1)                                             # (L, K) E(y|z) per latent
    v_i = y.var(dim=1, unbiased=True)                               # (L, K) Var(y|z) per latent

    sigma2_bio = m_i.var(dim=0, unbiased=True).numpy()              # (K,) Var_z[E(y|z)]
    sigma2_dec = v_i.mean(dim=0).numpy()                            # (K,) E_z[Var(y|z)]
    return sigma2_dec, sigma2_bio


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()
    gen = torch.Generator().manual_seed(args.seed)

    cache = load_cache(Path(args.data_dir) / args.artifact)
    split_col = cache.split_cells if args.split == "cells" else cache.split_combo
    is_test = (split_col == SPLIT_TEST)
    libsize = float(np.median(np.asarray(cache.libsize)))
    logger.info("library size (median, as the calibration eval uses): %.0f", libsize)

    names = np.asarray(cache.pert_names)
    de = cache.de_genes["per_pert"]

    if args.model == "flow":
        bundle = (load_operator(exp.checkpoints / "operator.pt", device) if args.stage_b == "operator"
                  else load_cond_flow(exp.checkpoints / "cond_flow.pt", device))
        decoder = load_count_decoder(args.decoder or (exp.checkpoints / "count_decoder.pt"), device)
        logger.info("flow decoder flags: anchored_mean=%s state_dispersion=%s",
                    decoder.anchored_mean, decoder.state_dispersion)

        @torch.no_grad()
        def decode(pid: int):
            z = sample_perturbed_latents(bundle, pid, args.n_latents, steps=args.steps,
                                         device=device, generator=gen)
            out = decoder(z, library_size=torch.full((args.n_latents,), libsize, device=device))
            return out["rho"], out["kappa"]
    else:
        ck = torch.load(exp.checkpoints / "cvae_baseline.pt", map_location=device)
        model = ConditionalNBVAE(n_genes=ck["n_genes"], pert_gene=ck["pert_gene"], latent_dim=ck["latent_dim"],
                                 cond_dim=ck["cond_dim"], gene_dim=ck["gene_dim"], hidden=ck["hidden"],
                                 compose=ck["compose"]).to(device)
        model.load_state_dict(ck["state"]); model.eval()

        @torch.no_grad()
        def decode(pid: int):
            z = torch.randn(args.n_latents, model.latent_dim, generator=gen).to(device)
            pidt = torch.full((args.n_latents,), int(pid), dtype=torch.long, device=device)
            out = model.decode(z, pidt)
            return out["rho"], out["kappa"]

    per_pert: dict[str, dict] = {}
    for name in [p for p in names if p != "control" and p in de]:
        pid = int(np.where(names == name)[0][0])
        mask = (cache.pert_id == pid) & is_test
        if int(mask.sum()) < args.min_test_cells:
            continue
        top_idx = np.array([i for i in de[name]["top_idx"][:args.top_k] if i < cache.hvg_X.shape[1]], dtype=int)
        if top_idx.size == 0:
            continue

        rho, kappa = decode(pid)
        s2_dec, s2_bio = decompose(rho.cpu(), kappa.cpu(), top_idx, libsize, args.n_reps, gen)
        var_real = np.asarray(cache.hvg_X)[mask][:, top_idx].var(axis=0, ddof=1)

        # Aggregate the VARIANCES across genes first, then form ratios. Averaging per-gene
        # ratios lets a gene with near-zero real variance dominate the mean and blow the
        # summary up to nonsense.
        v_real, v_dec, v_bio = var_real.mean(), s2_dec.mean(), s2_bio.mean()
        total = v_dec + v_bio
        per_pert[name] = {
            "n_test_cells": int(mask.sum()),
            "var_real": float(v_real),
            "sigma2_dec": float(v_dec),
            "sigma2_bio": float(v_bio),
            "total_pred": float(total),
            "bio_fraction": float(v_bio / max(total, 1e-12)),
            "spread_ratio": float(total / max(v_real, 1e-12)),          # >1 too wide, <1 too narrow
            "residual_target": float(v_real - v_bio),                   # what sigma2_dec SHOULD be
        }

    def agg(k):
        v = np.array([d[k] for d in per_pert.values()], dtype=float)
        return float(np.nanmean(v)) if v.size else float("nan")

    summary = {k: agg(k) for k in
               ["var_real", "sigma2_dec", "sigma2_bio", "total_pred", "bio_fraction",
                "spread_ratio", "residual_target"]}
    summary["n_perturbations"] = len(per_pert)

    verdict = ("TOO WIDE (over-dispersed)" if summary["spread_ratio"] > 1.15
               else "TOO NARROW (under-dispersed)" if summary["spread_ratio"] < 0.87
               else "about right")
    logger.info("=" * 78)
    logger.info("VARIANCE DECOMPOSITION  [%s / %s / %s]  over %d perturbations, top-%d DE genes",
                args.experiment, args.model, args.split, len(per_pert), args.top_k)
    logger.info("  real per-gene variance           %.4f", summary["var_real"])
    logger.info("  predicted total  (dec + bio)     %.4f   = %.2fx the real spread -> %s",
                summary["total_pred"], summary["spread_ratio"], verdict)
    logger.info("    sigma2_dec  (shared decoder)   %.4f", summary["sigma2_dec"])
    logger.info("    sigma2_bio  (latent cloud)     %.4f", summary["sigma2_bio"])
    logger.info("  BIO FRACTION of predicted spread %.3f   <- the latent distribution's entire share",
                summary["bio_fraction"])
    logger.info("  residual target for sigma2_dec   %.4f   (= var_real - sigma2_bio)",
                summary["residual_target"])
    logger.info("=" * 78)

    exp.write_report(f"variance_decomposition_{args.model}", {
        "experiment": args.experiment, "model": args.model, "split": args.split,
        "n_latents": args.n_latents, "n_reps": args.n_reps, "top_k": args.top_k,
        "libsize": libsize, **summary, "per_pert": per_pert,
    })
    logger.info("wrote report -> %s", exp.reports / f"variance_decomposition_{args.model}.json")


if __name__ == "__main__":
    main()
