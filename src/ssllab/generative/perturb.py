"""Generate perturbed-cell predictions from the trained conditional flow + decoder.

Sampling side of the perturbation pipeline: draw outcome latents from the
conditional flow given a perturbation (baseline drawn from the control pool),
then read out the decoder's normalized expression. Shared by the sampling script
(05) and the effect-size eval (06).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from ssllab.generative.condition import (
    ConditionEncoder,
    GeneSetConditionEncoder,
    GeneSetEmbedding,
)
from ssllab.generative.count_decoder import CountDecoder
from ssllab.generative.flow import VelocityMLP, euler_sample


def _build_condition(ck: dict, device: torch.device | str) -> nn.Module:
    """Rebuild the saved condition module from a Stage-B checkpoint.

    Four cases across (flow_base, cond_type). For the ``control`` transport base the
    condition is the **perturbation embedding alone** (``cond(pert_id)``); for the
    ``gaussian`` prior it fuses ``(z_b, pert_id)``.
    """
    flow_base = ck.get("flow_base", "gaussian")
    geneset = ck.get("cond_type", "table") == "geneset"
    if flow_base == "control":
        if geneset:
            cond = GeneSetEmbedding(ck["pert_gene"], gene_dim=ck.get("gene_dim", 64),
                                    out_dim=ck["cond_dim"], compose=ck.get("compose", "additive"))
        else:
            cond = nn.Embedding(ck["n_perts"], ck["cond_dim"])
    elif geneset:
        cond = GeneSetConditionEncoder(
            latent_dim=ck["dim"], pert_gene=ck["pert_gene"], pert_dim=ck["pert_dim"],
            gene_dim=ck.get("gene_dim", 64), cond_dim=ck["cond_dim"], compose=ck.get("compose", "additive"))
    else:
        cond = ConditionEncoder(latent_dim=ck["dim"], n_perts=ck["n_perts"],
                                pert_dim=ck["pert_dim"], cond_dim=ck["cond_dim"])
    cond = cond.to(device)
    cond.load_state_dict(ck["cond"])
    cond.eval()
    return cond


def load_cond_flow(path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Load the Stage-B checkpoint -> ready flow, condition module, and stats.

    Rebuilds the condition per the saved ``(flow_base, cond_type)``; the sampler
    branches on ``flow_base`` (``control`` transports a baseline latent; ``gaussian``
    starts from noise).
    """
    ck = torch.load(path, map_location=device)
    flow = VelocityMLP(data_dim=ck["dim"], hidden=ck["hidden"], n_layers=ck["n_layers"],
                       cond_dim=ck["cond_dim"]).to(device)
    flow.load_state_dict(ck["flow"])
    flow.eval()
    cond = _build_condition(ck, device)
    return {
        "flow": flow, "cond": cond,
        "mean": ck["mean"].to(device), "std": ck["std"].to(device),
        "ctrl_pool": ck["ctrl_pool"].to(device), "dim": ck["dim"], "n_perts": ck["n_perts"],
        "cond_type": ck.get("cond_type", "table"), "compose": ck.get("compose", "additive"),
        "flow_base": ck.get("flow_base", "gaussian"),
    }


def load_count_decoder(path: str | Path, device: torch.device | str = "cpu") -> CountDecoder:
    ck = torch.load(path, map_location=device)
    dec = CountDecoder(
        latent_dim=ck["latent_dim"], n_genes=ck["n_genes"], zinb=ck["zinb"],
        anchored_mean=ck.get("anchored_mean", False),
        state_dispersion=ck.get("state_dispersion", False),
    ).to(device)
    dec.load_state_dict(ck["state"])
    dec.eval()
    return dec


@torch.no_grad()
def sample_perturbed_latents(
    bundle: dict, pert_id: int, n: int, guidance: float = 1.0, steps: int = 100,
    device: torch.device | str = "cpu", generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``n`` outcome latents for one perturbation, integrate the flow, de-standardize.

    ``control`` base: draw baselines z_b from the control pool and transport them
    (source z0=z_b, condition = z_p) — the sample is anchored to a real baseline.
    ``gaussian`` base: start from noise, condition on the fused c=cond(z_b, z_p)."""
    flow, cond = bundle["flow"], bundle["cond"]
    pool = bundle["ctrl_pool"]
    idx = torch.randint(len(pool), (n,), generator=generator)
    z_b = pool[idx].to(device)
    pid = torch.full((n,), int(pert_id), dtype=torch.long, device=device)
    if bundle.get("flow_base", "gaussian") == "control":
        z_std = euler_sample(flow, n, bundle["dim"], n_steps=steps, device=device,
                             c=cond(pid), guidance=guidance, z0=z_b)
    else:
        z_std = euler_sample(flow, n, bundle["dim"], n_steps=steps, device=device,
                             c=cond(z_b, pid), guidance=guidance)
    return z_std * bundle["std"] + bundle["mean"]            # (n, dim), encoder-space latents


@torch.no_grad()
def predicted_expression(
    bundle: dict, decoder: CountDecoder, pert_id: int, n: int,
    guidance: float = 1.0, steps: int = 100, device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean predicted log1p-CP10K expression over a generated population ``(G,)`` —
    directly comparable to the cache's normalized ``hvg_X``.

    The decoder's gene-rate ``rho`` (sums to 1) is library-size-free, so the mean uses
    ``log1p(1e4 * rho)`` with no sampling — a clean estimate of the *mean* response (the
    effect-size metric). Per-cell *spread* needs count sampling; see :func:`predicted_population`.
    """
    z = sample_perturbed_latents(bundle, pert_id, n, guidance, steps, device, generator)
    rho = decoder(z, library_size=torch.ones(n, device=device))["rho"]   # (n, G)
    return torch.log1p(1e4 * rho).mean(0)                                  # (G,)


@torch.no_grad()
def predicted_population(
    bundle: dict, decoder: CountDecoder, pert_id: int, n: int, library_size: torch.Tensor,
    guidance: float = 1.0, steps: int = 100, device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-cell predicted log1p-CP10K expression for a generated population ``(n, G)``.

    Unlike :func:`predicted_expression`, this **samples counts** from the decoder's NB so the
    per-cell spread carries the real technical (count) noise that dominates scRNA-seq — without
    it a population of decoded rates is near-degenerate and no calibration is measurable.
    ``library_size`` ``(n,)`` sets the sequencing depth; counts are renormalized to log1p-CP10K.
    """
    z = sample_perturbed_latents(bundle, pert_id, n, guidance, steps, device, generator)
    ls = library_size.to(device)
    counts = decoder.sample_counts(z, ls)                                  # (n, G)
    return torch.log1p(1e4 * counts / ls.unsqueeze(-1))                    # (n, G)
