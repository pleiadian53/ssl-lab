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

from ssllab.generative.condition import ConditionEncoder
from ssllab.generative.count_decoder import CountDecoder
from ssllab.generative.flow import VelocityMLP, euler_sample


def load_cond_flow(path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Load the Stage-B checkpoint -> ready flow, condition encoder, and stats."""
    ck = torch.load(path, map_location=device)
    flow = VelocityMLP(data_dim=ck["dim"], hidden=ck["hidden"], n_layers=ck["n_layers"],
                       cond_dim=ck["cond_dim"]).to(device)
    flow.load_state_dict(ck["flow"])
    flow.eval()
    cond = ConditionEncoder(latent_dim=ck["dim"], n_perts=ck["n_perts"],
                            pert_dim=ck["pert_dim"], cond_dim=ck["cond_dim"]).to(device)
    cond.load_state_dict(ck["cond"])
    cond.eval()
    return {
        "flow": flow, "cond": cond,
        "mean": ck["mean"].to(device), "std": ck["std"].to(device),
        "ctrl_pool": ck["ctrl_pool"].to(device), "dim": ck["dim"], "n_perts": ck["n_perts"],
    }


def load_count_decoder(path: str | Path, device: torch.device | str = "cpu") -> CountDecoder:
    ck = torch.load(path, map_location=device)
    dec = CountDecoder(latent_dim=ck["latent_dim"], n_genes=ck["n_genes"], zinb=ck["zinb"]).to(device)
    dec.load_state_dict(ck["state"])
    dec.eval()
    return dec


@torch.no_grad()
def sample_perturbed_latents(
    bundle: dict, pert_id: int, n: int, guidance: float = 1.0, steps: int = 100,
    device: torch.device | str = "cpu", generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``n`` outcome latents for one perturbation: draw baselines z_b from the
    control pool, build c = cond(z_b, pert), integrate the flow, de-standardize."""
    flow, cond = bundle["flow"], bundle["cond"]
    pool = bundle["ctrl_pool"]
    idx = torch.randint(len(pool), (n,), generator=generator)
    z_b = pool[idx].to(device)
    pid = torch.full((n,), int(pert_id), dtype=torch.long, device=device)
    c = cond(z_b, pid)
    z_std = euler_sample(flow, n, bundle["dim"], n_steps=steps, device=device, c=c, guidance=guidance)
    return z_std * bundle["std"] + bundle["mean"]            # (n, dim), encoder-space latents


@torch.no_grad()
def predicted_expression(
    bundle: dict, decoder: CountDecoder, pert_id: int, n: int,
    guidance: float = 1.0, steps: int = 100, device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean predicted log1p-CP10K expression over a generated population for one
    perturbation — directly comparable to the cache's normalized ``hvg_X``.

    The decoder's gene-rate ``rho`` (sums to 1) is library-size-free, so the
    normalized expression is ``log1p(1e4 * rho)`` with no library-size needed.
    """
    z = sample_perturbed_latents(bundle, pert_id, n, guidance, steps, device, generator)
    rho = decoder(z, library_size=torch.ones(n, device=device))["rho"]   # (n, G)
    return torch.log1p(1e4 * rho).mean(0)                                  # (G,)
