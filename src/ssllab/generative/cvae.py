"""Conditional NB-VAE — the from-scratch generative baseline.

The control that makes the flow's effect-size number interpretable: a vanilla
conditional generative model trained end-to-end on counts, with **no JEPA
pretraining and no flow prior**. If the full method (JEPA latent + conditional
flow + NB decoder) does not beat this, its extra machinery is not earning its keep.

Design (CPA/scVI-flavored, kept deliberately plain):

    x  --enc-->  q(z | x) = N(mu, sigma)            # MLP encoder on log1p-CP10K features
    z  ~  reparam                                    # latent, prior p(z) = N(0, I)
    [z, c]  --dec-->  rho = softmax(.)               # condition-injected count head
    mu = library_size * rho ;  NB(counts | mu, kappa)

The condition ``c`` is the SAME gene-compositional embedding the flow uses
(:class:`GeneSetEmbedding`), so the baseline also generalizes to held-out combos and
the comparison isolates the *generative machinery*, not the perturbation encoding.

Generation mirrors the flow's eval interface: draw ``z ~ N(0, I)``, inject the
perturbation condition, decode ``rho``, and read out ``log1p(1e4 * rho)`` (library-
size-free) — directly comparable to the cache's normalized ``hvg_X``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssllab.generative.condition import GeneSetEmbedding
from ssllab.generative.count_decoder import nb_nll


class ConditionalNBVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        pert_gene: torch.Tensor,
        latent_dim: int = 256,
        cond_dim: int = 128,
        gene_dim: int = 64,
        hidden: int = 512,
        compose: str = "additive",
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cond = GeneSetEmbedding(pert_gene, gene_dim=gene_dim, out_dim=cond_dim, compose=compose)
        self.enc = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.rate_head = nn.Linear(hidden, n_genes)            # -> rho via softmax
        self.log_kappa = nn.Parameter(torch.zeros(n_genes))    # per-gene NB dispersion

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        # Clamp logvar: an unbounded posterior log-variance lets exp(logvar) and the KL
        # term blow up to NaN once the encoder grows (seen at ~epoch 16 without this).
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def decode(self, z: torch.Tensor, pert_id: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.cond(pert_id)
        h = self.dec(torch.cat([z, c], dim=-1))
        rho = F.softmax(self.rate_head(h), dim=-1)
        kappa = F.softplus(self.log_kappa) + 1e-4
        return {"rho": rho, "kappa": kappa}

    def loss(
        self, x: torch.Tensor, counts: torch.Tensor, library_size: torch.Tensor,
        pert_id: torch.Tensor, beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mu_z, logvar = self.encode(x)
        z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar)     # reparam
        out = self.decode(z, pert_id)
        mu = library_size.unsqueeze(-1) * out["rho"]
        recon = nb_nll(counts, mu, out["kappa"])
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu_z.pow(2) - logvar.exp(), dim=1))
        return recon + beta * kl, {"recon": float(recon), "kl": float(kl)}

    @torch.no_grad()
    def generate_expression(
        self, pert_id: int, n: int, device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Mean predicted log1p-CP10K expression for one perturbation (eval interface)."""
        # Draw on CPU with the seeded generator, then move to device — a CUDA tensor
        # cannot take a CPU generator, and this keeps sampling deterministic across devices.
        z = torch.randn(n, self.latent_dim, generator=generator).to(device)
        pid = torch.full((n,), int(pert_id), dtype=torch.long, device=device)
        rho = self.decode(z, pid)["rho"]
        return torch.log1p(1e4 * rho).mean(0)
