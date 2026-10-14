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

Where the condition enters (``encoder_cond``).
    By default the encoder is **condition-blind**, ``q(z | x)``: it sees the cell's
    expression only, and ``c`` reaches the model through the decoder alone. This follows
    CPA's basal-state reading, in which ``z`` is the cell's state apart from the
    perturbation, but without CPA's adversarial classifier that strips perturbation
    information out of ``z``. Because the observed ``x`` already contains the perturbation's
    effect, part of that effect can leak into ``z``; at generation ``z`` comes from the prior
    and carries none of it, so a leak would attenuate the predicted effect. The KL term
    discourages the leak (``c`` is free to the decoder while information in ``z`` costs KL),
    but nothing forbids it.

    ``encoder_cond=True`` gives the textbook conditional VAE, ``q(z | x, c)``: the encoder
    also reads the condition embedding, so it can account for the perturbation and put only
    the residual cell-to-cell variation into ``z``. The ELBO is valid either way; leaving
    ``c`` out only restricts the variational family, so the bound can be looser. Generation
    is identical in both modes (``z ~ N(0, I)`` plus ``c`` into the decoder), so the flag
    changes what training teaches the decoder, never the sampling path.

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
        encoder_cond: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_cond = encoder_cond
        self.cond = GeneSetEmbedding(pert_gene, gene_dim=gene_dim, out_dim=cond_dim, compose=compose)
        enc_in = n_genes + (cond_dim if encoder_cond else 0)
        self.enc = nn.Sequential(
            nn.Linear(enc_in, hidden), nn.SiLU(),
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

    @classmethod
    def from_checkpoint(cls, ck: dict, device: torch.device | str = "cpu") -> "ConditionalNBVAE":
        """Rebuild a trained baseline from the dict ``08_train_cvae_baseline.py`` saves, in eval mode.

        Checkpoints written before ``encoder_cond`` existed load as the condition-blind encoder,
        which is what they were trained with.
        """
        model = cls(
            n_genes=ck["n_genes"], pert_gene=ck["pert_gene"], latent_dim=ck["latent_dim"],
            cond_dim=ck["cond_dim"], gene_dim=ck["gene_dim"], hidden=ck["hidden"],
            compose=ck["compose"], encoder_cond=ck.get("encoder_cond", False),
        ).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        return model

    def encode(
        self, x: torch.Tensor, pert_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_cond:
            if pert_id is None:
                raise ValueError("encoder_cond=True: encode() needs pert_id to embed the condition")
            x = torch.cat([x, self.cond(pert_id)], dim=-1)
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
        mu_z, logvar = self.encode(x, pert_id)
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

    @torch.no_grad()
    def generate_population(
        self, pert_id: int, n: int, library_size: torch.Tensor,
        device: torch.device | str = "cpu", generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Per-cell predicted log1p-CP10K expression ``(n, G)``, **sampling counts** from the NB
        so the population carries realistic technical noise (for calibration)."""
        z = torch.randn(n, self.latent_dim, generator=generator).to(device)
        pid = torch.full((n,), int(pert_id), dtype=torch.long, device=device)
        out = self.decode(z, pid)
        ls = library_size.to(device)
        mu = ls.unsqueeze(-1) * out["rho"]
        kappa = out["kappa"].expand_as(mu)
        lam = torch.distributions.Gamma(kappa, kappa / (mu + 1e-8)).sample()   # NB = Gamma-Poisson
        counts = torch.poisson(lam)
        return torch.log1p(1e4 * counts / ls.unsqueeze(-1))
