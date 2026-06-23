"""Count decoder: map a cell latent ``z`` to a distribution over gene counts.

The G2 piece for single-cell data, and the part where **effect size** is
recovered ([docs/generative_jepa/06](../../../docs/generative_jepa/06-route-a-latent-decoder-head.md)).
Single-cell RNA-seq counts are non-negative integers, heavily over-dispersed, and
dropout-dominated, so a Gaussian/MSE decoder is the wrong measurement model. The
decoder instead emits the *parameters of a count distribution* — a negative
binomial (NB), optionally zero-inflated (ZINB) — and is trained by the count
likelihood of the real counts.

Convention (matches the docs):
    rho   = softmax(net(z))           a relative gene-rate profile (sums to 1)
    mu    = library_size * rho        the NB mean (library size is a *given* covariate)
    kappa = softplus(per-gene param)  the NB dispersion (variance = mu + mu^2 / kappa)
    x | z, ell ~ NB(mu, kappa)        the count of each gene

The losses below are the (stable form of the) negative log-likelihood the docs
write out factor by factor; minimizing them *is* maximizing the count likelihood.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def nb_nll(x: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative-binomial negative log-likelihood, averaged over cells.

    ``x`` ``(B, G)`` integer counts, ``mu`` ``(B, G)`` mean, ``kappa`` dispersion
    ``(G,)`` or ``(B, G)`` (broadcast). Numerically stable scVI-style form; equals
    the per-gene NLL ``-[logΓ(x+κ) - logΓ(κ) - logΓ(x+1) + κ log(κ/(κ+μ)) + x log(μ/(κ+μ))]``.
    """
    log_k_mu = torch.log(kappa + mu + eps)
    ll = (
        kappa * (torch.log(kappa + eps) - log_k_mu)
        + x * (torch.log(mu + eps) - log_k_mu)
        + torch.lgamma(x + kappa)
        - torch.lgamma(kappa)
        - torch.lgamma(x + 1.0)
    )
    return -ll.sum(-1).mean()


def zinb_nll(
    x: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, pi_logits: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Zero-inflated NB NLL. ``pi_logits`` ``(B, G)`` are the dropout-gate logits
    (probability of a structural zero). Each count is a mixture: a structural zero
    with prob sigmoid(pi_logits), else an NB draw."""
    softplus_pi = F.softplus(-pi_logits)
    log_k_mu = torch.log(kappa + mu + eps)
    nb_ll = (
        kappa * (torch.log(kappa + eps) - log_k_mu)
        + x * (torch.log(mu + eps) - log_k_mu)
        + torch.lgamma(x + kappa)
        - torch.lgamma(kappa)
        - torch.lgamma(x + 1.0)
    )
    case_nonzero = nb_ll - softplus_pi
    # A zero can come from the dropout gate OR the NB producing a zero.
    case_zero = F.softplus(pi_logits + kappa * (torch.log(kappa + eps) - log_k_mu)) - softplus_pi
    ll = torch.where(x < eps, case_zero, case_nonzero)
    return -ll.sum(-1).mean()


class CountDecoder(nn.Module):
    """Latent ``z`` -> count-distribution parameters (rho, kappa[, pi]).

    The mean is assembled as ``mu = library_size * rho`` (library size enters as a
    given covariate, not a prediction). Dispersion ``kappa`` is a learned per-gene
    parameter (the common scVI choice). With ``zinb=True`` it also emits a per-gene
    dropout-gate logit.
    """

    def __init__(
        self,
        latent_dim: int,
        n_genes: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        zinb: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.zinb = zinb
        dims = [latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.rate_head = nn.Linear(dims[-1], n_genes)         # -> rho via softmax
        self.log_kappa = nn.Parameter(torch.zeros(n_genes))   # per-gene dispersion
        self.dropout_head = nn.Linear(dims[-1], n_genes) if zinb else None

    def forward(self, z: torch.Tensor, library_size: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        rho = F.softmax(self.rate_head(h), dim=-1)            # (B, G), sums to 1
        mu = library_size.unsqueeze(-1) * rho                 # (B, G)
        kappa = F.softplus(self.log_kappa) + 1e-4             # (G,), positive
        out = {"rho": rho, "mu": mu, "kappa": kappa}
        if self.dropout_head is not None:
            out["pi_logits"] = self.dropout_head(h)
        return out

    def nll(self, z: torch.Tensor, counts: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Training loss: NLL of the real ``counts`` under the decoded distribution."""
        out = self.forward(z, library_size)
        if self.zinb:
            return zinb_nll(counts, out["mu"], out["kappa"], out["pi_logits"])
        return nb_nll(counts, out["mu"], out["kappa"])

    @torch.no_grad()
    def sample_counts(self, z: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Draw integer counts from the decoded NB/ZINB (for generated cells)."""
        out = self.forward(z, library_size)
        mu, kappa = out["mu"], out["kappa"].expand_as(out["mu"])
        # NB as a Gamma-Poisson mixture: lambda ~ Gamma(kappa, mu/kappa), x ~ Poisson(lambda).
        lam = torch.distributions.Gamma(kappa, kappa / (mu + 1e-8)).sample()
        x = torch.poisson(lam)
        if self.zinb:
            drop = torch.rand_like(x) < torch.sigmoid(out["pi_logits"])
            x = torch.where(drop, torch.zeros_like(x), x)
        return x

    @torch.no_grad()
    def expected_counts(self, z: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """The decoded mean counts ``mu`` (ZINB: scaled by the non-dropout prob)."""
        out = self.forward(z, library_size)
        mu = out["mu"]
        if self.zinb:
            mu = mu * (1.0 - torch.sigmoid(out["pi_logits"]))
        return mu
