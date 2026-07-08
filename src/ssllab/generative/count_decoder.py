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

Two optional levers (off by default; see the conditional-flow-jepa chapter 8):
    anchored_mean     rho = softmax(log rho_base + delta(z)); the head models the
                      *deviation* from a learned baseline profile, so it starts at
                      "no effect" (the readout twin of near-identity operator init).
    state_dispersion  kappa = softplus(kappa_head(z)) varies per cell/condition
                      instead of one constant per gene, fixing the coverage-at-1.00
                      symptom. Pair with the moment-of-moments anchor in ``loss`` so
                      a flexible kappa models measurement noise rather than absorbing
                      the biological variance the latent distribution should own.

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
    given covariate, not a prediction). By default dispersion ``kappa`` is a learned
    per-gene parameter (the common scVI choice) and ``rho`` is a bare softmax. The
    two flags below opt into the chapter-8 levers.

    Args:
        anchored_mean: model ``rho`` as a deviation from a learned baseline profile,
            ``rho = softmax(log_rho_base + delta(z))`` with ``delta`` zero-initialized,
            so at init ``rho`` equals the baseline. Set the baseline to the control
            profile with :meth:`set_baseline_profile`.
        state_dispersion: make ``kappa`` a function of ``z`` (per cell/condition)
            instead of one constant per gene.
    """

    def __init__(
        self,
        latent_dim: int,
        n_genes: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        zinb: bool = False,
        dropout: float = 0.0,
        anchored_mean: bool = False,
        state_dispersion: bool = False,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.zinb = zinb
        self.anchored_mean = anchored_mean
        self.state_dispersion = state_dispersion
        dims = [latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        # Mean head: bare softmax, or an identity-anchored deviation on a baseline.
        self.rate_head = nn.Linear(dims[-1], n_genes)             # -> rho (softmax of its output)
        if anchored_mean:
            nn.init.zeros_(self.rate_head.weight)                 # delta = 0 at init -> rho = baseline
            nn.init.zeros_(self.rate_head.bias)
            self.log_rho_base = nn.Parameter(torch.zeros(n_genes))
        else:
            self.log_rho_base = None

        # Dispersion: one constant per gene, or a per-cell head.
        if state_dispersion:
            self.kappa_head = nn.Linear(dims[-1], n_genes)
            nn.init.zeros_(self.kappa_head.weight)                # kappa = softplus(0) at init (matches constant)
            nn.init.zeros_(self.kappa_head.bias)
            self.log_kappa = None
        else:
            self.kappa_head = None
            self.log_kappa = nn.Parameter(torch.zeros(n_genes))   # per-gene dispersion

        self.dropout_head = nn.Linear(dims[-1], n_genes) if zinb else None

    @torch.no_grad()
    def set_baseline_profile(self, control_rate: torch.Tensor, eps: float = 1e-6) -> None:
        """Initialize the anchored mean head's baseline to a control rate profile.

        ``control_rate`` ``(G,)`` is a per-gene rate (the control-population mean of
        ``counts / library_size``); it need not sum to 1, it is renormalized here.
        Afterwards, at ``delta = 0`` the decoder emits ``rho == control_rate``.
        """
        if self.log_rho_base is None:
            raise RuntimeError("set_baseline_profile requires anchored_mean=True")
        p = control_rate.detach().to(self.log_rho_base.device).clamp_min(eps)
        p = p / p.sum()
        self.log_rho_base.copy_(torch.log(p))

    @staticmethod
    def moment_dispersion(
        mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-4, kappa_max: float = 1e4
    ) -> torch.Tensor:
        """Method-of-moments NB dispersion per gene from observed count mean & variance.

        NB variance is ``mu + mu^2 / kappa``, so ``kappa = mu^2 / (var - mu)``, defined
        only where ``var > mu`` (genuine over-dispersion); elsewhere it clamps toward
        ``kappa_max`` (near-Poisson). ``mean``/``var`` are ``(G,)`` over some reference
        population. Returns a ``(G,)`` dispersion target for the anchor in :meth:`loss`.
        """
        excess = (var - mean).clamp_min(eps)
        kappa = mean.clamp_min(eps) ** 2 / excess
        return kappa.clamp(min=eps, max=kappa_max)

    def forward(self, z: torch.Tensor, library_size: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        logits = self.rate_head(h)
        if self.log_rho_base is not None:
            logits = logits + self.log_rho_base                  # deviation on a learned baseline
        rho = F.softmax(logits, dim=-1)                          # (B, G), sums to 1
        mu = library_size.unsqueeze(-1) * rho                    # (B, G)
        if self.kappa_head is not None:
            kappa = F.softplus(self.kappa_head(h)) + 1e-4        # (B, G), per cell
        else:
            kappa = F.softplus(self.log_kappa) + 1e-4            # (G,), per gene
        out = {"rho": rho, "mu": mu, "kappa": kappa}
        if self.dropout_head is not None:
            out["pi_logits"] = self.dropout_head(h)
        return out

    def _nll_from_out(self, out: dict[str, torch.Tensor], counts: torch.Tensor) -> torch.Tensor:
        if self.zinb:
            return zinb_nll(counts, out["mu"], out["kappa"], out["pi_logits"])
        return nb_nll(counts, out["mu"], out["kappa"])

    def nll(self, z: torch.Tensor, counts: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Training loss: NLL of the real ``counts`` under the decoded distribution."""
        return self._nll_from_out(self.forward(z, library_size), counts)

    def loss(
        self,
        z: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor,
        kappa_target: torch.Tensor | None = None,
        anchor_weight: float = 0.0,
    ) -> torch.Tensor:
        """NLL plus an optional dispersion anchor (one forward pass).

        When ``kappa_target`` ``(G,)`` and ``anchor_weight > 0`` are given, add
        ``anchor_weight * mean_g (log kappa_g - log kappa_target_g)^2`` where
        ``kappa_g`` is the batch-averaged per-gene dispersion. The anchor pulls a
        flexible ``state_dispersion`` head toward a method-of-moments technical scale
        so it does not silently absorb biological variance (chapter 8's identifiability
        guard). With the constant per-gene dispersion the anchor is a mild prior.
        """
        out = self.forward(z, library_size)
        total = self._nll_from_out(out, counts)
        if kappa_target is not None and anchor_weight > 0:
            kappa = out["kappa"]
            kappa_g = kappa.mean(0) if kappa.dim() == 2 else kappa
            kappa_target = kappa_target.to(kappa_g.device)
            anchor = (torch.log(kappa_g + 1e-8) - torch.log(kappa_target + 1e-8)).pow(2).mean()
            total = total + anchor_weight * anchor
        return total

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
