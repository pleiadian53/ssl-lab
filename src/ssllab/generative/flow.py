"""Flow-matching prior over the JEPA latent (rectified flow on vectors).

This is the "make JEPA sampleable" piece. We learn a velocity field over the
frozen JEPA latent ``z`` and sample by integrating an ODE from noise to data.
Condensed from genai-lab's ``flow_matching/`` (VelocityMLP + linear interpolant
+ CFM loss + Euler sampler), specialized to flat vector latents.

Conventions (rectified flow / linear interpolant):
    z_t = (1 - t) * z0 + t * z1        with z0 ~ N(0, I), z1 = data latent
    u_t = z1 - z0                      (the target velocity, constant in t)
    loss = || v_theta(z_t, t [, c]) - u_t ||^2
Sampling integrates dz/dt = v_theta(z, t [, c]) from t=0 (noise) to t=1 (data).

Conditioning (the Part 9 step).
    Pass ``cond_dim > 0`` to make the field *conditional*: it takes an extra
    condition vector ``c`` ``(B, cond_dim)`` and FiLM-injects it alongside time,
    so it learns a different flow per condition and samples ``p(z | c)`` instead
    of the marginal ``p(z)``. ``c`` is modality-agnostic — a class embedding here
    (MNIST proxy), the ``(z_b, z_p)`` perturbation pair later. With ``cond_dim=0``
    the module is exactly the original unconditional field (``c`` must be None).

    A learned ``null_cond`` token supports classifier-free guidance (CFG): during
    training each sample's condition is dropped to ``null_cond`` with probability
    ``p_drop``; at sampling, ``guidance`` blends the conditional and unconditional
    velocities to trade fidelity against diversity. ``guidance=1`` is plain
    conditional sampling (the null token stays inert).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map scalar times ``(B,)`` in [0, 1] to ``(B, dim)`` sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:  # pad odd dims
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class VelocityMLP(nn.Module):
    """Velocity field ``v(z_t, t [, c])`` with FiLM time (and optional condition).

    Set ``cond_dim > 0`` for a conditional field; leave it 0 for the original
    unconditional prior (backward-compatible — existing callers pass no ``c``).
    """

    def __init__(
        self,
        data_dim: int,
        hidden: int = 256,
        n_layers: int = 4,
        time_dim: int = 64,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden)
        )  # -> (scale, shift) for FiLM
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden)
            )  # -> (scale, shift) for a second FiLM
            # Learned "no condition" token for classifier-free guidance.
            self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.in_proj = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(hidden, data_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if (c is None) != (self.cond_dim == 0):
            raise ValueError(
                f"condition mismatch: cond_dim={self.cond_dim} but c is "
                f"{'None' if c is None else 'provided'}"
            )
        temb = sinusoidal_time_embedding(t, self.time_dim)
        scale, shift = self.time_mlp(temb).chunk(2, dim=-1)
        h = self.in_proj(z_t)
        h = h * (1.0 + scale) + shift  # FiLM (time)
        if c is not None:
            cscale, cshift = self.cond_mlp(c).chunk(2, dim=-1)
            h = h * (1.0 + cscale) + cshift  # FiLM (condition)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_proj(h)


def _apply_cond_dropout(model: VelocityMLP, c: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Replace a fraction ``p_drop`` of rows of ``c`` with the learned null token (CFG)."""
    if p_drop <= 0.0:
        return c
    drop = torch.rand(c.shape[0], device=c.device) < p_drop
    null = model.null_cond.to(c.dtype).expand_as(c)
    return torch.where(drop.unsqueeze(-1), null, c)


def linear_interpolant(
    z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z_t, u_t)`` for the rectified-flow linear path. ``t`` is ``(B,)``."""
    tb = t.view(-1, *([1] * (z1.dim() - 1)))
    z_t = (1.0 - tb) * z0 + tb * z1
    u_t = z1 - z0
    return z_t, u_t


def cfm_loss(
    model: VelocityMLP,
    z1: torch.Tensor,
    c: torch.Tensor | None = None,
    p_drop: float = 0.0,
    z0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conditional flow-matching MSE loss for data latents ``z1`` ``(B, D)``.

    ``c`` ``(B, cond_dim)`` is the per-sample condition (None for the
    unconditional prior). ``p_drop`` randomly nulls the condition per sample to
    train classifier-free guidance; ignored when ``c`` is None.

    ``z0`` ``(B, D)`` is the transport **source**. Default ``None`` draws Gaussian
    noise — the standard noise→data prior. Pass real samples (e.g. control-cell
    latents) to learn a **distribution-to-distribution transport** source→``z1``,
    so the field models the *displacement* between two real populations rather
    than sampling absolute state from noise.
    """
    b = z1.shape[0]
    if z0 is None:
        z0 = torch.randn_like(z1)
    t = torch.rand(b, device=z1.device)
    z_t, u_t = linear_interpolant(z0, z1, t)
    if c is not None:
        c = _apply_cond_dropout(model, c, p_drop)
    v_pred = model(z_t, t, c)
    return ((v_pred - u_t) ** 2).mean()


@torch.no_grad()
def euler_sample(
    model: VelocityMLP,
    n: int,
    dim: int,
    n_steps: int = 50,
    device: torch.device | str = "cpu",
    c: torch.Tensor | None = None,
    guidance: float = 1.0,
    z0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the ODE from the source (t=0) to data (t=1). Returns ``(n, dim)``.

    ``c`` ``(n, cond_dim)`` fixes the condition for every sample (None for the
    unconditional prior). ``guidance > 1`` applies classifier-free guidance,
    extrapolating away from the null-token (unconditional) velocity to sharpen
    the condition; ``guidance=1`` is plain conditional sampling.

    ``z0`` ``(n, dim)`` is the source state. Default ``None`` starts from Gaussian
    noise (the noise→data prior); pass real samples (control latents) to transport
    a baseline population toward its conditioned outcome.
    """
    model.eval()
    z = torch.randn(n, dim, device=device) if z0 is None else z0.to(device)
    dt = 1.0 / n_steps
    null = None
    if c is not None and guidance != 1.0:
        null = model.null_cond.to(z.dtype).to(device).expand(n, -1)
    for step in range(n_steps):
        t = torch.full((n,), step * dt, device=device)
        if null is not None:
            v = model(z, t, null) + guidance * (model(z, t, c) - model(z, t, null))
        else:
            v = model(z, t, c)
        z = z + dt * v
    return z
