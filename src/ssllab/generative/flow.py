"""Flow-matching prior over the JEPA latent (rectified flow on vectors).

This is the "make JEPA sampleable" piece. We learn a velocity field over the
frozen JEPA latent ``z`` and sample by integrating an ODE from noise to data.
Condensed from genai-lab's ``flow_matching/`` (VelocityMLP + linear interpolant
+ CFM loss + Euler sampler), specialized to flat vector latents.

Conventions (rectified flow / linear interpolant):
    z_t = (1 - t) * z0 + t * z1        with z0 ~ N(0, I), z1 = data latent
    u_t = z1 - z0                      (the target velocity, constant in t)
    loss = || v_theta(z_t, t) - u_t ||^2
Sampling integrates dz/dt = v_theta(z, t) from t=0 (noise) to t=1 (data).
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
    """Velocity field ``v(z_t, t)`` with FiLM time conditioning."""

    def __init__(self, data_dim: int, hidden: int = 256, n_layers: int = 4, time_dim: int = 64) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden)
        )  # -> (scale, shift) for FiLM
        self.in_proj = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(hidden, data_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = sinusoidal_time_embedding(t, self.time_dim)
        scale, shift = self.time_mlp(temb).chunk(2, dim=-1)
        h = self.in_proj(z_t)
        h = h * (1.0 + scale) + shift  # FiLM
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_proj(h)


def linear_interpolant(
    z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z_t, u_t)`` for the rectified-flow linear path. ``t`` is ``(B,)``."""
    tb = t.view(-1, *([1] * (z1.dim() - 1)))
    z_t = (1.0 - tb) * z0 + tb * z1
    u_t = z1 - z0
    return z_t, u_t


def cfm_loss(model: VelocityMLP, z1: torch.Tensor) -> torch.Tensor:
    """Conditional flow-matching MSE loss for data latents ``z1`` ``(B, D)``."""
    b = z1.shape[0]
    z0 = torch.randn_like(z1)
    t = torch.rand(b, device=z1.device)
    z_t, u_t = linear_interpolant(z0, z1, t)
    v_pred = model(z_t, t)
    return ((v_pred - u_t) ** 2).mean()


@torch.no_grad()
def euler_sample(
    model: VelocityMLP,
    n: int,
    dim: int,
    n_steps: int = 50,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Integrate the ODE from noise (t=0) to data (t=1). Returns ``(n, dim)``."""
    model.eval()
    z = torch.randn(n, dim, device=device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n,), step * dt, device=device)
        z = z + dt * model(z, t)
    return z
