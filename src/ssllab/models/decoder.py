"""Latent decoder: map a pooled latent ``z`` back to an image.

Trained on the *frozen* JEPA encoder's embeddings (two-stage recipe). Because
JEPA latents are not trained to be decodable, reconstructions/samples may be
soft — that is an expected, instructive property of this route, not a bug.

Returns a dict (mirroring genai-lab's decoder convention) with ``logits`` and a
``mean`` (sigmoid) for Bernoulli-style MNIST pixels in ``[0, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 512),
        out_pixels: int = 28 * 28,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], out_pixels)]
        self.net = nn.Sequential(*layers)
        self.out_pixels = out_pixels

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(z)
        return {"logits": logits, "mean": torch.sigmoid(logits)}

    def decode_images(self, z: torch.Tensor, img: int = 28) -> torch.Tensor:
        """Convenience: ``(B, latent_dim)`` -> ``(B, 1, img, img)`` in ``[0, 1]``."""
        mean = self.forward(z)["mean"]
        return mean.reshape(-1, 1, img, img)
