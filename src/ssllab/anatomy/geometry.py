"""Sizes used to build the components, defaulting to the production Norman configuration.

Every default is the value the training scripts use on the full dataset (``01`` for the
JEPA encoder, ``03`` for the count decoder, ``04`` for the flow, ``08`` for the cVAE), so a
walkthrough at the defaults shows the shapes of the models behind the reported results.
Any field can be overridden to explore an alternative, and :meth:`Geometry.small` gives a
tiny configuration for fast tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from ssllab.data.perturbseq import token_dim_for


@dataclass(frozen=True)
class Geometry:
    # Data.
    n_hvg: int = 5000                # genes per cell (highly variable genes)
    n_tokens: int = 50               # gene-group tokens per cell
    n_ctrl: int = 11_720             # control cells in the flow's source pool (combo split)
    # JEPA encoder and predictor (01_pretrain_stage_a.py).
    embed_dim: int = 256             # width of every token embedding and of the cell latent z
    enc_depth: int = 6
    pred_depth: int = 2
    n_heads: int = 4
    n_target: int = 12               # masked tokens the predictor must fill in
    # Conditional flow (04_train_cond_flow.py).
    flow_hidden: int = 256
    flow_layers: int = 4
    time_dim: int = 64
    cond_dim: int = 128
    pert_dim: int = 64
    gene_dim: int = 64
    steps: int = 100                 # Euler steps at sampling
    # Count decoder (03_train_count_decoder.py).
    dec_hidden: tuple[int, ...] = (512, 512)
    # Conditional NB-VAE baseline (08_train_cvae_baseline.py).
    vae_latent: int = 256
    vae_hidden: int = 512
    # Batch sizes.
    n: int = 200                     # generated cells per perturbation (06, 09, 10)
    batch: int = 256                 # training batch

    @property
    def token_dim(self) -> int:
        """Genes per token: ``ceil(n_hvg / n_tokens)``, as the tokenizer computes it."""
        return token_dim_for(self.n_hvg, self.n_tokens)

    @classmethod
    def small(cls) -> "Geometry":
        """A tiny configuration with the same structure, for fast tests."""
        return cls(n_hvg=40, n_tokens=8, n_ctrl=30, embed_dim=32, enc_depth=2, pred_depth=1,
                   n_heads=4, n_target=3, flow_hidden=32, flow_layers=2, time_dim=16, cond_dim=16,
                   pert_dim=8, gene_dim=8, steps=3, dec_hidden=(32, 32), vae_latent=16,
                   vae_hidden=32, n=6, batch=5)

    def with_overrides(self, overrides: dict[str, Any]) -> "Geometry":
        """A copy with fields replaced; values arrive as strings from the command line."""
        types = {f.name: f.type for f in fields(self)}
        parsed = {}
        for key, value in overrides.items():
            if key not in types:
                raise KeyError(f"unknown geometry field {key!r}; fields: {', '.join(types)}")
            if isinstance(value, str):
                if key == "dec_hidden":
                    value = tuple(int(v) for v in value.split(","))
                else:
                    value = int(value)
            parsed[key] = value
        return replace(self, **parsed)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
