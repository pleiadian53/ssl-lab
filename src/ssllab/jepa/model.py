"""The JEPA module: assembles student encoder + EMA target + predictor.

Forward pass (one training step on a token batch):

    1. sample a (context, target) position split
    2. student encodes the *context* tokens                -> ctx_embed
    3. predictor predicts embeddings at *target* positions -> pred
    4. EMA teacher encodes the *full* token set, select the
       target positions (detached)                         -> target
    5. loss = prediction_loss(pred, target) [+ VICReg or SIGReg regularizer]

``embed`` gives the pooled image-level latent ``z`` (no masking) used
downstream by the decoder, the flow prior, and the linear probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ssllab.jepa.ema import EMA
from ssllab.jepa.masking import batched, sample_block_masks, sample_span_masks
from ssllab.models.encoder import JEPAEncoder
from ssllab.models.predictor import JEPAPredictor
from ssllab.objectives.jepa_loss import jepa_loss


@dataclass
class JEPAConfig:
    token_dim: int = 49
    n_tokens: int = 16
    embed_dim: int = 128
    enc_depth: int = 4
    pred_depth: int = 2
    n_heads: int = 4
    n_target: int = 4
    pred_kind: str = "smooth_l1"
    reg_coef: float = 0.0
    var_coef: float = 1.0
    cov_coef: float = 0.04
    ema_base: float = 0.996
    ema_final: float = 1.0
    # Masking. "random": n_target scattered positions (the MNIST/perturb-seq default).
    # "span": n_spans contiguous spans of span_len tokens (ordered 1D sequences);
    # n_target must equal span_len * n_spans. Defaults keep old checkpoints loadable.
    mask_kind: str = "random"
    span_len: int = 0
    n_spans: int = 1
    # Anti-collapse regularizer. "vicreg" (the default, and what every checkpoint saved
    # before this field existed used) or "sigreg" (LeJEPA). reg_target says what it acts on:
    # "tokens", the student's context-token embeddings (where VICReg has always acted, over
    # all B*C rows at once; SIGReg tests each token position's B rows as one view and
    # averages, so its scale matches the pooled target), or "pooled", their mean over the
    # context tokens, one cell vector per cell (B rows; the student's view of the
    # representation downstream stages read).
    reg_kind: str = "vicreg"
    reg_target: str = "tokens"
    sigreg_slices: int = 1024
    sigreg_points: int = 17
    sigreg_t_max: float = 5.0

    def __post_init__(self) -> None:
        if self.reg_kind not in ("vicreg", "sigreg"):
            raise ValueError(f"reg_kind must be 'vicreg' or 'sigreg', got {self.reg_kind!r}")
        if self.reg_target not in ("tokens", "pooled"):
            raise ValueError(f"reg_target must be 'tokens' or 'pooled', got {self.reg_target!r}")
        if self.mask_kind not in ("random", "span"):
            raise ValueError(f"mask_kind must be 'random' or 'span', got {self.mask_kind!r}")
        if self.mask_kind == "span" and self.span_len * self.n_spans != self.n_target:
            raise ValueError(
                f"span masking needs n_target == span_len * n_spans, got {self.n_target} != {self.span_len} * {self.n_spans}"
            )


class JEPA(nn.Module):
    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = JEPAEncoder(
            token_dim=cfg.token_dim,
            embed_dim=cfg.embed_dim,
            depth=cfg.enc_depth,
            n_heads=cfg.n_heads,
            n_tokens=cfg.n_tokens,
        )
        self.predictor = JEPAPredictor(
            embed_dim=cfg.embed_dim,
            depth=cfg.pred_depth,
            n_heads=cfg.n_heads,
            n_tokens=cfg.n_tokens,
        )
        self.ema = EMA(self.encoder, base=cfg.ema_base, final=cfg.ema_final)
        # SIGReg's random directions come from their own generator, so the default VICReg
        # path draws exactly the same random numbers as before SIGReg existed. Not saved in
        # the state dict (old checkpoints load unchanged); created on first use, on the
        # embeddings' device, seeded 0, and advanced every step.
        self._sigreg_gen: torch.Generator | None = None

    def _directions_generator(self, device: torch.device) -> torch.Generator:
        if self._sigreg_gen is None or self._sigreg_gen.device != device:
            self._sigreg_gen = torch.Generator(device=device).manual_seed(0)
        return self._sigreg_gen

    # -- training ---------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """One JEPA step on ``tokens`` ``(B, N, token_dim)`` -> ``(loss, components)``."""
        b = tokens.shape[0]
        cfg = self.cfg
        if cfg.mask_kind == "span":
            ctx_1d, tgt_1d = sample_span_masks(cfg.n_tokens, cfg.span_len, cfg.n_spans, device=tokens.device)
        else:
            ctx_1d, tgt_1d = sample_block_masks(cfg.n_tokens, cfg.n_target, device=tokens.device)
        ctx_idx = batched(ctx_1d, b)
        tgt_idx = batched(tgt_1d, b)

        # Student: encode context, predict target positions.
        ctx_embed = self.encoder(tokens, idx=ctx_idx)
        pred = self.predictor(ctx_embed, ctx_idx, tgt_idx)

        # Teacher: encode full set (no grad), select target positions.
        with torch.no_grad():
            full = self.ema.teacher(tokens, idx=None)            # (B, N, D)
            gather = tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
            target = torch.gather(full, 1, gather)               # (B, T, D)

        z_reg = None
        if cfg.reg_coef > 0:
            if cfg.reg_target == "pooled":
                z_reg = ctx_embed.mean(dim=1)                           # (B, D)
            elif cfg.reg_kind == "sigreg":
                z_reg = ctx_embed.transpose(0, 1)                       # (C, B, D): one view per position
            else:
                z_reg = ctx_embed                                       # VICReg: all rows at once, as always
        return jepa_loss(
            pred,
            target,
            z_ctx=z_reg,
            pred_kind=cfg.pred_kind,
            reg_coef=cfg.reg_coef,
            var_coef=cfg.var_coef,
            cov_coef=cfg.cov_coef,
            reg_kind=cfg.reg_kind,
            sigreg_slices=cfg.sigreg_slices,
            sigreg_points=cfg.sigreg_points,
            sigreg_t_max=cfg.sigreg_t_max,
            generator=self._directions_generator(tokens.device) if cfg.reg_kind == "sigreg" else None,
        )

    @torch.no_grad()
    def update_target(self, step: int, total: int) -> float:
        """Advance the EMA teacher. Call once per optimizer step."""
        return self.ema.update(self.encoder, step, total)

    # -- inference --------------------------------------------------------
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pooled image-level latent ``z`` ``(B, embed_dim)`` from the student encoder."""
        return self.encoder.embed_pooled(tokens)


def build_jepa(**kwargs) -> JEPA:
    """Factory mirroring genai-lab's ``build_*`` helpers.

    Accepts any :class:`JEPAConfig` field as a keyword.
    """
    return JEPA(JEPAConfig(**kwargs))
