"""The JEPA module: assembles student encoder + EMA target + predictor.

Forward pass (one training step on a token batch):

    1. sample a (context, target) position split
    2. student encodes the *context* tokens                -> ctx_embed
    3. predictor predicts embeddings at *target* positions -> pred
    4. EMA teacher encodes the *full* token set, select the
       target positions (detached)                         -> target
    5. loss = prediction_loss(pred, target) [+ VICReg reg]

``embed`` gives the pooled image-level latent ``z`` (no masking) used
downstream by the decoder, the flow prior, and the linear probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ssllab.jepa.ema import EMA
from ssllab.jepa.masking import batched, sample_block_masks
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

    # -- training ---------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """One JEPA step on ``tokens`` ``(B, N, token_dim)`` -> ``(loss, components)``."""
        b = tokens.shape[0]
        cfg = self.cfg
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

        return jepa_loss(
            pred,
            target,
            z_ctx=ctx_embed if cfg.reg_coef > 0 else None,
            pred_kind=cfg.pred_kind,
            reg_coef=cfg.reg_coef,
            var_coef=cfg.var_coef,
            cov_coef=cfg.cov_coef,
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
