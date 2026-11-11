"""Checkpoint helpers for the JEPA encoder.

Saves the JEPA config alongside weights so downstream scripts (probe, decoder,
prior) can rebuild the exact architecture and load the frozen encoder.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from ssllab.jepa.ema import EMA
from ssllab.jepa.model import JEPA, JEPAConfig


def save_jepa(jepa: JEPA, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": dataclasses.asdict(jepa.cfg), "state": jepa.state_dict()}, str(path))
    return path


def load_jepa(path: str | Path, device: torch.device | str = "cpu") -> JEPA:
    """Rebuild a JEPA from a checkpoint and load weights (eval mode, frozen).

    The EMA teacher is not part of the state dict (``EMA`` is a plain object holding a copy of
    the encoder), so it is rebuilt here as an exact copy of the loaded encoder. Without this the
    teacher would keep the random weights it was given when the JEPA was constructed, which is
    harmless while the encoder is only used frozen but wrong for anything that resumes the
    masked-prediction loss.
    """
    ck = torch.load(str(path), map_location=device)
    jepa = JEPA(JEPAConfig(**ck["cfg"]))
    jepa.load_state_dict(ck["state"])
    jepa.ema = EMA(jepa.encoder, base=jepa.cfg.ema_base, final=jepa.cfg.ema_final)
    jepa.to(device).eval()
    jepa.ema.to(device)
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa


def random_jepa_like(path: str | Path, device: torch.device | str = "cpu") -> JEPA:
    """A JEPA with the architecture of the checkpoint at ``path`` and freshly initialized weights.

    Only the checkpoint's config is read; its weights are not. The model is built by the same
    constructor ``load_jepa`` calls, so it draws from the global random-number generator exactly
    as ``load_jepa`` does before loading weights. After the same ``set_seed``, everything drawn
    afterwards (a decoder's initialization, shuffling) is therefore identical between a pretrained
    start and a random one, and only the encoder's starting weights differ. The weights are the
    constructor's own initialization, the one JEPA pretraining itself starts from. Returned like
    ``load_jepa``: eval mode, every parameter frozen, a teacher that copies the encoder.
    """
    ck = torch.load(str(path), map_location="cpu")
    jepa = JEPA(JEPAConfig(**ck["cfg"]))
    jepa.ema = EMA(jepa.encoder, base=jepa.cfg.ema_base, final=jepa.cfg.ema_final)
    jepa.to(device).eval()
    jepa.ema.to(device)
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa
