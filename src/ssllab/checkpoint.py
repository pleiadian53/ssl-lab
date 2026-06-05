"""Checkpoint helpers for the JEPA encoder.

Saves the JEPA config alongside weights so downstream scripts (probe, decoder,
prior) can rebuild the exact architecture and load the frozen encoder.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from ssllab.jepa.model import JEPA, JEPAConfig


def save_jepa(jepa: JEPA, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": dataclasses.asdict(jepa.cfg), "state": jepa.state_dict()}, str(path))
    return path


def load_jepa(path: str | Path, device: torch.device | str = "cpu") -> JEPA:
    """Rebuild a JEPA from a checkpoint and load weights (eval mode, frozen)."""
    ck = torch.load(str(path), map_location=device)
    jepa = JEPA(JEPAConfig(**ck["cfg"]))
    jepa.load_state_dict(ck["state"])
    jepa.to(device).eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa
