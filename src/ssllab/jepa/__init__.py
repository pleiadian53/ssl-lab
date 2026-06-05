"""JEPA core: masking, EMA target, and the assembled model."""

from ssllab.jepa.ema import EMA
from ssllab.jepa.masking import sample_block_masks
from ssllab.jepa.model import JEPA, JEPAConfig, build_jepa

__all__ = ["JEPA", "JEPAConfig", "build_jepa", "EMA", "sample_block_masks"]
