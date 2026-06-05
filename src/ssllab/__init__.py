"""ssl-lab — self-supervised learning lab.

Studies JEPA (joint-embedding predictive architectures) and how to extend a
JEPA representation learner into a *sampleable* generative model: train an
encoder by predicting embeddings (not pixels), then bolt on a flow-matching
prior over the latent and a decoder back to data space.

Lean public API; reach into submodules for the rest.
"""

from ssllab.jepa import JEPA, JEPAConfig, build_jepa
from ssllab.utils import get_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "JEPA",
    "JEPAConfig",
    "build_jepa",
    "set_seed",
    "get_device",
]
