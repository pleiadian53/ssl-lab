"""The variant switches a walkthrough can take.

Option names match the training scripts' flags (``--flow-base``, ``--cond-type``,
``--compose``, ``--zinb``, ``--anchored-mean``, ``--state-dispersion``, ``--encoder-cond``),
so the flags of a run can be pasted in to see that run's architecture. The defaults are
the configuration behind the headline result: the transport flow (control-cell source)
with the gene-compositional, additive condition, and the plain NB decoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

_CHOICES = {
    "flow_base": ("control", "gaussian"),
    "cond_type": ("geneset", "table"),
    "compose": ("additive", "deepsets"),
    "mode": ("generate", "train"),
}


@dataclass(frozen=True)
class Options:
    pert: str | None = None          # perturbation name; default: a held-out combination
    t: float = 0.37                  # flow time for the single-step components
    flow_base: str = "control"       # control = transport from control latents; gaussian = from noise
    cond_type: str = "geneset"       # geneset = compose from target genes; table = one row per perturbation
    compose: str = "additive"        # how gene vectors combine (geneset only)
    guidance: float = 1.0            # classifier-free guidance weight; 1 = plain conditional
    zinb: bool = False
    anchored_mean: bool = False
    state_dispersion: bool = False
    encoder_cond: bool = False       # cVAE encoder q(z | x, c) instead of q(z | x)
    mode: str = "generate"           # cVAE path: generate (noise -> decoder) or train (full ELBO)
    reg_coef: float = 0.04           # VICReg weight in the JEPA step
    expand: bool = False             # open up the inside of repeated blocks

    def __post_init__(self) -> None:
        for key, allowed in _CHOICES.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, got {getattr(self, key)!r}")

    def as_dict(self) -> dict:
        return asdict(self)
