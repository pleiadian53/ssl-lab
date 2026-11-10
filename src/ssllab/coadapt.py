"""Co-adaptation, Phase 1: train the cell encoder and the count decoder together.

Every earlier round of the perturbation-response work trained the NB count decoder on a frozen
JEPA encoder. The ceiling analysis found the effect present in those frozen latents (a linear
readout recovers much more of it than the trained decoder does on the very same latents), so
Phase 1 asks whether letting the encoder and the decoder shape each other recovers what the
decoder loses. The transition (the conditional flow) is not part of this training at all.

Three modes share one recipe and differ only in what is allowed to train:

``dec``
    The encoder stays frozen and only a fresh decoder trains. This is the control: today's
    decoder recipe, re-seeded, under the same optimizer settings as the other two modes.
``joint``
    The encoder (initialized from its JEPA checkpoint) and a fresh decoder train together on the
    decoder's negative-binomial reconstruction loss. The encoder gets its own, smaller learning
    rate because it starts from a pretrained state.
``joint_anchor``
    As ``joint``, while maintaining the JEPA predictive objective during reconstruction
    fine-tuning: the masked-prediction loss on the same batch is added with a fixed weight. The
    EMA teacher follows the changing student exactly as in Stage A, so this keeps an *objective*
    satisfied; it does not penalize distance from the original encoder. How far the
    representation actually moves is measured separately (for example by CKA against the
    original encoder). A penalty against a frozen copy of the original teacher would be a
    different experiment.

What trains, explicitly (``load_jepa`` returns every parameter frozen; :func:`prepare_jepa`
unfreezes):

``dec``           the decoder only (learning rate ``lr``).
``joint``         the decoder, and every parameter of ``jepa.encoder`` (token projection,
                  positional embeddings, transformer blocks, final norm) at ``encoder_lr``. The
                  predictor is unused and stays frozen.
``joint_anchor``  as ``joint``, plus every parameter of ``jepa.predictor`` at ``encoder_lr``.
                  The EMA teacher is never optimized; it is updated by EMA after every step.

The teacher is rebuilt as an exact copy of the loaded encoder, which starts a new fine-tuning
run; it is not the historical Stage A teacher, which was never saved. There is no resume: the
saved encoder and decoder are exports, not training checkpoints (no optimizer, teacher, schedule
or RNG state), so an interrupted run is restarted from scratch.

Reconstruction discourages complete information loss, but it does not guarantee that the
optimizer keeps every useful direction or the perturbation information the benchmark needs. So
the trainer records, every epoch, the reconstruction loss and the effective rank and feature
spread of the cell vectors it produces; the task-relevant checks (linear readout, perturbation
probe) run after training, in the scripts that read the saved encoder.

The anchor's weight is fixed by a rule rather than tuned: it makes the two terms equal at the
scale of a *trained* readout, ``w = mean NB loss of a reference trained decoder / mean JEPA
loss``, both measured in eval mode, without gradients and before any parameter update, on the
first ``anchor_batches`` training batches (:func:`anchor_weight_by_rule`); the weight is then
fixed for the whole run. A freshly initialized decoder's loss is more than twice a trained one's,
so measuring with it would overweight the anchor for most of training. Equal loss values do not
mean equal influence on the encoder, so every epoch also records the encoder-gradient norm each
term contributes on that epoch's first batch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import torch

from ssllab.eval.collapse import collapse_report
from ssllab.generative.count_decoder import LIBRARY_SIZES, CountDecoder, library_size_of
from ssllab.jepa.ema import EMA
from ssllab.jepa.model import JEPA

MODES = ("dec", "joint", "joint_anchor")
MIN_JEPA_LOSS = 1e-6  # below this the anchor weight is undefined in practice; the run refuses to start


@dataclass
class CoadaptConfig:
    """Settings shared by every mode. Defaults are the round-10 recipe."""

    mode: str = "joint"
    epochs: int = 30
    lr: float = 1e-3            # the decoder's learning rate (today's Stage C value)
    encoder_lr: float = 1e-4    # encoder, and predictor under the anchor; smaller, since pretrained
    clip: float = 5.0           # gradient-norm clip over every trainable parameter, in every mode
    anchor_batches: int = 20    # batches used to fix the anchor weight
    anchor_weight: float | None = None  # set by the rule before training (joint_anchor only)
    library_size: str = "hvg"   # the total the NB likelihood uses (count_decoder.LIBRARY_SIZES)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.library_size not in LIBRARY_SIZES:
            raise ValueError(f"library_size must be one of {LIBRARY_SIZES}, got {self.library_size!r}")
        if self.epochs < 1 or self.lr <= 0 or self.encoder_lr <= 0 or self.clip <= 0:
            raise ValueError("epochs must be >= 1 and lr, encoder_lr and clip must be positive")


def prepare_jepa(jepa: JEPA, mode: str, device: torch.device | str) -> JEPA:
    """Set which parts of a loaded JEPA train for ``mode``, and give it a valid teacher.

    ``load_jepa`` returns every parameter frozen. Here the encoder is unfrozen for the joint
    modes and the predictor too under the anchor. The EMA teacher is rebuilt as an exact copy of
    the loaded encoder, because the teacher is not saved in the checkpoint.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    jepa.to(device)
    train_encoder = mode != "dec"
    for p in jepa.encoder.parameters():
        p.requires_grad_(train_encoder)
    for p in jepa.predictor.parameters():
        p.requires_grad_(mode == "joint_anchor")
    jepa.ema = EMA(jepa.encoder, base=jepa.cfg.ema_base, final=jepa.cfg.ema_final).to(device)
    jepa.train(train_encoder)
    return jepa


def _nb_loss(jepa: JEPA, decoder: CountDecoder, batch: dict, device, library_size: str = "all") -> torch.Tensor:
    z = jepa.embed(batch["tokens"].to(device))
    return decoder.loss(z, batch["counts"].float().to(device), library_size_of(batch, library_size).to(device))


@torch.no_grad()
def anchor_weight_by_rule(
    jepa: JEPA, reference_decoder: CountDecoder, batches: Iterable[dict], device,
) -> dict[str, float]:
    """``w = mean NB loss of the reference decoder / mean JEPA loss`` over ``batches``.

    Both are measured before any update, on the same batches, with the loaded encoder. The
    reference is an already-trained decoder, so the two loss terms are equal at the scale the
    reconstruction loss will settle at rather than at a fresh decoder's much larger starting loss.
    Returns the weight and the two means.
    """
    nb, jl, n = 0.0, 0.0, 0
    was_training = jepa.training
    jepa.eval()
    for batch in batches:
        nb += float(_nb_loss(jepa, reference_decoder, batch, device))
        jl += float(jepa(batch["tokens"].to(device))[0])
        n += 1
    jepa.train(was_training)
    if n == 0 or jl / n < MIN_JEPA_LOSS:
        raise ValueError(f"anchor weight needs a mean JEPA loss of at least {MIN_JEPA_LOSS:g}; "
                         f"got {jl / max(n, 1):.3g} over {n} batches, which would make the weight explode")
    return {"anchor_weight": nb / jl, "ref_nb_loss": nb / n, "jepa_loss": jl / n, "n_batches": n}


def _grad_norm(term: torch.Tensor, params: list[torch.Tensor]) -> float:
    """Norm of ``d term / d params`` without touching ``.grad`` (the graph is kept for backward)."""
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None)))


def train_coadapt(
    jepa: JEPA,
    decoder: CountDecoder,
    loader,
    cfg: CoadaptConfig,
    device: torch.device | str = "cpu",
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """Train for ``cfg.epochs`` and return one record per epoch.

    Call :func:`prepare_jepa` first. Each record holds the mean NB loss, the mean JEPA loss and
    the mean total under the anchor, and the effective rank and feature spread of the cell
    vectors produced during the epoch (on up to 4,096 cells, detached). In the joint modes it
    also holds ``grad_nb`` (and under the anchor ``grad_anchor``): the norm of the encoder
    gradient from each term, on the epoch's first batch, before clipping.
    """
    if cfg.mode == "joint_anchor" and cfg.anchor_weight is None:
        raise ValueError("joint_anchor needs cfg.anchor_weight; fix it with anchor_weight_by_rule first")
    groups = [{"params": list(decoder.parameters()), "lr": cfg.lr}]
    if cfg.mode != "dec":
        groups.append({"params": list(jepa.encoder.parameters()), "lr": cfg.encoder_lr})
    if cfg.mode == "joint_anchor":
        groups.append({"params": list(jepa.predictor.parameters()), "lr": cfg.encoder_lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups)
    total_steps = cfg.epochs * max(len(loader), 1)
    step = 0
    history: list[dict] = []
    for epoch in range(cfg.epochs):
        decoder.train()
        sums = {"nb_loss": 0.0, "jepa_loss": 0.0, "loss": 0.0}
        zs, n_z, n_batches = [], 0, 0
        grad_norms: dict[str, float] = {}
        for batch in loader:
            if cfg.mode == "dec":
                with torch.no_grad():
                    z = jepa.embed(batch["tokens"].to(device))
            else:
                z = jepa.embed(batch["tokens"].to(device))
            nb = decoder.loss(z, batch["counts"].float().to(device), library_size_of(batch, cfg.library_size).to(device))
            loss = nb
            if cfg.mode == "joint_anchor":
                jl = jepa(batch["tokens"].to(device))[0]
                loss = nb + cfg.anchor_weight * jl
                sums["jepa_loss"] += float(jl.detach())
            if n_batches == 0 and cfg.mode != "dec":
                enc = [p for p in jepa.encoder.parameters() if p.requires_grad]
                grad_norms["grad_nb"] = _grad_norm(nb, enc)
                if cfg.mode == "joint_anchor":
                    grad_norms["grad_anchor"] = _grad_norm(cfg.anchor_weight * jl, enc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            if cfg.mode == "joint_anchor":
                jepa.update_target(step, total_steps)
            step += 1
            n_batches += 1
            sums["nb_loss"] += float(nb.detach())
            sums["loss"] += float(loss.detach())
            if n_z < 4096:
                zs.append(z.detach().cpu())
                n_z += len(z)
        rec = {"epoch": epoch + 1, **{k: v / max(n_batches, 1) for k, v in sums.items()}, **grad_norms}
        if cfg.mode != "joint_anchor":
            rec.pop("jepa_loss")
        rec.update(collapse_report(torch.cat(zs)) if zs else {})
        history.append(rec)
        if log:
            log("epoch %d/%d  " % (epoch + 1, cfg.epochs)
                + "  ".join(f"{k}={v:.4g}" for k, v in rec.items() if k not in ("epoch", "embed_dim")))
    return history


def config_dict(cfg: CoadaptConfig) -> dict:
    return asdict(cfg)
