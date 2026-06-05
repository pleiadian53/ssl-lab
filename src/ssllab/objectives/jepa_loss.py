"""JEPA training objective.

Two pieces:

1. **Prediction loss** — match predicted target embeddings to the EMA target
   encoder's embeddings, in latent space, with the target *detached*
   (stop-gradient). This is the core "predict embeddings, not pixels" signal.
2. **VICReg-style regularizer** (optional) — a hinge variance term plus an
   off-diagonal covariance term on the student embeddings. JEPA's EMA target
   already discourages collapse; this is cheap insurance and gives an explicit
   knob. Style mirrors genai-lab ``objectives/regularizers.py``.

Loss functions are plain functions returning ``(scalar, dict_of_components)``
(the genai-lab convention) so callers can log components.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "smooth_l1") -> torch.Tensor:
    """Latent-space prediction loss. ``target`` is detached (stop-gradient)."""
    target = target.detach()
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "cosine":
        # 1 - cosine similarity, averaged over tokens/batch.
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    raise ValueError(f"unknown prediction loss kind: {kind!r}")


def vicreg_reg(
    z: torch.Tensor,
    var_coef: float = 1.0,
    cov_coef: float = 0.04,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VICReg variance + covariance regularization on embeddings.

    Parameters
    ----------
    z:
        ``(M, D)`` embeddings (flatten batch/token dims before calling).

    Returns
    -------
    ``(loss, components)`` where ``loss = var_coef*var + cov_coef*cov``.
    """
    if z.dim() != 2:
        z = z.reshape(-1, z.shape[-1])
    m, d = z.shape

    # Variance: hinge each dimension's std up to 1.
    std = torch.sqrt(z.var(dim=0) + eps)
    var_term = F.relu(1.0 - std).mean()

    # Covariance: penalize off-diagonal entries of the covariance matrix.
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(m - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = off_diag.pow(2).sum() / d

    loss = var_coef * var_term + cov_coef * cov_term
    return loss, {"var": float(var_term.detach()), "cov": float(cov_term.detach())}


def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    z_ctx: torch.Tensor | None = None,
    pred_kind: str = "smooth_l1",
    reg_coef: float = 0.0,
    var_coef: float = 1.0,
    cov_coef: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine prediction loss with optional VICReg regularization.

    Parameters
    ----------
    pred, target:
        ``(B, T, D)`` predicted and EMA-target embeddings.
    z_ctx:
        Student context embeddings ``(B, C, D)`` used for the regularizer. Only
        needed when ``reg_coef > 0``.
    reg_coef:
        Weight on the VICReg term (0 disables it).
    """
    pl = prediction_loss(pred, target, kind=pred_kind)
    components = {"pred": float(pl.detach())}
    total = pl
    if reg_coef > 0.0 and z_ctx is not None:
        reg, reg_comp = vicreg_reg(z_ctx, var_coef=var_coef, cov_coef=cov_coef)
        total = total + reg_coef * reg
        components.update(reg_comp)
    else:
        components.update({"var": 0.0, "cov": 0.0})
    components["loss"] = float(total.detach())
    return total, components
