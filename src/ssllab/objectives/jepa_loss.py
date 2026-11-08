"""JEPA training objective.

Two pieces:

1. **Prediction loss** — match predicted target embeddings to the EMA target
   encoder's embeddings, in latent space, with the target *detached*
   (stop-gradient). This is the core "predict embeddings, not pixels" signal.
2. **VICReg-style regularizer** (optional) — a hinge variance term plus an
   off-diagonal covariance term on the student embeddings. JEPA's EMA target
   already discourages collapse; this is cheap insurance and gives an explicit
   knob. Style mirrors genai-lab ``objectives/regularizers.py``.

3. **SIGReg** (alternative to VICReg) — Sketched Isotropic Gaussian Regularization from
   LeJEPA (Balestriero and LeCun, 2025, arXiv 2511.08544). VICReg constrains only second
   moments: each dimension's spread and the pairwise covariances. SIGReg asks the whole
   embedding distribution to look like an isotropic standard Gaussian. It projects the
   embeddings onto many random unit directions and, on each one-dimensional projection,
   measures how far the empirical characteristic function is from the standard normal's
   (the Epps–Pulley statistic); the loss is the average over directions.

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


def epps_pulley(x: torch.Tensor, n_points: int = 17, t_max: float = 5.0) -> torch.Tensor:
    """Epps–Pulley statistic of each column of ``x`` ``(..., N, M)`` against N(0, 1); returns ``(..., M)``.

    The samples run along the second-to-last axis; leading axes are independent groups.

    For one projected sample ``x_1, ..., x_N`` the empirical characteristic function is
    ``φ̂(t) = (1/N) Σ_j exp(i t x_j)`` and the standard normal's is ``φ(t) = exp(−t²/2)``. The
    statistic is ``T = N ∫ |φ̂(t) − φ(t)|² w(t) dt`` with the Gaussian weight ``w(t) = exp(−t²/2)``,
    integrated by the trapezoid rule on ``n_points`` knots over ``[−t_max, t_max]``. These are
    the values of the LeJEPA paper's code listing (17 knots on [−5, 5]). It is zero when the
    sample's characteristic function matches the standard normal's at every knot, and the factor
    ``N`` makes it the usual test statistic, so its size does not shrink with the batch.
    The real part of ``φ̂`` is the mean of ``cos(t x)`` and the imaginary part the mean of
    ``sin(t x)``; ``φ`` is real, so ``|φ̂ − φ|² = (mean cos − φ)² + (mean sin)²``.
    """
    n = x.shape[-2]
    t = torch.linspace(-t_max, t_max, n_points, device=x.device, dtype=x.dtype)
    phi = torch.exp(-0.5 * t * t)                                   # target CF, also the weight
    xt = x.unsqueeze(-1) * t                                        # (..., N, M, n_points)
    err = (torch.cos(xt).mean(-3) - phi).square() + torch.sin(xt).mean(-3).square()  # (..., M, n_points)
    return torch.trapezoid(err * phi, t, dim=-1) * n


def sigreg(
    z: torch.Tensor,
    n_slices: int = 1024,
    n_points: int = 17,
    t_max: float = 5.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SIGReg: mean Epps–Pulley statistic over ``n_slices`` random unit directions.

    Parameters
    ----------
    z:
        ``(N, D)`` embeddings, one group of ``N`` samples, or ``(V, N, D)``: ``V`` groups
        ("views") tested separately with the same directions and averaged, as LeJEPA applies
        SIGReg to each view's batch. The statistic carries a factor ``N``, so the group size
        sets its scale: the JEPA step passes the context tokens as one group per token
        position (``N`` = batch size), which keeps the token and pooled targets on the same
        scale. Not centered or standardized first: matching N(0, I) includes having mean
        zero and unit variance.
    n_slices:
        Number of random directions, drawn afresh on every call (LeJEPA: resampling each step
        beats a large fixed set). Each direction is a standard Gaussian vector scaled to unit
        length, so it is uniform on the sphere.
    generator:
        Optional generator for the directions, so they do not consume the global random stream.

    Returns
    -------
    ``(loss, components)`` with ``components["sigreg"]`` the same scalar, for logging.
    """
    if z.dim() not in (2, 3):
        raise ValueError(f"sigreg expects (N, D) or (V, N, D), got shape {tuple(z.shape)}")
    a = torch.randn(z.shape[-1], n_slices, device=z.device, dtype=z.dtype, generator=generator)
    a = a / a.norm(dim=0, keepdim=True)
    loss = epps_pulley(z @ a, n_points=n_points, t_max=t_max).mean()
    return loss, {"sigreg": float(loss.detach())}


def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    z_ctx: torch.Tensor | None = None,
    pred_kind: str = "smooth_l1",
    reg_coef: float = 0.0,
    var_coef: float = 1.0,
    cov_coef: float = 0.04,
    reg_kind: str = "vicreg",
    sigreg_slices: int = 1024,
    sigreg_points: int = 17,
    sigreg_t_max: float = 5.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine prediction loss with an optional anti-collapse regularizer.

    Parameters
    ----------
    pred, target:
        ``(B, T, D)`` predicted and EMA-target embeddings.
    z_ctx:
        Embeddings the regularizer acts on: the student context embeddings ``(B, C, D)``
        or a pooled ``(B, D)``. Only needed when ``reg_coef > 0``.
    reg_coef:
        Weight on the regularizer (0 disables it). The two kinds use it differently, each
        as its source defines it:

        - ``"vicreg"``: ``pred + reg_coef · VICReg`` (the default; unchanged from before
          SIGReg existed, so old runs reproduce exactly).
        - ``"sigreg"``: ``(1 − reg_coef) · pred + reg_coef · SIGReg``, LeJEPA's convex form,
          where the paper's default is ``reg_coef = 0.05``.
    reg_kind:
        ``"vicreg"`` or ``"sigreg"``.
    """
    if reg_kind not in ("vicreg", "sigreg"):
        raise ValueError(f"unknown reg_kind: {reg_kind!r}")
    pl = prediction_loss(pred, target, kind=pred_kind)
    components = {"pred": float(pl.detach())}
    total = pl
    if reg_kind == "sigreg":
        if reg_coef > 0.0 and z_ctx is not None:
            reg, reg_comp = sigreg(z_ctx, n_slices=sigreg_slices, n_points=sigreg_points,
                                   t_max=sigreg_t_max, generator=generator)
            total = (1.0 - reg_coef) * pl + reg_coef * reg
            components.update(reg_comp)
        else:
            components["sigreg"] = 0.0
    elif reg_coef > 0.0 and z_ctx is not None:
        reg, reg_comp = vicreg_reg(z_ctx, var_coef=var_coef, cov_coef=cov_coef)
        total = total + reg_coef * reg
        components.update(reg_comp)
    else:
        components.update({"var": 0.0, "cov": 0.0})
    components["loss"] = float(total.detach())
    return total, components
