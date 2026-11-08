"""SIGReg (LeJEPA) and the reproducibility guard for the VICReg default."""

from __future__ import annotations

import math

import torch

from ssllab.checkpoint import load_jepa, save_jepa
from ssllab.jepa import build_jepa
from ssllab.objectives.jepa_loss import epps_pulley, jepa_loss, sigreg

GEOM = dict(token_dim=10, n_tokens=12, embed_dim=32, enc_depth=2, pred_depth=1, n_heads=4, n_target=4)


def _step(**cfg):
    torch.manual_seed(0)
    j = build_jepa(**GEOM, **cfg)
    torch.manual_seed(1)
    x = torch.randn(8, 12, 10)
    loss, comp = j(x)
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in j.parameters() if p.grad is not None)
    return loss.item(), grad.item(), comp


def test_vicreg_default_is_bit_identical_to_before_sigreg_existed():
    # Values recorded on 2026-11-08 with the code before SIGReg was added (same seeds, geometry).
    loss, grad, comp = _step(reg_coef=0.04)
    assert loss == 0.7444103360176086
    assert grad == 80.24014282226562
    assert comp["pred"] == 0.7353634834289551


def test_old_config_without_the_new_fields_loads_as_vicreg(tmp_path):
    j = build_jepa(**GEOM, reg_coef=0.04)
    path = save_jepa(j, tmp_path / "enc.pt")
    ck = torch.load(path)
    for k in ("reg_kind", "reg_target", "sigreg_slices", "sigreg_points", "sigreg_t_max"):
        ck["cfg"].pop(k)                                   # what a pre-SIGReg checkpoint holds
    torch.save(ck, path)
    j2 = load_jepa(path)
    assert j2.cfg.reg_kind == "vicreg" and j2.cfg.reg_target == "tokens"


def test_epps_pulley_is_small_for_gaussian_and_large_for_shifted_or_scaled():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4000, 1, generator=g)
    t_gauss = epps_pulley(x).item()
    # Under the null E|φ̂(t) − φ(t)|² = (1 − φ(t)²)/N, so E[T] = ∫(1 − e^{−t²})e^{−t²/2} dt
    # = √(2π) − √(2π/3) ≈ 1.06 (over the whole line; [−5, 5] changes it negligibly).
    assert t_gauss < 5.0
    assert epps_pulley(x + 1.0).item() > 50 * t_gauss      # wrong mean
    assert epps_pulley(3.0 * x).item() > 50 * t_gauss      # wrong scale


def test_epps_pulley_matches_a_direct_integral():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, 1, dtype=torch.float64, generator=g) * 1.3 + 0.2
    t = torch.linspace(-5, 5, 2001, dtype=torch.float64)
    ecf = torch.exp(1j * x * t).mean(0)                    # (2001,)
    phi = torch.exp(-0.5 * t ** 2)
    direct = 50 * torch.trapezoid((ecf - phi).abs() ** 2 * phi, t)
    coarse = epps_pulley(x).item()                         # 17 knots, as in the paper
    assert math.isclose(coarse, direct.item(), rel_tol=0.02)


def test_sigreg_prefers_isotropic_gaussian_embeddings():
    g = torch.Generator().manual_seed(2)
    iso = torch.randn(512, 16, generator=g)
    collapsed = iso.clone()
    collapsed[:, 1:] = 0.01 * collapsed[:, 1:]             # all spread in one direction
    a, _ = sigreg(iso, n_slices=256, generator=torch.Generator().manual_seed(0))
    b, _ = sigreg(collapsed, n_slices=256, generator=torch.Generator().manual_seed(0))
    assert b.item() > 20 * a.item()


def test_sigreg_loss_uses_the_convex_form_and_trains():
    pred, target = torch.randn(4, 3, 8), torch.randn(4, 3, 8)
    z = torch.randn(64, 8, requires_grad=True)
    total, comp = jepa_loss(pred, target, z_ctx=z, reg_coef=0.05, reg_kind="sigreg",
                            sigreg_slices=32, generator=torch.Generator().manual_seed(0))
    assert math.isclose(total.item(), 0.95 * comp["pred"] + 0.05 * comp["sigreg"], rel_tol=1e-5)
    total.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_sigreg_jepa_step_runs_for_both_targets_and_leaves_global_rng_alone():
    def after_step(**cfg):
        torch.manual_seed(5)
        j = build_jepa(**GEOM, **cfg)
        loss, comp = j(torch.randn(8, 12, 10))
        loss.backward()
        return loss, comp, torch.rand(1).item()            # next draw from the global stream

    _, _, ref = after_step(reg_coef=0.04)
    for target in ("tokens", "pooled"):
        loss, comp, nxt = after_step(reg_coef=0.05, reg_kind="sigreg", reg_target=target, sigreg_slices=64)
        assert torch.isfinite(loss) and comp["sigreg"] > 0
        assert nxt == ref                                  # masks and init drew exactly as under VICReg


def test_views_are_tested_separately_and_averaged():
    g = torch.Generator().manual_seed(3)
    z = torch.randn(5, 64, 8, generator=g)
    gen = lambda: torch.Generator().manual_seed(0)       # noqa: E731
    joint, _ = sigreg(z, n_slices=16, generator=gen())
    each = torch.stack([sigreg(v, n_slices=16, generator=gen())[0] for v in z])
    assert math.isclose(joint.item(), each.mean().item(), rel_tol=1e-5)
