"""CPU-fast smoke tests for the generative-JEPA vertical slice.

These assert the pieces wire together (shapes, finite losses, gradients) on
random data — not learning quality. Run with: ``pytest -q``.
"""

from __future__ import annotations

import torch

from ssllab.data.mnist import N_TOKENS, TOKEN_DIM, patchify, unpatchify
from ssllab.eval.collapse import collapse_report
from ssllab.generative.flow import VelocityMLP, cfm_loss, euler_sample
from ssllab.jepa.model import build_jepa
from ssllab.models.decoder import LatentDecoder
from ssllab.utils import set_seed

set_seed(0)


def test_patchify_roundtrip():
    x = torch.rand(4, 1, 28, 28)
    tok = patchify(x)
    assert tok.shape == (4, N_TOKENS, TOKEN_DIM)
    assert torch.allclose(unpatchify(tok), x, atol=1e-6)


def test_jepa_forward_backward():
    jepa = build_jepa(token_dim=TOKEN_DIM, n_tokens=N_TOKENS, reg_coef=0.04)
    tok = torch.rand(8, N_TOKENS, TOKEN_DIM)
    loss, comp = jepa(tok)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    # Some encoder parameter received a gradient.
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in jepa.encoder.parameters())
    for k in ("pred", "var", "cov", "loss"):
        assert k in comp


def test_ema_update_changes_teacher():
    jepa = build_jepa(token_dim=TOKEN_DIM, n_tokens=N_TOKENS)
    before = [p.clone() for p in jepa.ema.teacher.parameters()]
    # Perturb the student so the EMA step has something to move toward.
    with torch.no_grad():
        for p in jepa.encoder.parameters():
            p.add_(0.1)
    m = jepa.update_target(step=0, total=10)
    assert 0.0 <= m <= 1.0
    after = list(jepa.ema.teacher.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_embed_pooled_shape():
    jepa = build_jepa(token_dim=TOKEN_DIM, n_tokens=N_TOKENS, embed_dim=128)
    z = jepa.embed(torch.rand(8, N_TOKENS, TOKEN_DIM))
    assert z.shape == (8, 128)
    rep = collapse_report(z)
    assert rep["effective_rank"] >= 1.0


def test_flow_roundtrip():
    dim = 16
    v = VelocityMLP(data_dim=dim, hidden=64, n_layers=2)
    z1 = torch.randn(32, dim)
    loss = cfm_loss(v, z1)
    assert torch.isfinite(loss)
    loss.backward()
    samples = euler_sample(v, n=10, dim=dim, n_steps=8)
    assert samples.shape == (10, dim) and torch.isfinite(samples).all()


def test_conditional_flow():
    dim, cond_dim, b = 16, 8, 32
    v = VelocityMLP(data_dim=dim, hidden=64, n_layers=2, cond_dim=cond_dim)
    z1 = torch.randn(b, dim)
    c = torch.randn(b, cond_dim)

    # Conditional CFM loss is finite and trains the condition path (cond_mlp + null token).
    loss = cfm_loss(v, z1, c=c, p_drop=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert v.cond_mlp[0].weight.grad is not None and torch.isfinite(v.cond_mlp[0].weight.grad).all()
    assert v.null_cond.grad is not None  # dropout exercised the null token

    # Fixed-condition sampling returns the right shape, with and without guidance.
    s = euler_sample(v, n=10, dim=dim, n_steps=8, c=torch.randn(10, cond_dim))
    assert s.shape == (10, dim) and torch.isfinite(s).all()
    s_cfg = euler_sample(v, n=10, dim=dim, n_steps=8, c=torch.randn(10, cond_dim), guidance=2.0)
    assert s_cfg.shape == (10, dim) and torch.isfinite(s_cfg).all()

    # The condition actually steers the field: different c -> different velocity.
    z_t, t = torch.randn(4, dim), torch.rand(4)
    c0, c1 = torch.zeros(4, cond_dim), torch.ones(4, cond_dim)
    with torch.no_grad():
        assert not torch.allclose(v(z_t, t, c0), v(z_t, t, c1))


def test_conditional_flow_guards_mismatch():
    # Unconditional field rejects a condition; conditional field requires one.
    uncond = VelocityMLP(data_dim=8, hidden=32, n_layers=1)
    cond = VelocityMLP(data_dim=8, hidden=32, n_layers=1, cond_dim=4)
    z_t, t = torch.randn(2, 8), torch.rand(2)
    import pytest

    with pytest.raises(ValueError):
        uncond(z_t, t, torch.randn(2, 4))
    with pytest.raises(ValueError):
        cond(z_t, t)


def test_decoder_shape():
    dec = LatentDecoder(latent_dim=128)
    out = dec(torch.randn(8, 128))
    assert out["logits"].shape == (8, 28 * 28)
    assert dec.decode_images(torch.randn(8, 128)).shape == (8, 1, 28, 28)
