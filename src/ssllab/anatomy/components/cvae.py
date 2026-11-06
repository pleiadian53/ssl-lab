"""Walkthrough of the conditional NB-VAE baseline (``ssllab.generative.cvae``), built as
``08_train_cvae_baseline.py`` builds it.

Two paths. ``--mode train`` shows the full objective on a training batch: encoder,
reparameterized sample, condition-injected decoder, NB reconstruction and KL.
``--mode generate`` shows how it predicts a perturbation: noise straight into the decoder,
with no encoder and no flow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ssllab.anatomy.components._common import describe, rng_checkpoint, run_sequential
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.registry import component
from ssllab.anatomy.trace import Trace, Walkthrough
from ssllab.anatomy.vocab import Vocab
from ssllab.generative.count_decoder import nb_nll
from ssllab.generative.cvae import ConditionalNBVAE


def build_cvae(geo: Geometry, vocab: Vocab, opts: Options) -> ConditionalNBVAE:
    return ConditionalNBVAE(n_genes=geo.n_hvg, pert_gene=vocab.pert_gene, latent_dim=geo.vae_latent,
                            cond_dim=geo.cond_dim, gene_dim=geo.gene_dim, hidden=geo.vae_hidden,
                            compose=opts.compose, encoder_cond=opts.encoder_cond)


def replay_decode(tr: Trace, m: ConditionalNBVAE, z: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
    tr.section("decoder")
    c = tr.record("c", m.cond(pid), "gene-compositional condition, the same kind of embedding as the flow's "
                  "(see condition-embedding)", module=m.cond, label="condition")
    h = tr.record("[z, c]", torch.cat([z, c], dim=-1), "concatenate latent and condition", label="concat",
                  note="the condition enters the decoder directly")
    h = run_sequential(tr, m.dec, h, "dec", name="h")
    logits = tr.record("logits", m.rate_head(h), f"rate_head: {describe(m.rate_head)}", module=m.rate_head,
                       label="rate_head")
    return tr.record("rho", F.softmax(logits, dim=-1), "softmax over genes", label="softmax")


@component("cvae", group="baseline", aliases=("nb-vae", "vae", "baseline"),
           summary="the from-scratch conditional NB-VAE: training objective or generation",
           options=("mode", "encoder_cond", "compose", "pert"),
           variants=({}, {"mode": "train"}, {"mode": "train", "encoder_cond": True}))
def cvae(geo: Geometry, vocab: Vocab, opts: Options) -> Walkthrough:
    m = build_cvae(geo, vocab, opts)
    tr = Trace()
    shown = {"mode": opts.mode, "encoder_cond": opts.encoder_cond, "compose": opts.compose}
    if opts.mode == "generate":
        name = vocab.resolve(opts.pert)
        pid0 = vocab.pert_id(name)
        tr.section("source")
        z = tr.record("z", torch.randn(geo.n, geo.vae_latent, generator=torch.Generator().manual_seed(0)),
                      "Gaussian noise N(0, I): no encoder at generation", label="noise")
        pid = torch.full((geo.n,), pid0, dtype=torch.long)
        rho = replay_decode(tr, m, z, pid)
        tr.section("read-out")
        expr = tr.record("pred_mean", torch.log1p(1e4 * rho).mean(0), "log1p(10⁴·rho), averaged over the cells",
                         label="mean")
        ref = m.generate_expression(pid0, geo.n, generator=torch.Generator().manual_seed(0))
        return Walkthrough(component="cvae", title=f"predicting {name} with the baseline", options=shown | {"pert": name},
                           geometry=geo.as_dict(), steps=tr.steps, modules={"cVAE": m}, output=expr, reference=ref,
                           checked="pred_mean", inputs=f"{geo.n} cells under {name}")

    b = geo.batch
    x = torch.randn(b, geo.n_hvg).abs()
    counts = torch.poisson(torch.full((b, geo.n_hvg), 2.0))
    ell = counts.sum(1).clamp_min(1.0)
    pid = torch.randint(vocab.n_perts, (b,))
    with rng_checkpoint() as rng:
        tr.section("encoder")
        tr.record("x", x, "log1p-CP10K expression of each training cell", label="")
        tr.record("pert_id", pid, "each cell's perturbation (a mixed batch)", label="")
        h = x
        if opts.encoder_cond:
            c_enc = tr.record("c", m.cond(pid), "condition vector for the encoder", module=m.cond, label="condition")
            h = tr.record("[x, c]", torch.cat([x, c_enc], dim=-1), "concatenate: the encoder reads the condition",
                          label="concat", note="q(z | x, c), the textbook conditional VAE")
        h = run_sequential(tr, m.enc, h, "enc", name="h")
        mu_z = tr.record("mu_z", m.to_mu(h), f"to_mu: {describe(m.to_mu)}", module=m.to_mu, label="to_mu")
        logvar = tr.record("logvar", m.to_logvar(h).clamp(-10.0, 10.0), f"to_logvar: {describe(m.to_logvar)}, "
                           "clamped to [-10, 10]", module=m.to_logvar, label="to_logvar")
        tr.section("sample")
        z = tr.record("z", mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar),
                      "reparameterize: mu_z + noise·exp(logvar / 2)", label="reparam")
        rho = replay_decode(tr, m, z, pid)
        tr.section("loss")
        kappa = tr.record("kappa", F.softplus(m.log_kappa) + 1e-4, "per-gene dispersion", params=m.log_kappa.numel(),
                          label="kappa")
        mu = tr.record("mu", ell.unsqueeze(-1) * rho, "NB mean: library size × rho", label="× ell")
        recon = tr.record("recon", nb_nll(counts, mu, kappa), "NB negative log-likelihood of the real counts",
                          label="NB NLL")
        q = "q(z | x, c)" if opts.encoder_cond else "q(z | x)"
        kl = tr.record("kl", -0.5 * torch.mean(torch.sum(1 + logvar - mu_z.pow(2) - logvar.exp(), dim=1)),
                       f"KL({q} ‖ N(0, I)), averaged over cells", label="KL")
        loss = tr.record("loss", recon + 1.0 * kl, "recon + β·KL (β = 1)", label="total")
        rng.rewind()
        ref, _ = m.loss(x, counts, ell, pid)
    return Walkthrough(component="cvae", title="one training step of the baseline", options=shown,
                       geometry=geo.as_dict(), steps=tr.steps, modules={"cVAE": m}, output=loss, reference=ref,
                       checked="the loss", inputs=f"{b} training cells (random stand-in expression and counts)")
