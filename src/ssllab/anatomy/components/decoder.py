"""Walkthrough of Stage C: the negative-binomial count decoder and its two read-outs.

The module is the real ``ssllab.generative.count_decoder.CountDecoder`` with the flags
``03_train_count_decoder.py`` exposes (``--zinb``, ``--anchored-mean``,
``--state-dispersion``). The mean read-out is what the effect-size score uses; the
population read-out samples counts and is what the calibration metrics use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ssllab.anatomy.components._common import describe, run_sequential
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.registry import component
from ssllab.anatomy.trace import Trace, Walkthrough
from ssllab.anatomy.vocab import Vocab
from ssllab.generative.count_decoder import CountDecoder


def build_decoder(geo: Geometry, opts: Options) -> CountDecoder:
    return CountDecoder(latent_dim=geo.embed_dim, n_genes=geo.n_hvg, hidden_dims=geo.dec_hidden, zinb=opts.zinb,
                        anchored_mean=opts.anchored_mean, state_dispersion=opts.state_dispersion)


def replay_decoder(tr: Trace, dec: CountDecoder, z: torch.Tensor, ell: torch.Tensor) -> dict[str, torch.Tensor]:
    """``CountDecoder.forward``: trunk, mean head (+ baseline), dispersion, optional dropout gate."""
    tr.section("trunk")
    tr.record("z", z, "cell latents, one row per cell", label="")
    h = run_sequential(tr, dec.trunk, z, "trunk", name="h")

    tr.section("mean")
    logits = tr.record("logits", dec.rate_head(h), f"rate_head: {describe(dec.rate_head)}, one score per gene",
                       module=dec.rate_head, label="rate_head",
                       note="zero-initialized: starts at the baseline" if dec.log_rho_base is not None else "")
    if dec.log_rho_base is not None:
        logits = tr.record("logits", logits + dec.log_rho_base,
                           "add the learned baseline log-profile (set to the control profile before training)",
                           params=dec.log_rho_base.numel(), label="+ baseline")
    rho = tr.record("rho", F.softmax(logits, dim=-1), "softmax over genes: each cell's share of expression per gene",
                    label="softmax", note="each row sums to 1; no library size yet")
    tr.record("ell", ell, "library size of each cell (given, not predicted)", label="")
    mu = tr.record("mu", ell.unsqueeze(-1) * rho, "NB mean: mu = ell·rho", label="× ell")

    tr.section("dispersion")
    if dec.kappa_head is not None:
        kappa = tr.record("kappa", F.softplus(dec.kappa_head(h)) + 1e-4,
                          f"kappa_head: {describe(dec.kappa_head)}, then softplus: a dispersion per cell and gene",
                          module=dec.kappa_head, label="kappa_head")
    else:
        kappa = tr.record("kappa", F.softplus(dec.log_kappa) + 1e-4,
                          "softplus of one learned number per gene: the same for every cell",
                          params=dec.log_kappa.numel(), label="per-gene kappa",
                          note="variance = mu + mu²/kappa")
    out = {"rho": rho, "mu": mu, "kappa": kappa}
    if dec.dropout_head is not None:
        out["pi_logits"] = tr.record("pi_logits", dec.dropout_head(h),
                                     f"dropout_head: {describe(dec.dropout_head)}: logit of a structural zero",
                                     module=dec.dropout_head, label="dropout gate")
    return out


@component("count-decoder", group="decoder", aliases=("decoder", "nb-decoder"),
           summary="cell latent z → gene rates rho → NB mean and dispersion → expression or counts",
           options=("zinb", "anchored_mean", "state_dispersion"),
           variants=({}, {"zinb": True}, {"anchored_mean": True}, {"state_dispersion": True}))
def count_decoder(geo: Geometry, vocab: Vocab, opts: Options) -> Walkthrough:
    dec = build_decoder(geo, opts)
    z = torch.randn(geo.n, geo.embed_dim)
    ell = torch.full((geo.n,), vocab.median_libsize)
    tr = Trace()
    out = replay_decoder(tr, dec, z, ell)
    ref = dec(z, ell)

    tr.section("read-out: mean profile (effect-size score)")
    expr = tr.record("expr", torch.log1p(1e4 * out["rho"]), "log1p(10⁴·rho): normalized expression per cell, no sampling",
                     label="log1p")
    tr.record("pred_mean", expr.mean(0), "average over the cells", label="mean",
              note="compared with real cells as Δ = pred_mean − control_mean on the top-20 genes")
    tr.section("read-out: population (calibration metrics)")
    counts = tr.record("counts", dec.sample_counts(z, ell), "sample integer counts: rate ~ Gamma(kappa, kappa/mu), "
                                                             "count ~ Poisson(rate)", label="sample NB",
                       note="zero-inflated: then zero each count with probability sigmoid(pi)" if opts.zinb else "")
    tr.record("expr", torch.log1p(1e4 * counts / ell.unsqueeze(-1)), "log1p(10⁴·counts / ell) per cell",
              label="log1p")

    keys = list(ref)
    flags = {k: getattr(opts, k) for k in ("zinb", "anchored_mean", "state_dispersion")}
    return Walkthrough(
        component="count-decoder", title="from a cell latent to gene expression", options=flags,
        geometry=geo.as_dict(), steps=tr.steps, modules={"decoder": dec},
        output=torch.cat([out[k].flatten() for k in keys]), reference=torch.cat([ref[k].flatten() for k in keys]),
        checked=", ".join(keys),
        inputs=f"{geo.n} random stand-in latents; library size {vocab.median_libsize:,.0f} (median of {vocab.source})",
    )
