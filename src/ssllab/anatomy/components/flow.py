"""Walkthroughs of Stage B: the time embedding, the condition embedding, the velocity field
and the conditional flow that integrates it.

The modules are the real ones from ``ssllab.generative.flow`` and
``ssllab.generative.condition``, built the way ``04_train_cond_flow.py`` builds them. Inputs
that would come from upstream stages (control latents, the condition vector) are random
stand-ins of the right shape; the shapes and operations do not depend on their values.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssllab.anatomy.components._common import (
    describe,
    mlp_summary,
    rng_checkpoint,
    run_residual_mlps,
    run_sequential,
)
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.registry import component
from ssllab.anatomy.trace import Trace, Walkthrough
from ssllab.anatomy.vocab import Vocab
from ssllab.data.perturbseq import target_genes
from ssllab.generative.condition import ConditionEncoder, GeneSetConditionEncoder, GeneSetEmbedding
from ssllab.generative.flow import VelocityMLP, sinusoidal_time_embedding
from ssllab.generative.perturb import sample_perturbed_latents

FLOW_VARIANTS = (
    {},
    {"compose": "deepsets"},
    {"cond_type": "table"},
    {"flow_base": "gaussian"},
    {"flow_base": "gaussian", "cond_type": "table"},
)


# --------------------------------------------------------------------------- #
# Builders (mirror 04_train_cond_flow.py).
# --------------------------------------------------------------------------- #
def build_flow(geo: Geometry) -> VelocityMLP:
    return VelocityMLP(data_dim=geo.embed_dim, hidden=geo.flow_hidden, n_layers=geo.flow_layers,
                       time_dim=geo.time_dim, cond_dim=geo.cond_dim)


def build_condition(geo: Geometry, vocab: Vocab, opts: Options) -> nn.Module:
    """The condition module for (flow_base, cond_type), exactly as ``04`` chooses it.

    Transport base: the condition is the perturbation embedding alone, ``c = z_p``.
    Noise base: the condition fuses a control latent with it, ``c = MLP([z_b, z_p])``.
    """
    geneset = opts.cond_type == "geneset"
    if opts.flow_base == "control":
        if geneset:
            return GeneSetEmbedding(vocab.pert_gene, gene_dim=geo.gene_dim, out_dim=geo.cond_dim, compose=opts.compose)
        return nn.Embedding(vocab.n_perts, geo.cond_dim)
    if geneset:
        return GeneSetConditionEncoder(latent_dim=geo.embed_dim, pert_gene=vocab.pert_gene, pert_dim=geo.pert_dim,
                                       gene_dim=geo.gene_dim, cond_dim=geo.cond_dim, compose=opts.compose)
    return ConditionEncoder(latent_dim=geo.embed_dim, n_perts=vocab.n_perts, pert_dim=geo.pert_dim,
                            cond_dim=geo.cond_dim)


def call_condition(cond: nn.Module, opts: Options, pid: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return cond(pid) if opts.flow_base == "control" else cond(z_b, pid)


def _variant(opts: Options, *keys: str) -> dict:
    out = {k: getattr(opts, k) for k in keys}
    if opts.cond_type == "table" and "compose" in out:
        out.pop("compose")                    # compose applies to the gene-set condition only
    return out


# --------------------------------------------------------------------------- #
# Replays shared between components.
# --------------------------------------------------------------------------- #
def replay_time_branch(tr: Trace, flow: VelocityMLP, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``t -> (scale_t, shift_t)``: the sinusoidal features, then the time MLP, then a split."""
    tr.section("time branch")
    tr.record("t", t, "current time, the same value in every row", label="")
    half = flow.time_dim // 2
    temb = sinusoidal_time_embedding(t, flow.time_dim)
    probe = sinusoidal_time_embedding(torch.tensor([0.0, 0.99]), flow.time_dim)
    moving = int(((probe[1] - probe[0]).abs() > 0.1).sum())
    tr.record("temb", temb, f"sin and cos of t at {half} frequencies, from 1 down to 1/10,000",
              label="sin/cos", note=f"only {moving} of {flow.time_dim} features change by more than 0.1 "
                                    f"between t = 0 and t = 0.99")
    h = run_sequential(tr, flow.time_mlp, temb, "time_mlp")
    scale, shift = h.chunk(2, dim=-1)
    tr.record("scale_t, shift_t", scale, "split the columns in half: a scale and a shift per feature",
              label="split", note="two tensors of this shape")
    return scale, shift


def replay_velocity(
    tr: Trace, flow: VelocityMLP, z: torch.Tensor, t: torch.Tensor, c: torch.Tensor, expand: bool,
) -> torch.Tensor:
    """One evaluation of ``v(z, t, c)``, operation by operation, as ``VelocityMLP.forward`` does it."""
    scale, shift = replay_time_branch(tr, flow, t)

    tr.section("condition branch")
    tr.record("c", c, "condition vector (see condition-embedding), the same in every row", label="")
    hc = run_sequential(tr, flow.cond_mlp, c, "cond_mlp")
    cscale, cshift = hc.chunk(2, dim=-1)
    tr.record("scale_c, shift_c", cscale, "split the columns in half", label="split",
              note="two tensors of this shape")

    tr.section("cell branch")
    tr.record("z", z, "current positions, one row per cell", label="")
    h = tr.record("h", flow.in_proj(z), f"in_proj: {describe(flow.in_proj)}", module=flow.in_proj, label="in_proj")
    h = tr.record("h", h * (1.0 + scale) + shift, "FiLM with time: h·(1 + scale_t) + shift_t", label="FiLM(t)")
    h = tr.record("h", h * (1.0 + cscale) + cshift, "FiLM with condition: h·(1 + scale_c) + shift_c",
                  label="FiLM(c)")
    h = run_residual_mlps(tr, flow.blocks, h, "blocks", expand)
    return tr.record("v", flow.out_proj(h), f"out_proj: {describe(flow.out_proj)}; one velocity per cell",
                     module=flow.out_proj, label="out_proj")


def replay_condition(tr: Trace, cond: nn.Module, vocab: Vocab, opts: Options, name: str,
                     pid: torch.Tensor, z_b: torch.Tensor | None) -> torch.Tensor:
    """Perturbation name -> condition vector ``c``, for each of the four (flow_base, cond_type) builds."""
    n = pid.shape[0]
    genes = target_genes(name)
    tr.section("identity")
    tr.record("pert_id", pid, f'"{name}" → index {int(pid[0])} of {vocab.n_perts} perturbations, '
                              f"repeated for the {n} cells", label="lookup")

    if opts.cond_type == "table":
        emb = cond if opts.flow_base == "control" else cond.pert_emb
        tr.section("perturbation embedding")
        z_p = tr.record("z_p" if opts.flow_base == "gaussian" else "c", emb(pid),
                        f"row {int(pid[0])} of a learned {describe(emb)}", module=emb, label="table row",
                        note="one free row per perturbation; a combination held out of training keeps its untrained row")
    else:
        emb = cond if opts.flow_base == "control" else cond.pert_emb
        mh = emb.pert_gene[pid]
        tr.record("multi-hot", mh, f"row of the fixed membership table ({vocab.n_perts} × {len(vocab.genes)}): "
                                   f"1 at {', '.join(genes) or 'no gene'}", label="membership",
                  note="not learned: read off the name")
        tr.section("perturbation embedding")
        if emb.compose == "additive":
            z_p = tr.record("z_p", mh @ emb.gene_emb.weight,
                            "sum of the targeted genes' vectors: " + (" + ".join(f"e({g})" for g in genes) or "0"),
                            module=emb.gene_emb, label="sum e(g)",
                            note=f"gene table: {describe(emb.gene_emb)}")
        else:
            hg = tr.record("psi(e)", emb.psi(emb.gene_emb.weight),
                           f"psi: [{mlp_summary(emb.psi)}] applied to every gene's vector",
                           params=sum(p.numel() for p in emb.gene_emb.parameters()) + sum(p.numel() for p in emb.psi.parameters()),
                           label="psi", note="shape is one row per gene, not per cell")
            s = tr.record("sum", mh @ hg, "sum over the targeted genes: " + (" + ".join(f"psi(e({g}))" for g in genes) or "0"),
                          label="sum")
            z_p = tr.record("z_p", emb.phi(s), f"phi: [{mlp_summary(emb.phi)}]", module=emb.phi, label="phi")
        z_p = tr.record("z_p" if opts.flow_base == "gaussian" else "c", emb.proj(z_p),
                        f"proj: {describe(emb.proj)}", module=emb.proj, label="proj",
                        note="identity because gene_dim equals the output width" if isinstance(emb.proj, nn.Identity) else "")

    if opts.flow_base == "control":
        return z_p
    tr.section("fuse with a control cell (noise-base flow only)")
    tr.record("z_b", z_b, "one control latent per cell (stand-in values)", label="")
    h = tr.record("[z_b, z_p]", torch.cat([z_b, z_p], dim=-1), "concatenate", label="concat")
    return run_sequential(tr, cond.net, h, "net", name="c")


# --------------------------------------------------------------------------- #
# Components.
# --------------------------------------------------------------------------- #
@component("time-embedding", group="flow", aliases=("time",),
           summary="flow time t → sinusoidal features → MLP → FiLM scale and shift", options=("t",))
def time_embedding(geo: Geometry, vocab: Vocab, opts: Options) -> Walkthrough:
    flow = build_flow(geo)
    t = torch.full((geo.n,), opts.t)
    tr = Trace()
    scale, shift = replay_time_branch(tr, flow, t)
    ref = flow.time_mlp(sinusoidal_time_embedding(t, geo.time_dim))
    return Walkthrough(
        component="time-embedding", title="how the flow reads the clock", options=_variant(opts, "t"),
        geometry=geo.as_dict(), steps=tr.steps, modules={"time MLP": flow.time_mlp},
        output=torch.cat([scale, shift], dim=-1), reference=ref, checked="scale and shift",
        inputs=f"t = {opts.t} for {geo.n} cells",
    )


@component("condition-embedding", group="flow", aliases=("condition", "cond"),
           summary="perturbation name → gene membership → embedding → condition vector c",
           options=("pert", "flow_base", "cond_type", "compose"), variants=FLOW_VARIANTS)
def condition_embedding(geo: Geometry, vocab: Vocab, opts: Options) -> Walkthrough:
    name = vocab.resolve(opts.pert)
    cond = build_condition(geo, vocab, opts)
    pid = torch.full((geo.n,), vocab.pert_id(name), dtype=torch.long)
    z_b = torch.randn(geo.n, geo.embed_dim) if opts.flow_base == "gaussian" else None
    tr = Trace()
    c = replay_condition(tr, cond, vocab, opts, name, pid, z_b)
    return Walkthrough(
        component="condition-embedding", title="how the flow is told which perturbation to apply",
        options=_variant(opts, "flow_base", "cond_type", "compose") | {"pert": name},
        geometry=geo.as_dict(), steps=tr.steps, modules={"condition module": cond},
        output=c, reference=call_condition(cond, opts, pid, z_b), checked="c",
        inputs=f"{geo.n} cells under {name} (vocabulary: {vocab.source})",
    )


@component("velocity-field", group="flow", aliases=("velocity",),
           summary="one evaluation of v(z, t, c): time and condition FiLM, residual MLP",
           options=("t", "guidance", "expand"), variants=({}, {"guidance": 2.0}, {"expand": True}))
def velocity_field(geo: Geometry, vocab: Vocab, opts: Options) -> Walkthrough:
    flow = build_flow(geo)
    z = torch.randn(geo.n, geo.embed_dim)
    c = torch.randn(geo.n, geo.cond_dim)
    t = torch.full((geo.n,), opts.t)
    tr = Trace()
    v = replay_velocity(tr, flow, z, t, c, opts.expand)
    ref = flow(z, t, c)
    if opts.guidance != 1.0:
        tr.section("classifier-free guidance")
        null = flow.null_cond.expand(geo.n, -1)
        v_null = tr.record("v_null", flow(z, t, null), "the same network again, with the learned null condition",
                           params=flow.null_cond.numel(), label="null pass")
        v = tr.record("v", v_null + opts.guidance * (v - v_null), f"v_null + {opts.guidance:g}·(v − v_null)",
                      label="guide")
        ref = flow(z, t, null) + opts.guidance * (flow(z, t, c) - flow(z, t, null))
    return Walkthrough(
        component="velocity-field", title="one evaluation of the velocity network v(z, t, c)",
        options=_variant(opts, "t", "guidance"), geometry=geo.as_dict(), steps=tr.steps,
        modules={"velocity field": flow}, output=v, reference=ref, checked="v",
        inputs=f"{geo.n} cells; z and c are random stand-ins of the right shape",
    )


@component("conditional-flow", group="flow", aliases=("flow",),
           summary="sampling: source cells → Euler integration of v → outcome latents",
           options=("pert", "flow_base", "cond_type", "compose", "guidance"), variants=FLOW_VARIANTS)
def conditional_flow(geo: Geometry, vocab: Vocab, opts: Options) -> Walkthrough:
    name = vocab.resolve(opts.pert)
    flow, cond = build_flow(geo), build_condition(geo, vocab, opts)
    bundle = {"flow": flow, "cond": cond, "flow_base": opts.flow_base, "dim": geo.embed_dim,
              "ctrl_pool": torch.randn(geo.n_ctrl, geo.embed_dim),
              "mean": torch.randn(geo.embed_dim), "std": torch.rand(geo.embed_dim) + 0.5}
    pid0 = vocab.pert_id(name)
    tr, n, steps = Trace(), geo.n, geo.steps

    with rng_checkpoint() as rng:
        g = torch.Generator().manual_seed(0)
        z = replay_flow(tr, bundle, vocab, opts, name, pid0, n, steps, g)
        rng.rewind()
        ref = sample_perturbed_latents(bundle, pid0, n, guidance=opts.guidance, steps=steps,
                                       generator=torch.Generator().manual_seed(0))
    return Walkthrough(
        component="conditional-flow", title="generating outcome latents for one perturbation",
        options=_variant(opts, "flow_base", "cond_type", "compose", "guidance") | {"pert": name},
        geometry=geo.as_dict(), steps=tr.steps, modules={"velocity field": flow, "condition module": cond},
        output=z, reference=ref, checked="the generated latents",
        inputs=f"{n} cells under {name}; control pool and standardization statistics are random stand-ins",
    )


def replay_flow(tr: Trace, bundle: dict, vocab: Vocab, opts: Options, name: str, pid0: int,
                n: int, steps: int, g: torch.Generator) -> torch.Tensor:
    """``sample_perturbed_latents`` + ``euler_sample``, step by step (velocity collapsed to one row)."""
    flow, cond, pool = bundle["flow"], bundle["cond"], bundle["ctrl_pool"]
    tr.section("source")
    tr.record("ctrl_pool", pool, "standardized latents of the training control cells", label="")
    idx = torch.randint(len(pool), (n,), generator=g)
    z_b = tr.record("z_b", pool[idx], f"draw {n} rows at random, with replacement", label="draw")
    pid = torch.full((n,), pid0, dtype=torch.long)

    tr.section("condition")
    c = call_condition(cond, opts, pid, z_b)
    tr.record("c", c, f'condition-embedding of "{name}"' + ("" if opts.flow_base == "control" else " fused with z_b"),
              module=cond, label="condition")

    tr.section("integrate")
    if opts.flow_base == "control":
        z = tr.record("z", z_b, "start at the control cells (t = 0)", label="start")
    else:
        z = tr.record("z", torch.randn(n, bundle["dim"]), "start from Gaussian noise N(0, I) (t = 0)", label="start")
    dt = 1.0 / steps
    null = flow.null_cond.expand(n, -1) if opts.guidance != 1.0 else None
    passes = "two forward passes (guided)" if null is not None else "one forward pass"
    for k in range(steps):
        t = torch.full((n,), k * dt)
        if null is not None:
            v = flow(z, t, null) + opts.guidance * (flow(z, t, c) - flow(z, t, null))
        else:
            v = flow(z, t, c)
        z = z + dt * v
        if k == 0:
            tr.record("v", v, f"step 0: velocity field at t = 0, {passes} (see velocity-field)",
                      module=flow, label="v(z,t,c)")
            tr.record("z", z, f"z + {dt:g}·v", label="Euler step")
            if steps > 2:
                tr.note(f"steps 1 … {steps - 2}: the same two rows at t = k/{steps}")
    if steps > 1:
        tr.record("z", z, f"after step {steps - 1} (t = {(steps - 1) * dt:g}): the outcome, standardized units",
                  label=f"{steps} steps")
    tr.section("output")
    return tr.record("z_out", z * bundle["std"] + bundle["mean"], "undo the standardization: z·std + mean",
                     label="de-standardize", note="back in the encoder's units, ready for the decoder")
