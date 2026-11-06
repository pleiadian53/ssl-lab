"""Walkthroughs of Stage A: tokenizing a cell, the JEPA encoder, its three training branches
(student, teacher, predictor), and the full training step.

The model is the real ``ssllab.jepa.model.JEPA`` built by ``build_jepa`` with the arguments
``01_pretrain_stage_a.py`` passes. Masks are drawn with ``sample_block_masks`` exactly as
``JEPA.forward`` draws them, so the replayed training step reproduces the real loss.
"""

from __future__ import annotations

import torch

from ssllab.anatomy.components._common import describe, rng_checkpoint, run_transformer_blocks
from ssllab.anatomy.geometry import Geometry
from ssllab.anatomy.options import Options
from ssllab.anatomy.registry import component
from ssllab.anatomy.trace import Trace, Walkthrough
from ssllab.anatomy.vocab import Vocab
from ssllab.data.perturbseq import make_gene_partition, tokenize_cells
from ssllab.jepa.masking import batched, sample_block_masks
from ssllab.jepa.model import JEPA, build_jepa
from ssllab.models.encoder import JEPAEncoder
from ssllab.objectives.jepa_loss import prediction_loss, vicreg_reg


def build(geo: Geometry, opts: Options) -> JEPA:
    return build_jepa(token_dim=geo.token_dim, n_tokens=geo.n_tokens, embed_dim=geo.embed_dim,
                      enc_depth=geo.enc_depth, pred_depth=geo.pred_depth, n_heads=geo.n_heads,
                      n_target=geo.n_target, reg_coef=opts.reg_coef)


def stand_in_tokens(geo: Geometry) -> torch.Tensor:
    return torch.randn(geo.batch, geo.n_tokens, geo.token_dim)


def replay_encoder(tr: Trace, enc: JEPAEncoder, tokens: torch.Tensor, idx: torch.Tensor | None,
                   expand: bool, who: str) -> torch.Tensor:
    """``JEPAEncoder.forward``: select tokens, embed, add positions, transformer, final norm."""
    b, n, _ = tokens.shape
    if idx is None:
        idx = torch.arange(n).unsqueeze(0).expand(b, -1)
        sel = tokens
    else:
        sel = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        tr.record("tokens[ctx]", sel, f"keep only the {idx.shape[1]} context tokens (gather by position)",
                  label="gather")
    x = tr.record("x", enc.patch_embed(sel), f"patch_embed: {describe(enc.patch_embed.proj)}, applied to each token",
                  module=enc.patch_embed, label="embed")
    pos = enc.trunk.pos_for(idx)
    tr.record("pos", pos, f"rows of the learned positional table ({enc.n_tokens} × {enc.embed_dim}) "
                          "for these positions", params=enc.trunk.pos_embed.numel(), label="positions",
              note="a position is a gene group's identity, not an order")
    x = tr.record("x", x + pos, "add positions", label="+ pos")
    x = run_transformer_blocks(tr, enc.trunk.blocks, x, f"{who}.blocks", expand)
    return tr.record("x", enc.trunk.norm(x), f"final LayerNorm({enc.embed_dim})", module=enc.trunk.norm,
                     label="LayerNorm")


def _masks(geo: Geometry, tr: Trace | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    ctx_1d, tgt_1d = sample_block_masks(geo.n_tokens, geo.n_target)
    if tr is not None:
        tr.section("mask")
        tr.record("ctx_idx", ctx_1d, f"positions the student sees ({geo.n_tokens - geo.n_target} of {geo.n_tokens})",
                  label="", note="one split per batch, shared by every cell")
        tr.record("tgt_idx", tgt_1d, f"positions to predict ({geo.n_target}), drawn at random", label="")
    return batched(ctx_1d, geo.batch), batched(tgt_1d, geo.batch)


def _walk(component_name: str, title: str, opts_shown: dict, geo: Geometry, tr: Trace, **kw) -> Walkthrough:
    return Walkthrough(component=component_name, title=title, options=opts_shown, geometry=geo.as_dict(),
                       steps=tr.steps, **kw)


@component("cell-tokens", group="data", aliases=("tokenizer", "tokens"),
           summary="a cell's gene vector → fixed random gene groups → token tensor")
def cell_tokens(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    hvg = torch.randn(geo.batch, geo.n_hvg).abs()
    partition = make_gene_partition(geo.n_hvg, geo.n_tokens, seed=vocab.partition_seed)
    tr = Trace()
    tr.section("tokenize")
    tr.record("hvg_X", hvg, "log1p-CP10K expression of the highly variable genes, one row per cell", label="")
    tr.record("partition", partition, f"fixed random partition: {geo.n_tokens} groups of {geo.token_dim} gene "
                                      f"indices (seed {vocab.partition_seed}, the same for every cell)", label="",
              note="padding with -1 when the genes do not divide evenly")
    valid = partition >= 0
    gathered = hvg[:, partition.clamp_min(0)]
    if not bool(valid.all()):
        gathered = torch.where(valid.unsqueeze(0), gathered, gathered.new_full((), 0.0))
    tokens = tr.record("tokens", gathered, "gather each group's genes: token k holds the genes of group k",
                       label="gather")
    return _walk("cell-tokens", "how a cell becomes a sequence of tokens", {}, geo, tr,
                 output=tokens, reference=tokenize_cells(hvg, partition), checked="tokens",
                 inputs=f"{geo.batch} cells × {geo.n_hvg} genes (random stand-in values)")


@component("jepa-encoder", group="jepa", aliases=("encoder",),
           summary="tokens → embed → positions → transformer → mean-pool → cell latent z",
           options=("expand",), variants=({}, {"expand": True}), needs=("encoder",))
def jepa_encoder(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    jepa = src.jepa(geo, opts)
    tokens = stand_in_tokens(geo)
    tr = Trace()
    tr.section("encode all tokens")
    tr.record("tokens", tokens, f"{geo.n_tokens} gene-group tokens of {geo.token_dim} genes per cell", label="")
    x = replay_encoder(tr, jepa.encoder, tokens, None, opts.expand, "encoder")
    tr.section("pool")
    z = tr.record("z", x.mean(dim=1), "average over the tokens: one latent per cell", label="mean-pool",
                  note="this is the z the flow and the decoder use (from the student encoder)")
    return _walk("jepa-encoder", "the cell encoder used after pretraining", {"expand": opts.expand}, geo, tr,
                 modules={"encoder": jepa.encoder}, output=z, reference=jepa.embed(tokens), checked="z",
                 inputs=f"{geo.batch} cells (random stand-in tokens)")


@component("student-branch", group="jepa", aliases=("student", "context-encoder"),
           summary="training: the encoder sees only the context tokens", options=("expand",), needs=("encoder",))
def student_branch(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    jepa = src.jepa(geo, opts)
    tokens = stand_in_tokens(geo)
    tr = Trace()
    ctx_idx, _ = _masks(geo, tr)
    tr.section("student encoder")
    tr.record("tokens", tokens, "all tokens of each cell", label="")
    x = replay_encoder(tr, jepa.encoder, tokens, ctx_idx, opts.expand, "encoder")
    tr.record("ctx_embed", x, "context embeddings, passed to the predictor", label="")
    return _walk("student-branch", "the student encoder in a training step", {"expand": opts.expand}, geo, tr,
                 modules={"student encoder": jepa.encoder}, output=x,
                 reference=jepa.encoder(tokens, idx=ctx_idx), checked="ctx_embed",
                 inputs=f"{geo.batch} cells (random stand-in tokens)")


@component("teacher-branch", group="jepa", aliases=("teacher", "target-encoder", "ema"),
           summary="training: the EMA teacher encodes all tokens, then keeps the targets",
           options=("expand",), needs=("encoder",))
def teacher_branch(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    jepa = src.jepa(geo, opts, teacher=True)
    teacher = jepa.ema.teacher
    tokens = stand_in_tokens(geo)
    tr = Trace()
    _, tgt_idx = _masks(geo, tr)
    tr.section("teacher encoder (no gradient)")
    tr.record("tokens", tokens, "all tokens of each cell, nothing hidden", label="")
    full = replay_encoder(tr, teacher, tokens, None, opts.expand, "teacher")
    tr.section("select targets")
    gather = tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
    target = tr.record("target", torch.gather(full, 1, gather),
                       f"keep the {geo.n_target} target positions", label="gather",
                       note="each target saw every token, so it carries the cell's full context")
    tr.note(f"after each optimizer step: teacher ← m·teacher + (1 − m)·student, "
            f"m rising from {jepa.cfg.ema_base} to {jepa.cfg.ema_final} on a cosine schedule")
    ref = torch.gather(teacher(tokens), 1, gather)
    return _walk("teacher-branch", "the EMA teacher in a training step", {"expand": opts.expand}, geo, tr,
                 modules={"teacher encoder": teacher}, frozen=("teacher encoder",), output=target,
                 reference=ref, checked="target", inputs=f"{geo.batch} cells (random stand-in tokens)")


@component("predictor", group="jepa", aliases=("jepa-predictor",),
           summary="training: fill in the target positions from the context embeddings",
           options=("expand",), needs=("encoder",))
def predictor(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    jepa = src.jepa(geo, opts)
    pred = jepa.predictor
    tr = Trace()
    ctx_idx, tgt_idx = _masks(geo, tr)
    n_ctx = ctx_idx.shape[1]
    ctx_embed = torch.randn(geo.batch, n_ctx, geo.embed_dim)
    tr.section("assemble")
    tr.record("ctx_embed", ctx_embed, "the student's context embeddings (stand-in values)", label="")
    ctx = tr.record("ctx", ctx_embed + pred.trunk.pos_for(ctx_idx),
                    "add the predictor's own positional table at the context positions",
                    params=pred.trunk.pos_embed.numel(), label="+ pos")
    mask = tr.record("mask", pred.mask_token.expand(geo.batch, geo.n_target, -1) + pred.trunk.pos_for(tgt_idx),
                     "one learned mask vector, copied to each target, plus the target positions",
                     params=pred.mask_token.numel(), label="mask + pos",
                     note="the only thing distinguishing two targets is their position")
    x = tr.record("x", torch.cat([ctx, mask], dim=1), "concatenate context and mask tokens", label="concat")
    tr.section("predict")
    x = run_transformer_blocks(tr, pred.trunk.blocks, x, "predictor.blocks", opts.expand)
    x = tr.record("x", pred.trunk.norm(x), f"final LayerNorm({geo.embed_dim})", module=pred.trunk.norm,
                  label="LayerNorm")
    out = tr.record("pred", x[:, n_ctx:n_ctx + geo.n_target, :].contiguous(),
                    f"read out the last {geo.n_target} slots: the predicted target embeddings", label="slice")
    return _walk("predictor", "the JEPA predictor in a training step", {"expand": opts.expand}, geo, tr,
                 modules={"predictor": pred}, output=out, reference=pred(ctx_embed, ctx_idx, tgt_idx),
                 checked="pred", inputs=f"{geo.batch} cells (random stand-in context embeddings)")


@component("jepa-step", group="jepa", aliases=("jepa", "jepa-training"),
           summary="one training step: student, predictor, teacher, prediction loss + VICReg",
           options=("reg_coef",), variants=({}, {"reg_coef": 0.0}), needs=("encoder",))
def jepa_step(geo: Geometry, vocab: Vocab, opts: Options, src) -> Walkthrough:
    jepa = src.jepa(geo, opts, teacher=True)
    cfg = jepa.cfg
    tokens = stand_in_tokens(geo)
    tr = Trace()
    with rng_checkpoint() as rng:
        ctx_idx, tgt_idx = _masks(geo, tr)
        tr.section("three branches")
        ctx_embed = tr.record("ctx_embed", jepa.encoder(tokens, idx=ctx_idx),
                              "student encoder on the context tokens (see student-branch)",
                              module=jepa.encoder, label="student")
        pred = tr.record("pred", jepa.predictor(ctx_embed, ctx_idx, tgt_idx),
                         "predictor fills in the targets (see predictor)", module=jepa.predictor, label="predictor")
        full = jepa.ema.teacher(tokens)
        gather = tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
        target = tr.record("target", torch.gather(full, 1, gather),
                           "teacher on all tokens, targets kept (see teacher-branch); no gradient",
                           module=jepa.ema.teacher, label="teacher")
        tr.section("loss")
        loss = tr.record("pred_loss", prediction_loss(pred, target, kind=cfg.pred_kind),
                         f"{cfg.pred_kind} between pred and target, averaged", label=cfg.pred_kind)
        if cfg.reg_coef > 0:
            flat = ctx_embed.reshape(-1, ctx_embed.shape[-1])
            tr.record("z_ctx", flat, "context embeddings flattened to rows for VICReg", label="flatten")
            reg, _ = vicreg_reg(flat, var_coef=cfg.var_coef, cov_coef=cfg.cov_coef)
            tr.record("vicreg", reg, f"{cfg.var_coef:g}·variance hinge + {cfg.cov_coef:g}·off-diagonal covariance",
                      label="VICReg", note="keeps the embeddings from collapsing to a constant")
            loss = tr.record("loss", loss + cfg.reg_coef * reg, f"pred_loss + {cfg.reg_coef:g}·vicreg", label="total")
        tr.note("backpropagate into the student and predictor only; then update the teacher by EMA")
        rng.rewind()
        ref, _ = jepa(tokens)
    return _walk("jepa-step", "one Stage A training step", {"reg_coef": cfg.reg_coef}, geo, tr,
                 modules={"student encoder": jepa.encoder, "predictor": jepa.predictor,
                          "teacher encoder": jepa.ema.teacher},
                 frozen=("teacher encoder",), output=loss, reference=ref, checked="the loss",
                 inputs=f"{geo.batch} cells (random stand-in tokens)")
