"""Span-masked JEPA pretraining on PTB-XL training folds (study v1, D17/D21).

Inductive: only training-fold recordings (folds 1-8, all of them, unknown-rhythm
ones included; labels unused). Validation and test never enter pretraining.
One run per repeat, seeded by ``pretrain_seed = r``; its checkpoint is reused
across that repeat's budgets and by both pretrained arms.

Each step hides ``n_spans`` contiguous spans of ``span_s`` seconds (4 tokens of
250 ms each by default) and predicts the EMA teacher's embeddings there.
Collapse diagnostics (feature std, effective rank) are logged on a fixed sample
of training recordings.

Not resumable. The checkpoint stores the student encoder and predictor only (no
EMA teacher, optimiser state, step counter or RNG state), so an interrupted run
restarts from scratch. The checkpoint and report are written atomically at the
end; a finished, compatible run is skipped unless ``--force``.

Usage
-----
    python examples/ecg_rhythm/03_pretrain_jepa.py --repeat 0
    python examples/ecg_rhythm/03_pretrain_jepa.py --repeat 0 --epochs 1 --limit 512   # smoke

Output
------
    output/<experiment>/checkpoints/jepa_r<r>.pt
    output/<experiment>/reports/pretrain_r<r>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import _study as S
from ssllab.checkpoint import save_jepa
from ssllab.data.ptbxl import load_cache
from ssllab.data.signal import SignalBatcher, SignalGeometry
from ssllab.eval.collapse import collapse_report
from ssllab.jepa.model import build_jepa
from ssllab.utils.reproducibility import set_seed

logger = logging.getLogger("03_pretrain_jepa")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--study", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None, help="override the study's epochs (smoke only)")
    p.add_argument("--limit", type=int, default=None, help="first N training recordings (smoke only)")
    p.add_argument("--force", action="store_true", help="retrain even if a complete, compatible run exists")
    args = p.parse_args()

    study = S.load_study(args.study)
    ok, why = S.pretrain_complete(study, args.output_root, args.experiment, args.repeat)
    if ok and not args.force:   # never overwrite a finished run implicitly, smoke flags included
        logger.info("repeat %d already pretrained and compatible; skipping (pass --force to retrain)", args.repeat)
        return
    logger.info("repeat %d: %s -> training from scratch (pretraining is not resumable)", args.repeat, why)
    cfg, enc = study["pretrain"], study["encoder"]
    epochs = args.epochs or cfg["epochs"]
    sd = S.seeds(args.repeat)
    set_seed(sd["pretrain_seed"])
    device = S.resolve_device(args.device)

    cache = load_cache(args.data_dir / study["data"]["artifact"])
    geom = SignalGeometry(fs=study["data"]["fs"], duration_s=study["data"]["duration_s"],
                          patch_s=study["data"]["patch_s"], span_s=cfg["span_s"])
    rows = S.cohort_rows(cache, "train", np.ones(cache.n, dtype=bool))  # all training recordings
    if args.limit:
        rows = rows[: args.limit]
    batcher = SignalBatcher(cache.signals, rows, geom.patch_len, device=device)
    n_leads = cache.signals.shape[1]

    jepa = build_jepa(
        token_dim=geom.token_dim(n_leads), n_tokens=geom.n_tokens, embed_dim=enc["embed_dim"],
        enc_depth=enc["depth"], pred_depth=cfg["pred_depth"], n_heads=enc["n_heads"],
        n_target=geom.span_tokens * cfg["n_spans"], mask_kind=cfg["mask_kind"],
        span_len=geom.span_tokens, n_spans=cfg["n_spans"], reg_coef=cfg["reg_coef"],
    ).to(device)
    jepa.ema.to(device)
    params = [p for p in list(jepa.encoder.parameters()) + list(jepa.predictor.parameters())]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, len(batcher) // cfg["batch_size"])
    total = epochs * steps_per_epoch
    warm = min(cfg["warmup_epochs"] * steps_per_epoch, max(1, total // 10))
    lr_at = lambda s: cfg["lr"] * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))  # noqa: E731

    g = torch.Generator().manual_seed(sd["pretrain_seed"])
    probe_idx = torch.randperm(len(batcher), generator=torch.Generator().manual_seed(12345))[:2048].to(batcher.x.device)

    def collapse() -> dict:
        jepa.eval()
        with torch.no_grad():
            z = torch.cat([jepa.embed(batcher.batch(probe_idx[i:i + 256])[0]) for i in range(0, len(probe_idx), 256)])
        jepa.train()
        return collapse_report(z)

    logger.info("pretrain r=%d: %d recordings, %d epochs x %d steps, tokens (%d, %d), span %d tokens x %d, device=%s",
                args.repeat, len(batcher), epochs, steps_per_epoch, geom.n_tokens, geom.token_dim(n_leads),
                geom.span_tokens, cfg["n_spans"], device)
    history, step, t0 = [], 0, time.time()
    jepa.train()
    for ep in range(epochs):
        losses = []
        for _, tok, _ in batcher.iterate(cfg["batch_size"], shuffle=True, generator=g, drop_last=True):
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            loss, comp = jepa(tok)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            jepa.update_target(step, total)
            losses.append(float(loss))
            step += 1
        rec = {"epoch": ep + 1, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == epochs:
            rec["collapse"] = collapse()
            logger.info("epoch %d/%d loss %.4f  eff_rank %.1f  std %.3f  (%.0fs)", ep + 1, epochs, rec["loss"],
                        rec["collapse"]["effective_rank"], rec["collapse"]["feature_std"], rec["seconds"])
        history.append(rec)

    wall = time.time() - t0
    # Atomic: an interrupted run leaves no checkpoint that could be mistaken for a finished one.
    ck = S.checkpoint_path(args.output_root, args.experiment, args.repeat)
    tmp = save_jepa(jepa, ck.with_suffix(".pt.tmp"))
    os.replace(tmp, ck)
    report = {
        **S.provenance(study), "repeat": args.repeat, "seeds": sd,
        "n_recordings": int(len(batcher)), "epochs": epochs, "steps": step, "geometry": geom.describe(n_leads),
        "config": cfg, "encoder": enc, "device": str(device), "gpu": S.gpu_name(),
        "wall_seconds": round(wall, 1), "seconds_per_epoch": round(wall / epochs, 2),
        "final_collapse": history[-1].get("collapse"), "history": history, "checkpoint": str(ck),
        "smoke": bool(args.limit or args.epochs),
    }
    rp = S.study_root(args.output_root, args.experiment) / "reports" / f"pretrain_r{args.repeat}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.with_suffix(".json.tmp").write_text(json.dumps(report, indent=2))
    os.replace(rp.with_suffix(".json.tmp"), rp)   # the report is the completion marker, written last
    logger.info("done in %.0fs; checkpoint %s; report %s", wall, ck, rp)


if __name__ == "__main__":
    main()
