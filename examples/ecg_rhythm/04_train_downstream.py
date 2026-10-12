"""Train one downstream run of study v1: an arm, a repeat, a budget (D16-D20).

Arms share the encoder architecture, head, loss, optimiser and step budget:

* ``scratch``  : encoder randomly initialised (``train_seed``), all weights trained
* ``finetune`` : encoder from the repeat's JEPA checkpoint, all weights trained
* ``probe``    : encoder from the repeat's JEPA checkpoint, frozen; head trained

The training recordings are the eligible (rhythm-annotated) recordings of the
patients saved for this (repeat, budget) by ``02_make_subsets.py``. Each recording
keeps its own labels. Labels with zero positive patients in the subset are masked
out of the loss and listed as unsupported. Model selection uses the validation
core macro AUROC; the best step is kept. Test predictions are written only with
``--eval-test`` (the pilot never produces them).

``--shuffle-labels`` permutes whole label vectors across the training recordings
(prevalence and co-occurrence preserved, link to the signal broken): a sanity
check that should land near 0.5 AUROC, not a proof that no leakage exists.

Usage
-----
    python examples/ecg_rhythm/04_train_downstream.py --arm finetune --repeat 0 --budget 0.1 --lr 3e-4
    python examples/ecg_rhythm/04_train_downstream.py --arm scratch --repeat 0 --budget 0.1 --lr 1e-3 --shuffle-labels

Output
------
    output/<experiment>/runs/<tag>/<arm>_b<budget>_r<r>_lr<lr>/{report.json, preds.npz}
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import _study as S
from ssllab.checkpoint import load_jepa
from ssllab.data.ptbxl import load_cache
from ssllab.data.signal import SignalBatcher, SignalGeometry
from ssllab.eval.classification import macro_auroc, per_label_metrics
from ssllab.models.classifier import EncoderClassifier
from ssllab.models.encoder import JEPAEncoder
from ssllab.utils.reproducibility import set_seed

logger = logging.getLogger("04_train_downstream")
ARMS = ("scratch", "finetune", "probe")


@torch.no_grad()
def predict(model, batcher: SignalBatcher, bs: int = 512) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(tok)).float().cpu() for _, tok, _ in batcher.iterate(bs, shuffle=False)]
    model.train()
    return torch.cat(out).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--tag", default="main", help="run group, e.g. pilot or main")
    p.add_argument("--shuffle-labels", action="store_true")
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--study", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-steps", type=int, default=None, help="override (smoke only)")
    p.add_argument("--eval-every", type=int, default=None, help="override (smoke only)")
    p.add_argument("--skip-if-complete", action="store_true",
                   help="skip when this run already finished under the same base protocol (grid restarts)")
    args = p.parse_args()

    study = S.load_study(args.study)
    ds, enc_cfg = study["downstream"], study["encoder"]
    max_steps = args.max_steps or ds["max_steps"]
    eval_every = args.eval_every or ds["eval_every"]
    sd = S.seeds(args.repeat)
    set_seed(sd["train_seed"])
    device = S.resolve_device(args.device)

    out = S.run_dir(args.output_root, args.experiment, args.tag, args.arm + ("_shuffled" if args.shuffle_labels else ""),
                    args.budget, args.repeat, args.lr)
    if args.skip_if_complete and (out / "report.json").exists():
        prev = json.loads((out / "report.json").read_text())
        why = S.check_compatible(prev, study, "run", str(out))   # stops on an unexplained incompatibility
        if (prev["eval_test"] or not args.eval_test) and not prev["smoke"]:
            logger.info("complete: %s; skipping (%s)", out, why)
            return

    # --- data: the saved subset, never recomputed -------------------------------------
    subsets = S.load_subsets(study, args.output_root, args.experiment)
    reused = {"subsets": {"path": str(S.subsets_path(args.output_root, args.experiment)), "reason": subsets["_reuse_reason"]}}
    sub = subsets["repeats"][str(args.repeat)][S.budget_key(args.budget)]
    cache = load_cache(args.data_dir / study["data"]["artifact"])
    rl = cache.rhythm()
    names, core = list(rl.codes), study["labels"]["core"]
    core_idx = [names.index(c) for c in core]
    geom = SignalGeometry(fs=study["data"]["fs"], duration_s=study["data"]["duration_s"], patch_s=study["data"]["patch_s"])
    n_leads = cache.signals.shape[1]

    tr_rows = S.cohort_rows(cache, "train", rl.has_rhythm, sub["patients"])
    if len(tr_rows) != sub["n_records_eligible"]:
        raise SystemExit(f"eligible training recordings {len(tr_rows)} != saved {sub['n_records_eligible']}")
    Ytr = rl.y[tr_rows].astype(np.float32)
    supported = np.array([sub["support"][c]["supported"] for c in names])
    if args.shuffle_labels:
        Ytr = Ytr[np.random.default_rng(sd["train_seed"] + 999).permutation(len(Ytr))]
    va_rows = S.cohort_rows(cache, "val", rl.has_rhythm)
    train_b = SignalBatcher(cache.signals, tr_rows, geom.patch_len, device=device, labels=Ytr)
    val_b = SignalBatcher(cache.signals, va_rows, geom.patch_len, device=device)
    Yva = rl.y[va_rows]

    # --- model -------------------------------------------------------------------------
    if args.arm == "scratch":
        encoder = JEPAEncoder(token_dim=geom.token_dim(n_leads), embed_dim=enc_cfg["embed_dim"], depth=enc_cfg["depth"],
                              n_heads=enc_cfg["n_heads"], n_tokens=geom.n_tokens)
        init = "random"
    else:
        ok, pre_why = S.pretrain_complete(study, args.output_root, args.experiment, args.repeat)
        if not ok and not (args.max_steps or args.eval_every):
            raise SystemExit(f"repeat {args.repeat} pretraining unusable: {pre_why}")
        ck = S.checkpoint_path(args.output_root, args.experiment, args.repeat)
        encoder = copy.deepcopy(load_jepa(ck, device="cpu").encoder)
        for prm in encoder.parameters():
            prm.requires_grad_(True)
        init = str(ck)
        reused["pretrain_checkpoint"] = {"path": str(ck), "reason": pre_why}
    torch.manual_seed(sd["train_seed"])
    model = EncoderClassifier(encoder, n_out=len(names), freeze_encoder=args.arm == "probe").to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=ds["weight_decay"])
    warm = min(100, max(1, max_steps // 10))
    lr_at = lambda s: args.lr * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm))))  # noqa: E731
    mask = torch.as_tensor(supported, dtype=torch.float32, device=device)

    # --- train with validation-based selection ------------------------------------------
    logger.info("%s r=%d budget=%g lr=%g: %d train recordings (%d patients), %d val; unsupported=%s%s",
                args.arm, args.repeat, args.budget, args.lr, len(tr_rows), sub["n_patients_eligible"], len(va_rows),
                sub["unsupported_labels"], " [SHUFFLED LABELS]" if args.shuffle_labels else "")
    g = torch.Generator().manual_seed(sd["train_seed"])
    best = {"score": -1.0, "step": 0, "state": None}
    curve, step, bad, t0 = [], 0, 0, time.time()
    model.train()
    while step < max_steps and bad < ds["patience_evals"]:
        for _, tok, y in train_b.iterate(ds["batch_size"], shuffle=True, generator=g, drop_last=True):
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            loss = (F.binary_cross_entropy_with_logits(model(tok), y, reduction="none") * mask).sum(1).mean() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % eval_every == 0 or step == max_steps:
                score, complete = macro_auroc(Yva, predict(model, val_b), core_idx)
                curve.append({"step": step, "train_loss": float(loss), "val_core_macro_auroc": score})
                if complete and score > best["score"]:
                    best = {"score": score, "step": step, "state": copy.deepcopy(model.state_dict())}
                    bad = 0
                else:
                    bad += 1
            if step >= max_steps or bad >= ds["patience_evals"]:
                break
    train_seconds = time.time() - t0
    if best["state"] is None:
        raise SystemExit("no evaluation produced a complete validation core macro AUROC")
    model.load_state_dict(best["state"])

    # --- predictions + report -------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    Sva = predict(model, val_b)
    preds = {"labels": np.array(names), "val_rows": va_rows, "val_scores": Sva}
    if args.eval_test:
        te_rows = S.cohort_rows(cache, "test", rl.has_rhythm)
        preds |= {"test_rows": te_rows, "test_scores": predict(model, SignalBatcher(cache.signals, te_rows, geom.patch_len, device=device))}
    np.savez_compressed(out / "preds.npz", **preds)
    val_macro, val_complete = macro_auroc(Yva, Sva, core_idx)
    report = {
        **S.provenance(study), "reused": reused, "arm": args.arm, "repeat": args.repeat,
        "budget": args.budget, "lr": args.lr, "seeds": sd, "init": init, "tag": args.tag,
        "shuffle_labels": args.shuffle_labels, "eval_test": args.eval_test,
        "n_train_records": int(len(tr_rows)), "n_train_patients": sub["n_patients_eligible"],
        "n_val_records": int(len(va_rows)), "unsupported_labels": sub["unsupported_labels"],
        "best_step": best["step"], "stopped_step": step, "max_steps": max_steps,
        "best_at_cap": best["step"] >= max_steps - eval_every,
        "val_core_macro_auroc": val_macro, "val_core_complete": val_complete,
        "val_per_label": per_label_metrics(Yva, Sva, names),
        "train_seconds": round(train_seconds, 1), "steps_per_second": round(step / max(train_seconds, 1e-9), 1),
        "device": str(device), "gpu": S.gpu_name(), "curve": curve,
        "smoke": bool(args.max_steps or args.eval_every),
    }
    (out / "report.json.tmp").write_text(json.dumps(report, indent=2))
    (out / "report.json.tmp").replace(out / "report.json")   # the report is the completion marker, written last
    logger.info("best val core macro AUROC %.4f at step %d (stopped %d, %.0fs) -> %s",
                val_macro, best["step"], step, train_seconds, out)


if __name__ == "__main__":
    main()
