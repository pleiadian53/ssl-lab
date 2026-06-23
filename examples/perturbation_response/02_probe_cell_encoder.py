"""Stage A check — is the frozen cell encoder any good?

Loads the pretrained JEPA encoder, extracts one pooled latent ``z`` per cell, and
asks two questions:

  1. Does ``z`` linearly separate perturbations? A logistic-regression probe
     predicts the perturbation label from ``z`` (chance ~ 1 / n_perturbations).
  2. Has the representation collapsed? Effective rank and per-dimension feature
     std on the latents (effective rank must stay well above 1).

This is the encoder-quality gate before Stage B/C. It is a *representation* check,
not the effect-size benchmark — that comes once the decoder and flow exist.

Note on the split: we probe on the **cells** split (random per-cell hold-out, so
the same perturbation vocabulary appears in train and test, making the multiclass
probe well-posed). Pretraining (01) uses the **combo** split to avoid leaking
held-out combinations; the two are independent.

Usage
-----
    python examples/perturbation_response/02_probe_cell_encoder.py
    python examples/perturbation_response/02_probe_cell_encoder.py --train-limit 8000 --test-limit 4000

Output
------
    output/<experiment>/reports/stage_a_probe.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.eval.collapse import collapse_report
from ssllab.eval.probe import linear_probe
from ssllab.experiment import experiment
from ssllab.extract import extract_latents
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe the pretrained cell encoder (perturbation label + collapse).")
    p.add_argument("--experiment", type=str, default="norman_stage_a")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--encoder", type=str, default=None, help="override encoder path")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--split", type=str, default="cells", choices=["cells", "combo"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-limit", type=int, default=8000, help="cap probe train cells")
    p.add_argument("--test-limit", type=int, default=4000, help="cap probe test cells")
    p.add_argument("--limit", type=int, default=None, help="cap loader train cells (passed through)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _prepare(batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["tokens"].to(device), batch["pert_id"]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()
    encoder_path = args.encoder or (exp.checkpoints / "encoder.pt")

    jepa = load_jepa(encoder_path, device)

    train_loader, val_loader, test_loader = get_perturbseq_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir, artifact=args.artifact,
        split=args.split, limit=args.limit, seed=0,
    )
    eval_loader = test_loader if len(test_loader.dataset) else val_loader
    n_perts = len(train_loader.meta["pert_names"])

    Ztr, Ytr = extract_latents(jepa.embed, train_loader, device, limit=args.train_limit, prepare=_prepare)
    Zte, Yte = extract_latents(jepa.embed, eval_loader, device, limit=args.test_limit, prepare=_prepare)
    logger.info("extracted latents: train %s  test %s  (%d perturbations)", tuple(Ztr.shape), tuple(Zte.shape), n_perts)

    scores = linear_probe(Ztr, Ytr, Zte, Yte)
    rep = collapse_report(Ztr)
    chance = 1.0 / max(n_perts, 1)
    logger.info(
        "probe pert-id  train_acc=%.4f  test_acc=%.4f  (chance=%.4f)  |  eff_rank=%.2f/%d  feat_std=%.3f",
        scores["train_acc"], scores["test_acc"], chance,
        rep["effective_rank"], rep["embed_dim"], rep["feature_std"],
    )

    report = exp.write_report("stage_a_probe", {
        "experiment": args.experiment,
        "artifact": args.artifact,
        "split": args.split,
        "n_perturbations": n_perts,
        "chance": chance,
        "probe": scores,
        **rep,
    })
    logger.info("wrote report -> %s", report)


if __name__ == "__main__":
    main()
