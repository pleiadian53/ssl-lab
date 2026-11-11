"""Co-adaptation, Phase 1: train the cell encoder and the count decoder together.

The ceiling analysis found that the frozen JEPA latents carry more of the perturbation effect than
the trained negative-binomial count decoder recovers from them: a linear readout of the same
latents scores well above the decoder. This script trains a fresh count decoder in one of three
modes that differ only in what is allowed to train (see :mod:`ssllab.coadapt`):

    dec            the encoder stays frozen; only the decoder trains (the control)
    joint          the encoder, initialized from its JEPA checkpoint, trains with the decoder
    joint_anchor   joint, maintaining the JEPA predictive objective during reconstruction fine-tuning

``--init random`` (joint only) starts the encoder from freshly initialized weights of the same
architecture instead of the JEPA checkpoint, which is then read only for its config: the control
that separates what pretraining contributes from what joint training does on its own. Everything
drawn after the encoder is built, the decoder's initialization included, is the same as under
``--init pretrained`` at the same seed (see ``ssllab.checkpoint.random_jepa_like``).

Everything else is the same recipe in every mode: the combo split's training cells, the decoder's
negative-binomial loss on raw counts with each cell's library size (as ``03_train_count_decoder.py``
uses), a fresh decoder of the default architecture, AdamW, gradient clipping. The transition
(the conditional flow) is not involved.

Reconstruction discourages complete information loss, but rank, reconstruction quality and
task-relevant information are still monitored: the report holds the per-epoch NB loss and the
effective rank and feature spread of the cell vectors; the linear readout and the perturbation
probe are run on the saved encoder by ``14_ceiling_analysis.py`` and ``02_probe_cell_encoder.py``.

Under ``joint_anchor`` the anchor weight is fixed by rule before training, so the two terms are
equal at a trained readout's scale: ``w = mean NB loss of --ref-decoder / mean JEPA loss`` on the
first ``--anchor-batches`` training batches, measured with the loaded encoder before any update. The
weight is then fixed. Which parameters train in each mode, and why the rebuilt EMA teacher starts
a new fine-tuning run rather than restoring Stage A's, are set out in :mod:`ssllab.coadapt`. There
is no resume: an interrupted run is restarted from scratch.

Outputs, in the formats every downstream script already reads:

    output/<experiment>/checkpoints/encoder.pt         the (co-adapted or unchanged) encoder, save_jepa format
    output/<experiment>/checkpoints/count_decoder.pt   the decoder, the format of 03_train_count_decoder.py
    output/<experiment>/reports/coadapt.json           config, anchor rule, per-epoch history

Usage
-----
    python examples/perturbation_response/co-adaptation/02_coadapt_decoder.py \\
        --experiment norman_coadapt_joint_s0 --mode joint --seed 0
    # smoke
    python examples/perturbation_response/co-adaptation/02_coadapt_decoder.py \\
        --experiment smoke_coadapt --mode joint_anchor --epochs 1 --limit 2000 --anchor-batches 2
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa, random_jepa_like, save_jepa
from ssllab.coadapt import CoadaptConfig, anchor_weight_by_rule, config_dict, prepare_jepa, train_coadapt
from ssllab.data.perturbseq import get_perturbseq_dataloaders
from ssllab.experiment import experiment
from ssllab.generative.count_decoder import CountDecoder
from ssllab.generative.perturb import load_count_decoder
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Co-adaptation Phase 1: encoder + count decoder.")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["dec", "joint", "joint_anchor"])
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--encoder", type=str, default="output/norman_stage_a/checkpoints/encoder.pt",
                   help="the JEPA checkpoint the encoder starts from")
    p.add_argument("--ref-decoder", type=str, default="output/norman_r11_dec_hvg_s0/checkpoints/count_decoder.pt",
                   help="a trained decoder that sets the anchor weight's scale (joint_anchor only); it must "
                        "have been trained with the run's --library-size")
    p.add_argument("--init", type=str, default="pretrained", choices=["pretrained", "random"],
                   help="encoder start: the --encoder checkpoint's weights, or fresh weights of its "
                        "architecture (joint only)")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3, help="decoder learning rate")
    p.add_argument("--encoder-lr", type=float, default=1e-4, help="encoder (and predictor) learning rate")
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--anchor-batches", type=int, default=20)
    p.add_argument("--library-size", type=str, default="hvg", choices=["all", "hvg"],
                   help="total the NB likelihood multiplies the rate by: the modeled genes (default since "
                        "round 11) or all measured genes (reproduces earlier runs)")
    p.add_argument("--limit", type=int, default=None, help="cap training cells (smoke only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()
    enc_path = Path(args.encoder)

    if args.init == "random" and args.mode != "joint":
        raise SystemExit("--init random is only defined for --mode joint")
    start = load_jepa(enc_path, device) if args.init == "pretrained" else random_jepa_like(enc_path, device)
    jepa = prepare_jepa(start, args.mode, device)
    train_loader, _, _ = get_perturbseq_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir, artifact=args.artifact,
        split=args.split, limit=args.limit, seed=0,
    )
    n_genes = train_loader.meta["n_hvg"]
    decoder = CountDecoder(latent_dim=jepa.cfg.embed_dim, n_genes=n_genes).to(device)
    cfg = CoadaptConfig(mode=args.mode, epochs=args.epochs, lr=args.lr, encoder_lr=args.encoder_lr,
                        clip=args.clip, anchor_batches=args.anchor_batches, library_size=args.library_size)
    logger.info("mode=%s  init=%s  encoder %s (md5 %s)  genes=%d  train cells=%d  device=%s",
                args.mode, args.init, enc_path, _md5(enc_path)[:8], n_genes, len(train_loader.dataset), device)

    rule = None
    if args.mode == "joint_anchor":
        ref = load_count_decoder(Path(args.ref_decoder), device)
        batches = [b for _, b in zip(range(args.anchor_batches), train_loader)]
        if getattr(ref, "library_size", "all") != args.library_size:
            raise SystemExit(f"--ref-decoder was trained with library size {getattr(ref, 'library_size', 'all')!r}, "
                             f"but this run uses {args.library_size!r}")
        rule = anchor_weight_by_rule(jepa, ref, batches, device, args.library_size)
        rule["ref_decoder"] = str(args.ref_decoder)
        rule["ref_decoder_md5"] = _md5(Path(args.ref_decoder))
        cfg.anchor_weight = rule["anchor_weight"]
        logger.info("anchor weight by rule: %.4g = ref NB %.4g / JEPA %.4g over %d batches",
                    rule["anchor_weight"], rule["ref_nb_loss"], rule["jepa_loss"], rule["n_batches"])
        del ref

    history = train_coadapt(jepa, decoder, train_loader, cfg, device, log=logger.info)

    jepa.eval()
    enc_out = save_jepa(jepa, exp.checkpoints / "encoder.pt")
    dec_out = exp.checkpoints / "count_decoder.pt"
    torch.save({"state": decoder.state_dict(), "latent_dim": jepa.cfg.embed_dim, "n_genes": n_genes,
                "zinb": False, "anchored_mean": False, "state_dispersion": False,
                "library_size": args.library_size}, dec_out)
    exp.write_report("coadapt", {
        "mode": args.mode, "init": args.init, "seed": args.seed, "split": args.split, "library_size": args.library_size,
        "train_cells": len(train_loader.dataset), "limit": args.limit,
        "init_encoder": str(enc_path), "init_encoder_md5": _md5(enc_path),
        "encoder_md5": _md5(enc_out), "decoder_md5": _md5(dec_out),
        "config": config_dict(cfg), "anchor_rule": rule, "history": history,
        "final_nb_loss": history[-1]["nb_loss"],
    })
    logger.info("saved encoder -> %s ; decoder -> %s", enc_out, dec_out)


if __name__ == "__main__":
    main()
