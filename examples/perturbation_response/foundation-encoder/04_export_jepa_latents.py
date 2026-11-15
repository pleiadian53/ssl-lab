#!/usr/bin/env python3
"""Export our JEPA encoder's latents for every processed cell as an embedding store.

**Why.** The Geneformer comparison reads every encoder's vectors through the same code path: an
embedding store (``ssllab.embeddings``), looked up by cache row. For the comparison to be fair, our
own JEPA encoder has to reach the count decoder and the flow by that same path, not by the
on-the-fly encoding every earlier round used. This script writes that store: the frozen
``norman_stage_a`` encoder applied to all 109,737 processed cells, one 256-number vector each.

**The space key.** The store's ``space_key`` is the hash of the encoder's weights
(``ssllab.embeddings.jepa_space_key``), the same key the live encoder reports. A decoder trained on
the store can therefore be paired with a flow trained on the live encoder and the other way round,
because they read the same vectors; a decoder trained on a Geneformer store cannot.

**The check.** After writing, a sample of rows (``--verify-rows``, the seed-0 sample of 2,000 the
feasibility run used) is encoded again on the fly with ``ssllab.latents.encode_rows`` and compared
with the store. The two must agree within ``--tol`` (1e-5 by default): any larger gap means the
export changed the vectors, and every comparison built on the store would be in doubt. The
comparison's second interface gate, scoring the published round-11 decoders on these stored
latents, is run by ``14_ceiling_analysis.py --latents``.

Run it on the CPU, on a laptop or on the pod. On the CPU the result does not depend on how the
cells are batched, and the check passes with a gap of exactly 0. On an A40 GPU the same encoder,
fed the same cells in batches of 1,024 and then of 256, gave latents up to 0.0016 apart, because
the GPU's matrix kernels change their summation order with the batch shape; a store made there
would fail the check for that reason alone. The CPU export reads the 2.2 GB expression matrix and
encodes every cell in about five minutes. Nothing here touches Geneformer.

Usage (from the repo root):
    python examples/perturbation_response/foundation-encoder/04_export_jepa_latents.py \\
        --out output/embeddings/norman_jepa_stage_a
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import DEFAULT_N_TOKENS, TOKENS_NPZ, cache_fingerprint_from_dir, make_gene_partition
from ssllab.embeddings import file_md5, jepa_space_key, provenance, write_store
from ssllab.latents import encode_rows
from ssllab.utils import get_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifact", default="norman2019")
    ap.add_argument("--encoder", default="output/norman_stage_a/checkpoints/encoder.pt")
    ap.add_argument("--out", default="output/embeddings/norman_jepa_stage_a")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--verify-rows", type=int, default=2000, help="rows re-encoded on the fly and compared")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    cache_dir = Path(args.data_dir) / args.artifact
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    hvg = np.load(cache_dir / TOKENS_NPZ, allow_pickle=True)["hvg_X"]   # uncompressed member, 2.2 GB
    tok = manifest.get("tokenization", {})
    partition = make_gene_partition(hvg.shape[1], int(tok.get("n_tokens", DEFAULT_N_TOKENS)),
                                    int(tok.get("partition_seed", 0)))
    jepa = load_jepa(args.encoder, device)
    rows = np.arange(hvg.shape[0], dtype=np.int64)

    t0 = time.time()
    Z = encode_rows(jepa, hvg, partition, rows, device, args.batch_size).numpy()
    seconds = time.time() - t0
    store = write_store(
        args.out, Z, rows,
        source="jepa",
        dataset_fingerprint=cache_fingerprint_from_dir(cache_dir),
        space_key=jepa_space_key(jepa),
        model="JEPA stage A (norman_stage_a)",
        mode="mean over the 50 output tokens (embed_pooled)",
        encoder=str(args.encoder),
        encoder_md5=file_md5(args.encoder),
        device=str(device),
        encode_seconds=round(seconds, 1),
        **provenance(__file__),
    )

    # Re-encode a sample on the fly, in a different batch layout, and compare with the store.
    rng = np.random.default_rng(0)
    check = np.sort(rng.choice(rows, min(args.verify_rows, len(rows)), replace=False))
    fresh = encode_rows(jepa, hvg, partition, check, device, 256).numpy()
    gap = float(np.abs(store.lookup(check) - fresh).max())
    report = {"store": str(store.path), "n": store.n, "width": store.width, "space_key": store.space_key,
              "content_md5": store.manifest["content_md5"], "encode_seconds": round(seconds, 1),
              "verify_rows": int(len(check)), "max_abs_gap": gap, "tol": args.tol, "passed": gap <= args.tol}
    (store.path / "export_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if gap > args.tol:
        sys.exit(f"EXPORT CHECK FAILED: stored and on-the-fly latents differ by {gap:.3g} > {args.tol:g}")


if __name__ == "__main__":
    main()
