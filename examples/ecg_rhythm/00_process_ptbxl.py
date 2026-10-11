"""Process PTB-XL into the ssl-lab cache (Session 1).

Reads the PhysioNet PTB-XL tree (two CSV tables plus WFDB records), verifies every
record's per-lead checksum, and writes the cache that ``ssllab.data.ptbxl``
loads at train time. No extra dependencies: the WFDB reader is numpy only.

Dataset: PTB-XL v1.0.3 (Wagner et al. 2020, *Scientific Data*; PhysioNet,
CC BY 4.0). 21,799 ten-second 12-lead ECGs from 18,869 patients, with an
official 100 Hz copy (``records100``) and the native 500 Hz (``records500``).

Acquire the raw tree once (open access, no credentials). The AWS mirror is much
faster than the PhysioNet zip::

    aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ \\
        $SSLLAB_DATA_ROOT/ecg/ptbxl/raw/1.0.3/
    (cd $SSLLAB_DATA_ROOT/ecg/ptbxl/raw/1.0.3 && sed -E 's/^([0-9a-f]{64}) /\\1  /' SHA256SUMS.txt | shasum -a 256 -c --quiet)

Usage
-----
    # real run: 100 Hz cache straight into the lake, then link it into data/
    python examples/ecg_rhythm/00_process_ptbxl.py --fs 100 \\
        --out-dir $SSLLAB_DATA_ROOT/ecg/ptbxl/cache100
    python -c "from ops.datasets import link_dataset; link_dataset('ecg/ptbxl/cache100', link_name='ptbxl100')"

    # tiny synthetic smoke run (no network, seconds): fake PhysioNet tree -> cache
    python examples/ecg_rhythm/00_process_ptbxl.py --smoke

Output
------
    <out-dir>/{signals.npy, records.json, manifest.json}
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.ptbxl import PTBXL_VERSION, build_cache_from_raw, write_toy_raw_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _default_raw_dir() -> Path | None:
    root = os.environ.get("SSLLAB_DATA_ROOT")
    return Path(root).expanduser() / "ecg" / "ptbxl" / "raw" / PTBXL_VERSION if root else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-dir", type=Path, default=_default_raw_dir(),
                   help="PhysioNet PTB-XL tree (default: $SSLLAB_DATA_ROOT/ecg/ptbxl/raw/1.0.3)")
    p.add_argument("--fs", type=int, default=100, choices=(100, 500), help="which official copy to cache")
    p.add_argument("--out-dir", type=Path, default=None, help="cache directory (default: data/ptbxl<fs>)")
    p.add_argument("--limit", type=int, default=None, help="first N records only (debugging)")
    p.add_argument("--no-verify", action="store_true", help="skip per-record checksum verification")
    p.add_argument("--smoke", action="store_true", help="synthetic tree, no network")
    args = p.parse_args()

    out_dir = args.out_dir or Path("data") / f"ptbxl{args.fs}"
    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="ptbxl_smoke_"))
        raw = write_toy_raw_tree(tmp / "raw", n_records=120, fs=args.fs)
        out_dir = args.out_dir or tmp / f"ptbxl{args.fs}"
        logger.info("smoke: synthetic tree at %s", raw)
    else:
        raw = args.raw_dir
        if raw is None or not (raw / "ptbxl_database.csv").exists():
            p.error(f"PTB-XL tree not found at {raw}; pass --raw-dir or set SSLLAB_DATA_ROOT")

    logger.info("building %d Hz cache from %s -> %s", args.fs, raw, out_dir)
    m = build_cache_from_raw(raw, out_dir, fs=args.fs, limit=args.limit, verify=not args.no_verify, logger=logger)
    logger.info(
        "done: %d records, %d patients, leads=%s, fs=%g Hz, %d samples; checksums verified=%s",
        m["n_records"], m["n_patients"], ",".join(m["leads"]), m["fs"], m["n_samples"], m["checksums_verified"],
    )
    logger.info("rhythm vocabulary (%d): %s", len(m["rhythm_codes"]), ", ".join(m["rhythm_codes"]))


if __name__ == "__main__":
    main()
