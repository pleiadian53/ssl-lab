#!/usr/bin/env python3
"""Stage large inputs (datasets, pretrained checkpoints, artifacts) to a running pod's volume.

Generalizes the data-staging pattern: rsync any local path(s) to the RunPod network volume of
a *running* SkyPilot cluster, mirroring the repo-relative layout under the project namespace
``/runpod-volume/ssl-lab/<path>``. The pod entrypoints read their inputs from those absolute
volume paths, so nothing large has to ride in the workdir (``.skyignore`` excludes ``/data``
and ``/output``, and datasets/checkpoints are too big or too transient to commit).

The model is: provision a pod once, stage the inputs once (they persist on the volume across
pod lifecycles), then launch training jobs that read them from the volume.

Steps per path:
  1. resolve the running cluster (``--cluster`` or auto-detect the single UP cluster)
  2. ``ssh <cluster> mkdir -p`` the destination on the volume
  3. ``rsync -Pavz -L --no-owner --no-group`` the local dir to the volume
     (``-L`` dereferences symlinks so a symlinked data cache copies real files;
      ``--no-owner --no-group`` because RunPod volumes disallow chown)

Usage:
    # provision a pod first (kept alive), e.g.:
    #   python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- true
    #
    # stage what a job needs (auto-detects the single running cluster):
    python examples/ops/ops_stage_data.py data/norman2019 output/norman_flow_control/checkpoints
    #
    # preview without transferring:
    python examples/ops/ops_stage_data.py --dry-run data/norman2019
    #
    # target an explicit cluster (if several are up):
    python examples/ops/ops_stage_data.py --cluster ssl-perturbation output/norman_stage_a/checkpoints
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_VOLUME_PREFIX = "/runpod-volume/ssl-lab"   # project namespace on the shared volume


def _run(cmd: list[str], check: bool = True) -> int:
    logger.info("run: %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if check and rc != 0:
        logger.error("command failed (exit %d)", rc)
        sys.exit(rc)
    return rc


def _running_clusters() -> list[str]:
    """Names of UP SkyPilot clusters, parsed from ``sky status``."""
    out = subprocess.run(["sky", "status"], capture_output=True, text=True, check=False).stdout
    names: list[str] = []
    for line in out.splitlines():
        cols = line.split()
        # rows look like: NAME  ...  STATUS ...  ; keep those whose STATUS column is UP
        if len(cols) >= 2 and "UP" in cols:
            if cols[0] not in ("NAME",):
                names.append(cols[0])
    return names


def _resolve_cluster(explicit: str | None) -> str:
    if explicit:
        return explicit
    up = _running_clusters()
    if not up:
        logger.error("No running cluster found. Provision one first, e.g.:")
        logger.error("  python examples/ops/ops_run_pipeline.py --execute --gpu a40 -- true")
        sys.exit(1)
    if len(up) == 1:
        logger.info("auto-detected cluster: %s", up[0])
        return up[0]
    logger.error("Multiple clusters are up (%s). Pass --cluster to choose one.", ", ".join(up))
    sys.exit(1)


def _rel_to_repo(path: Path) -> str:
    """Repo-relative POSIX path, so the remote layout mirrors the local one."""
    p = path.resolve()
    try:
        return p.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        # outside the repo (e.g. an absolute data-lake path): mirror by basename under data/
        return f"data/{p.name}"


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def stage_path(cluster: str, local_path: Path, volume_prefix: str, dry_run: bool) -> None:
    rel = _rel_to_repo(local_path)
    dest = f"{volume_prefix}/{rel}"
    size = _dir_size_mb(local_path)
    print(f"\n  source : {local_path.resolve()}")
    print(f"  volume : {cluster}:{dest}")
    print(f"  size   : {size:,.0f} MB")
    if dry_run:
        print("  [dry-run] not transferring.")
        return
    # Ensure the destination's parent exists (rsync won't create deep parents portably).
    parent = str(Path(dest).parent)
    _run(["ssh", cluster, f"mkdir -p {parent!r} {dest!r}"])
    src = str(local_path.resolve()) + ("/" if local_path.is_dir() else "")
    _run(["rsync", "-Pavz", "-L", "--no-owner", "--no-group", src, f"{cluster}:{dest}"])
    print(f"  staged -> {dest}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage datasets / checkpoints / artifacts to a running pod's volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n"
               "  python examples/ops/ops_stage_data.py data/norman2019 "
               "output/norman_flow_control/checkpoints\n",
    )
    p.add_argument("paths", nargs="+", type=Path,
                   help="local paths to stage (repo-relative, e.g. data/norman2019)")
    p.add_argument("--cluster", default=None, help="target cluster (auto-detects the single UP one)")
    p.add_argument("--volume-prefix", default=_DEFAULT_VOLUME_PREFIX,
                   help=f"project namespace on the volume (default: {_DEFAULT_VOLUME_PREFIX})")
    p.add_argument("--dry-run", action="store_true", help="show what would transfer, do nothing")
    args = p.parse_args()

    missing = [str(x) for x in args.paths if not x.exists()]
    if missing:
        for m in missing:
            logger.error("local path not found: %s", m)
        logger.error("Nothing staged. A missing INPUT means an upstream step has not been run "
                     "(train the model / build the dataset first); see the decoder-ablation runbook.")
        sys.exit(2)

    # A dry run needs no live cluster (it transfers nothing) -- use a placeholder so the
    # plan is inspectable offline; a real run resolves/validates the running cluster.
    cluster = args.cluster or ("<cluster>" if args.dry_run else _resolve_cluster(None))
    print("=" * 64)
    print(f"staging {len(args.paths)} path(s) -> {cluster}  (namespace {args.volume_prefix})")
    print("=" * 64)
    for path in args.paths:
        stage_path(cluster, path, args.volume_prefix, args.dry_run)
    print("\n" + "=" * 64)
    if args.dry_run:
        print("dry run complete -- remove --dry-run to transfer.")
    else:
        print(f"done. verify on the pod:  ssh {cluster} 'ls -R {args.volume_prefix} | head'")
    print("=" * 64)


if __name__ == "__main__":
    main()
