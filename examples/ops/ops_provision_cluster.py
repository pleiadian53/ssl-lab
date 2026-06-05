"""Provision a long-running GPU pod for interactive / iterative ssl-lab work.

Unlike ops_run_pipeline.py (one job, fetch, tear down), this brings up a pod and
leaves it running so you can SSH in and launch many jobs against the same cluster
without re-provisioning.

Default behavior PROVISIONS FOR REAL (A40 from gpu_config.yaml) — consistent with
agentic-spliceai. Use --dry-run to preview the plan without touching the cloud.

Usage
-----
    # Provision with defaults (A40, RunPod) — launches a real pod
    python examples/ops/ops_provision_cluster.py

    # Preview only, no cloud calls
    python examples/ops/ops_provision_cluster.py --dry-run

    # Specific GPU / name
    python examples/ops/ops_provision_cluster.py --gpu a100 --name ssl-workspace

    # Manage
    python examples/ops/ops_provision_cluster.py --status
    python examples/ops/ops_provision_cluster.py --down ssl-workspace
    python examples/ops/ops_provision_cluster.py --down-all

Then run jobs on it:
    python examples/ops/ops_run_pipeline.py --execute --cluster ssl-workspace \
        --no-teardown -- python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50
"""

from __future__ import annotations

import argparse

from ops.gpu_runner import (
    DEFAULT_CONFIG_PATH,
    InfraConfig,
    down,
    down_all,
    keepalive_config,
    print_dry_run,
    provision,
    status,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Provision/manage a long-running ssl-lab GPU pod.")
    p.add_argument("--gpu", default=None, help="override GPU (default: a40, from gpu_config.yaml)")
    p.add_argument("--name", default="ssl-workspace", help="cluster name")
    p.add_argument("--data-path", default=None, help="local input dir (default: from gpu_config.yaml)")
    p.add_argument("--use-volume", dest="use_volume", action="store_true", default=None)
    p.add_argument("--no-volume", dest="use_volume", action="store_false")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan without launching (default: provision for real)")
    # Management actions (each short-circuits provisioning).
    p.add_argument("--status", action="store_true", help="show running clusters")
    p.add_argument("--down", metavar="NAME", default=None, help="tear down a cluster by name")
    p.add_argument("--down-all", action="store_true", help="tear down all clusters")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.status:
        status()
        return
    if args.down:
        down(args.down)
        return
    if args.down_all:
        down_all()
        return

    infra = InfraConfig.from_yaml(args.config)
    infra.apply_overrides(gpu=args.gpu, use_volume=args.use_volume, data_path=args.data_path)
    config = keepalive_config(infra, args.name)

    if args.dry_run:
        print_dry_run(config, infra, infra.output_local)
        print(f"Data dir       : {infra.local_data_dir}/  (set $SSLLAB_DATA_ROOT to link shared datasets)")
        print("Remove --dry-run to provision this A40 pod for real." if infra.gpu == "a40"
              else f"Remove --dry-run to provision this {infra.gpu} pod for real.")
    else:
        try:
            provision(config, args.name)
        except RuntimeError as e:
            raise SystemExit(str(e))


if __name__ == "__main__":
    main()
