"""Run an ssl-lab job on a remote GPU pod (or dry-run the plan).

Everything after a literal ``--`` is the command to run on the pod. Without
``--execute`` this is a DRY RUN: it prints the generated SkyPilot YAML and a cost
estimate, makes zero cloud calls, and needs no SkyPilot install — the safe way to
inspect exactly what would launch.

Usage
-----
    # Dry-run: see the SkyPilot YAML + cost for training JEPA on an A40
    python examples/ops/ops_run_pipeline.py --gpu a40 -- \
        python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50

    # Launch for real — pod is KEPT ALIVE by default for follow-up training sessions
    python examples/ops/ops_run_pipeline.py --execute --gpu a100 -- \
        python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50

    # ... add --teardown to release the pod when you're done
    python examples/ops/ops_run_pipeline.py --execute --teardown --gpu a100 -- \
        python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50

Output
------
    ops/configs/skypilot/generated/<job>.yaml   the generated SkyPilot task
    runs/                                        results rsynced back (on --execute)
"""

from __future__ import annotations

import argparse
import sys

from ops.gpu_runner import (
    DEFAULT_CONFIG_PATH,
    InfraConfig,
    build_skypilot_config,
    launch,
    print_dry_run,
)


def split_argv(argv: list[str]) -> tuple[list[str], str]:
    """Split on the first ``--`` into (runner args, pod command string)."""
    if "--" not in argv:
        raise SystemExit("error: provide the pod command after '--'.\n"
                         "  e.g. ... -- python examples/jepa_basics/01_train_jepa_mnist.py")
    i = argv.index("--")
    return argv[:i], " ".join(argv[i + 1:]).strip()


def parse_args(runner_argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an ssl-lab job on a remote GPU pod.")
    p.add_argument("--execute", action="store_true", help="actually launch (default: dry-run)")
    p.add_argument("--gpu", default=None, help="override GPU (a40, a100, h100, ...)")
    p.add_argument("--cluster", default=None, help="reuse an existing cluster by name")
    # Pod is kept alive by default (reuse it across training sessions); pass
    # --teardown to tear it down after the run finishes.
    p.add_argument("--teardown", action="store_true",
                   help="tear down the pod after the run (default: keep alive for reuse)")
    p.add_argument("--no-teardown", dest="teardown", action="store_false",
                   help=argparse.SUPPRESS)  # back-compat no-op (keep-alive is already default)
    p.set_defaults(teardown=False)
    p.add_argument("--use-volume", dest="use_volume", action="store_true", default=None)
    p.add_argument("--no-volume", dest="use_volume", action="store_false")
    p.add_argument("--job-name", default=None)
    p.add_argument("--output-dir", default=None, help="local dir for fetched results (default: output)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return p.parse_args(runner_argv)


def main() -> None:
    runner_argv, run_command = split_argv(sys.argv[1:])
    args = parse_args(runner_argv)
    if not run_command:
        raise SystemExit("error: empty pod command after '--'.")

    infra = InfraConfig.from_yaml(args.config)
    infra.apply_overrides(gpu=args.gpu, use_volume=args.use_volume)
    output_local = args.output_dir or infra.output_local

    config = build_skypilot_config(infra, run_command, job_name=args.job_name)

    if args.execute:
        try:
            launch(config, infra, output_local=output_local,
                   cluster=args.cluster, teardown=args.teardown)
        except RuntimeError as e:
            raise SystemExit(str(e))
    else:
        print_dry_run(config, infra, output_local)


if __name__ == "__main__":
    main()
