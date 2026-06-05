"""Pre-flight compute check: what hardware am I on, and do I need a pod?

Run this before paying for a GPU. Reports the compute backend, GPU/VRAM, RAM,
and free disk for the current machine (local or a pod you've SSH'd into).

Usage
-----
    python examples/ops/ops_compute_check.py
    python examples/ops/ops_compute_check.py --json
"""

from __future__ import annotations

import argparse
import json

from ops.hardware import detect_hardware, print_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect local/pod compute capabilities.")
    p.add_argument("--path", default=".", help="path to check free disk for")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = detect_hardware(args.path)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
