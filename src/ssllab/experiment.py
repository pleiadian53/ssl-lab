"""Experiment output layout — keep artifacts organized, not flat.

A flat ``output/`` (or ``runs/``) dir turns unwieldy fast once you have multiple
models, runs, and artifact types. This helper gives every script a single,
self-explanatory hierarchy keyed by experiment name::

    output/<experiment>/
      checkpoints/   model weights (encoder.pt, decoder.pt, prior.pt, ...)
      samples/       generated images (samples.png, recon.png, ...)
      reports/       metrics + metadata as JSON (jepa_train.json, probe.json, ...)
      logs/          run logs (train.log; captured by the pod pipeline entrypoint)

Scripts in a pipeline share artifacts by using the *same* ``--experiment`` name
(stable, not a timestamp), so e.g. the decoder trainer finds the encoder the JEPA
trainer wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentDirs:
    """Resolved output directories for one experiment."""

    root: Path
    checkpoints: Path
    samples: Path
    reports: Path
    logs: Path

    def ensure(self) -> "ExperimentDirs":
        """Create all subdirectories (idempotent). Returns self for chaining."""
        for d in (self.checkpoints, self.samples, self.reports, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def write_report(self, name: str, payload: dict[str, Any]) -> Path:
        """Dump a metrics/metadata dict to ``reports/<name>.json``."""
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / (name if name.endswith(".json") else f"{name}.json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


def experiment(name: str = "jepa_mnist", output_root: str | Path = "output") -> ExperimentDirs:
    """Build the directory layout for an experiment (does not create dirs).

    Call ``.ensure()`` to create them. ``output_root`` lets the pod redirect all
    outputs onto a persistent volume without changing the experiment structure.
    """
    base = Path(output_root) / name
    return ExperimentDirs(
        root=base,
        checkpoints=base / "checkpoints",
        samples=base / "samples",
        reports=base / "reports",
        logs=base / "logs",
    )
