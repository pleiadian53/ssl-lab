"""Reusable remote-training spine: config -> SkyPilot YAML -> launch -> fetch.

SkyPilot is the orchestration layer; RunPod is the provider (``cloud: runpod``).
We never call the RunPod API directly — we emit provider-neutral SkyPilot configs
and let SkyPilot translate. ``sky`` is imported lazily inside :func:`launch` so
that compute-check and dry-run work without SkyPilot installed.

Intentionally lean: no per-model pip registry — ssl-lab's remote setup is a plain
install of the SSL library.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# GPU catalog. Hourly rates are indicative RunPod community-cloud ballparks for
# cost *estimates* only — verify against live pricing before relying on them.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSpec:
    accelerator: str  # SkyPilot accelerator string, e.g. "A40:1"
    vram_gb: int
    usd_per_hr: float  # indicative


GPU_SPECS: dict[str, GpuSpec] = {
    "l4": GpuSpec("L4:1", 24, 0.43),
    "rtx4090": GpuSpec("RTX4090:1", 24, 0.69),
    "rtx5090": GpuSpec("RTX5090:1", 32, 0.89),
    "a40": GpuSpec("A40:1", 48, 0.39),
    "a100": GpuSpec("A100:1", 80, 1.19),
    "h100": GpuSpec("H100:1", 80, 2.49),
}


class _BlockDumper(yaml.SafeDumper):
    """YAML dumper that renders multi-line strings as literal blocks (``|``)."""


def _str_representer(dumper: yaml.Dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _str_representer)


def dump_yaml(config: dict[str, Any]) -> str:
    """Serialize a SkyPilot config with readable block scalars for scripts."""
    return yaml.dump(config, Dumper=_BlockDumper, sort_keys=False)


def _default_config_dir() -> Path:
    """The packaged ``configs/`` directory (sibling of the ``ops`` package)."""
    return Path(__file__).resolve().parents[1] / "configs"


DEFAULT_CONFIG_PATH = _default_config_dir() / "gpu_config.yaml"
GENERATED_DIR = _default_config_dir() / "skypilot" / "generated"


# --------------------------------------------------------------------------
# Infra configuration
# --------------------------------------------------------------------------


@dataclass
class InfraConfig:
    gpu: str = "a40"
    cloud: str = "runpod"
    docker_image: str = "nvcr.io/nvidia/pytorch:25.02-py3"
    use_volume: bool = False
    volume_name: str = "ssl-lab volume"
    volume_mount: str = "/runpod-volume"
    workdir: str = "."
    data_path: str = "data"
    setup_cmd: str = "pip install -e .\n"
    output_remote: str = "/workspace/output"
    output_local: str = "runs"
    _extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "InfraConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        cfg._extra = extra
        return cfg

    def apply_overrides(self, **kwargs: Any) -> None:
        """Apply non-None CLI overrides onto the config in place."""
        for k, v in kwargs.items():
            if v is None:
                continue
            if not hasattr(self, k):
                raise AttributeError(f"unknown InfraConfig field: {k!r}")
            setattr(self, k, v)
        if self.gpu not in GPU_SPECS:
            raise ValueError(f"unknown gpu {self.gpu!r}; choices: {', '.join(GPU_SPECS)}")

    @property
    def spec(self) -> GpuSpec:
        return GPU_SPECS[self.gpu]

    @property
    def accelerator(self) -> str:
        return self.spec.accelerator

    @property
    def usd_per_hr(self) -> float:
        return self.spec.usd_per_hr

    @property
    def local_data_dir(self) -> str:
        """The project's local input directory (where datasets are referenced).

        Always ``data_path``. The shared-lake env var ``SSLLAB_DATA_ROOT`` is
        consumed only by the per-dataset linker (``ops.datasets``), not here —
        ssl-lab's own data stays local; only specific large datasets are linked
        into this dir from the lake.
        """
        return self.data_path


# --------------------------------------------------------------------------
# SkyPilot config assembly
# --------------------------------------------------------------------------


def _derive_job_name(run_command: str) -> str:
    """Derive a job name from the script in a 'python path/to/NN_name.py ...' command."""
    for tok in shlex.split(run_command):
        if tok.endswith(".py"):
            stem = Path(tok).stem
            # strip a leading NN_ milestone prefix
            parts = stem.split("_")
            if parts and parts[0].isdigit():
                stem = "_".join(parts[1:]) or stem
            return "ssl-" + stem.replace("_", "-")
    return "ssl-job"


def build_skypilot_config(
    infra: InfraConfig, run_command: str, job_name: str | None = None
) -> dict[str, Any]:
    """Assemble a SkyPilot task config (a dict ready for ``yaml.dump``)."""
    job_name = job_name or _derive_job_name(run_command)

    run_lines = [
        "set -e",  # fail the job if the command fails (don't let trailing echoes mask it)
        f"mkdir -p {infra.output_remote}",
        run_command,
        'echo "=================================================="',
        f'echo "[ssl-ops] job {job_name} done. Results in {infra.output_remote}"',
        f'echo "[ssl-ops] fetch with: rsync -Pavz <cluster>:{infra.output_remote}/ {infra.output_local}/"',
    ]

    config: dict[str, Any] = {
        "name": job_name,
        "workdir": infra.workdir,
        "resources": {
            "accelerators": infra.accelerator,
            "cloud": infra.cloud,
            "image_id": f"docker:{infra.docker_image}",
        },
        "setup": infra.setup_cmd,
        "run": "\n".join(run_lines) + "\n",
    }
    if infra.use_volume:
        config["volumes"] = {infra.volume_mount: infra.volume_name}
    return config


def _write_generated(config: dict[str, Any], job_name: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{job_name}.yaml"
    path.write_text(dump_yaml(config))
    return path


# --------------------------------------------------------------------------
# Dry-run (no cloud calls, no SkyPilot needed)
# --------------------------------------------------------------------------


def estimate_cost(infra: InfraConfig, hours: float = 1.0) -> float:
    return infra.usd_per_hr * hours


def print_dry_run(config: dict[str, Any], infra: InfraConfig, output_local: str | Path) -> Path:
    """Print the generated SkyPilot YAML + a cost estimate + manual commands.

    Writes the YAML to the generated dir and returns its path. Makes no cloud
    calls and does not require SkyPilot to be installed.
    """
    job_name = config["name"]
    path = _write_generated(config, job_name)
    bar = "=" * 64
    print(bar)
    print(f"[ssl-ops] DRY RUN — job '{job_name}'  (no pod launched)")
    print(bar)
    print(dump_yaml(config).rstrip())
    print(bar)
    print(f"GPU            : {infra.gpu}  ({infra.accelerator}, {infra.spec.vram_gb} GB)")
    print(f"Cloud          : {infra.cloud}")
    print(f"Est. cost      : ~${infra.usd_per_hr:.2f}/hr (indicative)")
    print(f"Generated YAML : {path}")
    print(f"Results land in: {output_local}/  (rsynced from {infra.output_remote}/)")
    print(bar)
    print("To launch for real:")
    print(f"  sky launch {path} -y")
    print("Or via the driver:")
    print("  python examples/ops/ops_run_pipeline.py --execute --gpu "
          f"{infra.gpu} -- <your command>")
    print(bar)
    return path


# --------------------------------------------------------------------------
# Live launch (requires SkyPilot)
# --------------------------------------------------------------------------


def _require_sky() -> None:
    """Ensure the SkyPilot CLI is usable (both the package and the `sky` binary)."""
    import importlib.util
    import shutil

    has_pkg = importlib.util.find_spec("sky") is not None
    has_cli = shutil.which("sky") is not None
    if not (has_pkg and has_cli):
        missing = "the `sky` CLI is not on PATH" if has_pkg else "SkyPilot is not installed"
        raise RuntimeError(
            f"Cannot launch: {missing}. Install the remote extra into this env:\n"
            "    pip install -e 'ops/[remote]'\n"
            "and confirm the CLI is available:  sky --version"
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"[ssl-ops] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)


def launch(
    config: dict[str, Any],
    infra: InfraConfig,
    output_local: str | Path | None = None,
    cluster: str | None = None,
    teardown: bool = True,
) -> None:  # pragma: no cover - requires SkyPilot + a real account
    """Launch the job on a pod, rsync results back, optionally tear down.

    The cluster is given a deterministic name (``-c <job_name>``) so the rsync
    host and teardown target are unambiguous (SkyPilot otherwise auto-generates
    a name we'd have to guess). Teardown runs in a ``finally`` so a failed job or
    rsync never leaves a pod billing — *unless* ``teardown=False``, in which case
    the pod is intentionally kept for inspection / re-runs.
    """
    _require_sky()
    job_name = config["name"]
    cluster_name = cluster or job_name
    output_local = Path(output_local or infra.output_local)
    output_local.mkdir(parents=True, exist_ok=True)
    yaml_path = _write_generated(config, job_name)

    launched = False
    try:
        try:
            _run(["sky", "launch", str(yaml_path), "-y", "-c", cluster_name])
            launched = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"job failed on cluster '{cluster_name}'. Inspect logs:\n"
                f"    sky logs {cluster_name}\n"
                f"    ssh {cluster_name}"
            ) from e
        # ``-u`` (--update) is load-bearing, not a nicety: it SKIPS any file that is newer on
        # the receiver. Without it this is a blanket overwrite of the whole results tree, and a
        # pod whose inputs are stale (an out-of-date cache on the volume, say) will silently
        # replace correct local results with wrong ones. That has happened. A result that
        # changes when you re-run the same command is the hardest kind of bug to notice, so the
        # fetch must never be able to move a result backwards.
        _run([
            "rsync", "-Pavz", "-u",
            f"{cluster_name}:{infra.output_remote}/",
            f"{output_local}/",
        ])
        print(f"[ssl-ops] results -> {output_local}/  (--update: newer local files were kept)")
    finally:
        if not launched:
            # Provisioning never succeeded, so there is no pod. Saying otherwise turns a
            # failure into something that reads like a success.
            print(f"[ssl-ops] cluster '{cluster_name}' was NOT provisioned; nothing to tear down.")
        elif teardown:
            # check=False: a teardown hiccup must not mask the original error.
            subprocess.run(["sky", "down", cluster_name, "-y"], check=False)
            print(f"[ssl-ops] torn down cluster '{cluster_name}'")
        else:
            print(f"[ssl-ops] pod '{cluster_name}' left running "
                  f"(ssh {cluster_name}); tear down with: sky down {cluster_name} -y")


# --------------------------------------------------------------------------
# Long-running cluster management (provision / status / teardown)
# --------------------------------------------------------------------------


def keepalive_config(infra: InfraConfig, name: str) -> dict[str, Any]:
    """A SkyPilot task that just sets up the env and stays available for SSH/jobs."""
    config = build_skypilot_config(infra, 'echo "[ssl-ops] workspace ready"', job_name=name)
    return config


def provision(config: dict[str, Any], name: str) -> None:  # pragma: no cover - needs SkyPilot
    """Bring up a persistent cluster by name (no rsync, no teardown)."""
    _require_sky()
    yaml_path = _write_generated(config, name)
    _run(["sky", "launch", str(yaml_path), "-c", name, "-y"])
    print(f"[ssl-ops] cluster '{name}' up. SSH: ssh {name}")
    print(f"[ssl-ops] run jobs on it: ops_run_pipeline.py --execute --cluster {name} --no-teardown -- <cmd>")


def status() -> None:  # pragma: no cover - needs SkyPilot
    _require_sky()
    _run(["sky", "status", "--refresh"])


def down(name: str) -> None:  # pragma: no cover - needs SkyPilot
    _require_sky()
    _run(["sky", "down", name, "-y"])


def down_all() -> None:  # pragma: no cover - needs SkyPilot
    _require_sky()
    _run(["sky", "down", "--all", "-y"])
