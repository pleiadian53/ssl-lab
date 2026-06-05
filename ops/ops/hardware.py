"""Local/pod compute detection — the pre-flight 'compute check'.

Generic and dependency-light: reports the compute backend, GPU VRAM, system RAM,
and free disk so you can decide whether a job fits locally or needs a pod. No
task-specific feasibility estimators (those belong with individual experiments).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class HardwareReport:
    backend: str            # "cuda" | "mps" | "cpu"
    gpu_name: str | None
    gpu_vram_gb: float | None
    n_gpus: int
    cpu_count: int | None
    ram_gb: float | None
    disk_free_gb: float | None
    platform: str

    def as_dict(self) -> dict:
        return asdict(self)


def _torch_probe() -> tuple[str, str | None, float | None, int]:
    """Return (backend, gpu_name, vram_gb, n_gpus) using torch if available."""
    try:
        import torch
    except ImportError:
        return "cpu", None, None, 0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "cuda", props.name, round(props.total_memory / 1e9, 1), torch.cuda.device_count()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS shares system RAM; VRAM is not separately reported.
        return "mps", "Apple Silicon (MPS)", None, 1
    return "cpu", None, None, 0


def _ram_gb() -> float | None:
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1e6, 1)  # kB -> GB
    except OSError:
        pass
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        return round(int(out.stdout.strip()) / 1e9, 1)
    except Exception:
        return None


def detect_hardware(path: str = ".") -> HardwareReport:
    backend, gpu_name, vram, n_gpus = _torch_probe()
    try:
        disk_free = round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        disk_free = None
    import os
    return HardwareReport(
        backend=backend,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        n_gpus=n_gpus,
        cpu_count=os.cpu_count(),
        ram_gb=_ram_gb(),
        disk_free_gb=disk_free,
        platform=platform.platform(),
    )


def print_report(report: HardwareReport) -> None:
    bar = "=" * 56
    print(bar)
    print("[ssl-ops] compute check")
    print(bar)
    print(f"  platform   : {report.platform}")
    print(f"  backend    : {report.backend}")
    if report.gpu_name:
        vram = f"{report.gpu_vram_gb} GB" if report.gpu_vram_gb else "shared (MPS)"
        print(f"  gpu        : {report.gpu_name}  x{report.n_gpus}  ({vram})")
    print(f"  cpu cores  : {report.cpu_count}")
    print(f"  ram        : {report.ram_gb} GB" if report.ram_gb else "  ram        : unknown")
    print(f"  disk free  : {report.disk_free_gb} GB" if report.disk_free_gb else "  disk free  : unknown")
    print(bar)
    if report.backend == "cuda":
        print("  verdict    : CUDA GPU present — train locally or on this pod.")
    else:
        print("  verdict    : no CUDA GPU. Fine for smoke tests; provision a pod")
        print("               for real training:")
        print("               python examples/ops/ops_provision_cluster.py --gpu a40")
    print(bar)
