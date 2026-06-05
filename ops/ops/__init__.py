"""ssllab-ops — remote-training infrastructure for ssl-lab.

Decoupled from the SSL library (`ssllab`): nothing here imports JEPA, and the
SSL library never imports this. Cloud infra is necessary scaffolding for
training practically useful models at scale, not part of SSL itself.

SkyPilot orchestrates; RunPod is the provider (a config value).
"""

from ops.gpu_runner import (
    GPU_SPECS,
    InfraConfig,
    build_skypilot_config,
    down,
    down_all,
    estimate_cost,
    keepalive_config,
    launch,
    print_dry_run,
    provision,
    status,
)
from ops.datasets import data_root, link_dataset
from ops.hardware import HardwareReport, detect_hardware, print_report

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "InfraConfig",
    "GPU_SPECS",
    "build_skypilot_config",
    "print_dry_run",
    "launch",
    "keepalive_config",
    "provision",
    "status",
    "down",
    "down_all",
    "estimate_cost",
    "detect_hardware",
    "print_report",
    "HardwareReport",
    "data_root",
    "link_dataset",
]
