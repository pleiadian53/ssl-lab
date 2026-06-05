"""Unit tests for the ops config + SkyPilot YAML assembly (no cloud, no SkyPilot)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops.gpu_runner import (
    DEFAULT_CONFIG_PATH,
    GPU_SPECS,
    InfraConfig,
    _derive_job_name,
    build_skypilot_config,
)


def test_from_yaml_roundtrips_default_config():
    infra = InfraConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert infra.gpu in GPU_SPECS
    assert infra.cloud == "runpod"
    assert infra.output_local == "output"


def test_apply_overrides_validates_gpu():
    infra = InfraConfig.from_yaml(DEFAULT_CONFIG_PATH)
    infra.apply_overrides(gpu="a100")
    assert infra.gpu == "a100"
    assert infra.accelerator == "A100:1"
    with pytest.raises(ValueError):
        infra.apply_overrides(gpu="does-not-exist")


def test_derive_job_name_strips_milestone_prefix():
    name = _derive_job_name("python examples/jepa_basics/01_train_jepa_mnist.py --epochs 5")
    assert name == "ssl-train-jepa-mnist"


def test_build_skypilot_config_shape():
    infra = InfraConfig.from_yaml(DEFAULT_CONFIG_PATH)
    infra.apply_overrides(gpu="a40", use_volume=False)  # pin the no-volume path
    cmd = "python examples/jepa_basics/01_train_jepa_mnist.py --epochs 50"
    cfg = build_skypilot_config(infra, cmd)

    assert cfg["name"] == "ssl-train-jepa-mnist"
    assert cfg["resources"]["accelerators"] == "A40:1"
    assert cfg["resources"]["cloud"] == "runpod"
    assert cfg["resources"]["image_id"].startswith("docker:")
    assert "pip install --no-deps ." in cfg["setup"]  # NGC-safe non-editable install
    assert cfg["run"].startswith("set -e")  # failures must not be masked
    assert cmd in cfg["run"]
    assert infra.output_remote in cfg["run"]
    # volume off by default -> no volumes key
    assert "volumes" not in cfg


def test_build_skypilot_config_with_volume():
    infra = InfraConfig.from_yaml(DEFAULT_CONFIG_PATH)
    infra.apply_overrides(gpu="a40", use_volume=True)
    cfg = build_skypilot_config(infra, "python train.py")
    assert cfg["volumes"] == {infra.volume_mount: infra.volume_name}


def test_print_dry_run_writes_generated_yaml(capsys):
    infra = InfraConfig.from_yaml(DEFAULT_CONFIG_PATH)
    cfg = build_skypilot_config(infra, "python examples/jepa_basics/01_train_jepa_mnist.py")
    from ops.gpu_runner import print_dry_run

    path = print_dry_run(cfg, infra, "runs")
    assert Path(path).exists()
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "A40:1" in out
