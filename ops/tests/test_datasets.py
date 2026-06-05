"""Tests for the storage-dedup helpers (env-var pointer + per-dataset symlinks)."""

from __future__ import annotations

import pytest

from ops.datasets import DATA_ROOT_ENV, data_root, link_dataset
from ops.gpu_runner import DEFAULT_CONFIG_PATH, InfraConfig


def test_infra_config_has_data_path():
    infra = InfraConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert infra.data_path == "data"
    assert infra.local_data_dir == "data"


def test_data_root_reads_env(monkeypatch, tmp_path):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    assert data_root() is None
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert data_root() == tmp_path


def test_link_dataset_creates_symlink(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    (lake / "GRCh38").mkdir(parents=True)
    (lake / "GRCh38" / "genome.fa").write_text("ACGT")
    local = tmp_path / "proj" / "data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(lake))

    link = link_dataset("GRCh38", data_dir=local)
    assert link.is_symlink()
    assert link.resolve() == (lake / "GRCh38").resolve()
    assert (link / "genome.fa").read_text() == "ACGT"

    # idempotent: re-link refreshes without error
    link2 = link_dataset("GRCh38", data_dir=local)
    assert link2.is_symlink()


def test_link_dataset_missing_lake(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError):
        link_dataset("GRCh38", data_dir=tmp_path / "data")


def test_link_dataset_missing_dataset(tmp_path, monkeypatch):
    (tmp_path / "lake").mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "lake"))
    with pytest.raises(FileNotFoundError):
        link_dataset("nope", data_dir=tmp_path / "data")


def test_link_dataset_refuses_real_dir(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    (lake / "ds").mkdir(parents=True)
    monkeypatch.setenv(DATA_ROOT_ENV, str(lake))
    local = tmp_path / "data"
    (local / "ds").mkdir(parents=True)  # a REAL dir already there
    with pytest.raises(IsADirectoryError):
        link_dataset("ds", data_dir=local)
