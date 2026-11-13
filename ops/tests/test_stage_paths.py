"""Where ``examples/ops/ops_stage_data.py`` puts a local path on the pod volume.

Regression for 2026-11-13: staging ``data/norman2019/foundation``, where ``data/norman2019`` is a
symlink into the shared data lake, landed at ``data/foundation`` on the volume, because the path
was resolved (following the symlink out of the repository) before being made repo-relative.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_DRIVER = Path(__file__).resolve().parents[2] / "examples" / "ops" / "ops_stage_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("ops_stage_data", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    repo, lake = tmp_path / "repo", tmp_path / "lake" / "scrna" / "norman2019"
    (lake / "foundation").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    (repo / "output" / "run" / "checkpoints").mkdir(parents=True)
    (repo / "data" / "norman2019").symlink_to(lake, target_is_directory=True)
    return repo, lake


def test_path_inside_symlinked_dataset_keeps_its_repo_location(tmp_path, monkeypatch):
    stage = _load()
    repo, _ = _layout(tmp_path)
    monkeypatch.chdir(repo)
    assert stage._rel_to_repo(Path("data/norman2019/foundation"), repo) == "data/norman2019/foundation"
    assert stage._rel_to_repo(Path("data/norman2019"), repo) == "data/norman2019"


def test_ordinary_repo_path_and_absolute_form_agree(tmp_path, monkeypatch):
    stage = _load()
    repo, _ = _layout(tmp_path)
    monkeypatch.chdir(repo)
    rel = "output/run/checkpoints"
    assert stage._rel_to_repo(Path(rel), repo) == rel
    assert stage._rel_to_repo(repo / rel, repo) == rel


def test_path_outside_repo_mirrors_basename_under_data(tmp_path):
    stage = _load()
    repo, lake = _layout(tmp_path)
    assert stage._rel_to_repo(lake, repo) == "data/norman2019"
