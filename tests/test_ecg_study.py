"""ECG study protocol: amendments, relevant-settings compatibility, pinned reuse."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "ecg_rhythm"
sys.path.insert(0, str(EX))
import _study as S  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    for f in ("study_v1.json", "study_v1_amend1.json", "study_v1_amend2.json"):
        shutil.copy(EX / "configs" / f, tmp_path / f)
    return tmp_path


def test_amendment_applies_only_the_grid_and_keeps_the_base_frozen(cfg):
    base = S.load_study(cfg / "study_v1.json")
    am = S.load_study(cfg / "study_v1_amend1.json")
    assert am["downstream"]["lr_grid"] == [0.0001, 0.0003, 0.001, 0.003]
    assert base["downstream"]["lr_grid"] == [0.0003, 0.001]
    assert am["_base_sha256"] == base["_sha256"] == "fb5c061233a07024"
    assert am["_sha256"] != base["_sha256"]
    for k in ("subsets", "pretrain", "run"):
        assert S.relevant_sha(am, k) == S.relevant_sha(base, k)  # the grid is relevant to none of them


def test_non_amendable_change_and_tampered_base_are_refused(cfg):
    doc = json.loads((cfg / "study_v1_amend1.json").read_text())
    doc["changes"]["downstream.max_steps"] = 8000
    (cfg / "bad.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="not amendable"):
        S.load_study(cfg / "bad.json")
    (cfg / "study_v1.json").write_text((cfg / "study_v1.json").read_text() + " ")
    with pytest.raises(SystemExit, match="amends"):
        S.load_study(cfg / "study_v1_amend1.json")


def test_run_hash_tracks_training_settings_but_not_the_grid(cfg):
    s = S.load_study(cfg / "study_v1_amend1.json")
    h = S.relevant_sha(s, "run")
    s["downstream"]["lr_grid"] = [1.0]
    assert S.relevant_sha(s, "run") == h
    s["downstream"]["max_steps"] = 1
    assert S.relevant_sha(s, "run") != h


def test_legacy_artifacts_accepted_only_with_a_stated_reason(cfg):
    s = S.load_study(cfg / "study_v1_amend1.json")
    why = S.check_compatible({"study_sha256": "fb5c061233a07024"}, s, "run", "pilot run")
    assert "downstream.lr_grid" in why and "unchanged" in why
    with pytest.raises(SystemExit, match="INCOMPATIBLE"):
        S.check_compatible({"study_sha256": "0000000000000000"}, s, "run", "foreign run")
    with pytest.raises(SystemExit, match="INCOMPATIBLE"):
        S.check_compatible({"study_sha256": "x", "relevant_sha": {"run": "deadbeef"}}, s, "run", "drifted run")
    new = {"study_sha256": s["_sha256"], **S.provenance(s)}
    assert "same run-relevant settings" in S.check_compatible(new, s, "run", "new run")


def test_pinned_subsets_must_match_byte_for_byte(cfg, tmp_path):
    s = S.load_study(cfg / "study_v1_amend1.json")
    root = tmp_path / "out" / "exp"
    (root / "subsets").mkdir(parents=True)
    (root / "subsets" / "subsets.json").write_text(json.dumps({"study_sha256": "fb5c061233a07024"}))
    with pytest.raises(SystemExit, match="pinned"):
        S.load_subsets(s, tmp_path / "out", "exp")


def test_amendment2_final_grid_supersedes_amendment1(cfg):
    a2 = S.load_study(cfg / "study_v1_amend2.json")
    assert a2["downstream"]["lr_grid"] == [0.0001, 0.0003, 0.001, 0.003, 0.01]
    assert a2["_base_sha256"] == "fb5c061233a07024"
    assert S.new_rates(a2) == [0.01] and S.reuse_tags(a2) == ["lrsearch_a1"]
    assert a2["_amendment"]["final_grid"] is True
    a1 = S.load_study(cfg / "study_v1_amend1.json")
    assert S.relevant_sha(a2, "run") == S.relevant_sha(a1, "run")  # amendment-1 runs stay reusable
    # A changed superseded amendment is detected.
    (cfg / "study_v1_amend1.json").write_text((cfg / "study_v1_amend1.json").read_text() + " ")
    with pytest.raises(SystemExit, match="supersedes"):
        S.load_study(cfg / "study_v1_amend2.json")


def test_default_protocol_is_amendment2():
    assert S.STUDY_PATH.name == "study_v1_amend2.json"
