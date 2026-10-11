"""PTB-XL adapter: label rules, AF cohorts, patient-level budgets, cache, loader contract.

Everything runs on a tiny synthetic PhysioNet-shaped tree (CSV tables + WFDB
records), so the real reader and cache builder are exercised end to end. The
final test checks the real cache only when it has been built locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ssllab.data.ptbxl import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TOY_RHYTHM,
    af_pilot_cohort,
    budget_patients,
    build_cache_from_raw,
    get_ptbxl_dataloaders,
    load_cache,
    parse_scp_codes,
    patient_support,
    rhythm_labels,
    write_toy_raw_tree,
)
from ssllab.data.signal import SignalGeometry
from ssllab.jepa.model import build_jepa

CODES = tuple(c for c, _ in TOY_RHYTHM)


def _rec(codes):
    return {"scp_codes": codes}


# --------------------------------------------------------------------------- #
# Label semantics.
# --------------------------------------------------------------------------- #
def test_parse_scp_codes():
    assert parse_scp_codes("{'SR': 0.0, 'NORM': 100.0}") == {"SR": 0.0, "NORM": 100.0}
    assert parse_scp_codes("{}") == {}


def test_present_code_with_zero_likelihood_is_positive():
    rl = rhythm_labels([_rec({"SR": 0.0}), _rec({"AFIB": 100.0}), _rec({"NORM": 100.0})], CODES)
    sr, af = CODES.index("SR"), CODES.index("AFIB")
    assert rl.y[0, sr] == 1 and rl.likelihood[0, sr] == 0.0
    assert rl.y[1, af] == 1 and rl.likelihood[1, af] == 100.0
    assert np.isnan(rl.likelihood[0, af]) and rl.y[0, af] == 0  # absent, not zero-likelihood


def test_record_without_rhythm_statement_is_unknown():
    rl = rhythm_labels([_rec({"NORM": 100.0}), _rec({"SR": 0.0})], CODES)
    assert rl.has_rhythm.tolist() == [False, True]
    for rule in ("primary", "sensitivity"):
        include, _ = af_pilot_cohort(rl, rule)
        assert not include[0]


def test_af_pilot_rules():
    recs = [
        _rec({"AFIB": 0.0}),                 # 0  AF only
        _rec({"AFIB": 0.0, "AFLT": 0.0}),    # 1  AF + flutter co-label
        _rec({"AFLT": 0.0}),                 # 2  flutter only
        _rec({"SR": 0.0, "NORM": 100.0}),    # 3  sinus
        _rec({"PACE": 0.0}),                 # 4  paced (non-AF, not healthy)
        _rec({"NORM": 100.0}),               # 5  no rhythm statement
    ]
    rl = rhythm_labels(recs, CODES)
    inc, y = af_pilot_cohort(rl, "primary")
    assert inc.tolist() == [True, False, False, True, True, False]
    assert y[inc].tolist() == [1, 0, 0]
    inc, y = af_pilot_cohort(rl, "sensitivity")
    assert inc.tolist() == [True, True, True, True, True, False]
    assert y[inc].tolist() == [1, 1, 0, 0, 0]  # AFIB+AFLT positive, flutter-only negative
    with pytest.raises(ValueError):
        af_pilot_cohort(rl, "other")


# --------------------------------------------------------------------------- #
# Patient-level budgets and support.
# --------------------------------------------------------------------------- #
def test_budgets_are_nested_patient_level_and_seeded():
    pats = np.repeat(np.arange(200), 2)  # 200 patients, 2 records each
    fracs = (0.01, 0.05, 0.10, 0.25, 1.0)
    sets = [set(budget_patients(pats, f, seed=3).tolist()) for f in fracs]
    assert [len(s) for s in sets] == [2, 10, 20, 50, 200]
    assert all(a <= b for a, b in zip(sets, sets[1:]))
    assert sets == [set(budget_patients(pats, f, seed=3).tolist()) for f in fracs]
    assert set(budget_patients(pats, 0.25, seed=4).tolist()) != sets[3]
    with pytest.raises(ValueError):
        budget_patients(pats, 0.0, seed=0)


def test_patient_support_counts_patients_not_records():
    y = np.array([1, 1, 0, 0, 1, 0])
    pid = np.array([7, 7, 7, 8, 9, 9])  # patient 7 has two positive records
    assert patient_support(y, pid) == (2, 1)


# --------------------------------------------------------------------------- #
# Cache built from a synthetic PhysioNet tree.
# --------------------------------------------------------------------------- #
@pytest.fixture()
def toy_cache(tmp_path: Path) -> Path:
    raw = write_toy_raw_tree(tmp_path / "raw", n_records=90, seed=1)
    build_cache_from_raw(raw, tmp_path / "data" / "ptbxl100")
    return tmp_path / "data"


def test_cache_contents(toy_cache):
    c = load_cache(toy_cache / "ptbxl100")
    assert c.signals.shape == (90, 12, 1000) and c.signals.dtype == np.int16
    assert c.leads[1] == "II" and c.fs == 100.0
    assert c.rhythm_codes == CODES
    assert c.manifest["checksums_verified"] is True
    assert [r["ecg_id"] for r in c.records] == sorted(r["ecg_id"] for r in c.records)


def test_patients_never_span_splits(toy_cache):
    c = load_cache(toy_cache / "ptbxl100")
    split, pid = c.split, c.patient_id
    groups = [set(pid[split == s].tolist()) for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


def test_loader_contract_12_lead_and_lead_ii(toy_cache):
    tr, va, te = get_ptbxl_dataloaders(batch_size=8, data_dir=toy_cache, task="af_primary")
    b = next(iter(tr))
    assert b["tokens"].shape == (8, 40, 300) and b["signal"].shape == (8, 12, 1000)
    assert b["labels"].shape == (8, 1)
    assert torch.allclose(b["signal"].mean(-1), torch.zeros(8, 12), atol=1e-4)
    assert tr.meta["n_tokens"] == 40 and tr.meta["token_dim"] == 300 and tr.meta["fs"] == 100

    tr1, _, _ = get_ptbxl_dataloaders(batch_size=4, data_dir=toy_cache, task="rhythm", leads="II")
    b1 = next(iter(tr1))
    assert b1["tokens"].shape == (4, 40, 25) and b1["labels"].shape == (4, len(CODES))
    assert tr1.meta["leads"] == ["II"]


def test_loader_rows_follow_cohort_and_budget(toy_cache):
    c = load_cache(toy_cache / "ptbxl100")
    rl = c.rhythm()
    inc, _ = af_pilot_cohort(rl, "primary")
    full, va, te = get_ptbxl_dataloaders(batch_size=4, data_dir=toy_cache, task="af_primary")
    for dl, s in ((full, SPLIT_TRAIN), (va, SPLIT_VAL), (te, SPLIT_TEST)):
        rows = dl.dataset.rows
        assert inc[rows].all() and (c.split[rows] == s).all()
    assert set(full.dataset.rows) == set(np.flatnonzero(inc & (c.split == SPLIT_TRAIN)))

    small, va2, _ = get_ptbxl_dataloaders(batch_size=4, data_dir=toy_cache, task="af_primary",
                                          label_frac=0.5, budget_seed=0)
    pats = set(c.patient_id[small.dataset.rows].tolist())
    # Whole patients only: every eligible training record of a chosen patient is in.
    expect = np.flatnonzero(inc & (c.split == SPLIT_TRAIN) & np.isin(c.patient_id, list(pats)))
    assert set(small.dataset.rows) == set(expect)
    assert set(small.dataset.rows) <= set(full.dataset.rows)
    assert np.array_equal(va2.dataset.rows, va.dataset.rows)  # val never subsampled


def test_loader_refuses_rate_mismatch(toy_cache):
    with pytest.raises(ValueError, match="does not match the cache"):
        get_ptbxl_dataloaders(data_dir=toy_cache, geometry=SignalGeometry(fs=500))


def test_jepa_runs_on_ecg_tokens(toy_cache):
    tr, _, _ = get_ptbxl_dataloaders(batch_size=4, data_dir=toy_cache, task="none")
    m = tr.meta
    jepa = build_jepa(token_dim=m["token_dim"], n_tokens=m["n_tokens"], embed_dim=32,
                      enc_depth=1, pred_depth=1, n_heads=2, n_target=4)
    loss, _ = jepa(next(iter(tr))["tokens"])
    loss.backward()
    assert torch.isfinite(loss)
    assert jepa.embed(next(iter(tr))["tokens"]).shape == (4, 32)


# --------------------------------------------------------------------------- #
# The real cache, when it exists locally.
# --------------------------------------------------------------------------- #
REAL = Path(__file__).resolve().parents[1] / "data" / "ptbxl100"


@pytest.mark.skipif(not (REAL / "manifest.json").exists(), reason="real PTB-XL cache not built")
def test_real_cache_matches_published_totals():
    c = load_cache(REAL)
    assert c.manifest["limit"] is None and c.manifest["checksums_verified"]
    assert c.n == 21799 and len(set(c.patient_id.tolist())) == 18869
    assert c.signals.shape[1:] == (12, 1000)
    assert len(c.rhythm_codes) == 12
    split, pid = c.split, c.patient_id
    tr, va, te = (set(pid[split == s].tolist()) for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST))
    assert not (tr & va) and not (tr & te) and not (va & te)
