"""Magnitude and interaction metrics (ssllab.eval.interaction) and the shared bootstrap."""

from __future__ import annotations

import math

import numpy as np

from ssllab.eval.bootstrap import maxt_family, percentile_interval, resample_indices
from ssllab.eval.effect_size import run_effect_size_eval
from ssllab.eval.interaction import (
    NUMERATOR_FLOOR,
    measured_single_deltas,
    pair_metrics,
    split_half_floor,
)

G = 30
TOP = list(range(0, 20))


def _setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(1.0, 0.2, G)
    da, db = rng.normal(0, 0.5, G), rng.normal(0, 0.5, G)
    inter = rng.normal(0, 0.3, G)
    true_pair = ctrl + da + db + inter
    return rng, ctrl, da, db, inter, true_pair


def test_additive_prediction_scores_exactly_one():
    _, ctrl, da, db, _, true_pair = _setup()
    m = pair_metrics(pred_pair=ctrl + da + db, true_pair=true_pair, control_mean=ctrl,
                     delta_a=da, delta_b=db, top_idx=TOP)
    assert math.isclose(m["r"], 1.0, rel_tol=1e-12)
    assert abs(m["log_r"]) < 1e-12
    assert math.isnan(m["interaction_r"])               # predicted interaction is exactly zero
    assert math.isclose(m["rel_err"], m["rel_gi"], rel_tol=1e-12)


def test_ratio_is_round8_error_over_additive_error():
    rng, ctrl, da, db, _, true_pair = _setup(1)
    pred = true_pair + rng.normal(0, 0.2, G)
    m = pair_metrics(pred_pair=pred, true_pair=true_pair, control_mean=ctrl, delta_a=da, delta_b=db, top_idx=TOP)
    assert math.isclose(m["r"], m["rel_err"] / m["rel_gi"], rel_tol=1e-12)
    assert math.isclose(m["log_r"], math.log(m["r"]), rel_tol=1e-12)


def test_denominator_cancels_in_a_contrast():
    rng, ctrl, da, db, _, true_pair = _setup(2)
    p1, p2 = true_pair + rng.normal(0, 0.1, G), true_pair + rng.normal(0, 0.3, G)
    kw = dict(true_pair=true_pair, control_mean=ctrl, delta_a=da, delta_b=db, top_idx=TOP)
    m1, m2 = pair_metrics(pred_pair=p1, **kw), pair_metrics(pred_pair=p2, **kw)
    idx = np.array(TOP)
    direct = math.log(np.linalg.norm(p1[idx] - true_pair[idx])) - math.log(np.linalg.norm(p2[idx] - true_pair[idx]))
    assert math.isclose(m1["log_r"] - m2["log_r"], direct, rel_tol=1e-10)


def test_perfect_prediction_is_finite_via_numerator_floor():
    _, ctrl, da, db, _, true_pair = _setup(3)
    m = pair_metrics(pred_pair=true_pair, true_pair=true_pair, control_mean=ctrl, delta_a=da, delta_b=db, top_idx=TOP)
    expected = math.log(NUMERATOR_FLOOR) - math.log(m["interaction_norm"])
    assert math.isclose(m["log_r"], expected, rel_tol=1e-12)


def test_numerically_zero_interaction_gives_nan_ratios():
    rng, ctrl, da, db, _, _ = _setup(4)
    true_pair = ctrl + da + db                            # exactly additive: I_p = 0
    m = pair_metrics(pred_pair=true_pair + rng.normal(0, 0.1, G), true_pair=true_pair, control_mean=ctrl,
                     delta_a=da, delta_b=db, top_idx=TOP, pred_a=ctrl + da, pred_b=ctrl + db)
    assert math.isnan(m["r"]) and math.isnan(m["log_r"]) and math.isnan(m["internal_log_r"])
    assert np.isfinite(m["rel_err"])


def test_internal_metrics_equal_the_measured_ones_when_singles_are_exact():
    rng, ctrl, da, db, _, true_pair = _setup(5)
    pred = true_pair + rng.normal(0, 0.2, G)
    m = pair_metrics(pred_pair=pred, true_pair=true_pair, control_mean=ctrl, delta_a=da, delta_b=db,
                     top_idx=TOP, pred_a=ctrl + da, pred_b=ctrl + db)
    assert math.isclose(m["internal_r"], m["interaction_r"], rel_tol=1e-10)
    assert math.isclose(m["internal_log_r"], m["log_r"], rel_tol=1e-10)


def test_measured_single_deltas_uses_training_cells_only():
    X = np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 9.0], [7.0, 1.0]])
    names = np.array(["control", "A"])
    pert_id = np.array([0, 1, 1, 1])
    is_train = np.array([True, True, False, True])
    d = measured_single_deltas(["A"], hvg_X=X, pert_names=names, pert_id=pert_id, is_train=is_train,
                               control_mean=np.array([1.0, 1.0]))
    assert np.allclose(d["A"], [4.0, 0.0])                # mean of rows 1 and 3, minus control


def test_split_half_floor_is_finite_and_small_for_many_cells():
    rng, ctrl, da, db, inter, _ = _setup(6)
    cells = (ctrl + da + db + inter) + rng.normal(0, 0.05, (400, G))
    f = split_half_floor(cells, control_mean=ctrl, delta_a=da, delta_b=db, top_idx=TOP, seed=0)
    assert np.isfinite(f["floor_log_r"]) and f["floor_r"] < 0.2 and f["n_cells"] == 400


def test_maxt_family_matches_the_compare_arms_arithmetic():
    rng = np.random.default_rng(0)
    d1, d2 = rng.normal(0.1, 0.3, 20), rng.normal(-0.05, 0.2, 20)
    idx = resample_indices(20, 2000, seed=0)
    fam = maxt_family({"a": d1, "b": d2}, idx)
    # Re-derive as 12_compare_arms.py does, inline.
    ts, ses, obs = [], [], []
    for d in (d1, d2):
        o, se = d.mean(), d.std(ddof=1) / np.sqrt(20)
        bs = d[idx]
        t = (bs.mean(1) - o) / (bs.std(axis=1, ddof=1) / np.sqrt(20))
        ts.append(np.abs(t)); ses.append(se); obs.append(o)
    crit = np.percentile(np.vstack(ts).max(0), 95)
    assert math.isclose(fam["critical_value"], crit, rel_tol=1e-12)
    for k, o, se in zip(("a", "b"), obs, ses):
        assert np.allclose(fam["contrasts"][k]["ci_simultaneous"], [o - crit * se, o + crit * se])
    assert crit > 1.96                                    # two contrasts: wider than a single test
    pi = percentile_interval(d1, idx)
    assert pi["ci"][0] < pi["mean"] < pi["ci"][1]


def test_collecting_profiles_does_not_change_the_score():
    rng = np.random.default_rng(0)
    names = np.array(["control", "A", "A+B"])
    pert_id = np.repeat([0, 1, 2], 30)
    X = rng.normal(1, 0.3, (90, G)) + (pert_id[:, None] == 2) * np.linspace(0, 1, G)
    is_test = np.tile([True] * 25 + [False] * 5, 3)
    de = {"A": {"top_idx": TOP}, "A+B": {"top_idx": TOP}}
    ctrl = X[pert_id == 0].mean(0)
    fn = lambda pid, name: X[pert_id == pid].mean(0) + 0.01 * pid
    kw = dict(hvg_X=X, pert_names=names, pert_id=pert_id, is_test=is_test, de_genes=de, control_mean=ctrl)
    base, _ = run_effect_size_eval(fn, **kw)
    prof: dict = {}
    again, _ = run_effect_size_eval(fn, profiles=prof, **kw)
    assert base == again and set(prof) == set(base)
    assert np.allclose(prof["A+B"], fn(2, "A+B"))
