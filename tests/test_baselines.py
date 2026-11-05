"""Tests for the training-free additive reference predictor."""

from __future__ import annotations

import numpy as np
import pytest

from ssllab.eval.baselines import additive_predictor
from ssllab.eval.effect_size import run_effect_size_eval

G = 6
NAMES = np.array(["control", "A", "B", "A+B"])
DELTA = {"A": np.array([1.0, 0, 2, 0, 0, 0.5]), "B": np.array([0, 3.0, 0, 1, 0, 0.5])}
CTRL = np.full(G, 0.2)


def _toy(n_per: int = 30, combo_delta: np.ndarray | None = None, seed: int = 0):
    """Cells whose per-perturbation means are exact; the combo cells are the only test cells."""
    rng = np.random.default_rng(seed)
    means = {"control": CTRL, "A": CTRL + DELTA["A"], "B": CTRL + DELTA["B"],
             "A+B": CTRL + (DELTA["A"] + DELTA["B"] if combo_delta is None else combo_delta)}
    X, pid = [], []
    for i, name in enumerate(NAMES):
        noise = rng.normal(size=(n_per, G))
        X.append(means[name] + noise - noise.mean(0))          # exact group mean
        pid += [i] * n_per
    X, pid = np.vstack(X), np.array(pid)
    is_train = pid != 3
    return X, pid, is_train


def test_additive_prediction_is_sum_of_training_singles():
    X, pid, is_train = _toy()
    predict = additive_predictor(hvg_X=X, pert_names=NAMES, pert_id=pid,
                                 is_train=is_train, control_mean=CTRL)
    np.testing.assert_allclose(predict(3, "A+B"), CTRL + DELTA["A"] + DELTA["B"], atol=1e-12)
    np.testing.assert_allclose(predict(1, "A"), CTRL + DELTA["A"], atol=1e-12)


def test_exactly_additive_combo_scores_one():
    X, pid, is_train = _toy()
    predict = additive_predictor(hvg_X=X, pert_names=NAMES, pert_id=pid,
                                 is_train=is_train, control_mean=CTRL)
    de = {"A+B": {"top_idx": list(range(G))}}
    per_pert, _ = run_effect_size_eval(predict, hvg_X=X, pert_names=NAMES, pert_id=pid,
                                       is_test=~is_train, de_genes=de, control_mean=CTRL,
                                       top_k=G, min_test_cells=5)
    assert abs(per_pert["A+B"] - 1.0) < 1e-9


def test_prediction_never_reads_test_cells():
    # Change only the held-out combo's cells; the prediction must not move.
    X1, pid, is_train = _toy()
    X2, _, _ = _toy(combo_delta=np.array([-5.0, 4, -3, 2, -1, 0]))
    p1 = additive_predictor(hvg_X=X1, pert_names=NAMES, pert_id=pid, is_train=is_train, control_mean=CTRL)
    p2 = additive_predictor(hvg_X=X2, pert_names=NAMES, pert_id=pid, is_train=is_train, control_mean=CTRL)
    np.testing.assert_array_equal(p1(3, "A+B"), p2(3, "A+B"))


def test_missing_single_raises():
    X, pid, is_train = _toy()
    predict = additive_predictor(hvg_X=X, pert_names=NAMES, pert_id=pid,
                                 is_train=is_train, control_mean=CTRL)
    with pytest.raises(KeyError):
        predict(3, "A+C")
    no_b_train = is_train & (pid != 2)
    predict = additive_predictor(hvg_X=X, pert_names=NAMES, pert_id=pid,
                                 is_train=no_b_train, control_mean=CTRL)
    with pytest.raises(ValueError):
        predict(3, "A+B")
