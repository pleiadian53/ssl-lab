"""Tests for the effect-size metric (Pearson Δ on top-DE genes)."""

from __future__ import annotations

import math

import numpy as np

from ssllab.eval.effect_size import delta_correlation, pearson, summarize


def test_pearson_basic():
    assert abs(pearson([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(pearson([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))   # zero-variance -> nan


def test_delta_correlation_perfect_and_shifted():
    control = np.zeros(5)
    true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    top = [0, 1, 2, 3, 4]
    # Predicting the true mean exactly -> r = 1.
    assert abs(delta_correlation(true, true, control, top) - 1.0) < 1e-9
    # A monotone-preserving prediction still correlates highly on the deltas.
    pred = 0.5 * true + 0.3
    assert delta_correlation(pred, true, control, top) > 0.99
    # An anti-correlated prediction -> r ~ -1.
    assert delta_correlation(-true, true, control, top) < -0.99


def test_delta_correlation_uses_control_reference():
    # Common control reference is subtracted from both sides; a flat (no-effect)
    # prediction has zero delta -> undefined correlation (nan), as expected.
    control = np.array([1.0, 1.0, 1.0])
    true = np.array([1.0, 2.0, 3.0])
    assert math.isnan(delta_correlation(control, true, control, [0, 1, 2]))


def test_summarize_ignores_nan():
    s = summarize({"a": 0.8, "b": float("nan"), "c": 0.6})
    assert s["n_perturbations"] == 2
    assert abs(s["mean_delta_r"] - 0.7) < 1e-9
