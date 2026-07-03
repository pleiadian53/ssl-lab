"""Tests for the calibration metrics (distributional spread of the response)."""

from __future__ import annotations

import numpy as np

from ssllab.eval.calibration import interval_coverage, mean_wasserstein, spread_correlation


def test_spread_correlation_recovers_per_gene_std_ranking():
    rng = np.random.default_rng(0)
    # 4 genes with increasing spread; predicted population shares the ranking (+ jitter).
    scales = np.array([0.1, 0.5, 1.0, 3.0])
    true = rng.standard_normal((500, 4)) * scales
    pred = rng.standard_normal((500, 4)) * (scales * 1.2)   # same ranking, different absolute
    r = spread_correlation(pred, true, [0, 1, 2, 3])
    assert r > 0.95


def test_interval_coverage_matches_nominal_when_distributions_agree():
    rng = np.random.default_rng(1)
    true = rng.standard_normal((2000, 3))
    pred = rng.standard_normal((2000, 3))            # same distribution
    cov = interval_coverage(pred, true, [0, 1, 2], lo=0.1, hi=0.9)
    assert abs(cov - 0.8) < 0.03                     # ~nominal


def test_interval_coverage_drops_when_model_underdisperses():
    rng = np.random.default_rng(2)
    true = rng.standard_normal((2000, 3)) * 3.0      # wide truth
    pred = rng.standard_normal((2000, 3)) * 0.3      # over-confident model
    cov = interval_coverage(pred, true, [0, 1, 2], lo=0.1, hi=0.9)
    assert cov < 0.3                                 # far below nominal 0.8


def test_wasserstein_zero_for_identical_and_grows_with_shift():
    x = np.linspace(-2, 2, 500).reshape(-1, 1)
    near = mean_wasserstein(x, x, [0])
    far = mean_wasserstein(x + 5.0, x, [0])
    assert near < 1e-9
    assert far > 4.5
