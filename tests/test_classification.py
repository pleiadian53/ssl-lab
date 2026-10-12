"""Classification metrics, the threshold rule, and the paired patient-cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from ssllab.data.signal import SignalBatcher, patchify_1d, zscore_per_lead
from ssllab.eval.classification import (
    macro_auroc,
    macro_auroc_metric,
    operating_point,
    paired_cluster_bootstrap,
    per_label_metrics,
    threshold_at_sensitivity,
)
from ssllab.models.classifier import EncoderClassifier
from ssllab.models.encoder import JEPAEncoder


def _toy(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, k)) < np.array([0.5, 0.2, 0.05])[:k]).astype(int)
    S = Y + rng.normal(0, 0.8, (n, k))
    return Y, S


def test_per_label_metrics_match_sklearn_and_flag_undefined():
    Y, S = _toy()
    Y[:, 2] = 0  # a label with no positives
    m = per_label_metrics(Y, S, ["a", "b", "c"])
    assert m["a"]["auroc"] == pytest.approx(roc_auc_score(Y[:, 0], S[:, 0]))
    assert m["c"]["defined"] is False and np.isnan(m["c"]["auroc"])


def test_macro_auroc_never_shrinks_the_label_set():
    Y, S = _toy()
    v, ok = macro_auroc(Y, S, [0, 1])
    assert ok and v == pytest.approx(np.mean([roc_auc_score(Y[:, j], S[:, j]) for j in (0, 1)]))
    Y[:, 1] = 0
    v, ok = macro_auroc(Y, S, [0, 1])
    assert not ok and np.isnan(v)


def test_threshold_rule_is_the_most_specific_meeting_the_target():
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.1, 0.52, 0.3, 0.2])
    t = threshold_at_sensitivity(y, s, 0.9)
    assert t == 0.5                                   # 9 of 10 positives at or above
    op = operating_point(y, s, t)
    assert op["sensitivity"] == pytest.approx(0.9) and op["fp"] == 1 and op["tn"] == 2
    assert operating_point(y, s, 0.51)["sensitivity"] < 0.9  # any higher threshold misses the target


def test_bootstrap_zero_difference_and_cluster_resampling():
    Y, S = _toy(n=300, k=2)
    clusters = np.repeat(np.arange(100), 3)
    metric = macro_auroc_metric([0, 1])
    same = paired_cluster_bootstrap(metric, Y, S, S, clusters, n_boot=200, seed=0)
    assert same.estimate == 0 and same.ci_low == 0 and same.ci_high == 0
    better = paired_cluster_bootstrap(metric, Y, S + 0.5 * Y, S, clusters, n_boot=300, seed=0)
    assert better.estimate > 0 and better.ci_low > 0
    # Repeats: mean over repeats of per-repeat differences, same resample for all repeats.
    A = np.stack([S + 0.5 * Y, S + 0.3 * Y])
    B = np.stack([S, S])
    rep = paired_cluster_bootstrap(metric, Y, A, B, clusters, n_boot=200, seed=0, repeats=True)
    d0 = metric(Y, A[0]) - metric(Y, B[0])
    d1 = metric(Y, A[1]) - metric(Y, B[1])
    assert rep.estimate == pytest.approx((d0 + d1) / 2)


def test_bootstrap_resamples_whole_patients():
    # A patient's records always travel together: with one record per cluster vs.
    # identical duplicated records, the duplicated clusters must yield the same spread.
    Y, S = _toy(n=200, k=1)
    metric = lambda y, s: roc_auc_score(y[:, 0], s[:, 0]) if 0 < y[:, 0].sum() < len(y) else np.nan  # noqa: E731
    one = paired_cluster_bootstrap(metric, Y, S + 0.3 * Y, S, np.arange(200), n_boot=300, seed=3)
    dup = paired_cluster_bootstrap(metric, np.repeat(Y, 2, 0), np.repeat(S + 0.3 * Y, 2, 0), np.repeat(S, 2, 0),
                                   np.repeat(np.arange(200), 2), n_boot=300, seed=3)
    assert dup.ci_high - dup.ci_low == pytest.approx(one.ci_high - one.ci_low, rel=1e-6)


def test_signal_batcher_matches_reference_and_keeps_row_order():
    rng = np.random.default_rng(0)
    sig = rng.integers(-2000, 2000, size=(20, 3, 100)).astype(np.int16)
    rows = np.array([7, 2, 15, 4])
    lab = np.arange(4)[:, None].astype(float)
    b = SignalBatcher(sig, rows, patch_len=25, labels=lab)
    idx = torch.arange(4)
    tok, y = b.batch(idx)
    ref = patchify_1d(zscore_per_lead(torch.from_numpy(sig[rows].astype(np.float32))), 25)
    assert torch.allclose(tok, ref, atol=1e-5)
    assert y[:, 0].tolist() == [0, 1, 2, 3]
    seen = torch.cat([i for i, _, _ in b.iterate(3, shuffle=True, generator=torch.Generator().manual_seed(0))])
    assert sorted(seen.tolist()) == [0, 1, 2, 3]


def test_probe_arm_freezes_the_encoder():
    enc = JEPAEncoder(token_dim=12, embed_dim=16, depth=1, n_heads=2, n_tokens=8)
    m = EncoderClassifier(enc, n_out=5, freeze_encoder=True)
    before = [p.clone() for p in enc.parameters()]
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    m.train()
    m(torch.randn(4, 8, 12)).sum().backward()
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, enc.parameters()))
    assert not m.encoder.training  # frozen encoder stays in eval mode
