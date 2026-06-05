"""Evaluation: collapse diagnostics, linear probe, generative-sample metrics, viz."""

from ssllab.eval.collapse import collapse_report, effective_rank, feature_std
from ssllab.eval.generative import (
    classifier_metrics,
    density_coverage,
    fid,
    kid,
    nn_distance_stats,
    precision_recall,
)
from ssllab.eval.probe import linear_probe
from ssllab.eval.viz import save_image_grid

__all__ = [
    "feature_std",
    "effective_rank",
    "collapse_report",
    "linear_probe",
    "save_image_grid",
    "fid",
    "kid",
    "precision_recall",
    "density_coverage",
    "nn_distance_stats",
    "classifier_metrics",
]
