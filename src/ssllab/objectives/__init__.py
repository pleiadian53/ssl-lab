"""Training objectives."""

from ssllab.objectives.jepa_loss import jepa_loss, prediction_loss, vicreg_reg

__all__ = ["prediction_loss", "vicreg_reg", "jepa_loss"]
