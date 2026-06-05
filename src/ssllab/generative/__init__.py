"""Generative head: a sampleable prior over the JEPA latent."""

from ssllab.generative.flow import VelocityMLP, cfm_loss, euler_sample, linear_interpolant

__all__ = ["VelocityMLP", "linear_interpolant", "cfm_loss", "euler_sample"]
