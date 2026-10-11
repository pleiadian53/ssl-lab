"""Generic 1D-signal helpers: physical-time geometry, per-lead normalization, patching.

The time-series sibling of :func:`ssllab.data.mnist.patchify`. A multi-lead signal
``(C, L)`` is cut along time into ``N`` non-overlapping patches of ``P`` samples, and
each patch, across all leads, becomes one token:

    (B, C, L)  ->  (B, N, C * P)        with  L = N * P

Inside a token the layout is lead-major: the first ``P`` values are lead 0's
samples for that time window, the next ``P`` are lead 1's, and so on.

Geometry is specified in **physical time** (Hz and seconds), never in samples, so
the same configuration reads correctly at any sampling rate. The integer
quantities (samples per patch, tokens per record, tokens per masked span) are
derived, and a ratio that is not a whole number is refused rather than rounded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

_TOL = 1e-6


def _whole(x: float, what: str) -> int:
    r = round(x)
    if r < 1 or abs(x - r) > _TOL:
        raise ValueError(f"{what} must be a positive whole number, got {x:g}")
    return int(r)


@dataclass(frozen=True)
class SignalGeometry:
    """Token geometry in physical time.

    fs:
        Sampling rate in Hz.
    duration_s:
        Length of one input window in seconds.
    patch_s:
        Duration of one token's time window in seconds.
    span_s:
        Duration of one masked target span in seconds (SSL only; ``None`` when unused).
    """

    fs: float
    duration_s: float = 10.0
    patch_s: float = 0.25
    span_s: float | None = None

    def __post_init__(self) -> None:
        # Validate every derived integer up front.
        _ = self.n_samples, self.patch_len, self.n_tokens, self.span_tokens

    @property
    def n_samples(self) -> int:
        return _whole(self.fs * self.duration_s, f"fs*duration_s ({self.fs:g} Hz x {self.duration_s:g} s)")

    @property
    def patch_len(self) -> int:
        return _whole(self.fs * self.patch_s, f"fs*patch_s ({self.fs:g} Hz x {self.patch_s:g} s)")

    @property
    def n_tokens(self) -> int:
        return _whole(self.duration_s / self.patch_s, f"duration_s/patch_s ({self.duration_s:g} s / {self.patch_s:g} s)")

    @property
    def span_tokens(self) -> int | None:
        if self.span_s is None:
            return None
        n = _whole(self.span_s / self.patch_s, f"span_s/patch_s ({self.span_s:g} s / {self.patch_s:g} s)")
        if n >= self.n_tokens:
            raise ValueError(f"span_s={self.span_s:g} s leaves no context in a {self.duration_s:g} s window")
        return n

    def token_dim(self, n_leads: int) -> int:
        return n_leads * self.patch_len

    def describe(self, n_leads: int) -> dict:
        """Physical settings plus the derived integers, for manifests and run reports."""
        return {
            **asdict(self),
            "n_samples": self.n_samples,
            "patch_len": self.patch_len,
            "n_tokens": self.n_tokens,
            "span_tokens": self.span_tokens,
            "n_leads": n_leads,
            "token_dim": self.token_dim(n_leads),
        }


def zscore_per_lead(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize each lead of each record over time: ``(..., C, L)`` -> same shape.

    A flat lead (standard deviation below ``eps``, e.g. a disconnected electrode)
    maps to zeros instead of being divided by ~0.
    """
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True, unbiased=False)
    flat = sd < eps
    return torch.where(flat, torch.zeros_like(x), (x - mu) / sd.clamp_min(eps))


def patchify_1d(x: torch.Tensor, patch_len: int) -> torch.Tensor:
    """``(B, C, L)`` -> ``(B, N, C * patch_len)``, lead-major within each token."""
    B, C, L = x.shape
    if L % patch_len:
        raise ValueError(f"length {L} is not a multiple of patch_len {patch_len}")
    N = L // patch_len
    return x.reshape(B, C, N, patch_len).permute(0, 2, 1, 3).reshape(B, N, C * patch_len)


def unpatchify_1d(tokens: torch.Tensor, n_leads: int) -> torch.Tensor:
    """Inverse of :func:`patchify_1d`: ``(B, N, C * P)`` -> ``(B, C, N * P)``."""
    B, N, D = tokens.shape
    if D % n_leads:
        raise ValueError(f"token_dim {D} is not divisible by n_leads {n_leads}")
    P = D // n_leads
    return tokens.reshape(B, N, n_leads, P).permute(0, 2, 1, 3).reshape(B, n_leads, N * P)
