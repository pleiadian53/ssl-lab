"""Regenerate assets/one-gaussian-two-fates.svg, the figure in chapter 3a.

Toy model: a one-dimensional latent z ~ N(0, 1); a decoder whose mean switches between two fates,
f(z) = m1 + (m2 - m1) * sigmoid(z / tau); and x | z ~ N(f(z), s^2). The output density
p(x) = integral of N(x; f(z), s^2) times the standard-normal density of z is computed by quadrature.
A sharp decoder (small tau) gives a bimodal p(x); a smooth one (large tau) blurs the fates together.

Usage:
    python examples/perturbation_response/docs/conditional-flow-jepa/assets/make_one_gaussian_two_fates.py \
        examples/perturbation_response/docs/conditional-flow-jepa/assets/one-gaussian-two-fates.svg
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M1, M2, S = -2.0, 2.0, 0.45
TAU_SHARP, TAU_SMOOTH = 0.08, 1.2
BLUE, ORANGE, GREY, INK = "#2563eb", "#ea580c", "#6b7280", "#111827"

z = np.linspace(-3.5, 3.5, 1401)
phi = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
x = np.linspace(-4.5, 4.5, 501)


def f(zz, tau):
    return M1 + (M2 - M1) / (1 + np.exp(-zz / tau))


def p_x(tau):
    mean = f(z, tau)[None, :]
    k = np.exp(-(x[:, None] - mean) ** 2 / (2 * S**2)) / (S * np.sqrt(2 * np.pi))
    return np.trapezoid(k * phi[None, :], z, axis=1)


def halves(tau):
    """Output density split by which half of the latent space produced it."""
    out = []
    for mask in (z < 0, z >= 0):
        mean = f(z[mask], tau)[None, :]
        k = np.exp(-(x[:, None] - mean) ** 2 / (2 * S**2)) / (S * np.sqrt(2 * np.pi))
        out.append(np.trapezoid(k * phi[mask][None, :], z[mask], axis=1))
    return out


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "svg.fonttype": "none", "svg.hashsalt": "ssl-lab-3a"})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))

# Panel 1: the prior, halves colored
ax[0].fill_between(z, phi, where=z < 0, color=BLUE, alpha=0.35, lw=0)
ax[0].fill_between(z, phi, where=z >= 0, color=ORANGE, alpha=0.35, lw=0)
ax[0].plot(z, phi, color=INK, lw=1.4)
ax[0].set_title("1. Prior: one Gaussian over the latent $z$", fontsize=11, color=INK, loc="left")
ax[0].set_xlabel("latent $z$")
ax[0].set_yticks([])
ax[0].text(-2.55, 0.17, "half that will\nbecome fate 1", color=BLUE, ha="center", fontsize=9.5)
ax[0].text(2.55, 0.17, "half that will\nbecome fate 2", color=ORANGE, ha="center", fontsize=9.5)

# Panel 2: the decoder mean, sharp vs smooth
ax[1].plot(z, f(z, TAU_SHARP), color=INK, lw=2, label="sharp decoder")
ax[1].plot(z, f(z, TAU_SMOOTH), color=GREY, lw=1.6, ls="--", label="smooth decoder")
ax[1].axhline(M1, color=BLUE, lw=0.8, alpha=0.6)
ax[1].axhline(M2, color=ORANGE, lw=0.8, alpha=0.6)
ax[1].set_title("2. Decoder mean $f(z)$", fontsize=11, color=INK, loc="left")
ax[1].set_xlabel("latent $z$")
ax[1].set_ylabel("expression $x$")
ax[1].legend(frameon=False, fontsize=9, loc="lower right")

# Panel 3: the output distribution
lo, hi = halves(TAU_SHARP)
ax[2].fill_between(x, lo, color=BLUE, alpha=0.35, lw=0)
ax[2].fill_between(x, hi, color=ORANGE, alpha=0.35, lw=0)
ax[2].plot(x, p_x(TAU_SHARP), color=INK, lw=1.8, label="sharp decoder: two fates")
ax[2].plot(x, p_x(TAU_SMOOTH), color=GREY, lw=1.6, ls="--", label="smooth decoder: fates blur")
ax[2].set_title("3. Output $p(x)$: two fates from one Gaussian", fontsize=11, color=INK, loc="left")
ax[2].set_xlabel("expression $x$")
ax[2].set_yticks([])
ax[2].legend(frameon=False, fontsize=9, loc="upper center")
ax[2].set_ylim(0, max(p_x(TAU_SHARP)) * 1.35)

for a in ax:
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
out = sys.argv[1]
fig.savefig(out, format="svg", facecolor="white", metadata={"Date": None})
pmax = max(p_x(TAU_SHARP))
print("sharp: density at x=0 / at mode:", float(np.interp(0, x, p_x(TAU_SHARP)) / pmax))
print("smooth: density at x=0 / at its max:", float(np.interp(0, x, p_x(TAU_SMOOTH)) / max(p_x(TAU_SMOOTH))))
print("total mass sharp, smooth:", float(np.trapezoid(p_x(TAU_SHARP), x)), float(np.trapezoid(p_x(TAU_SMOOTH), x)))
