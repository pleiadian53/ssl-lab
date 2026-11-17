"""Regenerate assets/stadium-billiard.svg, the figure in the ergodic-theorem box of chapter 2.

A frictionless ball on a Bunimovich stadium: a rectangle of half-width A and half-height 1, capped by two
semicircles of radius 1 centred at (+A, 0) and (-A, 0). The ball moves in straight lines at constant speed
and reflects off the wall with angle of incidence equal to angle of reflection. Because the speed is
constant, the time spent in a region is proportional to the path length inside it, so the running time
average of "the ball is on the left half" is the left-half share of the path travelled so far.

Panel 1 draws the first bounces of a typical launch. Panel 2 plots the running time average against the
path length for that launch, which settles to the state average of one half, and for a trapped launch
fired straight across between the two flat sides, which stays at one forever. Panel 3 draws the trapped
launch.

Usage:
    python examples/perturbation_response/docs/action-operator/assets/make_stadium_billiard.py \
        examples/perturbation_response/docs/action-operator/assets/stadium-billiard.svg
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = 1.0                      # half-width of the straight section
N_BOUNCES = 20000            # long run for the running average
N_DRAWN = 45                 # bounces drawn in panel 1
EPS = 1e-9
BLUE, ORANGE, GREY, INK, SHADE = "#2563eb", "#ea580c", "#6b7280", "#111827", "#dbeafe"


def next_hit(p, d):
    """Earliest wall intersection of the ray p + t d, t > 0, and the outward wall normal there."""
    best_t, best_n = np.inf, None
    for y_wall, sign in ((1.0, 1.0), (-1.0, -1.0)):           # flat top and bottom walls
        if d[1] * sign > EPS:
            t = (y_wall - p[1]) / d[1]
            if t > EPS and abs(p[0] + t * d[0]) <= A and t < best_t:
                best_t, best_n = t, np.array([0.0, sign])
    for cx in (A, -A):                                          # semicircular caps
        c = np.array([cx, 0.0])
        q = p - c
        b = d @ q
        disc = b * b - (q @ q - 1.0)
        if disc < 0:
            continue
        for t in (-b - np.sqrt(disc), -b + np.sqrt(disc)):
            hit = p + t * d
            on_cap = hit[0] >= A - EPS if cx > 0 else hit[0] <= -A + EPS
            if t > EPS and on_cap and t < best_t:
                best_t, best_n = t, (hit - c)
    return best_t, best_n / np.linalg.norm(best_n)


def simulate(p, angle, n):
    """Bounce points and the running left-half time fraction after each segment."""
    d = np.array([np.cos(angle), np.sin(angle)])
    pts, total, left, frac, length = [p.copy()], 0.0, 0.0, [], []
    for _ in range(n):
        t, nrm = next_hit(p, d)
        q = p + t * d
        x0, x1 = p[0], q[0]
        if x0 < 0 and x1 < 0:
            share = 1.0
        elif x0 >= 0 and x1 >= 0:
            share = 0.0
        else:
            cross = x0 / (x0 - x1)                               # fraction of the segment before x = 0
            share = cross if x0 < 0 else 1.0 - cross
        total += t
        left += share * t
        frac.append(left / total)
        length.append(total)
        d = d - 2 * (d @ nrm) * nrm
        p = q
        pts.append(p.copy())
    return np.array(pts), np.array(length), np.array(frac)


def stadium_outline():
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    right = np.c_[A + np.cos(th), np.sin(th)]
    left = np.c_[-A - np.cos(th), -np.sin(th)]
    return np.vstack([right, left, right[:1]])


def draw_table(ax):
    outline = stadium_outline()
    ax.fill(outline[:, 0], outline[:, 1], color="white", zorder=0)
    lx = outline[outline[:, 0] <= 0]
    ax.fill(np.r_[lx[:, 0], 0, 0], np.r_[lx[:, 1], -1, 1], color=SHADE, lw=0, zorder=1)
    ax.plot(outline[:, 0], outline[:, 1], color=INK, lw=1.6, zorder=3)
    ax.plot([0, 0], [-1, 1], color=GREY, lw=0.8, ls="--", zorder=2)
    ax.text(-A - 0.45, 0, "left\nhalf", color=BLUE, ha="center", va="center", fontsize=9.5, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-A - 1.15, A + 1.15)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")


typical_pts, typical_len, typical_frac = simulate(np.array([0.3, 0.2]), 0.7, N_BOUNCES)
trapped_start = np.array([-0.5, 0.0])
trapped_pts, trapped_len, trapped_frac = simulate(trapped_start, np.pi / 2, N_BOUNCES)
print(f"typical launch, time on left half after {N_BOUNCES} bounces: {typical_frac[-1]:.4f}")
print(f"trapped launch, time on left half: {trapped_frac[-1]:.4f}")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "svg.hashsalt": "stadium-billiard"})
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})

draw_table(ax[0])
ax[0].plot(typical_pts[:N_DRAWN, 0], typical_pts[:N_DRAWN, 1], color=ORANGE, lw=0.8, alpha=0.85, zorder=2)
ax[0].set_title("1. A typical launch fills the table", fontsize=11, color=INK, loc="left")

ax[1].plot(typical_len, typical_frac, color=ORANGE, lw=1.6, label="typical launch")
ax[1].plot(trapped_len, trapped_frac, color=GREY, lw=1.6, label="trapped launch")
ax[1].axhline(0.5, color=BLUE, lw=1.2, ls="--")
ax[1].text(typical_len[-1], 0.535, "state average = ½", color=BLUE, ha="right", fontsize=9.5)
ax[1].set_xscale("log")
ax[1].set_ylim(0, 1.08)
ax[1].set_xlabel("time watched (path length, log scale)")
ax[1].set_ylabel("share of time on left half")
ax[1].set_title("2. The time average settles", fontsize=11, color=INK, loc="left")
ax[1].legend(frameon=False, fontsize=9, loc="lower right")
for side in ("top", "right"):
    ax[1].spines[side].set_visible(False)

draw_table(ax[2])
ax[2].plot([trapped_start[0]] * 2, [-1, 1], color=GREY, lw=2.2, zorder=2)
ax[2].annotate("", xy=(trapped_start[0] + 0.18, 0.65), xytext=(trapped_start[0] + 0.18, -0.65),
               arrowprops={"arrowstyle": "<->", "color": GREY, "lw": 1.0}, zorder=4)
ax[2].set_title("3. A rare launch is trapped", fontsize=11, color=INK, loc="left")

fig.tight_layout()
fig.savefig(sys.argv[1], metadata={"Date": None} if sys.argv[1].endswith(".svg") else None, dpi=110)
print(f"wrote {sys.argv[1]}")
