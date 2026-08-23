"""Phase 0 — the Koopman-linearity probe. Is the frozen latent space already Koopman coordinates?

The co-adaptation program's premise is that the frozen JEPA encoder retained the perturbation signal
(a linear readout of the frozen latents scores Δr = 0.852, above the NB-VAE bar) but *not in linearizing
coordinates* (fitting the latent response demanded generators of ‖M‖≈12, a large rotation, not a small
near-identity motion). This script replaces that premise with a number, cheaply, before we spend the
engineering of Phase 2 (co-adapting the encoder with the operator under a Koopman objective).

The question (chapter 2 §10 of the action-operator series, made operational): on the *frozen* latents, is
the control→perturbed transformation **affine** — the coordinates close approximately linearly under the
intervention, which is exactly Koopman invariance — or **genuinely nonlinear**? A perturbed cloud that is
an affine image of the control cloud can be matched by a linear map; one that is a non-affine warp cannot,
but a flexible nonlinear map can. So the gap between the best affine match and the best nonlinear match is
a marginal-level estimate of how non-Koopman the coordinates are.

Why latent space only (no decoder, no Δr). §10 is explicit: *do not test this on the downstream benchmark.*
The ceiling analysis already showed the transition stage has ≤0.03 of Δr headroom, so Δr cannot distinguish
a linear latent transition from a nonlinear one. This probe stays entirely in latent space.

Why energy distance, not per-pair error. Sequencing is destructive, so the data is unpaired: there is no
"same cell before and after". Both arms are therefore graded by matching *marginals* with the energy
distance — the project's standing objective (``ssllab.generative.operator_perturb.energy_distance``) — which
needs no correspondence between the two clouds. The paired residual ‖r‖ of §10 is unavailable; this is the
honest surrogate the destructive regime affords, and it is the same objective every operator round trained on.

Why a PCA subspace. The frozen latent is 256-dim but a perturbation cloud has only a few hundred cells, so a
free 256×256 affine (or a 256-dim covariance) overfits noise directions and generalizes worse than doing
nothing — the same reason the ceiling's readout is ridge-regularized and the operator uses a low-rank basis.
The probe therefore works in the top-``--pca-dim`` principal subspace of a full-data sample (so the dominant
*response* directions are represented, not just control's intrinsic variance), where both arms are
well-determined and generalize. A linear projection preserves the affine-vs-nonlinear question in the
directions that carry the response; the discarded low-variance directions are where finite clouds cannot
measure anything anyway. The retained variance fraction is reported.

The instrument, per perturbation p (clouds with ≥ ``--min-cells`` cells):

    1. Encode the shared control cloud and p's perturbed cloud with the frozen encoder (row-exact).
    2. Project both onto the top-k PCA subspace; standardize per-PC by the control cloud (affine, so it
       cannot change the linear-vs-nonlinear verdict; it only conditions the optimizer).
    3. Split each cloud into fit / eval halves; split fit further into train / val — a generalization test
       so the flexible arm cannot memorize the marginal, and an honest early-stopping signal.
    4. Fit two maps on the fit halves (train minibatches, early-stopped on val), both minimizing the
       energy distance:
         L (affine):     z' = A z + b, initialized at the closed-form 2nd-moment-optimal (Bures/Gaussian-OT)
                         affine with covariance shrinkage so it cannot underfit, then gradient fine-tuned.
         N (nonlinear):  z' = z + MLP(z), a small residual MLP, zero-initialized (starts at identity),
                         weight-decayed and capacity-swept so a gap must be real structure, not the biggest MLP.
    5. Score both on the eval halves with the *unbiased* energy distance, against a reference ladder:
         E_id     = energy(ctrl_eval, pert_eval)          how far the perturbation moves the cloud
         E_floor  = energy(pert_eval_a, pert_eval_b)       finite-sample floor (perturbed eval split in two)
         E_shift  = energy(ctrl_eval + Δμ, pert_eval)      crudest match: align means only (= the Δ benchmark)
         E_lin    = energy(L(ctrl_eval), pert_eval)
         E_nl     = energy(N(ctrl_eval), pert_eval)
       reduction_x   = (E_id − E_x) / (E_id − E_floor)     fraction of the response arm x explains
       non_koopman_gap = reduction_nl − reduction_lin      ← the headline number

Strata (for statistical power — Round 4's lesson that the powered split is what confirms or kills): the
single-gene perturbations (the cleanest elementary interventions, the primary/powered set), the training
combos, and the 20 held-out combos. Each stratum is summarized with a bootstrap CI over its perturbations,
averaged across ``--seeds`` random cell-splits/inits.

The pre-registered decision rule (see the report's ``decision`` block) gates **Phase 2 only** — Phase 1
(E+decoder) runs regardless because it attacks the larger, already-measured decoder bottleneck.

Nothing is trained on the encoder here; it stays frozen. CPU, minutes.

Usage
-----
    python examples/perturbation_response/co-adaptation/01_koopman_linearity_probe.py \
        --encoder output/norman_stage_a/checkpoints/encoder.pt --split combo

    # fast smoke:
    python .../01_koopman_linearity_probe.py --limit-perts 3 --seeds 1 --steps 150 \
        --strata singles heldout_combos

Output
------
    output/<experiment>/reports/koopman_probe.json   (+ samples/koopman_gap.png if matplotlib is present)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.checkpoint import load_jepa
from ssllab.data.perturbseq import (
    DEFAULT_N_TOKENS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    load_cache,
    make_gene_partition,
)
from ssllab.experiment import experiment
from ssllab.generative.operator_perturb import energy_distance
from ssllab.latents import encode_rows
from ssllab.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STRATA = ("singles", "train_combos", "val_combos", "heldout_combos")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0: is the frozen latent space Koopman-linear? (E stays frozen)")
    p.add_argument("--experiment", type=str, default="norman_koopman_probe")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--encoder", type=str, default="output/norman_stage_a/checkpoints/encoder.pt")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"],
                   help="which split column defines the held-out combos")
    p.add_argument("--strata", type=str, nargs="+", default=list(STRATA), choices=list(STRATA))
    p.add_argument("--pca-dim", type=int, default=20, help="probe in the top-k PCA subspace (fits are well-posed here)")
    p.add_argument("--min-cells", type=int, default=150, help="skip perturbations with fewer cells")
    p.add_argument("--max-cells-per-pert", type=int, default=1000, help="cap encoded cells per perturbation")
    p.add_argument("--control-pool-size", type=int, default=3000, help="subsample the control cloud to this many cells")
    p.add_argument("--pca-per-pert", type=int, default=200, help="cells per perturbation contributed to the PCA fit")
    p.add_argument("--n-cloud", type=int, default=128, help="minibatch cloud size for energy-distance fitting")
    p.add_argument("--n-eval", type=int, default=512, help="control-eval cloud size cap for scoring")
    p.add_argument("--steps", type=int, default=400, help="gradient steps per map fit")
    p.add_argument("--lr", type=float, default=5e-3, help="Adam lr for the nonlinear arm")
    p.add_argument("--lr-affine", type=float, default=2e-3, help="Adam lr for the affine fine-tune (starts near-optimal)")
    p.add_argument("--wd", type=float, default=1e-3, help="weight decay for the nonlinear arm")
    p.add_argument("--wd-affine", type=float, default=1e-4, help="weight decay for the affine fine-tune")
    p.add_argument("--nl-hidden", type=int, default=64, help="headline residual-MLP hidden width")
    p.add_argument("--capacity-sweep", type=int, nargs="+", default=[8, 32, 128],
                   help="MLP hidden widths for the monotonicity check (run on --sweep-perts perturbations)")
    p.add_argument("--sweep-perts", type=int, default=8, help="how many singles to run the capacity sweep on")
    p.add_argument("--seeds", type=int, default=3, help="random cell-split / init seeds (one seed is directional only)")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for the CI across perturbations")
    p.add_argument("--null-tol", type=float, default=0.05,
                   help="acceptance gate: median synthetic-affine-null gap must be below this for the "
                        "artifact subtraction to be trustworthy (else the config overfits — lower --pca-dim)")
    p.add_argument("--limit-perts", type=int, default=None, help="cap perturbations per stratum (smoke runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-fig", action="store_true", help="skip the gap-distribution figure")
    return p.parse_args()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The two arms
# --------------------------------------------------------------------------- #
class AffineMap(nn.Module):
    """z' = A z + b. Initialized from a closed-form Bures (Gaussian-OT) map, then fine-tuned."""

    def __init__(self, A0: torch.Tensor, b0: torch.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A0.clone())
        self.b = nn.Parameter(b0.clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.A.transpose(-1, -2) + self.b


class ResidualMLP(nn.Module):
    """z' = z + MLP(z). Final layer zero-initialized, so it starts at the identity map."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


def _sym_sqrt(mat: torch.Tensor, inverse: bool, eps: float) -> torch.Tensor:
    """Symmetric (inverse) square root of a symmetric PSD matrix via eigendecomposition."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp(min=eps)
    scaled = evals.rsqrt() if inverse else evals.sqrt()
    return (evecs * scaled) @ evecs.transpose(-1, -2)


def _shrink_cov(Z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    """Covariance with Ledoit-Wolf-style shrinkage toward a scaled identity (well-posed in small samples)."""
    d = Z.shape[1]
    S = torch.cov(Z.T)
    target = (torch.diagonal(S).mean()) * torch.eye(d, dtype=S.dtype)
    return (1 - gamma) * S + gamma * target + eps * torch.eye(d, dtype=S.dtype)


def bures_affine(Zc: torch.Tensor, Zp: torch.Tensor, gamma: float = 0.1, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """The 2nd-moment-optimal affine map control→perturbed: the Monge map between the two Gaussians.

    A = Σc^{-1/2} (Σc^{1/2} Σp Σc^{1/2})^{1/2} Σc^{-1/2},  b = μp − A μc, with shrunk covariances so the
    square roots are well-conditioned from a few dozen cells. This is the affine map that optimally aligns
    the clouds' means and covariances, so it is a linear arm that cannot be accused of underfitting; any
    residual left after it is genuinely non-affine structure.
    """
    Zc, Zp = Zc.double(), Zp.double()
    mu_c, mu_p = Zc.mean(0), Zp.mean(0)
    Sc = _shrink_cov(Zc, gamma, eps)
    Sp = _shrink_cov(Zp, gamma, eps)
    Sc_half = _sym_sqrt(Sc, inverse=False, eps=eps)
    Sc_inv_half = _sym_sqrt(Sc, inverse=True, eps=eps)
    mid = _sym_sqrt(Sc_half @ Sp @ Sc_half, inverse=False, eps=eps)
    A = Sc_inv_half @ mid @ Sc_inv_half
    b = mu_p - A @ mu_c
    return A.float(), b.float()


# --------------------------------------------------------------------------- #
# Energy distance (unbiased scoring estimator)
# --------------------------------------------------------------------------- #
def energy_u(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased (U-statistic) energy distance — the scoring estimator.

    The project's ``energy_distance`` is the V-statistic (``cdist(x,x).mean()`` averages over all n² pairs,
    including the n zero self-pairs). That underestimates the within-cloud spread, and the underestimate
    grows as the sample shrinks — so a floor computed on half-size clouds comes out *larger* than E_id and
    the reduction ratios explode. Excluding the diagonal (dividing by n(n−1) instead of n²) makes the
    estimator unbiased for the population energy distance: it has expectation 0 for two samples from the
    same distribution at any size, so E_floor and the arms are comparable even at different cloud sizes.
    The V-statistic stays as the *fit* loss (a consistent minibatch objective across arms); scoring uses this.
    """
    n, m = x.shape[0], y.shape[0]
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).sum() / (n * (n - 1)) if n > 1 else x.new_tensor(0.0)
    d_yy = torch.cdist(y, y).sum() / (m * (m - 1)) if m > 1 else y.new_tensor(0.0)
    return 2.0 * d_xy - d_xx - d_yy


@torch.no_grad()
def _energy(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(energy_u(x, y))


def _sample(cloud: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    idx = torch.randint(len(cloud), (n,), generator=gen)
    return cloud[idx]


def fit_map(model: nn.Module, ctrl_tr: torch.Tensor, pert_tr: torch.Tensor,
            ctrl_val: torch.Tensor, pert_val: torch.Tensor,
            steps: int, lr: float, wd: float, n_cloud: int, gen: torch.Generator) -> nn.Module:
    """Fit a map to push control onto perturbed (energy distance), early-stopped on a held-out val split.

    Trains on the train minibatches with the V-statistic energy distance (the operator's objective) and
    keeps the parameters with the best *unbiased* energy on the val split, so the reported map is selected
    to generalize rather than to memorize the fit cloud — the guard that makes linear-vs-nonlinear honest.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_e = float("inf")
    for step in range(steps):
        opt.zero_grad()
        x = model(_sample(ctrl_tr, n_cloud, gen))
        loss = energy_distance(x, _sample(pert_tr, n_cloud, gen))
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                e = float(energy_u(model(ctrl_val), pert_val))
            if e < best_e:
                best_e, best_state = e, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


# --------------------------------------------------------------------------- #
# Per-perturbation scoring
# --------------------------------------------------------------------------- #
def _floor(pert_eval: torch.Tensor, gen: torch.Generator, reps: int = 6) -> float:
    """Finite-sample energy-distance floor: split the perturbed eval cloud in two, averaged over reps."""
    n = len(pert_eval)
    vals = []
    for _ in range(reps):
        perm = torch.randperm(n, generator=gen)
        h = n // 2
        vals.append(_energy(pert_eval[perm[:h]], pert_eval[perm[h:2 * h]]))
    return float(np.mean(vals))


def _three_way(cloud: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a cloud into (train, val, eval): fit half = 50% (train 40% / val 10%), eval half = 50%."""
    perm = torch.randperm(len(cloud), generator=gen)
    n = len(cloud)
    n_eval = n // 2
    n_val = max(1, (n - n_eval) // 5)
    ev = cloud[perm[:n_eval]]
    val = cloud[perm[n_eval:n_eval + n_val]]
    tr = cloud[perm[n_eval + n_val:]]
    return tr, val, ev


def score_perturbation(ctrl_k: torch.Tensor, pert_k: torch.Tensor, args, gen: torch.Generator,
                       hidden: int) -> dict:
    """Fit L and N on fit halves, score the arm ladder on eval halves (all in the PCA subspace)."""
    ctrl_tr, ctrl_val, ctrl_eval = _three_way(ctrl_k, gen)
    pert_tr, pert_val, pert_eval = _three_way(pert_k, gen)
    if len(ctrl_eval) > args.n_eval:
        ctrl_eval = _sample(ctrl_eval, args.n_eval, gen)
    dim = ctrl_k.shape[1]

    # Arm L: affine, initialized at the Bures map, gradient fine-tuned (early-stopped on val).
    A0, b0 = bures_affine(ctrl_tr, pert_tr)
    lin = fit_map(AffineMap(A0, b0), ctrl_tr, pert_tr, ctrl_val, pert_val,
                  args.steps, args.lr_affine, args.wd_affine, args.n_cloud, gen)

    # Arm N: residual MLP, zero-init (identity), gradient-fit (early-stopped on val).
    nl = fit_map(ResidualMLP(dim, hidden), ctrl_tr, pert_tr, ctrl_val, pert_val,
                 args.steps, args.lr, args.wd, args.n_cloud, gen)

    # Reference ladder + arms, all scored on the held-out eval halves.
    e_id = _energy(ctrl_eval, pert_eval)
    e_floor = _floor(pert_eval, gen)
    dmu = pert_tr.mean(0) - ctrl_tr.mean(0)                        # mean-shift from TRAIN cells (no leakage)
    e_shift = _energy(ctrl_eval + dmu, pert_eval)
    with torch.no_grad():
        e_lin = _energy(lin(ctrl_eval), pert_eval)
        e_nl = _energy(nl(ctrl_eval), pert_eval)

    denom = e_id - e_floor
    no_signal = denom <= 0.05 * max(abs(e_id), 1e-8)              # cloud barely moves beyond noise
    def reduction(e):
        return None if no_signal else (e_id - e) / denom
    return {
        "n_train": len(pert_tr), "n_eval_pert": len(pert_eval), "n_eval_ctrl": len(ctrl_eval),
        "E_id": e_id, "E_floor": e_floor, "E_shift": e_shift, "E_lin": e_lin, "E_nl": e_nl,
        "reduction_shift": reduction(e_shift), "reduction_lin": reduction(e_lin), "reduction_nl": reduction(e_nl),
        "linear_residual_frac": None if no_signal else (e_lin - e_floor) / denom,
        "non_koopman_gap": None if no_signal else (e_lin - e_nl) / denom,
        "no_signal": bool(no_signal),
    }


def make_affine_null(ctrl_k: torch.Tensor, pert_k: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """A synthetic perturbed cloud that is a *known* affine image of the control distribution.

    Fit the Bures affine control→perturbation, then apply it to a fresh independent control sample of the
    perturbation's size. The result satisfies ``z = A z_ctrl + b`` exactly, so the true nonlinear gap is 0
    and its mean/covariance match the real perturbation (so the reduction denominator is comparable). Any
    gap the probe reports on this cloud is the *overfitting artifact* of fitting maps from a few hundred
    cells in the PCA subspace — the quantity we subtract to isolate genuine non-affine structure.
    """
    n = len(pert_k)
    ref = _sample(ctrl_k, min(len(ctrl_k), 4 * n), gen)          # cells used to estimate A, b
    A, b = bures_affine(ref, pert_k)
    src = _sample(ctrl_k, n, gen)                                # independent source cells
    return src @ A.transpose(-1, -2) + b


def probe_perturbation(ctrl_k: torch.Tensor, pert_k: torch.Tensor, args, gen: torch.Generator,
                       hidden: int) -> dict:
    """Score a perturbation and its matched synthetic-affine null; the excess is the calibrated signal."""
    real = score_perturbation(ctrl_k, pert_k, args, gen, hidden)
    syn = make_affine_null(ctrl_k, pert_k, gen)
    null = score_perturbation(ctrl_k, syn, args, gen, hidden)

    def sub(a, b):
        return None if (a is None or b is None) else a - b
    return {
        "gap_real": real["non_koopman_gap"], "gap_null": null["non_koopman_gap"],
        "gap_excess": sub(real["non_koopman_gap"], null["non_koopman_gap"]),
        "lin_resid_real": real["linear_residual_frac"], "lin_resid_null": null["linear_residual_frac"],
        "lin_resid_excess": sub(real["linear_residual_frac"], null["linear_residual_frac"]),
        "reduction_shift": real["reduction_shift"], "reduction_lin": real["reduction_lin"],
        "reduction_nl": real["reduction_nl"], "E_id": real["E_id"], "E_floor": real["E_floor"],
        "no_signal": real["no_signal"],
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _median_ci(values: list, B: int, gen: np.random.Generator) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"median": None, "ci_lo": None, "ci_hi": None, "n": 0}
    boot = [np.median(gen.choice(v, size=len(v), replace=True)) for _ in range(B)]
    return {"median": float(np.median(v)), "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)), "n": int(len(v))}


def _mean_over_seeds(per_seed: list) -> dict:
    keys = ["gap_real", "gap_null", "gap_excess", "lin_resid_real", "lin_resid_null", "lin_resid_excess",
            "reduction_shift", "reduction_lin", "reduction_nl", "E_id", "E_floor"]
    out = {}
    for k in keys:
        vals = [s[k] for s in per_seed if s is not None and s[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out["no_signal"] = all(s is None or s["no_signal"] for s in per_seed)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    exp = experiment(args.experiment, args.output_root).ensure()

    enc_path = Path(args.encoder)
    if not enc_path.exists():
        raise FileNotFoundError(f"missing encoder checkpoint: {enc_path}")

    jepa = load_jepa(enc_path, device)
    cache = load_cache(Path(args.data_dir) / args.artifact)
    tok = cache.manifest.get("tokenization", {})
    partition = make_gene_partition(cache.n_hvg, int(tok.get("n_tokens", DEFAULT_N_TOKENS)),
                                    int(tok.get("partition_seed", 0)))
    hvg = cache.hvg_X
    is_control = cache.is_control.astype(bool)
    pert_id = cache.pert_id
    pert_names = list(cache.pert_names)
    split_col = cache.split_combo if args.split == "combo" else cache.split_cells

    logger.info("encoder %s (md5 %s)  split=%s  %d genes", enc_path, _md5(enc_path)[:8], args.split, cache.n_hvg)

    rng = np.random.default_rng(args.seed)

    # --- encode the shared control cloud (raw encoder space) -------------------------------------- #
    ctrl_rows = np.flatnonzero(is_control)
    if len(ctrl_rows) > args.control_pool_size:
        ctrl_rows = np.sort(rng.choice(ctrl_rows, size=args.control_pool_size, replace=False))
    Z_ctrl_raw = encode_rows(jepa, hvg, partition, ctrl_rows, device)
    logger.info("control cloud: %d cells encoded (latent dim %d)", len(Z_ctrl_raw), Z_ctrl_raw.shape[1])

    # --- classify perturbations into strata + encode each cloud once ------------------------------ #
    def stratum_of(name: str, code: int) -> str | None:
        if name == "control":
            return None
        if "+" not in name:
            return "singles"
        return {SPLIT_TRAIN: "train_combos", SPLIT_VAL: "val_combos", SPLIT_TEST: "heldout_combos"}.get(int(code))

    perts_by_stratum: dict[str, list] = {s: [] for s in STRATA}
    for pid, name in enumerate(pert_names):
        rows = np.flatnonzero((pert_id == pid) & ~is_control)
        if len(rows) < args.min_cells:
            continue
        s = stratum_of(name, int(split_col[rows[0]]))             # splits are perturbation-level
        if s in perts_by_stratum:
            perts_by_stratum[s].append((pid, name))

    active_strata = [s for s in args.strata if perts_by_stratum.get(s)]
    for s in active_strata:
        if args.limit_perts is not None:
            perts_by_stratum[s] = perts_by_stratum[s][: args.limit_perts]
        logger.info("stratum %-15s %d perturbations", s, len(perts_by_stratum[s]))

    want = {pid for s in active_strata for pid, _ in perts_by_stratum[s]}
    raw_clouds: dict[int, torch.Tensor] = {}
    for pid in sorted(want):
        rows = np.flatnonzero((pert_id == pid) & ~is_control)
        if len(rows) > args.max_cells_per_pert:
            rows = np.sort(rng.choice(rows, size=args.max_cells_per_pert, replace=False))
        raw_clouds[pid] = encode_rows(jepa, hvg, partition, rows, device)
    logger.info("encoded %d perturbation clouds", len(raw_clouds))

    # --- fit PCA on a full-data sample (so response directions are represented) ------------------- #
    pool = [Z_ctrl_raw]
    for pid, Zr in raw_clouds.items():
        take = min(len(Zr), args.pca_per_pert)
        idx = torch.from_numpy(np.sort(rng.choice(len(Zr), size=take, replace=False)))
        pool.append(Zr[idx])
    pool = torch.cat(pool, 0)
    pool_mean = pool.mean(0)
    U, S, Vt = torch.linalg.svd(pool - pool_mean, full_matrices=False)
    k = min(args.pca_dim, Vt.shape[0])
    V = Vt[:k].T                                                  # (D, k)
    explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
    logger.info("PCA subspace k=%d  retained variance %.3f  (pool %d cells)", k, explained, len(pool))

    ctrl_proj0 = (Z_ctrl_raw - pool_mean) @ V
    pc_std = ctrl_proj0.std(0) + 1e-6                             # standardize per-PC by the control cloud

    def project(Zr: torch.Tensor) -> torch.Tensor:
        return ((Zr - pool_mean) @ V) / pc_std

    Z_ctrl = project(Z_ctrl_raw)
    clouds_k = {pid: project(Zr) for pid, Zr in raw_clouds.items()}

    # --- run the probe over strata × perturbations × seeds (real + synthetic-affine null) --------- #
    per_pert_records: list = []
    seed_stratum_excess: dict[str, list] = {s: [[] for _ in range(args.seeds)] for s in active_strata}
    for s in active_strata:
        for pid, name in perts_by_stratum[s]:
            cloud = clouds_k[pid]
            per_seed = []
            for si in range(args.seeds):
                gen = torch.Generator().manual_seed(1000 * args.seed + 97 * si + pid)
                m = probe_perturbation(Z_ctrl, cloud, args, gen, args.nl_hidden)
                per_seed.append(m)
                seed_stratum_excess[s][si].append(m["gap_excess"])
            agg = _mean_over_seeds(per_seed)
            per_pert_records.append({"pert": name, "stratum": s, "n_cells_used": int(len(cloud)), **agg})
        logger.info("scored stratum %-15s (%d perturbations)", s, len(perts_by_stratum[s]))

    # --- capacity sweep on a subset of singles (monotonicity guard) ------------------------------- #
    capacity_sweep = {}
    sweep_pool = perts_by_stratum.get("singles", [])[: args.sweep_perts] if "singles" in active_strata else []
    for pid, name in sweep_pool:
        row = {}
        for h in args.capacity_sweep:
            gen = torch.Generator().manual_seed(1000 * args.seed + pid + h)
            m = probe_perturbation(Z_ctrl, clouds_k[pid], args, gen, h)
            row[str(h)] = {"reduction_nl": m["reduction_nl"], "gap_excess": m["gap_excess"]}
        capacity_sweep[name] = row
    if sweep_pool:
        logger.info("capacity sweep on %d singles over hidden=%s", len(sweep_pool), args.capacity_sweep)

    # --- null sanity: control vs control should carry no signal (no_signal), an acceptance check --- #
    perm = torch.randperm(len(Z_ctrl), generator=torch.Generator().manual_seed(args.seed))
    half = len(Z_ctrl) // 2
    gen = torch.Generator().manual_seed(args.seed)
    null_metrics = score_perturbation(Z_ctrl[perm[:half]], Z_ctrl[perm[half:2 * half]], args, gen, args.nl_hidden)
    logger.info("null check (control vs control): E_id=%.4f E_floor=%.4f no_signal=%s",
                null_metrics["E_id"], null_metrics["E_floor"], null_metrics["no_signal"])

    # --- summarize each stratum ------------------------------------------------------------------- #
    boot_rng = np.random.default_rng(args.seed)
    strata_summary = {}
    for s in active_strata:
        recs = [r for r in per_pert_records if r["stratum"] == s]
        seed_medians = []
        for si in range(args.seeds):
            vals = [g for g in seed_stratum_excess[s][si] if g is not None]
            if vals:
                seed_medians.append(float(np.median(vals)))
        strata_summary[s] = {
            "n_perturbations": len(recs),
            "n_no_signal": sum(1 for r in recs if r["no_signal"]),
            "gap_excess": _median_ci([r["gap_excess"] for r in recs], args.bootstrap, boot_rng),
            "gap_real": _median_ci([r["gap_real"] for r in recs], args.bootstrap, boot_rng)["median"],
            "gap_null": _median_ci([r["gap_null"] for r in recs], args.bootstrap, boot_rng)["median"],
            "lin_resid_real": _median_ci([r["lin_resid_real"] for r in recs], args.bootstrap, boot_rng)["median"],
            "lin_resid_null": _median_ci([r["lin_resid_null"] for r in recs], args.bootstrap, boot_rng)["median"],
            "lin_resid_excess": _median_ci([r["lin_resid_excess"] for r in recs], args.bootstrap, boot_rng),
            "median_reduction_lin": _median_ci([r["reduction_lin"] for r in recs], args.bootstrap, boot_rng)["median"],
            "median_reduction_nl": _median_ci([r["reduction_nl"] for r in recs], args.bootstrap, boot_rng)["median"],
            "seed_spread_gap_excess": float(np.std(seed_medians)) if len(seed_medians) > 1 else None,
        }

    # --- the pre-registered decision (gates Phase 2 only) ----------------------------------------- #
    decision = evaluate_decision(strata_summary, args.null_tol)
    logger.info("DECISION (Phase 2 gate): %s", decision["verdict"])
    logger.info("  rationale: %s", decision["rationale"])

    # --- write report ----------------------------------------------------------------------------- #
    report = {
        "phase": "0 — Koopman-linearity probe",
        "config": vars(args),
        "encoder": str(enc_path), "encoder_md5": _md5(enc_path),
        "latent_dim": int(Z_ctrl_raw.shape[1]), "pca_dim": k, "pca_retained_variance": explained,
        "n_control_encoded": int(len(Z_ctrl)), "headline_capacity": args.nl_hidden, "seeds": args.seeds,
        "metric_note": ("non_koopman_gap = reduction_nl − reduction_lin, in [~0,1] fraction-of-response units. "
                        "gap_excess = gap_real − gap_null subtracts the overfitting artifact measured on a "
                        "matched synthetic AFFINE null (true gap 0). gap_excess is the calibrated signal the "
                        "decision uses."),
        "strata_summary": strata_summary,
        "capacity_sweep": capacity_sweep,
        "null_check": null_metrics,
        "decision": decision,
        "per_perturbation": sorted(per_pert_records, key=lambda r: (r["stratum"], r["pert"])),
    }
    exp.write_report("koopman_probe", report)
    logger.info("wrote report -> %s", exp.reports / "koopman_probe.json")

    if not args.no_fig:
        try:
            _make_figure(per_pert_records, active_strata, exp.samples / "koopman_gap.png")
            logger.info("wrote figure -> %s", exp.samples / "koopman_gap.png")
        except Exception as e:  # a missing/broken matplotlib must not kill the diagnostic
            logger.warning("figure skipped: %s", e)

    # console summary
    logger.info("")
    logger.info("%-16s  %9s  %8s  %8s  %10s  %5s  %s",
                "stratum", "gap_exces", "gap_real", "gap_null", "lin_resid", "n", "gap_excess 95% CI")
    for s in active_strata:
        ge = strata_summary[s]["gap_excess"]
        logger.info("%-16s  %9.3f  %8.3f  %8.3f  %10.3f  %5d  [%.3f, %.3f]",
                    s, _nan(ge["median"]), _nan(strata_summary[s]["gap_real"]), _nan(strata_summary[s]["gap_null"]),
                    _nan(strata_summary[s]["lin_resid_real"]), ge["n"], _nan(ge["ci_lo"]), _nan(ge["ci_hi"]))


def _nan(x):
    return float("nan") if x is None else x


def evaluate_decision(strata_summary: dict, null_tol: float) -> dict:
    """The pre-registered rule, evaluated on the singles stratum (the powered primary set).

    The metric is ``gap_excess`` = (nonlinear's advantage over affine on the real perturbation) − (the same
    advantage measured on a matched synthetic AFFINE null, where the truth is affine by construction). The
    subtraction removes the overfitting artifact of fitting maps from a few hundred cells, so ``gap_excess``
    isolates *genuine* non-affine structure and is robust to the PCA dimension. (The plan pre-registered a
    rule on the raw gap; it is updated here to the calibrated ``gap_excess`` because the control and
    synthetic-null checks revealed the raw gap is confounded by overfitting — a methodological correction
    made before reading the aggregate result, not a response to it.)

    Acceptance gate first: the synthetic-affine null measures how much gap the fitting manufactures when the
    truth is exactly affine, at this pca-dim and cell count. If that artifact (median ``gap_null``) is not
    itself small (≤ ``--null-tol``), the subtraction is being asked to correct too much and ``gap_excess`` is
    unreliable — the honest move is to lower --pca-dim (or raise --min-cells) until the artifact is small.

    Phase 2 (E+operator, Koopman objective) is JUSTIFIED iff the nonlinear arm reliably beats affine beyond
    the artifact: median ``gap_excess`` ≥ 0.10 with a bootstrap CI excluding 0. It is NOT justified if the
    excess is ≈ 0 (median < 0.03 and CI includes 0): the response is affine wherever it can be measured and
    the bottleneck is downstream (the decoder). Otherwise inconclusive. Phase 0 gates Phase 2 only; Phase 1
    runs regardless.
    """
    primary = "singles" if "singles" in strata_summary else next(iter(strata_summary), None)
    if primary is None:
        return {"stratum": None, "verdict": "inconclusive", "rationale": "no strata scored"}

    gap_null = strata_summary[primary]["gap_null"]
    gate_ok = gap_null is not None and gap_null <= null_tol
    ge = strata_summary[primary]["gap_excess"]
    gm, glo = ge["median"], ge["ci_lo"]
    lrr = strata_summary[primary]["lin_resid_real"]
    lrn = strata_summary[primary]["lin_resid_null"]
    if gm is None:
        return {"stratum": primary, "verdict": "inconclusive", "null_gate_passed": gate_ok,
                "rationale": "primary stratum had no signal-bearing perturbations"}

    if not gate_ok:
        verdict = "inconclusive"
        rationale = (f"ACCEPTANCE GATE FAILED: the synthetic-affine-null artifact gap_null={gap_null:.3f} "
                     f"exceeds --null-tol={null_tol:.3f}; the fits overfit at this pca-dim/min-cells and "
                     f"gap_excess={gm:.2f} cannot be trusted. Lower --pca-dim or raise --min-cells and re-run.")
    elif glo is not None and glo > 0.0 and gm >= 0.10:
        verdict = "justified"
        rationale = (f"beyond the affine-null artifact, the nonlinear arm adds median gap_excess={gm:.2f} "
                     f"(95% CI [{glo:.2f}, {ge['ci_hi']:.2f}] excludes 0); affine leaves {lrr:.2f} of the "
                     f"response on real vs {lrn:.2f} on a known-affine null → the frozen coordinates are not "
                     f"Koopman-invariant and Phase 2 is justified.")
    elif gm < 0.03 and (glo is None or glo <= 0.0):
        verdict = "not_justified"
        rationale = (f"gap_excess={gm:.2f}≈0 with CI including 0, and the affine arm leaves about as much on "
                     f"real ({lrr:.2f}) as on a known-affine null ({lrn:.2f}) → the response is affine wherever "
                     f"it can be measured; the bottleneck is downstream (decoder). Phase 2 low priority.")
    else:
        verdict = "inconclusive"
        rationale = (f"gap_excess median={gm:.2f} (CI lo={glo}); neither the justified nor the not-justified "
                     f"threshold is met. Proceed with Phase 1 regardless; revisit Phase 2.")
    return {"stratum": primary, "verdict": verdict, "null_gate_passed": gate_ok,
            "gap_null_median": gap_null, "null_tol": null_tol,
            "gap_excess_median": gm, "gap_excess_ci_lo": glo, "gap_excess_ci_hi": ge["ci_hi"],
            "lin_resid_real_median": lrr, "lin_resid_null_median": lrn, "rationale": rationale}


def _make_figure(records: list, strata: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data, labels = [], []
    for s in strata:
        vals = [r["gap_excess"] for r in records if r["stratum"] == s and r["gap_excess"] is not None]
        if vals:
            data.append(vals); labels.append(f"{s}\n(n={len(vals)})")
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        for i, vals in enumerate(data, start=1):
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, alpha=0.4, color="tab:blue")
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.axhline(0.10, color="tab:red", lw=0.8, ls=":", label="justified threshold (0.10)")
    ax.set_ylabel("gap_excess  (nonlinear − affine, artifact-corrected)")
    ax.set_title("Phase 0: non-affine response structure beyond the overfitting null")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
