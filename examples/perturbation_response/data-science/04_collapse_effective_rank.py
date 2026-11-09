"""What effective rank measures, and why a random-slice penalty tolerates a low-rank embedding.

Two diagnostics guard the encoder against collapse, and they catch different failures.

**feature_std** (``ssllab.eval.collapse.feature_std``) is the mean per-dimension standard
deviation. It catches the crudest collapse, where embeddings stop varying at all.

**effective rank** (``ssllab.eval.collapse.effective_rank``) is

    exp(entropy of the normalized singular values of the centered embedding matrix)

which ranges over [1, D]. It catches the collapse feature_std cannot see: embeddings that vary
plenty but only along a few directions. Read ``exp(entropy)`` as an *effective count*, the same
construction as perplexity in language modeling or a Hill number in ecology. If the variance is
spread evenly over k directions and absent from the rest, the answer is k; if it all sits in
one direction, the answer is 1, however large the variance is.

The second half of this script is about Round 9. The SIGReg arms trained to embeddings with
about 20 dimensions of variance out of 256, and the post-hoc question was how an isotropy
penalty permitted that. SIGReg (``ssllab.objectives.jepa_loss.sigreg``) tests whether the
projection onto each of many **random unit directions** looks like N(0, 1), averaged over
directions. The demonstration below is that this is a statement about *marginals*, and a
low-rank Gaussian whose total variance is right has marginals that are very nearly N(0, 1)
along almost every direction. The penalty is satisfied by something it was meant to forbid.

Everything is computed with the production functions; nothing here reimplements a statistic.

Sections
--------
    1. effective rank on spectra built to order
    2. the two collapse modes, and why one diagnostic is not enough
    3. what a random direction sees: SIGReg on low-rank embeddings
    4. real encoder latents (optional, needs --encoder)

Usage
-----
    python examples/perturbation_response/data-science/04_collapse_effective_rank.py
    python .../04_collapse_effective_rank.py --encoder output/norman_stage_a/checkpoints/encoder.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
for _p in (str(_HERE), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ssllab.eval.collapse import effective_rank, feature_std  # noqa: E402
from ssllab.objectives.jepa_loss import sigreg  # noqa: E402


def _low_rank(n: int, d: int, rank: int, *, match_total: bool, gen: torch.Generator) -> torch.Tensor:
    """``(n, d)`` Gaussian supported on ``rank`` random orthonormal directions.

    ``match_total=False`` gives each active direction unit variance, so total variance is
    ``rank``. ``match_total=True`` scales the active directions so the total variance is ``d``,
    the same as a full-rank standard Gaussian. Section 3 turns on exactly this difference.
    """
    basis = torch.linalg.qr(torch.randn(d, d, generator=gen))[0][:, :rank]  # (d, rank)
    coef = torch.randn(n, rank, generator=gen)
    if match_total:
        coef = coef * np.sqrt(d / rank)
    return coef @ basis.T


def section_1_spectra(n: int, d: int, seed: int) -> None:
    print("=" * 78)
    print(f"1. Effective rank on spectra built to order (n={n}, D={d})")
    print("=" * 78)
    gen = torch.Generator().manual_seed(seed)
    cases: list[tuple[str, torch.Tensor, str]] = [
        ("isotropic, full rank", torch.randn(n, d, generator=gen), f"should be near {d}"),
        ("rank 20", _low_rank(n, d, 20, match_total=False, gen=gen), "should be near 20"),
        ("rank 2", _low_rank(n, d, 2, match_total=False, gen=gen), "should be near 2"),
        ("rank 1", _low_rank(n, d, 1, match_total=False, gen=gen), "should be near 1"),
    ]
    # One direction with huge variance plus isotropic noise: high std, low rank.
    z = torch.randn(n, d, generator=gen)
    z[:, 0] *= 30.0
    cases.append(("1 dominant + noise", z, "variance is large but concentrated"))
    # Constant embeddings: the textbook collapse.
    cases.append(("constant (collapsed)", torch.zeros(n, d) + 0.7, "no variance at all"))
    # Geometric decay: a realistic spectrum.
    decay = torch.randn(n, d, generator=gen) * torch.logspace(0, -2, d).unsqueeze(0)
    cases.append(("geometric decay", decay, "a realistic spectrum"))

    print(f"\n  {'embedding':<24}{'feature_std':>13}{'eff. rank':>11}   note")
    for label, zz, note in cases:
        er = effective_rank(zz)
        print(f"  {label:<24}{feature_std(zz):>13.3f}{er:>11.1f}   {note}")
    print("\n  The last three rows are the point. 'Constant' is caught by feature_std (0.000).")
    print("  '1 dominant + noise' is NOT: its feature_std is the largest in the table, and only")
    print("  effective rank reveals that the variance lives in essentially one direction.")
    print("  A realistic encoder looks like the decay row: no hard cutoff, so effective rank")
    print("  is a soft count rather than a threshold, and is only meaningful compared against")
    print("  itself across runs.")


def section_2_modes(n: int, d: int, seed: int) -> None:
    print("\n" + "=" * 78)
    print("2. The two collapse modes")
    print("=" * 78)
    gen = torch.Generator().manual_seed(seed + 1)
    print("\n  Interpolating from full rank to rank 1, holding TOTAL variance fixed, so")
    print("  feature_std stays put while the geometry degenerates:\n")
    print(f"  {'rank':>6}{'feature_std':>13}{'eff. rank':>11}{'eff.rank/D':>12}")
    for rank in [d, 128, 64, 20, 8, 2, 1]:
        z = _low_rank(n, d, rank, match_total=True, gen=gen) if rank < d else torch.randn(n, d, generator=gen)
        print(f"  {rank:>6}{feature_std(z):>13.3f}{effective_rank(z):>11.1f}"
              f"{effective_rank(z) / d:>12.2f}")
    print("\n  feature_std barely moves across the whole range. Any monitor that watches only")
    print("  the standard deviation would report a healthy encoder at rank 1.")
    print("\n  Reference points from the real runs: VICReg Stage A reports effective rank 176")
    print("  of 256 with feature_std 0.755, and the SIGReg arms of Round 9 came out near 20.")
    print("  Note effective rank is batch and split dependent, which is why the same encoder")
    print("  is logged at 176 end-of-pretraining and 213 on the probe's 8,000-cell subset.")


def section_3_random_slices(n: int, d: int, seed: int, n_slices: int) -> None:
    print("\n" + "=" * 78)
    print("3. What a random direction sees")
    print("=" * 78)
    gen = torch.Generator().manual_seed(seed + 2)
    dgen = torch.Generator().manual_seed(seed + 99)  # fixed directions, so arms are comparable
    print(f"\n  SIGReg averages an Epps-Pulley statistic for N(0, 1) over {n_slices} random unit")
    print("  directions. It is NOT centered or standardized first, so matching N(0, I)")
    print("  includes having unit variance. Two ways to be rank 20 in 256 dimensions:\n")

    def stat(z: torch.Tensor) -> float:
        dgen.manual_seed(seed + 99)
        return float(sigreg(z, n_slices=n_slices, generator=dgen)[0])

    full = torch.randn(n, d, generator=gen)
    lr_unit = _low_rank(n, d, 20, match_total=False, gen=gen)
    lr_match = _low_rank(n, d, 20, match_total=True, gen=gen)
    rows = [
        ("full rank, N(0, I)", full, "the target the penalty asks for"),
        ("rank 20, unit per dim", lr_unit, "total variance only 20/256"),
        ("rank 20, total matched", lr_match, "total variance is right"),
        ("full rank, variance 4", 2.0 * full, "isotropic but wrong scale"),
        ("full rank, shifted", full + 1.5, "right shape, wrong mean"),
    ]
    dirs = torch.randn(d, 2048, generator=torch.Generator().manual_seed(7))
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    print(f"  {'embedding':<26}{'eff.rank':>9}{'slice var':>11}{'sd':>8}{'SIGReg':>10}   note")
    for label, z, note in rows:
        v = (z @ dirs).var(dim=0)
        print(f"  {label:<26}{effective_rank(z):>9.1f}{float(v.mean()):>11.3f}"
              f"{float(v.std()):>8.3f}{stat(z):>10.3f}   {note}")

    base, pen_unit, pen_match = stat(full), stat(lr_unit), stat(lr_match)
    print("\n  The 'sd' column is the mechanism, and it is worth being precise here rather")
    print("  than saying low rank is invisible, because it is not.")
    print("\n  Every one-dimensional projection of a Gaussian is Gaussian whatever the rank, so")
    print("  the SHAPE of a slice carries no rank information at all. What leaks is the slice's")
    print("  VARIANCE. For a rank-r embedding whose total variance is matched, a random")
    print("  direction picks up a fraction of the active subspace that fluctuates, with")
    print("  relative spread about sqrt(2/r):\n")
    for r in (256, 20, 8):
        print(f"    r = {r:<4} predicted sd {np.sqrt(2 / r):.3f}")
    print("\n  which the table confirms at r = 20 (measured sd ~0.31 against 0.316 predicted).")
    print("  At full rank the covariance is exactly isotropic, so every slice has variance 1")
    print("  and only sampling noise remains.")
    print(f"\n  So the penalty does see low rank: the matched rank-20 embedding scores")
    print(f"  {pen_match:.1f} against the target's {base:.2f}. The point is how that compares with")
    print(f"  what else the statistic is measuring. A global scale error scores {stat(2.0 * full):.0f} and a")
    print(f"  mean shift {stat(full + 1.5):.0f}, one to two orders of magnitude more. In a convex")
    print("  objective (1 - lambda)*prediction + lambda*SIGReg, the gradient is dominated by")
    print("  getting the mean and the scale right, and low rank is a comparatively faint")
    print("  signal that a prediction term can afford to trade against.")
    print("\n  That is the shape of the Round 9 outcome: encoders that satisfied the penalty")
    print("  while spending roughly 236 of 256 directions on nothing. The numbers here are an")
    print("  isolation of the statistic on synthetic Gaussians, NOT a reproduction of that")
    print("  run's figures (the round reports 3.85 against 1.07, on real embeddings with a")
    print("  different N and slice count); see the round 9 page for the authoritative numbers.")
    print("\n  Caveat on scope: this says a random-slice N(0, I) test is weak against low rank,")
    print("  not that LeJEPA's full method is. The paper's objective pairs it with a prediction")
    print("  term and other machinery that this isolation strips away.")


def section_4_real(encoder: str, data_dir: str, artifact: str, limit: int) -> None:
    print("\n" + "=" * 78)
    print("4. Real encoder latents")
    print("=" * 78)
    path = Path(encoder)
    if not path.exists():
        print(f"\n  {path} not found; skipping. Train Stage A first, or pass --encoder.")
        return
    from _data import load_narrow
    from ssllab.checkpoint import load_jepa
    from ssllab.data.perturbseq import DEFAULT_N_TOKENS, make_gene_partition
    from ssllab.latents import encode_rows
    from ssllab.utils import get_device

    data = load_narrow(Path(data_dir) / artifact, "hvg_X")
    device = get_device("cpu")
    jepa = load_jepa(path, device)
    tok = data.manifest.get("tokenization", {})
    partition = make_gene_partition(
        int(data.manifest["n_hvg"]), int(tok.get("n_tokens", DEFAULT_N_TOKENS)),
        int(tok.get("partition_seed", 0)),
    )
    rows = np.arange(min(limit, data.hvg_X.shape[0]))
    z = encode_rows(jepa, data.hvg_X, partition, rows, device, 256)
    print(f"\n  {path}")
    print(f"  {len(rows):,} cells -> z {tuple(z.shape)}")
    print(f"    feature_std      {feature_std(z):.4f}")
    print(f"    effective rank   {effective_rank(z):.1f} of {z.shape[1]}")
    print(f"    SIGReg statistic {float(sigreg(z, n_slices=256)[0]):.3f}")
    print("\n  Compare the effective rank against the run's own stage_a_train.json rather than")
    print("  against another run's: it depends on the batch and the split, as section 2 noted.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=2000, help="synthetic embeddings per case")
    p.add_argument("--dim", type=int, default=256, help="embedding width, as Stage A uses")
    p.add_argument("--slices", type=int, default=512, help="random directions for SIGReg")
    p.add_argument("--encoder", type=str, default="output/norman_stage_a/checkpoints/encoder.pt")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019")
    p.add_argument("--limit", type=int, default=4000, help="cells to encode in section 4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sections", type=str, default="1,2,3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    want = {s.strip() for s in args.sections.split(",")}
    if "1" in want:
        section_1_spectra(args.n, args.dim, args.seed)
    if "2" in want:
        section_2_modes(args.n, args.dim, args.seed)
    if "3" in want:
        section_3_random_slices(args.n, args.dim, args.seed, args.slices)
    if "4" in want:
        section_4_real(args.encoder, args.data_dir, args.artifact, args.limit)
    print()


if __name__ == "__main__":
    main()
