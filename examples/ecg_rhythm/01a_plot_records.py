"""Render real PTB-XL recordings for reading and teaching (no training).

Example records are chosen by a fixed rule, never by eye. For each rhythm code,
take the training-fold records (folds 1-8; validation and test are never browsed)
that carry exactly that one rhythm statement, have no signal-quality flag and a
human-validated label, and keep the lowest ``ecg_id``. Paced records may carry the
``pacemaker`` flag and bigeminy records the ``extra_beats`` flag, since each describes
the rhythm itself. The sinus example must
also carry the diagnostic ``NORM`` statement.

Figures
-------
    fig01_twelve_lead_sinus.png    all 12 leads of the sinus example, 10 s
    fig02_lead_ii_gallery.png      lead II for each rhythm, with R peaks and RR intervals
    fig03_rate_100_vs_500.png      one beat at 500 Hz versus the official 100 Hz copy
    fig04_tokens_and_span.png      250 ms patch boundaries and a 1 s masked span
    fig05_zscore.png               three leads before and after per-lead z-scoring
    examples.json                  the chosen records, their labels and RR statistics

The R-peak detector is a small Pan-Tompkins-style heuristic, good enough to read RR
intervals off clean examples; it is for illustration, not measurement.

Usage
-----
    python examples/ecg_rhythm/01a_plot_records.py
    python examples/ecg_rhythm/01a_plot_records.py --copy-to "<tutorial folder>/figures"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.ptbxl import QUALITY_FIELDS, SPLIT_TRAIN, load_cache
from ssllab.data.signal import SignalGeometry, zscore_per_lead
from ssllab.data.wfdb16 import read_format16, to_physical
from ssllab.experiment import experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GALLERY = ("SR", "SBRAD", "STACH", "SARRH", "AFIB", "AFLT", "PACE", "BIGU")


def choose_examples(cache) -> dict[str, int]:
    rl, split = cache.rhythm(), cache.split
    chosen = {}
    for code in GALLERY:
        j = rl.codes.index(code)
        best = None
        for i, r in enumerate(cache.records):
            if split[i] != SPLIT_TRAIN or rl.y[i].sum() != 1 or rl.y[i, j] != 1 or not r["validated_by_human"]:
                continue
            flags = [q for q in QUALITY_FIELDS if r[q]]
            # A flag that describes the rhythm itself is allowed for that rhythm.
            allowed = {"PACE": "pacemaker", "BIGU": "extra_beats"}.get(code)
            flags = [q for q in flags if q != allowed]
            if flags or (code == "SR" and "NORM" not in r["scp_codes"]):
                continue
            if best is None or r["ecg_id"] < cache.records[best]["ecg_id"]:
                best = i
        if best is not None:
            chosen[code] = best
    return chosen


def r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins-style R peaks: band-pass, squared slope, moving-window energy,
    then the largest absolute deflection near each energy peak."""
    b, a = butter(2, [5.0 / (fs / 2), min(20.0, 0.45 * fs) / (fs / 2)], btype="band")
    y = filtfilt(b, a, x)
    e = np.convolve(np.gradient(y) ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same")
    cand, _ = find_peaks(e, distance=int(0.25 * fs), height=0.3 * np.percentile(e, 98))
    hb, ha = butter(2, 0.5 / (fs / 2), btype="high")
    xh = filtfilt(hb, ha, x)
    w = int(0.08 * fs)
    return np.array(sorted({max(0, c - w) + int(np.argmax(np.abs(xh[max(0, c - w) : c + w + 1]))) for c in cand}), dtype=int)


def rr_stats(pk: np.ndarray, fs: float) -> dict:
    rr = np.diff(pk) / fs * 1000.0
    if rr.size < 2:
        return {"n_beats": int(pk.size)}
    return {
        "n_beats": int(pk.size),
        "rr_mean_ms": round(float(rr.mean()), 1),
        "rr_sd_ms": round(float(rr.std(ddof=1)), 1),
        "rr_cv": round(float(rr.std(ddof=1) / rr.mean()), 3),
        "heart_rate_bpm": round(60000.0 / float(rr.mean()), 1),
        "successive_rr_diff_mean_ms": round(float(np.abs(np.diff(rr)).mean()), 1),
    }


def _grid(ax, t_max, lo, hi):
    for x in np.arange(0, t_max + 1e-9, 0.2):
        ax.axvline(x, color="#f4c7c3", lw=0.4, zorder=0)
    for y in np.arange(np.floor(lo * 2) / 2, hi + 1e-9, 0.5):
        ax.axhline(y, color="#f4c7c3", lw=0.4, zorder=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--artifact", default="ptbxl100")
    p.add_argument("--experiment", default="ecg_rhythm")
    p.add_argument("--output-root", default="output")
    p.add_argument("--raw-dir", type=Path, default=None, help="PhysioNet tree, for the 500 Hz copy (default: from the manifest)")
    p.add_argument("--copy-to", type=Path, default=None, help="also copy the figures here")
    args = p.parse_args()

    cache = load_cache(args.data_dir / args.artifact)
    fs = cache.fs
    raw_dir = args.raw_dir or Path(cache.manifest["source"]["raw_dir"])
    lead_ii = cache.leads.index("II")
    t = np.arange(cache.signals.shape[-1]) / fs
    out = experiment(args.experiment, args.output_root).root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    mv = lambda i: cache.signals[i].astype(np.float32) / 1000.0  # noqa: E731  (microvolts -> mV)

    chosen = choose_examples(cache)
    logger.info("examples: %s", {c: cache.records[i]["ecg_id"] for c, i in chosen.items()})
    summary = {}
    for code, i in chosen.items():
        r = cache.records[i]
        pk = r_peaks(mv(i)[lead_ii], fs)
        summary[code] = {
            "ecg_id": r["ecg_id"], "patient_id": r["patient_id"], "strat_fold": r["strat_fold"],
            "scp_codes": r["scp_codes"], "age": r["age"], "sex": r["sex"],
            "quality_flags": {q: r[q] for q in QUALITY_FIELDS if r[q]},
            "lead_ii": rr_stats(pk, fs),
        }

    # fig01: twelve leads of the sinus example.
    i = chosen["SR"]
    x = mv(i)
    fig, axes = plt.subplots(12, 1, figsize=(12, 14), sharex=True)
    for k, ax in enumerate(axes):
        _grid(ax, t[-1], x[k].min(), x[k].max())
        ax.plot(t, x[k], color="black", lw=0.7)
        ax.set_ylabel(cache.leads[k], rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"PTB-XL ecg_id {cache.records[i]['ecg_id']}: sinus rhythm, 12 leads, {fs:g} Hz (grid: 0.2 s x 0.5 mV)")
    fig.tight_layout()
    fig.savefig(out / "fig01_twelve_lead_sinus.png", dpi=130)
    plt.close(fig)

    # fig02: lead II gallery with R peaks and RR intervals.
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.0 * len(chosen)), sharex=True)
    for ax, (code, i) in zip(np.atleast_1d(axes), chosen.items()):
        y = mv(i)[lead_ii]
        pk = r_peaks(y, fs)
        ax.plot(t, y, color="black", lw=0.7)
        ax.plot(t[pk], y[pk], "o", ms=3, color="#c0392b")
        for a, b in zip(pk[:-1], pk[1:]):
            ax.text((t[a] + t[b]) / 2, y.max(), f"{(b - a) / fs * 1000:.0f}", ha="center", va="bottom", fontsize=6, color="#555")
        s = summary[code]["lead_ii"]
        ax.set_title(f"{code}  (ecg_id {cache.records[i]['ecg_id']}, HR {s.get('heart_rate_bpm', float('nan')):.0f} bpm, "
                     f"RR CV {s.get('rr_cv', float('nan')):.2f})", fontsize=9, loc="left")
        ax.set_ylabel("mV")
    np.atleast_1d(axes)[-1].set_xlabel("time (s); numbers are RR intervals in ms")
    fig.tight_layout()
    fig.savefig(out / "fig02_lead_ii_gallery.png", dpi=130)
    plt.close(fig)

    # fig03: one beat at 500 Hz vs the official 100 Hz copy.
    i = chosen["SR"]
    hr_path = raw_dir / cache.records[i]["filename"].replace("records100", "records500").replace("_lr", "_hr")
    if hr_path.with_suffix(".hea").exists():
        d, hdr = read_format16(hr_path)
        x5 = to_physical(d, hdr)[lead_ii]
        t5 = np.arange(x5.size) / hdr.fs
        y1 = mv(i)[lead_ii]
        pk = r_peaks(y1, fs)
        c = t[pk[len(pk) // 2]]
        w5, w1 = (t5 > c - 0.45) & (t5 < c + 0.55), (t > c - 0.45) & (t < c + 0.55)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t5[w5], x5[w5], color="#7f8c8d", lw=1.2, label=f"500 Hz ({int(w5.sum())} samples)")
        ax.plot(t[w1], y1[w1], "o-", color="#c0392b", ms=3, lw=0.8, label=f"100 Hz ({int(w1.sum())} samples)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("lead II (mV)")
        ax.set_title(f"One beat, ecg_id {cache.records[i]['ecg_id']}: 2 ms vs 10 ms sample spacing")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fig03_rate_100_vs_500.png", dpi=130)
        plt.close(fig)
        summary["_rate_check"] = {
            "ecg_id": cache.records[i]["ecg_id"],
            "lead_ii_peak_mV_500hz": round(float(x5[w5].max()), 3),
            "lead_ii_peak_mV_100hz": round(float(y1[w1].max()), 3),
        }
    else:
        logger.warning("500 Hz record not found at %s; skipping fig03", hr_path)

    # fig04: patch boundaries and one masked span, on the AF example.
    g = SignalGeometry(fs=fs, duration_s=10.0, patch_s=0.25, span_s=1.0)
    i = chosen.get("AFIB", chosen["SR"])
    y = mv(i)[lead_ii]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, y, color="black", lw=0.7)
    for n in range(g.n_tokens + 1):
        ax.axvline(n * g.patch_s, color="#2980b9", lw=0.4, alpha=0.6)
    s0 = 16
    ax.axvspan(s0 * g.patch_s, (s0 + g.span_tokens) * g.patch_s, color="#e67e22", alpha=0.25, label="1 s target span (4 tokens)")
    ax.set_title(f"Lead II cut into {g.n_tokens} tokens of {g.patch_s * 1000:.0f} ms ({g.patch_len} samples each); "
                 f"12-lead token width = {g.token_dim(12)}", fontsize=9, loc="left")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig04_tokens_and_span.png", dpi=130)
    plt.close(fig)

    # fig05: per-lead z-scoring removes amplitude and offset differences between leads.
    import torch

    i = chosen["SR"]
    x = mv(i)
    z = zscore_per_lead(torch.from_numpy(x)).numpy()
    show = [cache.leads.index(n) for n in ("I", "AVR", "V2")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharex=True)
    for k in show:
        axes[0].plot(t, x[k], lw=0.7, label=cache.leads[k])
        axes[1].plot(t, z[k], lw=0.7, label=cache.leads[k])
    axes[0].set_title("raw (mV)", fontsize=9)
    axes[1].set_title("per-lead z-scored (unitless)", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 3)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig05_zscore.png", dpi=130)
    plt.close(fig)

    # fig06: a paced beat at both rates (is a pacing spike visible, and does 100 Hz keep it?).
    if "PACE" in chosen:
        i = chosen["PACE"]
        hr_path = raw_dir / cache.records[i]["filename"].replace("records100", "records500").replace("_lr", "_hr")
        if hr_path.with_suffix(".hea").exists():
            d, hdr = read_format16(hr_path)
            x5, x1 = to_physical(d, hdr), mv(i)
            t5 = np.arange(x5.shape[1]) / hdr.fs
            c = t[r_peaks(x1[lead_ii], fs)[0]]
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            for ax, lead in zip(axes, ("II", "V2")):
                k = cache.leads.index(lead)
                w5, w1 = (t5 > c - 0.3) & (t5 < c + 0.5), (t > c - 0.3) & (t < c + 0.5)
                ax.plot(t5[w5], x5[k][w5], color="#7f8c8d", lw=1.0, label="500 Hz")
                ax.plot(t[w1], x1[k][w1], "o-", color="#c0392b", ms=3, lw=0.8, label="100 Hz")
                ax.set_ylabel(f"{lead} (mV)")
                ax.legend(fontsize=8)
            axes[0].set_title(f"Paced beat, ecg_id {cache.records[i]['ecg_id']}, at 500 Hz and 100 Hz", fontsize=10)
            axes[1].set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(out / "fig06_paced_beat_rates.png", dpi=130)
            plt.close(fig)

    (out / "examples.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote figures to %s", out)
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for f in sorted(out.iterdir()):
            shutil.copy2(f, args.copy_to / f.name)
        logger.info("copied to %s", args.copy_to)


if __name__ == "__main__":
    main()
