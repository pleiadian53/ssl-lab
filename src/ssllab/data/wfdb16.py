"""A dependency-free reader (and writer) for WFDB records stored in format 16.

PhysioNet distributes PTB-XL as WFDB records: a text header (``.hea``) and a
binary signal file (``.dat``). PTB-XL uses only the simplest storage format,
**format 16**: little-endian ``int16`` samples, interleaved frame by frame
(sample ``t`` of lead 0, lead 1, ..., then sample ``t+1``). That is a few lines
of numpy, so we read it directly instead of adding the ``wfdb`` package.

The header carries two integrity fields per signal, the first sample value and a
16-bit checksum of all samples (their sum modulo 2**16). :func:`read_format16` verifies both by default,
so a truncated download or a mis-parsed interleave fails loudly.

Physical units: ``physical = (digital - baseline) / gain``. PTB-XL headers give
``gain = 1000`` per mV and ``baseline = 0``, so the stored integers are microvolts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# "1000.0(0)/mV" -> gain 1000.0, baseline 0, units mV. Baseline and units optional.
_GAIN_RE = re.compile(r"^(?P<gain>[-+0-9.eE]+)(?:\((?P<baseline>[-+0-9]+)\))?(?:/(?P<units>\S+))?$")


@dataclass(frozen=True)
class WfdbHeader:
    record_name: str
    n_sig: int
    fs: float
    n_samples: int
    dat_file: str
    gains: tuple[float, ...]
    baselines: tuple[int, ...]
    units: tuple[str, ...]
    init_values: tuple[int | None, ...]
    checksums: tuple[int | None, ...]
    sig_names: tuple[str, ...]


def _checksum16(x: np.ndarray) -> int:
    """WFDB checksum: the sum of all samples modulo 2**16, as PhysioNet writes it (0..65535).

    Headers may also carry the signed form (-32768..32767); compare both sides
    modulo 2**16, as :func:`read_format16` does.
    """
    return int(x.astype(np.int64).sum()) % 65536


def read_header(hea_path: str | Path) -> WfdbHeader:
    """Parse a single-segment WFDB header whose signals all live in one format-16 file."""
    lines = [ln.strip() for ln in Path(hea_path).read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    rec = lines[0].split()
    name, n_sig = rec[0], int(rec[1])
    fs = float(rec[2].split("/")[0]) if len(rec) > 2 else 250.0  # WFDB default 250 Hz
    n_samples = int(rec[3]) if len(rec) > 3 else 0
    if "/" in name:
        raise ValueError(f"{hea_path}: multi-segment records are not supported")

    dat_files, gains, baselines, units, inits, checks, names = [], [], [], [], [], [], []
    for ln in lines[1 : 1 + n_sig]:
        f = ln.split()
        # Plain "16" only: samples-per-frame (x), skew (:) and byte offset (+) would
        # all break the simple frame-major reshape below.
        if f[1] != "16":
            raise ValueError(f"{hea_path}: format {f[1]!r} is not supported (plain format 16 only)")
        m = _GAIN_RE.match(f[2]) if len(f) > 2 else None
        if m is None:
            raise ValueError(f"{hea_path}: cannot parse gain field {f[2] if len(f) > 2 else ''!r}")
        gain = float(m["gain"]) or 200.0  # WFDB: gain 0 means the default, 200
        dat_files.append(f[0])
        gains.append(gain)
        # WFDB: an omitted baseline defaults to the ADC zero (field 5), else 0.
        adc_zero = int(f[4]) if len(f) > 4 else 0
        baselines.append(int(m["baseline"]) if m["baseline"] is not None else adc_zero)
        units.append(m["units"] or "mV")
        inits.append(int(f[5]) if len(f) > 5 else None)
        checks.append(int(f[6]) if len(f) > 6 else None)
        names.append(" ".join(f[8:]) if len(f) > 8 else f"sig{len(names)}")
    if len(set(dat_files)) != 1:
        raise ValueError(f"{hea_path}: signals split across several .dat files are not supported")
    return WfdbHeader(
        record_name=name, n_sig=n_sig, fs=fs, n_samples=n_samples, dat_file=dat_files[0],
        gains=tuple(gains), baselines=tuple(baselines), units=tuple(units),
        init_values=tuple(inits), checksums=tuple(checks), sig_names=tuple(names),
    )


def read_format16(record_path: str | Path, verify: bool = True) -> tuple[np.ndarray, WfdbHeader]:
    """Read a record (path without extension). Returns ``(digital int16 (n_sig, n_samples), header)``.

    With ``verify``, the first sample and the checksum of every signal must match
    the header; a mismatch raises ``ValueError``.
    """
    record_path = Path(record_path)
    hdr = read_header(record_path.with_suffix(".hea"))
    raw = np.fromfile(record_path.parent / hdr.dat_file, dtype="<i2")
    if hdr.n_samples and raw.size != hdr.n_sig * hdr.n_samples:
        raise ValueError(
            f"{record_path}: {raw.size} samples on disk, header expects {hdr.n_sig} x {hdr.n_samples}"
        )
    x = raw.reshape(-1, hdr.n_sig).T.copy()  # frames -> (n_sig, n_samples)
    if verify:
        for i in range(hdr.n_sig):
            if hdr.init_values[i] is not None and int(x[i, 0]) != hdr.init_values[i]:
                raise ValueError(f"{record_path}: signal {i} first sample {x[i, 0]} != header {hdr.init_values[i]}")
            if hdr.checksums[i] is not None and _checksum16(x[i]) != hdr.checksums[i] % 65536:
                raise ValueError(f"{record_path}: signal {i} checksum mismatch")
    return x, hdr


def to_physical(digital: np.ndarray, hdr: WfdbHeader) -> np.ndarray:
    """Digital ``(n_sig, n_samples)`` -> physical units (float32), per the header's gain and baseline."""
    gain = np.asarray(hdr.gains, dtype=np.float32)[:, None]
    base = np.asarray(hdr.baselines, dtype=np.float32)[:, None]
    return (digital.astype(np.float32) - base) / gain


def write_format16(
    record_path: str | Path,
    digital: np.ndarray,
    fs: float,
    sig_names: list[str] | tuple[str, ...],
    gain: float = 1000.0,
    units: str = "mV",
) -> Path:
    """Write ``digital int16 (n_sig, n_samples)`` as a WFDB format-16 record (header + .dat).

    Used to build synthetic PTB-XL-shaped trees for smoke runs and tests.
    """
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    digital = np.asarray(digital, dtype=np.int16)
    n_sig, n_samples = digital.shape
    name = record_path.name
    digital.T.astype("<i2").tofile(record_path.with_suffix(".dat"))
    fs_txt = f"{fs:g}"
    lines = [f"{name} {n_sig} {fs_txt} {n_samples}"]
    for i in range(n_sig):
        lines.append(
            f"{name}.dat 16 {gain:.1f}(0)/{units} 16 0 {int(digital[i, 0])} "
            f"{_checksum16(digital[i])} 0 {sig_names[i]}"
        )
    record_path.with_suffix(".hea").write_text("\n".join(lines) + "\n")
    return record_path
