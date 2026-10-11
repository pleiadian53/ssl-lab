"""The dependency-free WFDB format-16 reader: round-trip, interleave order, integrity checks."""

from __future__ import annotations

import numpy as np
import pytest

from ssllab.data.wfdb16 import read_format16, read_header, to_physical, write_format16


def _record(n_sig=3, n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-3000, 3000, size=(n_sig, n)).astype(np.int16)


def test_roundtrip_and_header(tmp_path):
    x = _record()
    write_format16(tmp_path / "r1", x, 100, ["I", "II", "V1"])
    y, hdr = read_format16(tmp_path / "r1")
    np.testing.assert_array_equal(x, y)
    assert hdr.n_sig == 3 and hdr.n_samples == 50 and hdr.fs == 100.0
    assert hdr.sig_names == ("I", "II", "V1")
    assert hdr.gains == (1000.0,) * 3 and hdr.baselines == (0,) * 3


def test_interleave_is_frame_major(tmp_path):
    # On disk: s0[t0], s1[t0], s0[t1], s1[t1], ...
    x = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.int16)
    write_format16(tmp_path / "r", x, 100, ["a", "b"])
    raw = np.fromfile(tmp_path / "r.dat", dtype="<i2")
    np.testing.assert_array_equal(raw, [1, 10, 2, 20, 3, 30])


def test_physical_units_are_millivolts(tmp_path):
    x = np.array([[1000, -500]], dtype=np.int16)
    write_format16(tmp_path / "r", x, 100, ["II"])
    d, hdr = read_format16(tmp_path / "r")
    np.testing.assert_allclose(to_physical(d, hdr), [[1.0, -0.5]])


def test_checksum_mismatch_raises(tmp_path):
    x = _record()
    write_format16(tmp_path / "r", x, 100, ["I", "II", "V1"])
    raw = np.fromfile(tmp_path / "r.dat", dtype="<i2")
    raw[7] += 1  # corrupt one sample (not the first frame)
    raw.tofile(tmp_path / "r.dat")
    with pytest.raises(ValueError, match="checksum"):
        read_format16(tmp_path / "r")
    read_format16(tmp_path / "r", verify=False)  # opting out still reads


def test_truncated_file_raises(tmp_path):
    write_format16(tmp_path / "r", _record(), 100, ["I", "II", "V1"])
    raw = np.fromfile(tmp_path / "r.dat", dtype="<i2")
    raw[:-3].tofile(tmp_path / "r.dat")
    with pytest.raises(ValueError, match="samples on disk"):
        read_format16(tmp_path / "r")


def test_header_comments_and_unsupported_format(tmp_path):
    (tmp_path / "c.hea").write_text(
        "# a comment\nc 1 500 4\nc.dat 16 1000.0(0)/mV 16 0 5 0 0 II\n# trailing\n"
    )
    assert read_header(tmp_path / "c.hea").fs == 500.0
    (tmp_path / "f.hea").write_text("f 1 500 4\nf.dat 212 200/mV 12 0 0 0 0 II\n")
    with pytest.raises(ValueError, match="format"):
        read_header(tmp_path / "f.hea")


def test_checksum_accepts_physionet_unsigned_and_signed_forms(tmp_path):
    # Regression: PhysioNet headers store the checksum unsigned (PTB-XL record 1,
    # lead III: samples sum to -778, header says 64758). Both forms must verify.
    x = np.array([[64, -500, -342]], dtype=np.int16)  # sum = -778
    for stored in (64758, -778):
        x.T.astype("<i2").tofile(tmp_path / "r.dat")
        (tmp_path / "r.hea").write_text(f"r 1 100 3\nr.dat 16 1000.0(0)/mV 16 0 64 {stored} 0 III\n")
        y, _ = read_format16(tmp_path / "r")
        np.testing.assert_array_equal(x, y)
