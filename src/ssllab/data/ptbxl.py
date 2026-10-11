"""PTB-XL data adapter (12-lead clinical ECG, rhythm labels).

The ECG sibling of :mod:`ssllab.data.perturbseq`. It turns a 10 s, 12-lead record
into the modality-agnostic token tensor the JEPA stack consumes,

    a record ``(C, L)``  ->  time-patch tokens ``(N, C * P)``      (see :mod:`ssllab.data.signal`)

PTB-XL ships as WFDB files plus two CSV tables, so building the cache needs only
the standard library and numpy (:mod:`ssllab.data.wfdb16` reads the signals and
verifies their checksums). The cache layout::

    <data_dir>/<artifact>/
      signals.npy       int16 (n_records, n_leads, n_samples), microvolts; memory-mapped
      records.json      one dict per row: ids, fold, demographics, quality flags, scp_codes
      manifest.json     provenance, leads, sampling rate, rhythm vocabulary, label rules

Label semantics, fixed here so that every script agrees:

* A statement is **present** when its code appears in the record's ``scp_codes``,
  whatever its likelihood. PTB-XL stores likelihood 0 for many rhythm and form
  statements, where it means "not graded", not "absent".
* A record with no rhythm statement is **unknown** for every rhythm task. It is
  excluded, never treated as negative.
* AF pilot, ``primary`` rule: exclude unknown records and any record with AFLT
  (AFIB+AFLT co-labels included); among the rest, AFIB decides positivity.
  ``sensitivity`` rule: keep flutter-only records as negatives; any record with
  AFIB stays positive.
* "Non-AF" is a clinical population (other arrhythmias, paced rhythms, sinus
  records with MI or conduction disease). It is not a healthy-control group.

Splits follow PTB-XL's ``strat_fold``, which confines each patient to one fold:
folds 1-8 train, 9 validation, 10 test.

Label budgets are patient-level and nested. One seeded permutation of *all*
training patients is drawn, and a budget of ``k`` percent takes its first
``ceil(k/100 * n_patients)`` patients. Every record of a chosen patient is in;
no other record is. Because the permutation does not depend on the task, the
same patients carry labels across tasks and methods for a given seed.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ssllab.data.signal import SignalGeometry, patchify_1d, zscore_per_lead
from ssllab.data.wfdb16 import read_format16, write_format16

ARTIFACT_VERSION = "1"
PTBXL_VERSION = "1.0.3"

SIGNALS_NPY = "signals.npy"
RECORDS_JSON = "records.json"
MANIFEST_JSON = "manifest.json"

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLDS = (9,)
TEST_FOLDS = (10,)

AF_RULES = ("primary", "sensitivity")
TASKS = ("af_primary", "af_sensitivity", "rhythm", "none")
QUALITY_FIELDS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker")


# --------------------------------------------------------------------------- #
# Label parsing and cohorts.
# --------------------------------------------------------------------------- #
def parse_scp_codes(text: str) -> dict[str, float]:
    """PTB-XL ``scp_codes`` column (a Python dict literal) -> ``{code: likelihood}``."""
    d = ast.literal_eval(text) if text else {}
    if not isinstance(d, dict):
        raise ValueError(f"scp_codes is not a dict literal: {text!r}")
    return {str(k): float(v) for k, v in d.items()}


@dataclass(frozen=True)
class RhythmLabels:
    codes: tuple[str, ...]
    y: np.ndarray            # (n, K) uint8; 1 if the code is present, any likelihood
    likelihood: np.ndarray   # (n, K) float32; NaN where the code is absent
    has_rhythm: np.ndarray   # (n,) bool; False -> unknown for every rhythm task


def rhythm_labels(records: list[dict[str, Any]], codes: Iterable[str]) -> RhythmLabels:
    codes = tuple(codes)
    n, k = len(records), len(codes)
    y = np.zeros((n, k), dtype=np.uint8)
    lik = np.full((n, k), np.nan, dtype=np.float32)
    col = {c: j for j, c in enumerate(codes)}
    for i, r in enumerate(records):
        for code, val in r["scp_codes"].items():
            j = col.get(code)
            if j is not None:
                y[i, j] = 1
                lik[i, j] = val
    return RhythmLabels(codes=codes, y=y, likelihood=lik, has_rhythm=y.any(axis=1))


def af_pilot_cohort(labels: RhythmLabels, rule: str = "primary") -> tuple[np.ndarray, np.ndarray]:
    """``(include (n,) bool, y (n,) int8)`` for the binary AF pilot under ``rule``."""
    if rule not in AF_RULES:
        raise ValueError(f"rule must be one of {AF_RULES}, got {rule!r}")
    afib = labels.y[:, labels.codes.index("AFIB")].astype(bool)
    aflt = labels.y[:, labels.codes.index("AFLT")].astype(bool)
    include = labels.has_rhythm.copy()
    if rule == "primary":
        include &= ~aflt
    return include, afib.astype(np.int8)


def split_codes(strat_fold: np.ndarray) -> np.ndarray:
    out = np.full(strat_fold.shape, -1, dtype=np.int8)
    out[np.isin(strat_fold, TRAIN_FOLDS)] = SPLIT_TRAIN
    out[np.isin(strat_fold, VAL_FOLDS)] = SPLIT_VAL
    out[np.isin(strat_fold, TEST_FOLDS)] = SPLIT_TEST
    if (out < 0).any():
        raise ValueError(f"unexpected strat_fold values: {sorted(set(strat_fold[out < 0].tolist()))}")
    return out


# --------------------------------------------------------------------------- #
# Patient-level label budgets and support.
# --------------------------------------------------------------------------- #
def budget_patients(train_patients: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Patients that carry labels at budget ``frac`` (0 < frac <= 1), nested across fracs.

    ``train_patients`` is the full set of training-split patient ids (any order,
    duplicates allowed). The result is sorted.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    pats = np.unique(np.asarray(train_patients))
    perm = np.random.default_rng(seed).permutation(pats)
    k = min(len(pats), math.ceil(frac * len(pats) - 1e-9))
    return np.sort(perm[:k])


def patient_support(y: np.ndarray, patient_ids: np.ndarray) -> tuple[int, int]:
    """``(positive patients, negative patients)`` for binary ``y`` over records.

    A patient is positive when any of their records is positive, negative when
    none is. (A patient can have records with different rhythms.)
    """
    y = np.asarray(y).astype(bool)
    pats = np.asarray(patient_ids)
    pos = np.unique(pats[y])
    n_all = len(np.unique(pats))
    return int(len(pos)), int(n_all - len(pos))


# --------------------------------------------------------------------------- #
# Cache I/O.
# --------------------------------------------------------------------------- #
@dataclass
class PTBXLCache:
    signals: np.ndarray           # int16 (n, C, L), microvolts (memory-mapped)
    records: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def leads(self) -> tuple[str, ...]:
        return tuple(self.manifest["leads"])

    @property
    def fs(self) -> float:
        return float(self.manifest["fs"])

    @property
    def rhythm_codes(self) -> tuple[str, ...]:
        return tuple(self.manifest["rhythm_codes"])

    @property
    def patient_id(self) -> np.ndarray:
        return np.array([r["patient_id"] for r in self.records], dtype=np.int64)

    @property
    def strat_fold(self) -> np.ndarray:
        return np.array([r["strat_fold"] for r in self.records], dtype=np.int64)

    @property
    def split(self) -> np.ndarray:
        return split_codes(self.strat_fold)

    def rhythm(self) -> RhythmLabels:
        return rhythm_labels(self.records, self.rhythm_codes)


def load_cache(cache_dir: str | Path, mmap: bool = True) -> PTBXLCache:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / MANIFEST_JSON).read_text())
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"{cache_dir}: artifact_version {manifest.get('artifact_version')!r} != {ARTIFACT_VERSION!r}; rebuild the cache"
        )
    signals = np.load(cache_dir / SIGNALS_NPY, mmap_mode="r" if mmap else None)
    records = json.loads((cache_dir / RECORDS_JSON).read_text())
    if len(records) != signals.shape[0]:
        raise ValueError(f"{cache_dir}: {len(records)} records but {signals.shape[0]} signals")
    return PTBXLCache(signals=signals, records=records, manifest=manifest)


def _num(x: str) -> float | None:
    x = (x or "").strip()
    return float(x) if x else None


def _rhythm_vocab(scp_csv: Path) -> dict[str, str]:
    """Rhythm statements from ``scp_statements.csv``: ``{code: description}``, file order."""
    with scp_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    key = next(iter(rows[0]))  # the unnamed first column holds the code
    return {r[key]: r["description"] for r in rows if (_num(r.get("rhythm", "")) or 0.0) == 1.0}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache_from_raw(
    raw_dir: str | Path,
    cache_dir: str | Path,
    fs: int = 100,
    limit: int | None = None,
    verify: bool = True,
    source_zip: str | Path | None = None,
    log_every: int = 2000,
    logger=None,
) -> dict[str, Any]:
    """Read a PhysioNet PTB-XL tree and write the cache. Returns the manifest.

    ``fs`` picks the official copy: 100 (``filename_lr``) or 500 (``filename_hr``).
    Every record's per-lead checksum is verified unless ``verify=False``.
    """
    if fs not in (100, 500):
        raise ValueError(f"fs must be 100 or 500 (the two official copies), got {fs}")
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    with (raw_dir / "ptbxl_database.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(float(r["ecg_id"])))
    if limit is not None:
        rows = rows[:limit]
    vocab = _rhythm_vocab(raw_dir / "scp_statements.csv")
    fname_col = "filename_lr" if fs == 100 else "filename_hr"

    first, hdr0 = read_format16(raw_dir / rows[0][fname_col], verify=verify)
    n_leads, n_samples = first.shape
    cache_dir.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(
        cache_dir / SIGNALS_NPY, mode="w+", dtype=np.int16, shape=(len(rows), n_leads, n_samples)
    )
    records: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        x, hdr = (first, hdr0) if i == 0 else read_format16(raw_dir / r[fname_col], verify=verify)
        if hdr.sig_names != hdr0.sig_names or x.shape != (n_leads, n_samples) or hdr.fs != hdr0.fs:
            raise ValueError(f"{r[fname_col]}: geometry/lead order differs from the first record")
        if set(hdr.gains) != {1000.0} or set(hdr.baselines) != {0}:
            raise ValueError(f"{r[fname_col]}: expected gain 1000/mV and baseline 0, got {hdr.gains} {hdr.baselines}")
        signals[i] = x
        records.append({
            "ecg_id": int(float(r["ecg_id"])),
            "patient_id": int(float(r["patient_id"])),
            "strat_fold": int(float(r["strat_fold"])),
            "age": _num(r.get("age", "")),
            "sex": _num(r.get("sex", "")),
            "site": _num(r.get("site", "")),
            "device": (r.get("device") or "").strip() or None,
            "validated_by_human": (r.get("validated_by_human", "").strip().lower() == "true"),
            "scp_codes": parse_scp_codes(r["scp_codes"]),
            **{q: ((r.get(q) or "").strip() or None) for q in QUALITY_FIELDS},
            "filename": r[fname_col],
        })
        if logger is not None and log_every and (i + 1) % log_every == 0:
            logger.info("read %d / %d records", i + 1, len(rows))
    signals.flush()
    del signals

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source": {
            "dataset": "PTB-XL",
            "version": PTBXL_VERSION,
            "url": f"https://physionet.org/content/ptb-xl/{PTBXL_VERSION}/",
            "license": "CC BY 4.0",
            "raw_dir": str(raw_dir),
            "zip_sha256": _sha256(Path(source_zip)) if source_zip and Path(source_zip).exists() else None,
        },
        "fs": float(hdr0.fs),
        "n_samples": int(n_samples),
        "duration_s": n_samples / float(hdr0.fs),
        "leads": list(hdr0.sig_names),
        "units": "int16 microvolts (WFDB digital, gain 1000/mV, baseline 0)",
        "n_records": len(records),
        "n_patients": len({r["patient_id"] for r in records}),
        "checksums_verified": bool(verify),
        "rhythm_codes": list(vocab),
        "rhythm_descriptions": vocab,
        "splits": {"train_folds": list(TRAIN_FOLDS), "val_folds": list(VAL_FOLDS), "test_folds": list(TEST_FOLDS)},
        "label_rules": {
            "present": "code in scp_codes, any likelihood (0 included)",
            "no_rhythm_statement": "unknown; excluded from rhythm tasks",
            "af_primary": "exclude unknown and any AFLT (incl. AFIB+AFLT); y = AFIB",
            "af_sensitivity": "exclude unknown; flutter-only negative; y = AFIB",
        },
        "limit": limit,
    }
    (cache_dir / RECORDS_JSON).write_text(json.dumps(records))
    (cache_dir / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# A synthetic PhysioNet-shaped tree, for --smoke runs and tests.
# --------------------------------------------------------------------------- #
TOY_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
TOY_RHYTHM = (
    ("SR", "sinus rhythm"), ("AFIB", "atrial fibrillation"), ("STACH", "sinus tachycardia"),
    ("SARRH", "sinus arrhythmia"), ("SBRAD", "sinus bradycardia"), ("PACE", "normal functioning artificial pacemaker"),
    ("SVARR", "supraventricular arrhythmia"), ("BIGU", "bigeminal pattern"), ("AFLT", "atrial flutter"),
    ("SVTAC", "supraventricular tachycardia"), ("PSVT", "paroxysmal supraventricular tachycardia"),
    ("TRIGU", "trigeminal pattern"),
)


def _toy_beats(rng: np.random.Generator, n_samples: int, fs: float, irregular: bool, hr: float) -> np.ndarray:
    """One synthetic lead: P, QRS and T bumps at beat times, in microvolts."""
    t = np.arange(n_samples) / fs
    sig = np.zeros(n_samples)
    rr_mean = 60.0 / hr
    beat = rng.uniform(0, rr_mean)
    while beat < t[-1]:
        for dt, amp, width in ((-0.16, 120.0, 0.025), (0.0, 1000.0, 0.012), (0.25, 250.0, 0.04)):
            if irregular and dt < 0:
                continue  # no organised P wave in AF
            sig += amp * np.exp(-0.5 * ((t - beat - dt) / width) ** 2)
        beat += rr_mean * (rng.uniform(0.6, 1.4) if irregular else rng.normal(1.0, 0.02))
    return sig + rng.normal(0, 15.0, n_samples)


def write_toy_raw_tree(
    root: str | Path,
    specs: list[dict[str, Any]] | None = None,
    n_records: int = 60,
    fs: int = 100,
    duration_s: float = 10.0,
    seed: int = 0,
) -> Path:
    """Write a tiny PTB-XL-shaped tree (CSV tables + WFDB records) under ``root``.

    ``specs`` lists records as ``{"patient_id", "strat_fold", "scp_codes"}``. When
    omitted, ``n_records`` records are generated, with 1-3 records per patient and
    each patient confined to one fold.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    if specs is None:
        specs, pid = [], 1
        choices = [{"SR": 0.0}, {"SR": 0.0, "NORM": 100.0}, {"AFIB": 0.0}, {"STACH": 0.0},
                   {"SBRAD": 0.0}, {"AFLT": 0.0}, {"AFIB": 0.0, "AFLT": 0.0}, {"NORM": 100.0}]
        while len(specs) < n_records:
            fold = int(rng.integers(1, 11))
            for _ in range(int(rng.integers(1, 4))):
                specs.append({"patient_id": pid, "strat_fold": fold,
                              "scp_codes": dict(choices[int(rng.integers(len(choices)))])})
            pid += 1
        specs = specs[:n_records]
    n_samples = int(round(fs * duration_s))
    sub = "records100" if fs == 100 else "records500"
    suffix = "lr" if fs == 100 else "hr"
    db_rows = []
    for i, s in enumerate(specs, start=1):
        codes = s["scp_codes"]
        irregular = "AFIB" in codes or "AFLT" in codes
        hr = 120.0 if "STACH" in codes else 50.0 if "SBRAD" in codes else 72.0
        base = _toy_beats(rng, n_samples, fs, irregular, hr)
        gains = rng.uniform(0.3, 1.2, size=len(TOY_LEADS)) * rng.choice([-1, 1], size=len(TOY_LEADS))
        digital = np.clip(np.round(gains[:, None] * base[None, :]), -32768, 32767).astype(np.int16)
        rel = f"{sub}/{(i // 1000) * 1000:05d}/{i:05d}_{suffix}"
        write_format16(root / rel, digital, fs, TOY_LEADS)
        db_rows.append({
            "ecg_id": i, "patient_id": float(s["patient_id"]), "age": float(rng.integers(20, 90)),
            "sex": float(rng.integers(0, 2)), "site": 0.0, "device": "toy",
            "scp_codes": repr(codes), "validated_by_human": "True",
            **{q: "" for q in QUALITY_FIELDS},
            "strat_fold": s["strat_fold"], "filename_lr": rel if fs == 100 else "",
            "filename_hr": rel if fs == 500 else "",
        })
    root.mkdir(parents=True, exist_ok=True)
    with (root / "ptbxl_database.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(db_rows[0]))
        w.writeheader()
        w.writerows(db_rows)
    with (root / "scp_statements.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["", "description", "diagnostic", "form", "rhythm"])
        w.writerow(["NORM", "normal ECG", "1.0", "", ""])
        for code, desc in TOY_RHYTHM:
            w.writerow([code, desc, "", "", "1.0"])
    return root


# --------------------------------------------------------------------------- #
# Dataset + loaders.
# --------------------------------------------------------------------------- #
class PTBXLDataset(Dataset):
    """Rows of one split; each item is a per-lead z-scored ``(C, L)`` float signal."""

    def __init__(self, cache: PTBXLCache, rows: np.ndarray, lead_idx: list[int], labels: np.ndarray | None) -> None:
        self.c = cache
        self.rows = np.asarray(rows, dtype=np.int64)
        self.lead_idx = lead_idx
        self.labels = labels
        self._pid = cache.patient_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.rows[i])
        x = torch.from_numpy(np.asarray(self.c.signals[r, self.lead_idx], dtype=np.float32))
        item = {
            "signal": zscore_per_lead(x),
            "row": torch.tensor(r),
            "patient_id": torch.tensor(int(self._pid[r])),
        }
        if self.labels is not None:
            item["labels"] = torch.from_numpy(np.atleast_1d(self.labels[r]).astype(np.float32))
        return item


def _make_collate(patch_len: int):
    def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        out["tokens"] = patchify_1d(out["signal"], patch_len)  # (B, N, C * P)
        return out

    return collate


def resolve_leads(cache_leads: tuple[str, ...], leads: str | Iterable[str]) -> list[int]:
    """``"all"``, a single lead name such as ``"II"``, or a list of names -> indices."""
    if leads == "all":
        return list(range(len(cache_leads)))
    names = [leads] if isinstance(leads, str) else list(leads)
    upper = [c.upper() for c in cache_leads]
    missing = [n for n in names if n.upper() not in upper]
    if missing:
        raise ValueError(f"unknown lead(s) {missing}; cache has {list(cache_leads)}")
    return [upper.index(n.upper()) for n in names]


def task_labels(cache: PTBXLCache, task: str) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """``(include (n,) bool, labels or None, label_names)`` for ``task``."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if task == "none":
        return np.ones(cache.n, dtype=bool), None, []
    rl = cache.rhythm()
    if task == "rhythm":
        return rl.has_rhythm, rl.y, list(rl.codes)
    include, y = af_pilot_cohort(rl, rule=task.split("_", 1)[1])
    return include, y, ["AFIB"]


def get_ptbxl_dataloaders(
    batch_size: int = 64,
    data_dir: str | Path = "data",
    artifact: str = "ptbxl100",
    task: str = "af_primary",
    leads: str | Iterable[str] = "all",
    geometry: SignalGeometry | None = None,
    label_frac: float = 1.0,
    budget_seed: int = 0,
    num_workers: int = 0,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train, val, test)`` loaders of tokenized PTB-XL records.

    task:
        ``af_primary`` / ``af_sensitivity`` (binary AF pilot), ``rhythm`` (one
        sigmoid per rhythm statement), or ``none`` (no labels, every record).
    leads:
        ``"all"``, one lead name (e.g. ``"II"``), or a list of names.
    geometry:
        Physical-time token geometry. Its ``fs`` and ``duration_s`` must match the
        cache; a 100 Hz cache never feeds a 500 Hz model. Defaults to the cache
        rate, 10 s windows and 250 ms patches.
    label_frac, budget_seed:
        Patient-level labelled budget of the *train* split (nested across
        ``label_frac`` for a fixed seed). Val and test are never subsampled.
    limit:
        Cap the train split after budgeting (smoke runs only).

    Batches are dicts with ``tokens (B, N, C*P)``, ``signal (B, C, L)``,
    ``row``, ``patient_id`` and, unless ``task="none"``, ``labels (B, K)``. Each
    loader carries ``.meta``.
    """
    cache = load_cache(Path(data_dir) / artifact)
    geom = geometry or SignalGeometry(fs=cache.fs, duration_s=cache.manifest["duration_s"])
    if abs(geom.fs - cache.fs) > 1e-9 or geom.n_samples != cache.signals.shape[-1]:
        raise ValueError(
            f"geometry (fs={geom.fs:g} Hz, {geom.n_samples} samples) does not match the cache "
            f"(fs={cache.fs:g} Hz, {cache.signals.shape[-1]} samples); build a cache at that rate"
        )
    lead_idx = resolve_leads(cache.leads, leads)
    include, labels, names = task_labels(cache, task)
    split, pid = cache.split, cache.patient_id

    train_pool = np.flatnonzero(split == SPLIT_TRAIN)
    chosen = budget_patients(pid[train_pool], label_frac, budget_seed)
    train_rows = train_pool[np.isin(pid[train_pool], chosen) & include[train_pool]]
    if limit is not None:
        train_rows = train_rows[:limit]
    val_rows = np.flatnonzero((split == SPLIT_VAL) & include)
    test_rows = np.flatnonzero((split == SPLIT_TEST) & include)

    meta = {
        **geom.describe(len(lead_idx)),
        "leads": [cache.leads[i] for i in lead_idx],
        "task": task,
        "label_names": names,
        "label_frac": label_frac,
        "budget_seed": budget_seed,
        "n_budget_patients": int(len(chosen)),
        "n_train_records": int(len(train_rows)),
        "n_val_records": int(len(val_rows)),
        "n_test_records": int(len(test_rows)),
        "artifact": artifact,
    }
    collate = _make_collate(geom.patch_len)

    def loader(rows: np.ndarray, shuffle: bool, drop_last: bool) -> DataLoader:
        dl = DataLoader(
            PTBXLDataset(cache, rows, lead_idx, labels),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            drop_last=drop_last, collate_fn=collate,
        )
        dl.meta = meta  # type: ignore[attr-defined]
        return dl

    return (
        loader(train_rows, shuffle=True, drop_last=len(train_rows) > batch_size),
        loader(val_rows, shuffle=False, drop_last=False),
        loader(test_rows, shuffle=False, drop_last=False),
    )
