"""Shared plumbing for the ECG rhythm study scripts (02-06).

One place for: the frozen study config, the seeds a repeat implies, where saved
subsets and run outputs live, which recordings belong to a split's rhythm
cohort, and device selection. Scripts import this module from their own folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.ptbxl import SPLIT_NAMES, PTBXLCache  # noqa: E402

CONFIGS = Path(__file__).resolve().parent / "configs"
STUDY_PATH = CONFIGS / "study_v1_amend2.json"   # the protocol in force (base + amendment 2, final grid)

# An amendment may change only these fields: none of them affects the saved subsets,
# pretraining, or how a single run is trained once its learning rate is given.
AMENDABLE = {"downstream.lr_grid"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")


def _sha16(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(path: str | Path | None = None) -> dict:
    """Load a protocol: a base config, or an amendment applied to its base.

    The returned dict carries ``_base_sha256`` (the frozen pre-pilot protocol),
    ``_sha256`` (the effective protocol, base plus amendment) and ``_amendment``.
    The base file is verified against the hash the amendment names, and an
    amendment that touches a field outside :data:`AMENDABLE` is refused.
    """
    p = Path(path) if path else STUDY_PATH
    doc = json.loads(p.read_text())
    if "amendment" not in doc:
        doc["_base_sha256"] = doc["_sha256"] = _sha16(p)
        doc["_amendment"] = None
        return doc
    base_p = p.parent / doc["base"]
    base_sha = _sha16(base_p)
    if base_sha != doc["base_sha256_16"]:
        raise SystemExit(f"{base_p} has sha {base_sha}, but {p.name} amends {doc['base_sha256_16']}")
    sup = doc.get("supersedes")
    if sup:
        got = _sha16(base_p, p.parent / sup["file"])
        if got != sup["sha256_16"]:
            raise SystemExit(f"{p.name} supersedes {sup['file']} at {sup['sha256_16']}, but that file now hashes to {got}")
    study = json.loads(base_p.read_text())
    for key, val in doc["changes"].items():
        if key not in AMENDABLE:
            raise SystemExit(f"{p.name}: field {key!r} is not amendable (allowed: {sorted(AMENDABLE)})")
        sec, field = key.split(".")
        study[sec][field] = val
    study["_base_sha256"], study["_sha256"], study["_amendment"] = base_sha, _sha16(base_p, p), doc
    return study


# What each kind of artifact depends on. Two protocols with the same relevant hash
# produce interchangeable artifacts of that kind.
RELEVANT = {
    "subsets": ("data", "labels", "budgets", "budget_denominator", "repeats"),
    "pretrain": ("data", "encoder", "pretrain", "repeats"),
    "run": ("data", "labels", "budgets", "budget_denominator", "repeats", "encoder", "pretrain", "downstream"),
}
# Fields excluded from the run hash: the grid and how a rate is chosen do not change how one
# run trains once its learning rate is given explicitly.
RUN_EXCLUDED = {"downstream": ("lr_grid", "lr_selection")}


def relevant_sha(study: dict, kind: str) -> str:
    sel = {}
    for sec in RELEVANT[kind]:
        v = study.get(sec)
        if kind == "run" and isinstance(v, dict) and sec in RUN_EXCLUDED:
            v = {k: x for k, x in v.items() if k not in RUN_EXCLUDED[sec]}
        sel[sec] = v
    return hashlib.sha256(json.dumps(sel, sort_keys=True).encode()).hexdigest()[:16]


def provenance(study: dict) -> dict:
    """Fields every artifact records, so later scripts can check compatibility."""
    am = study["_amendment"]
    return {"study": study["study"], "study_sha256": study["_sha256"], "study_base_sha256": study["_base_sha256"],
            "amendment": am["amendment"] if am else None,
            "relevant_sha": {k: relevant_sha(study, k) for k in RELEVANT}}


def base_sha_of(record: dict) -> str:
    """The base protocol an artifact was made under (older artifacts carry only ``study_sha256``)."""
    return record.get("study_base_sha256", record["study_sha256"])


def _base_protocol(study: dict) -> dict:
    am = study["_amendment"]
    return load_study(CONFIGS / am["base"]) if am else study


def check_compatible(record: dict, study: dict, kind: str, what: str) -> str:
    """Accept an artifact only if nothing it depends on differs; return the reason, else stop.

    New artifacts carry their relevant hashes and are compared directly. Older ones
    (made before relevant hashes existed) are accepted only when made under the
    current base protocol and the amendment leaves their relevant settings identical.
    """
    want = relevant_sha(study, kind)
    rec = record.get("relevant_sha", {})
    if kind in rec:
        if rec[kind] != want:
            raise SystemExit(f"INCOMPATIBLE {what}: its {kind}-relevant settings ({rec[kind]}) differ from the protocol's ({want})")
        return f"same {kind}-relevant settings ({want})"
    if base_sha_of(record) != study["_base_sha256"]:
        raise SystemExit(f"INCOMPATIBLE {what}: made under base protocol {base_sha_of(record)}, current base is {study['_base_sha256']}")
    base_rel = relevant_sha(_base_protocol(study), kind)
    if base_rel != want:
        raise SystemExit(f"INCOMPATIBLE {what}: the amendment changes {kind}-relevant settings ({base_rel} -> {want})")
    am = study["_amendment"]
    changed = sorted(am["changes"]) if am else []
    return (f"made under base protocol {study['_base_sha256']}; amendment changes only {changed}, "
            f"which leave the {kind}-relevant settings unchanged ({want})")


def require_base(record: dict, study: dict, what: str, kind: str = "run") -> str:
    return check_compatible(record, study, kind, what)


def verify_pinned(study: dict, root: Path, key: str, field: str = "sha256", path_field: str = "path") -> None:
    """If the amendment pins an artifact's hash, the file must match it byte for byte."""
    am = study["_amendment"]
    rec = (am or {}).get("reuse", {}).get(key)
    if not rec or field not in rec:
        return
    f = root / rec[path_field]
    got = sha256_file(f)
    if got != rec[field]:
        raise SystemExit(f"{f}: sha256 {got[:16]}... differs from the pinned {rec[field][:16]}... ({key})")


def load_subsets(study: dict, output_root, experiment) -> dict:
    """The saved subsets, verified: same base protocol, and byte-identical if pinned."""
    root = study_root(output_root, experiment)
    path = subsets_path(output_root, experiment)
    subsets = json.loads(path.read_text())
    subsets["_reuse_reason"] = check_compatible(subsets, study, "subsets", str(path))
    verify_pinned(study, root, "subsets")
    return subsets


def pretrain_complete(study: dict, output_root, experiment, r: int) -> tuple[bool, str]:
    """Is repeat ``r``'s pretraining finished and compatible? Returns ``(ok, reason)``.

    Pretraining is not resumable: the checkpoint holds the student encoder and
    predictor only (no EMA teacher, optimiser state, step or RNG state). A run
    that did not finish is restarted from scratch.
    """
    root = study_root(output_root, experiment)
    ck, rp = checkpoint_path(output_root, experiment, r), root / "reports" / f"pretrain_r{r}.json"
    if not ck.exists() or not rp.exists():
        return False, "no completed checkpoint + report"
    rep = json.loads(rp.read_text())
    if rep.get("smoke"):
        return False, "smoke run"
    if rep["epochs"] != study["pretrain"]["epochs"]:
        return False, f"epochs {rep['epochs']} != {study['pretrain']['epochs']}"
    try:
        reason = check_compatible(rep, study, "pretrain", str(rp))
    except SystemExit as e:
        return False, str(e)
    if r == 0:
        verify_pinned(study, root, "pretrain_r0", "checkpoint_sha256", "checkpoint")
        verify_pinned(study, root, "pretrain_r0", "report_sha256", "report")
    return True, f"complete; {reason}"


def selection_path(output_root, experiment, tag: str) -> Path:
    return study_root(output_root, experiment) / f"lr_selection_{tag}.json"


def selection_tag(study: dict) -> str:
    am = study["_amendment"]
    return am["selection_tag"] if am else "pilot"


def reuse_tags(study: dict) -> list[str]:
    """Run tags whose (compatible) runs an amendment declares reusable for its selection."""
    return list((study["_amendment"] or {}).get("reuse_runs_from_tags", []))


def new_rates(study: dict) -> list[float]:
    """Rates an amendment adds (only these need new runs); the whole grid if undeclared."""
    am = study["_amendment"] or {}
    return list(am.get("new_rates", study["downstream"]["lr_grid"]))


def budget_key(frac: float) -> str:
    return f"{frac:g}"


def seeds(r: int) -> dict[str, int]:
    """What changes between repeats (D17)."""
    return {"budget_seed": r, "train_seed": 100 + r, "pretrain_seed": r}


def study_root(output_root: str | Path, experiment: str) -> Path:
    return Path(output_root) / experiment


def subsets_path(output_root: str | Path, experiment: str) -> Path:
    return study_root(output_root, experiment) / "subsets" / "subsets.json"


def checkpoint_path(output_root: str | Path, experiment: str, r: int) -> Path:
    return study_root(output_root, experiment) / "checkpoints" / f"jepa_r{r}.pt"


def run_dir(output_root: str | Path, experiment: str, tag: str, arm: str, frac: float, r: int, lr: float) -> Path:
    return study_root(output_root, experiment) / "runs" / tag / f"{arm}_b{budget_key(frac)}_r{r}_lr{lr:g}"


def split_code(name: str) -> int:
    return {v: k for k, v in SPLIT_NAMES.items()}[name]


def cohort_rows(cache: PTBXLCache, split: str, eligible: np.ndarray, patients=None) -> np.ndarray:
    """Rows of ``split`` that are ``eligible`` (and, if given, belong to ``patients``)."""
    m = (cache.split == split_code(split)) & eligible
    if patients is not None:
        m &= np.isin(cache.patient_id, np.asarray(patients))
    return np.flatnonzero(m)


def resolve_device(name: str):
    from ssllab.utils.reproducibility import get_device

    return get_device(name)


def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
