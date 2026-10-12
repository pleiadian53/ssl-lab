"""Choose each arm's learning rate on validation, then freeze it (D19, amendment 1).

Reads the runs of one tag at repeat 0 and budget 0.10, one per rate in the
protocol's grid, and picks the rate with the highest validation core macro AUROC
per arm. Every candidate run must have been made under the current base protocol,
and every grid rate must be present. All arms share the grid and the rule, which
is what "comparable tuning resources" means here. Test data plays no part.

The selection is written per tag (``lr_selection_<tag>.json``), so the pilot's
selection stays untouched. For each arm it also records two boundary checks:
whether the chosen rate sits at an edge of the grid, and whether its best step
fell at the step cap (still improving when training stopped).

Usage
-----
    python examples/ecg_rhythm/05_select_lr.py                # the amendment's selection tag
    python examples/ecg_rhythm/05_select_lr.py --tag pilot --study examples/ecg_rhythm/configs/study_v1.json

Output
------
    output/<experiment>/lr_selection_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import logging

import _study as S

logger = logging.getLogger("05_select_lr")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", default=None, help="run group (default: the protocol's selection tag)")
    p.add_argument("--experiment", default="ecg_rhythm_v1")
    p.add_argument("--output-root", default="output")
    p.add_argument("--study", default=None)
    args = p.parse_args()

    study = S.load_study(args.study)
    tag = args.tag or S.selection_tag(study)
    grid = study["downstream"]["lr_grid"]
    root = S.study_root(args.output_root, args.experiment)
    subsets = S.load_subsets(study, args.output_root, args.experiment)   # provenance check of the saved subsets
    # Candidates: this tag's runs plus runs from tags the amendment declares reusable.
    runs = []
    for t in [tag] + S.reuse_tags(study):
        for f in sorted((root / "runs" / t).glob("*/report.json")):
            r = json.loads(f.read_text())
            r["_source"] = t
            runs.append(r)
    smoke = [f"{r['_source']}/{r['arm']} lr={r['lr']:g}" for r in runs if r["smoke"]]
    if smoke:
        raise SystemExit(f"smoke runs among the candidates {smoke}; a selection must use real runs only")
    sel = {**S.provenance(study), "subsets_reuse_reason": None, "tag": tag, "budget": 0.1, "repeat": 0, "grid": grid,
           "rule": study["downstream"]["selection_metric"], "arms": {}}
    sel["subsets_reuse_reason"] = subsets["_reuse_reason"]
    am = study["_amendment"] or {}
    sel["reuse_runs_from_tags"] = S.reuse_tags(study)
    sel["final_grid"] = bool(am.get("final_grid"))
    sel["stop_rule"] = am.get("stop_rule")
    for arm in ("scratch", "finetune", "probe"):
        cand = [r for r in runs if r["arm"] == arm and not r["shuffle_labels"] and r["repeat"] == 0
                and abs(r["budget"] - 0.1) < 1e-9]
        if not cand:
            raise SystemExit(f"tag {tag}: no runs for arm {arm}; every arm needs the full grid")
        off = sorted({r["lr"] for r in cand} - set(grid))
        if off:
            raise SystemExit(f"tag {tag}: {arm} has runs at rates outside the protocol grid {grid}: {off}")
        seen = {}
        for r in cand:
            if r["lr"] in seen:
                raise SystemExit(f"{arm} lr={r['lr']:g} appears in both {seen[r['lr']]} and {r['_source']}; refusing to choose silently")
            seen[r["lr"]] = r["_source"]
        for r in cand:
            r["_compat"] = S.check_compatible(r, study, "run", f"run {tag}/{arm} lr={r['lr']}")
        missing = sorted(set(grid) - {r["lr"] for r in cand})
        if missing:
            raise SystemExit(f"{arm}: grid incomplete, missing lr {missing}")
        score = lambda r: r["val_core_macro_auroc"] if (r["val_core_macro_auroc"] == r["val_core_macro_auroc"] and not r.get("diverged")) else float("-inf")  # noqa: E731
        best = max(cand, key=score)
        if score(best) == float("-inf"):
            raise SystemExit(f"{arm}: every rate diverged or was incomplete")
        edge = "low" if best["lr"] == min(grid) else "high" if best["lr"] == max(grid) else None
        sel["arms"][arm] = {
            "lr": best["lr"],
            "val_core_macro_auroc": {f"{r['lr']:g}": r["val_core_macro_auroc"] for r in sorted(cand, key=lambda r: r["lr"])},
            "best_step": best["best_step"], "best_at_cap": best["best_at_cap"], "grid_edge": edge,
            "runs": {f"{r['lr']:g}": {"source_tag": r["_source"], "compatibility": r["_compat"],
                                     "diverged": bool(r.get("diverged")), "reused": r.get("reused", {})} for r in cand},
        }
        logger.info("%s: lr=%g (best step %d%s%s); by lr: %s", arm, best["lr"], best["best_step"],
                    ", AT STEP CAP" if best["best_at_cap"] else "", f", at {edge} grid edge" if edge else "",
                    sel["arms"][arm]["val_core_macro_auroc"])
    path = S.selection_path(args.output_root, args.experiment, tag)
    if path.exists():
        raise SystemExit(f"{path} exists; a frozen selection is never overwritten")
    path.write_text(json.dumps(sel, indent=2))
    print(" ".join(f"{a}={v['lr']:g}" for a, v in sel["arms"].items()))


if __name__ == "__main__":
    main()
