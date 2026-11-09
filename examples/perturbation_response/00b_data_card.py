"""Norman data card: how many cells inform each target gene, per split.

Reads the cache written by ``00_process_norman.py``. Trains nothing. The card
answers "what did the model actually see?" before any model output is read:

* cells, control cells, single-gene perturbations and combinations per split
* per target gene, the cells that carry it in each split: as a single, inside
  combinations, and in total
* for one or more named genes (``--gene``), the breakdown by perturbation
* for a named perturbation (``--perturbation CEBPE+RUNX1T1``), its own record: the
  genes it targets, which split its cells live in, and how much evidence each of
  its target genes has in training
* ``--as-table`` prints one row per perturbation with the cell counts as columns,
  which is the form to paste into a note or pipe to a file

Two keys, two questions. The gene table answers "how much evidence trained the
vector for gene g", which is what the gene-compositional condition raises. The
perturbation table answers "what is perturbation p and how many cells is its
score estimated from", which is what every per-perturbation metric raises.

Why per gene. The gene-compositional condition embeds ``A+B`` as
``e(A) + e(B)`` from learned per-gene vectors, so ``e(CEBPE)`` is trained by every
training cell whose perturbation targets CEBPE: the CEBPE single and every
training combination that includes it. The ``train`` total is that count. Under
the combo split the held-out combinations sit in val/test, so their cells never
reach the gene vectors.

Only the label arrays are read from the cache (perturbation, split codes,
control flags), not the expression matrices, so the card is fast on the full
dataset.

Usage
-----
    python examples/perturbation_response/00b_data_card.py                          # all target genes
    python examples/perturbation_response/00b_data_card.py --gene CEBPE              # one gene, by perturbation
    python examples/perturbation_response/00b_data_card.py --gene CEBPE --gene KLF1 --split cells
    python examples/perturbation_response/00b_data_card.py --perturbation CEBPE+RUNX1T1
    python examples/perturbation_response/00b_data_card.py --as-table --kind combo --in-split test

Output
------
    output/<experiment>/reports/data_card.json
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TOKENS_NPZ,
    perturbation_support,
    target_gene_support,
    target_genes,
)
from ssllab.experiment import experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SPLITS = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cells per target gene and split for the processed Norman cache.")
    p.add_argument("--experiment", type=str, default="norman_stage_a", help="experiment name (output/<name>/)")
    p.add_argument("--output-root", type=str, default="output")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--artifact", type=str, default="norman2019", help="processed cache under <data-dir>/")
    p.add_argument("--split", type=str, default="combo", choices=["combo", "cells"],
                   help="which split scheme to count under (combo = the headline held-out combinations)")
    p.add_argument("--perturbation", "-p", action="append", default=[],
                   help="a perturbation label ('CEBPE' or 'CEBPE+RUNX1T1'); repeatable. "
                        "Gene order does not matter, the label is alphabetized as in the cache")
    p.add_argument("--as-table", action="store_true",
                   help="one row per perturbation, cell counts as columns")
    p.add_argument("--kind", type=str, default=None, choices=["control", "single", "combo"],
                   help="restrict --as-table to one kind of label")
    p.add_argument("--in-split", type=str, default=None, choices=["train", "val", "test"],
                   help="restrict --as-table to perturbations whose cells are in this split")
    p.add_argument("--gene", action="append", default=[],
                   help="show this target gene by perturbation (repeatable; case-insensitive)")
    return p.parse_args()


def load_labels(cache_dir: Path, split: str) -> dict[str, np.ndarray]:
    """The per-cell label arrays only; ``np.load`` on an npz reads members lazily."""
    npz = np.load(cache_dir / TOKENS_NPZ, allow_pickle=True)
    return {
        "pert_id": npz["pert_id"],
        "pert_names": npz["pert_names"],
        "is_control": npz["is_control"],
        "split_codes": npz["split_combo" if split == "combo" else "split_cells"],
    }


def overview(labels: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Cells, control cells, and distinct singles and combinations present in each split."""
    out = {}
    for split, code in SPLITS.items():
        m = labels["split_codes"] == code
        present = [labels["pert_names"][p] for p in np.unique(labels["pert_id"][m])]
        n_targets = [len(target_genes(name)) for name in present]
        out[split] = {
            "cells": int(m.sum()),
            "control_cells": int((m & labels["is_control"].astype(bool)).sum()),
            "singles": sum(k == 1 for k in n_targets),
            "combos": sum(k >= 2 for k in n_targets),
        }
    return out


def resolve_genes(requested: list[str], known: list[str]) -> list[str]:
    """Match requested gene symbols to the target-gene vocabulary, ignoring case."""
    by_upper = {g.upper(): g for g in known}
    resolved = []
    for g in requested:
        hit = by_upper.get(g.upper())
        if hit is None:
            close = difflib.get_close_matches(g.upper(), list(by_upper), n=5)
            hint = f"; close matches: {', '.join(by_upper[c] for c in close)}" if close else ""
            raise SystemExit(f"{g!r} is not a target gene in this cache{hint}")
        resolved.append(hit)
    return resolved


def print_overview(ov: dict[str, dict[str, int]], split: str) -> None:
    print(f"\nSplit scheme: {split}")
    print(f"{'':>6}  {'cells':>8}  {'controls':>8}  {'singles':>7}  {'combos':>6}")
    for s, r in ov.items():
        print(f"{s:>6}  {r['cells']:>8,}  {r['control_cells']:>8,}  {r['singles']:>7}  {r['combos']:>6}")


def print_gene_table(support: dict) -> None:
    """One row per target gene, most training cells first."""
    rows = sorted(support.items(), key=lambda kv: (-kv[1]["train"]["total"], kv[0]))
    print(f"\nCells per target gene ({len(rows)} genes), sorted by training cells")
    print(f"{'gene':<10}  {'train single':>12}  {'train combos':>12}  {'(n combos)':>10}  "
          f"{'train total':>11}  {'val':>6}  {'test':>6}")
    for g, rec in rows:
        tr = rec["train"]
        print(f"{g:<10}  {tr['single']:>12,}  {tr['in_combos']:>12,}  {len(tr['combos']):>10}  "
              f"{tr['total']:>11,}  {rec['val']['total']:>6,}  {rec['test']['total']:>6,}")


def print_gene_detail(gene: str, rec: dict) -> None:
    """One gene, split by split, with each perturbation that carries it."""
    print(f"\n{gene}")
    for split in SPLITS:
        r = rec[split]
        if r["total"] == 0:
            print(f"  {split:<5}  0 cells")
            continue
        parts = []
        if r["single"]:
            parts.append(f"{r['single']:,} as a single")
        if r["combos"]:
            k = len(r["combos"])
            parts.append(f"{r['in_combos']:,} in {k} combination{'s' if k > 1 else ''}")
        print(f"  {split:<5}  {r['total']:,} cells = {' + '.join(parts)}")
        if r["single"]:
            print(f"           {gene:<22} {r['single']:>6,}")
        for name, n in sorted(r["combos"].items(), key=lambda kv: -kv[1]):
            print(f"           {name:<22} {n:>6,}")


def resolve_perts(requested: list[str], known: list[str]) -> list[str]:
    """Match requested perturbation labels to the cache's vocabulary.

    Case-insensitive, and gene order is irrelevant: ``"runx1t1+cebpe"`` resolves to
    ``"CEBPE+RUNX1T1"``, because the cache alphabetizes a combination's genes
    (``normalize_label`` in ``00_process_norman.py``) and so must any lookup.
    """
    def key(name: str) -> str:
        genes = target_genes(name)
        return "+".join(sorted(g.upper() for g in genes)) if genes else name.upper()

    by_key = {key(n): n for n in known}
    resolved = []
    for q in requested:
        hit = by_key.get(key(q))
        if hit is None:
            close = difflib.get_close_matches(key(q), list(by_key), n=5)
            hint = f"; close matches: {', '.join(by_key[c] for c in close)}" if close else ""
            raise SystemExit(f"{q!r} is not a perturbation in this cache{hint}")
        resolved.append(hit)
    return resolved


def print_pert_detail(name: str, rec: dict, support: dict) -> None:
    """One perturbation: its own cells, then the training evidence for each target gene."""
    print(f"\n{name}   ({rec['kind']}, {rec['n_genes']} target gene{'s' if rec['n_genes'] != 1 else ''})")
    print(f"  cells   train {rec['train']:,}   val {rec['val']:,}   test {rec['test']:,}"
          f"   total {rec['total']:,}   [{rec['split']}]")
    if rec["split"] in ("val", "test"):
        print(f"  held out under this split: these {rec['total']:,} cells are the only cells this "
              f"perturbation has, and every score reported for it is estimated from them")
    if not rec["genes"]:
        return
    print(f"\n  training evidence per target gene")
    print(f"  {'gene':<10}  {'single':>8}  {'in combos':>10}  {'(n)':>4}  {'train total':>11}")
    for g in rec["genes"]:
        tr = support[g]["train"]
        print(f"  {g:<10}  {tr['single']:>8,}  {tr['in_combos']:>10,}  {len(tr['combos']):>4}  "
              f"{tr['total']:>11,}")


def print_pert_table(perts: dict, kind: str | None, in_split: str | None) -> None:
    """One row per perturbation, counts as columns, most cells first."""
    rows = [(n, r) for n, r in perts.items()
            if (kind is None or r["kind"] == kind) and (in_split is None or r["split"] == in_split)]
    rows.sort(key=lambda kv: (-kv[1]["total"], kv[0]))
    what = " ".join(x for x in [kind, f"in {in_split}" if in_split else None] if x) or "all"
    print(f"\nPerturbations ({what}): {len(rows)} rows, sorted by total cells")
    print(f"{'perturbation':<24}  {'kind':<7}  {'train':>8}  {'val':>6}  {'test':>6}  "
          f"{'total':>8}  {'split':<6}")
    for n, r in rows:
        print(f"{n:<24}  {r['kind']:<7}  {r['train']:>8,}  {r['val']:>6,}  {r['test']:>6,}  "
              f"{r['total']:>8,}  {r['split']:<6}")
    if rows:
        tot = sum(r["total"] for _, r in rows)
        cells = [r["total"] for _, r in rows]
        print(f"{'':<24}  {'':<7}  {'':>8}  {'':>6}  {'':>6}  {tot:>8,}  (min {min(cells):,} "
              f"median {int(np.median(cells)):,} max {max(cells):,})")


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.data_dir) / args.artifact
    labels = load_labels(cache_dir, args.split)
    support = target_gene_support(labels["pert_id"], labels["pert_names"], labels["split_codes"])
    perts = perturbation_support(labels["pert_id"], labels["pert_names"], labels["split_codes"])
    ov = overview(labels)
    genes = resolve_genes(args.gene, list(support))
    named = resolve_perts(args.perturbation, list(perts))

    print_overview(ov, args.split)
    if named:
        for n in named:
            print_pert_detail(n, perts[n], support)
    elif args.as_table:
        print_pert_table(perts, args.kind, args.in_split)
    elif genes:
        for g in genes:
            print_gene_detail(g, support[g])
    else:
        print_gene_table(support)

    exp = experiment(args.experiment, args.output_root).ensure()
    path = exp.write_report("data_card", {
        "artifact": args.artifact,
        "split": args.split,
        "overview": ov,
        "target_genes": support,
        "perturbations": perts,
    })
    logger.info("saved data card -> %s", path)


if __name__ == "__main__":
    main()
