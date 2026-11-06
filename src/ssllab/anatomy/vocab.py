"""The perturbation vocabulary the condition components embed.

With the processed Norman cache present, the vocabulary is the real one (237
perturbations over 105 target genes), so a walkthrough can show ``CEBPE+RUNX1T1`` turning
into a multi-hot row with ones at CEBPE and RUNX1T1. Only the label arrays are read from
the cache, never the expression matrices. Without a cache a small toy vocabulary stands
in, which is enough for every shape.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ssllab.data.perturbseq import TOKENS_NPZ, target_genes
from ssllab.generative.condition import build_pert_gene_matrix

TOY_PERTS = ("control", "A", "B", "C", "A+B", "B+C")
DEFAULT_COMBO = "CEBPE+RUNX1T1"      # a held-out test combination in the Norman combo split


@dataclass
class Vocab:
    pert_names: list[str]
    pert_gene: torch.Tensor          # (n_perts, n_genes) multi-hot membership
    genes: list[str]                 # column order of pert_gene
    source: str                      # where the names came from
    median_libsize: float = 10_000.0

    @property
    def n_perts(self) -> int:
        return len(self.pert_names)

    def resolve(self, name: str | None) -> str:
        """Find a perturbation by name, ignoring case and the order of its genes.

        ``None`` picks a held-out combination (``CEBPE+RUNX1T1`` on the real vocabulary).
        """
        if name is None:
            if DEFAULT_COMBO in self.pert_names:
                return DEFAULT_COMBO
            return next(p for p in self.pert_names if len(target_genes(p)) >= 2)
        key = "+".join(sorted(g.upper() for g in name.split("+")))
        by_key = {"+".join(sorted(g.upper() for g in p.split("+"))): p for p in self.pert_names}
        if key in by_key:
            return by_key[key]
        close = difflib.get_close_matches(name.upper(), [p.upper() for p in self.pert_names], n=5)
        hint = f"; close matches: {', '.join(close)}" if close else ""
        raise KeyError(f"{name!r} is not in the perturbation vocabulary ({self.source}){hint}")

    def pert_id(self, name: str) -> int:
        return self.pert_names.index(name)


def load_vocab(data_dir: str | Path = "data", artifact: str = "norman2019") -> Vocab:
    """The cache's perturbation vocabulary if it exists, else the toy one."""
    npz_path = Path(data_dir) / artifact / TOKENS_NPZ
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        names = [str(p) for p in npz["pert_names"]]
        median_libsize = float(np.median(npz["libsize"]))
        source = str(Path(data_dir) / artifact)
    else:
        names, median_libsize, source = list(TOY_PERTS), 10_000.0, "toy vocabulary"
    pert_gene, genes = build_pert_gene_matrix(names)
    return Vocab(pert_names=names, pert_gene=pert_gene, genes=genes, source=source,
                 median_libsize=median_libsize)


def toy_vocab() -> Vocab:
    names = list(TOY_PERTS)
    pert_gene, genes = build_pert_gene_matrix(names)
    return Vocab(pert_names=names, pert_gene=pert_gene, genes=genes, source="toy vocabulary")
