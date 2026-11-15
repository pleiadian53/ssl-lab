#!/usr/bin/env python3
"""Which of the 5,000 modeled genes, and of each held-out pair's scored genes, Geneformer can see.

**Why.** Geneformer has tokens only for the genes in its vocabulary. A gene outside it never
reaches the model, although our count decoder still has to predict it from the embedding. The
comparison reports every endpoint twice: on all of each held-out pair's top 20 genes, as every
earlier round did, and on only the top genes Geneformer can see, for both encoders. The second
reading separates a gap caused by the vocabulary from a gap caused by the representation. This
script writes the list that reading needs.

**How a gene is matched.** The panel is named by gene symbols; the raw file pairs each symbol with
its Ensembl ID. Each ID is then looked up the way Geneformer's tokenizer looks it up
(``in_vocab_gene_counts`` in ``02_embed_geneformer.py``): upper-cased, mapped through the model
series' Ensembl mapping dictionary, which also renames retired IDs, and kept if the result is in the
series' token dictionary. The V1 models use the ``gc30M`` dictionaries and the V2 models the
``gc104M`` ones, so the two series are checked separately.

**The scored genes.** The 20 held-out pairs and their top 20 genes are chosen exactly as the
scorer chooses them (``Scorer`` in ``23_score_round10.py``): every combination in the
differential-expression table with at least 20 held-out cells on the combo split, and the first 20
entries of its ``top_idx``.

Reads gene tables only: the raw file's gene annotation, the cache's panel and DE table, and four
small dictionaries shipped with Geneformer's source. Nothing touches Geneformer's weights.

Usage (from the repo root):
    python examples/perturbation_response/foundation-encoder/06_visible_genes.py \\
        --dict-dir <folder with token_dictionary_gc30M.pkl, ensembl_mapping_dict_gc30M.pkl, *_gc104M.pkl>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import SPLIT_TEST, TOKENS_NPZ

SERIES = {"V1": "gc30M", "V2": "gc104M"}


def raw_symbol_to_ensembl(raw_h5ad: Path) -> dict[str, str]:
    import h5py

    with h5py.File(raw_h5ad, "r") as f:
        var = f["var"]

        def col(key: str) -> np.ndarray:
            x = var[key]
            if isinstance(x, h5py.Group):                     # categorical: codes + categories
                return x["categories"][:][x["codes"][:]].astype(str)
            return x[:].astype(str)

        symbols, ensembl = col(var.attrs["_index"]), col("index")
    if len(set(symbols)) != len(symbols):
        raise ValueError("gene symbols in the raw file are not unique; the panel cannot be matched by symbol")
    return dict(zip(symbols, ensembl))


def visible_mask(ensembl_ids: list[str], mapping: dict, vocab) -> np.ndarray:
    """True where Geneformer's tokenizer would give the gene a token."""
    out = []
    for e in ensembl_ids:
        m = mapping.get(str(e).upper())
        out.append(m is not None and m in vocab)
    return np.array(out, dtype=bool)


def scored_pairs(de: dict, pert_names: np.ndarray, pert_id: np.ndarray, split_combo: np.ndarray,
                 top_k: int = 20, min_test_cells: int = 20) -> dict[str, list[int]]:
    """The held-out pairs and their top-k HVG indices, chosen as the scorer chooses them."""
    is_test = split_combo == SPLIT_TEST
    out = {}
    for pid, name in enumerate(pert_names):
        name = str(name)
        if "+" not in name or name not in de["per_pert"]:
            continue
        if int(((pert_id == pid) & is_test).sum()) < min_test_cells:
            continue
        out[name] = [int(i) for i in de["per_pert"][name]["top_idx"][:top_k]]
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--artifact", default="norman2019")
    ap.add_argument("--dict-dir", required=True, help="folder holding Geneformer's token and Ensembl mapping dictionaries")
    ap.add_argument("--out", default=None, help="default: <data-dir>/<artifact>/foundation/visible_hvg.json")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.artifact
    npz = np.load(root / TOKENS_NPZ, allow_pickle=True)
    gene_ids = [str(g) for g in npz["gene_ids"]]
    de = json.loads((root / "de_genes.json").read_text())
    if de.get("gene_space", "hvg_index") != "hvg_index":
        raise ValueError(f"de_genes.json indexes {de.get('gene_space')}, expected hvg_index")
    sym2ens = raw_symbol_to_ensembl(root / "raw" / "norman_2019.h5ad")
    missing = [g for g in gene_ids if g not in sym2ens]
    if missing:
        raise ValueError(f"{len(missing)} panel genes are not in the raw file's annotation, e.g. {missing[:3]}")
    panel_ens = [sym2ens[g] for g in gene_ids]
    pairs = scored_pairs(de, npz["pert_names"], npz["pert_id"], npz["split_combo"])

    report: dict = {"n_panel_genes": len(gene_ids), "n_pairs": len(pairs), "series": {}}
    for series, tag in SERIES.items():
        d = Path(args.dict_dir)
        vocab = pickle.loads((d / f"token_dictionary_{tag}.pkl").read_bytes())
        mapping = pickle.loads((d / f"ensembl_mapping_dict_{tag}.pkl").read_bytes())
        vis = visible_mask(panel_ens, mapping, vocab)
        per_pair, unseen = {}, {}
        for name, top in pairs.items():
            hidden = [gene_ids[i] for i in top if not vis[i]]
            per_pair[name] = {"n_visible": len(top) - len(hidden), "unseen": hidden}
            for g in hidden:
                unseen[g] = unseen.get(g, 0) + 1
        hist = {}
        for v in per_pair.values():
            hist[str(v["n_visible"])] = hist.get(str(v["n_visible"]), 0) + 1
        report["series"][series] = {
            "dictionary": tag,
            "vocabulary_size": len(vocab),
            "n_panel_visible": int(vis.sum()),
            "visible_hvg_idx": np.flatnonzero(vis).tolist(),
            "scored_entries_visible": int(sum(v["n_visible"] for v in per_pair.values())),
            "scored_entries_total": int(sum(len(t) for t in pairs.values())),
            "pairs_by_n_visible": dict(sorted(hist.items(), reverse=True)),
            "unseen_scored_genes": dict(sorted(unseen.items(), key=lambda kv: -kv[1])),
            "per_pair": per_pair,
        }
        s = report["series"][series]
        print(f"{series} ({tag}): vocabulary {len(vocab)} tokens; panel {s['n_panel_visible']}/{len(gene_ids)} visible; "
              f"scored {s['scored_entries_visible']}/{s['scored_entries_total']}; pairs by visible count "
              f"{s['pairs_by_n_visible']}; unseen {s['unseen_scored_genes']}")

    out = Path(args.out) if args.out else root / "foundation" / "visible_hvg.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
