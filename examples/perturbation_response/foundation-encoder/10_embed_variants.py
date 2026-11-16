#!/usr/bin/env python3
"""Round 15's embedding stores: V2 variants that each change one property of the round-14 V2 arm. Pod only.

**Why.** In round 14, Geneformer V2-104M's gene-mean embedding read held-out Norman cells clearly
better than our JEPA encoder (readout ceiling 0.908 against 0.863), and V1-10M did not (0.862).
V2-104M differs from our encoder in several ways at once: it was pretrained on about 104 million
cells, it reads each cell's whole transcriptome rather than our 5,000-gene panel, it reads up to
4,094 genes per cell where V1 reads 2,048, and it is large. Round 15 changes one of these at a time
and asks which change costs the gain. This script writes one embedding store per variant, every
one with V2's gene-mean mode at the second-to-last layer, exactly as ``02_embed_geneformer.py``
does for the round-14 arm:

=============  =========================================================================  ======
variant        what changes from round 14's V2-104M gene mean                            arm
=============  =========================================================================  ======
``random``     the same architecture with random weights (``BertForMaskedLM(config)``     v2rand_cell
               at seed 0): no pretraining
``panel``      only the 5,000 panel genes enter each cell's ranking; the library-size     v2panel_cell
               total stays the cell's total over every measured gene, so each gene's
               normalized value is unchanged and only the set of genes ranked shrinks
``cut2048``    every gene list cut after its first 2,048 genes, V1's limit                v2cut_cell
``big``        V2-316M in place of V2-104M (18 layers, 1,152 wide)                         v2big_cell
=============  =========================================================================  ======

**How the panel variant keeps normalization fixed.** Geneformer's tokenizer divides each cell's
counts by ``obs["n_counts"]`` as supplied, never by a total it recomputes from the genes it keeps
(read in ``geneformer/tokenizer.py`` at the pinned revision). The panel input is therefore built
from the full count matrix, so ``n_counts`` is each cell's full-transcriptome total, and only then
cut to the panel's columns.

**Inputs.** The full V2 tokenized dataset of round 14 (``geneformer_full/tokens_V2.dataset``) for
the random, cut and big variants; if it is missing, the variant re-tokenizes every cell. The panel
variant reads the raw file itself. Panel genes are the processed cache's 5,000 genes, matched to
Ensembl IDs through the raw file's gene table (``06_visible_genes.raw_symbol_to_ensembl``).

**Outputs** (under ``--out-dir``): ``stores/<arm>/`` per variant, each with the manifest fields of a
round-14 store plus ``variant`` and ``base_model``; ``variants.json`` with timing and shape; the
random model's folder ``Geneformer-V2-104M-random/`` under ``--gf-root``.

Usage (on the pod, from the repo root):
    python examples/perturbation_response/foundation-encoder/10_embed_variants.py \\
        --gf-root /runpod-volume/ssl-lab/models/geneformer --data-dir /runpod-volume/ssl-lab/data \\
        --out-dir /runpod-volume/ssl-lab/output/geneformer_r15 \\
        --tokens /runpod-volume/ssl-lab/output/geneformer_full/tokens_V2.dataset --variants random panel cut2048 big
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ssllab.data.perturbseq import TOKENS_NPZ, cache_fingerprint_from_dir  # noqa: E402
from ssllab.embeddings import provenance, write_store  # noqa: E402


def _module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = {
    #  variant: (arm, model folder name, MODELS key whose width and series apply, input size)
    "random": ("v2rand_cell", "Geneformer-V2-104M-random", "V2-104M", 4096),
    "panel": ("v2panel_cell", "Geneformer-V2-104M", "V2-104M", 4096),
    "cut2048": ("v2cut_cell", "Geneformer-V2-104M", "V2-104M", 2048),
    "big": ("v2big_cell", "Geneformer-V2-316M", "V2-316M", 4096),
}
RANDOM_SEED = 0
CUT = 2048


def panel_mask(ensembl_ids: np.ndarray, panel_ens: set[str]) -> np.ndarray:
    """Columns of the raw matrix that belong to the panel."""
    mask = np.array([e in panel_ens for e in ensembl_ids])
    if mask.sum() != len(panel_ens):
        raise ValueError(f"{mask.sum()} raw columns match the {len(panel_ens)} panel genes")
    return mask


def panel_adata(gf02, counts, ensembl_ids: np.ndarray, cells: np.ndarray, mask: np.ndarray):
    """Tokenizer input over the panel's columns only, with ``n_counts`` the full-transcriptome total."""
    adata = gf02.build_count_adata(counts, ensembl_ids, cells)   # n_counts over every measured gene
    return adata[:, mask].copy()


def random_model_dir(gf_root: Path, base: str, name: str, seed: int) -> Path:
    """Save an untrained copy of ``base``'s architecture (``BertForMaskedLM`` from its config)."""
    import torch
    from transformers import BertConfig, BertForMaskedLM

    out = gf_root / name
    cfg = BertConfig.from_pretrained(str(gf_root / base))
    torch.manual_seed(seed)
    model = BertForMaskedLM(cfg)
    model.save_pretrained(str(out))
    return out


def full_tokens(args, gf02, root: Path, out: Path) -> Path:
    """Round 14's full V2 dataset, or a fresh tokenization of every cell if it is gone."""
    if args.tokens and Path(args.tokens).exists():
        return Path(args.tokens)
    index = np.load(root / "foundation" / "cell_index.npz")
    cells = np.arange(len(index["raw_row"]), dtype=np.int64)
    counts, ens = gf02.read_count_rows(root / "raw" / "norman_2019.h5ad", index["raw_row"][cells])
    (out / "full_input").mkdir(parents=True, exist_ok=True)
    gf02.build_count_adata(counts, ens, cells).write_h5ad(out / "full_input" / "norman.h5ad")
    path, _ = gf02.tokenize(out / "full_input", out / "full", "V2", args.nproc, counts, ens, cells)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gf-root", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokens", default=None, help="round 14's full tokens_V2.dataset (re-tokenized if absent)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--n-cells", type=int, default=0, help="0 embeds every cell; a sample is for smoke runs only")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--gf-revision", default=None)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("Geneformer's extractor needs CUDA; run this on the pod.")
    sys.path.insert(0, str(Path(args.gf_root)))
    from datasets import load_from_disk

    gf02 = _module("gf_embed", "02_embed_geneformer.py")
    gf06 = _module("gf_visible", "06_visible_genes.py")
    gf08 = _module("gf_mechanism", "08_mechanism_diagnostics.py")
    gf_root, out = Path(args.gf_root), Path(args.out_dir)
    root = Path(args.data_dir) / "norman2019"
    fingerprint = cache_fingerprint_from_dir(root)
    index = np.load(root / "foundation" / "cell_index.npz")
    cells = gf02.select_cells(len(index["raw_row"]), args.n_cells, 0)
    (out / "emb").mkdir(parents=True, exist_ok=True)
    report = {"n_cells_requested": int(len(cells)), "gpu": torch.cuda.get_device_name(0), "variants": {}}

    def subset(ds):
        """For a smoke run, keep only the sampled cells of a full dataset."""
        if args.n_cells <= 0:
            return ds
        keep = set(cells.tolist())
        return ds.filter(lambda ex: ex["cell_idx"] in keep)

    tokens_path = None
    for variant in args.variants:
        arm, model_name, spec_key, input_size = VARIANTS[variant]
        spec = gf02.MODELS[spec_key]
        t0 = time.perf_counter()
        extra: dict = {}
        if variant == "panel":
            gene_ids = np.load(root / TOKENS_NPZ, allow_pickle=True)["gene_ids"]
            sym2ens = gf06.raw_symbol_to_ensembl(root / "raw" / "norman_2019.h5ad")
            panel_ens = {sym2ens[str(g)] for g in gene_ids}
            counts, ens = gf02.read_count_rows(root / "raw" / "norman_2019.h5ad", index["raw_row"][cells])
            mask = panel_mask(ens, panel_ens)
            adata = panel_adata(gf02, counts, ens, cells, mask)
            if not np.allclose(adata.obs["n_counts"].to_numpy(), index["total_counts"][cells]):
                sys.exit("panel input: n_counts is not the full-transcriptome total")
            (out / "panel_input").mkdir(parents=True, exist_ok=True)
            (out / "panel").mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(out / "panel_input" / "norman.h5ad")
            del adata
            dataset, tok = gf02.tokenize(out / "panel_input", out / "panel", "V2", args.nproc,
                                         counts[:, mask], ens[mask], cells)
            del counts
            extra = {"panel_genes": int(mask.sum()), "tokenization": tok}
        else:
            if tokens_path is None:
                tokens_path = full_tokens(args, gf02, root, out)
            ds = subset(load_from_disk(str(tokens_path)))
            if variant == "cut2048":
                n_cut = int(np.sum(np.asarray(ds["length"]) - 2 > CUT))   # V2 lists carry <cls> and <eos>
                ds = gf08.cut_lists(ds, CUT)
                dataset = out / "cut2048" / f"tokens_V2_cut{CUT}.dataset"
                ds.save_to_disk(str(dataset))
                extra = {"cut": CUT, "n_cells_cut": n_cut}
            else:
                dataset = out / f"{variant}_tokens" / "tokens_V2.dataset"
                if args.n_cells > 0:
                    ds.save_to_disk(str(dataset))
                else:
                    dataset = tokens_path
            del ds
        model_dir = (random_model_dir(gf_root, gf02.MODELS["V2-104M"]["dir"], model_name, RANDOM_SEED)
                     if variant == "random" else gf_root / model_name)
        if variant == "random":
            extra["random_seed"] = RANDOM_SEED
        emb, idx, timing = gf02.embed(model_dir, Path(dataset), "V2", "cell", args.batch_size, args.nproc,
                                      out / "emb", spec["dim"])
        store = write_store(
            out / "stores" / arm, emb, idx, source="geneformer", dataset_fingerprint=fingerprint,
            model=model_name, mode="cell", layer="second-to-last (emb_layer=-1)", model_version="V2",
            input_size=input_size, gf_revision=args.gf_revision, variant=variant, base_model=spec_key,
            n_cells_requested=int(len(cells)), n_dropped=int(len(cells) - len(idx)), **provenance(__file__),
        )
        report["variants"][variant] = {"arm": arm, **timing, **gf02.describe(emb), **extra,
                                       "seconds_with_preparation": time.perf_counter() - t0,
                                       "n_dropped": int(len(cells) - len(idx)), "store": str(store.path),
                                       "content_md5": store.manifest["content_md5"]}
        print(variant, json.dumps(report["variants"][variant]), flush=True)
        (out / "variants.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out / "variants.json")


if __name__ == "__main__":
    main()
